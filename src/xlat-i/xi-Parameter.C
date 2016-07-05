#include "xi-Parameter.h"
#include "xi-Type.h"
#include "xi-Value.h"
#include "xi-Entry.h"

namespace xi {

/******************* C/C++ Parameter Marshalling ******************
For entry methods like:
	entry void foo(int nx,double xarr[nx],complex<float> yarr[ny],long ny);

We generate code on the call-side (in the proxy entry method) to
create a message and copy the user's parameters into it.  Scalar
fields are PUP'd, arrays are just memcpy'd.

The message looks like this:

messagestart>--------- PUP'd data ----------------
	|  PUP'd nx
	|  PUP'd offset-to-xarr (from array start, int byte count)
	|  PUP'd length-of-xarr (in elements)
	|  PUP'd offset-to-yarr
	|  PUP'd length-of-yarr (in elements)
	|  PUP'd ny
	+-------------------------------------------
	|  alignment gap (to multiple of 16 bytes)
arraystart>------- xarr data ----------
	| xarr[0]
	| xarr[1]
	| ...
	| xarr[nx-1]
	+------------------------------
	|  alignment gap (for yarr-elements)
	+--------- yarr data ----------
	| yarr[0]
	| yarr[1]
	| ...
	| yarr[ny-1]
	+------------------------------

On the recieve side, all the scalar fields are PUP'd to fresh
stack copies, and the arrays are passed to the user as pointers
into the message data-- so there's no copy on the receive side.

The message is freed after the user entry returns.
*/
Parameter::Parameter(int Nline,Type *Ntype,const char *Nname,
                     const char *NarrLen,Value *Nvalue)
  : type(Ntype)
  , name(Nname)
  , arrLen(NarrLen)
  , val(Nvalue)
  , line(Nline)
  , byConst(false)
  , conditional(0)
  , given_name(Nname)
  , podType(false)
{
	if (isMessage()) {
		name="impl_msg";
        }
	if (name==NULL && !isVoid())
	{/*Fabricate a unique name for this marshalled param.*/
		static int unnamedCount=0;
		name=new char[50];
		sprintf((char *)name,"impl_noname_%x",unnamedCount++);
	}
	byReference=false;
        declaredReference = false;
	if ((arrLen==NULL)&&(val==NULL))
	{ /* Consider passing type by reference: */
		if (type->isNamed())
		{ /* Some user-defined type: pass by reference */
			byReference=true;
		}
		if (type->isReference()) {
			byReference=true;
                        declaredReference = true;
			/* Clip off the ampersand--we'll add
			   it back ourselves in Parameter::print. */
			type=type->deref();
		}
                if (type->isConst()) {
                  byConst = true;
                  type = type->deref();
                }
	}
}

ParamList::ParamList(ParamList *pl) : manyPointers(false), param(pl->param), next(pl->next) {}

void ParamList::print(XStr &str,int withDefaultValues,int useConst)
{
    	param->print(str,withDefaultValues,useConst);
    	if (next) {
    		str<<", ";
    		next->print(str,withDefaultValues,useConst);
    	}
}

void ParamList::printTypes(XStr &str,int withDefaultValues,int useConst)
{
    XStr typeStr;
    param->getType()->print(typeStr);
    str << typeStr;
    if (next) {
      str << ", ";
      next->printTypes(str,withDefaultValues,useConst);
    }
}

void Parameter::print(XStr &str,int withDefaultValues,int useConst)
{
	if (arrLen!=NULL)
	{ //Passing arrays by const pointer-reference
		if (useConst) str<<"const ";
		str<<type<<" *";
		if (name!=NULL) str<<name;
	}
	else {
	    if (conditional) {
	        str<<type<<" *"<<name; 
	    }
	    else if (byReference)
		{ //Pass named types by const C++ reference
                        if (useConst || byConst) str<<"const ";
			str<<type<<" &";
		        if (name!=NULL) str<<name;
		}
		else
		{ //Pass everything else by value
                  // @TODO uncommenting this requires that PUP work on const types
                  //if (byConst) str << "const ";
			str<<type;
			if (name!=NULL) str<<" "<<name;
			if (withDefaultValues && val!=NULL)
			    {str<<" = ";val->print(str);}
		}
	}
}

void ParamList::printAddress(XStr &str)
{
    	param->printAddress(str);
    	if (next) {
    		str<<", ";
    		next->printAddress(str);
    	}
}

void Parameter::printAddress(XStr &str)
{
    	type->print(str);
    	str<<"*";
    	if (name!=NULL)
    		str<<" "<<name;
}

void ParamList::printValue(XStr &str)
{
    	param->printValue(str);
    	if (next) {
    		str<<", ";
    		next->printValue(str);
    	}
}

void Parameter::printValue(XStr &str)
{
    	if (arrLen==NULL)
    	  	str<<"*";
    	if (name!=NULL)
    		str<<name;
}

int ParamList::orEach(pred_t f)
{
	ParamList *cur=this;
	int ret=0;
	do {
		ret|=((cur->param)->*f)();
	} while (NULL!=(cur=cur->next));
	return ret;
}

void ParamList::callEach(fn_t f,XStr &str)
{
	ParamList *cur=this;
	do {
		((cur->param)->*f)(str);
	} while (NULL!=(cur=cur->next));
}

int ParamList::hasConditional() {
  return orEach(&Parameter::isConditional);
}

/** marshalling: pack fields into flat byte buffer **/
void ParamList::marshall(XStr &str, XStr &entry)
{
	if (isVoid())
		str<<"  void *impl_msg = CkAllocSysMsg();\n";
	else if (isMarshalled())
	{
		str<<"  //Marshall: ";print(str,0);str<<"\n";
		//First pass: find sizes
		str<<"  int impl_off=0;\n";
		int hasArrays=orEach(&Parameter::isArray);
		if (hasArrays) {
		  str<<"  int impl_arrstart=0;\n";
		  callEach(&Parameter::marshallArraySizes,str);
		}
		str<<"  { //Find the size of the PUP'd data\n";
		str<<"    PUP::sizer implP;\n";
		callEach(&Parameter::pup,str);
		if (hasArrays)
		{ /*round up pup'd data length--that's the first array*/
		  str<<"    impl_arrstart=CK_ALIGN(implP.size(),16);\n";
		  str<<"    impl_off+=impl_arrstart;\n";
		}
		else  /*No arrays--no padding*/
		  str<<"    impl_off+=implP.size();\n";
		str<<"  }\n";
		//Now that we know the size, allocate the packing buffer
		if (hasConditional()) str<<"  MarshallMsg_"<<entry<<" *impl_msg=CkAllocateMarshallMsgT<MarshallMsg_"<<entry<<" >(impl_off,impl_e_opts);\n";
		else str<<"  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);\n";
		//Second pass: write the data
		str<<"  { //Copy over the PUP'd data\n";
		str<<"    PUP::toMem implP((void *)impl_msg->msgBuf);\n";
		callEach(&Parameter::pup,str);
		callEach(&Parameter::copyPtr,str);
		str<<"  }\n";
		if (hasArrays)
		{ //Marshall each array
		  str<<"  char *impl_buf=impl_msg->msgBuf+impl_arrstart;\n";
		  callEach(&Parameter::marshallArrayData,str);
		}
	}
}
void Parameter::marshallArraySizes(XStr &str)
{
	Type *dt=type->deref();//Type, without &
	if (dt->isPointer())
		XLAT_ERROR_NOCOL("can't pass pointers across processors--\n"
		                 "Indicate the array length with []'s, or pass a reference",
		                 line);
	if (isArray()) {
		str<<"  int impl_off_"<<name<<", impl_cnt_"<<name<<";\n";
		str<<"  impl_off_"<<name<<"=impl_off=CK_ALIGN(impl_off,sizeof("<<dt<<"));\n";
		str<<"  impl_off+=(impl_cnt_"<<name<<"=sizeof("<<dt<<")*("<<arrLen<<"));\n";
	}
}
void Parameter::pup(XStr &str) {
	if (isArray()) {
	   str<<"    implP|impl_off_"<<name<<";\n";
	   str<<"    implP|impl_cnt_"<<name<<";\n";
	}
	else if (!conditional) {
	  if (byReference) {
	    str<<"    //Have to cast away const-ness to get pup routine\n";
	    str<<"    implP|("<<type<<" &)"<<name<<";\n";
	  }
	  else
	    str<<"    implP|"<<name<<";\n";
	}
}
void Parameter::marshallArrayData(XStr &str)
{
	if (isArray())
		str<<"  memcpy(impl_buf+impl_off_"<<name<<
			","<<name<<",impl_cnt_"<<name<<");\n";
}
void Parameter::copyPtr(XStr &str)
{
  if (isConditional()) {
    str<<"    impl_msg->"<<name<<"="<<name<<";\n";
  }
}

void ParamList::beginRednWrapperUnmarshall(XStr &str, bool isSDAGGen) {
  if (isSDAGGen) {
    str << *entry->genClosureTypeNameProxyTemp << "*"
        << " genClosure = new " << *entry->genClosureTypeNameProxyTemp << "()" << ";\n";
  }

    if (isMarshalled())
    {
        str<<"  /*Unmarshall pup'd fields: ";print(str,0);str<<"*/\n";
        str<<"  PUP::fromMem implP(impl_buf);\n";
        if (next != NULL && next->next == NULL) {
	  // 2 argument case - special cases for an array and its length, in either order
	  if (isArray() && !next->isArray()) {
              if (!isSDAGGen) {
                Type* dtLen = next->param->type->deref();
                str << "  " << dtLen << " " << next->param->name << "; "
                    << next->param->name << " = "
                    << "((CkReductionMsg*)impl_msg)->getLength() / sizeof("
                    << param->type->deref() << ");\n";
                Type *dt = param->type->deref();
                str << "  " << dt << "* " << param->name << "; "
                    << param->name << " = (" << dt << "*)impl_buf;\n";
              } else {
                str << "  genClosure->" << next->param->name << " = "
                    << "((CkReductionMsg*)impl_msg)->getLength() / sizeof("
                    << param->type->deref() << ");\n";
                Type* dt = param->type->deref();
                str << "  genClosure->" << param->name << " = (" << dt << "*)impl_buf;\n";
              }
	  } else if (!isArray() && next->isArray()) {
              if (!isSDAGGen) {
                Type* dt = param->type->deref();
                str << "  " << dt << " " << param->name << "; "
                    << param->name << " = "
                    << "((CkReductionMsg*)impl_msg)->getLength() / sizeof("
                    << next->param->type->deref() << ");\n";
                dt = next->param->type->deref();
                str << "  " << dt << "* " << next->param->name << "; "
                    << next->param->name << " = (" << dt << "*)impl_buf;\n";
              } else {
                str << "  genClosure->" << param->name << " = "
                    << "((CkReductionMsg*)impl_msg)->getLength() / sizeof("
                    << next->param->type->deref() << ");\n";
                Type* dt = next->param->type->deref();
                str << "  genClosure->" << next->param->name << " = (" << dt << "*)impl_buf;\n";
              }
	  } else {
              if (!isSDAGGen)
                callEach(&Parameter::beginUnmarshall,str);
              else
                callEach(&Parameter::beginUnmarshallSDAGCall,str);
	  }
        } else if (next == NULL && isArray()) {
	  // 1 argument case - special case for a standalone array
	  Type *dt = param->type->deref();
	  if (!isSDAGGen) {
	    str << "  " << dt << "* " << param->name << "; "
		<< param->name << " = (" << dt << "*)impl_buf;\n";
	  } else {
	    str << "  genClosure->" << param->name << " = (" << dt << "*)impl_buf;\n";
	  }
	} else {
            str << "  /* non two-param case */\n";
            if (!isSDAGGen)
              callEach(&Parameter::beginUnmarshall,str);
            else
              callEach(&Parameter::beginUnmarshallSDAGCall,str);
            str<<"  impl_buf+=CK_ALIGN(implP.size(),16);\n";
            str<<"  /*Unmarshall arrays:*/\n";
            if (!isSDAGGen)
              callEach(&Parameter::unmarshallArrayData,str);
            else
              callEach(&Parameter::unmarshallArrayDataSDAGCall,str);
        }
    }
}

/** unmarshalling: unpack fields from flat buffer **/
void ParamList::beginUnmarshall(XStr &str)
{
    if (isMarshalled())
    {
        str<<"  /*Unmarshall pup'd fields: ";print(str,0);str<<"*/\n";
        str<<"  PUP::fromMem implP(impl_buf);\n";
        callEach(&Parameter::beginUnmarshall,str);
        str<<"  impl_buf+=CK_ALIGN(implP.size(),16);\n";
        str<<"  /*Unmarshall arrays:*/\n";
        callEach(&Parameter::unmarshallArrayData,str);
    }
}
void Parameter::beginUnmarshall(XStr &str)
{ //First pass: unpack pup'd entries
	Type *dt=type->deref();//Type, without &
	if (isArray()) {
		str<<"  int impl_off_"<<name<<", impl_cnt_"<<name<<"; \n";
		str<<"  implP|impl_off_"<<name<<";\n";
		str<<"  implP|impl_cnt_"<<name<<";\n";
	}
	else if (isConditional())
        str<<"  "<<dt<<" *"<<name<<"=impl_msg_typed->"<<name<<";\n";
	else
		str<<"  "<<dt<<" "<<name<<"; implP|"<<name<<";\n";
}

void Parameter::beginUnmarshallSDAGCall(XStr &str) {
  Type *dt=type->deref();
  if (isArray()) {
    str << "  int impl_off_" << name << ", impl_cnt_" << name << "; \n";
    str << "  implP|impl_off_" << name << ";\n";
    str << "  implP|impl_cnt_" << name << ";\n";
  } else
    str << "  implP|" << (podType ? "" : "*") << "genClosure->" << name << ";\n";
}


/** unmarshalling: unpack fields from flat buffer **/
void ParamList::beginUnmarshallSDAGCall(XStr &str, bool usesImplBuf) {
  bool hasArray = false;
  for (ParamList* pl = this; pl != NULL; pl = pl->next) {
    hasArray = hasArray || pl->param->isArray();
  }

  if (isMarshalled()) {
    str << "  PUP::fromMem implP(impl_buf);\n";
    str << "  " << *entry->genClosureTypeNameProxyTemp << "*" <<
      " genClosure = new " << *entry->genClosureTypeNameProxyTemp << "()" << ";\n";
    callEach(&Parameter::beginUnmarshallSDAGCall, str);
    str << "  impl_buf+=CK_ALIGN(implP.size(),16);\n";
    callEach(&Parameter::unmarshallArrayDataSDAGCall,str);
    if (hasArray) {
      if (!usesImplBuf) {
        str << "  genClosure->_impl_marshall = impl_msg_typed;\n";
        str << "  CmiReference(UsrToEnv(genClosure->_impl_marshall));\n";
      } else {
        str << "  genClosure->_impl_buf_in = impl_buf;\n";
        str << "  genClosure->_impl_buf_size = implP.size();\n";
      }
    }
  }
}
void ParamList::beginUnmarshallSDAG(XStr &str) {
  if (isMarshalled()) {
    str << "          PUP::fromMem implP(impl_buf);\n";
    callEach(&Parameter::beginUnmarshall,str);
    str << "          impl_buf+=CK_ALIGN(implP.size(),16);\n";
    callEach(&Parameter::unmarshallArrayDataSDAG,str);
  }
}
void Parameter::unmarshallArrayDataSDAG(XStr &str) {
  if (isArray()) {
    Type *dt=type->deref();//Type, without &
    str << "          " << name << " = ("<<dt<<" *)(impl_buf+impl_off_" << name << ");\n";
  }
}
void Parameter::unmarshallArrayDataSDAGCall(XStr &str) {
  if (isArray()) {
    Type *dt=type->deref();//Type, without &
    str << "  genClosure->" << name << " = (" << dt << " *)(impl_buf+impl_off_" << name << ");\n";
  }
}

void ParamList::unmarshallSDAGCall(XStr &str, int isFirst) {
  if (isFirst && isMessage()) str<<"("<<param->type<<")impl_msg";
  else if (!isVoid()) {
    str << "genClosure->";
    str << param->getName();
    if (next) {
      str<<", ";
      next->unmarshallSDAGCall(str, 0);
    }
  }
}


void Parameter::unmarshallArrayData(XStr &str)
{ //Second pass: unpack pointed-to arrays
	if (isArray()) {
		Type *dt=type->deref();//Type, without &
		str<<"  "<<dt<<" *"<<name<<"=("<<dt<<" *)(impl_buf+impl_off_"<<name<<");\n";
	}
}

void ParamList::unmarshall(XStr &str, int isFirst)  //Pass-by-value
{
    	if (isFirst && isMessage()) str<<"("<<param->type<<")impl_msg";
    	else if (!isVoid()) {
    		str<<param->getName();
		if (next) {
    			str<<", ";
    			next->unmarshall(str, 0);
    		}
    	}
}

void ParamList::unmarshallAddress(XStr &str, int isFirst)  //Pass-by-reference, for Fortran
{
    	if (isFirst && isMessage()) str<<"("<<param->type<<")impl_msg";
    	else if (!isVoid()) {
    		if (param->isArray()) str<<param->getName(); //Arrays are already pointers
		else str<<"& "<<param->getName(); //Take address of simple types and structs
		if (next) {
    			str<<", ";
    			next->unmarshallAddress(str, 0);
    		}
    	}
}

void ParamList::pupAllValues(XStr &str) {
	if (isMarshalled())
		callEach(&Parameter::pupAllValues,str);
}

void Parameter::pupAllValues(XStr &str) {
	str<<"  if (implDestP.hasComments()) implDestP.comment(\""<<name<<"\");\n";
	if (isArray()) {
	  str<<
	  "  implDestP.synchronize(PUP::sync_begin_array);\n"
	  "  for (int impl_i=0;impl_i*(sizeof(*"<<name<<"))<impl_cnt_"<<name<<";impl_i++) {\n"
	  "    implDestP.synchronize(PUP::sync_item);\n"
	  "    implDestP|"<<name<<"[impl_i];\n"
	  "  }\n"
	  "  implDestP.synchronize(PUP::sync_end_array);\n"
	  ;
	}
	else /* not an array */ {
	  if (isConditional()) str<<"  pup_pointer(&implDestP, (void**)&"<<name<<");\n";
	  else str<<"  implDestP|"<<name<<";\n";
	}
}

void ParamList::endUnmarshall(XStr &)
{
	/* Marshalled entry points now have the "SNOKEEP" attribute...
    	if (isMarshalled()) {
    		str<<"  delete (CkMarshallMsg *)impl_msg;\n";
    	}
	*/
}

void ParamList::printMsg(XStr& str) {
    ParamList *pl;
    param->printMsg(str);
    pl = next;
    while (pl != NULL)
    {
       str <<", ";
       pl->param->printMsg(str);
       pl = pl->next;
    } 
}

void Parameter::printMsg(XStr& str) {
  type->print(str);
  if(given_name!=0)
    str <<given_name;
}

int Parameter::isMessage(void) const {return type->isMessage();}
int Parameter::isVoid(void) const {return type->isVoid();}
int Parameter::isCkArgMsgPtr(void) const {return type->isCkArgMsgPtr();}
int Parameter::isCkMigMsgPtr(void) const {return type->isCkMigMsgPtr();}
int Parameter::isArray(void) const {return arrLen!=NULL;}
int Parameter::isConditional(void) const {return conditional;}

int Parameter::operator==(const Parameter &parm) const {
  return *type == *parm.type;
}

void Parameter::setConditional(int c) { conditional = c; if (c) byReference = false; }

void Parameter::setAccelBufferType(int abt) {
  accelBufferType = ((abt < ACCEL_BUFFER_TYPE_MIN || abt > ACCEL_BUFFER_TYPE_MAX) ? (ACCEL_BUFFER_TYPE_UNKNOWN) : (abt));
}

int   Parameter::getAccelBufferType() { return accelBufferType; }
void  Parameter::setAccelInstName(XStr* ain) { accelInstName = ain; }
XStr* Parameter::getAccelInstName(void) { return accelInstName; }

ParamList::ParamList(Parameter *Nparam,ParamList *Nnext)
  :param(Nparam), next(Nnext) { 
      manyPointers = false;
      if(next != NULL && (param->isMessage() || next->isMessage())){
        manyPointers = true;
      }
}

int ParamList::isNamed(void) const {return param->type->isNamed();}
int ParamList::isBuiltin(void) const {return param->type->isBuiltin();}
int ParamList::isMessage(void) const {
    return (next==NULL) && param->isMessage();
}
const char *ParamList::getArrayLen(void) const {return param->getArrayLen();}
int ParamList::isArray(void) const {return param->isArray();}
int ParamList::isReference(void) const {return param->type->isReference() || param->byReference;}
int ParamList::declaredReference(void) const {return param->type->isReference() || param->declaredReference; }
bool ParamList::isConst(void) const {return param->type->isConst() || param->byConst;}
int ParamList::isVoid(void) const {
    return (next==NULL) && param->isVoid();
}
int ParamList::isPointer(void) const {return param->type->isPointer();}
const char *ParamList::getGivenName(void) const {return param->getGivenName();}
void ParamList::setGivenName(const char* s) {param->setGivenName(s);}
const char *ParamList::getName(void) const {return param->getName();}
int ParamList::isMarshalled(void) const {
    return !isVoid() && !isMessage();
}
int ParamList::isCkArgMsgPtr(void) const {
    return (next==NULL) && param->isCkArgMsgPtr();
}
int ParamList::isCkMigMsgPtr(void) const {
    return (next==NULL) && param->isCkMigMsgPtr();
}
int ParamList::getNumStars(void) const {return param->type->getNumStars(); }
const char *ParamList::getBaseName(void) {
    return param->type->getBaseName();
}
void ParamList::genMsgProxyName(XStr &str) {
    param->type->genMsgProxyName(str);
}

void ParamList::checkParamList(){
  if (manyPointers) {
    XLAT_ERROR_NOCOL("multiple pointers passed to a non-local entry method\n"
                     "You may pass only a single pointer to it, which should point to a message.",
                     param->line);
  }
}

int ParamList::operator==(ParamList &plist) {
  if (!(*param == *(plist.param))) return 0;
  if (!next && !plist.next) return 1;
  if (!next || !plist.next) return 0;
  return *next ==  *plist.next;
}

}   // namespace xi
