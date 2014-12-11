#include "xi-Entry.h"
#include "xi-Parameter.h"
#include "xi-Value.h"
#include "xi-SdagCollection.h"
#include "xi-Chare.h"

#include "sdag/constructs/When.h"

#include <list>
using std::list;

namespace xi {

extern int fortranMode;
extern int internalMode;
const char *python_doc;

XStr Entry::proxyName(void) {return container->proxyName();}
XStr Entry::indexName(void) {return container->indexName();}

void Entry::print(XStr& str)
{
  if(isThreaded())
    str << "threaded ";
  if(isSync())
    str << "sync ";
  if(retType) {
    retType->print(str);
    str << " ";
  }
  str << name<<"(";
  if(param)
    param->print(str);
  str << ")";
  if(stacksize) {
    str << " stacksize = ";
    stacksize->print(str);
  }
  str << ";\n";
}

void Entry::check() {
  if (!external) {
    if (isConstructor() && retType && !retType->isVoid())
      XLAT_ERROR_NOCOL("constructors cannot return a value",
                       first_line_);

    if (!isConstructor() && !retType)
      XLAT_ERROR_NOCOL("non-constructor entry methods must specify a return type (probably void)",
                       first_line_);

    if (isConstructor() && (isSync() || isIget())) {
      XLAT_ERROR_NOCOL("constructors cannot have the 'sync' attribute",
                       first_line_);
      attribs ^= SSYNC;
    }

    if (param->isCkArgMsgPtr() && (!isConstructor() || !container->isMainChare()))
      XLAT_ERROR_NOCOL("CkArgMsg can only be used in mainchare's constructor",
                       first_line_);

    if (isExclusive() && isConstructor())
      XLAT_ERROR_NOCOL("constructors cannot be 'exclusive'",
		       first_line_);

    if (isImmediate() && !container->isNodeGroup())
      XLAT_ERROR_NOCOL("[immediate] entry methods are only allowed on 'nodegroup' types",
		       first_line_);

    if (isLocal() && (container->isChare() || container->isNodeGroup()))
      XLAT_ERROR_NOCOL("[local] entry methods are only allowed on 'array' and 'group' types",
		       first_line_);
  }

  if (!isThreaded() && stacksize)
    XLAT_ERROR_NOCOL("the 'stacksize' attribute is only applicable to methods declared 'threaded'",
                     first_line_);

  if (retType && !isSync() && !isIget() && !isLocal() && !retType->isVoid())
    XLAT_ERROR_NOCOL("non-void return type in a non-sync/non-local entry method\n"
                     "To return non-void, you need to declare the method as [sync], which means it has blocking semantics,"
                     " or [local].",
                     first_line_);

  if (!isLocal() && param)
    param->checkParamList();

  if (isPython() && !container->isPython())
    XLAT_ERROR_NOCOL("python entry method declared in non-python chare",
                     first_line_);

  // check the parameter passed to the function, it must be only an integer
  if (isPython() && (!param || param->next || !param->param->getType()->isBuiltin() || !((BuiltinType*)param->param->getType())->isInt()))
    XLAT_ERROR_NOCOL("python entry methods take only one parameter, which is of type 'int'",
                     first_line_);

  if (isExclusive() && !container->isNodeGroup())
      XLAT_ERROR_NOCOL("only nodegroup methods can be 'exclusive'",
                       first_line_);

  // (?) Check that every when statement has a corresponding entry method
  // declaration. Otherwise, print all candidates tested (as in clang, gcc.)
  if (isSdag()) {
    list<CEntry*> whenEntryList;
    sdagCon->generateEntryList(whenEntryList, NULL);

    for (list<CEntry*>::iterator en = whenEntryList.begin(); en != whenEntryList.end(); ++en) {
      container->lookforCEntry(*en);
      (*en)->check();
    }
  }

  if (isTramTarget()) {
    if (param && (!param->isMarshalled() || param->isVoid() || param->next != NULL))
      XLAT_ERROR_NOCOL("'aggregate' entry methods must be parameter-marshalled "
                       "and take a single argument",
                       first_line_);

    if (!((container->isGroup() && !container->isNodeGroup()) || container->isArray()))
      XLAT_ERROR_NOCOL("'aggregate' entry methods can only be used in regular groups and chare arrays",
                       first_line_);
  }
}

void Entry::lookforCEntry(CEntry *centry)
{
   // compare name
   if (strcmp(name, *centry->entry) != 0) return;

   centry->addCandidate(this);

   // compare param
   if (param && !centry->paramlist) return;
   if (!param && centry->paramlist) return;
   if (param && !(*param == *centry->paramlist)) return;

   isWhenEntry = 1;
   centry->decl_entry = this;
}

Entry::Entry(int l, int a, Type *r, const char *n, ParamList *p, Value *sz, SdagConstruct *sc, const char *e, int fl, int ll) :
  attribs(a), retType(r), stacksize(sz), sdagCon(sc), name((char *)n), targs(0), intExpr(e), param(p), genClosureTypeName(0), genClosureTypeNameProxy(0), genClosureTypeNameProxyTemp(0), entryPtr(0), first_line_(fl), last_line_(ll)
{
  line=l; container=NULL;
  entryCount=-1;
  isWhenEntry=0;
  if (param && param->isMarshalled() && !isThreaded()) attribs|=SNOKEEP;

  if (isPython()) pythonDoc = python_doc;
  ParamList *plist = p;
  while (plist != NULL) {
    plist->entry = this;
    if (plist->param) plist->param->entry = this;
    plist = plist->next;
  }
}

void Entry::setChare(Chare *c) {
	Member::setChare(c);
        // mainchare constructor parameter is not allowed
	/* ****************** REMOVED 10/8/2002 ************************
        if (isConstructor()&&container->isMainChare() && param != NULL)
          if (!param->isCkArgMsgPtr())
           die("MainChare Constructor doesn't allow parameter!", line);
	Removed old treatment for CkArgMsg to allow argc, argv or void
	constructors for mainchares.
	* **************************************************************/
        if (isConstructor() && param->isVoid()) {
          if (container->isMainChare()) {
            //Main chare always magically takes CkArgMsg
            Type *t = new PtrType(new NamedType("CkArgMsg"));
            param=new ParamList(new Parameter(line,t));
            std::cerr << "Charmxi> " << line << ": Deprecation warning: mainchare constructors should explicitly take CkArgMsg* if that's how they're implemented.\n";
          }
          if (container->isArray()) {
            Array *a = dynamic_cast<Array*>(c);
            a->hasVoidConstructor = true;
          }
        }

	entryCount=c->nextEntry();

	//Make a special "callmarshall" method, for communication optimizations to use:
	hasCallMarshall=param->isMarshalled() && !isThreaded() && !isSync() && !isExclusive() && !fortranMode;
	if (isSdag())
	{
	  container->setSdag(1);

	  list<CEntry*> whenEntryList;
	  sdagCon->generateEntryList(whenEntryList, NULL);

	  for (list<CEntry*>::iterator i = whenEntryList.begin(); i != whenEntryList.end(); ++i) {
	    container->lookforCEntry(*i);
	  }
	}
}

void Entry::preprocessSDAG()
{
  if (isSdag() || isWhenEntry) {
    if (container->isNodeGroup())
      {
	attribs |= SLOCKED; // Make the method [exclusive] to preclude races on SDAG control structures
      }
  }
}

// "parameterType *msg" or "void".
// Suitable for use as the only parameter
XStr Entry::paramType(int withDefaultVals,int withEO,int useConst)
{
  XStr str;
  param->print(str,withDefaultVals,useConst);
  if (withEO) str<<eo(withDefaultVals,!param->isVoid());
  return str;
}

// "parameterType *msg," if there is a non-void parameter,
// else empty.  Suitable for use with another parameter following.
XStr Entry::paramComma(int withDefaultVals,int withEO)
{
  XStr str;
  if (!param->isVoid()) {
    str << paramType(withDefaultVals,withEO);
    str << ", ";
  }
  return str;
}
XStr Entry::eo(int withDefaultVals,int priorComma) {
  XStr str;
  if (param->isMarshalled()) {//FIXME: add options for void methods, too...
    if (priorComma) str<<", ";
    str<<"const CkEntryOptions *impl_e_opts";
    if (withDefaultVals) str<<"=NULL";
  }
  return str;
}

void Entry::collectSdagCode(SdagCollection *sc)
{
  if (isSdag()) {
    sc->addNode(this);
  }
}

XStr Entry::marshallMsg(void)
{
  XStr ret;
  XStr epName = epStr();
  param->marshall(ret, epName);
  return ret;
}

XStr Entry::epStr(bool isForRedn, bool templateCall)
{
  XStr str;
  if (isForRedn)
    str<<"redn_wrapper_";
  str << name << "_";

  if (param->isMessage()) {
    str<<param->getBaseName();
    str.replace(':', '_');
  }
  else if (param->isVoid())
    str<<"void";
  else
    str<<"marshall"<<entryCount;

  if (tspec && templateCall) {
    str << "< ";
    tspec->genShort(str);
    str << " >";
  }

  return str;
}

XStr Entry::epIdx(int fromProxy, bool isForRedn)
{
  XStr str;
  if (fromProxy) {
    str << indexName()<<"::";
    // If the chare is also templated, then we must avoid a parsing ambiguity
    if (tspec)
      str << "template ";
  }
  str << "idx_" << epStr(isForRedn, true) << "()";
  return str;
}

XStr Entry::epRegFn(int fromProxy, bool isForRedn)
{
  XStr str;
  if (fromProxy)
    str << indexName() << "::";
  str << "reg_" << epStr(isForRedn, true) << "()";
  return str;
}

XStr Entry::chareIdx(int fromProxy)
{
  XStr str;
  if (fromProxy)
    str << indexName()<<"::";
  str << "__idx";
  return str;
}

XStr Entry::syncPreCall(void) {
  XStr str;
  if(retType->isVoid())
    str << "  void *impl_msg_typed_ret = ";
  else if(retType->isMessage()) 
    str << "  "<< retType <<" impl_msg_typed_ret = ("<< retType <<")";
  else
    str << "  CkMarshallMsg *impl_msg_typed_ret = (CkMarshallMsg *)";
  return str;
}

XStr Entry::syncPostCall(void) {
  XStr str;
  if(retType->isVoid())
    str << "  CkFreeSysMsg(impl_msg_typed_ret); \n";
  else if (!retType->isMessage()){
      str <<"  char *impl_buf_ret=impl_msg_typed_ret->msgBuf; \n";
      str <<"  PUP::fromMem implPS(impl_buf_ret); \n";
      str <<"  "<<retType<<" retval; implPS|retval; \n";
      str <<"  CkFreeMsg(impl_msg_typed_ret); \n";
      str <<"  return retval; \n"; 
  }
  else{
     str <<"  return impl_msg_typed_ret;\n";   
  }
  return str;
}



/*************************** Chare Entry Points ******************************/

void Entry::genChareDecl(XStr& str)
{
  if(isConstructor()) {
    genChareStaticConstructorDecl(str);
  } else {
    // entry method declaration
    str << "    " << generateTemplateSpec(tspec) << "\n"
        << "    " << retType << " " << name << "(" << paramType(1,1) << ");\n";
  }
}

void Entry::genChareDefs(XStr& str)
{
  if(isConstructor()) {
    genChareStaticConstructorDefs(str);
  } else {
    XStr params; params<<epIdx()<<", impl_msg, &ckGetChareID()";
    // entry method definition
    XStr retStr; retStr<<retType;
    str << makeDecl(retStr,1)<<"::"<<name<<"("<<paramType(0,1)<<")\n";
    str << "{\n  ckCheck();\n"<<marshallMsg();
    if(isSync()) {
      str <<  syncPreCall() << "CkRemoteCall("<<params<<");\n";
      str << syncPostCall(); 
    } else {//Regular, non-sync message
      str << "  if (ckIsDelegated()) {\n";
      str << "    int destPE=CkChareMsgPrep("<<params<<");\n";
      str << "    if (destPE!=-1) ckDelegatedTo()->ChareSend(ckDelegatedPtr(),"<<params<<",destPE);\n";
      str << "  }\n";
      XStr opts;
      opts << ",0";
      if (isSkipscheduler())  opts << "+CK_MSG_EXPEDITED";
      if (isInline())  opts << "+CK_MSG_INLINE";
      str << "  else CkSendMsg("<<params<<opts<<");\n";
    }
    str << "}\n";
  }
}

void Entry::genChareStaticConstructorDecl(XStr& str)
{
  str << "    static CkChareID ckNew("<<paramComma(1)<<"int onPE=CK_PE_ANY"<<eo(1)<<");\n";
  str << "    static void ckNew("<<paramComma(1)<<"CkChareID* pcid, int onPE=CK_PE_ANY"<<eo(1)<<");\n";
  if (!param->isVoid())
    str << "    "<<container->proxyName(0)<<"("<<paramComma(1)<<"int onPE=CK_PE_ANY"<<eo(1)<<");\n";
}

void Entry::genChareStaticConstructorDefs(XStr& str)
{
  str << makeDecl("CkChareID",1)<<"::ckNew("<<paramComma(0)<<"int impl_onPE"<<eo(0)<<")\n";
  str << "{\n"<<marshallMsg();
  str << "  CkChareID impl_ret;\n";
  str << "  CkCreateChare("<<chareIdx()<<", "<<epIdx()<<", impl_msg, &impl_ret, impl_onPE);\n";
  str << "  return impl_ret;\n";
  str << "}\n";

  str << makeDecl("void",1)<<"::ckNew("<<paramComma(0)<<"CkChareID* pcid, int impl_onPE"<<eo(0)<<")\n";
  str << "{\n"<<marshallMsg();
  str << "  CkCreateChare("<<chareIdx()<<", "<<epIdx()<<", impl_msg, pcid, impl_onPE);\n";
  str << "}\n";

  if (!param->isVoid()) {
    str << makeDecl(" ",1)<<"::"<<container->proxyName(0)<<"("<<paramComma(0)<<"int impl_onPE"<<eo(0)<<")\n";
    str << "{\n"<<marshallMsg();
    str << "  CkChareID impl_ret;\n";
    str << "  CkCreateChare("<<chareIdx()<<", "<<epIdx()<<", impl_msg, &impl_ret, impl_onPE);\n";
    str << "  ckSetChareID(impl_ret);\n";
    str << "}\n";
  }
}

/***************************** Array Entry Points **************************/

void Entry::genArrayDecl(XStr& str)
{
  if(isConstructor()) {
    str << "    " << generateTemplateSpec(tspec) << "\n";
    genArrayStaticConstructorDecl(str);
  } else {
    if ((isSync() || isLocal()) && !container->isForElement()) return; //No sync broadcast
    str << "    " << generateTemplateSpec(tspec) << "\n";
    if(isIget())
      str << "    "<<"CkFutureID"<<" "<<name<<"("<<paramType(1,1)<<") ;\n"; //no const
    else if(isLocal())
      str << "    "<<retType<<" "<<name<<"("<<paramType(1,1,0)<<") ;\n";
    else
      str << "    "<<retType<<" "<<name<<"("<<paramType(1,1)<<") ;\n"; //no const
  }
}

void Entry::genArrayDefs(XStr& str)
{
  if(isIget() && !container->isForElement()) return;

  if (isConstructor())
    genArrayStaticConstructorDefs(str);
  else
  {//Define array entry method
    const char *ifNot="CkArray_IfNotThere_buffer";
    if (isCreateHere()) ifNot="CkArray_IfNotThere_createhere";
    if (isCreateHome()) ifNot="CkArray_IfNotThere_createhome";

    if ((isSync() || isLocal()) && !container->isForElement()) return; //No sync broadcast

    XStr retStr; retStr<<retType;
    if(isIget())
      str << makeDecl("CkFutureID ",1)<<"::"<<name<<"("<<paramType(0,1)<<") \n"; //no const
    else if(isLocal())
      str << makeDecl(retStr,1)<<"::"<<name<<"("<<paramType(0,1,0)<<") \n";
    else
      str << makeDecl(retStr,1)<<"::"<<name<<"("<<paramType(0,1)<<") \n"; //no const
    str << "{\n  ckCheck();\n";
    XStr inlineCall;
    if (!isNoTrace())
      inlineCall << "    _TRACE_BEGIN_EXECUTE_DETAILED(0,ForArrayEltMsg,(" << epIdx()
    		 << "),CkMyPe(), 0, ((CkArrayIndex&)ckGetIndex()).getProjectionID(((CkGroupID)ckGetArrayID()).idx), obj);\n";
    if(isAppWork())
      inlineCall << "    _TRACE_BEGIN_APPWORK();\n";
    inlineCall << "#if CMK_LBDB_ON\n"
               << "    LDObjHandle objHandle;\n"
               << "    int objstopped=0;\n"
               << "    objHandle = obj->timingBeforeCall(&objstopped);\n"
               << "#endif\n";
    inlineCall <<
      "#if CMK_CHARMDEBUG\n"
      "    CpdBeforeEp("<<epIdx()<<", obj, NULL);\n"
      "#endif\n";
    inlineCall << "    ";
    if (!retType->isVoid())
      inlineCall << retType << " retValue = ";
    inlineCall << "obj->" << name << "(";
    param->unmarshall(inlineCall);
    inlineCall << ");\n";
    inlineCall <<
      "#if CMK_CHARMDEBUG\n"
      "    CpdAfterEp("<<epIdx()<<");\n"
      "#endif\n";
    inlineCall << "#if CMK_LBDB_ON\n    obj->timingAfterCall(objHandle,&objstopped);\n#endif\n";
    if(isAppWork())
      inlineCall << "    _TRACE_END_APPWORK();\n";
    if (!isNoTrace()) inlineCall << "    _TRACE_END_EXECUTE();\n";
    if (!retType->isVoid()) {
      inlineCall << "    return retValue;\n";
    } else {
      inlineCall << "    return;\n";
    }

    XStr prepareMsg;
    prepareMsg << marshallMsg();
    prepareMsg << "  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);\n";
    prepareMsg << "  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;\n";
    prepareMsg << "  impl_amsg->array_setIfNotThere("<<ifNot<<");\n";

    if (!isLocal()) {
      if (isInline() && container->isForElement()) {
	str << "  " << container->baseName() << " *obj = ckLocal();\n";
	str << "  if (obj) {\n"
	    << inlineCall
	    << "  }\n";
      }
      str << prepareMsg;
    } else {
      str << "  "<< container->baseName() << " *obj = ckLocal();\n";
      str << "#if CMK_ERROR_CHECKING\n";
      str << "  if (obj==NULL) CkAbort(\"Trying to call a LOCAL entry method on a non-local element\");\n";
      str << "#endif\n";
      str << inlineCall;
    }
    if(isIget()) {
	    str << "  CkFutureID f=CkCreateAttachedFutureSend(impl_amsg,"<<epIdx()<<",ckGetArrayID(),ckGetIndex(),&CProxyElement_ArrayBase::ckSendWrapper);"<<"\n";
    }

    if(isSync()) {
      str << syncPreCall() << "ckSendSync(impl_amsg, "<<epIdx()<<");\n";
      str << syncPostCall(); 
    }
    else if (!isLocal())
    {
      XStr opts;
      opts << ",0";
      if (isSkipscheduler())  opts << "+CK_MSG_EXPEDITED";
      if (isInline())  opts << "+CK_MSG_INLINE";
      if(!isIget()) {
      if (container->isForElement() || container->isForSection()) {
        str << "  ckSend(impl_amsg, "<<epIdx()<<opts<<");\n";
      }
      else
        str << "  ckBroadcast(impl_amsg, "<<epIdx()<<opts<<");\n";
      }
    }
    if(isIget()) {
	    str << "  return f;\n";
    }
    str << "}\n";
  }
}

void Entry::genArrayStaticConstructorDecl(XStr& str)
{
  if (!container->isArray())
    die("Internal error - array declarations called for on non-array Chare type");

  if (container->getForWhom() == forIndividual)
      str<< //Element insertion routine
      "    void insert("<<paramComma(1,0)<<"int onPE=-1"<<eo(1)<<");";
  else if (container->getForWhom() == forAll) {
      //With options to specify size (including potentially empty, covering the param->isVoid() case)
      str << "    static CkArrayID ckNew(" << paramComma(1,0) << "const CkArrayOptions &opts = CkArrayOptions()" << eo(1) << ");\n";
      str << "    static void      ckNew(" << paramComma(1,0) << "const CkArrayOptions &opts, CkCallback _ck_array_creation_cb" << eo(1) << ");\n";

      XStr dim = ((Array*)container)->dim();
      if (dim == (const char*)"1D") {
        str << "    static CkArrayID ckNew(" << paramComma(1,0) << "const int s1" << eo(1)<<");\n";
        str << "    static void ckNew("      << paramComma(1,0) << "const int s1, CkCallback _ck_array_creation_cb" << eo(1) <<");\n";
      } else if (dim == (const char*)"2D") {
        str << "    static CkArrayID ckNew(" << paramComma(1,0) << "const int s1, const int s2"<<eo(1)<<");\n";
        str << "    static void ckNew("      << paramComma(1,0) << "const int s1, const int s2, CkCallback _ck_array_creation_cb" << eo(1) <<");\n";
      } else if (dim == (const char*)"3D") {
        str << "    static CkArrayID ckNew(" << paramComma(1,0) << "const int s1, const int s2, const int s3" << eo(1)<<");\n";
        str << "    static void ckNew("      << paramComma(1,0) << "const int s1, const int s2, const int s3, CkCallback _ck_array_creation_cb" << eo(1) <<");\n";
      /*} else if (dim==(const char*)"4D") {
        str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const short s1, const short s2, const short s3, const short s4"<<eo(1)<<");\n";
      } else if (dim==(const char*)"5D") {
        str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const short s1, const short s2, const short s3, const short s4, const short s5"<<eo(1)<<");\n";
      } else if (dim==(const char*)"6D") {
        str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const short s1, const short s2, const short s3, const short s4, const short s5, const short s6"<<eo(1)<<");\n"; */
      }
  }
  else if (container->getForWhom() == forSection) { }
}

void Entry::genArrayStaticConstructorDefs(XStr& str)
{
  if (!container->isArray())
    die("Internal error - array definitions called for on non-array Chare type");

  if (container->getForWhom() == forIndividual)
      str<<
      makeDecl("void",1)<<"::insert("<<paramComma(0,0)<<"int onPE"<<eo(0)<<")\n"
      "{ \n"<<marshallMsg()<<
      "   UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);\n"
      "   ckInsert((CkArrayMessage *)impl_msg,"<<epIdx()<<",onPE);\n}\n";
  else if (container->getForWhom() == forAll) {
    XStr syncPrototype, asyncPrototype, head, syncTail, asyncTail;
    syncPrototype << makeDecl("CkArrayID", 1) << "::ckNew";
    asyncPrototype << makeDecl("void", 1) << "::ckNew";

    head << "{\n"
         << marshallMsg();

    syncTail << "  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);\n"
         << "  CkArrayID gId = ckCreateArray((CkArrayMessage *)impl_msg, "
         << epIdx() << ", opts);\n";

    genTramInstantiation(syncTail);
    syncTail << "  return gId;\n}\n";

    asyncTail  << "  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);\n"
               << "  CkSendAsyncCreateArray(" << epIdx() << ", _ck_array_creation_cb, opts, impl_msg);\n"
               << "}\n";

    str << syncPrototype << "(" << paramComma(0) << "const CkArrayOptions &opts" << eo(0) << ")\n"
        << head << syncTail;
    str << asyncPrototype << "(" << paramComma(0) << "const CkArrayOptions &opts, CkCallback _ck_array_creation_cb"<< eo(0) << ")\n"
        << head << asyncTail;

    XStr dim = ((Array*)container)->dim();
    XStr sizeParams, sizeArgs;
    bool emit = true;

    if (dim == (const char*)"1D") {
      sizeParams << "const int s1";
      sizeArgs << "s1";
    } else if (dim == (const char*)"2D") {
      sizeParams << "const int s1, const int s2";
      sizeArgs << "s1, s2";
    } else if (dim == (const char*)"3D") {
      sizeParams << "const int s1, const int s2, const int s3";
      sizeArgs << "s1, s2, s3";
    }
#if 0
    else if (dim==(const char*)"4D") {
      sizeParams << "const short s1, const short s2, const short s3, const short s4";
      sizeArgs << "s1, s2, s3, s4";
    } else if (dim==(const char*)"5D") {
      sizeParams << "const short s1, const short s2, const short s3, const short s4, "
                 << "const short s5";
      sizeArgs << "s1, s2, s3, s4, s5";
    } else if (dim==(const char*)"6D") {
      sizeParams << "const short s1, const short s2, const short s3, const short s4, "
                 << "const short s5, const short s6";
      sizeArgs << "s1, s2, s3, s4, s5, s6";
    }
#endif
    else {
      emit = false;
    }

    if (emit) {
      str << syncPrototype << "(" << paramComma(0) << sizeParams << eo(0) << ")\n"
          << head << "  CkArrayOptions opts(" << sizeArgs << ");\n" << syncTail;
      str << asyncPrototype << "(" << paramComma(0) << sizeParams << ", CkCallback _ck_array_creation_cb" << eo(0) << ")\n"
          << head << "  CkArrayOptions opts(" << sizeArgs << ");\n" << asyncTail;
    }
  }
}


/******************************** Group Entry Points *********************************/

void Entry::genGroupDecl(XStr& str)
{
  if(isConstructor()) {
    str << "    " << generateTemplateSpec(tspec) << "\n";
    genGroupStaticConstructorDecl(str);
  } else {
    if ((isSync() || isLocal()) && !container->isForElement()) return; //No sync broadcast
    str << "    " << generateTemplateSpec(tspec) << "\n";
    if (isLocal())
      str << "    "<<retType<<" "<<name<<"("<<paramType(1,1,0)<<");\n";
    else
      str << "    "<<retType<<" "<<name<<"("<<paramType(1,1)<<");\n";
    // entry method on multiple PEs declaration
    if(!container->isForElement() && !container->isForSection() && !isSync() && !isLocal() && !container->isNodeGroup()) {
      str << "    " << generateTemplateSpec(tspec) << "\n";
      str << "    "<<retType<<" "<<name<<"("<<paramComma(1,0)<<"int npes, int *pes"<<eo(1)<<");\n";
      str << "    " << generateTemplateSpec(tspec) << "\n";
      str << "    "<<retType<<" "<<name<<"("<<paramComma(1,0)<<"CmiGroup &grp"<<eo(1)<<");\n";
    }
  }
}

void Entry::genGroupDefs(XStr& str)
{
  //Selects between NodeGroup and Group
  char *node = (char *)(container->isNodeGroup()?"Node":"");

  if(isConstructor()) {
    genGroupStaticConstructorDefs(str);
  } else {
    int forElement=container->isForElement();
    XStr params; params<<epIdx()<<", impl_msg";
    XStr paramg; paramg<<epIdx()<<", impl_msg, ckGetGroupID()";
    XStr parampg; parampg<<epIdx()<<", impl_msg, ckGetGroupPe(), ckGetGroupID()";
    // append options parameter
    XStr opts; opts<<",0";
    if (isImmediate()) opts << "+CK_MSG_IMMEDIATE";
    if (isInline())  opts << "+CK_MSG_INLINE";
    if (isSkipscheduler())  opts << "+CK_MSG_EXPEDITED";

    if ((isSync() || isLocal()) && !container->isForElement()) return; //No sync broadcast

    XStr retStr; retStr<<retType;
    XStr msgTypeStr;
    if (isLocal())
      msgTypeStr<<paramType(0,1,0);
    else
      msgTypeStr<<paramType(0,1);
    str << makeDecl(retStr,1)<<"::"<<name<<"("<<msgTypeStr<<")\n";
    str << "{\n  ckCheck();\n";
    if (!isLocal()) str <<marshallMsg();

    if (isLocal()) {
      XStr unmarshallStr; param->unmarshall(unmarshallStr);
      str << "  "<<container->baseName()<<" *obj = ckLocalBranch();\n";
      str << "  CkAssert(obj);\n";
      if (!isNoTrace()) str << "  _TRACE_BEGIN_EXECUTE_DETAILED(0,ForBocMsg,("<<epIdx()<<"),CkMyPe(),0,NULL, NULL);\n";
      if(isAppWork())
	str << " _TRACE_BEGIN_APPWORK();\n";
      str <<
	"#if CMK_LBDB_ON\n"
	"  // if there is a running obj being measured, stop it temporarily\n"
	"  LDObjHandle objHandle;\n"
	"  int objstopped = 0;\n"
	"  LBDatabase *the_lbdb = (LBDatabase *)CkLocalBranch(_lbdb);\n"
	"  if (the_lbdb->RunningObject(&objHandle)) {\n"
	"    objstopped = 1;\n"
	"    the_lbdb->ObjectStop(objHandle);\n"
	"  }\n"
	"#endif\n";
      str <<
	"#if CMK_CHARMDEBUG\n"
	"  CpdBeforeEp("<<epIdx()<<", obj, NULL);\n"
	"#endif\n  ";
      if (!retType->isVoid()) str << retType << " retValue = ";
      str << "obj->"<<name<<"("<<unmarshallStr<<");\n";
      str << "#if CMK_CHARMDEBUG\n"
	"  CpdAfterEp("<<epIdx()<<");\n"
	"#endif\n";
      str << "#if CMK_LBDB_ON\n"
	"  if (objstopped) the_lbdb->ObjectStart(objHandle);\n"
	"#endif\n";
      if(isAppWork())
	str << " _TRACE_END_APPWORK();\n";
      if (!isNoTrace()) str << "  _TRACE_END_EXECUTE();\n";
      if (!retType->isVoid()) str << "  return retValue;\n";
    } else if(isSync()) {
      str << syncPreCall() <<
        "CkRemote"<<node<<"BranchCall("<<paramg<<", ckGetGroupPe());\n";
      str << syncPostCall();
    } else { // Non-sync, non-local entry method
      if (forElement)
      {// Send
        str << "  if (ckIsDelegated()) {\n";
        str << "     Ck"<<node<<"GroupMsgPrep("<<paramg<<");\n";
        str << "     ckDelegatedTo()->"<<node<<"GroupSend(ckDelegatedPtr(),"<<parampg<<");\n";
        str << "  } else CkSendMsg"<<node<<"Branch"<<"("<<parampg<<opts<<");\n";
      }
      else if (container->isForSection())
      {// Multicast
        str << "  if (ckIsDelegated()) {\n";
        str << "     Ck"<<node<<"GroupMsgPrep("<<paramg<<");\n";
        str << "     ckDelegatedTo()->"<<node<<"GroupSectionSend(ckDelegatedPtr(),"<<params<<", ckGetNumSections(), ckGetSectionIDs());\n";
        str << "  } else {\n";
        str << "    void *impl_msg_tmp = (ckGetNumSections()>1) ? CkCopyMsg((void **) &impl_msg) : impl_msg;\n";
        str << "    for (int i=0; i<ckGetNumSections(); ++i) {\n";
        str << "       impl_msg_tmp= (i<ckGetNumSections()-1) ? CkCopyMsg((void **) &impl_msg):impl_msg;\n";
        str << "       CkSendMsg"<<node<<"BranchMulti("<<epIdx()<<", impl_msg_tmp, ckGetGroupIDn(i), ckGetNumElements(i), ckGetElements(i)"<<opts<<");\n";
        str << "    }\n";
        str << "  }\n";
      }
      else
      {// Broadcast
        str << "  if (ckIsDelegated()) {\n";
        str << "     Ck"<<node<<"GroupMsgPrep("<<paramg<<");\n";
        str << "     ckDelegatedTo()->"<<node<<"GroupBroadcast(ckDelegatedPtr(),"<<paramg<<");\n";
        str << "  } else CkBroadcastMsg"<<node<<"Branch("<<paramg<<opts<<");\n";
      }
    }
    str << "}\n";

    // entry method on multiple PEs declaration
    if(!forElement && !container->isForSection() && !isSync() && !isLocal() && !container->isNodeGroup()) {
      str << ""<<makeDecl(retStr,1)<<"::"<<name<<"("<<paramComma(1,0)<<"int npes, int *pes"<<eo(0)<<") {\n";
      str << marshallMsg();
      str << "  CkSendMsg"<<node<<"BranchMulti("<<paramg<<", npes, pes"<<opts<<");\n";
      str << "}\n";
      str << ""<<makeDecl(retStr,1)<<"::"<<name<<"("<<paramComma(1,0)<<"CmiGroup &grp"<<eo(0)<<") {\n";
      str << marshallMsg();
      str << "  CkSendMsg"<<node<<"BranchGroup("<<paramg<<", grp"<<opts<<");\n";
      str << "}\n";
    }
  }
}

XStr Entry::aggregatorIndexType() {
  XStr indexType;
  if (container->isGroup()) {
    indexType << "int";
  }
  else if (container->isArray()) {
    XStr dim, arrayIndexType;
    dim << ( (Array*) container)->dim();
    if (dim == "1D") {
      indexType << "int";
    }
    else {
      indexType << "CkArrayIndex";
    }
  }
  return indexType;
}

XStr Entry::dataItemType() {
  XStr itemType;
  if (container->isGroup()) {
    itemType << param->param->type;
  }
  else if (container->isArray()) {
    itemType << "ArrayDataItem< " << param->param->type << ", "
             << aggregatorIndexType() << " >";
  }
  return itemType;
}

XStr Entry::aggregatorType() {
  XStr groupType;
  if (container->isGroup()) {
    groupType << "GroupMeshStreamer<" << param->param->type
              << ", " << container->baseName() << ", SimpleMeshRouter"
              << ", " << container->indexName() << "::_callmarshall_" << epStr()
              << " >";
  }
  else if (container->isArray()) {
    groupType << "ArrayMeshStreamer<" << param->param->type
              << ", " << aggregatorIndexType() <<", "
              << container->baseName() << ", "
              << "SimpleMeshRouter, "
              << container->indexName() << "::_callmarshall_" << epStr()
              << " >";
  }
  return groupType;
}

XStr Entry::aggregatorName() {
  XStr aggregatorName;
  aggregatorName << epStr() << "TramAggregator";
  return aggregatorName;
}

void Entry::genTramTypes() {
  if (isTramTarget()) {
    XStr typeString, nameString, itemTypeString;
    typeString << aggregatorType();
    nameString << aggregatorName();
    itemTypeString << dataItemType();
    container->tramInstances.
      push_back(TramInfo(typeString.get_string(), nameString.get_string(),
                         itemTypeString.get_string()));
    tramInstanceIndex = container->tramInstances.size();
  }
}

void Entry::genTramDefs(XStr &str) {

  XStr retStr; retStr<<retType;
  XStr msgTypeStr;

  if (isLocal())
    msgTypeStr<<paramType(0,1,0);
  else
    msgTypeStr<<paramType(0,1);
  str << makeDecl(retStr,1)<<"::"<<name<<"("<<msgTypeStr<<") {\n"
      << "  if (" << aggregatorName() << " == NULL) {\n";

  if (container->isGroup()) {
    str << "    CkGroupID gId = ckGetGroupID();\n";
  }
  else if (container->isArray()) {
    str << "    CkArray *aMgr = ckLocalBranch();\n"
        << "    CkGroupID gId = aMgr->getGroupID();\n";
  }

  str  << "    CkGroupID tramGid;\n"
       << "    tramGid.idx = gId.idx + "<< tramInstanceIndex <<";\n"
       << "    " << aggregatorName() << " = (" << aggregatorType() << "*)"
       << " CkLocalBranch(tramGid);\n  }\n";

  if (container->isGroup()) {
    str << "  " << aggregatorName() << "->insertData(" << param->param->name
        << ", " << "ckGetGroupPe());\n}\n";
  }
  else if (container->isArray()) {
    XStr dim; dim << ((Array*)container)->dim();
    str << "  const CkArrayIndex &myIndex = ckGetIndex();\n"
        << "  " << aggregatorName() << "->insertData(" << param->param->name;
    if (dim==(const char*)"1D") {
      str << ", " << "myIndex.data()[0]);\n}\n";
    }
    else {
      str << ", " << "myIndex);\n}\n";
    }
  }
}

// size of TRAM buffers in bytes
const static int tramBufferSize = 16384;

void Entry::genTramInstantiation(XStr& str) {
  if (!container->tramInstances.empty()) {
    str << "  int pesPerNode = CkMyNodeSize();\n"
        << "  if (pesPerNode == 1) {\n"
        << "    pesPerNode = CmiNumCores();\n"
        << "  }\n"
        << "  int nDims = 2;\n"
        << "  int dims[nDims];\n"
        << "  dims[0] = CkNumPes() / pesPerNode;\n"
        << "  dims[1] = pesPerNode;\n"
        << "  if (dims[0] * dims[1] != CkNumPes()) {\n"
        << "    dims[0] = CkNumPes();\n"
        << "    dims[1] = 1;\n"
        << "  }\n"
        << "  int tramBufferSize = " << tramBufferSize <<";\n";
    for (int i = 0; i < container->tramInstances.size(); i++) {
      str << "  {\n"
          << "  int itemsPerBuffer = tramBufferSize / sizeof("
          << container->tramInstances[i].itemType.c_str() <<");\n"
          << "  if (itemsPerBuffer == 0) {\n"
          << "    itemsPerBuffer = 1;\n"
          << "  };\n"
          << "  CProxy_" << container->tramInstances[i].type.c_str()
          << " tramProxy =\n"
          << "  CProxy_" << container->tramInstances[i].type.c_str()
          << "::ckNew(2, dims, gId, itemsPerBuffer, false, 10.0);\n"
          << "  tramProxy.enablePeriodicFlushing();\n"
          << "  }";
    }
  }
}

XStr Entry::tramBaseType()
{
  XStr baseTypeString;
  baseTypeString << "MeshStreamer<" << dataItemType()
                 << ", SimpleMeshRouter >";

  return baseTypeString;
}

void Entry::genTramRegs(XStr& str)
{
  if (isTramTarget()) {
    XStr messageTypeString;
    messageTypeString << "MeshStreamerMessage< " << dataItemType() << " >";

    XStr baseTypeString = tramBaseType();

    NamedType messageType(messageTypeString.get_string());
    Message helper(-1, &messageType);
    helper.genReg(str);

    str << "\n  /* REG: group " << aggregatorType() << ": IrrGroup;\n  */\n"
        << "  CkIndex_" << aggregatorType() << "::__register(\""
        << aggregatorType() << "\", sizeof(" << aggregatorType() << "));\n"
        << "  /* REG: group " << baseTypeString << ": IrrGroup;\n  */\n"
        << "  CkIndex_" << baseTypeString << "::__register(\""
        << baseTypeString << "\", sizeof(" << baseTypeString << "));\n";

  }
}

void Entry::genTramPups(XStr& decls, XStr& defs)
{
  if (isTramTarget()) {
    XStr aggregatorTypeString = aggregatorType();
    container->genRecursivePup(aggregatorTypeString, "template <>\n", decls, defs);
  }
}

void Entry::genGroupStaticConstructorDecl(XStr& str)
{
  if (container->isForElement()) return;
  if (container->isForSection()) return;

  str << "    static CkGroupID ckNew("<<paramType(1,1)<<");\n";
  if (!param->isVoid()) {
    str << "    "<<container->proxyName(0)<<"("<<paramType(1,1)<<");\n";
  }
}

void Entry::genGroupStaticConstructorDefs(XStr& str)
{
  if (container->isForElement()) return;
  if (container->isForSection()) return;

  //Selects between NodeGroup and Group
  char *node = (char *)(container->isNodeGroup()?"Node":"");
  str << makeDecl("CkGroupID",1)<<"::ckNew("<<paramType(0,1)<<")\n";
  str << "{\n"<<marshallMsg();
  str << "  UsrToEnv(impl_msg)->setMsgtype(" << node << "BocInitMsg);\n";
  if (param->isMarshalled()) {
    str << "  if (impl_e_opts)\n";
    str << "    UsrToEnv(impl_msg)->setGroupDep(impl_e_opts->getGroupDepID());\n";
  }
  str << "  CkGroupID gId = CkCreate"<<node<<"Group("<<chareIdx()<<", "<<epIdx()<<", impl_msg);\n";

  genTramInstantiation(str);

  str << "  return gId;\n";
  str << "}\n";

  if (!param->isVoid()) {
    str << makeDecl(" ",1)<<"::"<<container->proxyName(0)<<"("<<paramType(0,1)<<")\n";
    str << "{\n"<<marshallMsg();
    str << "  UsrToEnv(impl_msg)->setMsgtype(" << node << "BocInitMsg);\n";
    if (param->isMarshalled()) {
      str << "  if (impl_e_opts)\n";
      str << "    UsrToEnv(impl_msg)->setGroupDep(impl_e_opts->getGroupDepID());\n";
    }
    str << "  ckSetGroupID(CkCreate"<<node<<"Group("<<chareIdx()<<", "<<epIdx()<<", impl_msg));\n";
    str << "}\n";
  }
}

/******************* Python Entry Point Code **************************/
void Entry::genPythonDecls(XStr& str) {
  str <<"/* STATIC DECLS: "; print(str); str << " */\n";
  if (isPython()) {
    str << "PyObject *_Python_"<<container->baseName()<<"_"<<name<<"(PyObject *self, PyObject *arg);\n";
  }
}

void Entry::genPythonDefs(XStr& str) {
  str <<"/* DEFS: "; print(str); str << " */\n";
  if (isPython()) {

    str << "PyObject *_Python_"<<container->baseName()<<"_"<<name<<"(PyObject *self, PyObject *arg) {\n";
    str << "  PyObject *dict = PyModule_GetDict(PyImport_AddModule(\"__main__\"));\n";
    str << "  int pyNumber = PyInt_AsLong(PyDict_GetItemString(dict,\"__charmNumber__\"));\n";
    str << "  PythonObject *pythonObj = (PythonObject *)PyLong_AsVoidPtr(PyDict_GetItemString(dict,\"__charmObject__\"));\n";
    str << "  "<<container->baseName()<<" *object = static_cast<"<<container->baseName()<<" *>(pythonObj);\n";
    str << "  object->pyWorkers[pyNumber].arg=arg;\n";
    str << "  object->pyWorkers[pyNumber].result=&CtvAccess(pythonReturnValue);\n";
    str << "  object->pyWorkers[pyNumber].pythread=PyThreadState_Get();\n";
    str << "  CtvAccess(pythonReturnValue) = 0;\n";

    str << "  //pyWorker->thisProxy."<<name<<"(pyNumber);\n";
    str << "  object->"<<name<<"(pyNumber);\n";

    str << "  //CthSuspend();\n";

    str << "  if (CtvAccess(pythonReturnValue)) {\n";
    str << "    return CtvAccess(pythonReturnValue);\n";
    str << "  } else {\n";
    str << "    Py_INCREF(Py_None); return Py_None;\n";
    str << "  }\n";
    str << "}\n";
  }
}

void Entry::genPythonStaticDefs(XStr& str) {
  if (isPython()) {
    str << "  {\""<<name<<"\",_Python_"<<container->baseName()<<"_"<<name<<",METH_VARARGS},\n";
  }
}

void Entry::genPythonStaticDocs(XStr& str) {
  if (isPython()) {
    str << "\n  \""<<name<<" -- \"";
    if (pythonDoc) str <<(char*)pythonDoc;
    str <<"\"\\\\n\"";
  }
}


/******************* Accelerator (Accel) Entry Point Code ********************/

void Entry::genAccelFullParamList(XStr& str, int makeRefs) {

  if (!isAccel()) return;

  ParamList* curParam = NULL;
  int isFirst = 1;

  // Parameters (which are read only by default)
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {

    if (!isFirst) { str << ", "; }

    Parameter* param = curParam->param;

    if (param->isArray()) {
      str << param->getType()->getBaseName() << "* " << param->getName();
    } else {
      str << param->getType()->getBaseName() << " " << param->getName();
    }

    isFirst = 0;
    curParam = curParam->next;
  }

  // Accel parameters
  curParam = accelParam;
  while (curParam != NULL) {

    if (!isFirst) { str << ", "; }

    Parameter* param = curParam->param;
    int bufType = param->getAccelBufferType();
    int needWrite = makeRefs && ((bufType == Parameter::ACCEL_BUFFER_TYPE_READWRITE) || (bufType == Parameter::ACCEL_BUFFER_TYPE_WRITEONLY));
    if (param->isArray()) {
      str << param->getType()->getBaseName() << "* " << param->getName();
    } else {
      str << param->getType()->getBaseName() << ((needWrite) ? (" &") : (" ")) << param->getName();
    }

    isFirst = 0;
    curParam = curParam->next;
  }

  // Implied object pointer
  if (!isFirst) { str << ", "; }
  str << container->baseName() << "* impl_obj";
}

void Entry::genAccelFullCallList(XStr& str) {
  if (!isAccel()) return;

  int isFirstFlag = 1;

  // Marshalled parameters to entry method
  ParamList* curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    if (!isFirstFlag) str << ", ";
    isFirstFlag = 0;
    str << curParam->param->getName();
    curParam = curParam->next;
  }

  // General variables (prefix with "impl_obj->" for member variables of the current object)
  curParam = accelParam;
  while (curParam != NULL) {
    if (!isFirstFlag) str << ", ";
    isFirstFlag = 0;
    str << (*(curParam->param->getAccelInstName()));
    curParam = curParam->next;
  }

  // Implied object
  if (!isFirstFlag) str << ", ";
  isFirstFlag = 0;
  str << "impl_obj";
}

void Entry::genAccelIndexWrapperDecl_general(XStr& str) {
  str << "    static void _accelCall_general_" << epStr() << "(";
  genAccelFullParamList(str, 1);
  str << ");\n";
}

void Entry::genAccelIndexWrapperDef_general(XStr& str) {
  str << makeDecl("void") << "::_accelCall_general_" << epStr() << "(";
  genAccelFullParamList(str, 1);
  str << ") {\n\n";

  //// DMK - DEBUG
  //str << "  // DMK - DEBUG\n";
  //str << "  CkPrintf(\"[DEBUG-ACCEL] :: [PPE] - "
  //    << makeDecl("void") << "::_accelCall_general_" << epStr()
  //    << "(...) - Called...\\n\");\n\n";

  str << (*accelCodeBody);

  str << "\n\n";
  str << "  impl_obj->" << (*accelCallbackName) << "();\n";
  str << "}\n";
}

void Entry::genAccelIndexWrapperDecl_spe(XStr& str) {

  // Function to issue work request
  str << "    static void _accelCall_spe_" << epStr() << "(";
  genAccelFullParamList(str, 0);
  str << ");\n";

  // Callback function that is a member of CkIndex_xxx
  str << "    static void _accelCall_spe_callback_" << epStr() << "(void* userPtr);\n";
}

// DMK - Accel Support
#if CMK_CELL != 0
  #include "spert.h"
#endif

void Entry::genAccelIndexWrapperDef_spe(XStr& str) {

  XStr containerType = container->baseName();

  // Some blank space for readability
  str << "\n\n";


  ///// Generate struct that will be passed to callback function /////

  str << "typedef struct __spe_callback_struct_" << epStr() << " {\n"
      << "  " << containerType << "* impl_obj;\n"
      << "  WRHandle wrHandle;\n"
      << "  void* scalar_buf_ptr;\n";

  // Pointers for marshalled parameter buffers
  ParamList* curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    if (curParam->param->isArray()) {
      str << "  void* param_buf_ptr_" << curParam->param->getName() << ";\n";
    }
    curParam = curParam->next;
  }
  curParam = accelParam;
  while (curParam != NULL) {
    if (curParam->param->isArray()) {
      str << "  void* accelParam_buf_ptr_" << curParam->param->getName() << ";\n";
    }
    curParam = curParam->next;
  }

  str << "} SpeCallbackStruct_" << epStr() << ";\n\n";


  ///// Generate callback function /////

  str << "void _accelCall_spe_callback_" << container->baseName() << "_" << epStr() << "(void* userPtr) {\n"
      << "  " << container->indexName() << "::_accelCall_spe_callback_" << epStr() << "(userPtr);\n"
      << "}\n";

  str << makeDecl("void") << "::_accelCall_spe_callback_" << epStr() << "(void* userPtr) {\n";
  str << "  SpeCallbackStruct_" << epStr() << "* cbStruct = (SpeCallbackStruct_" << epStr() << "*)userPtr;\n";
  str << "  " << containerType << "* impl_obj = cbStruct->impl_obj;\n";

  // Write scalars that are 'out' or 'inout' from the scalar buffer back into memory

  if (accel_numScalars > 0) {

    // Get the pointer to the scalar buffer
    int dmaList_scalarBufIndex = 0;
    if (accel_dmaList_scalarNeedsWrite) {
      dmaList_scalarBufIndex += accel_dmaList_numReadOnly;
    }
    str << "  char* __scalar_buf_offset = (char*)(cbStruct->scalar_buf_ptr);\n";

    // Parameters
    curParam = param;
    if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
    while (curParam != NULL) {
      if (!(curParam->param->isArray())) {
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }

    // Read only accel parameters
    curParam = accelParam;
    while (curParam != NULL) {
      if ((!(curParam->param->isArray())) && (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READONLY)) {
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }

    // Read write accel parameters
    curParam = accelParam;
    while (curParam != NULL) {
      if ((!(curParam->param->isArray())) && (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READWRITE)) {
        str << "  " << (*(curParam->param->getAccelInstName())) << " = *((" << curParam->param->getType()->getBaseName() << "*)__scalar_buf_offset);\n";
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }

    // Write only accel parameters
    curParam = accelParam;
    while (curParam != NULL) {
      if ((!(curParam->param->isArray())) && (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_WRITEONLY)) {
        str << "  " << (*(curParam->param->getAccelInstName())) << " = *((" << curParam->param->getType()->getBaseName() << "*)__scalar_buf_offset);\n";
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }
  }

  // Call the callback function
  str << "  (cbStruct->impl_obj)->" << (*accelCallbackName) << "();\n";

  // Free memory
  str << "  if (cbStruct->scalar_buf_ptr != NULL) { free_aligned(cbStruct->scalar_buf_ptr); }\n";
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    if (curParam->param->isArray()) {
      str << "  if (cbStruct->param_buf_ptr_" << curParam->param->getName() << " != NULL) { "
          <<      "free_aligned(cbStruct->param_buf_ptr_" << curParam->param->getName() << "); "
          <<   "}\n";
    }
    curParam = curParam->next;
  }
  str << "  delete cbStruct;\n";

  str << "}\n\n";


  ///// Generate function to issue work request /////

  str << makeDecl("void") << "::_accelCall_spe_" << epStr() << "(";
  genAccelFullParamList(str, 0);
  str << ") {\n\n";

  //// DMK - DEBUG
  //str << "  // DMK - DEBUG\n"
  //    << "  CkPrintf(\"[DEBUG-ACCEL] :: [PPE] - "
  //    << makeDecl("void") << "::_accelCall_spe_" << epStr()
  //    << "(...) - Called... (funcIndex:%d)\\n\", accel_spe_func_index__" << epStr() << ");\n\n";


  str << "  // Allocate a user structure to be passed to the callback function\n"
      << "  SpeCallbackStruct_" << epStr() << "* cbStruct = new SpeCallbackStruct_" << epStr() << ";\n"
      << "  cbStruct->impl_obj = impl_obj;\n"
      << "  cbStruct->wrHandle = INVALID_WRHandle;  // NOTE: Set actual value later...\n"
      << "  cbStruct->scalar_buf_ptr = NULL;\n";
  // Set all parameter buffer pointers in the callback structure to NULL
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    if (curParam->param->isArray()) {
      str << "  cbStruct->param_buf_ptr_" << curParam->param->getName() << " = NULL;\n";
    }
    curParam = curParam->next;
  }
  curParam = accelParam;
  while (curParam != NULL) {
    if (curParam->param->isArray()) {
      str << "  cbStruct->accelParam_buf_ptr_" << curParam->param->getName() << " = NULL;\n";
    }
    curParam = curParam->next;
  }
  str << "\n";


  // Create the DMA list
  int dmaList_curIndex = 0;
  int numDMAListEntries = accel_numArrays;
  if (accel_numScalars > 0) { numDMAListEntries++; }
  if (numDMAListEntries <= 0) {
    XLAT_ERROR_NOCOL("accel entry with no parameters",
                     first_line_);
  }

  // DMK - NOTE : TODO : FIXME - For now, force DMA lists to only be the static length or less.
  //   Fix this in the future to handle any length supported by hardware.  Also, for now,
  //   #if this check since non-Cell architectures do not have SPE_DMA_LIST_LENGTH defined and
  //   this code should not be called unless this is a Cell architecture.
  #if CMK_CELL != 0
  if (numDMAListEntries > SPE_DMA_LIST_LENGTH) {
    die("Accel entries do not support parameter lists of length > SPE_DMA_LIST_LENGTH yet... fix me...");
  }
  #endif

  // Do a pass of all the parameters, determine the size of all scalars (to pack them)
  if (accel_numScalars > 0) {
    str << "  // Create a single buffer to hold all the scalar values\n";
    str << "  int scalar_buf_len = 0;\n";
    curParam = param;
    if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
    while (curParam != NULL) {
      if (!(curParam->param->isArray())) {
        str << "  scalar_buf_len += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }
    curParam = accelParam;
    while (curParam != NULL) {
      if (!(curParam->param->isArray())) {
        str << "  scalar_buf_len += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }
    str << "  scalar_buf_len = ROUNDUP_128(scalar_buf_len);\n"
        << "  cbStruct->scalar_buf_ptr = malloc_aligned(scalar_buf_len, 128);\n"
        << "  char* scalar_buf_offset = (char*)(cbStruct->scalar_buf_ptr);\n\n";
  }


  // Declare the DMA list
  str << "  // Declare and populate the DMA list for the work request\n";
  str << "  DMAListEntry dmaList[" << numDMAListEntries << "];\n\n";


  // Parameters: read only by default & arrays need to be copied since message will be deleted
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {

    // Check to see if the scalar buffer needs slipped into the dma list here
    if (accel_numScalars > 0) {
      if (((dmaList_curIndex == 0) && (!(accel_dmaList_scalarNeedsWrite))) ||
          ((dmaList_curIndex == accel_dmaList_numReadOnly) && (accel_dmaList_scalarNeedsWrite))
	 ) {

        str << "  /*** Scalar Buffer ***/\n"
            << "  dmaList[" << dmaList_curIndex << "].size = scalar_buf_len;\n"
	    << "  dmaList[" << dmaList_curIndex << "].ea = (unsigned int)(cbStruct->scalar_buf_ptr);\n\n";
        dmaList_curIndex++;
      }
    }

    // Add this parameter to the dma list (somehow)
    str << "  /*** Param: '" << curParam->param->getName() << "' ***/\n";
    if (curParam->param->isArray()) {
      str << "  {\n"
	  << "    int bufSize = sizeof(" << curParam->param->getType()->getBaseName() << ") * (" << curParam->param->getArrayLen() << ");\n"
	  << "    bufSize = ROUNDUP_128(bufSize);\n"
          << "    cbStruct->param_buf_ptr_" << curParam->param->getName() << " = malloc_aligned(bufSize, 128);\n"
	  << "    memcpy(cbStruct->param_buf_ptr_" << curParam->param->getName() << ", " << curParam->param->getName() << ", bufSize);\n"
	  << "    dmaList[" << dmaList_curIndex << "].size = bufSize;\n"
	  << "    dmaList[" << dmaList_curIndex << "].ea = (unsigned int)(cbStruct->param_buf_ptr_" << curParam->param->getName() << ");\n"
	  << "  }\n";
      dmaList_curIndex++;
    } else {
      str << "  *((" << curParam->param->getType()->getBaseName() << "*)scalar_buf_offset) = "
          << curParam->param->getName() << ";\n"
          << "  scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
    }
    curParam = curParam->next;
    str << "\n";
  }

  // Read only accel params
  curParam = accelParam;
  while (curParam != NULL) {

    // Check to see if the scalar buffer needs slipped into the dma list here
    if (accel_numScalars > 0) {
      if (((dmaList_curIndex == 0) && (!(accel_dmaList_scalarNeedsWrite))) ||
          ((dmaList_curIndex == accel_dmaList_numReadOnly) && (accel_dmaList_scalarNeedsWrite))
	 ) {

        str << "  /*** Scalar Buffer ***/\n"
            << "  dmaList[" << dmaList_curIndex << "].size = scalar_buf_len;\n"
	    << "  dmaList[" << dmaList_curIndex << "].ea = (unsigned int)(cbStruct->scalar_buf_ptr);\n\n";
        dmaList_curIndex++;
      }
    }

    // Add this parameter
    if (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READONLY) {
      str << "  /*** Accel Param: '" << curParam->param->getName() << " ("
          << (*(curParam->param->getAccelInstName())) << ")' ***/\n";
      if (curParam->param->isArray()) {
        str << "  dmaList[" << dmaList_curIndex << "].size = ROUNDUP_128("
	    << "sizeof(" << curParam->param->getType()->getBaseName() << ") * "
            << "(" << curParam->param->getArrayLen() << "));\n"
  	    << "  dmaList[" << dmaList_curIndex << "].ea = (unsigned int)(" << (*(curParam->param->getAccelInstName())) << ");\n";
        dmaList_curIndex++;
      } else {
        str << "  *((" << curParam->param->getType()->getBaseName() << "*)scalar_buf_offset) = "
            << (*(curParam->param->getAccelInstName())) << ";\n"
            << "  scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      str << "\n";
    }

    curParam = curParam->next;
  }

  // Read/write accel params
  curParam = accelParam;
  while (curParam != NULL) {

    // Check to see if the scalar buffer needs slipped into the dma list here
    if (accel_numScalars > 0) {
      if (((dmaList_curIndex == 0) && (!(accel_dmaList_scalarNeedsWrite))) ||
          ((dmaList_curIndex == accel_dmaList_numReadOnly) && (accel_dmaList_scalarNeedsWrite))
	 ) {

        str << "  /*** Scalar Buffer ***/\n"
            << "  dmaList[" << dmaList_curIndex << "].size = scalar_buf_len;\n"
	    << "  dmaList[" << dmaList_curIndex << "].ea = (unsigned int)(cbStruct->scalar_buf_ptr);\n\n";
        dmaList_curIndex++;
      }
    }

    // Add this parameter
    if (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READWRITE) {
      str << "  /*** Accel Param: '" << curParam->param->getName() << " ("
          << (*(curParam->param->getAccelInstName())) << ")' ***/\n";
      if (curParam->param->isArray()) {
        str << "  dmaList[" << dmaList_curIndex << "].size = ROUNDUP_128("
	    << "sizeof(" << curParam->param->getType()->getBaseName() << ") * "
            << "(" << curParam->param->getArrayLen() << "));\n"
  	    << "  dmaList[" << dmaList_curIndex << "].ea = (unsigned int)(" << (*(curParam->param->getAccelInstName())) << ");\n";
        dmaList_curIndex++;
      } else {
        str << "  *((" << curParam->param->getType()->getBaseName() << "*)scalar_buf_offset) = "
            << (*(curParam->param->getAccelInstName())) << ";\n"
            << "  scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      str << "\n";
    }

    curParam = curParam->next;
  }

  // Write only accel params
  curParam = accelParam;
  while (curParam != NULL) {

    // Add this parameter
    if (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_WRITEONLY) {
      str << "  /*** Accel Param: '" << curParam->param->getName() << " ("
          << (*(curParam->param->getAccelInstName())) << ")' ***/\n";
      if (curParam->param->isArray()) {
        str << "  dmaList[" << dmaList_curIndex << "].size = ROUNDUP_128("
	    << "sizeof(" << curParam->param->getType()->getBaseName() << ") * "
            << "(" << curParam->param->getArrayLen() << "));\n"
  	    << "  dmaList[" << dmaList_curIndex << "].ea = (unsigned int)(" << (*(curParam->param->getAccelInstName())) << ");\n";
        dmaList_curIndex++;
      } else {
        str << "  *((" << curParam->param->getType()->getBaseName() << "*)scalar_buf_offset) = "
            << (*(curParam->param->getAccelInstName())) << ";\n"
            << "  scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      str << "\n";
    }

    curParam = curParam->next;
  }

  str << "  // Issue the work request\n";
  str << "  cbStruct->wrHandle = sendWorkRequest_list(accel_spe_func_index__" << epStr() << ",\n"
      << "                                            0,\n"
      << "                                            dmaList,\n"
      << "                                            " << accel_dmaList_numReadOnly << ",\n"
      << "                                            " << accel_dmaList_numReadWrite << ",\n"
      << "                                            " << accel_dmaList_numWriteOnly << ",\n"
      << "                                            cbStruct,\n"
      << "                                            WORK_REQUEST_FLAGS_NONE,\n"
      << "                                            _accelCall_spe_callback_" << container->baseName() << "_" << epStr() << "\n"
      << "                                           );\n";

  str << "}\n\n";


  // Some blank space for readability
  str << "\n";
}

int Entry::genAccels_spe_c_funcBodies(XStr& str) {

  // Make sure this is an accelerated entry method (just return if not)
  if (!isAccel()) { return 0; }

  // Declare the spe function
  str << "void __speFunc__" << indexName() << "__" << epStr() << "(DMAListEntry* dmaList) {\n";

  ParamList* curParam = NULL;
  int dmaList_curIndex = 0;

  // Identify the scalar buffer if there is one
  if (accel_numScalars > 0) {
    if (accel_dmaList_scalarNeedsWrite) {
      str << "  void* __scalar_buf_ptr = (void*)(dmaList[" << accel_dmaList_numReadOnly << "].ea);\n";
    } else {
      str << "  void* __scalar_buf_ptr = (void*)(dmaList[0].ea);\n";
      dmaList_curIndex++;
    }
    str << "  char* __scalar_buf_offset = (char*)(__scalar_buf_ptr);\n";
  }

  // Pull out all the parameters
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    if (curParam->param->isArray()) {
      str << "  " << curParam->param->getType()->getBaseName() << "* " << curParam->param->getName() << " = (" << curParam->param->getType()->getBaseName() << "*)(dmaList[" << dmaList_curIndex << "].ea);\n";
      dmaList_curIndex++;
    } else {
      str << "  " << curParam->param->getType()->getBaseName() << " " << curParam->param->getName() << " = *((" << curParam->param->getType()->getBaseName() << "*)__scalar_buf_offset);\n";
      str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
    }
    curParam = curParam->next;
  }

  // Read only accel params
  curParam = accelParam;
  while (curParam != NULL) {
    if (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READONLY) {
      if (curParam->param->isArray()) {
        str << "  " << curParam->param->getType()->getBaseName() << "* " << curParam->param->getName() << " = (" << curParam->param->getType()->getBaseName() << "*)(dmaList[" << dmaList_curIndex << "].ea);\n";
        dmaList_curIndex++;
      } else {
        str << "  " << curParam->param->getType()->getBaseName() << " " << curParam->param->getName() << " = *((" << curParam->param->getType()->getBaseName() << "*)__scalar_buf_offset);\n";
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
    }
    curParam = curParam->next;
  }

  // Reset the dmaList_curIndex to the read-write portion of the dmaList
  dmaList_curIndex = accel_dmaList_numReadOnly;
  if ((accel_numScalars > 0) && (accel_dmaList_scalarNeedsWrite)) {
    dmaList_curIndex++;
  }

  // Read-write accel params
  curParam = accelParam;
  while (curParam != NULL) {
    if (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READWRITE) {
      if (curParam->param->isArray()) {
        str << "  " << curParam->param->getType()->getBaseName() << "* " << curParam->param->getName() << " = (" << curParam->param->getType()->getBaseName() << "*)(dmaList[" << dmaList_curIndex << "].ea);\n";
        dmaList_curIndex++;
      } else {
        str << "  " << curParam->param->getType()->getBaseName() << " " << curParam->param->getName() << " = *((" << curParam->param->getType()->getBaseName() << "*)__scalar_buf_offset);\n";
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
    }
    curParam = curParam->next;
  }

  // Write only accel params
  curParam = accelParam;
  while (curParam != NULL) {
    if (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_WRITEONLY) {
      if (curParam->param->isArray()) {
        str << "  " << curParam->param->getType()->getBaseName() << "* " << curParam->param->getName() << " = (" << curParam->param->getType()->getBaseName() << "*)(dmaList[" << dmaList_curIndex << "].ea);\n";
        dmaList_curIndex++;
      } else {
        str << "  " << curParam->param->getType()->getBaseName() << " " << curParam->param->getName() << " = *((" << curParam->param->getType()->getBaseName() << "*)__scalar_buf_offset);\n";
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
    }
    curParam = curParam->next;
  }


  // Function body from the interface file
  str << "  {\n    " << (*accelCodeBody) << "\n  }\n";


  // Write the scalar values that are not read only back into the scalar buffer
  if ((accel_numScalars > 0) && (accel_dmaList_scalarNeedsWrite)) {

    str << "  __scalar_buf_offset = (char*)(__scalar_buf_ptr);\n";

    // Parameters
    curParam = param;
    if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
    while (curParam != NULL) {
      if (!(curParam->param->isArray())) {
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }

    // Read only accel parameters
    curParam = accelParam;
    while (curParam != NULL) {
      if ((!(curParam->param->isArray())) && (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READONLY)) {
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }

    // Read only accel parameters
    curParam = accelParam;
    while (curParam != NULL) {
      if ((!(curParam->param->isArray())) && (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READWRITE)) {
        str << "  *((" << curParam->param->getType()->getBaseName() << "*)__scalar_buf_offset) = " << curParam->param->getName() << ";\n";
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }

    // Read only accel parameters
    curParam = accelParam;
    while (curParam != NULL) {
      if ((!(curParam->param->isArray())) && (curParam->param->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_WRITEONLY)) {
        str << "  *((" << curParam->param->getType()->getBaseName() << "*)__scalar_buf_offset) = " << curParam->param->getName() << ";\n";
        str << "  __scalar_buf_offset += sizeof(" << curParam->param->getType()->getBaseName() << ");\n";
      }
      curParam = curParam->next;
    }

  }

  str << "}\n\n\n";

  return 1;
}

void Entry::genAccels_spe_c_regFuncs(XStr& str) {
  if (isAccel()) {
    str << "  funcLookupTable[curIndex  ].funcIndex = curIndex;\n"
        << "  funcLookupTable[curIndex++].funcPtr = __speFunc__" << indexName() << "__" << epStr() << ";\n";
  }
}

void Entry::genAccels_ppe_c_regFuncs(XStr& str) {
  if (isAccel()) {
    str << "  " << indexName() << "::accel_spe_func_index__" << epStr() << " = curIndex++;\n";
  }
}


/******************* Shared Entry Point Code **************************/
void Entry::genIndexDecls(XStr& str)
{
  str << "    /* DECLS: "; print(str); str << "     */";

  XStr templateSpecLine;
  templateSpecLine << "\n    " << generateTemplateSpec(tspec);

  // Entry point index storage
  str << "\n    // Entry point registration at startup"
      << templateSpecLine
      << "\n    static int reg_" << epStr() << "();" ///< @note: Should this be generated as private?
      << "\n    // Entry point index lookup"
      << templateSpecLine
      << "\n    inline static int idx_" << epStr() << "() {"
      << "\n      static int epidx = " << epRegFn(0) << ";"
      << "\n      return epidx;"
      << "\n    }\n";

  if (!isConstructor()) {
    str << templateSpecLine
        << "\n    inline static int idx_" << name << "("
        << retType
        << " (" << container->baseName() << "::*)(";
    if (param)
      param->print(str);
    str << ") ) {"
        << "\n      return " << epIdx(0) << ";"
        << "\n    }\n\n";
  }

  // DMK - Accel Support - Also declare the function index for the Offload API call
  #if CMK_CELL != 0
    if (isAccel()) {
      str << "    static int accel_spe_func_index__" << epStr() << ";\n";
    }
  #endif

  // Index function, so user can find the entry point number
  str << templateSpecLine
      << "\n    static int ";
  if (isConstructor()) str <<"ckNew";
  else str <<name;
  str << "("<<paramType(1,0)<<") { return "<<epIdx(0)<<"; }";

  // DMK - Accel Support
  if (isAccel()) {
    genAccelIndexWrapperDecl_general(str);
    #if CMK_CELL != 0
      genAccelIndexWrapperDecl_spe(str);
    #endif
  }

  if (isReductionTarget()) {
      str << "\n    // Entry point registration at startup"
          << templateSpecLine
          << "\n    static int reg_"<< epStr(true) <<"();" ///< @note: Should this be generated as private?
          << "\n    // Entry point index lookup"
          << templateSpecLine
          << "\n    inline static int idx_" << epStr(true) << "() {"
          << "\n      static int epidx = "<< epRegFn(0, true) <<";"
          << "\n      return epidx;"
          << "\n    }"
          << templateSpecLine
          << "\n    static int " << "redn_wrapper_" << name
          << "(CkReductionMsg* impl_msg) { return " << epIdx(0, true) << "; }"
          << templateSpecLine
          << "\n    static void _call_" << epStr(true) << "(void* impl_msg, void* impl_obj_void);";
  }

  // call function declaration
  str << templateSpecLine
      << "\n    static void _call_" << epStr() << "(void* impl_msg, void* impl_obj);";
  str << templateSpecLine
      << "\n    static void _call_sdag_" << epStr() << "(void* impl_msg, void* impl_obj);";
  if(isThreaded()) {
    str  << templateSpecLine
         << "\n    static void _callthr_"<<epStr()<<"(CkThrCallArg *);";
  }
  if (hasCallMarshall) {
    str << templateSpecLine
        << "\n    static int _callmarshall_" << epStr()
        << "(char* impl_buf, void* impl_obj_void);";
  }
  if (param->isMarshalled()) {
    str << templateSpecLine
        << "\n    static void _marshallmessagepup_"<<epStr()<<"(PUP::er &p,void *msg);";
  }
  str << "\n";
}

void Entry::genDecls(XStr& str)
{
  if (external)
    return;

  str << "/* DECLS: "; print(str); str << " */\n";

  if (isMigrationConstructor())
    {} //User cannot call the migration constructor
  else if(container->isGroup()) {
    genGroupDecl(str);
  } else if(container->isArray()) {
    if(!isIget())
      genArrayDecl(str);
    else if(container->isForElement())
      genArrayDecl(str);
  } else { // chare or mainchare
    genChareDecl(str);
  }
}

void Entry::genClosureEntryDecls(XStr& str) {
  genClosure(str, false);
}

void Entry::genClosureEntryDefs(XStr& str) {
  templateGuardBegin(tspec || container->isTemplated(), str);
  genClosure(str, true);
  templateGuardEnd(str);
}

void Entry::genClosure(XStr& decls, bool isDef) {
  if (isConstructor() || (isLocal() && !sdagCon)) return;

  bool hasArray = false, isMessage = false;
  XStr messageType;
  int i = 0;
  XStr structure, toPup, alloc, getter, dealloc;
  for(ParamList* pl = param; pl != NULL; pl = pl->next, i++) {
    Parameter* sv = pl->param;

    if (XStr(sv->type->getBaseName()) == "CkArrayOptions") continue;

    structure << "      ";
    getter << "      ";

    if ((sv->isMessage() != 1) && (sv->isVoid() != 1)) {
       structure << sv->type << " ";
       getter << sv->type << " ";
       if (sv->isArray() != 0) {
         structure << "*";
         getter << "*";
       }

       if (sv->isArray() != 0) {
         hasArray = hasArray || true;
       } else {
         toPup << "        " << "__p | " << sv->name << ";\n";
         sv->podType = true;
       }

       if (sv->name != 0) {
         structure << sv->name << ";\n";
         getter << "& " << "getP" << i << "() { return " << sv->name << ";}\n";
       }

    }
    else if (sv->isVoid() != 1){
      if (sv->isMessage()) {
        isMessage = true;
        structure << sv->type << " " << sv->name << ";\n";
        toPup << "        " << "CkPupMessage(__p, (void**)&" << sv->name << ");\n";
        messageType << sv->type->deref();
      }
    }
  }

  structure << "\n";

  toPup << "        packClosure(__p);\n";

  XStr initCode;
  initCode << "        init();\n";

  if (hasArray) {
    structure << "      " << "CkMarshallMsg* _impl_marshall;\n";
    structure << "      " << "char* _impl_buf_in;\n";
    structure << "      " << "int _impl_buf_size;\n";
    dealloc << "        if (_impl_marshall) CmiFree(UsrToEnv(_impl_marshall));\n";

    initCode << "        _impl_marshall = 0;\n";
    initCode << "        _impl_buf_in = 0;\n";
    initCode << "        _impl_buf_size = 0;\n";

    toPup << "        __p | _impl_buf_size;\n";
    toPup << "        bool hasMsg = (_impl_marshall != 0); __p | hasMsg;\n";
    toPup << "        " << "if (hasMsg) CkPupMessage(__p, (void**)&" << "_impl_marshall" << ");\n";
    toPup << "        " << "else PUParray(__p, _impl_buf_in, _impl_buf_size);\n";
    toPup << "        if (__p.isUnpacking()) {\n";
    toPup << "          char *impl_buf = _impl_marshall ? _impl_marshall->msgBuf : _impl_buf_in;\n";
    param->beginUnmarshallSDAG(toPup);
    toPup << "        }\n";
  }

  // Generate code for ensuring we don't migrate active local closures
  if (isLocal()) {
    toPup.clear();
    toPup << "        CkAbort(\"Can\'t migrate while a local SDAG method is active.\");\n";
  }

  if (!isMessage) {
    genClosureTypeName = new XStr();
    genClosureTypeNameProxy = new XStr();
    *genClosureTypeNameProxy << "Closure_" << container->baseName() << "::";
    *genClosureTypeNameProxy << name << "_" << entryCount << "_closure";
    *genClosureTypeName << name << "_" << entryCount << "_closure";

    container->sdagPUPReg << "  PUPable_reg(SINGLE_ARG(" << *genClosureTypeNameProxy << "));\n";

    if (isDef) {
      if (container->isTemplated()) {
        decls << container->tspec(false) << "\n";
      }
      decls << generateTemplateSpec(tspec) << "\n";
      decls << "    struct " << *genClosureTypeNameProxy <<" : public SDAG::Closure" << " {\n";
      decls << structure << "\n";
      decls << "      " << *genClosureTypeName << "() {\n";
      decls << initCode;
      decls << "      }\n";
      decls << "      " << *genClosureTypeName << "(CkMigrateMessage*) {\n";
      decls << initCode;
      decls << "      }\n";
      decls << getter;
      decls << "      void pup(PUP::er& __p) {\n";
      decls << toPup;
      decls << "      }\n";
      decls << "      virtual ~" << *genClosureTypeName << "() {\n";
      decls << dealloc;
      decls << "      }\n";
      decls << "      " << ((container->isTemplated() || tspec) ? "PUPable_decl_template" : "PUPable_decl") << "(SINGLE_ARG(" << *genClosureTypeName;
      if (tspec) {
        decls << "<";
        tspec->genShort(decls);
        decls << ">";
      }
      decls << "));\n";
      decls << "    };\n";
    } else {
      decls << generateTemplateSpec(tspec) << "\n";
      decls << "    struct " <<  *genClosureTypeName << ";\n";
    }
  } else {
    genClosureTypeName = new XStr();
    genClosureTypeNameProxy = new XStr();
    *genClosureTypeNameProxy << messageType;
    *genClosureTypeName << messageType;
  }

  genClosureTypeNameProxyTemp = new XStr();
  *genClosureTypeNameProxyTemp << (container->isTemplated() ? "typename " : "") << genClosureTypeNameProxy;
}

//This routine is only used in Entry::genDefs.
// It ends the current procedure with a call to awaken another thread,
// and defines the thread function to handle that call.
XStr Entry::callThread(const XStr &procName,int prependEntryName)
{
  XStr str,procFull;
  procFull<<"_callthr_";
  if(prependEntryName) procFull<<name<<"_";
  procFull<<procName;

  str << "  CthThread tid = CthCreate((CthVoidFn)"<<procFull
   <<", new CkThrCallArg(impl_msg,impl_obj), "<<getStackSize()<<");\n";
  str << "  ((Chare *)impl_obj)->CkAddThreadListeners(tid,impl_msg);\n";
  // str << "  CkpvAccess(_traces)->CkAddThreadListeners(tid);\n";
#if CMK_BIGSIM_CHARM
  str << "  BgAttach(tid);\n";
#endif
  str << "  CthResume(tid);\n";
  str << "}\n";

  str << makeDecl("void")<<"::"<<procFull<<"(CkThrCallArg *impl_arg)\n";
  str << "{\n";\
  str << "  void *impl_msg = impl_arg->msg;\n";
  str << "  "<<container->baseName()<<" *impl_obj = ("<<container->baseName()<<" *) impl_arg->obj;\n";
  str << "  delete impl_arg;\n";
  return str;
}

/*
  Generate the code to actually unmarshall the parameters and call
  the entry method.
*/
void Entry::genCall(XStr& str, const XStr &preCall, bool redn_wrapper, bool usesImplBuf)
{
  bool isArgcArgv=false;
  bool isMigMain=false;
  bool isSDAGGen = sdagCon || isWhenEntry;

  if (isConstructor() && container->isMainChare() &&
      (!param->isVoid()) && (!param->isCkArgMsgPtr())){
  	if(param->isCkMigMsgPtr()) isMigMain = true;
	else isArgcArgv = true;
  } else {
    //Normal case: Unmarshall variables
    if (redn_wrapper) param->beginRednWrapperUnmarshall(str, isSDAGGen);
    else {
      if (isSDAGGen)
        param->beginUnmarshallSDAGCall(str, usesImplBuf);
      else
        param->beginUnmarshall(str);

      if (param->isVoid() && !isNoKeep())
        str<<"  CkFreeSysMsg(impl_msg);\n";
    }
  }

  str << preCall;
  if (!isConstructor() && fortranMode) {
    if (!container->isArray()) { // Currently, only arrays are supported
      cerr << (char *)container->baseName() << ": only chare arrays are currently supported\n";
      exit(1);
    }
    str << "/* FORTRAN */\n";
    XStr dim; dim << ((Array*)container)->dim();
    if (dim==(const char*)"1D")
      str << "  int index1 = impl_obj->thisIndex;\n";
    else if (dim==(const char*)"2D") {
      str << "  int index1 = impl_obj->thisIndex.x;\n";
      str << "  int index2 = impl_obj->thisIndex.y;\n";
    }
    else if (dim==(const char*)"3D") {
      str << "  int index1 = impl_obj->thisIndex.x;\n";
      str << "  int index2 = impl_obj->thisIndex.y;\n";
      str << "  int index3 = impl_obj->thisIndex.z;\n";
    }
    str << "  ::" << fortranify(name)
	<< "((char **)(impl_obj->user_data), &index1";
    if (dim==(const char*)"2D" || dim==(const char*)"3D")
        str << ", &index2";
    if (dim==(const char*)"3D")
        str << ", &index3";
    if (!param->isVoid()) { str << ", "; param->unmarshallAddress(str); }
    str<<");\n";
    str << "/* FORTRAN END */\n";
  }

  // DMK : Accel Support
  else if (isAccel()) {

    #if CMK_CELL != 0
      str << "  if (1) {   // DMK : TODO : For now, hardcode the condition (i.e. for now, do not dynamically load-balance between host and accelerator)\n";
      str << "    _accelCall_spe_" << epStr() << "(";
      genAccelFullCallList(str);
      str << ");\n";
      str << "  } else {\n  ";
    #endif

    str << "  _accelCall_general_" << epStr() << "(";
    genAccelFullCallList(str);
    str << ");\n";

    #if CMK_CELL != 0
      str << "  }\n";
    #endif

  }

  else { //Normal case: call regular method
    if (isArgcArgv) str<<"  CkArgMsg *m=(CkArgMsg *)impl_msg;\n"; //Hack!

    if(isConstructor()) {//Constructor: call "new (obj) foo(parameters)"
  	str << "  new (impl_obj) "<<container->baseName();
    } else {//Regular entry method: call "obj->bar(parameters)"
      str << "  impl_obj->" << (tspec ? "template " : "") << name;
      if (tspec) {
        str << "< ";
        tspec->genShort(str);
        str << " >";
      }
    }

    if (isArgcArgv) { //Extract parameters from CkArgMsg (should be parameter marshalled)
        str<<"(m->argc,m->argv);\n";
        str<<"  delete m;\n";
    }else if(isMigMain){
        str<<"((CkMigrateMessage*)impl_msg);\n";
    }
    else {//Normal case: unmarshall parameters (or just pass message)
      if (isSDAGGen) {
        str << "(";
        if (param->isMessage()) {
          param->unmarshall(str);
          //} else if (param->isVoid()) {
          // no parameter
        } else if (!param->isVoid()) {
	  str << "genClosure";
        }
        str << ");\n";
	if (!param->isMessage() && !param->isVoid())
	  str << "  genClosure->deref();\n";
      } else {
        str<<"("; param->unmarshall(str); str<<");\n";
      }
    }
  }
}

void Entry::genDefs(XStr& str)
{
  if (external)
    return;
  XStr containerType=container->baseName();
  XStr preMarshall,preCall,postCall;

  templateGuardBegin(tspec || container->isTemplated(), str);
  str << "/* DEFS: "; print(str); str << " */\n";

  if(isMigrationConstructor())
    {} //User cannot call the migration constructor
  else if (isTramTarget() && container->isForElement()) {
    genTramDefs(str);
  }
  else if(container->isGroup()){
    genGroupDefs(str);
  } else if(container->isArray()) {
    genArrayDefs(str);
  } else
    genChareDefs(str);

  if (container->isMainChare() || container->isChare() || container->isForElement()) {
      if (isReductionTarget()) {
          XStr retStr; retStr<<retType;
          str << makeDecl(retStr);
          //str << retType << " " << indexName(); //makeDecl(retStr, 1)
          str << "::_call_" << epStr(true) << "(void* impl_msg, void* impl_obj_void)"
              << "\n{"
              << "\n  " << container->baseName() << "* impl_obj = static_cast<"
              << container->baseName() << "*> (impl_obj_void);\n"
              << "  char* impl_buf = (char*)((CkReductionMsg*)impl_msg)->getData();\n";
          XStr precall;
          genCall(str, precall, true, false);
          if (!(sdagCon || isWhenEntry))
            str << "  delete (CkReductionMsg*)impl_msg;\n}\n\n";
          else
            str << "  \n}\n\n";
      }
  }


  //Prevents repeated call and __idx definitions:
  if (container->getForWhom()!=forAll) {
    templateGuardEnd(str);
    return;
  }

  // Define the entry point registration functions
  str << "\n// Entry point registration function"
      << "\n" << makeDecl("int") << "::reg_" << epStr() << "() {"
      << "\n  int epidx = " << genRegEp() << ";";
  if (hasCallMarshall)
    str << "\n  CkRegisterMarshallUnpackFn(epidx, "
        << "_callmarshall_" << epStr(false, true) << ");";
  if (param->isMarshalled()) {
    str << "\n  CkRegisterMessagePupFn(epidx, "
        << "_marshallmessagepup_" << epStr(false, true) << ");\n";
  }
  else if (param->isMessage() && !isMigrationConstructor()) {
    str << "\n  CkRegisterMessagePupFn(epidx, (CkMessagePupFn)";
    param->param->getType()->deref()->print(str);
    str << "::ckDebugPup);";
  }
  str << "\n  return epidx;"
      << "\n}\n\n";

  if (isReductionTarget())
  {
    str << "\n// Redn wrapper registration function"
        << "\n" << makeDecl("int") << "::reg_"<< epStr(true) <<"() {"
        << "\n  return " << genRegEp(true) << ";"
        << "\n}\n\n";
  }

  // DMK - Accel Support
  #if CMK_CELL != 0
    if (isAccel()) {
      str << "int " << indexName() << "::" << "accel_spe_func_index__" << epStr() << "=0;\n";
    }
  #endif

  // Add special pre- and post- call code
  if(isSync() || isIget()) {
  //A synchronous method can return a value, and must finish before
  // the caller can proceed.
    preMarshall<< "  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);\n";
    if(retType->isVoid() || retType->isMessage())
      preCall<< "  void *impl_retMsg=";
    if(retType->isVoid()) {
      preCall << "CkAllocSysMsg();\n  ";
    } else if(retType->isMessage()){
      preCall << "(void *) ";
    } else{
      preCall<< "  "<<retType<<" impl_ret_val= ";  
      postCall<<"  //Marshall: impl_ret_val\n";
      postCall<<"  int impl_ret_size=0;\n";
      postCall<<"  { //Find the size of the PUP'd data\n";
      postCall<<"    PUP::sizer implPS;\n";
      postCall<<"    implPS|impl_ret_val;\n";
      postCall<<"    impl_ret_size+=implPS.size();\n";
      postCall<<"  };\n";
      postCall<<"  CkMarshallMsg *impl_retMsg=CkAllocateMarshallMsg(impl_ret_size, NULL);\n";
      postCall<<"  { //Copy over the PUP'd data;\n";
      postCall<<"    PUP::toMem implPS((void *)impl_retMsg->msgBuf);\n";
      postCall<<"    implPS|impl_ret_val;\n";
      postCall<<"  };\n";
    }
    postCall << "  CkSendToFutureID(impl_ref, impl_retMsg, impl_src);\n";
  } else if(isExclusive()) {
  //An exclusive method
    preMarshall << "  if(CmiTryLock(impl_obj->__nodelock)) {\n"; /*Resend msg. if lock busy*/
    /******* DANGER-- RESEND CODE UNTESTED **********/
    if (param->isMarshalled()) {
      preMarshall << "    impl_msg = CkCopyMsg(&impl_msg);\n";
    }
    preMarshall << "    CkSendMsgNodeBranch("<<epIdx()<<",impl_msg,CkMyNode(),impl_obj->CkGetNodeGroupID());\n";
    preMarshall << "    return;\n";
    preMarshall << "  }\n";

    postCall << "  CmiUnlock(impl_obj->__nodelock);\n";
  }

  if (!isConstructor() && fortranMode) { // Fortran90
      str << "/* FORTRAN SECTION */\n";

      XStr dim; dim << ((Array*)container)->dim();
      // Declare the Fortran Entry Function
      // This is called from C++
      str << "extern \"C\" void " << fortranify(name) << "(char **, " << container->indexList();
      if (!param->isVoid()) { str << ", "; param->printAddress(str); }
      str << ");\n";

      // Define the Fortran interface function
      // This is called from Fortran to send the message to a chare.
      str << "extern \"C\" void "
        //<< container->proxyName() << "_"
          << fortranify("SendTo_", container->baseName(), "_", name)
          << "(long* aindex, " << container->indexList();
      if (!param->isVoid()) { str << ", "; param->printAddress(str); }
      str << ")\n";
      str << "{\n";
      str << "  CkArrayID *aid = (CkArrayID *)*aindex;\n";
      str << "\n";
      str << "  " << container->proxyName() << " h(*aid);\n";
      str << "  if (*index1 == -1) \n";
      str << "    h." << name << "(";
      if (!param->isVoid()) param->printValue(str);
      str << ");\n";
      str << "  else\n";
      if (dim==(const char*)"1D")
        str << "    h[*index1]." << name << "(";
      else if (dim==(const char*)"2D")
        str << "    h[CkArrayIndex2D(*index1, *index2)]." << name << "(";
      else if (dim==(const char*)"3D")
        str << "    h[CkArrayIndex3D(*index1, *index2, *index3)]." << name << "(";
      if (!param->isVoid()) param->printValue(str);
      str << ");\n";
      str << "}\n";
      str << "/* FORTRAN SECTION END */\n";
    }

  // DMK - Accel Support
  //   Create the wrapper function for the acceleration call
  //   TODO : FIXME : For now, just use the standard C++ code... create OffloadAPI wrappers later
  if (isAccel()) {
    genAccelIndexWrapperDef_general(str);
    #if CMK_CELL != 0
      genAccelIndexWrapperDef_spe(str);
    #endif
  }

  //Generate the call-method body
  str << makeDecl("void")<<"::_call_"<<epStr()<<"(void* impl_msg, void* impl_obj_void)\n";
  str << "{\n"
      << "  " << container->baseName() << "* impl_obj = static_cast<"
      << container->baseName() << " *>(impl_obj_void);\n";
  if (!isLocal()) {
    if(isThreaded()) str << callThread(epStr());
    str << preMarshall;
    if (param->isMarshalled()) {
      if (param->hasConditional()) str << "  MarshallMsg_"<<epStr()<<" *impl_msg_typed=(MarshallMsg_"<<epStr()<<" *)impl_msg;\n";
      else str << "  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;\n";
      str << "  char *impl_buf=impl_msg_typed->msgBuf;\n";
    }
    genCall(str, preCall, false, false);
    param->endUnmarshall(str);
    str << postCall;
    if(isThreaded() && param->isMarshalled()) str << "  delete impl_msg_typed;\n";
  } else {
    str << "  CkAbort(\"This method should never be called as it refers to a LOCAL entry method!\");\n";
  }
  str << "}\n";

  if (hasCallMarshall) {
    str << makeDecl("int") << "::_callmarshall_" << epStr()
        <<"(char* impl_buf, void* impl_obj_void) {\n";
    str << "  " << containerType << "* impl_obj = static_cast< " << containerType << " *>(impl_obj_void);\n";
    if (!isLocal()) {
      if (!param->hasConditional()) {
        genCall(str, preCall, false, true);
        /*FIXME: implP.size() is wrong if the parameter list contains arrays--
        need to add in the size of the arrays.
         */
        str << "  return implP.size();\n";
      } else {
        str << "  CkAbort(\"This method is not implemented for EPs using conditional packing\");\n";
        str << "  return 0;\n";
      }
    } else {
      str << "  CkAbort(\"This method should never be called as it refers to a LOCAL entry method!\");\n";
      str << "  return 0;\n";
    }
    str << "}\n";
  }
  if (param->isMarshalled()) {
     str << makeDecl("void")<<"::_marshallmessagepup_"<<epStr()<<"(PUP::er &implDestP,void *impl_msg) {\n";
     if (!isLocal()) {
       if (param->hasConditional()) str << "  MarshallMsg_"<<epStr()<<" *impl_msg_typed=(MarshallMsg_"<<epStr()<<" *)impl_msg;\n";
       else str << "  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;\n";
       str << "  char *impl_buf=impl_msg_typed->msgBuf;\n";
       param->beginUnmarshall(str);
       param->pupAllValues(str);
     } else {
       str << "  /*Fake pupping since we don't really have a message */\n";
       str << "  int n=0;\n";
       str << "  if (implDestP.hasComments()) implDestP.comment(\"LOCAL message\");\n";
       str << "  implDestP|n;\n";
     }
     str << "}\n";
  }

  // to match the registry, generate register call even if there is no SDAG code
  //if ((param->isMarshalled() || param->isVoid()) /* && (sdagCon || isWhenEntry) */)
  if ((param->isMarshalled() || param->isVoid()) && genClosureTypeNameProxy) {
    if (container->isTemplated())
      str << container->tspec(false);
    if (tspec) {
      str << "template <";
      tspec->genLong(str, false);
      str << "> ";
    }

    str << ((container->isTemplated() || tspec) ? "PUPable_def_template_nonInst" : "PUPable_def") << "(SINGLE_ARG(" << *genClosureTypeNameProxy;
    if (tspec) {
      str << "<";
      tspec->genShort(str);
      str << ">";
    }
    str << "))\n";
  }

  templateGuardEnd(str);
}

XStr Entry::genRegEp(bool isForRedn)
{
  XStr str;
  str << "CkRegisterEp(\"";
  if (isForRedn)
      str << "redn_wrapper_" << name << "(CkReductionMsg *impl_msg)\",\n";
  else
      str << name << "("<<paramType(0)<<")\",\n";
  str << "      _call_" << epStr(isForRedn, true);
  str << ", ";
  /* messageIdx: */
  if (param->isMarshalled()) {
    if (param->hasConditional())  str<<"MarshallMsg_"<<epStr()<<"::__idx";
    else str<<"CkMarshallMsg::__idx";
  } else if(!param->isVoid() && !isMigrationConstructor()) {
    param->genMsgProxyName(str);
    str <<"::__idx";
  } else if (isForRedn) {
    str << "CMessage_CkReductionMsg::__idx";
  } else {
    str << "0";
  }
  /* chareIdx */
  str << ", __idx";
  /* attributes */
  str << ", 0";
  // reductiontarget variants should not be nokeep. The actual ep will be
  // parameter marshalled (and hence flagged as nokeep), but we'll delete the
  // CkReductionMsg in generated code, not runtime code. (so that we can cast
  // it to CkReductionMsg not CkMarshallMsg)
  if ( !isForRedn && (attribs & SNOKEEP) ) str << "+CK_EP_NOKEEP";
  if (attribs & SNOTRACE) str << "+CK_EP_TRACEDISABLE";
  if (attribs & SIMMEDIATE) str << "+CK_EP_TRACEDISABLE";
  if (attribs & SAPPWORK) str << "+CK_EP_APPWORK";

  /*MEICHAO*/
  if (attribs & SMEM) str << "+CK_EP_MEMCRITICAL";
  
  if (internalMode) str << "+CK_EP_INTRINSIC";
  str << ")";
  return str;
}

void Entry::genReg(XStr& str)
{
  if (tspec)
    return;

  if (external) {
    str << "  CkIndex_" << label << "::idx_" << name;
    if (targs)
        str << "< " << targs << " >";
    str << "( static_cast< "
        << retType << " (" << label << "::*)(" << paramType(0,0) << ") >(NULL) );\n";
    return;
  }

  str << "  // REG: "<<*this;
  str << "  " << epIdx(0) << ";\n";
  if (isReductionTarget())
    str << "  " << epIdx(0, true) << ";\n";
  if (isConstructor()) {
    if(container->isMainChare() && !isMigrationConstructor())
      str << "  CkRegisterMainChare(__idx, "<<epIdx(0)<<");\n";
    if(param->isVoid())
      str << "  CkRegisterDefaultCtor(__idx, "<<epIdx(0)<<");\n";
    if(isMigrationConstructor())
      str << "  CkRegisterMigCtor(__idx, "<<epIdx(0)<<");\n";
  }
}

void Entry::preprocess() {
  ParamList *pl = param;
  if (pl != NULL && pl->hasConditional()) {
    XStr str;
    str << "MarshallMsg_" << epStr();
    NamedType *nt = new NamedType(strdup(str));
    MsgVar *var = new MsgVar(new BuiltinType("char"), "msgBuf", 0, 1);
    MsgVarList *list = new MsgVarList(var);
    do {
      if (pl->param->isConditional()) {
        var = new MsgVar(pl->param->getType(), pl->param->getName(), 1, 0);
        list = new MsgVarList(var, list);
      }
    } while (NULL!=(pl=pl->next));
    Message *m = new Message(-1, nt, list);
    m->setModule(container->containerModule);
    container->containerModule->prependConstruct(m);
  }

  // DMK - Accel Support
  // Count the total number of scalar and array parameters if this is an accelerated entry
  accel_numScalars = 0;
  accel_numArrays = 0;
  accel_dmaList_numReadOnly = 0;
  accel_dmaList_numReadWrite = 0;
  accel_dmaList_numWriteOnly = 0;
  accel_dmaList_scalarNeedsWrite = 0;
  if (isAccel()) {
    ParamList* curParam = param;
    if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
    while (curParam != NULL) {
      if (curParam->param->isArray()) {
        accel_numArrays++;
        accel_dmaList_numReadOnly++;
      } else {
        accel_numScalars++;
      }
      curParam = curParam->next;
    }
    curParam = accelParam;
    while (curParam != NULL) {
      if (curParam->param->isArray()) {
        accel_numArrays++;
        switch (curParam->param->getAccelBufferType()) {
          case Parameter::ACCEL_BUFFER_TYPE_READWRITE:  accel_dmaList_numReadWrite++;  break;
          case Parameter::ACCEL_BUFFER_TYPE_READONLY:   accel_dmaList_numReadOnly++;   break;
          case Parameter::ACCEL_BUFFER_TYPE_WRITEONLY:  accel_dmaList_numWriteOnly++;  break;
          default:     XLAT_ERROR_NOCOL("unknown accel param type", first_line_);      break;
        }
      } else {
        accel_numScalars++;
        switch (curParam->param->getAccelBufferType()) {
          case Parameter::ACCEL_BUFFER_TYPE_READWRITE:  accel_dmaList_scalarNeedsWrite++;  break;
          case Parameter::ACCEL_BUFFER_TYPE_READONLY:                                      break;
          case Parameter::ACCEL_BUFFER_TYPE_WRITEONLY:  accel_dmaList_scalarNeedsWrite++;  break;
          default:     XLAT_ERROR_NOCOL("unknown accel param type", first_line_);          break;
        }
      }
      curParam = curParam->next;
    }
    if (accel_numScalars > 0) {
      if (accel_dmaList_scalarNeedsWrite) {
        accel_dmaList_numReadWrite++;
      } else {
        accel_dmaList_numReadOnly++;
      }
    }
  }
}


int Entry::paramIsMarshalled(void) {
    return param->isMarshalled();
}

int Entry::getStackSize(void) {
    return (stacksize ? stacksize->getIntVal() : 0);
}

void Entry::setAccelParam(ParamList* apl) { accelParam = apl; }
void Entry::setAccelCodeBody(XStr* acb) { accelCodeBody = acb; }
void Entry::setAccelCallbackName(XStr* acbn) { accelCallbackName = acbn; }

int Entry::isThreaded(void) { return (attribs & STHREADED); }
int Entry::isSync(void) { return (attribs & SSYNC); }
int Entry::isIget(void) { return (attribs & SIGET); }
int Entry::isConstructor(void) { return !strcmp(name, container->baseName(0).get_string());}
bool Entry::isMigrationConstructor() { return isConstructor() && (attribs & SMIGRATE); }
int Entry::isExclusive(void) { return (attribs & SLOCKED); }
int Entry::isImmediate(void) { return (attribs & SIMMEDIATE); }
int Entry::isSkipscheduler(void) { return (attribs & SSKIPSCHED); }
int Entry::isInline(void) { return attribs & SINLINE; }
int Entry::isLocal(void) { return attribs & SLOCAL; }
int Entry::isCreate(void) { return (attribs & SCREATEHERE)||(attribs & SCREATEHOME); }
int Entry::isCreateHome(void) { return (attribs & SCREATEHOME); }
int Entry::isCreateHere(void) { return (attribs & SCREATEHERE); }
int Entry::isPython(void) { return (attribs & SPYTHON); }
int Entry::isNoTrace(void) { return (attribs & SNOTRACE); }
int Entry::isAppWork(void) { return (attribs & SAPPWORK); }
int Entry::isNoKeep(void) { return (attribs & SNOKEEP); }
int Entry::isSdag(void) { return (sdagCon!=0); }
int Entry::isTramTarget(void) { return (attribs & SAGGREGATE); }

// DMK - Accel support
int Entry::isAccel(void) { return (attribs & SACCEL); }

int Entry::isMemCritical(void) { return (attribs & SMEM); }
int Entry::isReductionTarget(void) { return (attribs & SREDUCE); }

char *Entry::getEntryName() { return name; }
int Entry::getLine() { return line; }

}   // namespace xi
