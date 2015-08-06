#include "xi-Entry.h"
#include "xi-Parameter.h"
#include "xi-Value.h"
#include "xi-SdagCollection.h"
#include "xi-Chare.h"

#include "sdag/constructs/When.h"

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

  if (isSync() && retType && !(retType->isVoid() || retType->isMessage()))
    XLAT_ERROR_NOCOL("sync methods must return either void or a message",
                     first_line_);

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
	if (isSdag()) container->setSdag(1);
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

void Entry::collectSdagCode(WhenStatementEChecker *wsec)
{
  if (isSdag()) {
    wsec->addNode(this);
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

XStr Entry::syncReturn(void) {
  XStr str;
  if(retType->isVoid())
    str << "  CkFreeSysMsg(";
  else
    str << "  return ("<<retType<<") (";
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
  if (isImmediate()) {
      cerr << (char *)container->baseName() << ": Chare does not allow immediate message.\n";
      exit(1);
  }
  if (isLocal()) {
    cerr << (char*)container->baseName() << ": Chare does not allow LOCAL entry methods.\n";
    exit(1);
  }

  if(isConstructor()) {
    genChareStaticConstructorDefs(str);
  } else {
    XStr params; params<<epIdx()<<", impl_msg, &ckGetChareID()";
    // entry method definition
    XStr retStr; retStr<<retType;
    str << makeDecl(retStr,1)<<"::"<<name<<"("<<paramType(0,1)<<")\n";
    str << "{\n  ckCheck();\n"<<marshallMsg();
    if(isSync()) {
      str << syncReturn() << "CkRemoteCall("<<params<<"));\n";
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
  if (isImmediate()) {
      cerr << (char *)container->baseName() << ": Chare Array does not allow immediate message.\n";
      exit(1);
  }

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
    if (!isLocal()) {
      str << marshallMsg();
      str << "  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);\n";
      str << "  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;\n";
      str << "  impl_amsg->array_setIfNotThere("<<ifNot<<");\n";
    } else {
      XStr unmarshallStr; param->unmarshall(unmarshallStr);
      str << "  LDObjHandle objHandle;\n  int objstopped=0;\n";
      str << "  "<<container->baseName()<<" *obj = ckLocal();\n";
      str << "#if CMK_ERROR_CHECKING\n";
      str << "  if (obj==NULL) CkAbort(\"Trying to call a LOCAL entry method on a non-local element\");\n";
      str << "#endif\n";
      if (!isNoTrace())
	  str << "  _TRACE_BEGIN_EXECUTE_DETAILED(0,ForArrayEltMsg,(" << epIdx()
	      << "),CkMyPe(), 0, ((CkArrayIndex&)ckGetIndex()).getProjectionID(((CkGroupID)ckGetArrayID()).idx));\n";
      if(isAppWork())
      str << " _TRACE_BEGIN_APPWORK();\n";
      str << "#if CMK_LBDB_ON\n  objHandle = obj->timingBeforeCall(&objstopped);\n#endif\n";
      str << "#if CMK_CHARMDEBUG\n"
      "  CpdBeforeEp("<<epIdx()<<", obj, NULL);\n"
      "#endif\n   ";
      if (!retType->isVoid()) str << retType<< " retValue = ";
      str << "obj->"<<name<<"("<<unmarshallStr<<");\n";
      str << "#if CMK_CHARMDEBUG\n"
      "  CpdAfterEp("<<epIdx()<<");\n"
      "#endif\n";
      str << "#if CMK_LBDB_ON\n  obj->timingAfterCall(objHandle,&objstopped);\n#endif\n";
      if(isAppWork())
      str << " _TRACE_END_APPWORK();\n";
      if (!isNoTrace()) str << "  _TRACE_END_EXECUTE();\n";
      if (!retType->isVoid()) str << "  return retValue;\n";
    }
    if(isIget()) {
	    str << "  CkFutureID f=CkCreateAttachedFutureSend(impl_amsg,"<<epIdx()<<",ckGetArrayID(),ckGetIndex(),&CProxyElement_ArrayBase::ckSendWrapper);"<<"\n";
    }

    if(isSync()) {
      str << syncReturn() << "ckSendSync(impl_amsg, "<<epIdx()<<"));\n";
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
         << "  return ckCreateArray((CkArrayMessage *)impl_msg, "
         << epIdx() << ", opts);\n"
       "}\n";

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
#if 0
  if (isImmediate() && !container->isNodeGroup()) {
      cerr << (char *)container->baseName() << ": Group does not allow immediate message.\n";
      exit(1);
  }
#endif
  if (isLocal() && container->isNodeGroup()) {
    cerr << (char*)container->baseName() << ": Nodegroup does not allow LOCAL entry methods.\n";
    exit(1);
  }

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
      if (!isNoTrace()) str << "  _TRACE_BEGIN_EXECUTE_DETAILED(0,ForBocMsg,("<<epIdx()<<"),CkMyPe(),0,NULL);\n";
      if(isAppWork())
      str << " _TRACE_BEGIN_APPWORK();\n";
      str << "#if CMK_LBDB_ON\n"
"  // if there is a running obj being measured, stop it temporarily\n"
"  LDObjHandle objHandle;\n"
"  int objstopped = 0;\n"
"  LBDatabase *the_lbdb = (LBDatabase *)CkLocalBranch(_lbdb);\n"
"  if (the_lbdb->RunningObject(&objHandle)) {\n"
"    objstopped = 1;\n"
"    the_lbdb->ObjectStop(objHandle);\n"
"  }\n"
"#endif\n";
      str << "#if CMK_CHARMDEBUG\n"
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
      str << " _TRACE_BEGIN_APPWORK();\n";
      if (!isNoTrace()) str << "  _TRACE_END_EXECUTE();\n";
      if (!retType->isVoid()) str << "  return retValue;\n";
    } else if(isSync()) {
      str << syncReturn() <<
        "CkRemote"<<node<<"BranchCall("<<paramg<<", ckGetGroupPe()));\n";
    }
    else
    { //Non-sync entry method
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
  str << "  return CkCreate"<<node<<"Group("<<chareIdx()<<", "<<epIdx()<<", impl_msg);\n";
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
   #if (CMK_CUDA != 0) || (CMK_CELL != 0)
    str << "    static void _accelCall_generalDelayRestart_" << epStr() << "(void *ptr);\n";
    str << "    static void _accelCall_generalDelay_" << epStr() << "(";
    genAccelFullParamList(str, 1);
    str << ");\n";
    str << "    static int accel_cuda_func_index__" << epStr() << ";\n";
  #endif

  str << "\n";

}

/* This code might be useful, need to fix this: Harshit
  str << "    static void _accelCall_general_" << epStr() << "(";
  genf (isAccel()) {
     str << "  // AEM : \"" << container->baseName() << "::" << epStr() << "\"\n"
         << "  // Check to see if there is enough room in the table (increase the size of the table if not) and\n"
         << "  //   and then add the entry for \"" << container->baseName() << "::" << epStr() << "\".\n"
         << "  if (curIndex >= funcLookupTable_maxLen) {\n"
         << "    int newTableLen = funcLookupTable_maxLen + 16;\n"
         << "    FuncLookupTableEntry *newTable = new FuncLookupTableEntry[newTableLen];\n"
         << "    if (funcLookupTable_maxLen > 0 && funcLookupTable != NULL) {\n"
         << "      memcpy(newTable, funcLookupTable, sizeof(FuncLookupTableEntry) * curIndex);\n"
         << "      delete [] funcLookupTable;\n"
         << "    }\n"
         << "    funcLookupTable = newTable;\n"
         << "    funcLookupTable_maxLen = newTableLen;\n"
         << "  }\n"
         << "  funcLookupTable[curIndex  ].funcIndex = curIndex;\n"
 << "  funcLookupTable[curIndex  ].funcIndex = curIndex;\n"
         << "  funcLookupTable[curIndex++].funcPtr = __cudaFuncHost__" << container->baseName() << "__" << epStr() << ";\n\n";
   }
 }*/


void Entry::genAccels_cuda_c_regFuncs(XStr& str) {
   //if (isAccel() && isTriggered()) {
   if (isAccel()) {
     str << "  // AEM : \"" << container->baseName() << "::" << epStr() << "\"\n"
         << "  // Check to see if there is enough room in the table (increase the size of the table if not) and\n"
         << "  //   and then add the entry for \"" << container->baseName() << "::" << epStr() << "\".\n"
         << "  if (curIndex >= funcLookupTable_maxLen) {\n"
         << "    int newTableLen = funcLookupTable_maxLen + 16;\n"
         << "    FuncLookupTableEntry *newTable = new FuncLookupTableEntry[newTableLen];\n"
         << "    if (funcLookupTable_maxLen > 0 && funcLookupTable != NULL) {\n"
         << "      memcpy(newTable, funcLookupTable, sizeof(FuncLookupTableEntry) * curIndex);\n"
         << "      delete [] funcLookupTable;\n"
         << "    }\n"
         << "    funcLookupTable = newTable;\n"
         << "    funcLookupTable_maxLen = newTableLen;\n"
         << "  }\n"
         << "  funcLookupTable[curIndex  ].funcIndex = curIndex;\n"
         << "  funcLookupTable[curIndex++].funcPtr = __cudaFuncHost__" << container->baseName() << "__" << epStr() << ";\n\n";
   }
 }


void Entry::genAccels_cuda_host_c_regFuncs(XStr& str) {
  //if (isAccel() && isTriggered()) {
  if (isAccel()) {
    // DMK - DEBUG - register user events for performance testing

    //str << "  traceRegisterUserEvent(\"__cudaFuncHost__" << container->baseName() << "__" << epStr() << "\", 54718 + curIndex);\n";
    str << "  traceRegisterUserEvent(\"__cudaFuncHost__" << container->baseName() << "__" << epStr() << " - contribute\", 54718 + (3 * curIndex));\n";
    str << "  traceRegisterUserEvent(\"__cudaFuncHost__" << container->baseName() << "__" << epStr() << " - active\", 54718 + (3 * curIndex) + 1);\n";
    str << "  traceRegisterUserEvent(\"__cudaFuncHost__" << container->baseName() << "__" << epStr() << " - callback\", 54718 + (3 * curIndex) + 2);\n";

    //str << "  traceRegisterUserEvent(\"__cudaFunc_contributeTo__" << container->baseName() << "__" << epStr() << "\", 24875 + curIndex);\n";

    str << "  " << indexName() << "::accel_cuda_func_index__" << epStr() << " = curIndex++;\n";
  }
}





void Entry::genAccelIndexWrapperDef_general(XStr& str) {
ParamList *curParam = NULL;

  ///// The delay restart function for the general function on the host /////

  // If there is no actual accelerator (support), then there is no need for delayed calls
  #if (CMK_CUDA != 0) || (CMK_CELL != 0)

    curParam = param;
    int numPassedParams = 0;
    if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
    while (curParam != NULL) {
      numPassedParams++;
      curParam = curParam->next;
    }
    if (numPassedParams > 0) {
 str << "// AEM : \"" << container->baseName() << "::" << epStr() << "\"\n"
          << "//   A data structure for storing the passed parameters when the AEM is delayed (allowing the message to be free'd)\n"
          << "typedef struct __accelCall_" << container->baseName() << "_" << epStr() << "_passedParams {\n";
      curParam = param;
      if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
      while (curParam != NULL) {
        Parameter *p = curParam->param;
        const char *n = p->getName();
        const char *t = p->getType()->getBaseName();
        if (p->isArray()) {
          str << "  " << t << " *" << n << ";\n";
        } else {
          str << "  " << t << " " << n << ";\n";
        }
        curParam = curParam->next;
      }
      str << "} AccelCall_" << container->baseName() << "_" << epStr() << "_PassedParams;\n\n";
    }

    str << "// A function used to 'restart' a delayed AEM invocation\n"
        << "void _accelCall_generalDelayRestart_" << container->baseName() << "_" << epStr() << "(AccelDelayCallData* dataPtr) {\n"
        << "  " << indexName() << "::_accelCall_generalDelayRestart_" << epStr() << "(dataPtr);\n"
        << "};\n"
        << makeDecl("void") << "::_accelCall_generalDelayRestart_" << epStr() << "(void* ptr) {\n\n"
        << "  // Grab the data and object pointers related to the delayed AEM invocation\n"
        << "  AccelDelayCallData *dataPtr = (AccelDelayCallData*)ptr;\n"
 << "  if (dataPtr->funcPtr != ::_accelCall_generalDelayRestart_" << container->baseName() << "_" << epStr() << ") {\n"
        << "    CkPrintf(\"[ACCEL-ERROR] :: PE %d :: (generated code - host delay restart) - function pointer mismatch detected! (" << container->baseName() << " " << epStr() << ")\\n\", CkMyPe());\n"
        << "  }\n"
        << "  " << container->baseName() << " *impl_obj = (" << container->baseName() << "*)(dataPtr->objPtr);\n\n";
if (numPassedParams > 0) {
      str << "  // Unpack the passed parameters\n"
          << "  AccelCall_" << container->baseName() << "_" << epStr() << "_PassedParams *passedParams = (AccelCall_" << container->baseName() << "_" << epStr() << "_PassedParams*)(dataPtr->passedParams);\n";
      curParam = param;
      if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
      while (curParam != NULL) {
        Parameter *p = curParam->param;
        const char *n = p->getName();
        const char *t = p->getType()->getBaseName();
        if (p->isArray()) {
          str << "  " << t << " *" << n << " = passedParams->" << n << ";\n";
        } else {
          str << "  " << t << " " << n << " = passedParams->" << n << ";\n";
        }
        curParam = curParam->next;
      }
      str << "\n";
    }

    str << "  // Invoke the AEM\n"
        << " _accelCall_general_" << epStr() << "("; genAccelFullCallList(str); str << ");\n\n";

    if (numPassedParams > 0) {
      str << "  // Delete the passed parameters\n";
      curParam = param;
 if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
      while (curParam != NULL) {
        Parameter *p = curParam->param;
        const char *n = p->getName();
        const char *t = p->getType()->getBaseName();
        if (p->isArray()) {
          str << "  delete [] passedParams->" << n << ";\n";
        }
        curParam = curParam->next;
      }
      str << "  delete passedParams; passedParams = NULL; dataPtr->passedParams = NULL;\n\n";
    }

    str << "  // 'Free' the data structure related to the delayed call\n"
        << "  AccelManager *manager = AccelManager::getAccelManager();\n"
        << "  if (manager == NULL) { CkPrintf(\"[ACCEL-ERROR] :: PE %d :: (generated code - host delay restart) - Unable to get pointer to accelerator manager! (" << container->baseName() << ", " << epStr() << ")\\n\", CkMyPe()); fflush(NULL); }\n"
        << "  manager->freeAccelDelayCallData(dataPtr);\n"
        << "}\n\n";

    str << "// A function to 'delay' AEM invocations on the host core so that they can be executed at a later time\n"
        << makeDecl("void") << "::_accelCall_generalDelay_" << epStr() << "(";
    genAccelFullParamList(str, 1);
    str << ") {\n\n"
        << "  // Get a pointer to the accelerator manager\n"
        << "  AccelManager *manager = AccelManager::getAccelManager();\n"
        << "  if (manager == NULL) { CkPrintf(\"[ACCEL-ERROR] :: PE %d :: (generated code - host delay) - Unable to get pointer to accelerator manager! (" << container->baseName() << ", " << epStr() << ")\\n\", CkMyPe()); fflush(NULL); }\n\n"
        << "  // Get a 'delayed call' data structure and fill it in with information regarding this AEM invocation\n"
        << "  AccelDelayCallData *structPtr = manager->allocAccelDelayCallData();\n"
        << "  if (structPtr == NULL) { CkPrintf(\"[ACCEL-ERROR] :: PE %d :: (generated code - host delay) - Unable to allocate accel call delay data structure! (" << container->baseName() << ", " << epStr() << ")\\n\", CkMyPe()); fflush(NULL); }\n"
   << "  structPtr->objPtr = impl_obj;\n"
        << "  structPtr->funcPtr = ::_accelCall_generalDelayRestart_" << container->baseName() << "_" << epStr() << ";\n"
 << "  structPtr->objPtr = impl_obj;\n"
        << "  structPtr->funcPtr = ::_accelCall_generalDelayRestart_" << container->baseName() << "_" << epStr() << ";\n"
    #if CMK_CUDA != 0
        << "  structPtr->funcIndex = " << indexName() << "::accel_cuda_func_index__" << epStr() << ";\n"
    #else
        << "  structPtr->funcIndex = " << indexName() << "::accel_spe_func_index__" << epStr() << ";\n"
    #endif
        << "\n";

    if (numPassedParams > 0) {
      str << "  // Copy the passed parameters so that the message object can be free'd before the AEM is actually executed\n"
          << "  //   since keeping messages around for a while is bad for some machine layers\n"
          << "  AccelCall_" << container->baseName() << "_" << epStr() << "_PassedParams *passedParams = new AccelCall_" << container->baseName() << "_" << epStr() << "_PassedParams;\n";
      curParam = param;
      if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
      while (curParam != NULL) {
        Parameter *p = curParam->param;
        const char *n = p->getName();
        const char *t = p->getType()->getBaseName();
        if (p->isArray()) {
          str << "  passedParams->" << n << " = new " << t << "[" << p->getArrayLen() << "]; memcpy(passedParams->" << n << ", " << n << ", sizeof(" << t << ") * (" << p->getArrayLen() << "));\n";
        } else {
   str << "  passedParams->" << n << " = " << n << ";\n";
        }
        curParam = curParam->next;
      }
      str << "  structPtr->passedParams = (void*)passedParams;\n\n";
    } else {
      str << "  // No passed parameters for this AEM, so set the passedParams pointer to NULL\n"
          << "  structPtr->passedParams = NULL;\n\n";
    }

    str << "  // Register this AEM invocation as a delayed invocation with the accelerator manager\n"
        << "  manager->delayGeneralCall(structPtr);\n"
        <<" }\n\n";

  #endif  // need to create delayed function calls for this AEM

  ///// The general function for executing the code on the host /////

  str << "// The host version of the AEM used to execute the AEM on the host core\n"
      << makeDecl("void") << "::_accelCall_general_" << epStr() << "(";
  genAccelFullParamList(str, 1);
  str << ") {\n\n";

  // If accelerators are actually supported in this build of the runtime system, then go ahead and include code
  //   that allows the host version of the host to interoperate with the device versions of the code.
  #if (CMK_CUDA != 0) || (CMK_CELL != 0)

    str << "  // Begin timing\n"
        << "  double __startTime_userCode = CmiWallTimer();\n\n";

    str << "  // Get a pointer to the accelerator manager\n"
        << "  AccelManager *manager = AccelManager::getAccelManager();\n\n";
        // Pull any persistent data associated with the AEM
 #if CMK_CUDA != 0
      curParam = accelParam;
      while (curParam != NULL) {  // For each local parameter
        Parameter *p = curParam->param;
        if (p->isArray() && p->isPersist()) {
          const char *n = p->getName();
          str << "  // Accel Param: \"" << n << "\" (persist buffer)\n"
              << "  AccelPersistBuffer *persistBuffer_" << n << " = manager->getPersistBuffer(" << n << ");\n"
              << "  if (persistBuffer_" << n << " != NULL) {  // NOTE: If there is no persist buffer object, the host copy is up to date\n";
          // NOTE: If the buffer is write only, its actual contents don't matter, so don't bother pulling
          if (p->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READWRITE || p->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READONLY) {
            str << "    persistBuffer_" << n << "->pullFromDevice(); // NOTE: Will only actually pull if the host copy is dirty\n";
          }
          str << "  }\n"
              << "  persistBuffer_" << n << "->markDataAsOnHost(); // NOTE: Not actually dirty yet, but it will be once this function has completed\n"
              << "\n";
        }
        curParam = curParam->next;
      }
    #endif // end if CMK_CUDA != 0 (need to process persist buffers)
  #endif

  // If this is a splittable entry method, create the numSplits loop
  XStr* splitAmount = getAccelSplitAmount();
if (splitAmount != NULL) {
    str << "  // Since this AEM is splittable, determine the number of splits that will be used and create\n"
        << "  //   a loop that executes the splits serially.\n"
    #if CMK_CUDA != 0
        << "  int numSplits = (" << (*splitAmount) << ") >> 8;\n"
        << "  if (numSplits <= 0) { numSplits = 1; }\n"
    #else
        << "  int numSplits = 1;\n"
    #endif
        << "  for (int splitIndex = 0; splitIndex < numSplits; splitIndex++) {\n";
  }

  // Include the function body of the AEM
  str << "    { /***** START USER CODE (AEM FUNCTION BODY) *****/\n"
      << (*accelCodeBody)
      << "\n"
      << "    } /***** END USER CODE (AEM FUNCTION BODY) *****/\n";

  // If this is a splittable entry method, finish the numSplits loop and include
  //   a gpu process call for each iteration of the loop
  if (splitAmount != NULL) {
    #if CMK_CUDA != 0
      str << "\n    // Since GPGPUs are supported include a progress call to the Hybrid API between splits so\n"
          << "    //   that the host core is more responsive to the GPGPU's needs.\n"
          << "    gpuProgressFn();\n";
    #endif
 str << "  }\n";
  }
  str << "\n";
// If accelerators are actually supported in this build of the runtime system, then go ahead and include code
  //   that allows the host version of the host to interoperate with the device versions of the code.
  #if (CMK_CUDA != 0) || (CMK_CELL != 0)

    // Push any persistent data associated with the AEM
    #if CMK_CUDA != 0
      curParam = accelParam;
      while (curParam != NULL) {  // For each local parameter
        Parameter *p = curParam->param;
        if (p->isArray() && p->isPersist()) {
          const char *n = p->getName();
          str << "  // Accel Param: " << n << "\n"
              << "  if (persistBuffer_" << n << " != NULL) {  // NOTE: If there is no persist buffer object, the host copy is up to date\n";
          // NOTE: If the buffer is read only, its actual contents don't matter, so don't bother pushing
          if (p->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READWRITE || p->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_WRITEONLY) {
            str << "    persistBuffer_" << n << "->pushToDevice(); // NOTE: Will only actually pull if the host copy is dirty\n";
          }
          str << "  }\n"
       //<< "  persistBuffer_" << n << "->markDataAsOnDevice(); // NOTE: Not actually dirty yet, but will be once this function has completed\n"
              << "\n";
        }
        curParam = curParam->next;
      }
    #endif

    str << "  // Stop timing\n"
        << "  double __endTime_userCode = CmiWallTimer();\n\n";

    str << "  // Pass the timing information to the accelerator manager\n"
    #if CMK_CUDA != 0
        << "  if (manager != NULL) { manager->userHostCodeTiming(accel_cuda_func_index__" << epStr() << ", __startTime_userCode, __endTime_userCode); }\n"
    #elif CMK_CELL != 0
        << "  if (manager != NULL) { manager->userHostCodeTiming(accel_spe_func_index__" << epStr() << ", __startTime_userCode, __endTime_userCode); }\n"
    #endif
        << "\n";

  #endif

  str << "  // Finish of the host version of the AEM by immediately calling the callback function\n"
      << "  impl_obj->" << (*accelCallbackName) << "();\n"
      << "}\n\n";


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
void Entry::genAccelIndexWrapperDecl_cuda(XStr& str) {

  if (!(isAccel())) { return; }

  // Function to issue work request and callback function (as members of CkIndex_xxxx class)
  str << "    // AEM : \"" << container->baseName() << "::" << epStr() << "\"\n"
      << "    static void _accelCall_cuda_" << epStr() << "(";
  genAccelFullParamList(str, 0);
  str << ", int numElements, int issueFlag);\n"
      << "    static void _accelCall_cuda_callback_" << epStr() << "(void* userPtr);\n\n";
}

void Entry::genAccelIndexWrapperDef_cuda(XStr& str) {

  XStr containerType = container->baseName();
  ParamList *curParam = NULL;

  if (!(isAccel())) { return; }

  // Precompute the number of scalar parameters
  int numScalars = 0;
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    if (!(curParam->param->isArray())) { numScalars++; }
    curParam = curParam->next;
  }
  curParam = accelParam;
  while (curParam != NULL) {
    if (!(curParam->param->isArray())) { numScalars++; }
    curParam = curParam->next;
  }

  ///// Generate the kernel callback function /////

  // NOTE: Create a global function that simply calls the member function on the related CkIndex class for this chare class
  str << "// AEM : \"" << container->baseName() << "::" << epStr() << "\"\n"
      << "//   A global function that simply calls the CkIndex_xxxx generated callback function\n"
      << "void _accelCall_cuda_callback_" << containerType << "_" << epStr() << "(void *param, void *msg) {\n"
      << "  " << container->indexName() << "::_accelCall_cuda_callback_" << epStr() << "(param);\n"
      << "}\n\n";

  // NOTE: Create the actual function that is executed as the callback when a kernel completes
str << "// AEM : \"" << container->baseName() << "::" << epStr() << "\"\n"
      << "//   A callback function that is invoked after the kernel is finished executing\n"
      << makeDecl("void") << "::_accelCall_cuda_callback_" << epStr() << "(void* userPtr) {\n\n"
      << "  // Notify the accelerator manager that a kernel has finished\n"
      << "  markKernelEnd();\n\n"
      << "  // Grab a pointer to the accelerator manager\n"
      << "  AccelManager *accelManager = AccelManager::getAccelManager();\n\n"
      << "  // Grab the pointer to the data structure representing the kernel (AEM batch)\n"
      << "  AccelCallbackStruct *cbStruct = (AccelCallbackStruct*)userPtr;\n\n"
      << "  // Check to see if the individual callback functions should be executed (i.e. the batch has not been abandoned)\n"
      << "  if (cbStruct->abandonFlag == 0) {\n\n"
      << "    // Note the time that the callbacks are starting to be processed\n"
      << "    cbStruct->callbackStartTime = CmiWallTimer();\n\n"
      << "    // Cast the batch data pointer to an int pointer for easy access to the header fields\n"
      << "    int *__dataPtr_int = (int*)(cbStruct->wrData);\n\n"
      << "    // Test for the bit patterns written by the kernel to verify that the kernel was actually executed\n"
      << "    if ((__dataPtr_int[ACCEL_CUDA_KERNEL_BIT_PATTERN_0_INDEX] != ACCEL_CUDA_KERNEL_BIT_PATTERN_0) && \n"
      << "        (__dataPtr_int[ACCEL_CUDA_KERNEL_BIT_PATTERN_1_INDEX] != ACCEL_CUDA_KERNEL_BIT_PATTERN_1)\n"
      << "     ) {\n"
      << "      CkPrintf(\"[ACCEL-ERROR] :: Invalid bit patterns detected in callback for \\\"" << containerType << "_" << epStr() << "\\\" - The kernel may not have executed.\\n\");\n"
      << "    }\n\n"
      << "    // Test for an error code being passed back from the device\n"
      << "    if (__dataPtr_int[ACCEL_CUDA_KERNEL_ERROR_INDEX] != 0) {\n"
      << "      CkPrintf(\"[ACCEL-ERROR] :: Error code set for \\\"" << containerType << "_" << epStr() << "\\\": error code = %d...\\n\", __dataPtr_int[ACCEL_CUDA_KERNEL_ERROR_INDEX]);\n"
      << "    }\n\n"
      << "    // Cast the batch data pointer to a char pointer\n"
      << "    char *__dataPtr = (char*)(cbStruct->wrData);\n\n"
      << "    // Create a loop that will iterate through each individual AEM invocation within this batch of invocations\n"
      << "    for (int i = 0; i < cbStruct->numElements_count; i++) {\n\n"
      << "      // Get the object and associated data pointers for AEM invocation i\n"
      << "      " << containerType << "* impl_obj = (" << containerType << "*)(cbStruct->impl_objs[i]);\n"
      << "      int __dataOffset = __dataPtr_int[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << numScalars << " + i];\n\n";

  // For each scalar, declare the associated variable with its value
  // NOTE: Declare scalars before arrays since array lengths rely on scalar values
  int scalarArrayIndex = 0;
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (!(p->isArray())) {
      str << "      // Passed Parameter (Scalar) : \"" << n << "\"\n"
          << "      " << t << " " << n << " = ((" << t << "*)(__dataPtr + (__dataPtr_int[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << scalarArrayIndex << "])))[i];\n\n";
      scalarArrayIndex++;
    }
    curParam = curParam->next;
  }
  curParam = accelParam;
 while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (!(p->isArray())) {
      str << "      // Local Parameter (Scalar) : \"" << n << "\"\n"
          << "      " << t << " " << n << " = ((" << t << "*)(__dataPtr + (__dataPtr_int[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << scalarArrayIndex << "])))[i];\n\n";
      scalarArrayIndex++;
    }
    curParam = curParam->next;
  }

  // Process the array parameters
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (p->isArray()) {
      str << "      // Passed Parameter (Array) : " << n << "\n"
          << "      if (__dataOffset % sizeof(" << t << ") != 0) {\n"
          << "        __dataOffset += (sizeof(" << t << ") - (__dataOffset % sizeof(" << t << ")));\n"
          << "      }\n"
          << "      __dataOffset += sizeof(" << t << ") * (" << p->getArrayLen() << ");\n\n";
    }
    curParam = curParam->next;
  }
  curParam = accelParam;
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (p->isArray()) {
      if (p->isPersist()) {
        str << "      // Local Parameter (Persist Array) : \"" << n << "\"\n"
            << "      if (__dataOffset % sizeof(void*) != 0) {\n"
            << "        __dataOffset += (sizeof(void*) - (__dataOffset % sizeof(void*)));\n"
            << "      }\n"
            << "      __dataOffset += sizeof(void*);\n\n";
      } else if (p->isShared()) {
        str << "      // Local Parameter (Shared Array) : \"" << n << "\"\n"
            << "      if (__dataOffset % sizeof(int) != 0) {  // Alignment\n"
            << "        __dataOffset += (sizeof(int) - (__dataOffset % sizeof(int)));\n"
            << "      }\n"
            << "      " << t << " *__" << n << " = NULL;  // NOTE: Will be present on first reference, should remain NULL for all following references so as not to copy multiple times\n"
            << "      int __dataSkip_" << n << " = ((int*)(__dataPtr + __dataOffset))[1];\n"
            << "      __dataOffset += (2 * sizeof(int));\n"
            << "      if (__dataSkip_" << n << " > 0) {  // The actual data is included here\n"
            << "        if (__dataOffset % sizeof(" << t << ") != 0) {\n"
            << "          __dataOffset += (sizeof(" << t << ") - (__dataOffset % sizeof(" << t << ")));\n"
            << "        }\n"
            << "        __" << n << " = (" << t << "*)(__dataPtr + __dataOffset);\n"
            << "      }\n"
            << "      __dataOffset += __dataSkip_" << n << ";\n\n";
 } else {
        str << "      // Local Parameter (Array) : \"" << n << "\"\n"
            << "      if (__dataOffset % sizeof(" << t << ") != 0) {  // Alignment\n"
            << "        __dataOffset += (sizeof(" << t << ") - (__dataOffset % sizeof(" << t << ")));\n"
            << "      }\n"
            << "      " << t << " *__" << n << " = (" << t << "*)(__dataPtr + __dataOffset);\n"
            << "      __dataOffset += sizeof(" << t << ") * (" << p->getArrayLen() << ");\n\n";
      }
    }
    curParam = curParam->next;
  }

  // Copy the data from the kernel buffer in to the chare's memory
  // NOTE: Remote parameters are read-only by definition, so skip them
  str << "      // Copy back the data that actually requires copying back into the object\n";
  curParam = accelParam;
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if ((p->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READWRITE) ||
        (p->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_WRITEONLY)
       ) {
      if (p->isArray()) {
        if (!p->isPersist()) {
          if (p->isShared()) {
            str << "      if (__" << n << " != NULL) {\n  ";  // NOTE: Extra spaces to indent next line
          }
          str << "      memcpy((void*)(" << (*(p->getAccelInstName())) << "), (void*)(__" << n << "), sizeof(" << t << ") * (" << p->getArrayLen() << "));\n";
          if (p->isShared()) {
            str << "      }\n";
          }
        }
      } else {
        str << "      " << (*(p->getAccelInstName())) << " = " << n << ";\n";
      }
    }

    curParam = curParam->next;
  }
  str << "\n";

  str << "      // Call the callback function on the object associated with this AEM invocation\n"
      << "      impl_obj->" << (*accelCallbackName) << "();\n\n"
      << "    } // end for (i < cbStruct->numElements_count)\n\n"
      << "    // End timing and submit user events related to this batch of AEMs\n"
      << "    double callbackEndTime = CmiWallTimer();\n"
      << "    int userEventIndex = 54718 + (3 * " << indexName() << "::accel_cuda_func_index__" << epStr() << ");\n"
      << "    // traceUserBracketEvent(userEventIndex, cbStruct->contribStartTime, cbStruct->issueTime);\n"
      << "    traceUserBracketEvent(userEventIndex + 1, cbStruct->issueTime, cbStruct->callbackStartTime);\n"
      << "    traceUserBracketEvent(userEventIndex + 2, cbStruct->callbackStartTime, callbackEndTime);\n\n"
      << "    // Adjust the IDLE time tracking in the AccelManager\n"
      << "    if (accelManager != NULL) { accelManager->adjustCallbackTime(callbackEndTime - cbStruct->callbackStartTime); }\n\n"
      << "  } else { // else if (cbStruct->abandonFlag != 0)\n\n"
      << "    // Notify the accelerator manager that an abandoned batch has completed (record keeping)\n"
 << "    accelManager->notifyAbandonedRequestCompletion(cbStruct);\n\n"
      << "  } // end if (cbStruct->abandonFlag != 0)\n\n"
      << "  // Free the memory associated with the callback structure\n"
      << "  kernelCleanup(cbStruct->wr, cbStruct->di);\n"
      #if CMK_CUDA!= 0
        << "  hapi_poolFree(cbStruct->wrData);\n"
      #else
        << "  delete [] ((char*)(cbStruct->wrData));\n"
      #endif
      << "  cbStruct->wrData = NULL;\n"
      << "  if (accelManager != NULL) { accelManager->freeAccelCallbackStruct(cbStruct); }\n"
      << "}\n\n";

  ///// Generate the function to issue work request /////

  str << "// AEM : \"" << container->baseName() << "::" << epStr() << "\"\n"
      << "//   A function that batches and issues AEMs to the GPGPU device via the accelerator manager\n"
      << makeDecl("void") << "::_accelCall_cuda_" << epStr() << "(";
  genAccelFullParamList(str, 0);  // NOTE: This will always output at least one parameter (impl_obj)
  str << ", int numElements, int issueFlag) {\n\n"
      << "  // Get a pointer to the accelerator manager for use within this function\n"
      << "  AccelManager *accelManager = AccelManager::getAccelManager();\n\n";

  // For splittable AEMs, get the number of splits and set a splitMultiplier for later use
  //   in this generation code
  XStr *numSplitsStr = getAccelSplitAmount();
  int splitMultiplier = 1;
  if (numSplitsStr != NULL) {
    str << "  // Since this is a splittable AEM, get the number of splits indicated by the user\n"
        << "  int numSplits = (int)(" << numSplitsStr->get_string() << ");\n\n";
    splitMultiplier = 2;
  }

  str << "  // Get the callback structure from the accelerator manager (used for batch related info)\n"
      << "  AccelCallbackStruct *cbStruct = accelManager->getCurrentCallbackStruct(accel_cuda_func_index__" << epStr() << ");\n"
      << "  if (cbStruct == NULL || cbStruct->issueTime > 0.0) {    // If no current batch or last batch was issued\n"
      << "    cbStruct = accelManager->allocAccelCallbackStruct();  //   then get and initialize a new data structure\n"
      << "    cbStruct->numElements = numElements;\n"
      << "    cbStruct->funcIndex = accel_cuda_func_index__" << epStr() << ";\n"
      << "    cbStruct->callbackPtr = (void*)(new CkCallback(_accelCall_cuda_callback_" << containerType << "_" << epStr() << ", cbStruct));\n"
      << "    cbStruct->numSplitsSubArray = NULL;\n"
      << "    accelManager->setCurrentCallbackStruct(accel_cuda_func_index__" << epStr() << ", cbStruct);\n"
      << "  }\n\n"


      << "  // If this is the first element being included in the batch, note the time.  Otherwise, verify\n"
      << "  //   the number of total local elements to verify that load balancing did not occur during the\n"
      << "  //   the batching process.\n"
      << "  if (cbStruct->numElements_count == 0) {\n"
      << "    cbStruct->contribStartTime = CmiWallTimer();\n"
      << "  } else {\n"
      << "    if (cbStruct->numElements != numElements) {\n"
      << "      CkPrintf(\"[ACCEL-ERROR] :: PE %d :: (generated host code) - numElements mismatch detected for \\\"" << container->baseName() << "::" << epStr() << "\\\"! An element may have migrated during batching period. (%d vs %d)\\n\", CkMyPe(), cbStruct->numElements, numElements);\n"
      << "    }\n"
      << "  }\n\n"
      << "  // Record this impl_obj pointer\n"
 << "  cbStruct->impl_objs[cbStruct->numElements_count] = impl_obj;\n\n"
      << "  ///// Calculate the total size of all the data associated with this entry method /////\n\n"
      << "  int dataSize = 0;\n\n";

  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (p->isArray()) {
      str << "  // Passed Parameter (Array) : " << n << "\n"
          << "  int dataSize_" << n << "_align = 0;\n"
          << "  if (dataSize % sizeof(" << t << ") != 0) {  // If not aligned, add some bytes for alignment\n"
          << "    dataSize_" << n << "_align = (sizeof(" << t << ") - (dataSize % sizeof(" << t << ")));\n"
          << "  }\n"
          << "  int dataSize_" << n << " = sizeof(" << t << ") * (" << p->getArrayLen() << ");  // Data size associated with the parameter\n"
          << "  dataSize += (dataSize_" << n << " + dataSize_" << n << "_align);\n\n";
    }
    curParam = curParam->next;
  }
  curParam = accelParam;
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (p->isArray()) {
      if (p->isPersist()) {
        str << "  // Local Parameter (Persist Array): " << n << "\n"
            << "  int dataSize_" << n << "_align = 0;\n"
            << "  if (dataSize % sizeof(void*) != 0) {  // If not aligned, add some bytes for alignment\n"
            << "    dataSize_" << n << "_align = (sizeof(void*) - (dataSize % sizeof(void*)));\n"
            << "  }\n"
            << "  int dataSize_" << n << " = sizeof(void*);  // Include enough room for a device pointer\n"
            << "  dataSize += (dataSize_" << n << " + dataSize_" << n << "_align);\n\n";
      } else if (p->isShared()) {
        str << "  // Local Parameter (Shared Array): " << n << "\n"
            << "  int sharedOffset_" << n << " = cbStruct->sharedLookup->lookupOffset(" << n << ", " << p->getArrayLen() << "); // Lookup to see if data has already been included\n"
            << "  int dataSize_" << n << "_align = 0;          // An int[2] is always included\n"
            << "  int dataSize_" << n << " = 2 * sizeof(int);  //   (offset, skip amount)\n"
            << "  int dataSize_" << n << "_data_align = 0;     // Bytes associated with actual data\n"
            << "  int dataSize_" << n << "_data = 0;\n"
            << "  if (dataSize % sizeof(int) != 0) {  // Align for int[2]\n"
            << "    dataSize_" << n << "_align = (sizeof(int) - (dataSize % sizeof(int)));\n"
            << "  }\n"
            << "  dataSize += (dataSize_" << n << " + dataSize_" << n << "_align);\n"
            << "  if (sharedOffset_" << n << " < 0) {  // If data hasn't been included so far, include it now\n"
            << "    if (dataSize % sizeof(" << t << ") != 0) {  // Alignment\n"
            << "      dataSize_" << n << "_data_align = (sizeof(" << t << ") - (dataSize % sizeof(" << t << ")));\n"
            << "    }\n"
            << "    dataSize_" << n << "_data = sizeof(" << t << ") * (" << p->getArrayLen() << ");\n"
            << "  }\n"
            << "  dataSize += (dataSize_" << n << "_data_align + dataSize_" << n << "_data);\n\n";
      } else {
        str << "  // Local Parameter (Array): " << n << "\n"
     << "  int dataSize_" << n << "_align = 0;\n"
            << "  if (dataSize % sizeof(" << t << ") != 0) {  // Alignment\n"
            << "    dataSize_" << n << "_align = (sizeof(" << t << ") - (dataSize % sizeof(" << t << ")));\n"
            << "  }\n"
            << "  int dataSize_" << n << " = sizeof(" << t << ") * (" << p->getArrayLen() << ");\n"
            << "  dataSize += (dataSize_" << n << " + dataSize_" << n << "_align);\n\n";
      }
    }
    curParam = curParam->next;
  }

  str << "  // Enforce alignment for the start of individual element records (push forward offset so\n"
      << "  //   next element will be aligned and padding is included at the end so DMA transfers lengths\n"
      << "  //   are multiples of cache line sizes via setting ACCEL_CUDA_ELEMENT_ALIGN for performance)\n"
      << "  if (dataSize % ACCEL_CUDA_ELEMENT_ALIGN != 0) { dataSize += (ACCEL_CUDA_ELEMENT_ALIGN - (dataSize % ACCEL_CUDA_ELEMENT_ALIGN)); }\n\n"
      << "  ///// Grow the current buffer if needed /////\n\n"
      << "  if (cbStruct->wrDataLen + dataSize > cbStruct->wrDataLen_max) {\n\n"
      << "    // Calculate the number of elements total (setSize) and the number of elements remaining to be batched\n"
      << "    int setSize = ((numElements < ACCEL_AEMs_PER_GPU_KERNEL) ? (numElements) : (ACCEL_AEMs_PER_GPU_KERNEL));\n"
      << "    int elementsRemaining = setSize - cbStruct->numElements_count + 1;\n\n"
      << "    // Try to estimate the amount of memory that will be required by the batch\n"
      << "    int newWRDataLen_max = (cbStruct->wrDataLen + (dataSize * elementsRemaining) + (10 * 1024));\n"
      << "    if (cbStruct->wrData == NULL) { // If this is the first element, include header and scalar array sizes\n"
      << "      newWRDataLen_max += (sizeof(int) * (ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << numScalars << " + (" << splitMultiplier << " * setSize))); // header + scalar array offsets + element data offsets [ + individual numSplits ]\n";

  // Add memory for each of the scalar arrays (+ 128 byte aligned = 32 4-byte words... based on banked shared memory hardware)
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (!(p->isArray())) {
      str << "      newWRDataLen_max += (128 + (setSize * sizeof(" << t << ")));  // Passed Parameter (Scalar) : \"" << n << "\"\n";
    }
    curParam = curParam->next;
  }
  curParam = accelParam;
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (!(p->isArray())) {
      str << "      newWRDataLen_max += (128 + (setSize * sizeof(" << t << ")));  // Local Parameter (Scalar) : \"" << n << "\"\n";
    }
     curParam = curParam->next;
  }

  str << "    }\n"
      << "    newWRDataLen_max += 1024 - (newWRDataLen_max % 1024); // enforce 1K multiple\n\n"
      << "    // Allocate the batch data buffer\n"
      #if CMK_CUDA != 0
      << "    void *newWRData = hapi_poolMalloc(newWRDataLen_max);\n"
#else
        << "    void *newWRData = (void*)(new char[newWRDataLen_max]);\n"
      #endif
      << "    if (newWRData == NULL) { CkPrintf(\"[ERROR] :: Unable to allocate memory for newWRData...\\n\"); }\n\n"
      << "    // If there was data previously, copy it into the new buffer and delete the old batch data buffer\n"
      << "    if (cbStruct->wrData != NULL) {\n"
      << " CkPrintf(\" memcpy \\n\");\n\n"
      << "      memcpy(newWRData, cbStruct->wrData, cbStruct->wrDataLen);\n"
      #if CMK_CUDA != 0
        << "      hapi_poolFree(cbStruct->wrData);\n"
      #else
        << "      delete [] (char*)(cbStruct->wrData);\n"
      #endif
      << "    } else { // Otherwise, this is the first buffer being allocated, so do some setup\n"
      << "      int scalarArrayOffset = sizeof(int) * (ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << numScalars << " + (" << splitMultiplier << " * setSize));\n";

  // Add memory for each of the scalar arrays (128 byte aligned for performance)
  scalarArrayIndex = 0;
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (!(p->isArray())) {
      str << "      // Passed Parameter (Scalar) : \"" << n << "\"\n"
          << "       if (scalarArrayOffset % 128 != 0) { scalarArrayOffset += (128 - (scalarArrayOffset % 128)); }\n"
          << "       ((int*)newWRData)[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << scalarArrayIndex << "] = scalarArrayOffset;\n"
          << "       scalarArrayOffset += sizeof(" << t << ") * setSize;\n";
      scalarArrayIndex++;
    }
    curParam = curParam->next;
  }
  curParam = accelParam;
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (!(p->isArray())) {
      str << "      // Local Parameter (Scalar) : \"" << n << "\"\n"
          << "       if (scalarArrayOffset % 128 != 0) { scalarArrayOffset += (128 - (scalarArrayOffset % 128)); }\n"
          << "       ((int*)newWRData)[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << scalarArrayIndex << "] = scalarArrayOffset;\n"
          << "       scalarArrayOffset += sizeof(" << t << ") * setSize;\n";
      scalarArrayIndex++;
    }
    curParam = curParam->next;
  }

  str << "      // After the scalar arrays have been included, align for the first element\n"
      << "       if (scalarArrayOffset % ACCEL_CUDA_ELEMENT_ALIGN != 0) { scalarArrayOffset += (ACCEL_CUDA_ELEMENT_ALIGN - (scalarArrayOffset % ACCEL_CUDA_ELEMENT_ALIGN)); }\n"
      << "      cbStruct->wrDataLen = scalarArrayOffset;  // Record the current offset\n";

  if (numSplitsStr != NULL) {
    str << "      ((int*)newWRData)[ACCEL_CUDA_KERNEL_NUM_SPLITS] = numSplits;  // This is the first element, so all splits are equal in size so far (this value > 0)\n";
  } else {
 str << "      ((int*)newWRData)[ACCEL_CUDA_KERNEL_NUM_SPLITS] = 1;  // There are no splits, so numSplits = 1\n";
  }

  str << "      ((int*)newWRData)[ACCEL_CUDA_KERNEL_SET_SIZE] = setSize;  // Indicates the maximum number of elements this batch data buffer can accommodate (not the actual number of elements)\n"
      << "    }\n"
      << "    cbStruct->wrData = newWRData;\n"
      << "    cbStruct->wrDataLen_max = newWRDataLen_max;\n"
      << "  }\n\n"
      << "  // Grab the setSize for this batch (maximum number of elements that can be included in this batch)\n"
      << "  const int setSize = ((int*)(cbStruct->wrData))[ACCEL_CUDA_KERNEL_SET_SIZE];\n\n"
      << "  // Record the start of this element's data\n"
      << "  ((int*)(cbStruct->wrData))[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << numScalars << " + cbStruct->numElements_count] = cbStruct->wrDataLen;\n\n";

  if (numSplitsStr != NULL) {
    str << "  // Record the number of splits for this element\n"
        << "  int *numSplitsSubArray = ((int*)(cbStruct->wrData)) + (ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << numScalars << " + setSize);\n"
        << "  cbStruct->numSplitsSubArray = numSplitsSubArray;\n"
        << "  numSplitsSubArray[cbStruct->numElements_count] = numSplits;\n"
        << "  if (numSplitsSubArray[0] != numSplits) { ((int*)(cbStruct->wrData))[ACCEL_CUDA_KERNEL_NUM_SPLITS] = -1; }  // Flag if there are unequal numSplits values across the elements\n\n";
  }

  str << "  ///// Serialize the parameters (remote and local) into the batch data buffer /////\n\n"
      << "  const char *dataBasePtr = (char*)(cbStruct->wrData);\n"
      << "  const int *dataBasePtr_int = (int*)dataBasePtr;\n"
      << "  char *curDataPtr = ((char*)dataBasePtr) + cbStruct->wrDataLen;\n\n";

  scalarArrayIndex = 0;
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (p->isArray()) {
      str << "  // Passed Parameter (Array) : " << n << "\n"
          << "  curDataPtr += dataSize_" << n << "_align;\n"
          << "  memcpy((void*)curDataPtr, " << n << ", dataSize_" << n << ");\n"
          << "  curDataPtr += dataSize_" << n << ";\n\n";
    } else {
      str << "  // Passed Parameter (Scalar) : " << n << "\n"
          << "  ((" << t << "*)(dataBasePtr + (dataBasePtr_int[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << scalarArrayIndex << "])))[cbStruct->numElements_count] = " << n << ";\n\n";
      scalarArrayIndex++;
    }
    curParam = curParam->next;
  }
curParam = accelParam;
  while (curParam != NULL) {
    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();
    if (p->isArray()) {
      if (p->isPersist()) {
        str << "  // Local Parameter (Persist Array) : " << n << "\n"
            << "  curDataPtr += dataSize_" << n << "_align;  // Alignment\n"
            << "  AccelPersistBuffer *persistBuffer_" << n << " = accelManager->getPersistBuffer(" << n << ");\n"
            << "  if (persistBuffer_" << n << " == NULL) {\n"
            << "    int persistDataSize = sizeof(" << t << ") * (" << p->getArrayLen() << ");\n"
            << "    persistBuffer_" << n << " = accelManager->newPersistBuffer(impl_obj, " << n << ", persistDataSize);\n"
            << "    if (persistBuffer_" << n << " == NULL) { CkPrintf(\"[ACCEL-ERROR] :: PE %d :: (generated host code) - Unable to create persist buffer pointer for persist local parameter \\\"" << n << "\\\".\\n\", CkMyPe()); }\n"
            << "  } else {\n"
            << "    if (persistBuffer_" << n << "->getObjectPtr() != impl_obj) { CkPrintf(\"[ACCEL-ERROR] :: PE %d :: (generated host code) - Pointer mismatch detected (persist buffer object pointer != impl_obj) for \\\"" << n << "\\\".\\n\", CkMyPe()); }\n"
            << "  }\n"
            << "  ((void**)curDataPtr)[0] = (void*)(persistBuffer_" << n << "->getDeviceBuffer());\n";
        if (p->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READWRITE || p->getAccelBufferType() == Parameter::ACCEL_BUFFER_TYPE_READONLY) {  // Don't need to push writeonly since actual contents aren't required
          str << "  persistBuffer_" << n << "->pushToDevice();  // NOTE: Will only actually push if device copy is dirty (e.g. if the previous invocation of this AEM was on the host)\n";
        }
        str << "  persistBuffer_" << n << "->markDataAsOnDevice();\n"
            << "  curDataPtr += dataSize_" << n << ";\n\n";
      } else if (p->isShared()) {
        str << "  // Local Parameter (Shared Array) : " << n << "\n"
            << "  curDataPtr += dataSize_" << n << "_align; // Alignment of int[2]\n"
            << "  if (sharedOffset_" << n << " >= 0) {\n"
            << "    *(((int*)curDataPtr) + 0) = sharedOffset_" << n << ";  // Offset within batch data buffer\n"
            << "    *(((int*)curDataPtr) + 1) = 0;                         // Skip amount (> 0 means data is included here)\n"
            << "  } else {\n"
            << "    *(((int*)curDataPtr) + 0) = (curDataPtr - dataBasePtr) + dataSize_" << n << " + dataSize_" << n << "_data_align;\n"
            << "    *(((int*)curDataPtr) + 1) = dataSize_" << n << "_data_align + dataSize_" << n << "_data;\n"
            << "  }\n"
            << "  curDataPtr += dataSize_" << n << ";\n"
            << "  if (sharedOffset_" << n << " < 0) {  // Need to include the data here\n"
            << "    curDataPtr += dataSize_" << n << "_data_align;  // Alignment of data\n"
            << "    int sharedOffset = curDataPtr - dataBasePtr;    // Offset of data within batch data buffer\n"
            << "    cbStruct->sharedLookup->insertOffset(" << n << ", " << p->getArrayLen() << ", sharedOffset);\n"
            << "    memcpy((void*)curDataPtr, " << n << ", dataSize_" << n << "_data);\n"
            << "    curDataPtr += dataSize_" << n << "_data;\n"
            << "  }\n\n";
      }
else {
        str << "  // Local Parameter (Array) : " << n << "\n"
            << "  curDataPtr += dataSize_" << n << "_align;  // Alignment\n";
        if (p->getAccelBufferType() != Parameter::ACCEL_BUFFER_TYPE_WRITEONLY) {
          str << "  memcpy((void*)curDataPtr, " << n << ", dataSize_" << n << ");  // Copy the actual data\n";
        }
        str << "  curDataPtr += dataSize_" << n << ";\n\n";
      }
    } else {
      str << "  // Local Parameter (Scalar) : " << n << "\n"
          << "  ((" << t << "*)(dataBasePtr + (dataBasePtr_int[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << scalarArrayIndex << "])))[cbStruct->numElements_count] = " << n << ";\n\n";
      scalarArrayIndex++;
    }
    curParam = curParam->next;
  }


  str << "  // Advance the offset to the start of the next element's data and count this element\n"
      << "  cbStruct->wrDataLen += dataSize;\n"
      << "  (cbStruct->numElements_count)++;\n\n"
      << "  // Update the time of the last contribute for this batch\n"
      << "  accelManager->updateLastContribTime(cbStruct);\n\n"
      << "  // If the decision returned by the accelerator manager indicates that the batch should be\n"
      << "  //   issued (or if the maximum batch size has been reached), then issue the batch\n"
      << "  if (issueFlag != 0 || cbStruct->numElements_count >= setSize || cbStruct->numElements_count >= ACCEL_AEMs_PER_GPU_KERNEL) {\n"
      << "    accelManager->submitPendingRequest(cbStruct);\n"
      << "  }\n";

 str << "}\n";
}

int Entry::genAccels_cuda_c_funcBodies(XStr& str) {

  XStr containerType = container->baseName();
  ParamList *curParam = NULL;

  // Make sure this is an accelerated entry method (just return if not)
  //if (!(isAccel() && isTriggered())) { return 0; }
  if (!(isAccel())) { return 0; }

  // Precompute the number of scalar parameters
  int numScalars = 0;
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {
    if (!(curParam->param->isArray())) { numScalars++; }
    curParam = curParam->next;
  }
  curParam = accelParam;
  while (curParam != NULL) {
    if (!(curParam->param->isArray())) { numScalars++; }
    curParam = curParam->next;
  }

  // Declare the cuda kernel
  str << "// AEM : \"" << container->baseName() << "::" << epStr() << "\"'s kernel function (executed on the GPGPU device)\n"
      << "__global__ void __cudaFuncKernel__" << containerType << "__" << epStr() << "(void *__wrData, int __totalThreads) {\n\n" //, int __splitAmount) {\n\n"
      << "  // Calculate the thread index (unique value from 0 to numThreads - 1) for each GPGPU thread\n"
      << "  int __threadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;\n\n"
      << "  // If extra threads were included as part of the block and grid calculations, have the extra threads\n"
 << "  if (__threadIndex < __totalThreads) {\n\n";

  // Declare __elementIndex, along with splitIndex and numSplits if needed
  // NOTE: To save the registers, only declare the last two if actually splittable
  // NOTE: Declare multiple variables incase user code overwrites something needed (e.g. numSplits
  //   used by user code, __splitAmount used by generated code)
  if (getAccelSplitAmount() != NULL) {
    str << "    // Yeah, this function is splittable!  Way to go bro.  Let's do this...\n"
        << "    // NOTE: numSplits < 0 indicates that each element can have a different number of splits.\n"
        << "    //       numSplits > 0 indicates all split amounts matched during the batching process.\n"
        << "    int numSplits = ((int*)__wrData)[ACCEL_CUDA_KERNEL_NUM_SPLITS];\n"
        << "    int __elementIndex = -1;\n"
        << "    int splitIndex = -1;\n"
        << "    if (numSplits < 0) { // Need to count because each element has a different number of splits\n"
        << "      int __setSize = ((int*)__wrData)[ACCEL_CUDA_KERNEL_SET_SIZE];\n"
        << "      int *numSplitsSubArray = ((int*)__wrData) + (ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << numScalars << " + __setSize);\n"
        << "      __elementIndex = 0;\n"
        << "      numSplits = __threadIndex; // Borrow numSplits variable to count down the threads\n"
        << "      for (int i = 0; i < __setSize; i++) {\n"
        << "        if (numSplits < numSplitsSubArray[i]) { break; }\n"
        << "        __elementIndex += 1;\n"
        << "        numSplits -= numSplitsSubArray[i];\n"
        << "      }\n"
        << "      splitIndex = numSplits;\n"
        << "      numSplits = numSplitsSubArray[__elementIndex];\n"
        << "      // NOTE: If __threadIndex >= __totalThreads, splitIndex and numSplits will be invalid, but will go unused\n"
        << "    } else { // All elements have the same splits amount, so take advantage to avoid looping\n"
        << "      __elementIndex = __threadIndex / numSplits;\n"
        << "      splitIndex = __threadIndex % numSplits;\n"
  << "    }\n\n";
  } else {
    str << "    int __elementIndex = __threadIndex;\n\n";
#if 0
    printf("[INFO] :: A non-splittable accelerated entry method was detected (\"%s::%s\")\n"
           "[INFO] ::   Consider making the AEM \"splittable\" for better performance.\n",
           *(container->baseName()), *(epStr())
          );
#endif
  }

  str << "    // Lookup the data offset for this element (start of the element's streamed parameters)\n"
      << "    int __dataOffset = ((int*)__wrData)[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << numScalars << " + __elementIndex];\n\n";

  // Create the parameters by name
  int scalarArrayIndex = 0;
  curParam = param;
  if ((curParam->param->getType()->isVoid()) && (curParam->param->getName() == NULL)) { curParam = curParam->next; }
  while (curParam != NULL) {

    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();

    str << "    // Passed Parameter: " << n << "\n";
if (p->isArray()) {
      str << "    if (__dataOffset % sizeof(" << t << ") != 0) { __dataOffset += sizeof(" << t << ") - (__dataOffset % sizeof(" << t << ")); }\n"
          << "    register " << t << " *" << n << " = (" << t << "*)(((char*)__wrData) + __dataOffset);\n"
          << "    __dataOffset += sizeof(" << t << ") * (" << p->getArrayLen() << ");\n\n";
    } else {
      str << "    register " << t << " &" << n << " = ((" << t << "*)(((char*)__wrData) + ((int*)__wrData)[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << scalarArrayIndex << "]))[__elementIndex];\n\n";
      scalarArrayIndex++;
    }

    curParam = curParam->next;
  }
  curParam = accelParam;
  while (curParam != NULL) {

    Parameter *p = curParam->param;
    const char *n = p->getName();
    const char *t = p->getType()->getBaseName();

    str << "    // Local Parameter: " << n << "\n";
                                                          if (p->isArray()) {
      if (p->isPersist()) {
        str << "    if (__dataOffset % sizeof(void*) != 0) { __dataOffset += sizeof(void*) - (__dataOffset % sizeof(void*)); }\n"
            << "    register " << t << " *" << n << " = ((" << t << "**)(((char*)__wrData) + __dataOffset))[0];\n"
            << "    __dataOffset += sizeof(void*);\n\n";
      } else if (p->isShared()) {
        str << "    if (__dataOffset % sizeof(int) != 0) { __dataOffset += sizeof(int) - (__dataOffset % sizeof(int)); }\n"
            << "    int __" << n << "_offset = ((int*)(((char*)__wrData) + __dataOffset))[0];\n"
            << "    int __" << n << "_skip = ((int*)(((char*)__wrData) + __dataOffset))[1];\n"
            << "    register " << t << " *" << n << " = (" << t << "*)(((char*)__wrData) + __" << n << "_offset);\n"
            << "    __dataOffset += (2 * sizeof(int)) + __" << n << "_skip;\n\n";
      } else {
        str << "    if (__dataOffset % sizeof(" << t << ") != 0) { __dataOffset += sizeof(" << t << ") - (__dataOffset % sizeof(" << t << ")); }\n"
            << "    register " << t << " *" << n << " = (" << t << "*)(((char*)__wrData) + __dataOffset);\n"
            << "    __dataOffset += sizeof(" << t << ") * (" << p->getArrayLen() << ");\n\n";
      }
    } else {
      str << "    register " << t << " &" << n << " = ((" << t << "*)(((char*)__wrData) + ((int*)__wrData)[ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE + " << scalarArrayIndex << "]))[__elementIndex];\n\n";
      scalarArrayIndex++;
    }

    curParam = curParam->next;
  }
// Include the function body (create a new scope for it so any variables we need to declare aren't overwritten)
  str  << "    // Include the function body of the AEM\n"
  //     <<"    printf(\"A[63]=\t %f \t B[63]=  %f  )\\n\", A[63], B[63]);\n"
       << "    {\n"
       << "      /********** Start User Code **********/\n"
       << accelCodeBody->get_string()
       << "      /********** End User Code **********/\n"
       << "    }\n\n";

  str << "  } // end if (__threadIndex < __totalThreads)\n\n";
// Write the bit patterns
  str << "  // Write out the header bit patterns (different thread for each pattern; host will check for these to ensure the kernel executed)\n"
      << "  ((int*)__wrData)[ACCEL_CUDA_KERNEL_BIT_PATTERN_0_INDEX] = ACCEL_CUDA_KERNEL_BIT_PATTERN_0; \n"
      << "  ((int*)__wrData)[ACCEL_CUDA_KERNEL_BIT_PATTERN_1_INDEX] = ACCEL_CUDA_KERNEL_BIT_PATTERN_1; \n";

  str << "}\n\n";

  XStr *threadsPerBlock = getCudaManualThreadsPerBlock();

  // Create a info message if threadsPerBlock is not set (which can be removed once CUDA error checking is
  //   reliable and doesn't cause further issues once one error occurs).
  // DMK - NOTE | TODO | FIXME - The CUDA error reporting mechanism is used by the generated code (below) to test
  //   various threads-per-block values.  However, when using the generated code, we noticed that a success status
  //   was sometimes read from the stream even though the kernel itself had failed to execute.  This usually
  //   happened if resources where just below the maximum available (e.g. using 98% of the physical registers).
  //   So, for now, I'm including this error message until the generated code is fixed and/or the CUDA error
  //   reporting mechanism is known to be 100% reliable.
#if 0
  if (threadsPerBlock == NULL) {
    printf("[INFO] :: An accelerated entry method that does not use \"threadsperblock\" was detected (\"%s::%s\")\n"
  "[INFO] ::   Consider specifying the number of threads per block to use.  Otherwise, code relying on\n"
           "[INFO] ::   the CUDA error mechanism will be used.  At the time this message was written, the CUDA error\n"
           "[INFO] ::   mechanism wasn't completely reliable, resulting in the generated code failing in some cases.\n",
           *(container->baseName()), *(epStr())
          );
  }

#endif
  // Declare the host side of the cuda kernel (launch location)
  str << "// AEM : \"" << container->baseName() << "::" << epStr() << "\"\n"
      << "// Declare the host side function that actually launches the kernel (called by kernelSelect function)\n"
      << "void __cudaFuncHost__" << containerType << "__" << epStr() << "(workRequest *wr) {\n\n"

      << "  // Declare a static variable that keeps track of how many threads to use per block\n"
      << "  // DMK - NOTE | TODO | FIXME - This mechanism is not SMP-safe (use of static variable)\n"
      << "  static int discThreadsPerSM = " << ((threadsPerBlock == NULL) ? ("-1") : (*threadsPerBlock)) << ";\n\n"

      << "  // Read the total number of threads that should be used for this kernel execution\n"
      << "  int totalThreads = wr->dimBlock.x;\n\n"

      << "  // If the number of threads to use has not been discovered, discover it now\n"
      << "  if (discThreadsPerSM == -1) {\n\n"

      << "    int maxThreadsPerSM = 256;\n"
      << "    int threadsPerWarp = 32;\n"
      << "    int threadsPerBlock = ((maxThreadsPerSM < totalThreads) ? (maxThreadsPerSM) : (totalThreads));\n"
      << "    if (threadsPerBlock % threadsPerWarp != 0) { threadsPerBlock += threadsPerWarp - (threadsPerBlock % threadsPerWarp); }\n"
 << "    int numBlocks = -1;\n\n"

      << "    // Create a loop that tests different threadsPerBlock values until a successful kernel launch occurs\n"
      << "    cudaError_t cudaErrorCode;\n"
      << "    for (; threadsPerBlock > 0; threadsPerBlock -= threadsPerWarp) {\n\n"
      << "      // Calculate the block and grid sizes based on threadsPerBlock\n"
      << "      discThreadsPerSM = threadsPerBlock;\n"
      << "      numBlocks = (totalThreads / threadsPerBlock) + ((totalThreads % threadsPerBlock) ? (1) : (0));\n"
      << "      wr->dimGrid.x = numBlocks;\n"
      << "      wr->dimBlock.x = threadsPerBlock;\n\n"
      << "      // Try to issue the kernel, checking any error code returned\n"
      << "      cudaStreamSynchronize(kernel_stream);\n"
      << "      __cudaFuncKernel__" << containerType << "__" << epStr() << "<<<wr->dimGrid, wr->dimBlock, wr->smemSize, kernel_stream>>>(devBuffers[wr->bufferInfo[0].bufferID], totalThreads);\n"
      << "      cudaStreamSynchronize(kernel_stream);\n"
      << "      cudaErrorCode = cudaStreamQuery(kernel_stream);\n"
      << "      if (cudaErrorCode == cudaErrorLaunchOutOfResources) {\n"
      << "        continue;  // Try again with fewer threads per block\n"
      << "      } else {\n"
      << "        break;  // Success or some other error besides 'out of resources'\n"
      << "      }\n"
      << "    } // end for (threadsPerBlock > 0)\n\n"

      << "    // If the above loop resulted in any error code besides cudaSuccess, report it\n"
      << "    if (cudaErrorCode != cudaSuccess) {\n"
      << "      printf(\"[ACCEL-ERROR] - __cudaFuncKernel__" << containerType << "__" << epStr() << " (generated code) -- kernel execution failed! (CUDA reports - %d: \\\"%s\\\")\\n\", cudaErrorCode, cudaGetErrorString(cudaErrorCode));\n"
      << "    } else {\n"
      << "      traceKernelIssueTime();\n"
      << "    }\n\n"

  << "  // Otherwise, just use the previously discovered amount\n"
      << "  } else {\n"
      << "    wr->dimGrid.x = (totalThreads / discThreadsPerSM) + ((totalThreads % discThreadsPerSM) ? (1) : (0));\n"
      << "    wr->dimBlock.x = discThreadsPerSM;\n"
      << "    traceKernelIssueTime();\n"
      << "    __cudaFuncKernel__" << containerType << "__" << epStr() << "<<<wr->dimGrid, wr->dimBlock, wr->smemSize, kernel_stream>>>(devBuffers[wr->bufferInfo[0].bufferID], totalThreads);\n" //, splitAmount);\n"
      << "  }\n"
      << "}\n\n";

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

 void Entry::setAccelSplitAmount(XStr* sa) { splitAmount = sa; }
 void Entry::setCudaManualThreadsPerBlock(XStr *cmtpb) { cudaManualThreadsPerBlock = cmtpb; }
 int Entry::isTriggered(void) { return (attribs & STRIGGERED); }
 XStr* Entry::getAccelSplitAmount() { return splitAmount; }
 XStr* Entry::getCudaManualThreadsPerBlock() { return cudaManualThreadsPerBlock; }


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
    #if CMK_CUDA != 0
      genAccelIndexWrapperDecl_cuda(str);
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

    // TODO : Add a test to see if this is an array element or a singleton

    #if CMK_CELL != 0 || CMK_CUDA != 0

      // Grab the local accelerator manager
      str << "\n  // Get a pointer to the accelerator manager\n"
          << "  AccelManager *accelMgr = AccelManager::getAccelManager();\n\n"
          << "  // Get some info related to the AEM and use that information as part of the decision process\n"
          << "  // NOTE | TODO | FIXME : This is a bit of a hack for now.  Calling CkLocMgr::numLocalElements()\n"
          << "  //   is very expensive (iterates through the element records), so we want to call it as few times\n"
          << "  //   as possible.  For now, we will have the accelerator manager buffer the element counts (which\n"
          << "  //   will be reset upon load balancing finishing).\n"
          << "  int numElements = accelMgr->numElementsLookup(((ArrayElement*)impl_obj)->getLocMgr());\n"
          << "  int isTriggered = " << ((isTriggered()) ? ("-1") : ("0")) << ";\n"
          << "  int isSplittable = " << ((getAccelSplitAmount() != NULL) ? ("-1") : ("0")) << ";\n"
      #if CMK_CELL != 0
          << "  int funcIndex = " << indexName() << "::accel_spe_func_index__" << epStr() << ";\n\n"
      #elif CMK_CUDA != 0
          << "  int funcIndex = " << indexName() << "::accel_cuda_func_index__" << epStr() << ";\n\n"
      #endif
          << "  // Get a decision about what should be done with this AEM invocation\n"
          << "  AccelDecision decision;\n"
          << "  AccelError err = accelMgr->decide(funcIndex, 1, numElements, isTriggered, isSplittable, decision, impl_obj);\n"
          << "  if (err != ACCEL_SUCCESS) { printf(\"[ERROR] :: AccelManager::decide returned with error %s...\\n\", accelErrorString(err)); }\n\n"
          << "  // Direct the AEM to the chosen device\n"
          << "  switch (decision.deviceType) {\n";

      // Add in a case for CELL SPEs
      #if CMK_CELL != 0
        str << "\n"
            << "    case ACCEL_DEVICE_SPE:  // The SPE cores on the Cell processor\n"
            << "      _accelCall_spe_" << epStr() << "(";
        genAccelFullCallList(str);
        str << ");\n"
            << "      break;\n";
      #endif

      // Add in a case for CUDA-based GPUs
      #if CMK_CUDA != 0
        str << "\n"
            << "    case ACCEL_DEVICE_GPU_CUDA:  // CUDA-based GPGPU devices\n"
            << "      _accelCall_cuda_" << epStr() << "("; genAccelFullCallList(str); str << ", numElements, decision.issueFlag);\n"
            << "      break;\n";
      #endif

        // Add in a case for delayed host function invocations
      str << "\n"
          << "    case ACCEL_DEVICE_HOST_DELAY:  // Delayed host invocations\n"
          << "      _accelCall_generalDelay_" << epStr() << "("; genAccelFullCallList(str); str << ");\n"
          << "      break;\n";

      // Start the default case and use the general call that is always
      //   included as the body of the case
      str << "\n"
            << "    default:\n    ";  // NOTE: Add some spaces for formatting

    #endif

    // NOTE: Do this outside of the accelerator support checks so that it is always present, even if there is not switch statement
    str << "  _accelCall_general_" << epStr() << "("; genAccelFullCallList(str); str << ");\n";

    // Finish off the switch statement started above
    #if CMK_CELL != 0 || CMK_CUDA != 0

      str << "      break; // Not needed, but included for completeness\n"
          << "  }\n\n";

      // NOTE: This might cause issues if the host calls are a mixture of HOST and HOST_DELAYED (strategies should choose one or the other)
      str << "  // If the accelerator manager indicated that pending delayed host calls should be executed after this\n"
          << "  //   AEM has been processed, then do so now\n"
          << "  if (decision.issueDelayedFlag == ACCEL_ISSUE_TRUE) {\n"
          << "    accelMgr->issueDelayedGeneralCalls(" << indexName() << "::accel_cuda_func_index__" << epStr() << ");\n"
          << "  }\n\n";
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
#if CMK_CUDA != 0
    //if (isAccel() && isTriggered())
    if (isAccel()) {
      str << "int " << indexName() << "::" << "accel_cuda_func_index__" << epStr() << " = 0;\n";
    }
  #endif

  // Add special pre- and post- call code
  if(isSync() || isIget()) {
  //A synchronous method can return a value, and must finish before
  // the caller can proceed.
    preMarshall<< "  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);\n";
    preCall<< "  void *impl_retMsg=";
    if(retType->isVoid()) {
      preCall << "CkAllocSysMsg();\n  ";
    } else {
      preCall << "(void *) ";
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
    #if CMK_CUDA != 0
      genAccelIndexWrapperDef_cuda(str);
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
      str << "));\n";
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

// DMK - Accel support
int Entry::isAccel(void) { return (attribs & SACCEL); }

int Entry::isMemCritical(void) { return (attribs & SMEM); }
int Entry::isReductionTarget(void) { return (attribs & SREDUCE); }

char *Entry::getEntryName() { return name; }
int Entry::getLine() { return line; }

}   // namespace xi
