#include <iostream.h>
#include <fstream.h>
#include <string.h>
#include <stdlib.h>
#include "xi-symbol.h"

void 
ConstructList::setExtern(int e) 
{
  Construct::setExtern(e);
  if(construct)
    construct->setExtern(e);
  if(next)
    next->setExtern(e);
}

void 
ConstructList::print(XStr& str) 
{
  if(construct)
    construct->print(str);
  if(next) 
    next->print(str);
}

void 
TParamList::print(XStr& str) 
{
  tparam->print(str); 
  if (next) { 
    str << ","; 
    next->print(str); 
  }
}

void 
PtrType::print(XStr& str)
{
  type->print(str);
  for(int i=0;i<numstars;i++)
    str << "*";
}

void 
TypeList::print(XStr& str)
{
  type->print(str);
  if(next) {
    str << ", ";
    next->print(str);
  }
}

void 
MemberList::print(XStr& str)
{
  member->print(str);
  if(next)
    next->print(str);
}


void 
Chare::print(XStr& str)
{
  if(external)
    str << "extern ";
  if(templat)
    templat->genSpec(str);
  switch(chareType) {
    case SCHARE: str << "chare "; break;
    case SMAINCHARE: str << "mainchare "; break;
    case SGROUP: str << "group "; break;
  }
  type->print(str);
  if(bases) { str << ": "; bases->print(str); }
  if(list) {
    str << "{\n"; list->print(str); str << "};\n";
  } else {
    str << ";\n";
  }
}

void 
Message::print(XStr& str)
{
  if(external)
    str << "extern ";
  if(templat)
    templat->genSpec(str);
  str << "message ";
  type->print(str);
  str << ";\n";
}

void 
TType::print(XStr& str)
{
  str << "class ";
  type->print(str);
  if(init) {
    str << "=";
    init->print(str);
  }
}

void 
TName::print(XStr& str)
{
  type->print(str);
  str << " " << name;
  if(val) {
    str << "=";
    str << val;
  }
}


void 
TVarList::print(XStr& str)
{
  tvar->print(str);
  if(next) {
    str << ", ";
    next->print(str);
  }
}

void 
Template::print(XStr& str)
{
  if(entity)
    entity->print(str);
}

void 
Entry::print(XStr& str)
{
  if(isThreaded())
    str << "threaded ";
  if(isSync())
    str << "sync ";
  if(retType) {
    retType->print(str);
    str << " ";
  }
  str << name << "(";
  if(param)
    param->print(str);
  str << ")";
  if(stacksize) {
    str << " stacksize = "; 
    stacksize->print(str);
  }
  str << ";\n";
}

void 
Module::print(XStr& str)
{
  if(external)
    str << "extern ";
  str << "module " << name;
  if(clist) {
    str << " {\n";
    clist->print(str);
    str << "}\n";
  } else {
    str << ";\n";
  }
}

void
Module::generate(void)
{
  XStr declstr, defstr;
  declstr<<"#ifndef _DECL_" << name << "_H_"<<endx;
  declstr<<"#define _DECL_" << name << "_H_"<<endx;
  declstr<<"#include \"charm++.h\""<<endx;
  clist->genDecls(declstr);
  declstr<<"#endif"<<endx;
  defstr<<"#ifndef _DEFS_" << name << "_H_"<<endx;
  defstr<<"#define _DEFS_" << name << "_H_"<<endx;
  clist->genDefs(defstr);
  defstr << "void _register" << name << "(void)" << endx;
  defstr << "{" << endx;
  defstr << "  static int _done = 0; if (_done) return; _done = 1;" << endx;
  clist->genReg(defstr);
  defstr << "}" << endx;
  if(isMain()) {
    defstr << "void CkRegisterMainModule(void) {" << endx;
    defstr << "  _register" << name << "();" << endx;
    defstr << "}" << endx;
  }
  defstr<<"#endif"<<endx;
  XStr topname, botname;
  topname << name << ".decl.h";
  botname << name << ".def.h";
  ofstream decl(topname.get_string()), def(botname.get_string());
  if(decl==0 || def==0) {
    cerr << "Cannot open " << name << ".decl.h "
         << "or " << name << ".def.h for writing!!" << endl;
    exit(1);
  }
  decl << declstr.get_string();
  def << defstr.get_string();
}

void 
ModuleList::print(XStr& str) 
{
  module->print(str);
  if(next)
    next->print(str);
}

void 
ModuleList::generate(void) 
{
  module->generate();
  if(next)
    next->generate();
}

void 
Readonly::print(XStr& str)
{
  if(external)
    str << "extern ";
  str << "readonly ";
  if(msg)
    str << "message ";
  type->print(str);
  if(msg)
    str << " *";
  else
    str << " ";
  str << name;
  if(dims)
    dims->print(str);
  str << ";\n";
}

void
MemberList::setChare(Chare *c)
{
  member->setChare(c);
  if(next)
    next->setChare(c);
}

void
ConstructList::genDecls(XStr& str)
{
  if(construct) {
    construct->genDecls(str);
    str<<endx;
  }
  if(next)
    next->genDecls(str);
}

void
ConstructList::genDefs(XStr& str)
{
  if(construct) {
    construct->genDefs(str);
    str<<endx;
  }
  if(next)
    next->genDefs(str);
}

void
ConstructList::genReg(XStr& str)
{
  if(construct) {
    construct->genReg(str);
    str<<endx;
  }
  if(next)
    next->genReg(str);
}

static const char *CIChareStart = // prefix, name
"{\n"
"  public:\n"
"    static int __idx;\n"
"    static void __register(char *s);\n"
;

static const char *CIChareEnd =
"};\n"
;

void
Chare::genChareDecls(XStr& str)
{
  str <<"class "<< chare_prefix();
  type->print(str);
  if(external || type->isTemplated()) {
    str <<";";
    return;
  }
  str << ": ";
  str <<" public virtual _CK_CID";
  if(bases!=0) {
    str << ", ";
    bases->genProxyNames(str);
  }
  str.spew(CIChareStart, chare_prefix(), getBaseName());
  str << "    ";
  str<<chare_prefix();
  type->print(str);
  str << "(CkChareID _cid) { _ck_cid = _cid; }\n";
  str << "    CkChareID ckGetChareId(void) { return _ck_cid; }\n";
  str << "    void ckSetChareId(CkChareID _cid) { _ck_cid = _cid; }\n";
  if(list)
    list->genDecls(str);
  str.spew(CIChareEnd);
}

void
Chare::genGroupDecls(XStr& str)
{
  str <<"class "<< group_prefix();
  type->print(str);
  if(external || type->isTemplated()) {
    str <<";";
    return;
  }
  str << ": ";
  str <<" public virtual _CK_GID";
  if(bases!=0) {
    str << ", ";
    bases->genProxyNames(str);
  }
  str.spew(CIChareStart, group_prefix(), getBaseName());
  str << "    ";
  str<<group_prefix();
  type->print(str);
  str << "(int _gid) { _ck_gid = _gid; }\n";
  str << "    int ckGetGroupId(void) { return _ck_gid; }\n";
  str << "    void ckSetGroupId(int _gid) { _ck_gid = _gid; }\n";
  str << "    ";
  type->print(str);
  if(templat) {
    templat->genVars(str);
  }
  str << "* ckLocalBranch(void) {\n";
  str << "      return (";
  type->print(str);
  if(templat) {
    templat->genVars(str);
  }
  str << " *) CkLocalBranch(_ck_gid);\n";
  str << "    }\n";
  str << "    static ";
  type->print(str);
  if(templat) {
    templat->genVars(str);
  }
  str << "* ckLocalBranch(int gID) {\n";
  str << "      return (";
  type->print(str);
  if(templat) {
    templat->genVars(str);
  }
  str << " *) CkLocalBranch(gID);\n";
  str << "    }\n";
  if(list)
    list->genDecls(str);
  str.spew(CIChareEnd);
}

void
Chare::genDecls(XStr& str)
{
  str << "/* DECLS: "; print(str); str << " */\n";
  if(templat)
    templat->genSpec(str);
  str << "class ";
  type->print(str);
  str << ";\n";
  if(templat)
    templat->genSpec(str);
  if(chareType==SCHARE||chareType==SMAINCHARE) {
    genChareDecls(str);
  } else if(chareType==SGROUP) {
    genGroupDecls(str);
  }
}

void
Chare::genDefs(XStr& str)
{
  str << "/* DEFS: "; print(str); str << " */\n";
  if(!templat) {
    if(external) {
      str << "extern int ";
      type->genProxyName(str);
      str << "::__idx;\n";
    } else {
      str << "int ";
      type->genProxyName(str);
      str << "::__idx=0;\n";
    }
  }
  if(external || type->isTemplated())
    return;
  if(templat)
    templat->genSpec(str);
  str << "void ";
  if(chareType==SCHARE||chareType==SMAINCHARE)
    str << chare_prefix();
  else
    str << group_prefix();
  type->print(str);
  if(templat)
    templat->genVars(str);
  str << "::__register(char *s)\n";
  str << "{\n";
  str << "  __idx = CkRegisterChare(s, sizeof(";
  type->print(str);
  if(templat)
    templat->genVars(str);
  str << "));\n";
  if(list)
    list->genReg(str);
  str << "}\n";
  if(list)
    list->genDefs(str);
}

void
Chare::genReg(XStr& str)
{
  str << "/* REG: "; print(str); str << " */\n";
  if(external || templat)
    return;
  str << "  ";
  if(chareType==SCHARE||chareType==SMAINCHARE)
    str << chare_prefix();
  else
    str << group_prefix();
  type->print(str);
  str << "::__register(\"";
  type->print(str);
  str << "\");\n";
}

static const char *CIMsgClass =
"{\n"
"  public:\n"
"    static int __idx;\n"
"    static void __register(char *s);\n"
"    void*operator new(size_t s){return CkAllocMsg(__idx,s,0);}\n"
"    void operator delete(void *p){CkFreeMsg(p);}\n"
"    void*operator new(size_t,void*p){return p;}\n"
"    void*operator new(size_t s, int p){return CkAllocMsg(__idx,s,p);}\n"
;

static const char *CIAllocDecl =
"    void *operator new(size_t s, int *sz, int p);\n"
;

void
Message::genDecls(XStr& str)
{
  str << "/* DECLS: "; print(str); str << " */\n";
  if(templat)
    templat->genSpec(str);
  str <<"class ";
  type->print(str);
  str <<";\n";
  if(templat)
    templat->genSpec(str);
  str <<"class "<< msg_prefix();
  type->print(str);
  if(external || type->isTemplated()) {
    str <<";";
    return;
  }
  str.spew(CIMsgClass);
  if(isVarsize()) {
    str.spew(CIAllocDecl);
  }
  str << "};\n";
}

void
Message::genDefs(XStr& str)
{
  str << "/* DEFS: "; print(str); str << " */\n";
  if(!(external||type->isTemplated())) {
    // generate register function
    if(templat) {
      templat->genSpec(str);
      str << " ";
    }
    str << "void ";
    str << msg_prefix();
    type->print(str);
    if(templat)
      templat->genVars(str);
    str << "::__register(char *s)\n";
    str << "{\n";
    str << "  __idx = CkRegisterMsg(s, ";
    if(isPacked()) {
      str << "(CkPackFnPtr) ";
      type->print(str);
      if(templat)
        templat->genVars(str);
      str << "::pack, ";
      str << "(CkUnpackFnPtr) ";
      type->print(str);
      if(templat)
        templat->genVars(str);
      str << "::unpack, ";
    } else {
      str << "0, 0, ";
    }
    str << "0, sizeof(";
    type->print(str);
    if(templat)
      templat->genVars(str);
    str << "));\n";
    str << "}\n";
    // generate varsize new operator
    if(isVarsize()) {
      if(templat) {
        templat->genSpec(str);
        str << " ";
      }
      str << "void *";
      str << msg_prefix();
      type->print(str);
      if(templat)
        templat->genVars(str);
      str << "::operator new(size_t s, int *sz, int p)\n";
      str << "{\n";
      str << "  return ";
      type->print(str);
      if(templat)
        templat->genVars(str);
      str << "::alloc(__idx, s, sz, p);\n";
      str << "}\n";
    }
  }
  if(!templat) {
    if(!external) {
      str << "int " << msg_prefix();
      type->print(str);
      str << "::__idx=0;\n";
    }
  }
}

void
Message::genReg(XStr& str)
{
  str << "/* REG: "; print(str); str << " */\n";
  if(!templat && !external) {
    str << "  " << msg_prefix();
    type->print(str);
    str << "::__register(\"";
    type->print(str);
    str << "\");\n";
  }
}

void
Template::genVars(XStr& str)
{
  str << " < ";
  if(tspec)
    tspec->genShort(str);
  str <<" > ";
}

void
Template::genSpec(XStr& str)
{
  str << "template ";
  str << "< ";
  if(tspec)
    tspec->genLong(str);
  str <<" > ";
}

void
Template::genDecls(XStr& str)
{
  if(!external && entity) {
    entity->genDecls(str);
  }
}

void
Template::genDefs(XStr& str)
{
  if(!external && entity)
    entity->genDefs(str);
}

void
Template::genReg(XStr& str)
{
}

void
TVarList::genLong(XStr& str)
{
  if(tvar)
    tvar->genLong(str);
  if(next) {
    str << ", ";
    next->genLong(str);
  }
}

void
TVarList::genShort(XStr& str)
{
  if(tvar)
    tvar->genShort(str);
  if(next) {
    str << ", ";
    next->genShort(str);
  }
}

void TType::genLong(XStr& str)
{
  str << "class ";
  if(type)
    type->print(str);
  if(init) {
    str << "=";
    init->print(str);
  }
}

void TType::genShort(XStr& str)
{
  if(type)
    type->print(str);
}

void TName::genLong(XStr& str)
{
  if(type)
    type->print(str);
  str << " " << name;
  if(val) {
    str << "=" << val;
  }
}

void TName::genShort(XStr& str)
{
  str << name;
}

static const char *CIExternModule = // modulename
"#include \"\1.decl.h\"\n"
"extern void _register\1(void);\n"
;

void
Module::genDecls(XStr& str)
{
  if(external) {
    str.spew(CIExternModule, name);
  } else {
    clist->genDecls(str);
  }
}

void
Module::genDefs(XStr& str)
{
  if(external) {
  } else {
    clist->genDefs(str);
  }
}

void
Module::genReg(XStr& str)
{
  if(external) {
    str << "  _register" << name << "();" << endx;
  } else {
    clist->genDefs(str);
  }
}

void
Readonly::genDecls(XStr& str)
{
  str << "/* DECLS: "; print(str); str << " */\n";
  if(container) { // local static var
  } else { // global var
    str<<"extern ";
    type->print(str);
    if(msg)
      str << "*";
    str<<" "<<name;
    if(dims)
      dims->print(str);
    str << ";";
  }
}

void
Readonly::genDefs(XStr& str)
{
  str << "/* DEFS: "; print(str); str << " */\n";
}

void
Readonly::genReg(XStr& str)
{
  if(external)
    return;
  if(msg) {
    if(dims) {
      cerr << "Readonly Message cannot be an array!!\n";
      exit(1);
    }
    str << "  CkRegisterReadonlyMsg((void **) &";
    if(container) {
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << "::";
    }
    str << name << ");\n";
  } else {
    str << "  CkRegisterReadonly(sizeof(";
    type->print(str);
    if(dims)
      dims->print(str);
    str << "), (void *) &";
    if(container) {
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << "::";
    }
    str << name << ");\n";
  }
}

void TParamList::genSpec(XStr& str)
{
  if(tparam)
    tparam->genSpec(str);
  if(next) {
    str << ", ";
    next->genSpec(str);
  }
}

void TypeList::genProxyNames(XStr& str)
{
  if(type) {
    str << "public ";
    type->genProxyName(str);
  }
  if(next) {
    str << ", ";
    type->genProxyName(str);
  }
}

void MemberList::genDecls(XStr& str)
{
  if(member)
    member->genDecls(str);
  if(next) {
    str << endx;
    next->genDecls(str);
  }
}

void MemberList::genDefs(XStr& str)
{
  if(member)
    member->genDefs(str);
  if(next) {
    str << endx;
    next->genDefs(str);
  }
}

void MemberList::genReg(XStr& str)
{
  if(member)
    member->genReg(str);
  if(next) {
    str << endx;
    next->genReg(str);
  }
}

void
Entry::genEpIdx(XStr& str)
{
  str << name;
  if(param)
    str << "_" << param->getBaseName();
}

void Entry::genEpIdxDecl(XStr& str)
{
  str << "    static int __idx_";
  genEpIdx(str);
  str << ";\n";
}

void Entry::genChareStaticConstructorDecl(XStr& str)
{
  str << "    static void ckNew(";
  if(param) {
    if(!param->isVoid()) {
      param->print(str);
      str << ", ";
    }
  }
  str << "int onPE=CK_PE_ANY);\n";
  str << "    static void ckNew(";
  if(param) {
    if(!param->isVoid()) {
      param->print(str);
      str << ", ";
    }
  }
  str << "CkChareID* pcid, int onPE=CK_PE_ANY);\n";
  str << "    ";
  container->genProxyName(str);
  str << "(";
  if(param) {
    if(!param->isVoid()) {
      param->print(str);
      str << ", ";
    }
  }
  str << "int onPE=CK_PE_ANY);\n";
}

void Entry::genGroupStaticConstructorDecl(XStr& str)
{
  str << "    static int ckNew(";
  if(param) {
    param->print(str);
    if(!param->isVoid())
      str << "msg";
  }
  str << ");\n";
  str << "    ";
  container->genProxyName(str);
  str << "(";
  if(param) {
    if(!param->isVoid()) {
      param->print(str);
      str << "msg,";
    }
  }
  str << "int retEP, CkChareID *cid);\n";
  str << "    ";
  container->genProxyName(str);
  str << "(";
  if(param) {
    param->print(str);
    if(!param->isVoid()) {
      str << "msg";
    }
  }
  str << ");\n";
}

void Entry::genChareDecl(XStr& str)
{
  if(isConstructor()) {
    genChareStaticConstructorDecl(str);
  } else {
    // entry method declaration
    str << "    ";
    if(retType==0) {
      cerr << "Entry methods must specify a return type: ";
      cerr << "use void if necessary\n";
      exit(1);
    }
    retType->print(str);
    str << " " << name;
    str << "(";
    assert(param!=0);
    param->print(str);
    str<< ");\n";
    // entry ptr declaration
    str << "    static int ckIdx_" << name << "(";
    assert(param!=0);
    param->print(str);
    str<< ") { return __idx_"; 
    genEpIdx(str); 
    str << "; }\n";
  }
}

void Entry::genGroupDecl(XStr& str)
{
  if(isConstructor()) {
    genGroupStaticConstructorDecl(str);
  } else {
    // entry method broadcast declaration
    str << "    ";
    if(retType==0) {
      cerr << "Entry methods must specify a return type: ";
      cerr << "use void if necessary\n";
      exit(1);
    }
    retType->print(str);
    str << " " << name;
    str << "(";
    if(param) {
      param->print(str);
      if(!param->isVoid())
        str << "msg";
    }
    str<< ") {\n";
    str << "      CkBroadcastMsgBranch(__idx_";
    genEpIdx(str);
    str << ", ";
    if(!param->isVoid())
      str << "msg, ";
    else
      str << "CkAllocSysMsg(), ";
    str << "_ck_gid);\n";
    str << "    }\n";
    // entry method onPE declaration
    str << "    ";
    retType->print(str);
    str << " " << name;
    str << "(";
    if(param && !param->isVoid()) {
      param->print(str);
      str << "msg, ";
    }
    str<< "int onPE) {\n";
    str << "      CkSendMsgBranch(__idx_";
    genEpIdx(str);
    str << ", ";
    if(!param->isVoid())
      str << "msg, ";
    else
      str << "CkAllocSysMsg(), ";
    str << "onPE, _ck_gid);\n";
    str << "    }\n";
    // entry ptr declaration
    str << "    static int ckIdx_" << name << "(";
    assert(param!=0);
    param->print(str);
    str<< ") { return __idx_"; 
    genEpIdx(str); 
    str << "; }\n";
  }
}

void Entry::genDecls(XStr& str)
{
  str << "/* DECLS: "; print(str); str << " */\n";
  genEpIdxDecl(str);
  if(container->getChareType()==SGROUP)
    genGroupDecl(str);
  else
    genChareDecl(str);
  // call function declaration
  str << "    static void ";
  str << " _call_";
  genEpIdx(str);
  str << "(";
  if(param) {
    str << "void* msg, ";
  } else {
    assert(isConstructor());
    str << "CkArgMsg* msg, ";
  }
  str << container->getBaseName();
  if(container->isTemplated())
    container->genVars(str);
  str<< "* obj);\n";
}

void Entry::genEpIdxDef(XStr& str)
{
  if(container->isTemplated())
    container->genSpec(str);
  str << "int ";
  container->genProxyName(str);
  if(container->isTemplated())
    container->genVars(str);
  str << "::__idx_";
  genEpIdx(str);
  str << "=0;\n";
}

void Entry::genChareStaticConstructorDefs(XStr& str)
{
  if(container->isTemplated())
    container->genSpec(str);
  str << "void ";
  container->genProxyName(str);
  if(container->isTemplated())
    container->genVars(str);
  str << "::ckNew(";
  if(param && !param->isVoid()) {
    param->print(str);
    str << "msg, ";
  }
  str << "int onPE)\n";
  str << "{\n";
  if(!param || param->isVoid()) {
    str << "  void *msg = CkAllocSysMsg();\n";
  }
  str << "  CkCreateChare(__idx, __idx_";
  genEpIdx(str);
  str << ", msg, 0, onPE);\n";
  str << "}\n";

  if(container->isTemplated())
    container->genSpec(str);
  str << "void ";
  container->genProxyName(str);
  if(container->isTemplated())
    container->genVars(str);
  str << "::ckNew(";
  if(param && !param->isVoid()) {
    param->print(str);
    str << "msg, ";
  }
  str << "CkChareID* pcid, int onPE)\n";
  str << "{\n";
  if(!param || param->isVoid()) {
    str << "  void *msg = CkAllocSysMsg();\n";
  }
  str << "  CkCreateChare(__idx, __idx_";
  genEpIdx(str);
  str << ", msg, pcid, onPE);\n";
  str << "}\n";

  if(container->isTemplated())
    container->genSpec(str);
  str << " ";
  container->genProxyName(str);
  if(container->isTemplated())
    container->genVars(str);
  str << "::";
  container->genProxyName(str);
  str << "(";
  if(param && !param->isVoid()) {
    param->print(str);
    str << "msg, ";
  }
  str << "int onPE)\n";
  str << "{\n";
  if(!param || param->isVoid()) {
    str << "  void *msg = CkAllocSysMsg();\n";
  }
  str << "  CkCreateChare(__idx, __idx_";
  genEpIdx(str);
  str << ", msg, &_ck_cid, onPE);\n";
  str << "}\n";
}

void Entry::genGroupStaticConstructorDefs(XStr& str)
{
  if(container->isTemplated())
    container->genSpec(str);
  str << "int ";
  container->genProxyName(str);
  if(container->isTemplated())
    container->genVars(str);
  str << "::ckNew(";
  if(param) {
    param->print(str);
    if(!param->isVoid())
      str << "msg";
  }
  str << ")\n";
  str << "{\n";
  if(param->isVoid()) {
    str << "  void *msg = CkAllocSysMsg();\n";
  }
  str << "  return CkCreateGroup(__idx, __idx_";
  genEpIdx(str);
  str << ", msg, 0, 0);\n";
  str << "}\n";

  if(container->isTemplated())
    container->genSpec(str);
  str << " ";
  container->genProxyName(str);
  if(container->isTemplated())
    container->genVars(str);
  str << "::";
  container->genProxyName(str);
  str << "(";
  if(param) {
    if(!param->isVoid()) {
      param->print(str);
      str << "msg, ";
    }
  }
  str << "int retEP, CkChareID *cid)\n";
  str << "{\n";
  if(param->isVoid()) {
    str << "  void *msg = CkAllocSysMsg();\n";
  }
  str << "  _ck_gid = CkCreateGroup(__idx, __idx_";
  genEpIdx(str);
  str << ", msg, retEP, cid);\n";
  str << "}\n";

  if(container->isTemplated())
    container->genSpec(str);
  str << " ";
  container->genProxyName(str);
  if(container->isTemplated())
    container->genVars(str);
  str << "::";
  container->genProxyName(str);
  str << "(";
  if(param) {
    param->print(str);
    if(!param->isVoid()) {
      str << "msg";
    }
  }
  str << ")\n";
  str << "{\n";
  if(param->isVoid()) {
    str << "  void *msg = CkAllocSysMsg();\n";
  }
  str << "  _ck_gid = CkCreateGroup(__idx, __idx_";
  genEpIdx(str);
  str << ", msg, 0, 0);\n";
  str << "}\n";
}

void Entry::genChareDefs(XStr& str)
{
  if(isConstructor()) {
    genChareStaticConstructorDefs(str);
  } else {
    // entry method declaration
    if(container->isTemplated())
      container->genSpec(str);
    assert(retType!=0);
    retType->print(str);
    str << " ";
    container->genProxyName(str);
    if(container->isTemplated())
      container->genVars(str);
    str << "::" << name << "(";
    assert(param!=0);
    param->print(str);
    if(!param->isVoid())
      str << "msg";
    str << ")\n{\n";
    if(param->isVoid())
      str << "  void *msg = CkAllocSysMsg();\n";
    str << "  CkSendMsg(__idx_";
    genEpIdx(str);
    str << ", msg, &_ck_cid);\n";
    str << "}\n";
  }
}

void Entry::genGroupDefs(XStr& str)
{
  if(isConstructor()) {
    genGroupStaticConstructorDefs(str);
  } else {
  }
}

void Entry::genDefs(XStr& str)
{
  str << "/* DEFS: "; print(str); str << " */\n";
  genEpIdxDef(str);
  if(container->getChareType()==SGROUP)
    genGroupDefs(str);
  else
    genChareDefs(str);
  // call function
  if(container->isTemplated())
    container->genSpec(str);
  str << " void ";
  container->genProxyName(str);
  if(container->isTemplated())
    container->genVars(str);
  str << "::_call_";
  genEpIdx(str);
  str << "(";
  if(param) {
    str << "void* msg, ";
  } else {
    assert(isConstructor());
    str << "CkArgMsg* msg, ";
  }
  str << container->getBaseName();
  if(container->isTemplated())
    container->genVars(str);
  str<< "* obj)\n";
  str << "{\n";
  if(isConstructor()) {
    str << "  new (obj) " << container->getBaseName();
    if(container->isTemplated())
      container->genVars(str);
    if(param) {
      if(!param->isVoid()) {
        str << "((";
        param->print(str);
        str << ")msg);\n";
      } else {
        str << "();\n";
        str<< "  CkFreeSysMsg(msg);\n";
      }
    } else {
      str << "((CkArgMsg*)msg);\n";
    }
  } else {
    str << "  obj->" << name << "(";
    if(param->isVoid()) {
      str << ");\n";
      str << "  CkFreeSysMsg(msg);\n";
    } else {
      str << "(";
      param->print(str);
      str << ") msg);\n";
    }
  }
  str << "}\n";
}

void Entry::genReg(XStr& str)
{
  str << "/* REG: "; print(str); str << " */\n";
  str << "  __idx_";
  genEpIdx(str);
  str << " = CkRegisterEp(\"" << name << "\", "
      << "(CkCallFnPtr)_call_";
  genEpIdx(str);
  str << ", ";
  if(param && !param->isVoid()) {
    param->genMsgProxyName(str);
    str << "::__idx, ";
  } else {
    str << "0, ";
  }
  str << "__idx);\n";
  if(container->getChareType()==SMAINCHARE && isConstructor()) {
    str << "  CkRegisterMainChare(__idx, __idx_";
    genEpIdx(str);
    str << ");\n";
  }
}
