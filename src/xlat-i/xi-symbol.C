#include <iostream.h>
#include <fstream.h>
#include <string.h>
#include <stdlib.h>
#include "xi-symbol.h"

CompileMode compilemode;

Value::Value(char *s)
{
  factor = 1;
  val = s;
  if(val == 0 || strlen(val)==0 ) return;
  int pos = strlen(val)-1;
  if(val[pos]=='K' || val[pos]=='k') {
    val[pos] = '\0';
    factor = 1024;
  }
  if(val[pos]=='M' || val[pos]=='m') {
    val[pos] = '\0';
    factor = 1024*1024;
  }
}


int
Value::getIntVal(void)
{
  if(val==0 || strlen(val)==0) return 0;
  return (atoi((const char *)val)*factor);
}

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
NamedType::print(XStr& str)
{
  str << name;
  if(tparams) {
    str << "<";
    tparams->print(str);
    str << ">";
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
    case SNODEGROUP: str << "nodegroup "; break;
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
Module::generate()
{
  XStr declstr, defstr;
  
  declstr<<"#ifndef _DECL_" << name << "_H_"<<endx;
  declstr<<"#define _DECL_" << name << "_H_"<<endx;
  declstr<<"#include \"charm++.h\""<<endx;
  clist->genDecls(declstr);
  declstr<<"extern void _register"<<name<<"(void);"<<endx;
  if(isMain()) {
    defstr << "extern void CkRegisterMainModule(void);" << endx;
  }
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
    defstr << "  _REGISTER_DONE();" << endx;
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
ModuleList::generate()
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

int
MemberList::isPure(void)
{
  if(member->isPure())
    return 1;
  if(next)
    return next->isPure();
  return 0;
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
"    static void __register(const char *s);\n"
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
    bases->genProxyNames(str, "public ", "", ", ");
  }
  str.spew(CIChareStart, chare_prefix(), getBaseName());
  if(isAbstract()) {
    str << "    ";
    str<<chare_prefix();
    type->print(str);
    str << "(void) {};\n";
  }
  str << "    ";
  str<<chare_prefix();
  type->print(str);
  str << "(CkChareID __cid) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames(str, "", "(__cid)", ", ");
  }
  str << "{ ckSetChareId(__cid); }\n";
  str << "    CkChareID ckGetChareId(void) { return _ck_cid; }\n";
  str << "    void ckSetChareId(CkChareID __cid){_CHECK_CID(__cid,__idx);_ck_cid=__cid;}\n";
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
  if(chareType==SGROUP)
    str <<" public virtual _CK_GID";
  else
    str <<" public virtual _CK_NGID";
  if(bases!=0) {
    str << ", ";
    bases->genProxyNames(str, "public ", "", ", ");
  }
  str.spew(CIChareStart, group_prefix(), getBaseName());
  if(isAbstract()) {
    str << "    ";
    str<<group_prefix();
    type->print(str);
    str << "(void) {};\n";
  }
  str << "    ";
  str<<group_prefix();
  type->print(str);
  str << "(CkGroupID _gid) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames(str, "", "(_gid)", ", ");
  }
  if(chareType==SGROUP) {
    str << "{ _ck_gid = _gid; _setChare(0); }\n";
  } else {
    str << "{ _ck_ngid = _gid; _setChare(0); }\n";
  }
  str << "    ";
  str<<group_prefix();
  type->print(str);
  str << "(CkChareID __cid) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames(str, "", "(__cid)", ", ");
  }
  str << "{ ckSetChareId(__cid); }\n";
  str << "    CkChareID ckGetChareId(void) { return _ck_cid; }\n";
  str << "    void ckSetChareId(CkChareID __cid){_CHECK_CID(__cid,__idx);_ck_cid=__cid;_setChare(1);}\n";
  if(chareType==SGROUP) {
    str << "    CkGroupID ckGetGroupId(void) { return _ck_gid; }\n";
    str << "   void ckSetGroupId(CkGroupID _gid){_ck_gid=_gid;_setChare(0);}\n";
  } else {
    str << "    CkGroupID ckGetGroupId(void) { return _ck_ngid; }\n";
    str << "  void ckSetGroupId(CkGroupID _gid){_ck_ngid=_gid;_setChare(0);}\n";
  }
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
  if(chareType==SGROUP)
    str << " *) CkLocalBranch(_ck_gid);\n";
  else
    str << " *) CkLocalNodeBranch(_ck_ngid);\n";
  str << "    }\n";
  str << "    static ";
  type->print(str);
  if(templat) {
    templat->genVars(str);
  }
  str << "* ckLocalBranch(CkGroupID gID) {\n";
  str << "      return (";
  type->print(str);
  if(templat) {
    templat->genVars(str);
  }
  if(chareType==SGROUP)
    str << " *) CkLocalBranch(gID);\n";
  else
    str << " *) CkLocalNodeBranch(gID);\n";
  str << "    }\n";
  if(list)
    list->genDecls(str);
  str.spew(CIChareEnd);
}

void
Chare::genArrayDecls(XStr& str)
{
  str <<"class "<< array_prefix();
  type->print(str);
  if(external || type->isTemplated()) {
    str <<";";
    return;
  }
  str << ": ";
  str <<" public virtual _CK_AID";
  if(bases!=0) {
    str << ", ";
    bases->genProxyNames(str, "public ", "", ", ");
  }
  str.spew(CIChareStart, array_prefix(), getBaseName());
  if(isAbstract()) {
    str << "    ";
    str<<array_prefix();
    type->print(str);
    str << "(void) {};\n";
  }
  str << "    ";
  str<<array_prefix();
  type->print(str);
  str << "(CkAID _aid) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames(str, "", "(_aid)", ", ");
  }
  str << "{ ckSetArrayId(_aid);}\n";
  str << "    ";
  str<<array_prefix();
  type->print(str);
  str << "(CkChareID __cid) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames(str, "", "(__cid)", ", ");
  }
  str << "{ ckSetChareId(__cid);}\n";
  str << "    CkAID ckGetArrayId(void) { return CkAID(_ck_aid, _elem); }\n";
  str << "    void ckSetArrayId(CkAID _aid) { \n";
  str << "      _setChare(0); _setAid(_aid._ck_aid); _elem = _aid._elem; \n";
  str << "    }\n";
  str << "    CkChareID ckGetChareId(void) { return _cid; }\n";
  str << "    void ckSetChareId(CkChareID __cid) { \n";
  str << "      _CHECK_CID(__cid, __idx); _setChare(1); _setCid(__cid); \n";
  str << "    }\n";
  str << "    " << array_prefix();
  type->print(str);
  str << " operator [] (int idx) {\n";
  str << "      return " << array_prefix();
  type->print(str);
  str << "(CkAID(_ck_aid, idx));\n";
  str << "    }\n";
  if(list)
    list->genDecls(str);
  str.spew(CIChareEnd);
}

void
Chare::genDecls(XStr& str)
{
  if(type->isTemplated())
    return;
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
  } else if(chareType==SGROUP || chareType==SNODEGROUP) {
    genGroupDecls(str);
  } else if(chareType==SARRAY) {
    genArrayDecls(str);
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
  if(chareType==SCHARE||chareType==SMAINCHARE) {
    str << chare_prefix();
  } else if(chareType==SGROUP || chareType==SNODEGROUP) {
    str << group_prefix();
  } else if(chareType==SARRAY) {
    str << array_prefix();
  }
  type->print(str);
  if(templat)
    templat->genVars(str);
  str << "::__register(const char *s)\n";
  str << "{\n";
  str << "  __idx = CkRegisterChare(s, sizeof(";
  type->print(str);
  if(templat)
    templat->genVars(str);
  str << "));\n";
  // register all bases
  if(bases !=0) {
    bases->genProxyNames(str, "_REGISTER_BASE(__idx, ", "::__idx);\n", "");
  }
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
  if(chareType==SCHARE||chareType==SMAINCHARE) {
    str << chare_prefix();
  } else if (chareType==SGROUP || chareType==SNODEGROUP) {
    str << group_prefix();
  } else if (chareType==SARRAY) {
    str << array_prefix();
  }
  type->print(str);
  str << "::__register(\"";
  type->print(str);
  str << "\");\n";
}

static const char *CIMsgClass =
"{\n"
"  public:\n"
"    static int __idx;\n"
"    static void __register(const char *s);\n"
"    void*operator new(size_t s){return CkAllocMsg(__idx,s,0);}\n"
"    void operator delete(void *p){CkFreeMsg(p);}\n"
"    void*operator new(size_t,void*p){return p;}\n"
"    void*operator new(size_t s, int p){return CkAllocMsg(__idx,s,p);}\n"
;

static const char *CIMsgClassAnsi =
"{\n"
"  public:\n"
"    static int __idx;\n"
"    static void __register(const char *s);\n"
"    void*operator new(size_t s){return CkAllocMsg(__idx,s,0);}\n"
"    void operator delete(void *p){CkFreeMsg(p);}\n"
"    void*operator new(size_t,void*p){return p;}\n"
"    void operator delete(void*,void*){}\n"
"    void*operator new(size_t s, int p){return CkAllocMsg(__idx,s,p);}\n"
"    void operator delete(void *,int){}\n"
;

static const char *CIAllocDecl =
"    void *operator new(size_t s, int *sz, int p);\n"
;

static const char *CIAllocDeclAnsi =
"    void *operator new(size_t s, int *sz, int p);\n"
"    void operator delete(void*,int *,int);\n"
;

void
Message::genDecls(XStr& str)
{
  if(type->isTemplated())
    return;
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
  if (compilemode==ansi)
    str.spew(CIMsgClassAnsi);
  else str.spew(CIMsgClass);

  if(isVarsize()) {
    if (compilemode==ansi)
      str.spew(CIAllocDeclAnsi);
    else str.spew(CIAllocDecl);
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
    str << "::__register(const char *s)\n";
    str << "{\n";
    str << "  __idx = CkRegisterMsg(s, ";
    if(isPacked()||isVarsize()) {
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

      if(compilemode==ansi) {
        // Generate corresponding delete
        if(templat) {
          templat->genSpec(str);
          str << " ";
        }
        str << "void ";
        str << msg_prefix();
        type->print(str);
        if(templat)
          templat->genVars(str);
        str << "::operator delete(void *p, int *, int)\n";
        str << "{\n";
        str << "  CkFreeMsg(p);\n";
        str << "}\n";
      }
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
  if(!external)
    clist->genDefs(str);
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

void TypeList::genProxyNames(XStr& str, const char *prefix, 
                             const char *suffix, const char *sep)
{
  if(type) {
    str << prefix;
    type->genProxyName(str);
    str << suffix;
  }
  if(next) {
    str << sep;
    next->genProxyNames(str, prefix, suffix, sep);
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
  if(container->getChareType()==SARRAY && !param && isConstructor()) {
    str << "    static int __idx_" << name << "_ArrayElementCreateMessage;\n";
    str << "    static int __idx_" << name << "_ArrayElementMigrateMessage;\n";
  } else {
    str << "    static int __idx_";
    genEpIdx(str);
    str << ";\n";
  }
}

void Entry::genChareStaticConstructorDecl(XStr& str)
{
  if(container->isAbstract())
    return;
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
  // entry ptr declaration
  str << "    static int ckIdx_" << name << "(";
  if(param!=0)
    param->print(str);
  else
    str << "void";
  str<< ") { return __idx_"; 
  genEpIdx(str); 
  str << "; }\n";
}

void Entry::genGroupStaticConstructorDecl(XStr& str)
{
  if(container->isAbstract())
    return;
  str << "    static CkGroupID ckNew(";
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
  // entry ptr declaration
  str << "    static int ckIdx_" << name << "(";
  if(param!=0)
    param->print(str);
  else
    str << "void";
  str<< ") { return __idx_"; 
  genEpIdx(str); 
  str << "; }\n";
}

void Entry::genArrayStaticConstructorDecl(XStr& str)
{
  if(container->isAbstract())
    return;
  str << "    static CkAID ckNew(int numElements)\n";
  str << "{\n";
  str << "  return CkAID(Array1D::CreateArray(numElements,_RRMapID,\n";
  str << "    __idx, ConstructorIndex(";
  str << name;
  str << ", ArrayElementCreateMessage), ConstructorIndex(";
  str << name;
  str << ", ArrayElementMigrateMessage)),-1);\n";
  str << "}\n";
  str << "    static CkAID ckNew(int numElements, CkGroupID mapID)\n";
  str << "{\n";
  str << "  return CkAID(Array1D::CreateArray(numElements,mapID,\n";
  str << "    __idx, ConstructorIndex(";
  str << name;
  str << ", ArrayElementCreateMessage), ConstructorIndex(";
  str << name;
  str << ", ArrayElementMigrateMessage)),-1);\n";
  str << "}\n";
  str << "    ";
  container->genProxyName(str);
  str << "(int numElements)";
  if(container->isDerived()) {
    str << ": ";
    container->genProxyBases(str, "", "(numElements)", ", ");
  }
  str << "\n{\n";
  str << "  _setAid(Array1D::CreateArray(numElements, _RRMapID,\n";
  str << "    __idx, ConstructorIndex(";
  str << name;
  str << ", ArrayElementCreateMessage), ConstructorIndex(";
  str << name;
  str << ", ArrayElementMigrateMessage))); _setChare(0); _elem=-1;\n";
  str << "}\n";
  str << "    ";
  container->genProxyName(str);
  str << "(int numElements, CkGroupID mapID)";
  if(container->isDerived()) {
    str << ": ";
    container->genProxyBases(str, "", "(numElements, mapID)", ", ");
  }
  str << "\n{\n";
  str << "  _setAid(Array1D::CreateArray(numElements, mapID,\n";
  str << "    __idx, ConstructorIndex(";
  str << name;
  str << ", ArrayElementCreateMessage), ConstructorIndex(";
  str << name;
  str << ", ArrayElementMigrateMessage))); _setChare(0); _elem=-1;\n";
  str << "}\n";
  // entry ptr declaration
  str << "    static int ckIdx_" << name << "(";
  if(param!=0)
    param->print(str);
  else
    str << "ArrayElementCreateMessage*";
  str<< ") { return __idx_" << name << "_ArrayElementCreateMessage; }\n"; 
  str << "    static int ckIdx_" << name << "(";
  if(param!=0)
    param->print(str);
  else
    str << "ArrayElementMigrateMessage*";
  str<< ") { return __idx_" << name << "_ArrayElementMigrateMessage; }\n"; 
}

void Entry::genChareDecl(XStr& str)
{
  if(isConstructor()) {
    genChareStaticConstructorDecl(str);
  } else {
    // entry method declaration
    str << "    ";
    if(isVirtual())
      str << "virtual ";
    if(retType==0) {
      cerr << "Entry methods must specify a return type: ";
      cerr << "use void if necessary\n";
      exit(1);
    }
    retType->print(str);
    str << " " << name;
    str << "(";
    if(param == 0) {
      cerr << "Entry methods must specify a message parameter: ";
      cerr << "use void if necessary\n";
      exit(1);
    }
    param->print(str);
    str<< ");\n";
    // entry method declaration with future
    if(isSync()) {
      str << "    ";
      if(isVirtual())
        str << "virtual ";
      str << " void " << name << "(";
      if(!param->isVoid()) {
        param->print(str);
        str << ",";
      }
      str<< "CkFutureID*);\n";
    }
    // entry ptr declaration
    str << "    static int ckIdx_" << name << "(";
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
    if(!isSync()) {
      str << "    ";
      if(isVirtual())
        str << "virtual ";
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
      str<< ")";
      str << "{\n";
      str << "        if(_isChare()) {\n";
      str << "          CkSendMsg(__idx_";
      genEpIdx(str);
      str << ", ";
      if(!param->isVoid())
        str << "msg, &_ck_cid);\n";
      else
        str << "CkAllocSysMsg(), &_ck_cid);\n";
      str << "        } else {\n";
      if(container->getChareType()==SGROUP)
        str << "        CkBroadcastMsgBranch(__idx_";
      else
        str << "        CkBroadcastMsgNodeBranch(__idx_";
      genEpIdx(str);
      str << ", ";
      if(!param->isVoid())
        str << "msg, ";
      else
        str << "CkAllocSysMsg(), ";
      if(container->getChareType()==SGROUP)
        str << "_ck_gid);\n";
      else
        str << "_ck_ngid);\n";
      str << "      }\n";
      str << "    }\n";
    }
    // entry method onPE declaration
    str << "    ";
    if(isVirtual())
      str << "virtual ";
    retType->print(str);
    str << " " << name;
    str << "(";
    if(param && !param->isVoid()) {
      param->print(str);
      str << "msg, ";
    }
    str<< "int onPE)";
    str << " {\n";
    if(isSync()) {
      if(retType->isVoid()) {
        if(container->getChareType()==SGROUP)
          str << "    CkFreeSysMsg(CkRemoteBranchCall(__idx_";
        else
          str << "    CkFreeSysMsg(CkRemoteNodeBranchCall(__idx_";
      } else {
        str << "      return (";
        retType->print(str);
        if(container->getChareType()==SGROUP)
          str << ") (CkRemoteBranchCall(__idx_";
        else
          str << ") (CkRemoteNodeBranchCall(__idx_";
      }
      genEpIdx(str);
      str << ", ";
      if(!param->isVoid())
        str << "msg, ";
      else
        str << "CkAllocSysMsg(), ";
      str << "_ck_gid, onPE));\n";
      str << "    }\n";
    } else {
      if(container->getChareType()==SGROUP)
        str << "      CkSendMsgBranch(__idx_";
      else
        str << "      CkSendMsgNodeBranch(__idx_";
      genEpIdx(str);
      str << ", ";
      if(!param->isVoid())
        str << "msg, ";
      else
        str << "CkAllocSysMsg(), ";
      if(container->getChareType()==SGROUP)
        str << "onPE, _ck_gid);\n";
      else
        str << "onPE, _ck_ngid);\n";
      str << "    }\n";
    }
    // entry method onPE declaration with future
    if(isSync()) {
      str << "    ";
      if(isVirtual())
        str << "virtual ";
      str << "void " << name << "(";
      if(param && !param->isVoid()) {
        param->print(str);
        str << "msg, ";
      }
      str<< "int onPE, CkFutureID *fut)";
      str << " {\n";
      str << "      *fut = ";
      if(container->getChareType()==SGROUP)
        str << "CkRemoteBranchCallAsync(__idx_";
      else
        str << "CkRemoteNodeBranchCallAsync(__idx_";
      genEpIdx(str);
      str << ", ";
      if(!param->isVoid())
        str << "msg, ";
      else
        str << "CkAllocSysMsg(), ";
      str << "_ck_gid, onPE);\n";
      str << "    }\n";
    }
    // entry method forChare declaration with future
    if(isSync()) {
      str << "    ";
      if(isVirtual())
        str << "virtual ";
      str << "void " << name << "(";
      if(param && !param->isVoid()) {
        param->print(str);
        str << "msg, ";
      }
      str<< "CkFutureID *fut)";
      str << " {\n";
      str << "      *fut = ";
      str << "CkRemoteCallAsync(__idx_";
      genEpIdx(str);
      str << ", ";
      if(!param->isVoid())
        str << "msg, ";
      else
        str << "CkAllocSysMsg(), ";
      str << "&_ck_cid);\n";
      str << "    }\n";
    }
    // entry ptr declaration
    str << "    static int ckIdx_" << name << "(";
    assert(param!=0);
    param->print(str);
    str<< ") { return __idx_"; 
    genEpIdx(str); 
    str << "; }\n";
  }
}

void Entry::genArrayDecl(XStr& str)
{
  if(isConstructor()) {
    genArrayStaticConstructorDecl(str);
  } else {
    // entry method broadcast declaration
    str << "    ";
    if(isVirtual())
      str << "virtual ";
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
    str<< ")";
    str << " {\n";
    str << "      if(_isChare())  {\n";
    str << "        CkSendMsg(__idx_";
    genEpIdx(str);
    str << ", ";
    if(!param->isVoid())
      str << "msg, ";
    else
      str << "CkAllocMsg(0, sizeof(ArrayMessage),0), ";
    str << "&_cid);\n";
    str << "        return;\n";
    str << "      };\n";
    str << "      if (_elem==(-1)) _array->broadcast((ArrayMessage*) ";
    if(!param->isVoid())
      str << "msg, ";
    else
      str << "CkAllocMsg(0, sizeof(ArrayMessage),0), ";
    str << " __idx_";
    genEpIdx(str);
    str << ");\n";
    str << "      else _array->send((ArrayMessage*) ";
    if(!param->isVoid())
      str << "msg, ";
    else
      str << "CkAllocMsg(0, sizeof(ArrayMessage),0), ";
    str << "_elem, __idx_";
    genEpIdx(str);
    str << ");\n";
    str << "    }\n";
    // entry method onPE declaration
    str << "    ";
    if(isVirtual())
      str << "virtual ";
    retType->print(str);
    str << " " << name;
    str << "(";
    if(param && !param->isVoid()) {
      param->print(str);
      str << "msg, ";
    }
    str<< "int onPE)";
    str << " {\n";
    str << "      _array->send((ArrayMessage*) ";
    if(!param->isVoid())
      str << "msg, ";
    else
      str << "CkAllocMsg(0, sizeof(ArrayMessage),0), ";
    str << "onPE, __idx_";
    genEpIdx(str);
    str << ");\n";
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
  if(container->getChareType()==SGROUP || container->getChareType()==SNODEGROUP) {
    genGroupDecl(str);
  } else if(container->getChareType()==SARRAY) {
    genArrayDecl(str);
  } else { // chare or mainchare
    genChareDecl(str);
  }
  // call function declaration
  if(isConstructor() && container->isAbstract())
    return;
  if(container->getChareType()==SARRAY && !param && isConstructor()) {
    str << "    static void ";
    str << " _call_" << name << "_ArrayElementCreateMessage(void* msg, ";
    str << container->getBaseName();
    if(container->isTemplated())
      container->genVars(str);
    str<< "* obj);\n";
    str << "    static void ";
    str << " _call_" << name << "_ArrayElementMigrateMessage(void* msg, ";
    str << container->getBaseName();
    if(container->isTemplated())
      container->genVars(str);
    str<< "* obj);\n";
    if(isThreaded()) {
      str << "    static void ";
      str << " _callthr_" << name;
      str << "_ArrayElementCreateMessage(CkThrCallArg *);\n";
      str << "    static void ";
      str << " _callthr_" << name;
      str << "_ArrayElementMigrateMessage(CkThrCallArg *);\n";
    }
  } else {
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
    if(isThreaded()) {
      str << "    static void _callthr_";
      genEpIdx(str);
      str << "(CkThrCallArg *);\n";
    }
  }
}

void Entry::genEpIdxDef(XStr& str)
{
  if(container->getChareType()==SARRAY && isConstructor() && !param) {
    if(container->isTemplated())
      container->genSpec(str);
    str << "int ";
    container->genProxyName(str);
    if(container->isTemplated())
      container->genVars(str);
    str << "::__idx_" << name << "_ArrayElementCreateMessage=0;\n";
    if(container->isTemplated())
      container->genSpec(str);
    str << "int ";
    container->genProxyName(str);
    if(container->isTemplated())
      container->genVars(str);
    str << "::__idx_" << name << "_ArrayElementMigrateMessage=0;\n";
  } else {
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
}

void Entry::genChareStaticConstructorDefs(XStr& str)
{
  if(container->isAbstract())
    return;
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
  if(container->isAbstract())
    return;
  if(container->isTemplated())
    container->genSpec(str);
  str << "CkGroupID ";
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
  if(container->getChareType()==SGROUP)
    str << "  return CkCreateGroup(__idx, __idx_";
  else
    str << "  return CkCreateNodeGroup(__idx, __idx_";
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
  if(container->getChareType()==SGROUP)
    str << "  _ck_gid = CkCreateGroup(__idx, __idx_";
  else
    str << "  _ck_ngid = CkCreateNodeGroup(__idx, __idx_";
  genEpIdx(str);
  str << ", msg, retEP, cid);\n";
  str << "  _setChare(0);\n";
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
  if(container->getChareType()==SGROUP)
    str << "  _ck_gid = CkCreateGroup(__idx, __idx_";
  else
    str << "  _ck_ngid = CkCreateNodeGroup(__idx, __idx_";
  genEpIdx(str);
  str << ", msg, 0, 0);\n";
  str << "  _setChare(0);\n";
  str << "}\n";
}

void Entry::genChareDefs(XStr& str)
{
  if(isConstructor()) {
    genChareStaticConstructorDefs(str);
  } else {
    // entry method definition
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
    if(isSync()) {
      if(retType->isVoid()) {
        str << "  CkFreeSysMsg(CkRemoteCall(__idx_";
        genEpIdx(str);
        str << ", msg, &_ck_cid));\n";
      } else {
        str << "  return ("; retType->print(str);
        str << ") CkRemoteCall(__idx_";
        genEpIdx(str);
        str << ", msg, &_ck_cid);\n";
      }
    } else {
      str << "  CkSendMsg(__idx_";
      genEpIdx(str);
      str << ", msg, &_ck_cid);\n";
    }
    str << "}\n";
    // entry method definition with future
    if(isSync()) {
      if(container->isTemplated())
        container->genSpec(str);
      str << " void ";
      container->genProxyName(str);
      if(container->isTemplated())
        container->genVars(str);
      str << "::" << name << "(";
      assert(param!=0);
      if(!param->isVoid()) {
        param->print(str);
        str << "msg,";
      }
      str << "CkFutureID *fut)\n{\n";
      if(param->isVoid())
        str << "  void *msg = CkAllocSysMsg();\n";
      str << "  *fut = CkRemoteCallAsync(__idx_";
      genEpIdx(str);
      str << ", msg, &_ck_cid);\n";
      str << "}\n";
    }
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
  if(container->getChareType()==SGROUP||container->getChareType()==SNODEGROUP){
    genGroupDefs(str);
  } else if (container->getChareType()==SARRAY) {
  } else
    genChareDefs(str);
  // call function
  if(isConstructor() && container->isAbstract())
    return; // no call function for a constructor of an abstract chare
  if(container->getChareType()==SARRAY && !param && isConstructor()) {
    if(container->isTemplated())
      container->genSpec(str);
    str << " void ";
    container->genProxyName(str);
    if(container->isTemplated())
      container->genVars(str);
    str << "::_call_" << name << "_ArrayElementCreateMessage(void* msg,";
    str << container->getBaseName();
    if(container->isTemplated())
      container->genVars(str);
    str<< "* obj)\n";
    str << "{\n";
    if(isThreaded()) {
      str << "  CthAwaken(CthCreate((CthVoidFn)_callthr_" << name;
      str << "_ArrayElementCreateMessage, new CkThrCallArg(msg,obj), ";
      str << getStackSize() << "));\n}\n";
      if(container->isTemplated())
        container->genSpec(str);
      str << " void ";
      container->genProxyName(str);
      if(container->isTemplated())
        container->genVars(str);
      str << "::_callthr_" << name;
      str << "_ArrayElementCreateMessage(CkThrCallArg *arg)\n";
      str << "{\n";
      str << "  void *msg = arg->msg;\n";
      str << "  ";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << " *obj = (";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << " *) arg->obj;\n";
      str << "  delete arg;\n";
    }
    str << "  new (obj) " << container->getBaseName();
    if(container->isTemplated())
      container->genVars(str);
    str << "((ArrayElementCreateMessage*)msg);\n}\n";
    if(container->isTemplated())
      container->genSpec(str);
    str << " void ";
    container->genProxyName(str);
    if(container->isTemplated())
      container->genVars(str);
    str << "::_call_" << name << "_ArrayElementMigrateMessage(void* msg,";
    str << container->getBaseName();
    if(container->isTemplated())
      container->genVars(str);
    str<< "* obj)\n";
    str << "{\n";
    if(isThreaded()) {
      str << "  CthAwaken(CthCreate((CthVoidFn)_callthr_" << name;
      str << "_ArrayElementMigrateMessage, new CkThrCallArg(msg,obj), ";
      str << getStackSize() << "));\n}\n";
      if(container->isTemplated())
        container->genSpec(str);
      str << " void ";
      container->genProxyName(str);
      if(container->isTemplated())
        container->genVars(str);
      str << "::_callthr_" << name;
      str << "_ArrayElementMigrateMessage(CkThrCallArg *arg)\n";
      str << "{\n";
      str << "  void *msg = arg->msg;\n";
      str << "  ";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << " *obj = (";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << " *) arg->obj;\n";
      str << "  delete arg;\n";
    }
    str << "  new (obj) " << container->getBaseName();
    if(container->isTemplated())
      container->genVars(str);
    str << "((ArrayElementMigrateMessage*)msg);\n}\n";
  } else if(isSync()) {
    if(isConstructor()) {
      cerr << "Constructors cannot be sync methods." << endl;
      exit(1);
    }
    if(container->isTemplated())
      container->genSpec(str);
    str << " void ";
    container->genProxyName(str);
    if(container->isTemplated())
      container->genVars(str);
    str << "::_call_";
    genEpIdx(str);
    str << "(void* msg, ";
    str << container->getBaseName();
    if(container->isTemplated())
      container->genVars(str);
    str<< "* obj)\n";
    str << "{\n";
    if(isThreaded()) {
      str << "  CthAwaken(CthCreate((CthVoidFn)_callthr_";
      genEpIdx(str);
      str << ", new CkThrCallArg(msg, obj), ";
      str << getStackSize() << "));\n}\n";
      if(container->isTemplated())
        container->genSpec(str);
      str << " void ";
      container->genProxyName(str);
      if(container->isTemplated())
        container->genVars(str);
      str << "::_callthr_";
      genEpIdx(str);
      str << "(CkThrCallArg *arg)\n";
      str << "{\n";
      str << "  void *msg = arg->msg;\n";
      str << "  ";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << "* obj = (";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << " *) arg->obj;\n";
      str << "  delete arg;\n";
    }
    str << "  int ref = CkGetRefNum(msg);\n";
    str << "  int src = CkGetSrcPe(msg);\n";
    str << "  void *retMsg;\n";
    if(retType->isVoid()) {
      str << "  retMsg = CkAllocSysMsg();\n";
      str << "  obj->" << name << "(";
    } else {
      str << "  retMsg = (void *) obj->" << name << "(";
    }
    if(param->isVoid()) {
      str << ");\n";
      if(container->getChareType() == SARRAY)
        str << "  CkFreeMsg(msg);\n";
      else
        str << "  CkFreeSysMsg(msg);\n";
    } else {
      str << "(";
      param->print(str);
      str << ") msg);\n";
    }
    str << "  CkSendToFuture(ref, retMsg, src);\n";
    str << "}\n";
  } else if (isExclusive()) {
    if(container->getChareType() != SNODEGROUP) {
      cerr << "Only entry methods of a nodegroup can be exclusive." << endl;
      exit(1);
    }
    if(isConstructor()) {
      cerr << "Constructors cannot be exclusive methods." << endl;
      exit(1);
    }
    if(param==0) {
      cerr << "Entry methods must specify a message parameter: ";
      cerr << "use void if necessary\n";
      exit(1);
    }
    if(container->isTemplated())
      container->genSpec(str);
    str << " void ";
    container->genProxyName(str);
    if(container->isTemplated())
      container->genVars(str);
    str << "::_call_";
    genEpIdx(str);
    str << "(void* msg, ";
    str << container->getBaseName();
    if(container->isTemplated())
      container->genVars(str);
    str<< "* obj)\n";
    str << "{\n";
    if(isThreaded()) {
      str << "  CthAwaken(CthCreate((CthVoidFn)_callthr_";
      genEpIdx(str);
      str << ", new CkThrCallArg(msg, obj), ";
      str << getStackSize() << "));\n}\n";
      if(container->isTemplated())
        container->genSpec(str);
      str << " void ";
      container->genProxyName(str);
      if(container->isTemplated())
        container->genVars(str);
      str << "::_callthr_";
      genEpIdx(str);
      str << "(CkThrCallArg *arg)\n";
      str << "{\n";
      str << "  void *msg = arg->msg;\n";
      str << "  ";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << " *obj = (";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << " *) arg->obj;\n";
      str << "  delete arg;\n";
    }
    str << "  if(CmiTryLock(obj->__nodelock)) {\n";
    str << "    "; 
    container->genProxyName(str);
    if(container->isTemplated())
      container->genVars(str);
    str << " pobj(CkGetNodeGroupID());\n";
    str << "    pobj." << name << "(";
    if(param->isVoid()) {
      str << "CkMyNode());\n";
      str << "    CkFreeSysMsg(msg);\n";
    } else {
      str << "(";
      param->print(str);
      str << ") msg, CkMyNode());\n";
    }
    str << "    return;\n";
    str << "  }\n";
    str << "  obj->" << name << "(";
    if(param->isVoid()) {
      str << ");\n";
      str << "  CkFreeSysMsg(msg);\n";
    } else {
      str << "(";
      param->print(str);
      str << ") msg);\n";
    }
    str << "  CmiUnlock(obj->__nodelock);\n";
    str << "}\n";
  } else {
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
    if(isThreaded()) {
      str << "  CthAwaken(CthCreate((CthVoidFn)_callthr_";
      genEpIdx(str);
      str << ", new CkThrCallArg(msg, obj), ";
      str << getStackSize() << "));\n}\n";
      if(container->isTemplated())
        container->genSpec(str);
      str << " void ";
      container->genProxyName(str);
      if(container->isTemplated())
        container->genVars(str);
      str << "::_callthr_";
      genEpIdx(str);
      str << "(CkThrCallArg *arg)\n{\n";
      str << "  void *msg = arg->msg;\n";
      str << "  ";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << " *obj = (";
      str << container->getBaseName();
      if(container->isTemplated())
        container->genVars(str);
      str << " *) arg->obj;\n";
      str << "  delete arg;\n";
    }
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
        if(container->getChareType() == SARRAY)
          str << "  CkFreeMsg(msg);\n";
        else
          str << "  CkFreeSysMsg(msg);\n";
      } else {
        str << "(";
        param->print(str);
        str << ") msg);\n";
      }
    }
    str << "}\n";
  }
}

void Entry::genReg(XStr& str)
{
  str << "/* REG: "; print(str); str << " */\n";
  if(isConstructor() && container->isAbstract())
    return;
  if(container->getChareType()==SARRAY && !param && isConstructor()) {
    str << "  __idx_" << name << "_ArrayElementCreateMessage";
    str << " = CkRegisterEp(\"" << name << "\", "
        << "(CkCallFnPtr)_call_" << name << "_ArrayElementCreateMessage,";
    str << "CMessage_ArrayElementCreateMessage::__idx, ";
    str << "__idx);\n";
    str << "  __idx_" << name << "_ArrayElementMigrateMessage";
    str << " = CkRegisterEp(\"" << name << "\", "
        << "(CkCallFnPtr)_call_" << name << "_ArrayElementMigrateMessage,";
    str << "CMessage_ArrayElementMigrateMessage::__idx, ";
    str << "__idx);\n";
  } else {
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
}
