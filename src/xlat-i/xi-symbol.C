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
  if(next) { 
    str << ","; 
    next->print(str); 
  }
}


void 
Type::genProxyName(XStr &str) 
{
  cerr<< getBaseName() << " type has no proxy!!\n";
  exit(1);
}
    
void 
NamedType::print(XStr& str)
{
  str << name;
  if(tparams) {
    str << "<"<<tparams<<">";
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
  
  str << chareTypeName()<<" "<<type;
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
  str << " "<<name;
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

void 
Module::print(XStr& str)
{
  if(external)
    str << "extern ";
  str << "module "<<name;
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
  
  declstr << 
  "#ifndef _DECL_"<<name<<"_H_\n"
  "#define _DECL_"<<name<<"_H_\n"
  "#include \"charm++.h\"\n";
  clist->genDecls(declstr);
  declstr << "extern void _register"<<name<<"(void);\n";
  if(isMain()) {
    declstr << "extern \"C\" void CkRegisterMainModule(void);\n";
  }
  declstr << "#endif"<<endx;
  // defstr << "#ifndef _DEFS_"<<name<<"_H_"<<endx;
  // defstr << "#define _DEFS_"<<name<<"_H_"<<endx;
  clist->genDefs(defstr);
  defstr << 
  "#ifndef CK_TEMPLATES_ONLY\n"
  "void _register"<<name<<"(void)\n"
  "{\n"
  "  static int _done = 0; if(_done) return; _done = 1;\n";
  clist->genReg(defstr);
  defstr << "}\n";
  if(isMain()) {
    defstr << 
    "extern \"C\" void CkRegisterMainModule(void) {\n"
    "  _register"<<name<<"();\n"
    "  _REGISTER_DONE();\n"
    "}\n";
  }
  defstr << "#endif\n";
  // defstr << "#endif"<<endx;
  XStr topname, botname;
  topname<<name<<".decl.h";
  botname<<name<<".def.h";
  ofstream decl(topname.get_string()), def(botname.get_string());
  if(decl==0 || def==0) {
    cerr<<"Cannot open "<<topname.get_string()<<"or "
	<<botname.get_string()<<" for writing!!\n";
    exit(1);
  }
  decl<<declstr.get_string();
  def<<defstr.get_string();
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
    str << endx;
  }
  if(next)
    next->genDecls(str);
}

void
ConstructList::genDefs(XStr& str)
{
  if(construct) {
    construct->genDefs(str);
    str << endx;
  }
  if(next)
    next->genDefs(str);
}

void
ConstructList::genReg(XStr& str)
{
  if(construct) {
    construct->genReg(str);
    str << endx;
  }
  if(next)
    next->genReg(str);
}

static const char *CIChareStart = // prefix, name
"{\n"
"  public:\n"
"    static int __idx;\n"
;

static const char *CIChareEnd =
"};\n"
;

void
Chare::genRegisterMethodDecl(XStr& str)
{
  if(external || type->isTemplated())
    return;
  str << "    static void __register(const char *s, size_t size);\n";
}

void
Chare::genRegisterMethodDef(XStr& str)
{
  if(external || type->isTemplated())
    return;
  if(!templat) {
    str << "#ifndef CK_TEMPLATES_ONLY\n";
  } else {
    str << "#ifdef CK_TEMPLATES_ONLY\n";
  }
  genTSpec(str);
  str <<  
  "    void "<<proxyName()<<"::__register(const char *s, size_t size) {\n"
  "      __idx = CkRegisterChare(s, size);\n";
  // register all bases
  if(bases !=0)
    bases->genProxyNames(str, "  _REGISTER_BASE(__idx, ", "::__idx);\n", "");
  if(list)
    list->genReg(str);
  str << "    }\n";
  str << "#endif\n";
}

void
Chare::genDecls(XStr& str)
{
  if(type->isTemplated())
    return;
  str << "/* DECLS: "; print(str); str << " */\n";
  if(templat)
    templat->genSpec(str);
  str << "class "<<type<<";\n";
  if(templat)
    templat->genSpec(str);
  genSubDecls(str);
}


void
Chare::genSubDecls(XStr& str)
{
  XStr ptype;
  ptype<<proxyPrefix()<<type;
  
  str << "class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << ": public virtual _CK_CID";
  if(bases!=0) {
    str << ", ";
    bases->genProxyNames(str, "public ", "", ", ");
  }
  str << CIChareStart;
  genRegisterMethodDecl(str);
  if(isAbstract())
    str << "    "<<ptype<<"(void) {};\n";
  str << "    "<<ptype<<"(CkChareID __cid) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames(str, "", "(__cid)", ", ");
  }
  str << "{ ckSetChareId(__cid); }\n";
  str << "    CkChareID ckGetChareId(void) { return _ck_cid; }\n";
  str << "    void ckSetChareId(CkChareID __cid) {_CHECK_CID(__cid,__idx);_ck_cid=__cid;}\n";
  if(list)
    list->genDecls(str);
  str << CIChareEnd;
}

void
Group::genSubDecls(XStr& str)
{
  XStr ptype,ttype;
  ptype<<proxyPrefix()<<type;
  ttype<<type;
  if(templat) {
    templat->genVars(ttype);
  }
  
  str << "class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << ": ";
  if(isNodeGroup())
    str << " public virtual _CK_NGID";
  else
    str << " public virtual _CK_GID";
  if(bases!=0) {
    str << ", ";
    bases->genProxyNames(str, "public ", "", ", ");
  }
  str << CIChareStart;
  genRegisterMethodDecl(str);
  if(isAbstract())
    str << "    "<<ptype<<"(void) {};\n";
  str << "    "<<ptype<<"(CkGroupID _gid) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames(str, "", "(_gid)", ", ");
  }
  str << "{ _ck_gid = _gid; _setChare(0); }\n";
  str << "    "<<ptype<<"(CkChareID __cid) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames(str, "", "(__cid)", ", ");
  }
  str << "{ ckSetChareId(__cid); }\n";
  str << "    CkChareID ckGetChareId(void) { return _ck_cid; }\n";
  str << "    void ckSetChareId(CkChareID __cid){_CHECK_CID(__cid,__idx);_ck_cid=__cid;_setChare(1);}\n";
  str << "    CkGroupID ckGetGroupId(void) { return _ck_gid; }\n";
  str << "    void ckSetGroupId(CkGroupID _gid){_ck_gid=_gid;_setChare(0);}\n";
  str << "    "<<ttype<<"* ckLocalBranch(void) {\n";
  str << "      return ("<<ttype;
  if(isNodeGroup())
    str << " *) CkLocalNodeBranch(_ck_gid);\n";
  else
    str << " *) CkLocalBranch(_ck_gid);\n";
  str << "    }\n";
  str << "    static "<<ttype;
  str << "* ckLocalBranch(CkGroupID gID) {\n";
  str << "      return ("<<ttype;
  if(isNodeGroup())
    str << " *) CkLocalNodeBranch(gID);\n";
  else
    str << " *) CkLocalBranch(gID);\n";
  str << "    }\n";
  if(list)
    list->genDecls(str);
  str << CIChareEnd;
}

void
Array::genSubDecls(XStr& str)
{
  XStr ptype;
  ptype<<proxyPrefix()<<type;
  
  str << "class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << ": public virtual CkArrayID";
  if(bases!=0) {
    str << ", ";
    bases->genProxyNames(str, "public virtual ", "", ", ");
  }
  str << CIChareStart;
  genRegisterMethodDecl(str);
  if(isAbstract())
    str << "    "<<ptype<<"(void) {};\n";
  str << "    "<<ptype<<"(CkArrayID _aid) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames(str, "", "(_aid)", ", ");
  }
  str << "{ ckSetArrayId(_aid);}\n";
  str << "    "<<ptype<<"(const "<<ptype<<" &_arr) ";
  if(bases !=0) {
    str << ":";
    bases->genProxyNames2(str, "", "((const ", " &)_arr)", ", ");
  }
  str << "{ *this = _arr;}\n";
  str << 
  "    CkArrayID ckGetArrayId(void) { return CkArrayID(_ck_aid, _elem); }\n"
  "    void ckSetArrayId(CkArrayID _aid) { \n"
  "      _setAid(_aid._ck_aid); _elem = _aid._elem; \n"
  "    }\n"
  "    "<<ptype<<" operator [] (int idx) {\n"
  "      return "<<ptype<<"(CkArrayID(_ck_aid, idx));\n"
  "    }\n";
  if(list)
    list->genDecls(str);
  str << CIChareEnd;
}

void
Chare::genDefs(XStr& str)
{
  str << "/* DEFS: "; print(str); str << " */\n";
  if(!templat) {
    str << "#ifndef CK_TEMPLATES_ONLY\n";
    if(external) str << "extern ";
    str << "int "<<proxyName()<<"::__idx";
    if(!external) str << "=0";
    str << ";\n";
    str << "#endif\n";
  }
  if(!external && !type->isTemplated())
    genRegisterMethodDef(str);
  if(list)
    list->genDefs(str);
}

void
Chare::genReg(XStr& str)
{
  str << "/* REG: "; print(str); str << "*/\n";
  if(external || templat)
    return;
  str << "  "<<proxyPrefix();
  str << type<<"::__register(\""<<type<<"\", sizeof("<<type<<"));\n";
}

static const char *CIMsgClass =
"{\n"
"  public:\n"
"    static int __idx;\n"
"    void*operator new(size_t s){return CkAllocMsg(__idx,s,0);}\n"
"    void operator delete(void *p){CkFreeMsg(p);}\n"
"    void*operator new(size_t,void*p){return p;}\n"
"    void*operator new(size_t s, int p){return CkAllocMsg(__idx,s,p);}\n"
;

static const char *CIMsgClassAnsi =
"{\n"
"  public:\n"
"    static int __idx;\n"
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
  XStr ptype;
  ptype<<proxyPrefix()<<type;
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
  str << "class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  if(compilemode==ansi)
    str << CIMsgClassAnsi;
  else str << CIMsgClass;

  if(isVarsize()) {
    if(compilemode==ansi)
      str << CIAllocDeclAnsi;
    else str << CIAllocDecl;
  }
  if(!(external||type->isTemplated())) {
   // generate register function
    str << "    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {\n";
    str << "      __idx = CkRegisterMsg(s, pack, unpack, 0, size);\n";
    str << "    }\n";
  }
  str << "};\n";
}

void
Message::genDefs(XStr& str)
{
  XStr ptype;
  ptype<<proxyPrefix()<<type;
  
  str << "/* DEFS: "; print(str); str << " */\n";
  if(!templat) {
    str << "#ifndef CK_TEMPLATES_ONLY\n";
  } else {
    str << "#ifdef CK_TEMPLATES_ONLY\n";
  }
  if(!(external||type->isTemplated())) {
    // generate varsize new operator
    if(isVarsize()) {
      if(templat) {
        templat->genSpec(str);
        str << " ";
      }
      str << "void *"<<ptype;
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
        str << "void "<<ptype;
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
      str << "int "<<ptype<<"::__idx=0;\n";
    }
  }
  str << "#endif\n";
}

void
Message::genReg(XStr& str)
{
  str << "// REG: "; print(str);
  if(!templat && !external) {
    str << "  "<<proxyPrefix()<<type<<"::";
    str << "__register(\""<<type<<"\", sizeof("<<type<<"),";
    if(isPacked()||isVarsize()) {
      str << "(CkPackFnPtr) "<<type;
      if(templat)
        templat->genVars(str);
      str << "::pack, ";
      str << "(CkUnpackFnPtr) "<<type;
      if(templat)
        templat->genVars(str);
      str << "::unpack);\n";
    } else {
      str << "0, 0);\n";
    }
  }
}

void
Template::genVars(XStr& str)
{
  str << " < ";
  if(tspec)
    tspec->genShort(str);
  str << " > ";
}

void
Template::genSpec(XStr& str)
{
  str << "template ";
  str << "< ";
  if(tspec)
    tspec->genLong(str);
  str << " > ";
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
  str << " "<<name;
  if(val) {
    str << "="<<val;
  }
}

void TName::genShort(XStr& str)
{
  str << name;
}

void
Module::genDecls(XStr& str)
{
  if(external) {
    str << "#include \""<<name<<".decl.h\"\n";
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
    str << "      _register"<<name<<"();"<<endx;
  } else {
    clist->genDefs(str);
  }
}

void
Readonly::genDecls(XStr& str)
{
  str << "/* DECLS: "; print(str); str << " */\n";
  /*
  if(container) { // local static var
  } else { // global var
    str << "extern ";
    type->print(str);
    if(msg)
      str << "*";
    str << " "<<name;
    if(dims)
      dims->print(str);
    str << ";";
  }
  */
}

void
Readonly::genDefs(XStr& str)
{
  str << "/* DEFS: "; print(str); str << " */\n";
  if(container) { // local static var
  } else { // global var
    str << "extern ";
    type->print(str);
    if(msg)
      str << "*";
    str << " "<<name;
    if(dims)
      dims->print(str);
    str << ";";
  }
}

void
Readonly::genReg(XStr& str)
{
  if(external)
    return;
  if(msg) {
    if(dims) {
      cerr<<"line "<<line<<":Readonly Message cannot be an array!!\n";
      exit(1);
    }
    str << "  CkRegisterReadonlyMsg((void **) &";
    if(container) {
      str << container->baseName()<<"::";
    }
    str << name<<");\n";
  } else {
    str << "  CkRegisterReadonly(sizeof(";
    type->print(str);
    if(dims)
      dims->print(str);
    str << "), (void *) &";
    if(container) {
      str << container->baseName()<<"::";
    }
    str << name<<");\n";
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

void TypeList::genProxyNames2(XStr& str, const char *prefix, 
                             const char *middle, const char *suffix, 
                             const char *sep)
{
  if(type) {
    str << prefix;
    type->genProxyName(str);
    str << middle;
    type->genProxyName(str);
    str << suffix;
  }
  if(next) {
    str << sep;
    next->genProxyNames2(str, prefix, middle, suffix, sep);
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

///////////////////////////// ENTRY ////////////////////////////

// "parameterType *msg" or "void".
// Suitable for use as the only parameter
XStr Entry::paramType(void)
{
  XStr str;
  if(param) {
    str << param;
    if(!param->isVoid())  
    	str << "msg";
  }
  return str;
}

// "parameterType *msg," if there is a non-void parameter, 
// else empty.  Suitable for use with another parameter following.
XStr Entry::paramComma(void)
{
  XStr str;
  if(param&&!param->isVoid())  
  	str << param<<"msg, ";
  return str;
}

// Returns a dummy message declaration if the current
// parameter type is void.
XStr Entry::voidParamDecl(void)
{
  if((!param) || param->isVoid())
    return "      void *msg = CkAllocSysMsg();\n";
  else
    return "";
}

XStr Entry::epIdx(int include__idx_)
{
  XStr str;
  if(include__idx_) str << "__idx_";
  str << name;
  if(param) str << "_"<<param->getBaseName();
  return str;
}

void Entry::genEpIdxDecl(XStr& str)
{
  if(container->isArray() && !param && isConstructor()) {
    str << "    static int __idx_"<<name<<"_ArrayElementCreateMessage;\n";
    str << "    static int __idx_"<<name<<"_ArrayElementMigrateMessage;\n";
  } else {
    str << "    static int "<<epIdx()<<";\n";
  }
}

void Entry::genChareStaticConstructorDecl(XStr& str)
{
  if(container->isAbstract())
    return;
  str << "    static void ckNew("<<paramComma()<<"int onPE=CK_PE_ANY);\n";
  str << "    static void ckNew("<<paramComma()<<"CkChareID* pcid, int onPE=CK_PE_ANY);\n";
  str << "    "<<container->proxyName(0)<<"("<<paramComma()<<"int onPE=CK_PE_ANY);\n";
  // entry ptr declaration
  str << "    static int ckIdx_"<<name<<"("<<paramType()<<") ";
  str << "{ return "<<epIdx()<<"; }\n";
}

void Entry::genGroupStaticConstructorDecl(XStr& str)
{
  if(container->isAbstract())
    return;
  str << "    static CkGroupID ckNew("<<paramType()<<");\n";
  str << "    static CkGroupID ckNewSync("<<paramType()<<");\n";
  str << "    "<<container->proxyName(0)<<"("<<paramComma()<<"int retEP, CkChareID *cid);\n";
  str << "    "<<container->proxyName(0)<<"("<<paramType()<<");\n";
  // entry ptr declaration
  str << "    static int ckIdx_"<<name<<"("<<paramType()<<") ";
  str << "{ return "<<epIdx()<<"; }\n";
}

void Entry::genArrayStaticConstructorDecl(XStr& str)
{
  if(container->isAbstract())
    return;
  str << 
  "    static CkGroupID ckNew_GID(int numElements, CkGroupID mapID)\n"
  "    {\n"
  "        return Array1D::CreateArray(numElements,mapID,__idx,\n"
  "            ConstructorIndex("<<name<<", ArrayElementCreateMessage), \n"
  "            ConstructorIndex("<<name<<", ArrayElementMigrateMessage));\n"
  "    }\n"
  "    static CkArrayID ckNew(int numElements, CkGroupID mapID=_RRMapID)\n"
  "        {return CkArrayID(ckNew_GID(numElements,mapID),-1);}\n"
  "    ";
  
  str << container->proxyName(0)<<"(int numElements,CkGroupID mapID=_RRMapID)";
  if(container->isDerived()) {
    str << ": ";
    container->genProxyBases(str, "", "(numElements,mapID)", ", ");
  }
  str << "\n"
  "    {\n"
  "        _setAid(ckNew_GID(numElements,mapID)); \n"
  "        _elem=-1;\n"
  "    }\n";
  
  // entry ptr declaration
  char *paramStr=NULL;
  if(param!=NULL)
  {
  	XStr paramXStr;
  	param->print(paramXStr);
  	paramStr=paramXStr.get_string();
  }
  str.spew(
  "    static int ckIdx_\001(\002)\n"
  "        { return __idx_\001_ArrayElementCreateMessage; }\n" 
  "    static int ckIdx_\001(\003)\n"
  "        { return __idx_\001_ArrayElementMigrateMessage; }\n",
  name,
  (paramStr!=0)?paramStr:"ArrayElementCreateMessage*",
  (paramStr!=0)?paramStr:"ArrayElementMigrateMessage*");
}

void Entry::genChareDecl(XStr& str)
{
  if(isConstructor()) {
    genChareStaticConstructorDecl(str);
  } else {
    // entry method declaration
    str << "    "<<Virtual()<<retType<<" "<<name<<"("<<paramType()<<");\n";
    // entry method declaration with future
    if(isSync()) {
      str << "    "<<Virtual()<<" void "<<name<<"("<<paramComma()<<"CkFutureID*);\n";
    }
    // entry ptr declaration
    str << "    static int ckIdx_"<<name<<"("<<paramType()<<") ";
    str << "{ return "<<epIdx()<<"; }\n";
  }
}

void Entry::genGroupDecl(XStr& str)
{  
  //Selects between NodeGroup and Group
  char *node = (char *)(container->isNodeGroup()?"Node":"");

  if(isConstructor()) {
    genGroupStaticConstructorDecl(str);
  } else {
    // entry method broadcast declaration
    if(!isSync()) {
      str << "    "<<Virtual()<<retType<<" "<<name<<"("<<paramType()<<")";
      str << "{\n"<<voidParamDecl();
      str << "      if(_isChare())\n";
      str << "        CkSendMsg("<<epIdx()<<", msg, &_ck_cid);\n";
      str << "      else\n";
      str << "        CkBroadcastMsg"<<node<<"Branch("<<epIdx()<<", msg, _ck_gid);\n";
      str << "    }\n";
    }
    // entry method onPE declaration
    str << "    "<<Virtual()<<retType<<" "<<name<<"("<<paramComma()<<"int onPE)";
    str << " {\n"<<voidParamDecl();
    if(isSync()) {
      if(retType->isVoid()) {
        str << "    CkFreeSysMsg(";
      } else {
        str << "    return ("<<retType<<") ";
      }
      str << "(CkRemote"<<node<<"BranchCall("<<epIdx()<<", "<<paramComma()<<"_ck_gid, onPE));\n";
      str << "    }\n";
    } else {
      str << "      CkSendMsg"<<node<<"Branch("<<epIdx()<<", msg, onPE, _ck_gid);\n";
      str << "    }\n";
    }
    // entry method onPE declaration with future
    if(isSync()) {
      str << "    "<<Virtual()<<"void "<<name<<"("<<paramComma()<<"int onPE, CkFutureID *fut)";
      str << " {\n"<<voidParamDecl();
      str << "      *fut = ";
      str << "CkRemote"<<node<<"BranchCallAsync("<<epIdx()<<", msg, _ck_gid, onPE);\n";
      str << "    }\n";
    }
    // entry method forChare declaration with future
    if(isSync()) {
      str << "    "<<Virtual()<<"void "<<name<<"("<<paramComma()<<"CkFutureID *fut)";
      str << " {\n"<<voidParamDecl();
      str << "      *fut = CkRemoteCallAsync("<<epIdx()<<", msg, &_ck_cid);\n";
      str << "    }\n";
    }
    // entry ptr declaration
    str << "    static int ckIdx_"<<name<<"(";
    if(param==0) {
      cerr << "line "<<line<<":No entry parameter specified.\n";
      exit(1);
    }
    param->print(str);
    str << ") { return "<<epIdx();
    str << "; }\n";
  }
}

void Entry::genArrayDecl(XStr& str)
{
  if(isConstructor()) {
    genArrayStaticConstructorDecl(str);
  } else {
    char *msg=(char *)(param->isVoid()?"CkAllocMsg(0, sizeof(ArrayMessage),0)":"msg");
    // entry method broadcast declaration
    str << "    "<<Virtual()<<retType<<" "<<name<<"("<<paramComma()<<"int index=-2) {\n";
    str << "      if(index==-2) index=_elem;\n";
    str << "      if(index==(-1)) \n";
    str << "        _array->broadcast((ArrayMessage*) "<<msg<<", "<<epIdx()<<");\n";
    str << "      else _array->send((ArrayMessage*) "<<msg<<", index, "<<epIdx()<<");\n";
    str << "    }\n";
    // entry ptr declaration
    str << "    static int ckIdx_"<<name<<"("<<paramType()<<") { return "<<epIdx()<<"; }\n";
  }
}

void Entry::genDecls(XStr& str)
{
  str << "/* DECLS: "; print(str); str << " */\n";
  if(retType==0 && !isConstructor()) {
      cerr<<"line "<<line<<":Entry methods must specify a return type. ";
      cerr<<"use void if necessary\n";
      exit(1);
  }
  genEpIdxDecl(str);
  if(container->isGroup()) {
    genGroupDecl(str);
  } else if(container->isArray()) {
    genArrayDecl(str);
  } else { // chare or mainchare
    genChareDecl(str);
  }
  // call function declaration
  if(isConstructor() && container->isAbstract())
    return;
  if(container->isArray() && !param && isConstructor()) {
    str << "    static void _call_"<<name<<"_ArrayElementCreateMessage(void* msg, ";
    str << container->baseName()<<"* obj);\n";
    str << "    static void _call_"<<name<<"_ArrayElementMigrateMessage(void* msg, ";
    str << container->baseName()<<"* obj);\n";
    if(isThreaded()) {
      str << "    static void _callthr_"<<name;
      str << "_ArrayElementCreateMessage(CkThrCallArg *);\n";
      str << "    static void _callthr_"<<name;
      str << "_ArrayElementMigrateMessage(CkThrCallArg *);\n";
    }
  } else {
    str << "    static void _call_"<<epIdx(0)<<"(";
    if(param) {
      str << "void* msg, ";
    } else {
      if(!isConstructor()) {
        cerr <<"line "<<line<<" Only constructors allowed to have empty parameter list\n";
        exit(1);
      }
      str << "CkArgMsg* msg, ";
    }
    str << container->baseName();
    str << "* obj);\n";
    if(isThreaded()) {
      str << "    static void _callthr_"<<epIdx(0)<<"(CkThrCallArg *);\n";
    }
  }
}

void Entry::genEpIdxDef(XStr& str)
{
  if(container->isArray() && isConstructor() && !param) {
    
    container->genTSpec(str);
    str << "int "<<container->proxyName()<<"::__idx_"<<name<<"_ArrayElementCreateMessage=0;\n";
    
    container->genTSpec(str);
    str << "int "<<container->proxyName()<<"::__idx_"<<name<<"_ArrayElementMigrateMessage=0;\n";
  } else {
    container->genTSpec(str);
    str << "int "<<container->proxyName()<<"::"<<epIdx()<<"=0;\n";
  }
}

//Return a templated proxy declaration string for 
// this Member's container with the given return type, e.g.
// template<int N,class foo> void CProxy_bar<N,foo>
// Works with non-templated Chares as well.
XStr Member::makeDecl(const char *returnType)
{
  XStr str;
  
  if (container->isTemplated())
    {container->genTSpec(str);str << " ";}
  str << returnType<<" "<<container->proxyName();
  return str;
}

void Entry::genChareStaticConstructorDefs(XStr& str)
{
  if(container->isAbstract())
    return;
  str << makeDecl("void")<<"::ckNew("<<paramComma()<<"int onPE)\n";
  str << "{\n"<<voidParamDecl();
  str << "  CkCreateChare(__idx, "<<epIdx()<<", msg, 0, onPE);\n";
  str << "}\n";

  str << makeDecl("void")<<"::ckNew("<<paramComma()<<"CkChareID* pcid, int onPE)\n";
  str << "{\n"<<voidParamDecl();
  str << "  CkCreateChare(__idx, "<<epIdx()<<", msg, pcid, onPE);\n";
  str << "}\n";

  str << makeDecl(" ")<<"::"<<container->proxyName(0)<<"("<<paramComma()<<"int onPE)\n";
  str << "{\n"<<voidParamDecl();
  str << "  CkCreateChare(__idx, "<<epIdx()<<", msg, &_ck_cid, onPE);\n";
  str << "}\n";
}

void Entry::genGroupStaticConstructorDefs(XStr& str)
{
  if(container->isAbstract())
    return;
  
  //Selects between NodeGroup and Group
  char *node=(char *)(container->isNodeGroup()?"Node":"");
  
  str << makeDecl("CkGroupID")<<"::ckNew("<<paramType()<<")\n";
  str << "{\n"<<voidParamDecl();
  str << "  return CkCreate"<<node<<"Group(__idx, "<<epIdx()<<", msg, 0, 0);\n";
  str << "}\n";

  str << makeDecl("CkGroupID")<<"::ckNewSync("<<paramType()<<")\n";
  str << "{\n"<<voidParamDecl();
  str << "  return CkCreate"<<node<<"GroupSync(__idx, "<<epIdx()<<", msg);\n";
  str << "}\n";

  str << makeDecl(" ")<<"::"<<container->proxyName(0)<<
         "("<<paramComma()<<"int retEP, CkChareID *cid)\n";
  str << "{\n"<<voidParamDecl();
  str << "  _ck_gid = CkCreate"<<node<<"Group(__idx, "<<epIdx()<<", msg, retEP, cid);\n";
  str << "  _setChare(0);\n";
  str << "}\n";

  str << makeDecl(" ")<<"::"<<container->proxyName(0)<<"("<<paramType()<<")\n";
  str << "{\n"<<voidParamDecl();
  str << "  _ck_gid = CkCreate"<<node<<"Group(__idx, "<<epIdx()<<", msg, 0, 0);\n";
  str << "  _setChare(0);\n";
  str << "}\n";
}

void Entry::genChareDefs(XStr& str)
{
  if(isConstructor()) {
    genChareStaticConstructorDefs(str);
  } else {
    // entry method definition
    
    container->genTSpec(str);
    str << retType<<" "<<container->proxyName()<<"::"<<name<<"("<<paramType()<<")\n";
    str << "{\n";
    if(param->isVoid())
      str << "  void *msg = CkAllocSysMsg();\n";
    if(isSync()) {
      if(retType->isVoid()) {
        str << "  CkFreeSysMsg(CkRemoteCall("<<epIdx()<<", msg, &_ck_cid));\n";
      } else {
        str << "  return ("<<retType<<") CkRemoteCall("<<epIdx()<<", msg, &_ck_cid);\n";
      }
    } else {
      str << "  CkSendMsg("<<epIdx()<<", msg, &_ck_cid);\n";
    }
    str << "}\n";
    // entry method definition with future
    if(isSync()) {
      str << makeDecl(" void")<<"::"<<name<<"("<<paramComma()<<"CkFutureID *fut)\n";
      str << "{\n"<<voidParamDecl();
      str << "  *fut = CkRemoteCallAsync("<<epIdx()<<", msg, &_ck_cid);\n";
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

//This define is only used in Entry::genDefs.
// It ends the current procedure with a call to awaken another thread,
// and defines the thread function to handle that call.
XStr Entry::callThread(const XStr &procName,int prependEntryName)
{
  XStr str,procFull;
  procFull<<"_callthr_";
  if(prependEntryName) procFull<<name<<"_";
  procFull<<procName;
  
  str << "  CthAwaken(CthCreate((CthVoidFn)"<<procFull
   <<", new CkThrCallArg(msg,obj), "<<getStackSize()<<"));\n}\n";
  str << makeDecl("void")<<"::"<<procFull<<"(CkThrCallArg *arg)\n";
  str << "{\n";\
  str << "  void *msg = arg->msg;\n";
  str << "  "<<container->baseName()<<" *obj = ("<<container->baseName()<<" *) arg->obj;\n";
  str << "  delete arg;\n";
  return str;
}

void Entry::genDefs(XStr& str)
{
  XStr containerType=container->baseName();
  XStr cpType,cpComma;
  char *freeMsgVoid=(char*) "";
  if((!param)||(param->isVoid()))
  {
    if(container->isArray())
      freeMsgVoid=(char*) "  CkFreeMsg(msg);\n";
    else
      freeMsgVoid=(char*) "  CkFreeSysMsg(msg);\n";
  }
  if(param&&!param->isVoid())
  {//Add type casts for the message parameter
     cpType<<"("<<param<<")msg";
     cpComma<<"("<<param<<")msg,";
  }
  
  str << "/* DEFS: "; print(str); str << " */\n";
  if(!(container->isTemplated())) {
    str << "#ifndef CK_TEMPLATES_ONLY\n";
  } else {
    str << "#ifdef CK_TEMPLATES_ONLY\n";
  }
  genEpIdxDef(str);
  if(container->isGroup()){
    genGroupDefs(str);
  } else if(container->isArray()) {
  } else
    genChareDefs(str);
  // call function
  if(isConstructor() && container->isAbstract())
    return; // no call function for a constructor of an abstract chare
  if(container->isArray() && !param && isConstructor()) {
    str << makeDecl("void")<<"::_call_"<<name<<"_ArrayElementCreateMessage";
    str << "(void* msg,"<<containerType<<"* obj)\n";
    str << "{\n";
    if(isThreaded()) str << callThread("ArrayElementCreateMessage",1);
    str << "  new (obj) "<<containerType<<"((ArrayElementCreateMessage*)msg);\n}\n";
    
    str << makeDecl("void")<<"::_call_"<<name<<"_ArrayElementMigrateMessage";
    str << "(void* msg,"<<containerType<<"* obj)\n";
    str << "{\n";
    if(isThreaded()) str << callThread("ArrayElementMigrateMessage",1);
    str << "  new (obj) "<<containerType<<"((ArrayElementMigrateMessage*)msg);\n}\n";
  } 
  else if(isSync()) {
  //A synchronous method can return a value, and must finish before
  // the caller can proceed.
    if(isConstructor()) {
      cerr<<"line "<<line<<":Constructors cannot be sync methods."<<endl;
      exit(1);
    }
    str << makeDecl("void")<<"::_call_"<<epIdx(0)<<"(void* msg, "<<containerType<<"* obj)\n";
    str << "{\n";
    if(isThreaded()) str << callThread(epIdx(0));
    str << "  int ref = CkGetRefNum(msg), src = CkGetSrcPe(msg);\n";
    str << "  void *retMsg=";
    if(retType->isVoid()) {
      str << "CkAllocSysMsg();\n  ";
    } else {
      str << "(void *) ";
    }
    str << "obj->"<<name<<"("<<cpType<<");\n"<<freeMsgVoid;
    str << "  CkSendToFuture(ref, retMsg, src);\n";
    str << "}\n";
  } else if(isExclusive()) {
  //An exclusive method 
    if(!container->isNodeGroup()) {
      cerr<<"line "<<line<<":Only entry methods of a nodegroup can be exclusive."<<endl;
      exit(1);
    }
    if(isConstructor()) {
      cerr<<"line "<<line<<":Constructors cannot be exclusive methods."<<endl;
      exit(1);
    }
    if(param==0) {
      cerr<<"line "<<line<<":Entry methods must specify a message parameter. ";
      cerr<<"use void if necessary\n";
      exit(1);
    }
    str << makeDecl("void")<<"::_call_"<<epIdx(0)<<"(void* msg, "<<containerType<<"* obj)\n";
    str << "{\n";
    if(isThreaded()) str << callThread(epIdx(0));
    str << "  if(CmiTryLock(obj->__nodelock)) {\n";
    str << "    "<<container->proxyName()<<" pobj(CkGetNodeGroupID());\n";
    str << "    pobj."<<name<<"("<<cpComma<<"CkMyNode());\n"<<freeMsgVoid;
    str << "    return;\n";
    str << "  }\n";
    str << "  obj->"<<name<<"("<<cpType<<");\n"<<freeMsgVoid;
    str << "  CmiUnlock(obj->__nodelock);\n";
    str << "}\n";
  } else {//Not sync, exclusive, or an array constructor-- just a regular method
    str << makeDecl("void")<<"::_call_"<<epIdx(0)
      <<"("<<((param)?"void":"CkArgMsg")<<"* msg, "<<containerType<<"* obj)\n";
    str << "{\n";
    if(isThreaded()) str << callThread(epIdx(0));
    if(isConstructor()) {
      str << "  new (obj) "<<containerType;
      if(param) {
        if(!param->isVoid()) {
          str << "(("<<param<<")msg);\n";
        } else {
          str << "();\n";
          str << "  CkFreeSysMsg(msg);\n";
        }
      } else {
        str << "((CkArgMsg*)msg);\n";
      }
    } else {//Not a constructor
      str << "  obj->"<<name<<"("<<cpType<<");\n"<<freeMsgVoid;
    }
    str << "}\n";
  }
  str << "#endif\n";
}

void Entry::genReg(XStr& str)
{
  str << "    // REG: "<<*this;
  if(isConstructor() && container->isAbstract())
    return;
  if(container->isArray() && !param && isConstructor()) {
    str << "      __idx_"<<name<<"_ArrayElementCreateMessage";
    str << " = CkRegisterEp(\""<<name<<"\", "
     <<"(CkCallFnPtr)_call_"<<name<<"_ArrayElementCreateMessage,";
    str << "CMessage_ArrayElementCreateMessage::__idx, __idx);\n";
    
    str << "      __idx_"<<name<<"_ArrayElementMigrateMessage";
    str << " = CkRegisterEp(\""<<name<<"\", "
     <<"(CkCallFnPtr)_call_"<<name<<"_ArrayElementMigrateMessage,";
    str << "CMessage_ArrayElementMigrateMessage::__idx, __idx);\n";
  } else {
    str << "      "<<epIdx()<<" = CkRegisterEp(\""<<name 
 <<"\", (CkCallFnPtr)_call_"<<epIdx(0)<<", ";
    if(param && !param->isVoid()) {
      param->genMsgProxyName(str);
      str << "::__idx, ";
    } else {
      str << "0, ";
    }
    str << "__idx);\n";
    if(container->isMainChare() && isConstructor()) {
      str << "      CkRegisterMainChare(__idx, "<<epIdx()<<");\n";
    }
  }
}













