/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <iostream.h>
#include <fstream.h>
//  #include <string>
#include <stdlib.h>
#include "xi-symbol.h"
#include <ctype.h> // for tolower()

CompileMode compilemode;
MessageList message_list;
int fortranMode;

// Make the name lower case
char* fortranify(const char *s)
{
  int i, len = strlen(s);
  char *retVal;
  retVal = new char[len+1];
  for(i = 0; i < len; i++)
    retVal[i] = tolower(s[i]);
  retVal[len] = 0;

  return retVal;
}

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
  printVars(str);
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
    if (fortranMode) defstr << "extern void _registerf90main(void);\n";
    defstr << "extern \"C\" void CkRegisterMainModule(void) {\n";
    if (fortranMode) { // For Fortran90
      defstr << "  // FORTRAN\n";
      defstr << "  _registerf90main();\n";
    }
    defstr << 
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
  if(!decl || !def) {
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

Chare::Chare(int ln, NamedType *t, TypeList *b, MemberList *l, int mig)
	 : type(t), list(l), bases(b), migratable(mig) 
{
	line = ln;
	setTemplate(0); 
	abstract=0;
	if (list)
	{
		list->setChare(this);
		if (list->isPure()) abstract=1;
		else /*not at abstract class--*/
      		//Add migration constructor to MemberList
		  if(!t->isTemplated() && migratable) {
			Entry *e=new Entry(ln,SMIGRATE,NULL,
			(char *)type->getBaseName(),
			new PtrType(new NamedType("CkMigrateMessage")));
			e->setChare(this);
			list=new MemberList(e,list);
		  }
	}
}


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
  if(migratable) {
    str << "      CkRegisterMigCtor(__idx, " << 
	            "__idx_"<<type->getBaseName()<<"_CkMigrateMessage);\n";
  }
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


//Array Constructor
Array::Array(int ln, NamedType *index,
	NamedType *t, TypeList *b, MemberList *l)  
    : Chare(ln,t,b,l,0x01) 
{
	index->print(indexSuffix);
	if (indexSuffix!=(const char*)"none")
		indexType<<"CkArrayIndex"<<indexSuffix;
	else indexType<<"CkArrayIndex";
//Add ArrayElement to the list of bases (if we're not ArrayElement ourselves)
	if((!bases)&&(0!=strcmp(type->getBaseName(),"ArrayElement")))
		bases = new TypeList(new NamedType("ArrayElement"), bases);
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
  str << " : ";
  bool fakeBase=false;
  if(bases==0) {
  	fakeBase=true;//<- "CkArrayBase" is a base class, but is used only for implementation.
  	bases=new TypeList(new NamedType("CkArrayBase"),NULL);
  }
  bases->genProxyNames(str, "public ", "", ", ");
  
  str << CIChareStart;
  genRegisterMethodDecl(str);
  
  //Create an empty array
  str <<"    static CkGroupID buildArrayGroup(CkGroupID mapID=_RRMapID,int numInitial=0)\n"
	"    {\n"
	"        return CkArray::CreateArray(mapID,numInitial);\n"
	"    }\n";

  if (is1D()) //Create an array with some 1D elements
  {
    str <<"    static CkGroupID buildArrayGroup(int ctorIndex,int numInitial,CkGroupID mapID,\n"
	"            CkArrayMessage *msg)\n"
	"    {\n"
	"        CkGroupID id=buildArrayGroup(mapID,numInitial);\n"
	"        CProxy_CkArrayBase(id).base_insert1D(ctorIndex,numInitial,msg);\n"
	"        return id;\n"
	"    }\n";
  }
  
  //This constructor is used for array indexing
  str << "  protected:\n"
         "    "<<ptype<<"(const CkArrayID &aid,const "<<indexType<<" &idx)\n"
         "        :";bases->genProxyNames(str, "", "(aid,idx)", ", ");str<<" {}\n";
//         "      :CProxy_CkArrayBase(aid,idx) {}\n";
  str << "  public:\n"
         "    "<<ptype<<"(const CkArrayID &aid) \n"
         "        :";bases->genProxyNames(str, "", "(aid)", ", ");str<<" {}\n";
//         "      :CProxy_CkArrayBase(aid) {}\n";
  str << "    "<<ptype<<"(void) {}\n";//An empty constructor
  
  str<< //Build a simple, empty array
  "    static CkArrayID ckNew(void) {return CkArrayID(buildArrayGroup());}\n"
  "    static CkArrayID ckNew_mapped(CkGroupID mapID) {return CkArrayID(buildArrayGroup(mapID));}\n";
  
  if (indexSuffix!=(const char*)"none")
  {
    str <<
    "//Generalized array indexing: (these KEEP the index you pass in!)\n"
    "    "<<ptype<<" operator [] (const "<<indexType<<" &idx) const\n"
    "        {return "<<ptype<<"(_aid, idx);}\n"
    "    "<<ptype<<" operator() (const "<<indexType<<" &idx) const\n"
    "        {return "<<ptype<<"(_aid, idx);}\n";
  }
  
  //Add specialized indexing for these common types
  if (indexSuffix==(const char*)"1D")
  {
    str << 
    "    "<<ptype<<" operator [] (int idx) const \n"
    "        {return "<<ptype<<"(_aid, CkArrayIndex1D(idx));}\n"
    "    "<<ptype<<" operator () (int idx) const \n"
    "        {return "<<ptype<<"(_aid, CkArrayIndex1D(idx));}\n";
  } else if (indexSuffix==(const char*)"2D") {
    str << 
    "    "<<ptype<<" operator () (int i0,int i1) const \n"
    "        {return "<<ptype<<"(_aid, CkArrayIndex2D(i0,i1));}\n";
  } else if (indexSuffix==(const char*)"3D") {
    str << 
    "    "<<ptype<<" operator () (int i0,int i1,int i2) const \n"
    "        {return "<<ptype<<"(_aid, CkArrayIndex3D(i0,i1,i2));}\n";
  }
  if (fakeBase) bases=NULL;//<- return bases to original value

  if(list)
    list->genDecls(str);
  str << CIChareEnd;
}

void
Chare::genDefs(XStr& str)
{
  str << "/* DEFS: "; print(str); str << " */\n";
  if (fortranMode) { // For Fortran90
    if (!isArray()) { // Currently, only arrays are supported
      cerr << (char *)baseName() << ": only chare arrays are currently supported\n";
      exit(1);
    }
    // We have to generate the chare array itself
    str << "/* FORTRAN */\n";
    str << "extern \"C\" void " << fortranify(baseName()) << "_allocate_(char **, void *);\n";
    str << "\n";
    str << "class " << baseName() << " : public ArrayElement1D\n";
    str << "{\n";
    str << "public:\n";
    str << "  char user_data[64];\n";
    str << "public:\n";
    str << "  " << baseName() << "()\n";
    str << "  {\n";
//    str << "    CkPrintf(\"" << baseName() << " %d created\\n\",thisIndex);\n";
    str << "    CkArrayID *aid = &thisArrayID;\n";
    str << "    " << fortranify(baseName()) << "_allocate_((char **)&user_data, &aid);\n";
    str << "  }\n";
    str << "\n";
    str << "  " << baseName() << "(CkMigrateMessage *m)\n";
    str << "  {\n";
    str << "    CkPrintf(\"" << baseName() << " %d migrating\\n\",thisIndex);\n";
    str << "  }\n";
    str << "\n";
    str << "};\n";
    str << "\n";
    str << "extern \"C\" void " << fortranify(baseName()) << "_cknew_(int *numElem, long *aindex)\n";
    str << "{\n";
    str << "    CkArrayID *aid = new CkArrayID;\n";
    str << "    *aid = CProxy_" << baseName() << "::ckNew(*numElem); \n";
    str << "    *aindex = (long)aid;\n";
    str << "}\n";

  }
  if(!type->isTemplated()) {
    if(!templat) {
      str << "#ifndef CK_TEMPLATES_ONLY\n";
    } else {
      str << "#ifdef CK_TEMPLATES_ONLY\n";
    }
    if(external) str << "extern ";
    genTSpec(str);
    str << "int "<<proxyName()<<"::__idx";
    if(!external) str << "=0";
    str << ";\n";
    str << "#endif\n";
  }
  if(!external && !type->isTemplated())
    genRegisterMethodDef(str);
  if(list) 
  {//Add definitions for all entry points
    
    if(isTemplated())
      str << "#ifdef CK_TEMPLATES_ONLY\n";
    else
      str << "#ifndef CK_TEMPLATES_ONLY\n";
    list->genDefs(str);
    str << "#endif /*CK_TEMPLATES_ONLY*/\n";
  }
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
"    void* operator new(size_t, const int pb=0);\n"
"    void operator delete(void *p){CkFreeMsg(p);}\n"
"    void* operator new(size_t, void*p) {return p;}\n"
"    void* operator new(size_t, int*, const int pb=0);\n"
"    static void*alloc(int,size_t,int*,int);\n"
;

static const char *CIMsgClassAnsi =
"{\n"
"  public:\n"
"    static int __idx;\n"
"    void* operator new(size_t,void*p) { return p; }\n"
"    void operator delete(void*p,void*){CkFreeMsg(p);}\n"
"    void* operator new(size_t,const int pb=0);\n"
"    void operator delete(void*p,const int pb=0){\n"
"      CkFreeMsg(p);}\n"
"    void* operator new(size_t, int*, const int pb=0);\n"
"    void operator delete(void*p,int*,const int pb=0){\n"
"      CkFreeMsg(p);}\n"
"    static void* alloc(int,size_t,int*,int);\n"
;

void
Message::genAllocDecl(XStr &str)
{
  int i, num;
  XStr mtype;
  mtype << type;
  if(templat) templat->genVars(mtype);
  if(compilemode==ansi)
    str << CIMsgClassAnsi;
  else str << CIMsgClass;
  str << "    static void *pack(" << mtype << " *p);\n";
  str << "    static " << mtype << "* unpack(void* p);\n";
  num = numVars();
  if(num>0) {
    str << "    void *operator new(size_t,";
    for(i=0;i<num;i++)
      str << "int, ";
    str << "const int);\n";
    if(compilemode==ansi) {
      str << "    void operator delete(void *p,";
      for(i=0;i<num;i++)
        str << "int, ";
      str << "const int){CkFreeMsg(p)}\n";
    }
  }
}

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
    str << ";\n";
    return;
  }
//OSL 4/11/2000-- make *all* messages inherit from CkArrayMessage.
//  This means users don't have to remember to type ",public ArrayMessage";
// and eventually CkArrayMessage will be integrated into the envelope anyway.
  str << ":public CkArrayMessage";

  genAllocDecl(str);

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
  int i, num = numVars();
  MsgVarList *ml;
  MsgVar *mv;
  XStr ptype, mtype, tspec;
  ptype<<proxyPrefix()<<type;
  if(templat) templat->genVars(ptype);
  mtype << type;
  if(templat) templat->genVars(mtype);
  if(templat) { templat->genSpec(tspec); tspec << " "; }
  
  str << "/* DEFS: "; print(str); str << " */\n";
  if(!templat) {
    str << "#ifndef CK_TEMPLATES_ONLY\n";
  } else {
    str << "#ifdef CK_TEMPLATES_ONLY\n";
  }
  if(!(external||type->isTemplated())) {
    // new (size_t, priobits)
    str << tspec << "void *" << ptype << "::operator new(size_t s,";
    str << "const int pb){\n";
    str << "  return " << mtype << "::alloc(__idx, s, 0, pb);\n}\n";
    // new (size_t, int*, priobits)
    str << tspec << "void *" << ptype << "::operator new(size_t s, int* sz,";
    str << "const int pb){\n";
    str << "  return " << mtype << "::alloc(__idx, s, sz, pb);\n}\n";
    // new (size_t, int, int, ..., int, priobits)
    if(num>0) {
      str << tspec << "void *"<< ptype << "::operator new(size_t s, ";
      for(i=0;i<num;i++)
        str << "int sz" << i << ", ";
      str << "const int p) {\n";
      str << "  int i; int sizes[" << num << "];\n";
      for(i=0;i<num;i++)
        str << "  sizes[" << i << "] = sz" << i << ";\n";
      str << "  return " << mtype << "::alloc(__idx, s, sizes, p);\n";
      str << "}\n";
    }
    // alloc(int, size_t, int*, priobits)
    str << tspec << "void* " << ptype;
    str << "::alloc(int msgnum, size_t sz, int *sizes, int pb) {\n";
    str << "  int offsets[" << num+1 << "];\n";
    str << "  offsets[0] = ALIGN8(sz);\n";
    for(i=0, ml=mvlist; i<num; i++, ml=ml->next) {
      mv = ml->msg_var;
      str << "  if(sizes==0)\n";
      str << "    offsets[" << i+1 << "] = offsets[0];\n";
      str << "  else\n";
      str << "    offsets[" << i+1 << "] = offsets[" << i << "] + ";
      str << "ALIGN8(sizeof(" << mv->type << ")*sizes[" << i << "]);\n";
    }
    str << "  " << mtype << " *newmsg = (" << mtype << " *) ";
    str << "CkAllocMsg(msgnum, offsets[" << num << "], pb);\n";
    for(i=0, ml=mvlist; i<num; i++,ml=ml->next) {
      mv = ml->msg_var;
      str << "  newmsg->" << mv->name << " = (" << mv->type << " *) ";
      str << "((char *)newmsg + offsets[" << i << "]);\n";
    }
    str << "  return (void *) newmsg;\n}\n";
    // pack
    str << tspec << "void* " << ptype << "::pack(" << mtype << " *msg) {\n";
    for(i=0, ml=mvlist; i<num; i++, ml=ml->next) {
      mv = ml->msg_var;
      str << "  msg->" << mv->name << " = (" <<mv->type << " *) ";
      str << "((char *)msg->" << mv->name << " - (char *)msg);\n";
    }
    str << "  return (void *) msg;\n}\n";
    // unpack
    str << tspec << mtype << "* " << ptype << "::unpack(void* buf) {\n";
    str << "  " << mtype << " *msg = (" << mtype << " *) buf;\n";
    for(i=0, ml=mvlist; i<num; i++, ml=ml->next) {
      mv = ml->msg_var;
      str << "  msg->" << mv->name << " = (" <<mv->type << " *) ";
      str << "((size_t)msg->" << mv->name << " + (char *)msg);\n";
    }
    str << "  return msg;\n}\n";
  }
  if(!templat) {
    if(!external && !type->isTemplated()) {
      str << "int "<< ptype <<"::__idx=0;\n";

      // Define the Marshalling message for Fortran
      if (fortranMode)
      {
        str << "/* FORTRAN */\n";
        str << "class " << type 
            << " : public CMessage_" << type << "\n";
        str << "{\npublic:\n";
        contents->genUnmarshalList2(str);
        str << "};\n";
      }
    }
  } else {
    str << tspec << "int "<< ptype <<"::__idx=0;\n";
  }
  str << "#endif\n";
}

void
Message::genReg(XStr& str)
{
  str << "/* REG: "; print(str); str << "*/\n";
  if(!templat && !external) {
    XStr ptype, mtype, tspec;
    ptype<<proxyPrefix()<<type;
    str << ptype << "::__register(\"" << type << "\", sizeof(" << type <<"),";
    str << "(CkPackFnPtr) " << type << "::pack,";
    str << "(CkUnpackFnPtr) " << type << "::unpack);\n";
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
  if(!container && !strchr(name, ':')) {
    str << "extern ";
    type->print(str);
    if(msg)
      str << "*";
    str << " "<<name;
    if(dims)
      dims->print(str);
    str << ";\n";
  }
  if (fortranMode) {
      str << "extern \"C\" void set_"
          << fortranify(name)
          << "_(int *n) { " << name << " = *n; }\n";
      str << "extern \"C\" void get_"
          << fortranify(name)
          << "_(int *n) { *n = " << name << "; }\n";
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

void TypeList::genUnmarshalList0(XStr& str, int depth)
{
  if(type) {
//    type->print(str);
    str << type->getBaseName();
    str << "*";
  }
  if(next) {
    str << ", ";
    next->genUnmarshalList0(str, depth+1);
  }
}

void TypeList::genUnmarshalList1(XStr& str, const char* message_name, int depth)
{
  if(type) {
    if (type->typeName() != arrayType)
      str << "&((" << message_name << "*)msg)->p" << depth;
    else
      str << "((" << message_name << "*)msg)->p" << depth;
  }
  if(next) {
    str << ", ";
    next->genUnmarshalList1(str, message_name, depth+1);
  }
}

void TypeList::genUnmarshalList2(XStr& str, int depth)
{
  if(type) {
//    str << "  " << type->getBaseName() << " p" << depth << ";\n";
    char p[10];
    sprintf(p, "p%d", depth);
    type->printVar(str, p);
    str<<";\n";
  }
  if(next) {
    next->genUnmarshalList2(str, depth+1);
  }
}

void TypeList::genUnmarshalList3(XStr& str, int depth)
{
  if(type) {
    str << type->getBaseName() << " *p" << depth;
  }
  if(next) {
    str << ", ";
    next->genUnmarshalList3(str, depth+1);
  }
}

void TypeList::genUnmarshalList4(XStr& str, int depth)
{
  if(type) {
    if (type->typeName() != arrayType) 
      str << "  msg->p" << depth << " = *p" << depth << ";\n";
    else {
      ArrayType *atype = (ArrayType *)type;
      str << "  memcpy(msg->p" << depth << ", p" << depth << ", ";
      atype->printDim(str);
      str << "*sizeof(" << atype->getBaseName() << "));\n";
    }
  }
  if(next) {
    next->genUnmarshalList4(str, depth+1);
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


Entry::Entry(int l, int a, EnType *r, char *n, EnType *p, Value *sz) :
      attribs(a), retType(r), name(n), param(p), stacksize(sz)
{ 
  line=l; setChare(0); 
  if(!isVirtual() && isPure()) {
    cerr << "Line "<<line<<": Non-virtual methods cannot be pure virtual!!\n";
    abort();
  }
  if(!isThreaded() && stacksize) {
    cerr << "Line "<<line<<": Non-Threaded methods cannot have stacksize spec.!!\n";
    abort();
  }
  if(retType && !isSync() && !retType->isVoid()) {
    cerr << "Line "<<line<<": Async methods cannot have non-void return type!!\n";
    abort();
  }
}


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

/*************************** Chare Entry Points ******************************/

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
  }
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

void Entry::genChareStaticConstructorDecl(XStr& str)
{
  if(container->isAbstract()) return;

  str << "    static void ckNew("<<paramComma()<<"int onPE=CK_PE_ANY);\n";
  str << "    static void ckNew("<<paramComma()<<"CkChareID* pcid, int onPE=CK_PE_ANY);\n";
  str << "    "<<container->proxyName(0)<<"("<<paramComma()<<"int onPE=CK_PE_ANY);\n";
    
}

void Entry::genChareStaticConstructorDefs(XStr& str)
{
  if(container->isAbstract()) return;
  
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

/***************************** Array Entry Points **************************/

void Entry::genArrayDecl(XStr& str)
{
  if(isConstructor()) {
    genArrayStaticConstructorDecl(str);
  } else {
    // entry method broadcast declaration
    str << "    "<<Virtual()<<retType<<" "<<name<<"("<<paramType()<<") const;\n";
  }
}

void Entry::genArrayDefs(XStr& str)
{
  if (isConstructor())
    genArrayStaticConstructorDefs(str);
  else
  {//Define array entry method
    container->genTSpec(str);
    str<<retType<<" "<<container->proxyName()<<"::"<<name<<"("<<paramType()<<") const\n";
    str<< "{\n";
    if (!param||param->isVoid())
      str << "      CkArrayMessage *msg=(CkArrayMessage*)CkAllocMsg(0, sizeof(CkArrayMessage),0);\n";
    str << "      if(_idx.nInts==-1) \n";
    str << "        base_broadcast(msg, "<<epIdx()<<");\n";
    str << "      else\n";
    str << "        base_send(msg, "<<epIdx()<<");\n";
    str << "}\n";
  }
}

void Entry::genArrayStaticConstructorDecl(XStr& str)
{
  if(container->isAbstract()) return;
  
  if ( ((Array *)container) ->is1D())
      str<< //With numInitial
      "    static CkArrayID ckNew("<<paramComma()<<"int numInitial,CkGroupID mapID=_RRMapID);\n";
    
  str<<
    "    void insert("<<paramComma()<<"int onPE=-1);";
}

void Entry::genArrayStaticConstructorDefs(XStr& str)
{
    if(container->isAbstract()) return;
    
    const char *callMsg;//"msg" or "NULL"
    if (!param||param->isVoid()) callMsg="NULL"; else callMsg="msg";
    
    //Add user-callable array constructors--
    if (((Array *)container)->is1D())
    //1D element constructors can take a number of initial elements
       str<< //With numInitial
       makeDecl("CkArrayID")<<"::ckNew("<<paramComma()<<"int numElements,CkGroupID mapID)\n"
       "{ return CkArrayID(buildArrayGroup("<<epIdx()<<",numElements,mapID,"<<callMsg<<"));}\n";
    str<<
    makeDecl("void")<<"::insert("<<paramComma()<<"int onPE)\n"
    "{ base_insert("<<epIdx()<<",onPE,"<<callMsg<<");}\n";
}


/******************************** Group Entry Points *********************************/

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
    // entry method on multi PEs declaration
    if(!isSync() && !container->isNodeGroup()) {
      str << "    "<<Virtual()<<retType<<" "<<name<<"("<<paramComma();
      str << "int npes, int *pes)";
      str << " {\n"<<voidParamDecl();
      str << "      CkSendMsg"<<node<<"BranchMulti(";
      str <<epIdx()<<", msg, npes, pes, _ck_gid);\n";
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
  }
}
void Entry::genGroupDefs(XStr& str)
{
  if(isConstructor()) {
    genGroupStaticConstructorDefs(str);
  } else {
  }
}

void Entry::genGroupStaticConstructorDecl(XStr& str)
{
  if(container->isAbstract()) return;

  str << "    static CkGroupID ckNew("<<paramType()<<");\n";
  str << "    static CkGroupID ckNewSync("<<paramType()<<");\n";
  str << "    "<<container->proxyName(0)<<"("<<paramComma()<<"int retEP, CkChareID *cid);\n";
  str << "    "<<container->proxyName(0)<<"("<<paramType()<<");\n";
}

void Entry::genGroupStaticConstructorDefs(XStr& str)
{
  if(container->isAbstract()) return;
  
  //Selects between NodeGroup and Group
  char *node = (char *)(container->isNodeGroup()?"Node":"");
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

/******************* Shared Entry Point Code **************************/

void Entry::genDecls(XStr& str)
{
  if(container->isArray()&&param==NULL) {param=new BuiltinType("void");}
  if(isConstructor() && retType && !retType->isVoid()) {
    cerr << "Line "<<line<<": Constructors cannot return a value!!\n";
    abort();
  }
  
  str << "/* DECLS: "; print(str); str << " */\n";
  if(retType==0 && !isConstructor()) {
      cerr<<"line "<<line<<":Entry methods must specify a return type. ";
      cerr<<"use void if necessary\n";
      exit(1);
  }  
  
  if (attribs&SMIGRATE) 
    {} //User cannot call the migration constructor
  else if(container->isGroup()) {
      genGroupDecl(str);
  } else if(container->isArray()) {
      genArrayDecl(str);
  } else { // chare or mainchare
      genChareDecl(str);
  }
  
  // Entry point index storage
  str << "    static int  "<<epIdx()<<";\n";
  
  // ckIdx, so user can find the entry point number
  str << "    static int  ckIdx_"<<name<<"("<<paramType()<<") { return "<<epIdx()<<"; }\n"; 

  // call function declaration
  if(isConstructor() && container->isAbstract())
    return;
  else {
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
    str << container->baseName()<< "* obj);\n";
    if(isThreaded()) {
      str << "    static void _callthr_"<<epIdx(0)<<"(CkThrCallArg *);\n";
    }
  }
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
  
  //Define storage for entry point number
  container->genTSpec(str);
  str << "int "<<container->proxyName()<<"::"<<epIdx()<<"=0;\n";

  if (attribs&SMIGRATE) 
    {} //User cannot call the migration constructor
  else if(container->isGroup()){
    genGroupDefs(str);
  } else if(container->isArray()) {
    genArrayDefs(str);
  } else
    genChareDefs(str);
  
  // call function
  if(isConstructor() && container->isAbstract())
    return; // no call function for a constructor of an abstract chare
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

    if (!isConstructor() && fortranMode) { // Fortran90
      const char* msg_name = param->getBaseName();
      TypeList *msg_contents = (message_list.find(msg_name))->getContents();

      str << "/* FORTRAN SECTION */\n";

      // Declare the Fortran Entry Function
      // This is called from C++
      str << "extern \"C\" void " << fortranify(name) << "_(char **, int*, ";
      msg_contents->genUnmarshalList0(str);
      str << ");\n";

/*
      // Define the Marshalling message
      str << "class " << msg_name 
          << " : public CMessage_" << msg_name << "\n";
      str << "{\npublic:\n";
      msg_contents->genUnmarshalList2(str);
      str << "};\n";
*/

      // Define the Fortran interface function
      // This is called from Fortran to send the message to a chare.
      str << "extern \"C\" void "
        //<< container->proxyName() << "_" 
          << fortranify("SendTo_")
	  << fortranify(container->baseName())
          << "_" << fortranify(name)
          << "_(long* aindex, int *index, ";
      msg_contents->genUnmarshalList3(str);
      str << ")\n";
      str << "{\n";
      str << "  CkArrayID *aid = (CkArrayID *)*aindex;\n";
      str << "\n";
      str << "  // marshalling\n";
      str << "  " << container->proxyName() << " h(*aid);\n";
      str << "  " << msg_name <<" *msg = new " << msg_name <<";\n";
      //      str << "    msg->data = *data;\n";
      msg_contents->genUnmarshalList4(str);
      str << "  h[*index]." << name << "(msg);\n";
      str << "}\n";
    }

    str << makeDecl("void")<<"::_call_"<<epIdx(0)
      <<"("<<((param)?"void":"CkArgMsg")<<"* msg, "<<containerType<<"* obj)\n";
    str << "{\n";
    if(isThreaded()) str << callThread(epIdx(0));
    if(isConstructor()) {
      str << "  new (obj) "<<containerType;
      if(param) {
        if(!param->isVoid()) {
          str << "(("<<param<<")msg);\n";
        } else {//Constructor takes no msg-- must blow away old parameter
          str << "();\n";
          if (container->isArray()) str << "  CkFreeMsg(msg);\n";
          else str << "  CkFreeSysMsg(msg);\n";
        }
      } else {//For main chare--
        str << "((CkArgMsg*)msg);\n";
      }
    } else {//Not a constructor
      if (fortranMode) { // Fortran90
        str << "  int index = obj->getIndex();\n";
        str << "  " << fortranify(name)
            << "_((char **)(obj->user_data), &index, ";
        const char* msg_name = param->getBaseName();
        TypeList *msg_contents = (message_list.find(msg_name))->getContents();
        msg_contents->genUnmarshalList1(str, param->getBaseName());
        str << ");\n";
      } else {
        if(isThreaded()&&(param->isVoid()))
          str << freeMsgVoid<<"  obj->"<<name<<"("<<cpType<<");\n";
	else
          str << "  obj->"<<name<<"("<<cpType<<");\n"<<freeMsgVoid;
      }
    }
    str << "}\n";
  }
}

void Entry::genReg(XStr& str)
{
  str << "    /* REG: "<<*this << "*/\n";
  if(isConstructor() && container->isAbstract())
    return;
    
  str << "      "<<epIdx()<<" = CkRegisterEp(\""<<name 
 <<"\", (CkCallFnPtr)_call_"<<epIdx(0)<<", ";
  if(param && !param->isVoid() && !(attribs&SMIGRATE)) {
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













