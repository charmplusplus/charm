/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include "xi-symbol.h"
#include <ctype.h> // for tolower()

int fortranMode;
const char *cur_file;

const char *Prefix::Proxy="CProxy_";
const char *Prefix::ProxyElement="CProxyElement_";
const char *Prefix::ProxySection="CProxySection_";
const char *Prefix::Message="CMessage_";
const char *Prefix::Index="CkIndex_";


//Fatal error function
void die(const char *why,int line)
{
	if (line==-1)
		fprintf(stderr,"%s: Charmxi fatal error> %s\n",cur_file,why);
	else
		fprintf(stderr,"%s:%d: Charmxi fatal error> %s\n",cur_file,line,why);
	exit(1);
}

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

Value::Value(const char *s)
{
  factor = 1;
  val=s;
  if(val == 0 || strlen(val)==0 ) return;
  char *v = (char *)malloc(strlen(val)+5);
  strcpy(v,val);
  int pos = strlen(v)-1;
  if(v[pos]=='K' || v[pos]=='k') {
    v[pos] = '\0';
    factor = 1024;
  }
  if(v[pos]=='M' || v[pos]=='m') {
    v[pos] = '\0';
    factor = 1024*1024;
  }
  val=v;
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
Type::genProxyName(XStr &str,forWhom forElement) 
{
  die("type::genProxyName called (INTERNAL ERROR)");
}
void 
Type::genIndexName(XStr &str) 
{
  die("type::genIndexName called (INTERNAL ERROR)");
}
void 
Type::genMsgProxyName(XStr &str) 
{
  die("type::genMsgProxyName called (INTERNAL ERROR)");
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
int TypeList::length(void) const 
{
  if (next) return next->length()+1;
  else return 1;
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
    die("cannot create output files (check directory permissions)\n");
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


XStr Chare::proxyName(int withTemplates)
{
  XStr str;
  str<<proxyPrefix()<<type;
  if (withTemplates) str<<tvars();
  return str;
}

XStr Chare::indexName(int withTemplates)
{
  XStr str;
  str<<Prefix::Index<<type;
  if (withTemplates) str<<tvars();
  return str;
}

static const char *forWhomStr(forWhom w)
{
  switch(w) {
  case forAll: return Prefix::Proxy;
  case forIndividual: return Prefix::ProxyElement;
  case forSection: return Prefix::ProxySection;
  case forIndex: return Prefix::Index;
  default: return NULL;
  };
}

void NamedType::genProxyName(XStr& str,forWhom forElement)
{ 
   const char *prefix=forWhomStr(forElement);
   if (prefix==NULL)
	   die("Unrecognized forElement type passed to NamedType::genProxyName");
   str << prefix;
   print(str);
}

void TypeList::genProxyNames(XStr& str, const char *prefix, const char *middle,
                             const char *suffix, const char *sep,forWhom forElement)
{
  if(type) {
    str << prefix;
    type->genProxyName(str,forElement);
    if (middle!=NULL) {
      str << middle;
      type->genProxyName(str,forElement);
    }
    str << suffix;
  }
  if(next) {
    str << sep;
    next->genProxyNames(str, prefix, middle, suffix, sep,forElement);
  }
}
void Chare::genProxyNames(XStr& str, const char *prefix,const char *middle, 
    	const char *suffix, const char *sep)
{
	bases->genProxyNames(str,prefix,middle,suffix,sep,forElement);
}
void Chare::genIndexNames(XStr& str, const char *prefix,const char *middle, 
    	const char *suffix, const char *sep)
{
	bases->genProxyNames(str,prefix,middle,suffix,sep,forIndex);
}
char *Chare::proxyPrefix(void)
{
  return (char *)forWhomStr(forElement);
}

//Common multiple inheritance disambiguation code
void Chare::sharedDisambiguation(XStr &str,const XStr &super)
{
    str<<"    void ckDelegate(CkGroupID to) {\n";
    genProxyNames(str,"      ",NULL,"::ckDelegate(to);\n","");
     str<<"    }\n";
    str<<"    void ckUndelegate(void) {\n";
    genProxyNames(str,"      ",NULL,"::ckUndelegate();\n","");
    str<<"    }\n";
    str<<"    void pup(PUP::er &p) {\n";
    genProxyNames(str,"      ",NULL,"::pup(p);\n","");
    str<<"    }\n";
}


static const char *CIClassStart = // prefix, name
"{\n"
"  public:\n"
;

static const char *CIClassEnd =
"};\n"
;

Chare::Chare(int ln, attrib_t Nattr, NamedType *t, TypeList *b, MemberList *l)
	 : attrib(Nattr), type(t), list(l), bases(b)
{
	line = ln;
	entryCount=1;
        hasElement=0;
	forElement=forAll;
	setTemplate(0); 
	if (list)
	{
		list->setChare(this);
      		//Add migration constructor to MemberList
		if(isMigratable()) {
			Entry *e=new Entry(ln,SMIGRATE,NULL,
			  (char *)type->getBaseName(),
			  new ParamList(new Parameter(line,
				new PtrType(new NamedType("CkMigrateMessage"))
			  )));
			e->setChare(this);
			list=new MemberList(e,list);
		}
	}
	if (bases==NULL) //Always add Chare as a base class
		bases = new TypeList(new NamedType("Chare"), bases);
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
  str <<  tspec() <<
  "void "<<indexName()<<"::__register(const char *s, size_t size) {\n"
  "  __idx = CkRegisterChare(s, size);\n";
  // register all bases
  genIndexNames(str, "  _REGISTER_BASE(__idx, ",NULL, "::__idx);\n", "");
  if(list)
    list->genReg(str);
  str << "}\n";
  str << "#endif\n";
}

extern void sdag_trans(XStr& classname, XStr& input, XStr& output);

void
Chare::genDecls(XStr& str)
{
  if(type->isTemplated())
    return;
  str << "/* DECLS: "; print(str); str << " */\n";
 //Forward declaration of the user-defined implementation class*/
  str << tspec()<<" class "<<type<<";\n";
  
 //Generate index class
  str << "/* --------------- index object ------------------ */\n";
  str << tspec()<< "class "<<Prefix::Index<<type;
  if(external || type->isTemplated()) 
  { //Just a template instantiation/forward declaration
    str << ";";
  }
  else 
  { //Actual implementation
    str << CIClassStart;
    str << "    static int __idx;\n";
    str << "    static void __register(const char *s, size_t size);\n";
    if(list)
      list->genIndexDecls(str);
    str << CIClassEnd;
  }  
  str << "/* --------------- element proxy ------------------ */\n";
  genSubDecls(str);
  if (hasElement) {
  	str << "/* ---------------- collective proxy -------------- */\n";
  	forElement=forAll;
  	genSubDecls(str);

        if (hasSection) {
  	  str << "/* ---------------- section proxy -------------- */\n";
  	  forElement=forSection;
  	  genSubDecls(str);
	}
  	forElement=forIndividual;
  }
  
  if(list) {
    int sdagPresent = 0;
    XStr sdagStr;
    list->collectSdagCode(sdagStr, sdagPresent);
    if(sdagPresent) {
      XStr classname;
      XStr sdag_output;
      classname << baseName(0);
      sdag_trans(classname, sdagStr, sdag_output);
      str << sdag_output;
    }
  }
}

void
Chare::genSubDecls(XStr& str)
{
  XStr ptype;
  ptype<<proxyPrefix()<<type;
  
  str << tspec()<< "class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << ":";
  genProxyNames(str, "public ",NULL, "", ", ");
  str << CIClassStart;
  str << "    "<<ptype<<"(void) {};\n";
  str << "    "<<ptype<<"(CkChareID __cid) : ";
  genProxyNames(str, "",NULL, "(__cid)", ", ");
  str << "{  }\n";
  //Multiple inheritance-- resolve inheritance ambiguity
    XStr super;
    bases->getFirst()->genProxyName(super,forElement);
    str<<"    CK_DISAMBIG_CHARE("<<super<<")\n";
    sharedDisambiguation(str,super);
    str<<"    void ckSetChareID(const CkChareID &c) {\n";
    genProxyNames(str,"      ",NULL,"::ckSetChareID(c);\n","");
    str<<"    }\n";

  str<<"    "<<type<<tvars()<<" *ckLocal(void) const\n";
  str<<"     { return ("<<type<<tvars()<<" *)CkLocalChare(&ckGetChareID()); }\n";
  
  if(list)
    list->genDecls(str);
  str << CIClassEnd;
  if (!isTemplated()) str << "PUPmarshall("<<ptype<<");\n";
}

Group::Group(int ln, attrib_t Nattr,
    	NamedType *t, TypeList *b, MemberList *l)
    	:Chare(ln,Nattr|CGROUP,t,b,l) 
{
        hasElement=1;
	forElement=forIndividual;
	hasSection=0;
	if (b==NULL) {//Add Group as a base class
		if (isNodeGroup())
			bases = new TypeList(new NamedType("NodeGroup"), NULL);
		else
			bases = new TypeList(new NamedType("Group"), NULL);
	}
}

void
Group::genSubDecls(XStr& str)
{
  XStr ptype; ptype<<proxyPrefix()<<type;
  XStr ttype; ttype<<type<<tvars();
  XStr super;
  bases->getFirst()->genProxyName(super,forElement);
  
  str << tspec()<< "class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << ": ";
  genProxyNames(str, "public ",NULL, "", ", ");
  str << CIClassStart;
  str << "    "<<ptype<<"(void) {}\n";
  if (forElement==forIndividual) 
  {//For a single element
    str << "    "<<ptype<<"(CkGroupID _gid,int _onPE,CkGroupID dTo) : ";
    genProxyNames(str, "", NULL,"(_gid,_onPE,dTo)", ", ");
    str << "{  }\n";
    str << "    "<<ptype<<"(CkGroupID _gid,int _onPE) : ";
    genProxyNames(str, "", NULL,"(_gid,_onPE)", ", ");
    str << "{  }\n";

    str<<"   CK_DISAMBIG_GROUP_ELEMENT("<<super<<")\n";
  } 
  else if (forElement==forAll)
  {//For whole group
    str << "    "<<ptype<<"(CkGroupID _gid,CkGroupID dTo) : ";
    genProxyNames(str, "", NULL,"(_gid,dTo)", ", ");
    str << "{  }\n";      
    str << "    "<<ptype<<"(CkGroupID _gid) : ";
    genProxyNames(str, "", NULL,"(_gid)", ", ");
    str << "{  }\n";      

    //Group proxy can be indexed into an element proxy:
    forElement=forIndividual;//<- for the proxyName below
    str << "    "<<proxyName(1)<<" operator[](int onPE) const\n";
    str << "      {return "<<proxyName(1)<<"(ckGetGroupID(),onPE,ckDelegatedIdx());}\n";
    forElement=forAll;

    str<<"   CK_DISAMBIG_GROUP("<<super<<")\n";
  }

  //Multiple inheritance-- resolve inheritance ambiguity
  sharedDisambiguation(str,super);
  str<<"    void ckSetGroupID(CkGroupID g) {\n";
  genProxyNames(str,"      ",NULL,"::ckSetGroupID(g);\n","");
  str<<"    }\n";
  
  str << "    "<<ttype<<"* ckLocalBranch(void) const {\n";
  str << "      return ckLocalBranch(ckGetGroupID());\n";
  str << "    }\n";
  str << "    static "<<ttype<< "* ckLocalBranch(CkGroupID gID) {\n";
  str << "      return ("<<ttype<<"*)";
  if(isNodeGroup())
    str << "CkLocalNodeBranch(gID);\n";
  else
    str << "CkLocalBranch(gID);\n";
  str << "    }\n";
  if(list)
    list->genDecls(str);
  str << CIClassEnd;
  if (!isTemplated()) str << "PUPmarshall("<<ptype<<");\n";

}


//Array Constructor
Array::Array(int ln, attrib_t Nattr, NamedType *index,
	NamedType *t, TypeList *b, MemberList *l)  
    : Chare(ln,Nattr|CARRAY|CMIGRATABLE,t,b,l) 
{
        hasElement=1;
	forElement=forIndividual;
	hasSection=1;
	index->print(indexSuffix);
	if (indexSuffix!=(const char*)"none")
		indexType<<"CkArrayIndex"<<indexSuffix;
	else indexType<<"CkArrayIndex";

	if(b==0) { //No other base class:
		if (0==strcmp(type->getBaseName(),"ArrayElement"))
			//ArrayElement has special "ArrayBase" superclass
			bases = new TypeList(new NamedType("ArrayBase"), NULL);
		else //Everybody else inherits from ArrayBase
			bases = new TypeList(new NamedType("ArrayElement"), NULL);
	}
}

void
Array::genSubDecls(XStr& str)
{
  XStr ptype; ptype<<proxyPrefix()<<type;
  
  str << tspec()<< " class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << " : ";
  genProxyNames(str, "public ",NULL, "", ", ");
  
  str << CIClassStart;
  
  str << "    "<<ptype<<"(void) {}\n";//An empty constructor
  
  //Resolve multiple inheritance ambiguity
  XStr super;
  bases->getFirst()->genProxyName(super,forElement);
  sharedDisambiguation(str,super);
  
  if (forElement==forIndividual) 
  {/*For an individual element (no indexing)*/
    str << "    CK_DISAMBIG_ARRAY_ELEMENT("<<super<<")\n";
    str << "    "<<type<<tvars()<<" *ckLocal(void) const\n";
    str << "      { return ("<<type<<tvars()<<" *)"<<super<<"::ckLocal(); }\n";  
    //This constructor is used for array indexing
    str <<
         "    "<<ptype<<"(const CkArrayID &aid,const "<<indexType<<" &idx,CkGroupID dTo)\n"
         "        :";genProxyNames(str, "",NULL, "(aid,idx,dTo)", ", ");str<<" {}\n";  
    str <<
         "    "<<ptype<<"(const CkArrayID &aid,const "<<indexType<<" &idx)\n"
         "        :";genProxyNames(str, "",NULL, "(aid,idx)", ", ");str<<" {}\n";  
  } 
  else if (forElement==forAll)
  {/*Collective, indexible version*/    
    str << "    CK_DISAMBIG_ARRAY("<<super<<")\n";
    
    str<< //Build a simple, empty array
    "    static CkArrayID ckNew(void) {return ckCreateEmptyArray();}\n";
    
    XStr etype; etype<<Prefix::ProxyElement<<type;
    if (indexSuffix!=(const char*)"none")
    {
      str <<
    "//Generalized array indexing:\n"
    "    "<<etype<<" operator [] (const "<<indexType<<" &idx) const\n"
    "        {return "<<etype<<"(ckGetArrayID(), idx, ckDelegatedIdx());}\n"
    "    "<<etype<<" operator() (const "<<indexType<<" &idx) const\n"
    "        {return "<<etype<<"(ckGetArrayID(), idx, ckDelegatedIdx());}\n";
    }
  
  //Add specialized indexing for these common types
    if (indexSuffix==(const char*)"1D")
    {
    str << 
    "    "<<etype<<" operator [] (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex1D(idx), ckDelegatedIdx());}\n"
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex1D(idx), ckDelegatedIdx());}\n";
    } else if (indexSuffix==(const char*)"2D") {
    str << 
    "    "<<etype<<" operator () (int i0,int i1) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex2D(i0,i1), ckDelegatedIdx());}\n";
    } else if (indexSuffix==(const char*)"3D") {
    str << 
    "    "<<etype<<" operator () (int i0,int i1,int i2) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex3D(i0,i1,i2), ckDelegatedIdx());}\n";
    }
    str <<"    "<<ptype<<"(const CkArrayID &aid,CkGroupID dTo) \n"
         "        :";genProxyNames(str, "",NULL, "(aid,dTo)", ", ");str<<" {}\n";
    str <<"    "<<ptype<<"(const CkArrayID &aid) \n"
         "        :";genProxyNames(str, "",NULL, "(aid)", ", ");str<<" {}\n";
  }
  else if (forElement==forSection)
  { /* for Section, indexible version*/    
    str << "    CK_DISAMBIG_ARRAY_SECTION("<<super<<")\n";

    XStr etype; etype<<Prefix::ProxyElement<<type;
    if (indexSuffix!=(const char*)"none")
    {
      str <<
    "//Generalized array indexing:\n"
    "    "<<etype<<" operator [] (const "<<indexType<<" &idx) const\n"
    "        {return "<<etype<<"(ckGetArrayID(), idx, ckDelegatedIdx());}\n"
    "    "<<etype<<" operator() (const "<<indexType<<" &idx) const\n"
    "        {return "<<etype<<"(ckGetArrayID(), idx, ckDelegatedIdx());}\n";
    }
  
  //Add specialized indexing for these common types
    if (indexSuffix==(const char*)"1D")
    {
    str << 
    "    "<<etype<<" operator [] (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], ckDelegatedIdx());}\n"
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], ckDelegatedIdx());}\n";
    } else if (indexSuffix==(const char*)"2D") {
    str << 
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex2D*)&ckGetArrayElements()[idx], ckDelegatedIdx());}\n";
    } else if (indexSuffix==(const char*)"3D") {
    str << 
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex3D*)&ckGetArrayElements()[idx], ckDelegatedIdx());}\n";
    }

    str <<"    "<<ptype<<"(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems, CkGroupID dTo) \n"
         "        :";genProxyNames(str, "",NULL, "(aid,elems,nElems,dTo)", ", ");str << " {}\n";
    str <<"    "<<ptype<<"(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems) \n"
         "        :";genProxyNames(str, "",NULL, "(aid,elems,nElems)", ", ");str<<" {}\n";
    str <<"    "<<ptype<<"(const CkSectionID &sid)"
	  "       :";genProxyNames(str, "",NULL, "(sid)", ", ");str<< " {}\n";
  }
  
  if(list)
    list->genDecls(str);
  str << CIClassEnd;
  if (!isTemplated()) str << "PUPmarshall("<<ptype<<");\n";
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
    str << "extern \"C\" void " << fortranify(baseName()) << "_allocate_(char **, void *, int *);\n";
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
    str << "    " << fortranify(baseName()) << "_allocate_((char **)&user_data, &aid, &thisIndex);\n";
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
    str << tspec()<<" int "<<indexName()<<"::__idx";
    if(!external) str << "=0";
    str << ";\n";
    str << "#endif\n";
  }
  if(list) 
  {//Add definitions for all entry points
    if(isTemplated())
      str << "#ifdef CK_TEMPLATES_ONLY\n";
    else
      str << "#ifndef CK_TEMPLATES_ONLY\n";
  
    list->genDefs(str);
    if (hasElement) 
    { //Define the entry points for the element
      forElement=forAll;
      list->genDefs(str);
      if (hasSection) {  // for Section
        forElement=forSection;
        list->genDefs(str);
      }
      forElement=forIndividual;
    }
    str << "#endif /*CK_TEMPLATES_ONLY*/\n";
  }
  if(!external && !type->isTemplated())
    genRegisterMethodDef(str);
}

void
Chare::genReg(XStr& str)
{
  str << "/* REG: "; print(str); str << "*/\n";
  if(external || templat)
    return;
  str << "  "<<indexName()<<"::__register(\""<<type<<"\", sizeof("<<type<<"));\n";
}

static const char *CIMsgClassAnsi =
"{\n"
"  public:\n"
"    static int __idx;\n"
"    void* operator new(size_t,void*p) { return p; }\n"
"    void* operator new(size_t,const int);\n"
"    void* operator new(size_t);\n"
"    void* operator new(size_t, int*, const int);\n"
"    void* operator new(size_t, int*);\n"
"#if CMK_MULTIPLE_DELETE\n"
"    void operator delete(void*p,void*){CkFreeMsg(p);}\n"
"    void operator delete(void*p,const int){CkFreeMsg(p);}\n"
"    void operator delete(void*p){ CkFreeMsg(p);}\n"
"    void operator delete(void*p,int*,const int){CkFreeMsg(p);}\n"
"    void operator delete(void*p,int*){CkFreeMsg(p);}\n"
"#endif\n"
"    void operator delete(void*p,size_t){CkFreeMsg(p);}\n"
"    static void* alloc(int,size_t,int*,int);\n"
;

void
Message::genAllocDecl(XStr &str)
{
  int i, num;
  XStr mtype;
  mtype << type;
  if(templat) templat->genVars(mtype);
  str << CIMsgClassAnsi;
  str << "    CMessage_" << mtype << "() {};\n";
  str << "    static void *pack(" << mtype << " *p);\n";
  str << "    static " << mtype << "* unpack(void* p);\n";
  num = numVars();
  if(num>0) {
    str << "    void *operator new(size_t,";
    for(i=0;i<num;i++)
      str << "int, ";
    str << "const int);\n";
    str << "#if CMK_MULTIPLE_DELETE\n";
    str << "    void operator delete(void *p,";
    for(i=0;i<num;i++)
        str << "int, ";
    str << "const int){CkFreeMsg(p);}\n";
    str << "#endif\n";
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
  str << ":public CkMessage";

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

    // new (size_t)
    str << tspec << "void *" << ptype << "::operator new(size_t s){\n";
    str << "  return " << mtype << "::alloc(__idx, s, 0, 0);\n}\n";
    // new (size_t, priobits)
    str << tspec << "void *" << ptype << "::operator new(size_t s,";
    str << "const int pb){\n";
    str << "  return " << mtype << "::alloc(__idx, s, 0, pb);\n}\n";
    // new (size_t, int*)
    str << tspec << "void *" << ptype << "::operator new(size_t s, int* sz){\n";
    str << "  return " << mtype << "::alloc(__idx, s, sz, 0);\n}\n";
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
      str << "  int sizes[" << num << "];\n";
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
}

void
Readonly::genIndexDecls(XStr& str)
{
  str << "/* DECLS: "; print(str); str << " */\n";
}

//Turn this string into a valid identifier
XStr makeIdent(const XStr &in)
{
  XStr ret;
  const char *i=in.get_string_const();
  while (*i!=0) {
    //Quote all "special" characters
    if (*i==':') ret<<"_QColon_";
    else if (*i==' ') ret<<"_QSpace_";
    else if (*i=='+') ret<<"_QPlus_";
    else if (*i=='-') ret<<"_QMinus_";
    else if (*i=='*') ret<<"_QTimes_";
    else if (*i=='/') ret<<"_QSlash_";
    else if (*i=='%') ret<<"_QPercent_";
    else if (*i=='&') ret<<"_QAmpersand_";
    else if (*i=='.') ret<<"_QDot_";
    else if (*i==',') ret<<"_QComma_";
    else if (*i=='\'') ret<<"_QSQuote_";
    else if (*i=='\"') ret<<"_QQuote_";
    else if (*i=='(') ret<<"_QLparen_";
    else if (*i==')') ret<<"_QRparen_";
    else if (*i=='<') ret<<"_QLess_";
    else if (*i=='>') ret<<"_QGreater_";
    else if (*i=='{') ret<<"_QLbrace_";
    else if (*i=='}') ret<<"_QRbrace_";
    else ret << *i; //Copy character unmodified
    i++; //Advance to next
  }
  return ret;
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

  if (!msg) { //Generate a pup for this readonly
    str << "extern \"C\" void __xlater_roPup_"<<makeIdent(qName());
    str <<    "(void *pup_er) {\n";
    str << "  PUP::er &p=*(PUP::er *)pup_er;\n";
    str << "  p|"<<qName()<<";\n";
    str << "}\n";
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
    if(dims) die("readonly Message cannot be an array",line);
    str << "  CkRegisterReadonlyMsg(\""<<qName()<<"\",\""<<type<<"\",";
    str << "(void **)&"<<qName()<<");\n";
  } else {
    str << "  CkRegisterReadonly(\""<<qName()<<"\",\""<<type<<"\",";
    str << "sizeof("<<qName()<<"),(void *) &"<<qName()<<",";
    str << "__xlater_roPup_"<<makeIdent(qName())<<");\n";
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

void MemberList::genIndexDecls(XStr& str)
{
  if(member)
    member->genIndexDecls(str);
  if(next) {
    str << endx;
    next->genIndexDecls(str);
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

void MemberList::collectSdagCode(XStr& str, int& sdagPresent)
{
  if(member)
    member->collectSdagCode(str, sdagPresent);
  if(next) {
    str << endx;
    next->collectSdagCode(str, sdagPresent);
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


Entry::Entry(int l, int a, Type *r, char *n, ParamList *p, Value *sz) :
      attribs(a), retType(r), name(n), param(p), stacksize(sz)
{ 
  line=l; container=NULL; 
  entryCount=-1;
  sdagCode = 0;

  if(!isThreaded() && stacksize) die("Non-Threaded methods cannot have stacksize",line);
  if(retType && !isSync() && !retType->isVoid()) 
    die("A remote method normally returns void.  To return non-void, you need to declare the method as [sync], which means it has blocking semantics.",line);
}
void Entry::setChare(Chare *c) {
	Member::setChare(c);
	if (param==NULL) 
	{//Fake a parameter list of the appropriate type
		Type *t;
		if (isConstructor()&&container->isMainChare())
			//Main chare always magically takes CkArgMsg
			t=new PtrType(new NamedType("CkArgMsg"));
		else
			t=new BuiltinType("void");
		param=new ParamList(new Parameter(line,t));
	}
	entryCount=c->nextEntry();
}

// "parameterType *msg" or "void".
// Suitable for use as the only parameter
XStr Entry::paramType(int withDefaultVals)
{
  XStr str;
  param->print(str,withDefaultVals);
  return str;
}

// "parameterType *msg," if there is a non-void parameter, 
// else empty.  Suitable for use with another parameter following.
XStr Entry::paramComma(int withDefaultVals)
{
  XStr str;
  if (!param->isVoid()) {
  	param->print(str,withDefaultVals);
  	str << ", ";
  }
  return str;
}

XStr Entry::marshallMsg(void)
{
  XStr ret;
  param->marshall(ret);
  return ret;
}

XStr Entry::epStr(void)
{
  XStr str;
  str << name << "_";
  if (param->isMessage()) str<<param->getBaseName();
  else if (param->isVoid()) str<<"void";
  else str<<"marshall"<<entryCount;
  return str;
}

XStr Entry::epIdx(int fromProxy)
{
  XStr str;
  if (fromProxy)
    str << indexName()<<"::";
  str << "__idx_"<<epStr();
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

//Return a templated proxy declaration string for 
// this Member's container with the given return type, e.g.
// template<int N,class foo> void CProxy_bar<N,foo>
// Works with non-templated Chares as well.
XStr Member::makeDecl(const XStr &returnType,int forProxy)
{
  XStr str;
  
  if (container->isTemplated())
    str << container->tspec() << " ";
  str << returnType<<" ";
  if (forProxy)
  	str<<container->proxyName();
  else
  	str<<container->indexName();
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
    str << "    "<<retType<<" "<<name<<"("<<paramType(1)<<");\n";
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
    str << makeDecl(retStr,1)<<"::"<<name<<"("<<paramType(0)<<")\n";
    str << "{\n"<<marshallMsg();
    if(isSync()) {
      str << syncReturn() << "CkRemoteCall("<<params<<"));\n";
    } else {//Regular, non-sync message
      str << "  if (ckIsDelegated())\n";
      str << "    ckDelegatedTo()->ChareSend("<<params<<");\n";
      str << "  else CkSendMsg("<<params<<");\n";
    }
    str << "}\n";
  }
}

void Entry::genChareStaticConstructorDecl(XStr& str)
{
  str << "    static CkChareID ckNew("<<paramComma(1)<<"int onPE=CK_PE_ANY);\n";
  str << "    static void ckNew("<<paramComma(1)<<"CkChareID* pcid, int onPE=CK_PE_ANY);\n";
  if (!param->isVoid())
    str << "    "<<container->proxyName(0)<<"("<<paramComma(1)<<"int onPE=CK_PE_ANY);\n";
}

void Entry::genChareStaticConstructorDefs(XStr& str)
{
  str << makeDecl("CkChareID",1)<<"::ckNew("<<paramComma(0)<<"int impl_onPE)\n";
  str << "{\n"<<marshallMsg();
  str << "  CkChareID impl_ret;\n";
  str << "  CkCreateChare("<<chareIdx()<<", "<<epIdx()<<", impl_msg, &impl_ret, impl_onPE);\n";
  str << "  return impl_ret;\n";
  str << "}\n";

  str << makeDecl("void",1)<<"::ckNew("<<paramComma(0)<<"CkChareID* pcid, int impl_onPE)\n";
  str << "{\n"<<marshallMsg();
  str << "  CkCreateChare("<<chareIdx()<<", "<<epIdx()<<", impl_msg, pcid, impl_onPE);\n";
  str << "}\n";
  
  if (!param->isVoid()) {
    str << makeDecl(" ",1)<<"::"<<container->proxyName(0)<<"("<<paramComma(0)<<"int impl_onPE)\n";
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
    genArrayStaticConstructorDecl(str);
  } else {
    if (isSync() && !container->isForElement()) return; //No sync broadcast
    str << "    "<<retType<<" "<<name<<"("<<paramType(1)<<") ;\n"; //no const
  }
}

void Entry::genArrayDefs(XStr& str)
{
  if (isConstructor())
    genArrayStaticConstructorDefs(str);
  else
  {//Define array entry method
    const char *ifNot="CkArray_IfNotThere_buffer";
    if (isCreateHere()) ifNot="CkArray_IfNotThere_createhere";
    if (isCreateHome()) ifNot="CkArray_IfNotThere_createhome";
    
    if (isSync() && !container->isForElement()) return; //No sync broadcast
    
    XStr retStr; retStr<<retType;
    str << makeDecl(retStr,1)<<"::"<<name<<"("<<paramType(0)<<") \n"; //no const
    str << "{\n"<<marshallMsg();
    str << "  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;\n";
    str << "  impl_amsg->array_setIfNotThere("<<ifNot<<");\n";
    if(isSync()) {
      str << syncReturn() << "ckSendSync(impl_amsg, "<<epIdx()<<"));\n";
    } 
    else 
    {
      if (container->isForElement() || container->isForSection())
        str << "    ckSend(impl_amsg, "<<epIdx()<<");\n";
      else
        str << "    ckBroadcast(impl_amsg, "<<epIdx()<<");\n";
    }
    str << "}\n";
  }
}

void Entry::genArrayStaticConstructorDecl(XStr& str)
{
  if (container->getForWhom()==forIndividual)
      str<< //Element insertion routine
      "    void insert("<<paramComma(1)<<"int onPE=-1);";
  else if (container->getForWhom()==forAll)
      str<< //With options
      "    static CkArrayID ckNew("<<paramComma(1)<<"const CkArrayOptions &opts);\n";
  else if (container->getForWhom()==forSection) 
      str << 
      "    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems) {\n"
      "      return CkSectionID(aid, elems, nElems);\n"
      "    } \n";
}

void Entry::genArrayStaticConstructorDefs(XStr& str)
{
  if (container->getForWhom()==forIndividual)
      str<<
      makeDecl("void",1)<<"::insert("<<paramComma(0)<<"int onPE)\n"
      "{ \n"<<marshallMsg()<<
      "   ckInsert((CkArrayMessage *)impl_msg,"<<epIdx()<<",onPE);\n}\n";
  else if (container->getForWhom()==forAll)
      str<<
      makeDecl("CkArrayID",1)<<"::ckNew("<<paramComma(0)<<"const CkArrayOptions &opts)\n"
       "{ \n"<<marshallMsg()<<
	 "   return ckCreateArray((CkArrayMessage *)impl_msg,"<<epIdx()<<",opts);\n"
       "}\n";
      
}


/******************************** Group Entry Points *********************************/

void Entry::genGroupDecl(XStr& str)
{  
  //Selects between NodeGroup and Group
  char *node = (char *)(container->isNodeGroup()?"Node":"");

  if(isConstructor()) {
    genGroupStaticConstructorDecl(str);
  } else {
    int forElement=container->isForElement();
    XStr params; params<<epIdx()<<", impl_msg";
    XStr paramg; paramg<<epIdx()<<", impl_msg, ckGetGroupID()";
    XStr parampg; parampg<<epIdx()<<", impl_msg, ckGetGroupPe(), ckGetGroupID()";

    if (isSync() && !container->isForElement()) return; //No sync broadcast
    
    str << "    "<<retType<<" "<<name<<"("<<paramType(1)<<")\n";
    str << "    {\n"<<marshallMsg();

    if(isSync()) {
      str << syncReturn() <<
        "CkRemote"<<node<<"BranchCall("<<paramg<<", ckGetGroupPe()));\n"; 
    } 
    else
    { //Non-sync entry method
      if (forElement)
      {// Send
        str << "      if (ckIsDelegated()) \n";
	str << "         ckDelegatedTo()->"<<node<<"GroupSend("<<parampg<<");\n";
        str << "      else CkSendMsg"<<node<<"Branch("<<parampg<<");\n";
      }
      else
      {// Broadcast
        str << "      if (ckIsDelegated()) \n";
        str << "         ckDelegatedTo()->"<<node<<"GroupBroadcast("<<paramg<<");\n";
        str << "      else CkBroadcastMsg"<<node<<"Branch("<<paramg<<");\n";
      }
    }
    str << "    }\n";

    // entry method on multiple PEs declaration
    if(!forElement && !isSync() && !container->isNodeGroup()) {
      str << "    "<<retType<<" "<<name<<"("<<paramComma(1)<<"int npes, int *pes)\n";
      str << "    {\n"<<marshallMsg();
      str << "      CkSendMsg"<<node<<"BranchMulti("<<params<<", npes, pes, ckGetGroupID());\n";
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
  if (container->isForElement()) return;
  
  str << "    static CkGroupID ckNew("<<paramType(1)<<");\n";
  if (!param->isVoid()) {
    str << "    "<<container->proxyName(0)<<"("<<paramType(1)<<");\n";
  }
}

void Entry::genGroupStaticConstructorDefs(XStr& str)
{
  if (container->isForElement()) return;
  
  //Selects between NodeGroup and Group
  char *node = (char *)(container->isNodeGroup()?"Node":"");
  str << makeDecl("CkGroupID",1)<<"::ckNew("<<paramType(0)<<")\n";
  str << "{\n"<<marshallMsg();
  str << "  return CkCreate"<<node<<"Group("<<chareIdx()<<", "<<epIdx()<<", impl_msg);\n";
  str << "}\n";

  if (!param->isVoid()) {
    str << makeDecl(" ",1)<<"::"<<container->proxyName(0)<<"("<<paramType(0)<<")\n";
    str << "{\n"<<marshallMsg();
    str << "  ckSetGroupID(CkCreate"<<node<<"Group("<<chareIdx()<<", "<<epIdx()<<", impl_msg));\n";
    str << "}\n";
  }
}

/******************* Shared Entry Point Code **************************/
void Entry::genIndexDecls(XStr& str)
{
  str << "/* DECLS: "; print(str); str << " */\n";
  
  // Entry point index storage
  str << "    static int "<<epIdx(0)<<";\n";
  
  // Index function, so user can find the entry point number
  str << "    static int ";
  if (isConstructor()) str <<"ckNew";
  else str <<name;
  str << "("<<paramType(1)<<") { return "<<epIdx(0)<<"; }\n"; 

  // call function declaration
  str << "    static void _call_"<<epStr()<<"(void* impl_msg,"<<
    container->baseName()<<"* impl_obj);\n";
  if(isThreaded()) {
    str << "    static void _callthr_"<<epStr()<<"(CkThrCallArg *);\n";
  }
}

void Entry::genDecls(XStr& str)
{
  if(isConstructor() && retType && !retType->isVoid())
    die("Constructors cannot return a value",line);
  
  str << "/* DECLS: "; print(str); str << " */\n";
  if(retType==0 && !isConstructor())
      die("Entry methods must specify a return type-- \n"
      	"use void if necessary",line);
  
  if (attribs&SMIGRATE) 
    {} //User cannot call the migration constructor
  else if(container->isGroup()) {
      genGroupDecl(str);
  } else if(container->isArray()) {
      genArrayDecl(str);
  } else { // chare or mainchare
      genChareDecl(str);
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
   <<", new CkThrCallArg(impl_msg,impl_obj), "<<getStackSize()<<"));\n}\n";
  str << makeDecl("void")<<"::"<<procFull<<"(CkThrCallArg *impl_arg)\n";
  str << "{\n";\
  str << "  void *impl_msg = impl_arg->msg;\n";
  str << "  "<<container->baseName()<<" *impl_obj = ("<<container->baseName()<<" *) impl_arg->obj;\n";
  str << "  delete impl_arg;\n";
  return str;
}

void Entry::genDefs(XStr& str)
{
  XStr containerType=container->baseName();
  XStr preCall,postCall;
  
  str << "/* DEFS: "; print(str); str << " */\n";
    
  if (attribs&SMIGRATE) 
    {} //User cannot call the migration constructor
  else if(container->isGroup()){
    genGroupDefs(str);
  } else if(container->isArray()) {
    genArrayDefs(str);
  } else
    genChareDefs(str);

  //Prevents repeated call and __idx definitions:
  if (container->getForWhom()!=forAll) return;
  
  //Define storage for entry point number
  str << container->tspec()<<" int "<<indexName()<<"::"<<epIdx(0)<<"=0;\n";

  // Add special pre- and post- call code
  if(isSync()) {
  //A synchronous method can return a value, and must finish before
  // the caller can proceed.
    if(isConstructor()) die("Constructors cannot be [sync]",line);
    preCall<< "  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);\n";
    preCall << "  void *impl_retMsg=";
    if(retType->isVoid()) {
      preCall << "CkAllocSysMsg();\n  ";
    } else {
      preCall << "(void *) ";
    }
    
    postCall << "  CkSendToFuture(impl_ref, impl_retMsg, impl_src);\n";
  } else if(isExclusive()) {
  //An exclusive method 
    if(!container->isNodeGroup()) die("only nodegroup methods can be exclusive",line);
    if(isConstructor()) die("Constructors cannot be [exclusive]",line);
    preCall << "  if(CmiTryLock(impl_obj->__nodelock)) {\n"; /*Resend msg. if lock busy*/
    /******* DANGER-- RESEND CODE UNTESTED **********/
    preCall << "    CkSendMsgNodeBranch("<<epIdx()<<",impl_msg,CkMyNode(),CkGetNodeGroupID());\n";
    preCall << "    return;\n";
    preCall << "  }\n";
    
    postCall << "  CmiUnlock(impl_obj->__nodelock);\n";
  }

  if (!isConstructor() && fortranMode) { // Fortran90
      const char* msg_name = param->getBaseName();

      str << "/* FORTRAN SECTION */\n";

      // Declare the Fortran Entry Function
      // This is called from C++
      str << "extern \"C\" void " << fortranify(name) << "_(char **, int*, ";
      param->printAddress(str);
      str << ");\n";

      // Define the Fortran interface function
      // This is called from Fortran to send the message to a chare.
      str << "extern \"C\" void "
        //<< container->proxyName() << "_" 
          << fortranify("SendTo_")
	  << fortranify(container->baseName())
          << "_" << fortranify(name)
          << "_(long* aindex, int *index, ";
      param->printAddress(str);
      str << ")\n";
      str << "{\n";
      str << "  CkArrayID *aid = (CkArrayID *)*aindex;\n";
      str << "\n";
      str << "  " << container->proxyName() << " h(*aid);\n";
      str << "  h[*index]." << name << "(";
      param->printValue(str);
      str << ");\n";
      str << "}\n";
      str << "/* FORTRAN SECTION END */\n";
    }
  
  //Generate the call-method body
  str << makeDecl("void")<<"::_call_"<<epStr()<<"(void* impl_msg,"<<containerType<<" * impl_obj)\n";
  str << "{\n";
  if(isThreaded()) str << callThread(epStr());
  str << preCall;
  param->beginUnmarshall(str);
  if (!isConstructor() && fortranMode) {
    str << "/* FORTRAN */\n";
    str << "  int index = impl_obj->thisIndex;\n";
    str << "  " << fortranify(name)
	<< "_((char **)(impl_obj->user_data), &index, ";
    param->unmarshallAddress(str); str<<");\n";
    str << "/* FORTRAN END */\n";
  }
  else {
  if(isConstructor()) str << "  new (impl_obj) "<<containerType;
  else str << "  impl_obj->"<<name;
  str<<"("; param->unmarshall(str); str<<");\n";
  }
  param->endUnmarshall(str);
  str << postCall;
  str << "}\n";
}

void Entry::genReg(XStr& str)
{
  str << "// REG: "<<*this;
  str << "  "<<epIdx(0)<<" = CkRegisterEp(\""<<name<<"("<<paramType(0)<<")\",\n"
  	"     (CkCallFnPtr)_call_"<<epStr()<<", ";
  if (param->isMarshalled()) {
    str<<"CkMarshallMsg::__idx";
  } else if(!param->isVoid() && !(attribs&SMIGRATE)) {
    param->genMsgProxyName(str);
    str <<"::__idx";
  } else {
    str << "0";
  }
  str << ", __idx);\n";
  if (isConstructor()) {
    if(container->isMainChare())
      str << "  CkRegisterMainChare(__idx, "<<epIdx(0)<<");\n";
    if(param->isVoid())
      str << "  CkRegisterDefaultCtor(__idx, "<<epIdx(0)<<");\n";
    if(attribs&SMIGRATE)
      str << "  CkRegisterMigCtor(__idx, "<<epIdx(0)<<");\n";
  }
}


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
	|  PUP'd offset-to-yarr
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
    	:type(Ntype), name(Nname), arrLen(NarrLen), val(Nvalue),line(Nline)
{
        given_name = Nname;
	if (isMessage()) {
		name="impl_msg";
        }
	if (name==NULL && !isVoid()) 
	{/*Fabricate a unique name for this marshalled param.*/
		static int unnamedCount=0;
		name=new char[50];
		sprintf((char *)name,"impl_noname_%x",unnamedCount++);
	}
	byReference=(type->isNamed())&&(arrLen==NULL)&&(val==NULL);
}

void ParamList::print(XStr &str,int withDefaultValues)
{
    	param->print(str,withDefaultValues);
    	if (next) {
    		str<<", ";
    		next->print(str,withDefaultValues);
    	}
}
void Parameter::print(XStr &str,int withDefaultValues) 
{
	if (arrLen!=NULL) 
	{ //Passing arrays by const pointer-reference
		str<<"const "<<type<<" *";
		if (name!=NULL) str<<name;
	}
	else {
		if (byReference) 
		{ //Pass named types by const C++ reference 
			str<<"const "<<type<<" &";
	                if (name!=NULL) str<<name;
		}
		else 
		{ //Pass everything else by value
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

/** marshalling: pack fields into flat byte buffer **/
void ParamList::marshall(XStr &str)
{
	if (isVoid())
		str<<"  void *impl_msg = CkAllocSysMsg();\n";
	else if (isMarshalled()) 
	{
		str<<"  //Marshall: ";print(str,0);str<<"\n";
		//First pass: find sizes
		str<<"  int impl_off=0,impl_arrstart=0;\n";
		int hasArrays=orEach(&Parameter::isArray);
		if (hasArrays) {
		  callEach(&Parameter::marshallArraySizes,str);
		}
		str<<"  { //Find the size of the PUP'd data\n";
		str<<"    PUP::sizer implP;\n";
		callEach(&Parameter::pup,str);
		str<<"    impl_arrstart=CK_ALIGN(implP.size(),16);\n";
		str<<"    impl_off+=impl_arrstart;\n";
		str<<"  }\n";
		//Now that we know the size, allocate the packing buffer
		str<<"  CkMarshallMsg *impl_msg=new (impl_off,0)CkMarshallMsg;\n";
		//Second pass: write the data
		str<<"  { //Copy over the PUP'd data\n";
		str<<"    PUP::toMem implP((void *)impl_msg->msgBuf);\n";
		callEach(&Parameter::pup,str);
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
		die("can't pass pointers across processors--\n"
		    "Indicate the array length with []'s, or pass a reference",line);
	if (isArray()) {
		str<<"  int impl_off_"<<name<<", impl_cnt_"<<name<<";\n";
		str<<"  impl_off_"<<name<<"=impl_off=CK_ALIGN(impl_off,sizeof("<<dt<<"));\n";
		str<<"  impl_off+=(impl_cnt_"<<name<<"=sizeof("<<dt<<")*("<<arrLen<<"));\n";
	}
}
void Parameter::pup(XStr &str) {
	if (isArray())  str<<"    implP|impl_off_"<<name<<";\n";
	else  {
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

/** unmarshalling: unpack fields from flat buffer **/
void ParamList::beginUnmarshall(XStr &str) 
{
    	if (isMarshalled()) 
    	{
    		str<<"  //Unmarshall: ";print(str,0);str<<"\n";
    		str<<"  char *impl_buf=((CkMarshallMsg *)impl_msg)->msgBuf;\n";
    		str<<"  PUP::fromMem implP(impl_buf);\n";
    		callEach(&Parameter::beginUnmarshall,str);
    		str<<"  impl_buf+=CK_ALIGN(implP.size(),16);\n";
    	}
	else if (isVoid()) {str<<"  CkFreeSysMsg(impl_msg);\n";}
}
void Parameter::beginUnmarshall(XStr &str) 
{
	Type *dt=type->deref();//Type, without &
	if (isArray())
		str<<"  int impl_off_"<<name<<"; implP|impl_off_"<<name<<";\n";
	else
		str<<"  "<<dt<<" "<<name<<"; implP|"<<name<<";\n";
}
void ParamList::unmarshall(XStr &str) 
{
    	if (isMessage()) str<<"("<<param->type<<")impl_msg";
    	else if (isMarshalled()) {
    		param->unmarshall(str);
    		if (next) {
    			str<<", ";
    			next->unmarshall(str);
    		}
    	}
}
void Parameter::unmarshall(XStr &str)
{
	if (isArray())
		str<<"("<<type->deref()<<" *)(impl_buf+impl_off_"<<name<<")";
	else
		str<<name;
}
void ParamList::unmarshallAddress(XStr &str) 
{
    	if (isMessage()) str<<"("<<param->type<<")impl_msg";
    	else if (isMarshalled()) {
    		param->unmarshallAddress(str);
    		if (next) {
    			str<<", ";
    			next->unmarshallAddress(str);
    		}
    	}
}
void Parameter::unmarshallAddress(XStr &str)
{
	if (isArray())
		str<<"("<<type->deref()<<" *)(impl_buf+impl_off_"<<name<<")";
	else
		str<<"&" <<name;
}
void ParamList::endUnmarshall(XStr &str) 
{
    	if (isMarshalled()) {
    		str<<"  delete (CkMarshallMsg *)impl_msg;\n";
    	}
}

/***************** InitCall **************/
InitCall::InitCall(int l, const char *n)
	    : name(n)
{ 
	line=l; setChare(0); 
}
void InitCall::print(XStr& str)
{
	str<<"  initcall void "<<name<<"(void);\n";
}
void InitCall::genDecls(XStr& str) {}
void InitCall::genIndexDecls(XStr& str) {}
void InitCall::genDefs(XStr& str) {}
void InitCall::genReg(XStr& str)
{
	str<<"      ";
	if (container)
		str<<container->baseName()<<"::";
	str<<name<<"();\n";
}

/****************** Registration *****************/

