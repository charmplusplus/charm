#include <list>
using std::list;
#include <algorithm>
using std::for_each;
#include <stdlib.h>
#include "xi-symbol.h"
#include <ctype.h> // for tolower()
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#if ! CMK_BOOL_DEFINED
typedef enum {false = 0, true = 1} bool;
#endif

#include <fstream>

namespace xi {
   
int fortranMode;
int internalMode;
const char *cur_file;
const char *python_doc;

const char *Prefix::Proxy="CProxy_";
const char *Prefix::ProxyElement="CProxyElement_";
const char *Prefix::ProxySection="CProxySection_";
const char *Prefix::Message="CMessage_";
const char *Prefix::Index="CkIndex_";
const char *Prefix::Python="CkPython_";


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
char* fortranify(const char *s, const char *suff1="", const char *suff2="", const char *suff3="")
{
  int i, len1 = strlen(s), len2 = strlen(suff1),
         len3 = strlen(suff2), len4 = strlen(suff3);
  int c = len1+len2+len3+len4;
  char str[1024], strUpper[1024];
  strcpy(str, s);
  strcat(str, suff1);
  strcat(str, suff2);
  strcat(str, suff3);
  for(i = 0; i < c+1; i++)
    str[i] = tolower(str[i]);
  for(i = 0; i < c+1; i++)
    strUpper[i] = toupper(str[i]);
  char *retVal;
  retVal = new char[2*c+20];
  strcpy(retVal, "FTN_NAME(");
  strcat(retVal, strUpper);
  strcat(retVal, ",");
  strcat(retVal, str);
  strcat(retVal, ")");

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


/**
   Apply fn_ on each Construct in the list l, passing it arg as
   the target. If between_ is passed, do that to arg between each
   element.
 */
template<typename T, typename U, typename A>
class perElemGenC
{
    void (U::*fn)(A);
    void (*between)(A);
    A arg;
public:
    perElemGenC(list<T*> &l,
	       A arg_,
	       void (U::*fn_)(A),
	       void (*between_)(A) = NULL)
	: fn(fn_), between(between_), arg(arg_)
	{
	    for_each(l.begin(), l.end(), *this);
	}
    void operator()(T* m)
	{
	    if (m)
	    {
		(m->*fn)(arg);
		if (between)
		    between(arg);
	    }
	}
};

template<typename T, typename U, typename A>
void perElemGen(list<T*> &l, A& arg_, void (U::*fn_)(A&),
// Sun Studio 7 (C++ compiler version 5.4) can't handle this
//		void (*between_)(A&) = NULL)
		void (*between_)(A&))
{
    perElemGenC<T, U, A&>(l, arg_, fn_, between_);
}

template<typename T, typename U, typename A>
void perElemGen(list<T*> &l, A& arg_, void (U::*fn_)(A&))
{
    perElemGenC<T, U, A&>(l, arg_, fn_, NULL);
}

template<typename T, typename U, typename A>
void perElemGen(list<T*> &l, A* arg_, void (U::*fn_)(A*),
// See above
//		void (*between_)(A*) = NULL)
		void (*between_)(A*))
{
    perElemGenC<T, U, A*>(l, arg_, fn_, between_);
}

template<typename T, typename U, typename A>
void perElemGen(list<T*> &l, A* arg_, void (U::*fn_)(A*))
{
    perElemGenC<T, U, A*>(l, arg_, fn_, NULL);
}

void newLine(XStr &str)
{
    str << endx;
}

ConstructList::ConstructList(int l, Construct *c, ConstructList *n)
{
    constructs.push_back(c);
    if (n)
	constructs.insert(constructs.end(),
			  n->constructs.begin(), n->constructs.end());
    line = l;
}

void
ConstructList::setExtern(int e)
{
  Construct::setExtern(e);
  perElemGen(constructs, e, &Construct::setExtern);
}

void
ConstructList::setModule(Module *m)
{
  Construct::setModule(m);
  perElemGen(constructs, m, &Construct::setModule);
}

void
ConstructList::print(XStr& str)
{
    perElemGen(constructs, str, &Construct::print);
}

int ConstructList::genAccels_spe_c_funcBodies(XStr& str) {
    int rtn = 0;
    for (list<Construct *>::iterator i = constructs.begin();
	 i != constructs.end(); ++i)
	if (*i) rtn += (*i)->genAccels_spe_c_funcBodies(str);
    return rtn;
}
void ConstructList::genAccels_spe_c_regFuncs(XStr& str) {
    perElemGen(constructs, str, &Construct::genAccels_spe_c_regFuncs);
}
void ConstructList::genAccels_spe_c_callInits(XStr& str) {
    perElemGen(constructs, str, &Construct::genAccels_spe_c_callInits);
}
void ConstructList::genAccels_spe_h_includes(XStr& str) {
    perElemGen(constructs, str, &Construct::genAccels_spe_h_includes);
}
void ConstructList::genAccels_spe_h_fiCountDefs(XStr& str) {
    perElemGen(constructs, str, &Construct::genAccels_spe_h_fiCountDefs);
}
void ConstructList::genAccels_ppe_c_regFuncs(XStr& str) {
    perElemGen(constructs, str, &Construct::genAccels_ppe_c_regFuncs);
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

std::string TParamList::to_string()
{
    XStr s;
    print(s);
    return s.get_string();
}


void
Type::genProxyName(XStr &str,forWhom forElement)
{
  (void)str; (void)forElement;
  die("type::genProxyName called (INTERNAL ERROR)");
}
void
Type::genIndexName(XStr &str)
{
  (void)str;
  die("type::genIndexName called (INTERNAL ERROR)");
}
void
Type::genMsgProxyName(XStr &str)
{
  (void)str;
  die("type::genMsgProxyName called (INTERNAL ERROR)");
}

void
NamedType::print(XStr& str)
{
  if (scope) str << scope;
  str << name;
  if (tparams) str << "<"<<tparams<<" >";
}

void NamedType::genIndexName(XStr& str) { 
    if (scope) str << scope;
    str << Prefix::Index; 
    str << name;
    if (tparams) str << "<"<<tparams<<" >";
}

void NamedType::genMsgProxyName(XStr& str) { 
    if (scope) str << scope;
    str << Prefix::Message;
    str << name;
    if (tparams) str << "<"<<tparams<<" >";
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

MemberList::MemberList(Member *m, MemberList *n)
{
    members.push_back(m);
    if (n)
	members.insert(members.end(), n->members.begin(), n->members.end());
}


void
MemberList::print(XStr& str)
{
    perElemGen(members, str, &Member::print);
}

void
MemberList::appendMember(Member *m)
{
    members.push_back(m);
}

int MemberList::genAccels_spe_c_funcBodies(XStr& str) {
    int rtn = 0;
    for (list<Member*>::iterator i = members.begin(); i != members.end(); ++i)
	if (*i)
	    rtn += (*i)->genAccels_spe_c_funcBodies(str);
    return rtn;
}
void MemberList::genAccels_spe_c_regFuncs(XStr& str) {
    perElemGen(members, str, &Member::genAccels_spe_c_regFuncs);
}
void MemberList::genAccels_spe_c_callInits(XStr& str) {
    perElemGen(members, str, &Member::genAccels_spe_c_callInits);
}
void MemberList::genAccels_spe_h_includes(XStr& str) {
    perElemGen(members, str, &Member::genAccels_spe_h_includes);
}
void MemberList::genAccels_spe_h_fiCountDefs(XStr& str) {
    perElemGen(members, str, &Member::genAccels_spe_h_fiCountDefs);
}
void MemberList::genAccels_ppe_c_regFuncs(XStr& str) {
    perElemGen(members, str, &Member::genAccels_ppe_c_regFuncs);
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
  using std::ofstream;
  XStr declstr, defstr;
  XStr pubDeclStr, pubDefStr, pubDefConstr;

  // DMK - Accel Support
  #if CMK_CELL != 0
  XStr accelstr_spe_c, accelstr_spe_h;
  #endif

  declstr <<
  "#ifndef _DECL_"<<name<<"_H_\n"
  "#define _DECL_"<<name<<"_H_\n"
  "#include \"charm++.h\"\n";
  if (fortranMode) declstr << "#include \"charm-api.h\"\n";
  if (clist) clist->genDecls(declstr);
  declstr << "extern void _register"<<name<<"(void);\n";
  if(isMain()) {
    declstr << "extern \"C\" void CkRegisterMainModule(void);\n";
  }
  declstr << "#endif"<<endx;
  // Generate the publish class if there are structured dagger connect entries
  int connectPresent = 0;
  if (clist) clist->genPub(pubDeclStr, pubDefStr, pubDefConstr, connectPresent);
  if (connectPresent == 1) {
     pubDeclStr << "};\n\n";
     pubDefConstr <<"}\n\n";
  }

  // defstr << "#ifndef _DEFS_"<<name<<"_H_"<<endx;
  // defstr << "#define _DEFS_"<<name<<"_H_"<<endx;
  genDefs(defstr);
  defstr <<
  "#ifndef CK_TEMPLATES_ONLY\n"
  "void _register"<<name<<"(void)\n"
  "{\n"
  "  static int _done = 0; if(_done) return; _done = 1;\n";
  if (clist) clist->genReg(defstr);
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
    "}\n";
  }
  defstr << "#endif\n";
  // defstr << "#endif"<<endx;


  // DMK - Accel Support
  #if CMK_CELL != 0

  /// Generate the SPE code file contents ///
  accelstr_spe_c << "#ifndef __ACCEL_" << name << "_C__\n"
                 << "#define __ACCEL_" << name << "_C__\n\n\n";
  int numAccelEntries = genAccels_spe_c_funcBodies(accelstr_spe_c);
  accelstr_spe_c << "\n\n#endif //__ACCEL_" << name << "_C__\n";

  /// Generate the SPE header file contents ///
  accelstr_spe_h << "#ifndef __ACCEL_" << name << "_H__\n"
                 << "#define __ACCEL_" << name << "_H__\n\n\n";
  genAccels_spe_h_includes(accelstr_spe_h);
  accelstr_spe_h << "\n\n";
  accelstr_spe_h << "#define MODULE_" << name << "_FUNC_INDEX_COUNT (" << numAccelEntries;
  genAccels_spe_h_fiCountDefs(accelstr_spe_h);
  accelstr_spe_h << ")\n\n\n";
  accelstr_spe_h << "#endif //__ACCEL_" << name << "_H__\n";

  #endif


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
  if (connectPresent == 1) {
    decl << pubDeclStr.charstar();
    def << pubDefConstr.charstar();
    def << pubDefStr.charstar();
  }

  // DMK - Accel Support
  #if CMK_CELL != 0

  /// Generate this module's code (actually create the files) ///
  XStr accelname_c, accelname_h;
  accelname_c << name << ".genSPECode.c";
  accelname_h << name << ".genSPECode.h";
  ofstream accel_c(accelname_c.get_string()), accel_h(accelname_h.get_string());
  if (!accel_c) {
    cerr << "Cannot open " << accelname_c.get_string() << " for writing!!\n";
    die("Cannot create output files (check directory permissions)\n");
  }
  if (!accel_h) {
    cerr << "Cannot open " << accelname_h.get_string() << " for writing!!\n";
    die("Cannot create output files (check directory permissions)\n");
  }
  accel_c << accelstr_spe_c.get_string();
  accel_h << accelstr_spe_h.get_string();
  
  // If this is the main module, generate the general C file and include this modules accel.h file
  if (isMain()) {

    XStr mainAccelStr_c;
    mainAccelStr_c << "#include \"main__funcLookup__.genSPECode.h" << "\"\n"
                   << "#include \"" << name << ".genSPECode.c" << "\"\n";
    ofstream mainAccel_c("main__funcLookup__.genSPECode.c");
    if (!mainAccel_c) {
      cerr << "Cannot open main__funcLookup__.genSPECode.c for writing!!\n";
      die("Cannot create output files (check directory permissions)");
    }
    mainAccel_c << mainAccelStr_c.get_string();

    XStr mainAccelStr_h;
    mainAccelStr_h << "#ifndef __MAIN_FUNCLOOKUP_H__\n"
                   << "#define __MAIN_FUNCLOOKUP_H__\n\n"
		   << "#include <spu_intrinsics.h>\n"
		   << "#include <stdlib.h>\n"
		   << "#include <stdio.h>\n"
		   << "#include \"spert.h\"\n\n"
		   << "#include \"simd.h\"\n"
                   << "#include \"" << name << ".genSPECode.h" << "\"\n\n"
                   << "#endif //__MAIN_FUNCLOOKUP_H__\n";
    ofstream mainAccel_h("main__funcLookup__.genSPECode.h");
    if (!mainAccel_h) {
      cerr << "Cannot open main__funcLookup__.genSPECode.h for writing!!\n";
      die("Cannot create output files (check directory permissions)");
    }
    mainAccel_h << mainAccelStr_h.get_string();

  }

  #endif
}

void
Module::preprocess()
{
  if (clist!=NULL) clist->preprocess();
}

void
Module::genDepend(const char *cifile)
{
  cout << name << ".decl.h " << name << ".def.h: "
       << cifile << ".stamp" << endl;
}

void
ModuleList::print(XStr& str)
{
    perElemGen(modules, str, &Module::print);
}

void
ModuleList::generate()
{
    for (list<Module*>::iterator i = modules.begin(); i != modules.end(); ++i)
	(*i)->generate();
}

void
ModuleList::preprocess()
{
    for (list<Module*>::iterator i = modules.begin(); i != modules.end(); ++i)
	(*i)->preprocess();
}

void
ModuleList::genDepends(std::string ciFileBaseName)
{
    perElemGen(modules, ciFileBaseName.c_str(), &Module::genDepend);
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
    perElemGen(members, c, &Member::setChare);
}

void
ConstructList::genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent)
{
    for (list<Construct*>::iterator i = constructs.begin(); 
	 i != constructs.end(); ++i)
	if (*i) {
	    (*i)->genPub(declstr, defstr, defconstr, connectPresent);
	    declstr << endx;
	}
}

void
ConstructList::genDecls(XStr& str)
{
    perElemGen(constructs, str, &Construct::genDecls, newLine);
}

void
ConstructList::genDefs(XStr& str)
{
    perElemGen(constructs, str, &Construct::genDefs, newLine);
}

void
ConstructList::genReg(XStr& str)
{
    perElemGen(constructs, str, &Construct::genReg, newLine);
}

void
ConstructList::preprocess()
{
    for (list<Construct*>::iterator i = constructs.begin(); 
	 i != constructs.end(); ++i)
	if (*i)
	    (*i)->preprocess();
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

XStr Chare::indexList()
{
  // generating "int *index1, int *index2, int *index3"
  XStr str;
  if (!isArray()) { // Currently, only arrays are supported
      cerr << (char *)baseName() << ": only chare arrays are currently supported\n";
      exit(1);
  }
  XStr dim = ((Array*)this)->dim();
  if (dim==(const char*)"1D")
    str << "const int *index1";
  else if (dim==(const char*)"2D")
    str << "const int *index1, const int *index2";
  else if (dim==(const char*)"3D")
    str << "const int *index1, const int *index2, const int *index3";
  else {
      cerr << (char *)baseName() << ": only up to 3 dimension chare arrays are currently supported\n";
      exit(1);
  }
  return str;
}

static const char *forWhomStr(forWhom w)
{
  switch(w) {
  case forAll: return Prefix::Proxy;
  case forIndividual: return Prefix::ProxyElement;
  case forSection: return Prefix::ProxySection;
  case forIndex: return Prefix::Index;
  case forPython: return "";
  default: return NULL;
  };
}

void NamedType::genProxyName(XStr& str,forWhom forElement)
{
    const char *prefix=forWhomStr(forElement);
    if (prefix==NULL)
        die("Unrecognized forElement type passed to NamedType::genProxyName");
    if (scope) str << scope;
    str << prefix;
    str << name;
    if (tparams) str << "<"<<tparams<<" >";
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
    (void)super;
    str<<"    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL) {\n";
    genProxyNames(str,"      ",NULL,"::ckDelegate(dTo,dPtr);\n","");
    str<<"    }\n";
    str<<"    void ckUndelegate(void) {\n";
    genProxyNames(str,"      ",NULL,"::ckUndelegate();\n","");
    str<<"    }\n";
    str<<"    void pup(PUP::er &p) {\n";
    genProxyNames(str,"      ",NULL,"::pup(p);\n","");
    str<<"    }\n";
    if (isPython()) {
      str<<"    void registerPython(const char *str) {\n";
      str<<"      CcsRegisterHandler(str, CkCallback("<<Prefix::Index<<type<<"::pyRequest(0), ";//<<Prefix::Proxy<<type<<"(";
      //if (isArray()) str<<"ckGetArrayID()";
      //else if (isGroup()) str <<"ckGetGroupID()";
      //else str<<"ckGetChareID()";
      str << "*this";
      str<<"));\n";
      str<<"    }\n";
    }
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
	hasSection=0;
	bases_CBase=NULL;
	setTemplate(0);
	hasSdagEntry=0;
	if (list)
	{
		list->setChare(this);
      		//Add migration constructor to MemberList
		if(isMigratable()) {
			Entry *e=new Entry(ln,SMIGRATE,NULL,
			  (char *)type->getBaseName(),
			  new ParamList(new Parameter(line,
				new PtrType(new NamedType("CkMigrateMessage")))),0,0,0);
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
  "  __idx = CkRegisterChare(s, size,";
  if (isMainChare()) str << " TypeMainChare";
  else if (isGroup()) str << " TypeGroup";
  else if (isNodeGroup()) str << " TypeNodeGroup";
  else if (isArray()) str << " TypeArray";
  else if (isChare()) str << " TypeChare";
  else str << " TypeInvalid";
  str << ");\n";
  if (internalMode) str << "  CkRegisterChareInCharm(__idx);\n";
  // register all bases
  genIndexNames(str, "  CkRegisterBase(__idx, ",NULL, "::__idx);\n", "");
  genSubRegisterMethodDef(str);
  if(list)
    list->genReg(str);
  if (hasSdagEntry) {
      str << "  " << baseName(0) << "::__sdag_register(); \n";
  }
  str << "}\n";
  str << "#endif\n";
}

void
Chare::genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent)
{
  if(type->isTemplated())
    return;
  else
  {
    if(list)
      list->genPub(declstr, defstr, defconstr, connectPresent);
  }
}

void
Chare::genDecls(XStr& str)
{
  if(type->isTemplated())
    return;
  str << "/* DECLS: "; print(str); str << " */\n";

  // include python header and add two methods called execute and iterate.
  // these cannot be added to the .ci file of the PythonCCS interface since
  // different charm object require different definitions...
  if (isPython()) {
    str << "#include \"PythonCCS.h\"\n";
    if (list) {
      Entry *etemp = new Entry(0,0,new BuiltinType("void"),"pyRequest",new ParamList(new Parameter(0,new PtrType(new NamedType("CkCcsRequestMsg",0)),"msg")),0,0,0,0);
      list->appendMember(etemp);
      etemp->setChare(this);
      //etemp = new Entry(0,0,new BuiltinType("void"),"getPrint",new ParamList(new Parameter(0,new PtrType(new NamedType("CkCcsRequestMsg",0)),"msg")),0,0,0,0);
      //list->appendMember(etemp);
      //etemp->setChare(this);
    }
  }

  //Forward declaration of the user-defined implementation class*/
  str << tspec()<<" class "<<type<<";\n";
  str << tspec()<<" class "<<Prefix::Index<<type<<";\n";
  str << tspec()<<" class "<<Prefix::Proxy<<type<<";\n";
  if (hasElement)
    str << tspec()<<" class "<<Prefix::ProxyElement<<type<<";\n";
  if (hasSection)
    str << tspec()<<" class "<<Prefix::ProxySection<<type<<";\n";
  if (isPython())
    str << tspec()<<" class "<<Prefix::Python<<type<<";\n";

 //Generate index class
  str << "/* --------------- index object ------------------ */\n";
  str << tspec()<< "class "<<Prefix::Index<<type;
  str << ":";
  genProxyNames(str, "public ",NULL, "", ", ");
  if(external || type->isTemplated())
  { //Just a template instantiation/forward declaration
    str << ";";
  }
  else
  { //Actual implementation
    str << CIClassStart;
    genTypedefs(str);
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
    forElement=forAll; genSubDecls(str); forElement=forIndividual;
  }
  if (hasSection) {
    str << "/* ---------------- section proxy -------------- */\n";
    forElement=forSection; genSubDecls(str); forElement=forIndividual;
  }
  if (isPython()) {
    str << "/* ---------------- python wrapper -------------- */\n";
    genPythonDecls(str);
  }

  if(list) {
    //handle the case that some of the entries may be sdag Entries
    int sdagPresent = 0;
    XStr sdagStr;
    CParsedFile *myParsedFile = new CParsedFile(this);
    list->collectSdagCode(myParsedFile, sdagPresent);
    if(sdagPresent) {
      XStr classname;
      XStr sdag_output;
      classname << baseName(0);
      resetNumbers();
      myParsedFile->doProcess(classname, sdag_output);
      str << sdag_output;
    }
  }

  // Create CBase_Whatever convenience type so that chare implementations can
  // avoid inheriting from a complex CBaseT templated type.
  TypeList *b=bases_CBase;
  if (b==NULL) b=bases; //Fall back to normal bases list if no CBase available
  if (templat) {
    templat->genSpec(str);
    str << "\nclass CBase_" << type << " : public ";
  } else {
    str << "typedef ";
  }
  str << "CBaseT" << b->length() << "<";
  if (isPython()) {
    str << Prefix::Python << type;
  } else {
    str << b;
  }
  str << ", CProxy_" << type;
  if (templat) {
    templat->genVars(str);
    str << " > { };\n";
  } else {
    str << "> CBase_" << type << ";\n";
  }
}

void
Chare::preprocess()
{
  if(list) list->preprocess();
}

/*This disambiguation code is needed to support
  multiple inheritance in Chares (Groups, Arrays).
  They resolve ambiguous accessor calls to the parent "super".
  Because mutator routines need to change *all* the base
  classes, mutators are generated in xi-symbol.C.
*/
static void
disambig_proxy(XStr &str, const XStr &super)
{
  str << "int ckIsDelegated(void) const "
      << "{return " << super << "::ckIsDelegated();}\n"
      << "inline CkDelegateMgr *ckDelegatedTo(void) const "
      << "{return " << super << "::ckDelegatedTo();}\n"
      << "inline CkDelegateData *ckDelegatedPtr(void) const "
      << "{return " << super << "::ckDelegatedPtr();}\n"
      << "CkGroupID ckDelegatedIdx(void) const "
      << "{return " << super << "::ckDelegatedIdx();}\n";
}

void
Chare::genSubDecls(XStr& str)
{
  XStr ptype;
  ptype<<proxyPrefix()<<type;

  // Class declaration
  str << tspec()<< "class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << ":";
  genProxyNames(str, "public ",NULL, "", ", ");
  str << CIClassStart;

  genTypedefs(str);

  // Various constructors:
  str << "    "<<ptype<<"(void) {};\n";

  str << "    "<<ptype<<"(CkChareID __cid) : ";
  genProxyNames(str, "",NULL, "(__cid)", ", ");
  str << "{  }\n";

  str << "    "<<ptype<<"(const Chare *c) : ";
  genProxyNames(str, "",NULL, "(c)", ", ");
  str << "{  }\n";

  //Multiple inheritance-- resolve inheritance ambiguity
    XStr super;
    bases->getFirst()->genProxyName(super,forElement);
    disambig_proxy(str, super);
    str << "inline void ckCheck(void) const {" << super << "::ckCheck();}\n"
	<< "const CkChareID &ckGetChareID(void) const\n"
	<< "{ return " << super << "::ckGetChareID(); }\n"
	<< "operator const CkChareID &(void) const {return ckGetChareID();}\n";

    sharedDisambiguation(str,super);
    str<<"    void ckSetChareID(const CkChareID &c) {\n";
    genProxyNames(str,"      ",NULL,"::ckSetChareID(c);\n","");
    str<<"    }\n";

  str<<"    "<<type<<tvars()<<" *ckLocal(void) const\n";
  str<<"     { return ("<<type<<tvars()<<" *)CkLocalChare(&ckGetChareID()); }\n";

  if(list)
    list->genDecls(str);
  str << CIClassEnd;
  if (!isTemplated()) str << "PUPmarshall("<<ptype<<")\n";
}

void Chare::genPythonDecls(XStr& str) {

  XStr ptype;
  ptype<<Prefix::Python<<type;

  // Class declaration
  str << tspec()<< "class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << ":";
  TypeList *b=bases_CBase;
  if (b==NULL) b=bases; //Fall back to normal bases list if no CBase available
  b->genProxyNames(str,"public ",NULL,"",", ",forPython);
  str << ", public PythonObject ";
  str << CIClassStart;

  // default constructor methods
  str << "    "<<ptype<<"(void) {}\n";
  str << "    "<<ptype<<"(CkMigrateMessage *msg): ";
  b->genProxyNames(str,"",NULL,"(msg)",", ",forPython);
  str << " {}\n";

  // define pupper
  str << "    void pup(PUP::er &p) {\n";
  b->genProxyNames(str,"      ",NULL,"::pup(p);","\n",forPython);
  str << "\n    }\n";

  // define the python custom methods and their documentation
  str << "    static PyMethodDef CkPy_MethodsCustom[];\n";
  str << "    PyMethodDef *getMethods(void) {return CkPy_MethodsCustom;}\n";
  str << "    static const char *CkPy_MethodsCustomDoc;\n";
  str << "    const char *getMethodsDoc(void) {return CkPy_MethodsCustomDoc;}\n";

  str << CIClassEnd;

  // declare all static python methods and CkPy_MethodsCustom
  if (list)
    list->genPythonDecls(str);
  str << "\n";

  if (!isTemplated()) str << "PUPmarshall("<<ptype<<")\n";
}

void Chare::genPythonDefs(XStr& str) {

  XStr ptype;
  ptype<<Prefix::Python<<type;

  // generate the python methods array
  str << "PyMethodDef "<<ptype<<"::CkPy_MethodsCustom[] = {\n";
  if (list)
    list->genPythonStaticDefs(str);
  str << "  {NULL, NULL}\n};\n\n";
  // generate documentaion for the methods
  str << "const char * "<<ptype<<"::CkPy_MethodsCustomDoc = \"charm.__doc__ = \\\"Available methods for object "<<type<<":\\\\n\"";
  if (list)
    list->genPythonStaticDocs(str);
  str << "\n  \"\\\"\";\n\n";

  if (list)
    list->genPythonDefs(str);

}

Group::Group(int ln, attrib_t Nattr,
    	NamedType *t, TypeList *b, MemberList *l)
    	:Chare(ln,Nattr|CGROUP,t,b,l)
{
        hasElement=1;
	forElement=forIndividual;
	hasSection=1;
	bases_CBase=NULL;
	if (b==NULL) {//Add Group as a base class
		if (isNodeGroup())
			bases = new TypeList(new NamedType("NodeGroup"), NULL);
		else {
			bases = new TypeList(new NamedType("IrrGroup"), NULL);
			bases_CBase = new TypeList(new NamedType("Group"), NULL);
		}
	}
}

void Group::genSubRegisterMethodDef(XStr& str) {
        if(!isTemplated()){
                str << "   CkRegisterGroupIrr(__idx,"<<type<<"::isIrreducible());\n";
        }else{
                str << "   CkRegisterGroupIrr(__idx," <<type<<tvars() <<"::isIrreducible());\n";
        }
}

static void
disambig_reduction_client(XStr &str, const XStr &super)
{
  str << "inline void setReductionClient(CkReductionClientFn fn,void *param=NULL) const\n"
      << "{ " << super << "::setReductionClient(fn,param); }\n"
      << "inline void ckSetReductionClient(CkReductionClientFn fn,void *param=NULL) const\n"
      << "{ " << super << "::ckSetReductionClient(fn,param); }\n"
      << "inline void ckSetReductionClient(CkCallback *cb) const\n"
      << "{ " << super << "::ckSetReductionClient(cb); }\n";
}

static void
disambig_group(XStr &str, const XStr &super)
{
  disambig_proxy(str, super);
  str << "inline void ckCheck(void) const {" << super << "::ckCheck();}\n"
      << "CkChareID ckGetChareID(void) const\n"
      << "   {return " << super << "::ckGetChareID();}\n"
      << "CkGroupID ckGetGroupID(void) const\n"
      << "   {return " << super << "::ckGetGroupID();}\n"
      << "operator CkGroupID () const { return ckGetGroupID(); }\n";
  disambig_reduction_client(str, super);
}

void
Group::genSubDecls(XStr& str)
{
  XStr ptype; ptype<<proxyPrefix()<<type;
  XStr ttype; ttype<<type<<tvars();
  XStr super;
  bases->getFirst()->genProxyName(super,forElement);

  // Class declaration:
  str << tspec()<< "class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << ": ";
  genProxyNames(str, "public ",NULL, "", ", ");
  str << CIClassStart;

  genTypedefs(str);

  // Basic constructors:
  str << "    "<<ptype<<"(void) {}\n";
  str << "    "<<ptype<<"(const IrrGroup *g) : ";
  genProxyNames(str, "", NULL,"(g)", ", ");
  str << "{  }\n";

  if (forElement==forIndividual)
  {//For a single element
    str << "    "<<ptype<<"(CkGroupID _gid,int _onPE,CK_DELCTOR_PARAM) : ";
    genProxyNames(str, "", NULL,"(_gid,_onPE,CK_DELCTOR_ARGS)", ", ");
    str << "{  }\n";
    str << "    "<<ptype<<"(CkGroupID _gid,int _onPE) : ";
    genProxyNames(str, "", NULL,"(_gid,_onPE)", ", ");
    str << "{  }\n";

    disambig_group(str, super);
    str << "int ckGetGroupPe(void) const\n"
	<< "{return " << super << "::ckGetGroupPe();}\n";

  }
  else if (forElement==forSection)
  {//For a section of the group
    str << "    "<<ptype<<"(const CkGroupID &_gid,const int *_pelist,int _npes,CK_DELCTOR_PARAM) : ";
    genProxyNames(str, "", NULL,"(_gid,_pelist,_npes,CK_DELCTOR_ARGS)", ", ");
    str << "{  }\n";
    str << "    "<<ptype<<"(const CkGroupID &_gid,const int *_pelist,int _npes) : ";
    genProxyNames(str, "", NULL,"(_gid,_pelist,_npes)", ", ");
    str << "{  }\n";
    str << "    "<<ptype<<"(int n,const CkGroupID *_gid, int const * const *_pelist,const int *_npes) : ";
    genProxyNames(str, "", NULL,"(n,_gid,_pelist,_npes)", ", ");
    str << "{  }\n";
    str << "    "<<ptype<<"(int n,const CkGroupID *_gid, int const * const *_pelist,const int *_npes,CK_DELCTOR_PARAM) : ";
    genProxyNames(str, "", NULL,"(n,_gid,_pelist,_npes,CK_DELCTOR_ARGS)", ", ");
    str << "{  }\n";
    
    disambig_group(str, super);
    str << "inline int ckGetNumSections() const\n" <<
      "{ return " << super << "::ckGetNumSections(); }\n" <<
      "inline CkSectionInfo &ckGetSectionInfo()\n" <<
      "{ return " << super << "::ckGetSectionInfo(); }\n" <<
      "inline CkSectionID *ckGetSectionIDs()\n" <<
      "{ return " << super << "::ckGetSectionIDs(); }\n" <<
      "inline CkSectionID &ckGetSectionID()\n" <<
      "{ return " << super << "::ckGetSectionID(); }\n" <<
      "inline CkSectionID &ckGetSectionID(int i)\n" <<
      "{ return " << super << "::ckGetSectionID(i); }\n" <<
      "inline CkGroupID ckGetGroupIDn(int i) const\n" <<
      "{ return " << super << "::ckGetGroupIDn(i); }\n" <<
      "inline int *ckGetElements() const\n" <<
      "{ return " << super << "::ckGetElements(); }\n" <<
      "inline int *ckGetElements(int i) const\n" <<
      "{ return " << super << "::ckGetElements(i); }\n" <<
      "inline int ckGetNumElements() const\n" <<
      "{ return " << super << "::ckGetNumElements(); } \n" <<
      "inline int ckGetNumElements(int i) const\n" <<
      "{ return " << super << "::ckGetNumElements(i); }\n";
  }
  else if (forElement==forAll)
  {//For whole group
    str << "    "<<ptype<<"(CkGroupID _gid,CK_DELCTOR_PARAM) : ";
    genProxyNames(str, "", NULL,"(_gid,CK_DELCTOR_ARGS)", ", ");
    str << "{  }\n";
    str << "    "<<ptype<<"(CkGroupID _gid) : ";
    genProxyNames(str, "", NULL,"(_gid)", ", ");
    str << "{  }\n";

    //Group proxy can be indexed into an element proxy:
    forElement=forIndividual;//<- for the proxyName below
    str << "    "<<proxyName(1)<<" operator[](int onPE) const\n";
    str << "      {return "<<proxyName(1)<<"(ckGetGroupID(),onPE,CK_DELCTOR_CALL);}\n";
    forElement=forAll;

    disambig_group(str, super);
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
  if (!isTemplated()) str << "PUPmarshall("<<ptype<<")\n";

}

XStr indexSuffix2object(const XStr &indexSuffix) {
	if (indexSuffix==(const char*)"1D") return "CkIndex1D";
	if (indexSuffix==(const char*)"2D") return "CkIndex2D";
	if (indexSuffix==(const char*)"3D") return "CkIndex3D";
	if (indexSuffix==(const char*)"4D") return "CkIndex4D";
	if (indexSuffix==(const char*)"5D") return "CkIndex5D";
	if (indexSuffix==(const char*)"6D") return "CkIndex6D";
	if (indexSuffix==(const char*)"Max") return "CkIndexMax";
	else return indexSuffix;
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
      //printf("indexSuffix = %s\n", indexSuffix.charstar());
	if (indexSuffix!=(const char*)"none")
		indexType<<"CkArrayIndex"<<indexSuffix;
	else indexType<<"CkArrayIndex";

	if(b==0) { //No other base class:
		if (0==strcmp(type->getBaseName(),"ArrayElement"))
			//ArrayElement has special "ArrayBase" superclass
			bases = new TypeList(new NamedType("ArrayBase"), NULL);
		else {//Everybody else inherits from ArrayElementT<indexType>
			bases=new TypeList(new NamedType("ArrayElement"),NULL);
			XStr indexObject(indexSuffix2object(indexSuffix));
			XStr parentClass;
			parentClass<<"ArrayElementT<"<<indexObject<<">";
			char *parentClassName=strdup(parentClass);
			bases_CBase = new TypeList(new NamedType(parentClassName), NULL);
		}
	}
}

static void
disambig_array(XStr &str, const XStr &super)
{
  disambig_proxy(str, super);
  str << "inline void ckCheck(void) const {" << super << "::ckCheck();}\n" <<
    "inline operator CkArrayID () const {return ckGetArrayID();}\n" <<
    "inline static CkArrayID ckCreateEmptyArray(void)" <<
    "{ return " << super << "::ckCreateEmptyArray(); }\n" <<
    "inline static CkArrayID ckCreateArray(CkArrayMessage *m,int ctor,const CkArrayOptions &opts)" <<
    "{ return " << super << "::ckCreateArray(m,ctor,opts); }\n" <<
    "inline void ckInsertIdx(CkArrayMessage *m,int ctor,int onPe,const CkArrayIndex &idx)" <<
    "{ " << super << "::ckInsertIdx(m,ctor,onPe,idx); }\n" <<
    "inline void ckBroadcast(CkArrayMessage *m, int ep, int opts=0) const" <<
    "{ " << super << "::ckBroadcast(m,ep,opts); }\n" <<
    "inline CkArrayID ckGetArrayID(void) const" <<
    "{ return " << super << "::ckGetArrayID();}\n" <<
    "inline CkArray *ckLocalBranch(void) const" <<
    "{ return " << super << "::ckLocalBranch(); }\n" <<
    "inline CkLocMgr *ckLocMgr(void) const" <<
    "{ return " << super << "::ckLocMgr(); }\n" <<
    "inline void doneInserting(void) { " << super << "::doneInserting(); }\n";
  disambig_reduction_client(str, super);
}

void
Array::genSubDecls(XStr& str)
{
  XStr ptype; ptype<<proxyPrefix()<<type;

  // Class declaration:
  str << tspec()<< " class "<<ptype;
  if(external || type->isTemplated()) {
    str << ";";
    return;
  }
  str << " : ";
  genProxyNames(str, "public ",NULL, "", ", ");
  str << CIClassStart;

  genTypedefs(str);

  str << "    "<<ptype<<"(void) {}\n";//An empty constructor
  if (forElement!=forSection)
  { //Generate constructor based on array element
	  str << "    "<<ptype<<"(const ArrayElement *e) : ";
    genProxyNames(str, "", NULL,"(e)", ", ");
    str << "{  }\n";
  }

  //Resolve multiple inheritance ambiguity
  XStr super;
  bases->getFirst()->genProxyName(super,forElement);
  sharedDisambiguation(str,super);

  if (forElement==forIndividual)
  {/*For an individual element (no indexing)*/
    disambig_array(str, super);
    str << "inline void ckInsert(CkArrayMessage *m,int ctor,int onPe)\n"
	<< "  { " << super << "::ckInsert(m,ctor,onPe); }\n"
	<< "inline void ckSend(CkArrayMessage *m, int ep, int opts = 0) const\n"
	<< "  { " << super << "::ckSend(m,ep,opts); }\n"
	<< "inline void *ckSendSync(CkArrayMessage *m, int ep) const\n"
	<< "  { return " << super << "::ckSendSync(m,ep); }\n"
	<< "inline const CkArrayIndex &ckGetIndex() const\n"
	<< "  { return " << super << "::ckGetIndex(); }\n";

    str << "    "<<type<<tvars()<<" *ckLocal(void) const\n";
    str << "      { return ("<<type<<tvars()<<" *)"<<super<<"::ckLocal(); }\n";
    //This constructor is used for array indexing
    str <<
         "    "<<ptype<<"(const CkArrayID &aid,const "<<indexType<<" &idx,CK_DELCTOR_PARAM)\n"
         "        :";genProxyNames(str, "",NULL, "(aid,idx,CK_DELCTOR_ARGS)", ", ");str<<" {}\n";
    str <<
         "    "<<ptype<<"(const CkArrayID &aid,const "<<indexType<<" &idx)\n"
         "        :";genProxyNames(str, "",NULL, "(aid,idx)", ", ");str<<" {}\n";
  }
  else if (forElement==forAll)
  {/*Collective, indexible version*/
    disambig_array(str, super);

    str<< //Build a simple, empty array
    "    static CkArrayID ckNew(void) {return ckCreateEmptyArray();}\n";

    XStr etype; etype<<Prefix::ProxyElement<<type<<tvars();
    if (indexSuffix!=(const char*)"none")
    {
      str <<
    "//Generalized array indexing:\n"
    "    "<<etype<<" operator [] (const "<<indexType<<" &idx) const\n"
    "        {return "<<etype<<"(ckGetArrayID(), idx, CK_DELCTOR_CALL);}\n"
    "    "<<etype<<" operator() (const "<<indexType<<" &idx) const\n"
    "        {return "<<etype<<"(ckGetArrayID(), idx, CK_DELCTOR_CALL);}\n";
    }

  //Add specialized indexing for these common types
    if (indexSuffix==(const char*)"1D")
    {
    str << "    " << etype << " operator [] (int idx) const \n"
	<< "        {return "<< etype <<"(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}\n"
	<< "    " << etype <<" operator () (int idx) const \n"
	<< "        {return "<< etype <<"(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}\n";
    } else if (indexSuffix==(const char*)"2D") {
    str <<
    "    "<<etype<<" operator () (int i0,int i1) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex2D(i0,i1), CK_DELCTOR_CALL);}\n"
    "    "<<etype<<" operator () (CkIndex2D idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex2D(idx), CK_DELCTOR_CALL);}\n";
    } else if (indexSuffix==(const char*)"3D") {
    str <<
    "    "<<etype<<" operator () (int i0,int i1,int i2) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex3D(i0,i1,i2), CK_DELCTOR_CALL);}\n"
    "    "<<etype<<" operator () (CkIndex3D idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex3D(idx), CK_DELCTOR_CALL);}\n";
    } else if (indexSuffix==(const char*)"4D") {
    str <<
    "    "<<etype<<" operator () (short int i0,short int i1,short int i2,short int i3) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex4D(i0,i1,i2,i3), CK_DELCTOR_CALL);}\n"
    "    "<<etype<<" operator () (CkIndex4D idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex4D(idx), CK_DELCTOR_CALL);}\n";
    } else if (indexSuffix==(const char*)"5D") {
    str <<
    "    "<<etype<<" operator () (short int i0,short int i1,short int i2,short int i3,short int i4) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex5D(i0,i1,i2,i3,i4), CK_DELCTOR_CALL);}\n"
    "    "<<etype<<" operator () (CkIndex5D idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex5D(idx), CK_DELCTOR_CALL);}\n";
    } else if (indexSuffix==(const char*)"6D") {
    str <<
    "    "<<etype<<" operator () (short int i0,short int i1,short int i2,short int i3,short int i4,short int i5) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex6D(i0,i1,i2,i3,i4,i5), CK_DELCTOR_CALL);}\n"
    "    "<<etype<<" operator () (CkIndex6D idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), CkArrayIndex6D(idx), CK_DELCTOR_CALL);}\n";
    }
    str <<"    "<<ptype<<"(const CkArrayID &aid,CK_DELCTOR_PARAM) \n"
         "        :";genProxyNames(str, "",NULL, "(aid,CK_DELCTOR_ARGS)", ", ");str<<" {}\n";
    str <<"    "<<ptype<<"(const CkArrayID &aid) \n"
         "        :";genProxyNames(str, "",NULL, "(aid)", ", ");str<<" {}\n";
  }
  else if (forElement==forSection)
  { /* for Section, indexible version*/
    disambig_array(str, super);
    str << "inline void ckSend(CkArrayMessage *m, int ep, int opts = 0)\n"
	<< " { " << super << "::ckSend(m,ep,opts); }\n"
	<< "inline CkSectionInfo &ckGetSectionInfo()\n"
	<< "  { return " << super << "::ckGetSectionInfo(); }\n"
	<< "inline CkSectionID *ckGetSectionIDs()\n"
	<< "  { return " << super << "::ckGetSectionIDs(); }\n"
	<< "inline CkSectionID &ckGetSectionID()\n"
	<< "  { return " << super << "::ckGetSectionID(); }\n"
	<< "inline CkSectionID &ckGetSectionID(int i)\n"
	<< "  { return " << super << "::ckGetSectionID(i); }\n"
	<< "inline CkArrayID ckGetArrayIDn(int i) const\n"
	<< "{return " << super << "::ckGetArrayIDn(i); } \n"
	<< "inline CkArrayIndex *ckGetArrayElements() const\n"
	<< "  { return " << super << "::ckGetArrayElements(); }\n"
	<< "inline CkArrayIndex *ckGetArrayElements(int i) const\n"
	<< "{return " << super << "::ckGetArrayElements(i); }\n"
	<< "inline int ckGetNumElements() const\n"
	<< "  { return " << super << "::ckGetNumElements(); } \n"
	<< "inline int ckGetNumElements(int i) const\n"
	<< "{return " << super << "::ckGetNumElements(i); } \n";

    XStr etype; etype<<Prefix::ProxyElement<<type<<tvars();
    if (indexSuffix!=(const char*)"none")
    {
      str <<
    "//Generalized array indexing:\n"
    "    "<<etype<<" operator [] (const "<<indexType<<" &idx) const\n"
    "        {return "<<etype<<"(ckGetArrayID(), idx, CK_DELCTOR_CALL);}\n"
    "    "<<etype<<" operator() (const "<<indexType<<" &idx) const\n"
    "        {return "<<etype<<"(ckGetArrayID(), idx, CK_DELCTOR_CALL);}\n";
    }

  //Add specialized indexing for these common types
    if (indexSuffix==(const char*)"1D")
    {
    str <<
    "    "<<etype<<" operator [] (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}\n"
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}\n"
    "    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex1D *elems, int nElems) {\n"
    "      return CkSectionID(aid, elems, nElems);\n"
    "    } \n"
    "    static CkSectionID ckNew(const CkArrayID &aid, int l, int u, int s) {\n"
    "      CkVec<CkArrayIndex1D> al;\n"
    "      for (int i=l; i<=u; i+=s) al.push_back(CkArrayIndex1D(i));\n"
    "      return CkSectionID(aid, al.getVec(), al.size());\n"
    "    } \n";
    } else if (indexSuffix==(const char*)"2D") {
    str <<
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex2D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}\n"
    "    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex2D *elems, int nElems) {\n"
    "      return CkSectionID(aid, elems, nElems);\n"
    "    } \n"
    "    static CkSectionID ckNew(const CkArrayID &aid, int l1, int u1, int s1, int l2, int u2, int s2) {\n"
    "      CkVec<CkArrayIndex2D> al;\n"
    "      for (int i=l1; i<=u1; i+=s1) \n"
    "        for (int j=l2; j<=u2; j+=s2) \n"
    "          al.push_back(CkArrayIndex2D(i, j));\n"
    "      return CkSectionID(aid, al.getVec(), al.size());\n"
    "    } \n";
    } else if (indexSuffix==(const char*)"3D") {
    str <<
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex3D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}\n"
    "    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex3D *elems, int nElems) {\n"
    "      return CkSectionID(aid, elems, nElems);\n"
    "    } \n"
    "    static CkSectionID ckNew(const CkArrayID &aid, int l1, int u1, int s1, int l2, int u2, int s2, int l3, int u3, int s3) {\n"
    "      CkVec<CkArrayIndex3D> al;\n"
    "      for (int i=l1; i<=u1; i+=s1) \n"
    "        for (int j=l2; j<=u2; j+=s2) \n"
    "          for (int k=l3; k<=u3; k+=s3) \n"
    "          al.push_back(CkArrayIndex3D(i, j, k));\n"
    "      return CkSectionID(aid, al.getVec(), al.size());\n"
    "    } \n";
    } else if (indexSuffix==(const char*)"4D") {
    str <<
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex4D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}\n"
    "    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex4D *elems, int nElems) {\n"
    "      return CkSectionID(aid, elems, nElems);\n"
    "    } \n"
    "    static CkSectionID ckNew(const CkArrayID &aid, int l1, int u1, int s1, int l2, int u2, int s2, int l3, int u3, int s3, int l4, int u4, int s4) {\n"
    "      CkVec<CkArrayIndex4D> al;\n"
    "      for (int i=l1; i<=u1; i+=s1) \n"
    "        for (int j=l2; j<=u2; j+=s2) \n"
    "          for (int k=l3; k<=u3; k+=s3) \n"
    "            for (int l=l4; l<=u4; l+=s4) \n"
    "              al.push_back(CkArrayIndex4D(i, j, k, l));\n"
    "      return CkSectionID(aid, al.getVec(), al.size());\n"
    "    } \n";
    } else if (indexSuffix==(const char*)"5D") {
    str <<
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex5D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}\n"
    "    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex5D *elems, int nElems) {\n"
    "      return CkSectionID(aid, elems, nElems);\n"
    "    } \n"
    "    static CkSectionID ckNew(const CkArrayID &aid, int l1, int u1, int s1, int l2, int u2, int s2, int l3, int u3, int s3, int l4, int u4, int s4, int l5, int u5, int s5) {\n"
    "      CkVec<CkArrayIndex5D> al;\n"
    "      for (int i=l1; i<=u1; i+=s1) \n"
    "        for (int j=l2; j<=u2; j+=s2) \n"
    "          for (int k=l3; k<=u3; k+=s3) \n"
    "            for (int l=l4; l<=u4; l+=s4) \n"
    "              for (int m=l5; m<=u5; m+=s5) \n"
    "                al.push_back(CkArrayIndex5D(i, j, k, l, m));\n"
    "      return CkSectionID(aid, al.getVec(), al.size());\n"
    "    } \n";
    } else if (indexSuffix==(const char*)"6D") {
    str <<
    "    "<<etype<<" operator () (int idx) const \n"
    "        {return "<<etype<<"(ckGetArrayID(), *(CkArrayIndex6D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}\n"
    "    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex6D *elems, int nElems) {\n"
    "      return CkSectionID(aid, elems, nElems);\n"
    "    } \n"
    "    static CkSectionID ckNew(const CkArrayID &aid, int l1, int u1, int s1, int l2, int u2, int s2, int l3, int u3, int s3, int l4, int u4, int s4, int l5, int u5, int s5, int l6, int u6, int s6) {\n"
    "      CkVec<CkArrayIndex6D> al;\n"
    "      for (int i=l1; i<=u1; i+=s1) \n"
    "        for (int j=l2; j<=u2; j+=s2) \n"
    "          for (int k=l3; k<=u3; k+=s3) \n"
    "            for (int l=l4; l<=u4; l+=s4) \n"
    "              for (int m=l5; m<=u5; m+=s5) \n"
    "                for (int n=l6; n<=u6; n+=s6) \n"
    "                  al.push_back(CkArrayIndex6D(i, j, k, l, m, n));\n"
    "      return CkSectionID(aid, al.getVec(), al.size());\n"
    "    } \n";
    }

    str <<"    "<<ptype<<"(const CkArrayID &aid, CkArrayIndex *elems, int nElems, CK_DELCTOR_PARAM) \n"
         "        :";genProxyNames(str, "",NULL, "(aid,elems,nElems,CK_DELCTOR_ARGS)", ", ");str << " {}\n";
    str <<"    "<<ptype<<"(const CkArrayID &aid, CkArrayIndex *elems, int nElems) \n"
         "        :";genProxyNames(str, "",NULL, "(aid,elems,nElems)", ", ");str<<" {}\n";
    str <<"    "<<ptype<<"(const CkSectionID &sid)"
	  "       :";genProxyNames(str, "",NULL, "(sid)", ", ");str<< " {}\n";
	str <<"    "<<ptype<<"(int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems, CK_DELCTOR_PARAM) \n"
	  "        :";genProxyNames(str, "",NULL, "(n,aid,elems,nElems,CK_DELCTOR_ARGS)", ", ");str << " {}\n";
	str <<"    "<<ptype<<"(int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems) \n"
	  "        :";genProxyNames(str, "",NULL, "(n,aid,elems,nElems)", ", ");str<<" {}\n";
    str <<
    "    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex *elems, int nElems) {\n"
    "      return CkSectionID(aid, elems, nElems);\n"
    "    } \n";
  }

  if(list){
    list->genDecls(str);
  }
  str << CIClassEnd;
  if (!isTemplated()) str << "PUPmarshall("<<ptype<<")\n";
}

void
Chare::genTypedefs(XStr &str) {
   str << "    typedef "<<baseName(1)<<" local_t;\n";
   str << "    typedef "<<Prefix::Index<<baseName(1)<<" index_t;\n";
   str << "    typedef "<<Prefix::Proxy<<baseName(1)<<" proxy_t;\n";

   if (hasElement)
     str << "    typedef "<<Prefix::ProxyElement<<baseName(1)<<" element_t;\n";
   else /* !hasElement, so generic proxy is element type */
     str << "    typedef "<<Prefix::Proxy<<baseName(1)<<" element_t;\n";

   if (hasSection)
     str << "    typedef "<<Prefix::ProxySection<<baseName(1)<<" section_t;\n";
   str << "\n";
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
    str << "extern \"C\" void " << fortranify(baseName(), "_allocate") << "(char **, void *, " << indexList() << ");\n";
    str << "extern \"C\" void " << fortranify(baseName(), "_pup") << "(pup_er p, char **, void *);\n";
    str << "extern \"C\" void " << fortranify(baseName(), "_resumefromsync") << "(char **, void *, " << indexList() << ");\n";
    str << "\n";
    XStr dim = ((Array*)this)->dim();
    str << "class " << baseName() << " : public ArrayElement" << dim << "\n";
    str << "{\n";
    str << "public:\n";
    str << "  char user_data[64];\n";
    str << "public:\n";
    str << "  " << baseName() << "()\n";
    str << "  {\n";
//    str << "    CkPrintf(\"" << baseName() << " %d created\\n\",thisIndex);\n";
    str << "    CkArrayID *aid = &thisArrayID;\n";
    if (dim==(const char*)"1D")
      str << "    " << fortranify(baseName(), "_allocate") << "((char **)&user_data, &aid, &thisIndex);\n";
    else if (dim==(const char*)"2D")
      str << "    " << fortranify(baseName(), "_allocate") << "((char **)&user_data, &aid, &thisIndex.x, &thisIndex.y);\n";
    else if (dim==(const char*)"3D")
      str << "    " << fortranify(baseName(), "_allocate") << "((char **)&user_data, &aid, &thisIndex.x, &thisIndex.y, &thisIndex.z);\n";
    str << "      usesAtSync = CmiTrue;\n";
    str << "  }\n";
    str << "\n";
    str << "  " << baseName() << "(CkMigrateMessage *m)\n";
    str << "  {\n";
    str << "    /* CkPrintf(\"" << baseName() << " %d migrating\\n\",thisIndex);*/ \n";
    str << "  }\n";
    str << "\n";

    str << "  virtual void pup(PUP::er &p)\n";
    str << "  {\n";
    str << "    ArrayElement" << dim << "::pup(p);\n";
    str << "    p(user_data, 64);\n";
    str << "    CkArrayID *aid = &thisArrayID;\n";
    str << "    ::" << fortranify(baseName(), "_pup") << "(&p, (char **)&user_data, &aid); \n";
    str << "  }\n";
    str << "\n";

      // Define the Fortran interface function for ResumeFromSync
    str << "  void ResumeFromSync()\n";
    str << "  {\n";
    str << "    CkArrayID *aid = &thisArrayID;\n";
    str << "    ::" << fortranify(baseName(), "_resumefromSync");
    if (dim == (const char*)"1D") {
      str << "((char **)&user_data, &aid, &thisIndex);\n";
    }
    else if (dim == (const char*)"2D") {
      str << "((char **)&user_data, &aid, &thisIndex.x, &thisIndex.y);\n";
    }
    else if (dim == (const char*)"3D") {
      str << "((char **)&user_data, &aid, &thisIndex.x, &thisIndex.y, &thisIndex.z);\n";
    }
    str << "  }\n";

    str << "};\n";
    str << "\n";
    if (dim==(const char*)"1D") {
      str << "extern \"C\" void " << fortranify(baseName(), "_cknew") << "(int *numElem, long *aindex)\n";
      str << "{\n";
      str << "    CkArrayID *aid = new CkArrayID;\n";
      str << "    *aid = CProxy_" << baseName() << "::ckNew(*numElem); \n";
    }
    else if (dim==(const char*)"2D") {
      str << "extern \"C\" void " << fortranify(baseName(), "_cknew") << "(int *numx, int *numy, long *aindex)\n";
      str << "{\n";
      str << "    CkArrayID *aid = new CkArrayID;\n";
      str << "    *aid = CProxy_" << baseName() << "::ckNew(); \n";
      str << "    CProxy_" << baseName() << " p(*aid);\n";
      str << "    for (int i=0; i<*numx; i++) \n";
      str << "      for (int j=0; j<*numy; j++) \n";
      str << "        p[CkArrayIndex2D(i, j)].insert(); \n";
      str << "    p.doneInserting(); \n";
    }
    else if (dim==(const char*)"3D") {
      str << "extern \"C\" void " << fortranify(baseName(), "_cknew") << "(int *numx, int *numy, int *numz, long *aindex)\n";
      str << "{\n";
      str << "    CkArrayID *aid = new CkArrayID;\n";
      str << "    *aid = CProxy_" << baseName() << "::ckNew(); \n";
      str << "    CProxy_" << baseName() << " p(*aid);\n";
      str << "    for (int i=0; i<*numx; i++) \n";
      str << "      for (int j=0; j<*numy; j++) \n";
      str << "        for (int k=0; k<*numz; k++) \n";
      str << "          p[CkArrayIndex3D(i, j, k)].insert(); \n";
      str << "    p.doneInserting(); \n";
    }
    str << "    *aindex = (long)aid;\n";
    str << "}\n";

      // Define the Fortran interface function for AtSync
    if (dim == (const char*)"1D") {
      str << "extern \"C\" void "
          << fortranify(baseName(), "_atsync")
          << "(long* aindex, int *index1)\n";
      str << "{\n";
      str << "  CkArrayID *aid = (CkArrayID *)*aindex;\n";
      str << "\n";
      str << "  CProxy_" << baseName() << " h(*aid);\n";
      str << "  h[*index1].ckLocal()->AtSync();\n";
    }
    else if (dim == (const char*)"2D") {
      str << "extern \"C\" void "
          << fortranify(baseName(), "_atsync")
          << "(long* aindex, int *index1, int *index2)\n";
      str << "{\n";
      str << "  CkArrayID *aid = (CkArrayID *)*aindex;\n";
      str << "\n";
      str << "  CProxy_" << baseName() << " h(*aid);\n";
      str << "  h[CkArrayIndex2D(*index1, *index2)].ckLocal()->AtSync();\n";
    }
    else if (dim == (const char*)"3D") {
      str << "extern \"C\" void "
          << fortranify(baseName(), "_atsync")
          << "(long* aindex, int *index1, int *index2, int *index3)\n";
      str << "{\n";
      str << "  CkArrayID *aid = (CkArrayID *)*aindex;\n";
      str << "\n";
      str << "  CProxy_" << baseName() << " h(*aid);\n";
      str << "  h[CkArrayIndex3D(*index1, *index2, *index3)].ckLocal()->AtSync();\n";
    }
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
  // define the python routines
  if (isPython()) {
    if(isTemplated())
      str << "#ifdef CK_TEMPLATES_ONLY\n";
    else
      str << "#ifndef CK_TEMPLATES_ONLY\n";
    str << "/* ---------------- python wrapper -------------- */\n";

    // write CkPy_MethodsCustom
    genPythonDefs(str);

    str << "#endif /*CK_TEMPLATES_ONLY*/\n";
  }

  if(!external && !type->isTemplated())
    genRegisterMethodDef(str);
  if (hasSdagEntry) {
    str << "\n";
    str << baseName(0) << "_SDAG_CODE_DEF\n\n";
  }
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
"    void* operator new(size_t, void*p) { return p; }\n"
"    void* operator new(size_t);\n"
"    void* operator new(size_t, int*, const int);\n"
"    void* operator new(size_t, int*);\n"
"#if CMK_MULTIPLE_DELETE\n"
"    void operator delete(void*p, void*){dealloc(p);}\n"
"    void operator delete(void*p){dealloc(p);}\n"
"    void operator delete(void*p, int*, const int){dealloc(p);}\n"
"    void operator delete(void*p, int*){dealloc(p);}\n"
"#endif\n"
"    void operator delete(void*p, size_t){dealloc(p);}\n"
"    static void* alloc(int,size_t, int*, int);\n"
"    static void dealloc(void *p);\n"
;

void
Message::genAllocDecl(XStr &str)
{
  int i, num;
  XStr mtype;
  mtype << type;
  if(templat) templat->genVars(mtype);
  str << CIMsgClassAnsi;
  str << "    CMessage_" << mtype << "();\n";
  str << "    static void *pack(" << mtype << " *p);\n";
  str << "    static " << mtype << "* unpack(void* p);\n";
  num = numArrays();
  if(num>0) {
    str << "    void *operator new(size_t";
    for(i=0;i<num;i++)
      str << ", int";
    str << ");\n";
  }
  str << "    void *operator new(size_t, ";
  for(i=0;i<num;i++)
    str << "int, ";
  str << "const int);\n";
  str << "#if CMK_MULTIPLE_DELETE\n";
  if(num>0) {
    str << "    void operator delete(void *p";
    for(i=0;i<num;i++)
        str << ", int";
    str << "){dealloc(p);}\n";
  }
  str << "    void operator delete(void *p, ";
  for(i=0;i<num;i++)
    str << "int, ";
  str << "const int){dealloc(p);}\n";
  str << "#endif\n";
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
    str << "      __idx = CkRegisterMsg(s, pack, unpack, dealloc, size);\n";
    str << "    }\n";
  }
  str << "};\n";
  
  if (strncmp(type->getBaseName(), "MarshallMsg_", 12) == 0) {
    MsgVarList *ml;
    MsgVar *mv;
    int i;
    str << "class " << type << " : public " << ptype << " {\n";
    str << "  public:\n";
    int num = numVars();
    for(i=0, ml=mvlist; i<num; i++, ml=ml->next) {
      mv = ml->msg_var;
      if (mv->isConditional() || mv->isArray()) {
        str << "    /* "; mv->print(str); str << " */\n";
        str << "    " << mv->type << " *" << mv->name << ";\n";
      }
    }
    str <<"};\n";
  }
}

void
Message::genDefs(XStr& str)
{
  int i, count, num = numVars();
  int numArray = numArrays();
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
    // new (size_t, int*)
    str << tspec << "void *" << ptype << "::operator new(size_t s, int* sz){\n";
    str << "  return " << mtype << "::alloc(__idx, s, sz, 0);\n}\n";
    // new (size_t, int*, priobits)
    str << tspec << "void *" << ptype << "::operator new(size_t s, int* sz,";
    str << "const int pb){\n";
    str << "  return " << mtype << "::alloc(__idx, s, sz, pb);\n}\n";
    // new (size_t, int, int, ..., int)
    if(numArray>0) {
      str << tspec << "void *" << ptype << "::operator new(size_t s";
      for(i=0;i<numArray;i++)
        str << ", int sz" << i;
      str << ") {\n";
      str << "  int sizes[" << numArray << "];\n";
      for(i=0;i<numArray;i++)
        str << "  sizes[" << i << "] = sz" << i << ";\n";
      str << "  return " << mtype << "::alloc(__idx, s, sizes, 0);\n";
      str << "}\n";
    }
    // new (size_t, int, int, ..., int, priobits)
    // degenerates to  new(size_t, priobits)  if no varsize
    str << tspec << "void *"<< ptype << "::operator new(size_t s, ";
    for(i=0;i<numArray;i++)
      str << "int sz" << i << ", ";
    str << "const int p) {\n";
    if (numArray>0) str << "  int sizes[" << numArray << "];\n";
    for(i=0, count=0, ml=mvlist ;i<num; i++, ml=ml->next)
      if (ml->msg_var->isArray()) {
        str << "  sizes[" << count << "] = sz" << count << ";\n";
        count ++;
      }
    str << "  return " << mtype << "::alloc(__idx, s, " << (numArray>0?"sizes":"0") << ", p);\n";
    str << "}\n";
    // alloc(int, size_t, int*, priobits)
    str << tspec << "void* " << ptype;
    str << "::alloc(int msgnum, size_t sz, int *sizes, int pb) {\n";
    str << "  CkpvAccess(_offsets)[0] = ALIGN8(sz);\n";
    for(i=0, count=0, ml=mvlist; i<num; i++, ml=ml->next) {
      mv = ml->msg_var;
      if (mv->isArray()) {
        str << "  if(sizes==0)\n";
        str << "    CkpvAccess(_offsets)[" << count+1 << "] = CkpvAccess(_offsets)[0];\n";
        str << "  else\n";
        str << "    CkpvAccess(_offsets)[" << count+1 << "] = CkpvAccess(_offsets)[" << count << "] + ";
        str << "ALIGN8(sizeof(" << mv->type << ")*sizes[" << count << "]);\n";
        count ++;
      }
    }
    str << "  return CkAllocMsg(msgnum, CkpvAccess(_offsets)[" << numArray << "], pb);\n";
    str << "}\n";

    str << tspec << ptype << "::" << proxyPrefix() << type << "() {\n";
    str << mtype << " *newmsg = (" << mtype << " *)this;\n";
    for(i=0, count=0, ml=mvlist; i<num; i++,ml=ml->next) {
      mv = ml->msg_var;
      if (mv->isArray()) {
        str << "  newmsg->" << mv->name << " = (" << mv->type << " *) ";
        str << "((char *)newmsg + CkpvAccess(_offsets)[" << count << "]);\n";
        count ++;
      }
    }
    str << "}\n";

    int numCond = numConditional();
    str << tspec << "void " << ptype << "::dealloc(void *p) {\n";
    if (numCond > 0) {
      str << "  " << mtype << " *msg = (" << mtype << "*) p;\n";
      for(i=0, count=0, ml=mvlist; i<num; i++, ml=ml->next) {
        mv = ml->msg_var;
        if (mv->isConditional()) {
          if (mv->type->isPointer()) die("conditional variable cannot be a pointer", line);
          str << "  CkConditional *cond_" << mv->name << " = static_cast<CkConditional*>(msg->" << mv->name << ");\n";
          str << "  if (cond_" << mv->name << "!=NULL) cond_" << mv->name << "->deallocate();\n";
        }
      }
    }
    str << "  CkFreeMsg(p);\n";
    str << "}\n";
    // pack
    str << tspec << "void* " << ptype << "::pack(" << mtype << " *msg) {\n";
    for(i=0, ml=mvlist; i<num; i++, ml=ml->next) {
      mv = ml->msg_var;
      if (mv->isArray()) {
        str << "  msg->" << mv->name << " = (" <<mv->type << " *) ";
        str << "((char *)msg->" << mv->name << " - (char *)msg);\n";
      }
    }
    if (numCond > 0) {
      str << "  int impl_off[" <<  numCond+1 << "];\n";
      str << "  impl_off[0] = UsrToEnv(msg)->getUsersize();\n";
      for(i=0, count=0, ml=mvlist; i<num; i++, ml=ml->next) {
        mv = ml->msg_var;
        if (mv->isConditional()) {
          str << "  if (msg->" << mv->name << "!=NULL) { /* conditional packing of ";
          mv->type->print(str); str << " " << mv->name << " */\n";
          str << "    PUP::sizer implP;\n";
          str << "    implP|*msg->" << mv->name << ";\n";
          str << "    impl_off[" << count+1 << "] = impl_off[" << count << "] + implP.size();\n";
          str << "  } else {\n";
          str << "    impl_off[" << count+1 << "] = impl_off[" << count << "];\n";
          str << "  }\n";
          count ++;
        }
      }
      str << "  " << mtype << " *newmsg = (" << mtype << "*) CkAllocMsg(__idx, impl_off["
          << numCond << "], UsrToEnv(msg)->getPriobits());\n";
      str << "  envelope *newenv = UsrToEnv(newmsg);\n";
      str << "  UInt newSize = newenv->getTotalsize();\n";
      str << "  CmiMemcpy(newenv, UsrToEnv(msg), impl_off[0]+sizeof(envelope));\n";
      str << "  newenv->setTotalsize(newSize);\n";
      str << "  if (UsrToEnv(msg)->getPriobits() > 0) CmiMemcpy(newenv->getPrioPtr(), UsrToEnv(msg)->getPrioPtr(), newenv->getPrioBytes());\n";
      for(i=0, count=0, ml=mvlist; i<num; i++, ml=ml->next) {
        mv = ml->msg_var;
        if (mv->isConditional()) {
          str << "  if (msg->" << mv->name << "!=NULL) { /* conditional packing of ";
          mv->type->print(str); str << " " << mv->name << " */\n";
          str << "    newmsg->" << mv->name << " = ("; mv->type->print(str);
          str << "*)(((char*)newmsg)+impl_off[" << count << "]);\n";
          str << "    PUP::toMem implP((void *)newmsg->" << mv->name << ");\n";
          str << "    implP|*msg->" << mv->name << ";\n";
          str << "    newmsg->" << mv->name << " = (" << mv->type << "*) ((char *)newmsg->" << mv->name << " - (char *)newmsg);\n";
          str << "  }\n";
          count++;
        }
      }
      str << "  CkFreeMsg(msg);\n";
      str << "  msg = newmsg;\n";
    }
    str << "  return (void *) msg;\n}\n";
    // unpack
    str << tspec << mtype << "* " << ptype << "::unpack(void* buf) {\n";
    str << "  " << mtype << " *msg = (" << mtype << " *) buf;\n";
    for(i=0, ml=mvlist; i<num; i++, ml=ml->next) {
      mv = ml->msg_var;
      if (mv->isArray()) {
        str << "  msg->" << mv->name << " = (" <<mv->type << " *) ";
        str << "((size_t)msg->" << mv->name << " + (char *)msg);\n";
      }
    }
    if (numCond > 0) {
      for(i=0, count=0, ml=mvlist; i<num; i++, ml=ml->next) {
        mv = ml->msg_var;
        if (mv->isConditional()) {
          str << "  if (msg->" << mv->name << "!=NULL) { /* conditional packing of ";
          mv->type->print(str); str << " " << mv->name << " */\n";
          str << "    PUP::fromMem implP((char*)msg + (size_t)msg->" << mv->name << ");\n";
          str << "    msg->" << mv->name << " = new " << mv->type << ";\n";
          str << "    implP|*msg->" << mv->name << ";\n";
          str << "  }\n";
          count ++;
        }
      }
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
Template::setExtern(int e)
{
  Construct::setExtern(e);
  entity->setExtern(e);
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
Template::genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent)
{
  if(!external && entity) {
    entity->genPub(declstr, defstr, defconstr, connectPresent);
  }
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

int Template::genAccels_spe_c_funcBodies(XStr& str) {
  int rtn = 0;
  if (!external && entity) { rtn += entity->genAccels_spe_c_funcBodies(str); }
  return rtn;
}

void Template::genAccels_spe_c_regFuncs(XStr& str) {
  if (!external && entity) { entity->genAccels_spe_c_regFuncs(str); }
}

void Template::genAccels_spe_c_callInits(XStr& str) {
  if (!external && entity) { entity->genAccels_spe_c_callInits(str); }
}

void Template::genAccels_spe_h_includes(XStr& str) {
  if (!external && entity) { entity->genAccels_spe_h_includes(str); }
}

void Template::genAccels_spe_h_fiCountDefs(XStr& str) {
  if (!external && entity) { entity->genAccels_spe_h_fiCountDefs(str); }
}

void Template::genAccels_ppe_c_regFuncs(XStr& str) {
  if (!external && entity) { entity->genAccels_ppe_c_regFuncs(str); }
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
Module::genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent)
{
  if(!external) {
    if (clist) clist->genPub(declstr, defstr, defconstr, connectPresent);
  }
}


void
Module::genDecls(XStr& str)
{
  if(external) {
    str << "#include \""<<name<<".decl.h\"\n";
  } else {
    if (clist) clist->genDecls(str);
  }

  #if CMK_CELL != 0
    str << "extern int register_accel_spe_funcs__module_" << name << "(int curIndex);\n";
    if (isMain()) {
      str << "extern \"C\" void register_accel_spe_funcs(void);\n";
    }
  #endif
}

void
Module::genDefs(XStr& str)
{
  if(!external)
    if (clist)
      clist->genDefs(str);

  // DMK - Accel Support
  #if CMK_CELL != 0

    if (!external) {

      // Protected this functioni with CK_TEMPLATES_ONLY check
      str << "#ifndef CK_TEMPLATES_ONLY\n";

      // Create the registration function
      // NOTE: Add a check so modules won't register more than once.  It is possible that to modules
      //   could have 'extern' references to each other, creating infinite loops in the registration
      //   process.  Avoid this problem.
      str << "int register_accel_spe_funcs__module_" << name << "(int curIndex) {\n"
          << "  static int hasAlreadyRegisteredFlag = 0;\n"
          << "  if (hasAlreadyRegisteredFlag) { return curIndex; };\n"
          << "  hasAlreadyRegisteredFlag = 1;\n";
      genAccels_ppe_c_regFuncs(str);
      str << "  return curIndex;\n"
          << "}\n";

      // Check to see if this is the main module (create top register function if so)
      if (isMain()) {
        str << "#include\"spert.h\"\n"  // NOTE: Make sure SPE_FUNC_INDEX_USER is defined
            << "extern \"C\" void register_accel_spe_funcs(void) {\n"
	    << "  register_accel_spe_funcs__module_" << name << "(SPE_FUNC_INDEX_USER);\n"
	    << "}\n";
      }

      str << "#endif /*CK_TEMPLATES_ONLY*/\n";
    }

  #endif
}

void
Module::genReg(XStr& str)
{
  if(external) {
    str << "      _register"<<name<<"();"<<endx;
  } else {
    if (clist) clist->genDefs(str);
  }
}


int Module::genAccels_spe_c_funcBodies(XStr& str) {

  // If this is an external module decloration, just place an include
  if (external) {
    str << "#include \"" << name << ".genSPECode.c\"\n";
    return 0;
  }

  // If this is the main module, generate the function lookup table
  if (isMain()) {
    str << "typedef void(*AccelFuncPtr)(DMAListEntry*);\n\n"
        << "typedef struct __func_lookup_table_entry {\n"
        << "  int funcIndex;\n"
        << "  AccelFuncPtr funcPtr;\n"
        << "} FuncLookupTableEntry;\n\n"
        << "FuncLookupTableEntry funcLookupTable[MODULE_" << name << "_FUNC_INDEX_COUNT];\n\n\n";
  }

  // Process each of the sub-constructs
  int rtn = 0;
  if (clist) { rtn += clist->genAccels_spe_c_funcBodies(str); }

  // Create the accelerated function registration function for accelerated entries local to this module
  // NOTE: Add a check so modules won't register more than once.  It is possible that to modules
  //   could have 'extern' references to each other, creating infinite loops in the registration
  //   process.  Avoid this problem.
  str << "int register_accel_funcs_" << name << "(int curIndex) {\n"
      << "  static int hasAlreadyRegisteredFlag = 0;\n"
      << "  if (hasAlreadyRegisteredFlag) { return curIndex; };\n"
      << "  hasAlreadyRegisteredFlag = 1;\n";
  genAccels_spe_c_regFuncs(str);
  str << "  return curIndex;\n"
      << "}\n\n\n";

  // If this is the main module, generate the funcLookup function
  if (isMain()) {

    str << "\n\n";
    str << "#ifdef __cplusplus\n"
        << "extern \"C\"\n"
        << "#endif\n"
        << "void funcLookup(int funcIndex,\n"
        << "                void* readWritePtr, int readWriteLen,\n"
        << "                void* readOnlyPtr, int readOnlyLen,\n"
        << "                void* writeOnlyPtr, int writeOnlyLen,\n"
        << "                DMAListEntry* dmaList\n"
        << "               ) {\n\n";

    str << "  if ((funcIndex >= SPE_FUNC_INDEX_USER) && (funcIndex < (SPE_FUNC_INDEX_USER + MODULE_" << name << "_FUNC_INDEX_COUNT))) {\n"

        //<< "    // DMK - DEBUG\n"
        //<< "    printf(\"[DEBUG-ACCEL] :: [SPE_%d] - Calling funcIndex %d...\\n\", (int)getSPEID(), funcIndex);\n"

        << "    (funcLookupTable[funcIndex - SPE_FUNC_INDEX_USER].funcPtr)(dmaList);\n"
        << "  } else if (funcIndex == SPE_FUNC_INDEX_INIT) {\n"
        << "    if (register_accel_funcs_" << name << "(0) != MODULE_" << name << "_FUNC_INDEX_COUNT) {\n"
        << "      printf(\"ERROR : register_accel_funcs_" << name << "() returned an invalid value.\\n\");\n"
	<< "    };\n";
    genAccels_spe_c_callInits(str);
    str << "  } else if (funcIndex == SPE_FUNC_INDEX_CLOSE) {\n"
        << "    // NOTE : Do nothing on close, but handle case...\n"
        << "  } else {\n"
	<< "    printf(\"ERROR : Unknown funcIndex (%d) passed to funcLookup()... ignoring.\\n\", funcIndex);\n"
	<< "  }\n";

    str << "}\n";
  }

  return rtn;
}

void Module::genAccels_spe_c_regFuncs(XStr& str) {
  if (external) {
    str << "  curIndex = register_accel_funcs_" << name << "(curIndex);\n";
  } else {
    if (clist) { clist->genAccels_spe_c_regFuncs(str); }
  }
}

void Module::genAccels_spe_c_callInits(XStr& str) {
  if (clist) { clist->genAccels_spe_c_callInits(str); }
}

void Module::genAccels_spe_h_includes(XStr& str) {
  if (external) {
    str << "#include \"" << name << ".genSPECode.h\"\n";
  }
  if (clist) { clist->genAccels_spe_h_includes(str); }
}

void Module::genAccels_spe_h_fiCountDefs(XStr& str) {
  if (external) {
    str << " + MODULE_" << name << "_FUNC_INDEX_COUNT";
  }
  if (clist) { clist->genAccels_spe_h_fiCountDefs(str); }
}

void Module::genAccels_ppe_c_regFuncs(XStr& str) {
  if (external) {
    str << "  curIndex = register_accel_spe_funcs__module_" << name << "(curIndex);\n";
  } else {
    if (clist) { clist->genAccels_ppe_c_regFuncs(str); }
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
    str << "#ifndef CK_TEMPLATES_ONLY\n";
    str << "extern \"C\" void __xlater_roPup_"<<makeIdent(qName());
    str <<    "(void *_impl_pup_er) {\n";
    str << "  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;\n";
    if(dims){
	    str << "  _impl_p("<<qName()<<","; dims->printValue(str); str<<");\n";
    }else{
	    str << "  _impl_p|"<<qName()<<";\n";
    }
    str << "}\n";
    str << "#endif\n";
  }

  if (fortranMode) {
      str << "extern \"C\" void "
          << fortranify("set_", name)
          << "(int *n) { " << name << " = *n; }\n";
      str << "extern \"C\" void "
          << fortranify("get_", name)
          << "(int *n) { *n = " << name << "; }\n";
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
    perElemGen(members, str, &Member::genIndexDecls, newLine);
}

void MemberList::genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent)
{
    for (list<Member*>::iterator i = members.begin(); i != members.end(); ++i)
	if (*i) {
	    (*i)->genPub(declstr, defstr, defconstr, connectPresent);
	    declstr << endx;
	}
}

void MemberList::genDecls(XStr& str)
{
    perElemGen(members, str, &Member::genDecls, newLine);
}

void MemberList::collectSdagCode(CParsedFile *pf, int& sdagPresent)
{
    for (list<Member*>::iterator i = members.begin(); i != members.end(); ++i)
	if (*i)
	    (*i)->collectSdagCode(pf, sdagPresent);
}

void MemberList::genDefs(XStr& str)
{
    perElemGen(members, str, &Member::genDefs, newLine);
}

void MemberList::genReg(XStr& str)
{
    perElemGen(members, str, &Member::genReg, newLine);
}

void MemberList::preprocess()
{
    for (list<Member*>::iterator i = members.begin(); i != members.end(); ++i)
	if (*i)
	    (*i)->preprocess();
}

void MemberList::lookforCEntry(CEntry *centry)
{
    perElemGen(members, centry, &Member::lookforCEntry);
}

void MemberList::genPythonDecls(XStr& str) {
    perElemGen(members, str, &Member::genPythonDecls, newLine);
}

void MemberList::genPythonDefs(XStr& str) {
    perElemGen(members, str, &Member::genPythonDefs, newLine);
}

void MemberList::genPythonStaticDefs(XStr& str) {
    perElemGen(members, str, &Member::genPythonStaticDefs);
}

void MemberList::genPythonStaticDocs(XStr& str) {
    perElemGen(members, str, &Member::genPythonStaticDocs);
}

void Entry::lookforCEntry(CEntry *centry)
{
   // compare name
   if (strcmp(name, *centry->entry) != 0) return;
   // compare param
   if (param && !centry->paramlist) return;
   if (!param && centry->paramlist) return;
   if (param && !(*param == *centry->paramlist)) return;

   isWhenEntry = 1;
   centry->decl_entry = this;
}

void Chare::lookforCEntry(CEntry *centry)
{
  if(list)
    list->lookforCEntry(centry);
  if (centry->decl_entry == NULL)  {
    cerr<<"Function \""<<centry->entry->get_string_const()
        <<"\" appears in Sdag When construct, but not defined as an entry function. "
        << endl;
    die("(FATAL ERROR)");
  }
}

///////////////////////////// CPARSEDFILE //////////////////////
/*void CParsedFile::print(int indent)
{
  for(CEntry *ce=entryList.begin(); !entryList.end(); ce=entryList.next())
  {
    ce->print(indent);
    printf("\n");
  }
  for(SdagConstruct *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next())
  {
    cn->print(indent);
    printf("\n");
  }
}
*/
XStr *CParsedFile::className = NULL;

void CParsedFile::numberNodes(void)
{
  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    if (cn->sdagCon != 0) {
      cn->sdagCon->numberNodes();
    }
  }
}

void CParsedFile::labelNodes(void)
{
  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    if (cn->sdagCon != 0) {
      cn->sdagCon->labelNodes();
    }
  }
}

void CParsedFile::propagateState(void)
{
  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    cn->sdagCon->propagateState(0);
  }
}

void CParsedFile::mapCEntry(void)
{
  CEntry *en;
  for(en=entryList.begin(); !entryList.end(); en=entryList.next()) {
    container->lookforCEntry(en);
  }
}

void CParsedFile::generateEntryList(void)
{
  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    cn->sdagCon->generateEntryList(entryList, 0);
  }
}

void CParsedFile::generateConnectEntryList(void)
{
  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    cn->sdagCon->generateConnectEntryList(connectEntryList);
  }
}

void CParsedFile::generateCode(XStr& op)
{
  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    cn->sdagCon->setNext(0,0);
    cn->sdagCon->generateCode(op);
  }
}

void CParsedFile::generateEntries(XStr& op)
{
  CEntry *en;
  SdagConstruct *sc;
  op << "public:\n";
  for(sc=connectEntryList.begin(); !connectEntryList.end(); sc=connectEntryList.next())
     sc->generateConnectEntries(op);
  for(en=entryList.begin(); !entryList.end(); en=entryList.next()) {
    en->generateCode(op);
  }
}

void CParsedFile::generateInitFunction(XStr& op)
{
  op << "private:\n";
  op << "  CDep *__cDep;\n";
  op << "  void __sdag_init(void) {\n";
  op << "    __cDep = new CDep("<<numEntries<<","<<numWhens<<");\n";
  CEntry *en;
  for(en=entryList.begin(); !entryList.end(); en=entryList.next()) {
    en->generateDeps(op);
  }
  op << "  }\n";
}


/**
    Create a merging point for each of the places where multiple
    dependencies lead into some future task.

    Used by Isaac's critical path detection
*/
void CParsedFile::generateDependencyMergePoints(XStr& op) 
{

  op << " \n";

  // Each when statement will have a set of message dependencies, and
  // also the dependencies from completion of previous task
  for(int i=0;i<numWhens;i++){
    op << "  MergeablePathHistory _when_" << i << "_PathMergePoint; /* For Critical Path Detection */ \n";
  }
  
  // The end of each overlap block will have multiple paths that merge
  // before the subsequent task is executed
  for(int i=0;i<numOlists;i++){
    op << "  MergeablePathHistory olist__co" << i << "_PathMergePoint; /* For Critical Path Detection */ \n";
  }

}


void CParsedFile::generatePupFunction(XStr& op)
{
  op << "public:\n";
  op << "  void __sdag_pup(PUP::er& p) {\n";
  op << "    if (__cDep) { __cDep->pup(p); }\n";
  op << "  }\n";
}

void CParsedFile::generateTrace()
{
  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    if (cn->sdagCon != 0) {
      cn->sdagCon->generateTrace();
    }
  }
}

void CParsedFile::generateRegisterEp(XStr& op)
{
  op << "  static void __sdag_register() {\n\n";

  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    if (cn->sdagCon != 0) {
      cn->sdagCon->generateRegisterEp(op);
    }
  }
  op << "  }\n";
}

void CParsedFile::generateTraceEpDecl(XStr& op)
{
  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    if (cn->sdagCon != 0) {
      cn->sdagCon->generateTraceEpDecl(op);
    }
  }
}

void CParsedFile::generateTraceEpDef(XStr& op)
{
  for(Entry *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    if (cn->sdagCon != 0) {
      cn->sdagCon->generateTraceEpDef(op);
    }
  }
}


////////////////////////// SDAGCONSTRUCT ///////////////////////
SdagConstruct::SdagConstruct(EToken t, SdagConstruct *construct1)
{
  con1 = 0;  con2 = 0; con3 = 0; con4 = 0;
  type = t;
  traceName=NULL;
  publishesList = new TList<SdagConstruct*>();
  constructs = new TList<SdagConstruct*>();
  constructs->append(construct1);
}

SdagConstruct::SdagConstruct(EToken t, SdagConstruct *construct1, SdagConstruct *aList)
{
  con1=0; con2=0; con3=0; con4=0;
  type = t;
  traceName=NULL;
  publishesList = new TList<SdagConstruct*>();
  constructs = new TList<SdagConstruct*>();
  constructs->append(construct1);
  SdagConstruct *sc;
  for(sc = aList->constructs->begin(); !aList->constructs->end(); sc=aList->constructs->next())
    constructs->append(sc);
}

SdagConstruct::SdagConstruct(EToken t, XStr *txt, SdagConstruct *c1, SdagConstruct *c2, SdagConstruct *c3,
			     SdagConstruct *c4, SdagConstruct *constructAppend, EntryList *el)
{
  text = txt;
  type = t;
  traceName=NULL;
  con1 = c1; con2 = c2; con3 = c3; con4 = c4;
  publishesList = new TList<SdagConstruct*>();
  constructs = new TList<SdagConstruct*>();
  if (constructAppend != 0) {
    constructs->append(constructAppend);
  }
  elist = el;
}

SdagConstruct::SdagConstruct(EToken t, const char *entryStr, const char *codeStr, ParamList *pl)
{
  type = t;
  traceName=NULL;
  text = new XStr(codeStr);
  connectEntry = new XStr(entryStr);
  con1 = 0; con2 = 0; con3 = 0; con4 =0;
  publishesList = new TList<SdagConstruct*>();
  constructs = new TList<SdagConstruct*>();
  param = pl;

}

///////////////////////////// ENTRY ////////////////////////////

void ParamList::checkParamList(){
  if(manyPointers){ 
    die("You may pass only a single pointer to a non-local entry method. It should point to a message.", param->line);
    abort();
  }
}

Entry::Entry(int l, int a, Type *r, const char *n, ParamList *p, Value *sz, SdagConstruct *sc, const char *e, int connect, ParamList *connectPList) :
      attribs(a), retType(r), stacksize(sz), sdagCon(sc), name((char *)n), intExpr(e), param(p), connectParam(connectPList), isConnect(connect)
{
  line=l; container=NULL;
  entryCount=-1;
  isWhenEntry=0;
  if (param && param->isMarshalled() && !isThreaded()) attribs|=SNOKEEP;

  if(!isThreaded() && stacksize) die("Non-Threaded methods cannot have stacksize",line);
  if(retType && !isSync() && !isIget() && !isLocal() && !retType->isVoid())
    die("A remote method normally returns void.  To return non-void, you need to declare the method as [sync], which means it has blocking semantics.",line);
  if (isPython()) pythonDoc = python_doc;
  if(!isLocal() && p){
    p->checkParamList();
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

void Entry::collectSdagCode(CParsedFile *pf, int& sdagPresent)
{
  if (isSdag()) {
    sdagPresent = 1;
    pf->nodeList.append(this);
  }
}

XStr Entry::marshallMsg(void)
{
  XStr ret;
  XStr epName = epStr();
  param->marshall(ret, epName);
  return ret;
}

XStr Entry::epStr(void)
{
  XStr str;
  str << name << "_";
  if (param->isMessage()) {
    str<<param->getBaseName();
    str.replace(':', '_');
  }
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
    str << "    "<<retType<<" "<<name<<"("<<paramType(1,1)<<");\n";
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
    genArrayStaticConstructorDecl(str);
  } else {
    if ((isSync() || isLocal()) && !container->isForElement()) return; //No sync broadcast
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
	  str << "  _TRACE_BEGIN_EXECUTE_DETAILED(0,ForArrayEltMsg," << epIdx()
	      << ",CkMyPe(), 0, ((CkArrayIndex&)ckGetIndex()).getProjectionID(((CkGroupID)ckGetArrayID()).idx));\n";
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
  if (container->getForWhom()==forIndividual)
      str<< //Element insertion routine
      "    void insert("<<paramComma(1,0)<<"int onPE=-1"<<eo(1)<<");";
  else if (container->getForWhom()==forAll) {
      str<< //With options
      "    static CkArrayID ckNew("<<paramComma(1,0)<<"const CkArrayOptions &opts"<<eo(1)<<");\n";
      if (container->isArray()) {
        XStr dim = ((Array*)container)->dim();
        if (dim==(const char*)"1D") {
          str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const int s1"<<eo(1)<<");\n";
        } else if (dim==(const char*)"2D") {
          str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const int s1, const int s2"<<eo(1)<<");\n";
        } else if (dim==(const char*)"3D") {
          str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const int s1, const int s2, const int s3"<<eo(1)<<");\n";
        /*} else if (dim==(const char*)"4D") {
          str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const short s1, const short s2, const short s3, const short s4"<<eo(1)<<");\n";
        } else if (dim==(const char*)"5D") {
          str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const short s1, const short s2, const short s3, const short s4, const short s5"<<eo(1)<<");\n";
        } else if (dim==(const char*)"6D") {
          str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const short s1, const short s2, const short s3, const short s4, const short s5, const short s6"<<eo(1)<<");\n"; */
        }
      }
  }
  else if (container->getForWhom()==forSection) { }
}

void Entry::genArrayStaticConstructorDefs(XStr& str)
{
  if (container->getForWhom()==forIndividual)
      str<<
      makeDecl("void",1)<<"::insert("<<paramComma(0,0)<<"int onPE"<<eo(0)<<")\n"
      "{ \n"<<marshallMsg()<<
      "   ckInsert((CkArrayMessage *)impl_msg,"<<epIdx()<<",onPE);\n}\n";
  else if (container->getForWhom()==forAll){
      str<<
      makeDecl("CkArrayID",1)<<"::ckNew("<<paramComma(0)<<"const CkArrayOptions &opts"<<eo(0)<<")\n"
       "{ \n"<<marshallMsg()<<
	 "   return ckCreateArray((CkArrayMessage *)impl_msg,"<<epIdx()<<",opts);\n"
       "}\n";
      if (container->isArray()) {
        XStr dim = ((Array*)container)->dim();
        if (dim==(const char*)"1D") {
          str<<
            makeDecl("CkArrayID",1)<<"::ckNew("<<paramComma(0)<<"const int s1"<<eo(0)<<")\n"
            "{ \n"<<marshallMsg()<<
            "   return ckCreateArray((CkArrayMessage *)impl_msg,"<<epIdx()<<",CkArrayOptions(s1));\n"
            "}\n";
        } else if (dim==(const char*)"2D") {
          str<<
            makeDecl("CkArrayID",1)<<"::ckNew("<<paramComma(0)<<"const int s1, const int s2"<<eo(0)<<")\n"
            "{ \n"<<marshallMsg()<<
            "   return ckCreateArray((CkArrayMessage *)impl_msg,"<<epIdx()<<",CkArrayOptions(s1, s2));\n"
            "}\n";
        } else if (dim==(const char*)"3D") {
          str<<
            makeDecl("CkArrayID",1)<<"::ckNew("<<paramComma(0)<<"const int s1, const int s2, const int s3"<<eo(0)<<")\n"
            "{ \n"<<marshallMsg()<<
            "   return ckCreateArray((CkArrayMessage *)impl_msg,"<<epIdx()<<",CkArrayOptions(s1, s2, s3));\n"
            "}\n";
        /*} else if (dim==(const char*)"4D") {
          str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const short s1, const short s2, const short s3, const short s4"<<eo(1)<<");\n";
        } else if (dim==(const char*)"5D") {
          str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const short s1, const short s2, const short s3, const short s4, const short s5"<<eo(1)<<");\n";
        } else if (dim==(const char*)"6D") {
          str<<"    static CkArrayID ckNew("<<paramComma(1,0)<<"const short s1, const short s2, const short s3, const short s4, const short s5, const short s6"<<eo(1)<<");\n";
        */
        }
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
    genGroupStaticConstructorDecl(str);
  } else {
    if ((isSync() || isLocal()) && !container->isForElement()) return; //No sync broadcast
    if (isLocal())
      str << "    "<<retType<<" "<<name<<"("<<paramType(1,1,0)<<");\n";
    else
      str << "    "<<retType<<" "<<name<<"("<<paramType(1,1)<<");\n";
    // entry method on multiple PEs declaration
    if(!container->isForElement() && !container->isForSection() && !isSync() && !isLocal() && !container->isNodeGroup()) {
      str << "    "<<retType<<" "<<name<<"("<<paramComma(1,0)<<"int npes, int *pes"<<eo(1)<<");\n";
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
      if (!isNoTrace()) str << "  _TRACE_BEGIN_EXECUTE_DETAILED(0,ForBocMsg,"<<epIdx()<<",CkMyPe(),0,NULL);\n";
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
  str << "  return CkCreate"<<node<<"Group("<<chareIdx()<<", "<<epIdx()<<", impl_msg);\n";
  str << "}\n";

  if (!param->isVoid()) {
    str << makeDecl(" ",1)<<"::"<<container->proxyName(0)<<"("<<paramType(0,1)<<")\n";
    str << "{\n"<<marshallMsg();
    str << "  ckSetGroupID(CkCreate"<<node<<"Group("<<chareIdx()<<", "<<epIdx()<<", impl_msg));\n";
    str << "}\n";
  }
}

/******************* Python Entry Point Code **************************/
void Entry::genPythonDecls(XStr& str) {
  str <<"/* STATIC DECLS: "; print(str); str << " */\n";
  if (isPython()) {
    // check the parameter passed to the function, it must be only an integer
    if (!param || param->next || !param->param->getType()->isBuiltin() || !((BuiltinType*)param->param->getType())->isInt()) {
      die("A python entry method must accept only one parameter of type `int`");
    }

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
    die("Accel entry with no parameters");
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
  str << "/* DECLS: "; print(str); str << " */\n";

  // Entry point index storage
  str << "    static int "<<epIdx(0)<<";\n";

  // DMK - Accel Support - Also declare the function index for the Offload API call
  #if CMK_CELL != 0
    if (isAccel()) {
      str << "    static int accel_spe_func_index__" << epStr() << ";\n";
    }
  #endif

  // Index function, so user can find the entry point number
  str << "    static int ";
  if (isConstructor()) str <<"ckNew";
  else str <<name;
  str << "("<<paramType(1,0)<<") { return "<<epIdx(0)<<"; }\n";

  // DMK - Accel Support
  if (isAccel()) {
    genAccelIndexWrapperDecl_general(str);
    #if CMK_CELL != 0
      genAccelIndexWrapperDecl_spe(str);
    #endif
  }

  if (isReductionTarget()) {
      str << "    static int __idx_" << name << "_redn_wrapper;\n"
          << "    static int " << name << "_redn_wrapper"
          << "(CkReductionMsg* impl_msg) { return __idx_" << name << "_redn_wrapper; }\n"
          << "    static void _" << name << "_redn_wrapper(void* impl_msg, "
          << container->baseName() <<"* impl_obj);\n";
  }

  // call function declaration
  str << "    static void _call_"<<epStr()<<"(void* impl_msg,"<<
    container->baseName()<<"* impl_obj);\n";
  if(isThreaded()) {
    str << "    static void _callthr_"<<epStr()<<"(CkThrCallArg *);\n";
  }
  if (hasCallMarshall) {
    str << "    static int _callmarshall_"<<epStr()<<"(char* impl_buf,"<<
      container->baseName()<<"* impl_obj);\n";
  }
  if (param->isMarshalled()) {
    str << "    static void _marshallmessagepup_"<<epStr()<<"(PUP::er &p,void *msg);\n";
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
      if(!isIget())
        genArrayDecl(str);
      else if(container->isForElement())
	genArrayDecl(str);
  } else { // chare or mainchare
      genChareDecl(str);
  }
}


void Entry::genPub(XStr &declstr, XStr& defstr, XStr& defconstr, int& connectPresent)
{
/*  if (isConnect == 1)
     printf("Entry is Connected %s\n", name);
  else
     printf("Entry is not Connected %s\n", name);
*/
  if ((isConnect == 1) && (connectPresent == 0)) {
     connectPresent = 1;
     declstr << "class publish\n";
     declstr << "{\n";
     declstr << "   public:\n";
     declstr << "      publish();\n";
     defconstr << "publish::publish()\n"  << "{\n";
  }
  if (isConnect == 1) {
     defconstr << "   publishflag_" <<getEntryName() << " = 0;\n";
     defconstr << "   getflag_" <<getEntryName() << " = 0;\n";
     declstr << "      void " <<getEntryName() <<"(";
     defstr << "void publish::" << getEntryName() <<"(";
     ParamList *pl = connectParam;
     XStr *parameters = new XStr("");
     int count = 0;
     int i, numStars;
     if (pl->isVoid() == 1) {
	declstr << "void);\n";
	defstr << "void);\n";
     }
     else if (pl->isMessage() == 1){
	declstr << pl->getBaseName() <<"* " << pl->getGivenName() <<");\n";
	defstr << pl->getBaseName() <<"* " << pl->getGivenName() <<");\n";
	defconstr << "   " << pl->getGivenName() <<" = new " << pl->getBaseName() <<"();\n";
	parameters->append("      ");
	parameters->append(pl->getBaseName());
	parameters->append("* ");
	parameters->append(pl->getGivenName());
	parameters->append("_msg;\n ");
     }
     else {
	defconstr << "   " << getEntryName() <<"_msg = new CkMarshallMsg();\n";
	parameters->append("      CkMarshallMsg *");
	parameters->append(getEntryName());
	parameters->append("_msg;\n");
        while(pl != NULL) {
	  if (count > 0) {
	    declstr << ", ";
	    defstr << ", ";
	  }
	  if (pl->isPointer() == 1) {
	  // FIX THE FOLLOWING - I think there could be problems if the original passed in value is deleted
	    declstr << pl->getBaseName();
	    defstr << pl->getBaseName();
	    numStars = pl->getNumStars();
	    for(i=0; i< numStars; i++) {
	      declstr << "*";
	      defstr << "*";
	    }
	    declstr << " " <<  pl->getGivenName();
	    defstr << " " <<  pl->getGivenName();
	  }
	  else if (pl->isReference() == 1) {
	    declstr << pl->getBaseName() <<"& " <<pl->getGivenName();
	    defstr << pl->getBaseName() <<"& " <<pl->getGivenName();
	  }
	  else if (pl->isArray() == 1){
	    declstr << pl->getBaseName() <<"* " <<pl->getGivenName();
	    defstr << pl->getBaseName() <<"* " <<pl->getGivenName();
	  }
	  else if ((pl->isBuiltin() == 1) || (pl->isNamed() == 1)) {
	    declstr << pl->getBaseName() <<" " <<pl->getGivenName();
	    defstr << pl->getBaseName() <<" " <<pl->getGivenName();
	  }
	  pl = pl->next;
	  count++;
	}
	declstr << "); \n";
	defstr << ") { \n";
     }
     declstr << "      void get_" << getEntryName() << "(CkCallback cb);\n";
     declstr << "      int publishflag_" << getEntryName() << ";\n";
     declstr << "      int getflag_" << getEntryName() << ";\n";
     declstr << "      CkCallback " << getEntryName() << "_cb;\n";
     declstr << parameters->charstar();

     // Following generates the def publish::connectFunction code

     // Traverse thru parameter list and set the local messages accordingly
     defstr <<"    const CkEntryOptions *impl_e_opts = NULL;\n";
     XStr epName = epStr();
     connectParam->marshall(defstr, epName);
     defstr << "   " << getEntryName() << "_msg = impl_msg;\n";
     defstr << "   " << "if (getflag_" << getEntryName() <<" == 1) {\n";
     // FIX THE FOLLOWING IN CASE MSG IS VOID
     defstr << "     " << getEntryName() << "_cb.send(" << getEntryName() <<"_msg);\n";
     defstr << "   }\n";
     defstr << "   else\n";
     defstr << "     publishflag_" << getEntryName() << " = 1;\n";
     defstr << "}\n\n";

     // Following generates the def publish::get_connectFunction code

     defstr << "void publish::get_" << getEntryName() << "(CkCallback cb) {\n";
     defstr << "   " << getEntryName() << "_cb = cb;\n";
     defstr << "   if (publishflag_" << getEntryName() << " == 1) {\n";
     defstr << "     cb.send(" << getEntryName() << "_msg);\n";
     defstr << "     publishflag_" << getEntryName() << " = 0 ;\n";
     defstr << "   }\n";
     defstr << "   else\n";
     defstr << "     getflag_" << getEntryName() << " = 1;\n";
     defstr << "}\n";
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

  str << "  CthThread tid = CthCreate((CthVoidFn)"<<procFull
   <<", new CkThrCallArg(impl_msg,impl_obj), "<<getStackSize()<<");\n";
  str << "  ((Chare *)impl_obj)->CkAddThreadListeners(tid,impl_msg);\n";
  // str << "  CkpvAccess(_traces)->CkAddThreadListeners(tid);\n";
#if CMK_BIGSIM_CHARM
  str << "  BgAttach(tid);\n";
#endif
  str << "  CthAwaken(tid);\n";
  str << "}\n";
//  str << "  CthAwaken(CthCreate((CthVoidFn)"<<procFull
//   <<", new CkThrCallArg(impl_msg,impl_obj), "<<getStackSize()<<"));\n}\n";

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
void Entry::genCall(XStr& str, const XStr &preCall, bool redn_wrapper)
{
  bool isArgcArgv=false;
  bool isMigMain=false;

  if (param->isCkArgMsgPtr() && (!isConstructor() || !container->isMainChare()))
    die("CkArgMsg can only be used in mainchare's constructor.\n");

  if (isConstructor() && container->isMainChare() &&
      (!param->isVoid()) && (!param->isCkArgMsgPtr())){
  	if(param->isCkMigMsgPtr()) isMigMain = true;
	else isArgcArgv = true;
  } else {
    //Normal case: Unmarshall variables
    if (redn_wrapper) param->beginRednWrapperUnmarshall(str);
    else param->beginUnmarshall(str);
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
  	str << "  impl_obj->"<<name;
    }

    if (isArgcArgv) { //Extract parameters from CkArgMsg (should be parameter marshalled)
        str<<"(m->argc,m->argv);\n";
        str<<"  delete m;\n";
    }else if(isMigMain){
        str<<"((CkMigrateMessage*)impl_msg);\n";
    }
    else {//Normal case: unmarshall parameters (or just pass message)
        str<<"("; param->unmarshall(str); str<<");\n";
    }
  }
}

void Entry::genDefs(XStr& str)
{
  XStr containerType=container->baseName();
  XStr preMarshall,preCall,postCall;

  str << "/* DEFS: "; print(str); str << " */\n";

  if (attribs&SMIGRATE)
    {} //User cannot call the migration constructor
  else if(container->isGroup()){
    genGroupDefs(str);
  } else if(container->isArray()) {
    genArrayDefs(str);
  } else
    genChareDefs(str);

  if (container->isChare() || container->isForElement()) {
      if (isReductionTarget()) {
          XStr retStr; retStr<<retType;
          str << retType << " " << indexName(); //makeDecl(retStr, 1)
          str << "::_" << name << "_redn_wrapper(void* impl_msg, "
              << container->baseName() << "* impl_obj)\n{\n"
              << "  char* impl_buf = (char*)((CkReductionMsg*)impl_msg)->getData();\n";
          XStr precall;
          genCall(str, precall, true);
          str << "\n}\n\n";
      }
  }


  //Prevents repeated call and __idx definitions:
  if (container->getForWhom()!=forAll) return;

  //Define storage for entry point number
  str << container->tspec()<<" int "<<indexName()<<"::"<<epIdx(0)<<"=0;\n";
  if (isReductionTarget()) {
      str << " int " << indexName() << "::__idx_" << name <<"_redn_wrapper=0;\n";
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
    if(isConstructor()) die("Constructors cannot be [sync]",line);
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
    if(!container->isNodeGroup()) die("only nodegroup methods can be exclusive",line);
    if(isConstructor()) die("Constructors cannot be [exclusive]",line);
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
  str << makeDecl("void")<<"::_call_"<<epStr()<<"(void* impl_msg,"<<containerType<<" * impl_obj)\n";
  str << "{\n";
  if (!isLocal()) {
    if(isThreaded()) str << callThread(epStr());
    str << preMarshall;
    if (param->isMarshalled()) {
      if (param->hasConditional()) str << "  MarshallMsg_"<<epStr()<<" *impl_msg_typed=(MarshallMsg_"<<epStr()<<" *)impl_msg;\n";
      else str << "  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;\n";
      str << "  char *impl_buf=impl_msg_typed->msgBuf;\n";
    }
    genCall(str,preCall);
    param->endUnmarshall(str);
    str << postCall;
    if(isThreaded() && param->isMarshalled()) str << "  delete impl_msg_typed;\n";
  } else {
    str << "  CkAbort(\"This method should never be called as it refers to a LOCAL entry method!\");\n";
  }
  str << "}\n";

  if (hasCallMarshall) {
    str << makeDecl("int")<<"::_callmarshall_"<<epStr()<<"(char* impl_buf,"<<containerType<<" * impl_obj) {\n";
    if (!isLocal()) {
      if (!param->hasConditional()) {
        genCall(str,preCall);
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
}

void Entry::genReg(XStr& str)
{
  str << "// REG: "<<*this;
  str << "  "<<epIdx(0)<<" = CkRegisterEp(\""<<name<<"("<<paramType(0)<<")\",\n"
  	"     (CkCallFnPtr)_call_"<<epStr()<<", ";
  /* messageIdx: */
  if (param->isMarshalled()) {
    if (param->hasConditional())  str<<"MarshallMsg_"<<epStr()<<"::__idx";
    else str<<"CkMarshallMsg::__idx";
  } else if(!param->isVoid() && !(attribs&SMIGRATE)) {
    param->genMsgProxyName(str);
    str <<"::__idx";
  } else {
    str << "0";
  }
  /* chareIdx */
  str << ", __idx";
  /* attributes */
  str << ", 0";
  if (attribs & SNOKEEP) str << "+CK_EP_NOKEEP";
  if (attribs & SNOTRACE) str << "+CK_EP_TRACEDISABLE";
  if (attribs & SIMMEDIATE) str << "+CK_EP_TRACEDISABLE";

  /*MEICHAO*/
  if (attribs & SMEM) str << "+CK_EP_MEMCRITICAL";
  
  if (internalMode) str << "+CK_EP_INTRINSIC";
  str << ");\n";
  if (isConstructor()) {
    if(container->isMainChare()&&!(attribs&SMIGRATE))
      str << "  CkRegisterMainChare(__idx, "<<epIdx(0)<<");\n";
    if(param->isVoid())
      str << "  CkRegisterDefaultCtor(__idx, "<<epIdx(0)<<");\n";
    if(attribs&SMIGRATE)
      str << "  CkRegisterMigCtor(__idx, "<<epIdx(0)<<");\n";
  }
  if (hasCallMarshall)
      str << "  CkRegisterMarshallUnpackFn("<<epIdx(0)<<
            ",(CkMarshallUnpackFn)_callmarshall_"<<epStr()<<");\n";

  if (param->isMarshalled()) {
      str << "  CkRegisterMessagePupFn("<<epIdx(0)<<
  	    ",(CkMessagePupFn)_marshallmessagepup_"<<epStr()<<");\n";
  }
  else if (param->isMessage() && !attribs&SMIGRATE) {
      str << "  CkRegisterMessagePupFn("<<epIdx(0)<<", (CkMessagePupFn)";
      str << param->param->getType()->getBaseName() <<"::ckDebugPup);\n";
  }
  if (isReductionTarget()) {
      str << "  " << "__idx_" << name << "_redn_wrapper = CkRegisterEp(\""
          << name << "_redn_wrapper(CkReductionMsg* impl_msg)\",\n"
          << "     (CkCallFnPtr)_" << name << "_redn_wrapper, "
          << "CMessage_CkReductionMsg::__idx, __idx, 0);";
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
          default:     die("Unknown accel param type");                            break;
        }
      } else {
        accel_numScalars++;
        switch (curParam->param->getAccelBufferType()) {
          case Parameter::ACCEL_BUFFER_TYPE_READWRITE:  accel_dmaList_scalarNeedsWrite++;  break;
          case Parameter::ACCEL_BUFFER_TYPE_READONLY:                                      break;
          case Parameter::ACCEL_BUFFER_TYPE_WRITEONLY:  accel_dmaList_scalarNeedsWrite++;  break;
          default:     die("Unknown accel param type");                                break;
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
    	:type(Ntype), name(Nname), arrLen(NarrLen), val(Nvalue),line(Nline)
{
        conditional=0;
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
	byReference=false;
	if ((arrLen==NULL)&&(val==NULL))
	{ /* Consider passing type by reference: */
		if (type->isNamed())
		{ /* Some user-defined type: pass by reference */
			byReference=true;
		}
		if (type->isReference()) {
			byReference=true;
			/* Clip off the ampersand--we'll add
			   it back ourselves in Parameter::print. */
			type=type->deref();
		}
	}
}

void ParamList::print(XStr &str,int withDefaultValues,int useConst)
{
    	param->print(str,withDefaultValues,useConst);
    	if (next) {
    		str<<", ";
    		next->print(str,withDefaultValues,useConst);
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
			if (useConst) str<<"const ";
			str<<type<<" &";
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
		die("can't pass pointers across processors--\n"
		    "Indicate the array length with []'s, or pass a reference",line);
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

void ParamList::beginRednWrapperUnmarshall(XStr &str)
{
    if (isMarshalled())
    {
        str<<"  /*Unmarshall pup'd fields: ";print(str,0);str<<"*/\n";
        str<<"  PUP::fromMem implP(impl_buf);\n";
        if (next != NULL && next->next == NULL) {
            if (isArray()) {
                Type* dt = next->param->type->deref();
                str << "  " << dt << " " << next->param->name << "; "
                    << next->param->name << " = "
                    << "((CkReductionMsg*)impl_msg)->getLength() / sizeof("
                    << param->type->deref() << ");\n";
                dt = param->type->deref();
                str << "  " << dt << "* " << param->name << "; "
                    << param->name << " = (" << dt << "*)impl_buf;\n";
            } else if (next->isArray()) {
                Type* dt = param->type->deref();
                str << "  " << dt << " " << param->name << "; "
                    << param->name << " = "
                    << "((CkReductionMsg*)impl_msg)->getLength() / sizeof("
                    << next->param->type->deref() << ");\n";
                dt = next->param->type->deref();
                str << "  " << dt << "* " << next->param->name << "; "
                    << next->param->name << " = (" << dt << "*)impl_buf;\n";
            } else {
                callEach(&Parameter::beginUnmarshall,str);
            }
        } else {
            str << "/* non two-param case */\n";
            callEach(&Parameter::beginUnmarshall,str);
            str<<"  impl_buf+=CK_ALIGN(implP.size(),16);\n";
            str<<"  /*Unmarshall arrays:*/\n";
            callEach(&Parameter::unmarshallArrayData,str);
        }
    } else if (isVoid()) {
        str<<"  CkFreeSysMsg(impl_msg);\n";
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
    else if (isVoid()) {str<<"  CkFreeSysMsg(impl_msg);\n";}
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
	  "  { for (int impl_i=0;impl_i*(sizeof(*"<<name<<"))<impl_cnt_"<<name<<";impl_i++) { \n"
	  "      implDestP.synchronize(PUP::sync_item);\n"
	  "      implDestP|"<<name<<"[impl_i];\n"
	  "  } } \n"
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

/***************** InitCall **************/
InitCall::InitCall(int l, const char *n, int nodeCall)
	    : name(n)
{
	line=l; setChare(0); isNodeCall=nodeCall;

        // DMK - Accel Support
        isAccelFlag = 0;
}
void InitCall::print(XStr& str)
{
	str<<"  initcall void "<<name<<"(void);\n";
}
void InitCall::genReg(XStr& str)
{
	str<<"      _registerInitCall(";
	if (container)
		str<<container->baseName()<<"::";
	str<<name;
	str<<","<<isNodeCall<<");\n";
}

void InitCall::genAccels_spe_c_callInits(XStr& str) {
  if (isAccel()) {
    str << "    " << name << "();\n";
  }
}


/***************** PUP::able support **************/
PUPableClass::PUPableClass(int l, NamedType* type_,PUPableClass *next_)
	    : type(type_), next(next_)
{
	line=l; setChare(0);
}
void PUPableClass::print(XStr& str)
{
	str << "  PUPable " << type <<";\n";
	if (next) next->print(str);
}
void PUPableClass::genDefs(XStr& str)
{
        if (type->isTemplated()) {
                str << "#ifdef CK_TEMPLATES_ONLY\n";
                str << "  #define _CHARMXI_CLASS_NAME " << type << "\n";
                str << "  PUPable_def_template(_CHARMXI_CLASS_NAME)\n";
                str << "  #undef _CHARMXI_CLASS_NAME\n";
                str << "#endif\n";
        } else {
                str<<"  PUPable_def(" << type << ")\n";
        }
	if (next) next->genDefs(str);
}
void PUPableClass::genReg(XStr& str)
{
        if (type->isTemplated()) {
                str<<"      #define _CHARMXI_CLASS_NAME " << type << "\n";
                str<<"      PUPable_reg2(_CHARMXI_CLASS_NAME,\"" << type << "\");\n";
                str<<"      #undef _CHARMXI_CLASS_NAME\n";
        } else {
                str<<"      PUPable_reg(" << type << ");\n";
        }
	if (next) next->genReg(str);
}


/***************** Include support **************/
IncludeFile::IncludeFile(int l, const char *n)
	    : name(n)
{
	line=l; setChare(0);
}
void IncludeFile::print(XStr& str)
{
	str<<"  include "<<name<<";\n";
}
void IncludeFile::genDecls(XStr& str) {
	str<<"#include "<<name<<"\n";
}


/***************** normal extern C Class support **************/
ClassDeclaration::ClassDeclaration(int l, const char *n)
	    : name(n)
{
	line=l; setChare(0);
}
void ClassDeclaration::print(XStr& str)
{
	str<<"  class "<<name<<";\n";
}
void ClassDeclaration::genDecls(XStr& str) {
	str<<"class "<<name<<";\n";
}


/****************** Registration *****************/

}
