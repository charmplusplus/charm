// BE_produce.cc - Produce the work of the BE - does nothing in the
//		   dummy BE

#include	"idl.hh"
#include	"idl_extern.hh"
#include	"be.hh"

#include <fstream.h>
#include <stdio.h>
#include <libgen.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define SZ 1024

class XString {
  private:
    char *s;
    unsigned int len, blklen;
  public:
    XString() {
      s = new char[SZ];
      *s = '\0';
      len = 0;
      blklen = SZ;
    }
    XString(char *_s) {
      len = strlen(_s);
      blklen = SZ;
      // gzheng
      blklen = (len/SZ + 1)*SZ;
      /*  HP doesn't support while in this inline constructor
      while ( len >= blklen ) {
        blklen += SZ;
      }
      */
      s = new char[blklen];
      strcpy(s, _s);
    }
    ~XString() { delete[] s; }
    char *get_string(void) { return s; }
    void append(char *_s) {
      len += strlen(_s);
      if ( len >= blklen) {
        while ( len >= blklen ) {
          blklen += SZ;
        }
        char *tmp = s;
        s = new char[blklen];
        strcpy(s, tmp);
        delete[] tmp;
      }
      strcat(s, _s);
    }
    void append(char c) {
      char tmp[2];
      tmp[0] = c;
      tmp[1] = '\0';
      append(tmp);
    }
};

static void
debugPrint(char *s)
{
  cerr << "DEBUG: " << s << endl;
}

static void
myAbort(char *s)
{
  cerr << "ABORT: " << s << endl;
  exit(1);
}

// Replace \1 by a1, \2 by a2, etc. in string b.  Sends output to
// outStream.
static void
spew(XString *ostr, const char*b,
     char *a1 = "ERROR", char *a2 = "ERROR",
     char *a3 = "ERROR", char *a4 = "ERROR", 
     char *a5 = "ERROR")
{
  int i;
  for(i=0; i<strlen(b); i++){
    switch(b[i]){
    case '\01':
      ostr->append(a1); break;
    case '\02':
      ostr->append(a2); break;
    case '\03':
      ostr->append(a3); break;
    case '\04':
      ostr->append(a4); break;
    case '\05':
      ostr->append(a5); break;
    default:
      ostr->append(b[i]);
    }
  }
}

//----------------------------------------------------------------------
// Given "path/file.ext" we return "file".
// @@ Given "path/file.b.ext" we return "file.b".
// Caller must delete the returned string.
static char *
getBasename(char *pathname)
{
  int i = 0;

  for(i=strlen(pathname)-1; i>=0; i--)
    if (pathname[i] == '/') {
      break;
    }
  i++;
  char *basenm = new char[strlen(pathname+i)+1];
  strcpy(basenm, pathname+i);

  for(i=0; basenm[i]; i++)
    if (basenm[i] == '.') {
      basenm[i] = '\0';
      break;
    }

  return basenm;
}

//----------------------------------------------------------------------
// Files to output to

ofstream *FC, *FH, *FCI;

//----------------------------------------------------------------------
// Strings to output to

XString *CIh, *CIc, *CIi;
XString *Icc, *Icg, *Im;
XString *Hcc, *Hcg, *Hm, *Hi;
XString *Ccc, *Ccg, *Cm, *Ci;

//----------------------------------------------------------------------
// Spew Code

const char *HiFormal0 = // type
"\01"
;

// gzheng
const char *HiFormal01 = // type
"\01 &"
;

const char *HiFormal1 =
", "
;

const char *HiFormal2 = // type, argname, arglen
"\01 \02[\03]"
;

// gzheng
const char *HiFormal21 = // type, argname, arglen
"\01 &\02[\03]"
;

const char *CiFormal0 = // type, argname
"\01 \02"
;

// gzheng
const char *CiFormal01 = // type, argname
"\01 &\02"
;

const char *CiFormal1 =
", "
;

const char *CiFormal2 = // type, argname, arglen
"\01 \02[\03]"
;

// gzheng
const char *CiFormal21 = // type, argname, arglen
"\01 &\02[\03]"
;

const char *CccFormal0 = // type, argname
"  \01 \02 = msg->\02;\n"
;

const char * CccFormal1 = // type, argname, arglen
"  \01 \02[\03];\n"
"  {\n"
"    int i;\n"
"    for(i=0;i<\03;i++)\n"
"      \02[i] = msg->\02[i];\n"
"  }\n"
;

const char *CcgFormal0 = // type, argname
"  \01 \02 = msg->\02;\n"
;

const char * CcgFormal1 = // type, argname, arglen
"  \01 \02[\03];\n"
"  {\n"
"    int i;\n"
"    for(i=0;i<\03;i++)\n"
"      \02[i] = msg->\02[i];\n"
"  }\n"
;

const char *HmFormal0 = // type, argname
"    \01 \02;\n"
;

const char *HmFormal1 = // type, argname, arglen
"    \01 \02[\03];\n"
;

const char *CiActual0 = // argname
"  msg->\01 = \01;\n"
;

const char *CiActual1 = // argname, arglen
"  {\n"
"    int i;\n"
"    for(i=0;i<\02;i++)\n"
"      msg->\01[i] = \01[i];\n"
"  }\n"
;

const char *CccActual0 = // argname
"\01"
;

const char *CccActual1 =
", "
;

const char *CcgActual0 = // argname
"\01"
;

const char *CcgActual1 =
", "
;

//----------------------------------------------------------------------
void
BE_produce_parameters(be_operation *bop, int passnum)
{
  UTL_ScopeActiveIterator   *i;
  AST_Decl		    *d;
  int argnum = 0;

  if(passnum == 1) { // Produce Param Lists
    i = new UTL_ScopeActiveIterator(bop, UTL_Scope::IK_decls);
    while (!(i->is_done())) {
      d = i->item();
      argnum++;
      if (d->node_type() == AST_Decl::NT_argument) {
        be_argument *a = be_argument::narrow_from_decl(d);
        if (a->isArray()) {
	  be_array *ar = be_array::narrow_from_decl(a->field_type());
	  char *arrayType = ar->base_type()->local_name()->get_string();
	  unsigned long arraySizeNum = ar->dims()[0]->ev()->u.ulval;
          char arraylen[128];
          sprintf(arraylen, "%ld", arraySizeNum);
          char argName[1024];
          sprintf(argName, "arg%d", argnum);
	  if(a->direction() == AST_Argument::dir_IN) {
            spew(Hi, HiFormal2, arrayType, argName, arraylen);
            spew(Ci, CiFormal2, arrayType, argName, arraylen);
            spew(Ccc, CccFormal1, arrayType, argName, arraylen);
            spew(Ccg, CcgFormal1, arrayType, argName, arraylen);
            spew(Hm, HmFormal1, arrayType, argName, arraylen);
	  } else {		// dir_OUT dir_INOUT
	    // gzheng
	    // add & before arguments
            spew(Hi, HiFormal21, arrayType, argName, arraylen); 
            spew(Ci, CiFormal21, arrayType, argName, arraylen);
            spew(Ccc, CccFormal1, arrayType, argName, arraylen);
            spew(Ccg, CcgFormal1, arrayType, argName, arraylen);
            spew(Hm, HmFormal1, arrayType, argName, arraylen);
//            cerr << "LIMIT: Out arguments not supported yet.\n";
	  }
        } else {
	  char *argType = a->field_type()->local_name()->get_string();
          char argName[1024];
          sprintf(argName, "arg%d", argnum);
	  if(a->direction() == AST_Argument::dir_IN) {
            spew(Hi, HiFormal0, argType);
            spew(Ci, CiFormal0, argType, argName);
            spew(Ccc, CccFormal0, argType, argName);
            spew(Ccg, CcgFormal0, argType, argName);
            spew(Hm, HmFormal0, argType, argName);
	  } else {	// gzheng
            spew(Hi, HiFormal01, argType);	       // add & before arguments
            spew(Ci, CiFormal01, argType, argName);
            spew(Ccc, CccFormal0, argType, argName);
            spew(Ccg, CcgFormal0, argType, argName);
            spew(Hm, HmFormal0, argType, argName);
//            cerr << "LIMIT: Out arguments not supported yet.\n";
	  }
        }
      } else { // not an argument
        cerr << "LIMIT: Only arguments within method parameter list\n";
      }
      i->next();
      if (!(i->is_done())) { // output commas between parameters
        spew(Hi, HiFormal1);
        spew(Ci, CiFormal1);
      }
    }
    delete i;
  } else if(passnum == 2) { // Produce Marshalling Code
    i = new UTL_ScopeActiveIterator(bop, UTL_Scope::IK_decls);
    while (!(i->is_done())) {
      d = i->item();
      argnum++;
      if (d->node_type() == AST_Decl::NT_argument) {
        be_argument *a = be_argument::narrow_from_decl(d);
        if (a->isArray()) {
	  be_array *ar = be_array::narrow_from_decl(a->field_type());
	  char *arrayType = ar->base_type()->local_name()->get_string();
	  unsigned long arraySizeNum = ar->dims()[0]->ev()->u.ulval;
          char arraylen[128];
          sprintf(arraylen, "%ld", arraySizeNum);
          char argName[1024];
          sprintf(argName, "arg%d", argnum);
	  if(a->direction() == AST_Argument::dir_IN || a->direction() == AST_Argument::dir_INOUT) {
            spew(Ci, CiActual1, argName, arraylen);
            spew(Ccc, CccActual0, argName);
            spew(Ccg, CcgActual0, argName);
	  } else if (a->direction() == AST_Argument::dir_OUT) {	// gzheng
            spew(Ccc, CccActual0, argName);
            spew(Ccg, CcgActual0, argName);
	  } else {
            cerr << "LIMIT: arguments not supported yet.\n";
          }
        } else {
	  char *argType = a->field_type()->local_name()->get_string();
          char argName[1024];
          sprintf(argName, "arg%d", argnum);
	  if(a->direction() == AST_Argument::dir_IN || a->direction() == AST_Argument::dir_INOUT) {
            spew(Ci, CiActual0, argName);
            spew(Ccc, CccActual0, argName);
            spew(Ccg, CcgActual0, argName);
	  } else if (a->direction() == AST_Argument::dir_OUT) {
            spew(Ccc, CccActual0, argName);
            spew(Ccg, CcgActual0, argName);
	  } else {
            cerr << "LIMIT: Out arguments not supported yet.\n";
          }
        }
      } else { // not an argument
        cerr << "LIMIT: Only arguments within method parameter list\n";
      }
      i->next();
      if (!(i->is_done())) { // output commas between parameters
        spew(Ccc, CccActual1);
        spew(Ccg, CcgActual1);
      }
    }
    delete i;
  }
}

const char *HiMethod0 = // returnType, methodname
"    \01 \02("
;

const char *HiMethod1 =
");\n"
;

const char *HmMethod0 = // messagename
"class \01 : public CMessage_\01 {\n"
"  public:\n"
;

const char *HmMethod1 =
"};\n"
"\n"
;

const char *HccMethod0 = // methodname, messagename
"    void \01(\02 *);\n"
;

// gzheng
const char *HccMethods_1 = // methodname, messagename
"    \02 * \01(\02 *);\n"
;

const char *HcgMethod0 = // methodname, messagename
"    void \01(\02 *);\n"
;

const char *CiMethod0 = // returnType, classname, methodname
"\01 CI\02::\03("
;

const char *CiMethod1 = // msgName
")\n"
"{\n"
"  \01 *msg = new \01;\n"
;

const char *CiMethod2 = // classname, methodname, messagename
"  if(isChare()) {\n"
"    CkChareID cid = cih.ciCID();\n"
"    CkSendMsg(CProxy_CC\01::__idx_\02_\03,msg,&cid);\n"
"  } else if(ciGetProc()==CI_PE_ALL) {\n"
"    CkBroadcastMsgBranch(CProxy_CG\01::__idx_\02_\03,msg,cih.ciGID());\n"
"  } else {\n"
"    CkSendMsgBranch(CProxy_CG\01::__idx_\02_\03,msg,cih.ciGID(),ciGetProc());\n"
"  }\n"
"}\n"
"\n"
;

// gzheng
const char *CiMethods_3 = // classname, methodname, messagename
"  if(isChare()) {\n"
"    CkChareID cid = cih.ciCID();\n"
"    msg = (\03 *)CkRemoteCall(CProxy_CC\01::__idx_\02_\03,msg,&cid);\n"
"  } else if(ciGetProc()==CI_PE_ALL) {\n"
"//    CkBroadcastMsgBranch(CProxy_CG\01::__idx_\02_\03,msg,cih.ciGID());\n"
"  } else {\n"
"//    CkRemoteBranchCall(CProxy_CG\01::__idx_\02_\03,msg,cih.ciGID(),ciGetProc());\n"
"  }\n"
;

const char *CiMethods_4 = //
"}\n"
"\n"
;

const char *CccMethod0 = // classname, methodname, messagename
"void CC\01::\02(\03 *msg)\n"
"{\n"
;

// gzheng
const char *CccMethods_1 = // classname, methodname, messagename
"\03 * CC\01::\02(\03 *msg)\n"
"{\n"
;

const char *CccMethod1 = // methodname
"  obj->\01("
;

// gzheng split CccMethod2 to two parts
const char *CccMethod2 =
");\n"
;

const char *CccMethod3 =
"}\n"
"\n"
;

const char *CccMethods_3 =
"  return msg;\n"
"}\n"
"\n"
;

const char *CcgMethod0 = // classname, methodname, messagename
"void CG\01::\02(\03 *msg)\n"
"{\n"
;

const char *CcgMethod1 = // methodname
"  obj->\01("
;

const char *CcgMethod2 =
");\n"
"}\n"
"\n"
;

const char *IccMethod0 = // methodname, messagename
"  entry void \01(\02 *);\n"
;

// gzheng
const char *IccMethods_1 = // methodname, messagename
"  entry [sync] void \01(\02 *);\n"
;

const char *IcgMethod0 = // methodname, messagename
"  entry void \01(\02 *);\n"
;

// gzheng
const char *IcgMethods_1 = // methodname, messagename
"  entry [sync] void \01(\02 *);\n"
;

const char *ImMethod0 = // messagename
"message \01;\n"
;

// gzheng
const char *CiMethod_1 = // argname
"  \01 = msg->\01;\n"
;

const char *CccMethod1_1 = // argname
"  msg->\01 = \01;\n"
;

void
BE_produce_operation(AST_Decl *d_in, AST_Interface *parent_interface)
{
  be_operation *bop = be_operation::narrow_from_decl(d_in);
  UTL_ScopeActiveIterator   *i;
  UTL_StrlistActiveIterator *si;
  UTL_ExceptlistActiveIterator *ei;
  AST_Decl		    *d;
  AST_Exception		    *e;
  String		    *s;

  // gzheng
  // check if has OUT
  int out_attr = 0;
  i = new UTL_ScopeActiveIterator(bop, UTL_Scope::IK_decls);
  while (!(i->is_done())) {
      d = i->item();
      if (d->node_type() == AST_Decl::NT_argument) {
        be_argument *a = be_argument::narrow_from_decl(d);
        if(a->direction() == AST_Argument::dir_OUT || 
           a->direction() == AST_Argument::dir_INOUT) {
		out_attr = 1;
 	}
      }
      i->next();
  }
  delete i;

  char *classname = parent_interface->local_name()->get_string();
  char *methodname = bop->local_name()->get_string();
  char msgName[1024];
  if(strcmp(classname,methodname)==0) { // Constructor
    if(!bop->hasParameter()) { // Default Constructor
      return;
    }
    spew(Hi, HiMethod0, "void", "ciCreate");
    spew(Ci, CiMethod0, "void", classname, "ciCreate");
  } else { // Ordinary Method
    char *returnTypeName = bop->return_type()->local_name()->get_string();
    spew(Hi, HiMethod0, returnTypeName, methodname);
    spew(Ci, CiMethod0, returnTypeName, classname, methodname);
    if(bop->hasParameter()) { // Message Needed
      int msgnum = bop->getMarshallMessageNumber();
      sprintf(msgName, "CIMsg%s%s%d", classname, methodname, msgnum);
      spew(Im, ImMethod0, msgName);
      spew(Hm, HmMethod0, msgName);
    } else {
      strcpy(msgName, "CIMsgEmpty");
    }
    // gzheng
    if( out_attr == 0 ) {
       spew(Icc, IccMethod0, methodname, msgName);
       spew(Icg, IcgMethod0, methodname, msgName);
       spew(Hcc, HccMethod0, methodname, msgName);
       spew(Hcg, HcgMethod0, methodname, msgName);
       spew(Ccc, CccMethod0, classname, methodname, msgName);
    }
    else {
       spew(Icc, IccMethods_1, methodname, msgName);
       spew(Icg, IcgMethods_1, methodname, msgName);
       spew(Hcc, HccMethods_1, methodname, msgName);
       spew(Hcg, HcgMethod0, methodname, msgName);
       spew(Ccc, CccMethods_1, classname, methodname, msgName);
    }
    spew(Ccg, CcgMethod0, classname, methodname, msgName);
  }
  BE_produce_parameters(bop, 1); // Produce Formals
  spew(Hi, HiMethod1);
  spew(Ci, CiMethod1, msgName);
  spew(Hm, HmMethod1);
  spew(Ccc, CccMethod1, methodname);
  spew(Ccg, CcgMethod1, methodname);
  BE_produce_parameters(bop, 2); // Produce Marshalling Code
  spew(Ccc, CccMethod2);
  spew(Ccg, CcgMethod2);

  // gzheng
  if( out_attr == 0 ) {
  	spew(Ci, CiMethod2, classname, methodname, msgName);
  	spew(Ccc, CccMethod3);
  }
  else  {
  	spew(Ci, CiMethods_3, classname, methodname, msgName);
	// pick up all the OUT arguments
	int argnum = 0;
  	i = new UTL_ScopeActiveIterator(bop, UTL_Scope::IK_decls);
  	while (!(i->is_done())) {
	    argnum++;
      	    d = i->item();
      	    if (d->node_type() == AST_Decl::NT_argument) {
        	be_argument *a = be_argument::narrow_from_decl(d);
        	if(a->direction() == AST_Argument::dir_OUT || 
                   a->direction() == AST_Argument::dir_INOUT) {
			// make assignment back
          		char argName[1024];
          		sprintf(argName, "arg%d", argnum);
  			spew(Ci, CiMethod_1, argName);
			// in CCclass, assign message back
  			spew(Ccc, CccMethod1_1, argName);
 		}
            }
            i->next();
  	}
  	delete i;
	// add "}" to close the function
  	spew(Ci, CiMethods_4);
  	spew(Ccc, CccMethods_3);
  }
}

//----------------------------------------------------------------------
void
BE_produce_attribute(AST_Decl *d)
{
  be_attribute *a = be_attribute::narrow_from_decl(d);

//   cout << "NT_attr " << a->field_type()->local_name()->get_string() << " "
//        << a->local_name()->get_string() << endl;

//   o << (pd_readonly == I_TRUE ? "readonly" : "") << " attribute ";
//   AST_Field::dump(o);
}

const char *HiClass0 = // classname
"class CI\01 {\n"
"  private:\n"
"    CIHandle cih;\n"
"    CIMethodParams cim;\n"
"  public:\n"
"    CI\01() {}\n"
"    CI\01(CIHandle hndl) { cih = hndl; }\n"
"    int isChare(void) { return cih.isChare(); }\n"
"    CkChareID ciCID(void) { return cih.ciCID(); }\n"
"    int ciGID(void) { return cih.ciGID(); }\n"
"    int ciGetProc(void) { return cih.ciGetProc(); }\n"
"    CI\01 & ciSetProc(int _proc) { \n"
"      cih.ciSetProc(_proc); return *this; \n"
"    }\n"
"    CI\01 & ciSetPrioWords(int words) { \n"
"      cim.ciSetPrioWords(words); return *this; \n"
"    }\n"
"    CI\01 & ciSetPrioVec(int *_vec) { \n"
"      cim.ciSetPrioVec(_vec); return *this; \n"
"    }\n"
"    CI\01 & ciSetPrio(int words, int *_vec) { \n"
"      cim.ciSetPrio(words, _vec); return *this; \n"
"    }\n"
"    CI\01 & ciSetIntPrio(int _prio) { \n"
"      cim.ciSetIntPrio(_prio); return *this; \n"
"    }\n"
"    CI\01 & ciSetSynch(int _synch) { \n"
"      cim.ciSetSynch(_synch); return *this; \n"
"    }\n"
"    void ciDelete(void);\n"
"    void ciCreate(void);\n"
;

const char *HiClass1 =
"};\n"
"\n"
;

const char *HccClass0 = // classname
"class CC\01 : public Chare {\n"
"  private:\n"
"    \01 *obj;\n"
"  public:\n"
"    CC\01(CIMsgEmpty *);\n"
"    void ciDelete(CIMsgEmpty *);\n"
;

const char *HccClass1 =
"};\n"
"\n"
;

const char *HcgClass0 = // classname
"class CG\01 : public Group {\n"
"  private:\n"
"    \01 *obj;\n"
"  public:\n"
"    CG\01(CIMsgEmpty *);\n"
"    void ciDelete(CIMsgEmpty *);\n"
;

const char *HcgClass1 =
"};\n"
"\n"
;

const char *CiClass0 = // classname
"void CI\01::ciDelete(void) {\n"
"  CIMsgEmpty *msg = new CIMsgEmpty;\n"
"  if(isChare()) {\n"
"    CkChareID cid = cih.ciCID();\n"
"    CkSendMsg(CProxy_CC\01::__idx_ciDelete_CIMsgEmpty,msg,&cid);\n"
"  } else if(ciGetProc()==CI_PE_ALL) {\n"
"    CkBroadcastMsgBranch(CProxy_CG\01::__idx_ciDelete_CIMsgEmpty,msg,cih.ciGID());\n"
"  } else {\n"
"    CkSendMsgBranch(CProxy_CG\01::__idx_ciDelete_CIMsgEmpty,msg,cih.ciGID(),ciGetProc());\n"
"  }\n"
"}\n"
"\n"
"void CI\01::ciCreate(void) {\n"
"  CIMsgEmpty *msg = new CIMsgEmpty;\n"
"  if(isChare()) {\n"
"    CkChareID cid;\n"
"    CProxy_CC\01::ckNew(msg,&cid,cih.ciGetProc());\n"
"    cih.setCID(cid);\n"
"  } else {\n"
"    cih.setGID(CProxy_CG\01::ckNew(msg));\n"
"  }\n"
"}\n"
"\n"
;

const char *CccClass0 = // classname
"void CC\01::ciDelete(CIMsgEmpty *msg) {\n"
"  delete msg;\n"
"  obj->~\01();\n"
"  char *orig = (char *) obj - sizeof(CIHandle);\n"
"  delete[] orig;\n"
"}\n"
"\n"
"CC\01::CC\01(CIMsgEmpty *msg) {\n"
"  delete msg;\n"
"  char *space = new char[sizeof(CIHandle)+sizeof(\01)];\n"
"  new (space) CIHandle(thishandle);\n"
"  obj = new (space+sizeof(CIHandle)) \01;\n"
"}\n"
"\n"
;

const char *CcgClass0 = // classname
"void CG\01::ciDelete(CIMsgEmpty *msg) {\n"
"  delete msg;\n"
"  obj->~\01();\n"
"  char *orig = (char *) obj - sizeof(CIHandle);\n"
"  delete[] orig;\n"
"}\n"
"\n"
"CG\01::CG\01(CIMsgEmpty *msg) {\n"
"  delete msg;\n"
"  char *space = new char[sizeof(CIHandle)+sizeof(\01)];\n"
"  new (space) CIHandle(thisgroup);\n"
"  obj = new (space+sizeof(CIHandle)) \01;\n"
"}\n"
"\n"
;

const char *IccClass0 = // classname
"chare CC\01 {\n"
"  entry CC\01(CIMsgEmpty *);\n"
"  entry void ciDelete(CIMsgEmpty *);\n"
;

const char *IccClass1 = 
"};\n"
"\n"
;

const char *IcgClass0 = // classname
"group CG\01 {\n"
"  entry CG\01(CIMsgEmpty *);\n"
"  entry void ciDelete(CIMsgEmpty *);\n"
;

const char *IcgClass1 = 
"};\n"
"\n"
;

//----------------------------------------------------------------------
void
BE_produce_interface(AST_Decl *d)
{
  UTL_ScopeActiveIterator	*i;
  AST_Interface			*m;

  m = AST_Interface::narrow_from_decl(d);

  char *classname = m->local_name()->get_string();
  spew(Hi, HiClass0, classname);
  spew(Hcc, HccClass0, classname);
  spew(Hcg, HcgClass0, classname);
  spew(Ci, CiClass0, classname);
  spew(Ccc, CccClass0, classname);
  spew(Ccg, CcgClass0, classname);
  spew(Icc, IccClass0, classname);
  spew(Icg, IcgClass0, classname);
  i = new UTL_ScopeActiveIterator(m, UTL_Scope::IK_both);
  int count = 0;
  while (!(i->is_done())) {
    d = i->item();
    count++;
    switch(d->node_type()){
      case AST_Decl::NT_attr:
        BE_produce_attribute(d);
        break;
      case AST_Decl::NT_op:
        BE_produce_operation(d, m);
        break;
      default:
        cerr << "LIMIT: Only Attributes and Operations within interface\n";
        break;
      }
      i->next();
    }
  delete i;
  spew(Hi, HiClass1);
  spew(Hcc, HccClass1);
  spew(Hcg, HcgClass1);
  spew(Icc, IccClass1);
  spew(Icg, IcgClass1);
}

const char *HiModule0 = // modulename
"// Begin Module \01\n"
"\n"
;

const char *HiModule1 = // modulename
"// End Module \01\n"
"\n"
;

const char *CiModule0 = // modulename
"// Begin Module \01\n"
"\n"
;

const char *CiModule1 = // modulename
"// End Module \01\n"
"\n"
;

//----------------------------------------------------------------------
void
BE_produce_module(AST_Decl *d)
{
  UTL_ScopeActiveIterator	*i;
  AST_Module			*m;

  m = AST_Module::narrow_from_decl(d);
  spew(Hi, HiModule0, m->local_name()->get_string());
  spew(Ci, CiModule0, m->local_name()->get_string());
  i = new UTL_ScopeActiveIterator(m, UTL_Scope::IK_both);
  int count = 0;
  while (!(i->is_done())) {
    d = i->item();
    count++;
    switch(d->node_type()){
      case AST_Decl::NT_interface:
        BE_produce_interface(d);
        break;
      default:
        if (!d->imported()) {
          cerr << "LIMIT: Only Interfaces within Modules at top level\n";
        }
        break;
    }
    i->next();
  }
  delete i;
  spew(Hi, HiModule1, m->local_name()->get_string());
  spew(Ci, CiModule1, m->local_name()->get_string());
}

static char *baseName;

const char *CIhTop0 = // basename
"#ifndef _CI_\01_H_\n"
"#define _CI_\01_H_\n"
"#include <new.h>\n"
"#include \"charm++.h\"\n"
"#include \"idl.h\"\n"
"#include \"\01.h\"\n"
"#include \"CI\01.decl.h\"\n"
"\n"
;

const char *CIhTop1 = // basename
"\n"
"#endif\n"
;

const char *CIiTop1 = // basename
"\n"
"}\n"
;

const char *CIcTop0 = // basename
"#include \"CI\01.h\"\n"
"\n"
;

const char *CIcTop1 = // basename
"\n"
"#include \"CI\01.def.h\"\n"
;

const char *CIiTop0 = // basename
"module CI\01 {\n"
"extern message CIMsgEmpty;\n"
;

//----------------------------------------------------------------------
// Create the various files that the BE outputs.  Output the
// file-level opening stuff.
void
initialize()
{
  CIi = new XString; 
  CIh = new XString; 
  CIc = new XString;
  Im = new XString; 
  Icc = new XString; 
  Icg = new XString;
  Hm = new XString; 
  Hi = new XString; 
  Hcc = new XString; 
  Hcg = new XString;
  Cm = new XString; 
  Ci = new XString; 
  Ccc = new XString; 
  Ccg = new XString;
  char *filename = idl_global->main_filename()->get_string();
  baseName = getBasename(filename);
  char *createName = new char[1024];
  sprintf(createName, "CI%s.h", baseName);
  FH = new ofstream(createName);
  sprintf(createName, "CI%s.C", baseName);
  FC = new ofstream(createName);
  sprintf(createName, "CI%s.ci", baseName);
  FCI = new ofstream(createName);
  delete createName;
  spew(CIh, CIhTop0, baseName);
  spew(CIc, CIcTop0, baseName);
  spew(CIi, CIiTop0, baseName);
}

// At the top level we can have modules, interfaces, constants or type
// declarations. @@
void
BE_produce_top_level()
{
  UTL_ScopeActiveIterator	*i;
  AST_Decl			*d;

  i = new UTL_ScopeActiveIterator(idl_global->root(), UTL_Scope::IK_both);

  int count = 0;
  while (!(i->is_done())) {
    d = i->item();
    count++;

    switch(d->node_type()){
      case AST_Decl::NT_module:
	BE_produce_module(d);
	break;
      case AST_Decl::NT_interface:
	BE_produce_interface(d);
	break;
      case AST_Decl::NT_const:
      case AST_Decl::NT_struct:
	break;
      default:
	if (!d->imported()) {
	  cerr << "LIMIT: Only Modules and Interfaces at top level\n";
	}
	break;
      }
    i->next();
  }
  delete i;
}

// Output the file-level closing stuff; Close the output files.
void
clean_up()
{
  CIh->append(Hm->get_string());
  CIh->append(Hi->get_string());
  CIh->append(Hcc->get_string());
  CIh->append(Hcg->get_string());
  delete Hm; delete Hi; delete Hcc; delete Hcg;

  CIc->append(Cm->get_string());
  CIc->append(Ci->get_string());
  CIc->append(Ccc->get_string());
  CIc->append(Ccg->get_string());
  delete Cm; delete Ci; delete Ccc; delete Ccg;

  CIi->append(Im->get_string());
  CIi->append(Icc->get_string());
  CIi->append(Icg->get_string());
  delete Im; delete Icg; delete Icc;

  spew(CIh, CIhTop1, baseName);
  spew(CIc, CIcTop1, baseName);
  spew(CIi, CIiTop1, baseName);

  *FH << CIh->get_string();
  *FC << CIc->get_string();
  *FCI << CIi->get_string();

  delete CIh; delete CIc; delete CIi;
  delete FH; delete FC; delete FCI;
  delete baseName;
}

/*
 * Do the work of this BE.
 */
void
BE_produce()
{
  initialize();

  BE_produce_top_level();
  
  clean_up();
}

//----------------------------------------------------------------------
/*
 * Abort this run of the BE
 */
void
BE_abort()
{
}

/*

COPYRIGHT

Copyright 1992, 1993, 1994 Sun Microsystems, Inc.  Printed in the United
States of America.  All Rights Reserved.

This product is protected by copyright and distributed under the following
license restricting its use.

The Interface Definition Language Compiler Front End (CFE) is made
available for your use provided that you include this license and copyright
notice on all media and documentation and the software program in which
this product is incorporated in whole or part. You may copy and extend
functionality (but may not remove functionality) of the Interface
Definition Language CFE without charge, but you are not authorized to
license or distribute it to anyone else except as part of a product or
program developed by you or with the express written consent of Sun
Microsystems, Inc. ("Sun").

The names of Sun Microsystems, Inc. and any of its subsidiaries or
affiliates may not be used in advertising or publicity pertaining to
distribution of Interface Definition Language CFE as permitted herein.

This license is effective until terminated by Sun for failure to comply
with this license.  Upon termination, you shall destroy or return all code
and documentation for the Interface Definition Language CFE.

INTERFACE DEFINITION LANGUAGE CFE IS PROVIDED AS IS WITH NO WARRANTIES OF
ANY KIND INCLUDING THE WARRANTIES OF DESIGN, MERCHANTIBILITY AND FITNESS
FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR ARISING FROM A COURSE OF
DEALING, USAGE OR TRADE PRACTICE.

INTERFACE DEFINITION LANGUAGE CFE IS PROVIDED WITH NO SUPPORT AND WITHOUT
ANY OBLIGATION ON THE PART OF Sun OR ANY OF ITS SUBSIDIARIES OR AFFILIATES
TO ASSIST IN ITS USE, CORRECTION, MODIFICATION OR ENHANCEMENT.

SUN OR ANY OF ITS SUBSIDIARIES OR AFFILIATES SHALL HAVE NO LIABILITY WITH
RESPECT TO THE INFRINGEMENT OF COPYRIGHTS, TRADE SECRETS OR ANY PATENTS BY
INTERFACE DEFINITION LANGUAGE CFE OR ANY PART THEREOF.

IN NO EVENT WILL SUN OR ANY OF ITS SUBSIDIARIES OR AFFILIATES BE LIABLE FOR
ANY LOST REVENUE OR PROFITS OR OTHER SPECIAL, INDIRECT AND CONSEQUENTIAL
DAMAGES, EVEN IF SUN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Use, duplication, or disclosure by the government is subject to
restrictions as set forth in subparagraph (c)(1)(ii) of the Rights in
Technical Data and Computer Software clause at DFARS 252.227-7013 and FAR
52.227-19.

Sun, Sun Microsystems and the Sun logo are trademarks or registered
trademarks of Sun Microsystems, Inc.

SunSoft, Inc.
2550 Garcia Avenue
Mountain View, California  94043

NOTE:

SunOS, SunSoft, Sun, Solaris, Sun Microsystems or the Sun logo are
trademarks or registered trademarks of Sun Microsystems, Inc.

 */

// #pragma ident "%@(#)BE_produce.cc	1.16% %92/06/10% Sun Microsystems"
