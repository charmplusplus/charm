// BE_produce.cc - Produce the work of the BE - does nothing in the
//		   dummy BE

#include	"idl.hh"
#include	"idl_extern.hh"
#include	"be.hh"

//----------------------------------------------------------------------
// TO DO

// test single class generation - DONE
// implement multi-class generation - DONE
// debug multi-file hello type program
// test multiple classes in a file - DONE
// primitive types: char, [unsigned] short, [unsigned] long, float, double- DONE
// struct's - DONE
// nested types - DONE for structs, etc.
// simple arrays - design done, code it. - DONE
// parameter passing: in, out, inout - design done, code it. - DONE
//   use futures to get back a value - DONE
// constructors - DONE
// can a constructor have inout parameters ? NO.  Why not ?
// Do we need to inherit ?  class myClass1 /*: public chare_object*/ {
// no, since user uses only stub, never the class.
// threaded keyword - DONE
// Actually, the word is just ignored; the user has to create a .ci
// file with the threaded stuff.

// intermixing itc++ and idl++ example

// const's - DONE
// enums - DONE, bug: spews twice, why ?

// return values
// cleanup M_Empty
// cleanup unnecessary params in return messages

// which items are used by the IDL, which items belong to the class
// being declared ?

// sequences: varsize arrays
// VARSIZE myClass1->myMethod1SETSIZE(-, 100, -)->myMethod1(a, b[], c)
// BOC's
// scoping
// structs within interfaces ?
// any - requires RTTI, right ?
// inheritance
// exceptions
// do away with user writing .ci file, idl need to understand message,
//   threaded.

//----------------------------------------------------------------------
#include <fstream.h>
#include <stdio.h>
#include <libgen.h>
#include <stdlib.h>

#include "idlString.hh"

const int MAX_STR_LEN = 10000;

#include<assert.h>

// Assumes s can grow to MAX_STR_LEN long.
// Basically, a safe form of strcat().
static void
string_append(char *const s, const char *const a)
{
  assert( strlen(s) + strlen(a) < MAX_STR_LEN );
  int i=0, j=0;
  for(i = strlen(s), j=0; a[j]!='\0'; i++, j++)
    s[i] = a[j];
  s[i] = '\0';
}

static void
DebugPrint(char *s)
{
  cerr << s << endl;
}

//----------------------------------------------------------------------
// Replace \1 by a1, \2 by a2, etc. in string b.  Sends output to
// outStream.
static void
spew(ostream& outStream, const char*b,
     char *a1 = "ERROR", char *a2 = "ERROR",
     char *a3 = "ERROR", char *a4 = "ERROR", char *a5 = "ERROR")
{
  int i;
  for(i=0; i<strlen(b); i++){
    switch(b[i]){
    case '\01':
      outStream << a1;
      break;
    case '\02':
      outStream << a2;
      break;
    case '\03':
      outStream << a3;
      break;
    case '\04':
      outStream << a4;
      break;
    case '\05':
      outStream << a5;
      break;
    default:
      outStream << b[i];
    }
  }
}

// Replace \1 by a1, \2 by a2, etc. in string b, and appends the
// result to outString.  Assumes outString is MAX_STR_LEN long.
static void
spew(char outString[], const char*b,
     char *a1 = "ERROR", char *a2 = "ERROR",
     char *a3 = "ERROR", char *a4 = "ERROR", char *a5 = "ERROR")
{
  int eosi = strlen(outString); // one past end of string index
  int i;
  for(i=0; i<strlen(b); i++){
    assert(eosi < MAX_STR_LEN-1);
    switch(b[i]){
    case '\01':
      outString[eosi] = '\0';
      string_append(outString, a1);
      eosi += strlen(a1);
      break;
    case '\02':
      outString[eosi] = '\0';
      string_append(outString, a2);
      eosi += strlen(a2);
      break;
    case '\03':
      outString[eosi] = '\0';
      string_append(outString, a3);
      eosi += strlen(a3);
      break;
    case '\04':
      outString[eosi] = '\0';
      string_append(outString, a4);
      eosi += strlen(a4);
      break;
    case '\05':
      outString[eosi] = '\0';
      string_append(outString, a5);
      eosi += strlen(a5);
      break;
    default:
      outString[eosi++] = b[i];
    }
  }
}

//----------------------------------------------------------------------
// Given "path/file.ext" we return "file".
// @@ Given "path/file.b.ext" we return "file.b".
// Caller must delete the returned string.
char *
get_basename_noextension(char *pathname)
{
	char c = '/';
	int i = 0;

	for(i=strlen(pathname)-1; i>=0; i--)
	  if (pathname[i] == c)
	    break;
	assert(i != 0);

	i++;
	char *basenm = new char[strlen(pathname) - i];

	int j;
	for(j=0; pathname[i]; i++, j++)
	  if (pathname[i] != '.')
	    basenm[j] = pathname[i];
	  else {
	    basenm[j] = '\0';
	    break;
	  }

	cout << "MODULE " << basenm << endl;
	return basenm;
}

// If the file is a/b/c/m.idl
//  we return a/b/c/m
// If the file is a/b/c/m.n.idl
//  we return a/b/c/m.n
// Caller must delete the returned string.
char *
get_fullname_noextension(char *pathname)
{
	char c = '.';
	int i = 0;

	for(i=strlen(pathname)-1; i>=0; i--)
	  if (pathname[i] == c)
	    break;
	assert(i != 0);

	char *fullnm = new char[i+1];
	strncpy(fullnm, pathname, i);
	fullnm[i] = '\0';

	return fullnm;
}

//----------------------------------------------------------------------
// Files to output to

ofstream *MESGH; // messages.h
ofstream *SKELH; // skeleton.h
ofstream *SKELC; // skeleton.C
ofstream *SKELCI; // front of skeleton.ci
ofstream *SKELCIREST; // tail of skeleton.ci
ofstream *STUBH; // stub.h
ofstream *STUBC; // stub.C
ofstream *STUBCI; // stub.ci
ofstream *STUBCIREST; // tail of stub.ci

// Global variables
char *fileBaseNameNoExt = 0;
char *fullNameNoExt = 0;

//----------------------------------------------------------------------
//This is the actual code that we output.  We use the spew function to
//replace \01, \02, etc. by the desired values.

// header stuff
const char *SKELHDR10 = // filename
"#ifndef \01_SKELETON_H\n"
"#define \01_SKELETON_H\n"
"// \01_skeleton.h\n"
"// The skeleton basically contains the class and forwards all messages\n"
"// received to it by invoking local function calls.\n"
"\n"
"//#include \"\01_messages.h\"\n"
"#include \"\01_skeleton.top.h\"\n"
"\n"
;

// declare skeleton class and system-defined functions
const char *SKELHDR20 = // classname
"class \01_skeleton : public skel_object {\n"
"  private:\n"
"    \01 *myObj1;\n"
"\n"
"  public:\n"
"    \01_skeleton(M_Empty *m);\n"
"    void delete_\01_skeleton(M_Empty *m);\n"
"\n"
;

// declare user-defined functions
const char *SKELHDR30 = // method return type, name, message type
"    \01 \02(\03 *m);\n"
;

// end classname
const char *SKELHDR40 =
"};\n"
;

// end filename
const char *SKELHDR50 =
"\n"
"#endif\n"
;

//----------------------------------------------------------------------

// header stuff
const char *SKELCEE10 = // filename
"// \01_skeleton.C\n"
"\n"
"#include \"\01_skeleton.h\"\n"
"\n"
;

// define system-defined functions
const char *SKELCEE20 = // classname
"//----------------------------------------\n"
"void \01_skeleton::delete_\01_skeleton(M_Empty *m) {\n"
"  delete m;\n"
"}\n"
"\n"
;

// define user-defined functions, call local object
// classname, methodname, message type, return type
const char *SKELCEE30 = 
"//----------------------------------------\n"
"\04 \01_skeleton::\02(\03 *m) {\n"
"  myObj1->\02("
;

// define user-defined constructor, call local object
// classname, methodname, message type, return type
const char *SKELCEE35 = 
"//----------------------------------------\n"
"\04 \01_skeleton::\02(\03 *m) {\n"
"  myObj1 = new \05("
;

// list arguments
const char *SKELCEE40 = // argument name
"m->\01"
;

// list arguments
// const char *SKELCEE42 = // array argument name, size
// "m->\01[\02]"
// ;

// also introduce comma's between arguments
const char *SKELCEE45 = //
", "
;

// end parameters
const char *SKELCEE50 =
");\n"
;

// return value stuff
const char *SKELCEE52 = // return message type
"  \01 *mret = new (MsgIndex(\01)) \01;\n"
;

// assign arguments to return message elements
const char *SKELCEE53 = // argument name
"  mret->\01 = m->\01;\n"
;

// return message
const char *SKELCEE54 =
"  return mret;\n"
;

// end method
const char *SKELCEE55 =
"  delete m;\n"
"}\n"
"\n"
;

// end filename
const char *SKELCEE60 = // filename
"//----------------------------------------\n"
"#include \"\01_skeleton.bot.h\"\n"
;

//----------------------------------------------------------------------

// header stuff
const char *SKELCI10 = // filename
"// \01_skeleton.ci\n"
"\n"
// @@ should be extern, but then we need message.ci
"message M_Empty;\n"
;

// declare marshalling messages
const char *SKELCI20 = // message type
"message \01;\n"
"\n"
;

// declare system-defined functions
const char *SKELCI30 = // class name
"chare \01_skeleton {\n"
"  entry \01_skeleton(M_Empty *);\n"
"  entry delete_\01_skeleton(M_Empty *);\n"
"\n"
;

// declare user-defined functions
const char *SKELCI40 = // operation name, message type, special return type
"  \03 entry \01(\02 *);\n"
;

// end classname and filename
const char *SKELCI50 =
"};\n"
;

//----------------------------------------------------------------------

// class specific messages

// header and system-defined stuff
const char *MESGOP10 = // filename
"#ifndef \01_MESSAGE_H\n"
"#define \01_MESSAGE_H\n"
"//#include \"ckdefs.h\"\n"
"//#include \"chare.h\"\n"
"//#include \"c++interface.h\"\n"
"#include \"charm++.h\"\n"
"\n"
"#ifndef COMMON_TEMPLATE\n"
"#define COMMON_TEMPLATE\n"
"template<class T>\n"
"void\n"
"copy_array(T to[], T from[], int size)\n"
"{\n"
"  for (int i=0; i < size; i++)\n"
"    to[i] = from[i];\n"
"};\n"
"#endif\n"
"\n"
"\n"
// @@ there should be a global messages file.
"#ifndef MESSAGE_H\n"
"#define MESSAGE_H\n"
"class skel_object;\n"
"class M_Empty;\n"
"// Right now contains nothing\n"
"class skel_object : public chare_object {\n"
"};\n"
"\n"
"class M_Empty : public comm_object {\n"
"};\n"
"#endif\n"
"\n"
"#include \"\01.h\"\n"
"\n"
;

//define messages
const char *MESGOP20 = // messagename
"class \01 : public comm_object {\n"
"public:\n"
;

const char *MESGOP30 = // argument type, name
"    \01 \02;\n"
;

const char *MESGOP35 = // array argument type, name, size
"    \01 \02[\03];\n"
;

// end messagename
const char *MESGOP40 =
"};\n"
;

// end filename
const char *MESGOP50 =
"\n"
"#endif\n"
"\n"
;

//----------------------------------------------------------------------
// @@ not yet implemented
const char *MESGCEE10 =
"\n"
"\n"
;

//----------------------------------------------------------------------

// header stuff
const char *STUBHDR10 = // filename
"#ifndef \01_STUB_H\n"
"#define \01_STUB_H\n"
"\n"
"#include \"\01_skeleton.h\"\n"
"\n"
;

// declare the stub class and system-defined functions
const char *STUBHDR20 = // classname
"class \01_stub : public chare_object {\n"
"  int shouldDelete; //virtual\n"
"  ChareIDType chare_Id; // virtual\n"
"public:\n"
"  \01_stub(int pe);\n"
"  \01_stub();\n"
"  \01_stub(ChareIDType id);\n"
"  ~\01_stub(void);\n"
"  ChareIDType getGlobalID() { return chare_Id; }\n"
"\n"
;

// declare user-defined functions
const char *STUBHDR30 = // method return type, name
"  \01 \02("
;

// pass by value
const char *STUBHDR40 = // argument type
"\01 "
;

// pass by reference
const char *STUBHDR41 = // argument type
"\01 &"
;

const char *STUBHDR42 = // array argument type, size
"\01[\02] "
;

// also introduce comma's between arguments
const char *STUBHDR45 =
", "
;

// end arguments
const char *STUBHDR50 =
");\n"
;

// declare system-defined helper function, if needed
const char *STUBHDR55 = // return message type with ptr, marshall message type,
 // methodname
"  \01 \03helper(\02 *);\n"
;

// end classname
const char *STUBHDR60 =
"};\n"
;

// end filename
const char *STUBHDR70 =
"\n"
"#endif\n"
;

//----------------------------------------------------------------------

// header stuff
const char *STUBCEE10 = // filename
"// \01_stub.C\n"
"\n"
"#include \"\01_stub.h\"\n"
"#include \"\01_skeleton.h\"\n"
"#include \"\01_stub.top.h\"\n"
"\n"
;

// define system-defined functions
const char *STUBCEE20 = // classname
"//\01_stub::\01_stub(int pe) {\n"
"//  // create new chare of type \01_skeleton on PE\n"
"//  // by sending a message of type M_Empty\n"
"//  // and store returned chareID in chareId;\n"
"//  M_Empty *msg = new (MsgIndex(M_Empty)) M_Empty;\n"
"//  new_chare2(\01_skeleton,M_Empty,msg,&chare_Id,pe);\n"
"//\n"
"//  shouldDelete = 1;\n"
"//}\n"
"\n"
"\01_stub::\01_stub() {\n"
"  // create new chare of type \01_skeleton on any PE\n"
"  // by sending a message of type M_Empty\n"
"  // and store returned chareID in chareId;\n"
"  M_Empty *msg = new (MsgIndex(M_Empty)) M_Empty;\n"
"  new_chare2(\01_skeleton,M_Empty,msg,&chare_Id,CK_PE_ANY);\n"
"\n"
"  shouldDelete = 1;\n"
"}\n"
"\n"
"\01_stub::\01_stub(ChareIDType id) {\n"
"  chare_Id = id;\n"
"  shouldDelete = 0;\n"
"}\n"
"\n"
"\01_stub::~\01_stub(void) {\n"
"  if(shouldDelete) {\n"
"    //M_Empty *m = new (MsgIndex(M_Empty)) M_Empty;\n"
"    //CSendMsg(\01_skeleton, delete_\01_skeleton, m, &chare_Id);\n"
"  } else {\n"
"    // do nothing\n"
"  }\n"
"}\n"
"\n"
"//------------------------------\n"
;

// begin to define user-defined function
const char *STUBCEE30 = // class name, method return type, method name
"\02 \01_stub::\03("
;

// OR begin to define user-defined constructor
const char *STUBCEE35 = // classname
"\01_stub::\01_stub("
;

// @@ make sure out/inout parameters are references

// list parameters
const char *STUBCEE40 = // argument type, name
"\01 \02"
;

// list parameters,  pass by reference
const char *STUBCEE41 = // argument type, name
"\01 &\02"
;

// list parameters
const char *STUBCEE42 = // array argument type, name, size
"\01 \02[\03]"
;

// also introduce comma's between parameters
const char *STUBCEE45 = //
", "
;

// end parameters
const char *STUBCEE50 =
") {\n"
;

// begin function body
const char *STUBCEE60 = // class name, method name, message type
//"  // CPrintf(\"\01_stub::\02: Message received %f\\n\", arg0);\n"
"  \03 *m = new (MsgIndex(\03)) \03;\n"
;

// // assign arguments to message elements
// const char *STUBCEE70 = // arguments
// "  m->%s = %s;\n"
// ;

// assign arguments to message elements
const char *STUBCEE71 = // arguments
"  m->\01 = \01;\n"
;

// // assign arguments to message elements
// const char *STUBCEE75 = // arguments
// "  copy_array(m->%s, %s, %s);\n"
// ;

// assign array arguments to message elements
const char *STUBCEE76 = // argument name, array size
"  copy_array(m->\01, \01, \02);\n"
;

// send off the asynchronous message
const char *STUBCEE80 = //class name,  method name
"  CSendMsg(\01_skeleton, \02, \03, m, &chare_Id);\n"
;

// OR send off a synchronous message
const char *STUBCEE82 = // return message type, method name
"  \01 * result;\n"
"  result = \02helper(m);\n"
"\n"
;

// assign return message elements to arguments 
const char *STUBCEE84 = // arguments
"  \01 = result->\01;\n"
;

// assign array message elements to arguments
const char *STUBCEE86 = // argument name, array size
"  copy_array(\01, result->\01, \02);\n"
;

// OR create the skeleton
const char *STUBCEE88 = //class name
// @@ what about specific PE's ?  We need to generate another constructor.
"  // create new chare of type \01_skeleton on PE\n"
"  // by sending a message of type M_Empty\n"
"  // and store returned chareID in chareId;\n"
"  new_chare2(\01_skeleton,\02,m,&chare_Id,0);\n"
"\n"
"  shouldDelete = 1;\n"
"\n"
;

// end user-defined function
const char *STUBCEE90 =
"}\n"
;

// if needed, helper function
const char *STUBCEE92 = // return message type, marshall message type,
 // class name, method name
"\01 * \03_stub::\04helper(\02 *m) {\n"
"  \01 * result;\n"
"  result = (\01 *) CRemoteCall(\03_skeleton,\04,\02,m,&chare_Id);\n"
"  return result;\n"
"}\n"
"\n"
;

// if needed, helper function without return type
const char *STUBCEE94 = // return message type (void), marshall message type,
 // class name, method name
"\01 \03_stub::\04helper(\02 *m) {\n"
"  \01 result;\n"
"  result = (\01 *) CRemoteCallFn(GetEntryPtr(\03_skeleton,\n"
"						      \04, \02), m, &chare_Id);\n"
"  return result;\n"
"}\n"
"\n"
;

// end
const char *STUBCEE95 = // filename
"\n"
"#include \"\01_stub.bot.h\"\n"
;

//----------------------------------------------------------------------

// declare marshalling messages
const char *STUBCI20 = // message type
"extern message \01;\n"
;

// declare system-defined functions
const char *STUBCI30 = // class name
"\n"
"chare \01_stub {\n"
;

// declare system-defined helper function, if needed
const char *STUBCI40 = // operation name, message type, special return type
"  threaded \03 entry \01helper(\02 *);\n"
;

// end classname and filename
const char *STUBCI50 =
"};\n"
;

//----------------------------------------------------------------------

// Traverse the method's parameters.
// Output code for:
//   1. the marshall message
//   2. the stub function (.h, .C)
//   3. the skeleton.C local invocation of the method
void
BE_produce_parameters(be_operation *bop,
		      char stubcPass1B[],
		      char mesghPass2[],
		      char skelcPass2[],
		      char stubcPass2[])
{
  UTL_ScopeActiveIterator   *i;
  AST_Decl		    *d;

  // @@ what about the return value ?

  i = new UTL_ScopeActiveIterator(bop, UTL_Scope::IK_decls);
  int argNum = -1;
  while (!(i->is_done())) {
    d = i->item();
    argNum++;

    if (d->node_type() == AST_Decl::NT_argument) {
      be_argument *a = be_argument::narrow_from_decl(d);
      //      cout << "NT_argument" << endl;
      char argName[100];
      sprintf(argName, "%s%d", "arg", argNum);

      if (a->isArray()) {
	be_array *ar = be_array::narrow_from_decl(a->field_type());

	char *arrayType = ar->base_type()->local_name()->get_string();
	unsigned long arraySizeNum = ar->dims()[0]->ev()->u.ulval;
	char arraySize[100];
        strcpy(arraySize,"");
	sprintf(arraySize, "%u", arraySizeNum);
	spew(*MESGH, MESGOP35, arrayType, argName, arraySize);
	if (bop->isReturnMessageNeeded())
	  spew(mesghPass2, MESGOP35, arrayType, argName, arraySize);
	spew(*SKELC, SKELCEE40, argName);
	if (bop->isReturnMessageNeeded())
	  spew(skelcPass2, SKELCEE53, argName);
	spew(*STUBH, STUBHDR42, arrayType, arraySize);
	spew(*STUBC, STUBCEE42, arrayType, argName, arraySize);

	//sprintf(ta, STUBCEE75, argName, argName, arraySize);
	spew(stubcPass1B, STUBCEE76, argName, arraySize);
	if (bop->isReturnMessageNeeded())
	  spew(stubcPass2, STUBCEE86, argName, arraySize);
      } else {
	char *argType = a->field_type()->local_name()->get_string();
	spew(*MESGH, MESGOP30, argType, argName);
	if (bop->isReturnMessageNeeded())
	  spew(mesghPass2, MESGOP30, argType, argName);
	spew(*SKELC, SKELCEE40, argName);
	if (bop->isReturnMessageNeeded())
	  spew(skelcPass2, SKELCEE53, argName);

	// do we need to use pass by reference for the parameter
	if(a->direction() == AST_Argument::dir_IN) {
	  spew(*STUBH, STUBHDR40, argType);
	  spew(*STUBC, STUBCEE40, argType, argName);
	} else {
	  spew(*STUBH, STUBHDR41, argType);
	  spew(*STUBC, STUBCEE41, argType, argName);
	}

	//sprintf(ta, STUBCEE70, argName);
	spew(stubcPass1B, STUBCEE71, argName);
	if (bop->isReturnMessageNeeded())
	  spew(stubcPass2, STUBCEE84, argName);
      }
      //      string_append(stubcPass1B, ta);

      //   o << direction_to_string(pd_direction) << " ";
      //   AST_Field::dump(o);
    } else {
      cerr << "LIMIT: Only arguments within method parameter list\n";
    }

    i->next();
    if (!(i->is_done())) { // output commas between parameters
      spew(*SKELC, SKELCEE45);
      spew(*STUBH, STUBHDR45);
      spew(*STUBC, STUBCEE45);
    }
  }
  delete i;

}

// There are three types of methods:
// - a constructor (no return values, creates a chare)
// - an asynchronous call (no return values)
// - a synchronous call (has a return value)
// Furthermore, the method may be non-threaded (default), or threaded.
void
BE_produce_operation(AST_Decl *d_in, AST_Interface *parent_interface)
  /* An operation is basically a C++ method. */
{
  //AST_Operation *op = AST_Operation::narrow_from_decl(d_in);
  be_operation *bop = be_operation::narrow_from_decl(d_in);

  if (bop->isConstructor())
    cerr << "DEBUG: isConstructor ";

  if (bop->isMarshallMessageNeeded())
    cerr << "DEBUG: isMarshallMessageNeeded ";

  if (bop->isReturnMessageNeeded())
    cerr << "DEBUG: isReturnMessageNeeded ";

  if (bop->isThreaded())
    cerr << "DEBUG: isThreaded found thread" << endl;
  else
    cerr << endl;

  ostream &o(cout);

  UTL_ScopeActiveIterator   *i;
  UTL_StrlistActiveIterator *si;
  UTL_ExceptlistActiveIterator *ei;
  AST_Decl		    *d;
  AST_Exception		    *e;
  String		    *s;

  // @@ if not threaded, set a flag to disallow use of futures in the function.
//   if (pd_flags == OP_oneway)
//     o << "oneway ";
//   else if (pd_flags == OP_idempotent)
//     o << "idempotent ";

  char *classname = parent_interface->local_name()->get_string();
  char *methodname = bop->local_name()->get_string();
  char *skelmethodname = new char[MAX_STR_LEN];
  strcpy(skelmethodname, methodname);
  char *stubmethodname = new char[MAX_STR_LEN];
  strcpy(stubmethodname, methodname);
  // If we have a constructor myClass::myClass, we need to change the
  // methodname to the constructor name for a stub and a skeleton.
  if (bop->isConstructor()) {
    strcat(skelmethodname, "_skeleton"); // in case of a ret mesg
    strcat(stubmethodname, "_stub");
  }

  // Create a message to marshall the method's parameters in.
  char marshallMesgTypeName[MAX_STR_LEN];
  sprintf(marshallMesgTypeName, "M%sM%d", classname,
	  bop->getMarshallMessageNumber());
  //  bop->p_marshallMesgTypeName = marshallMesgTypeName;

  // @@ document this: if there is no return value, the caller must
  // assume an asynchronous call was made to this function.

  // @@ I added the local_
  // Method's return type.
  char *methodReturnTypeName = bop->return_type()->local_name()->get_string();
  if (bop->isConstructor()) {
    methodReturnTypeName = "";
  }

  // If needed, create a message to marshall the method's return
  // parameters in.
  char returnMarshallMesgTypeName[MAX_STR_LEN];
  strcpy(returnMarshallMesgTypeName,"void");
  char returnMarshallMesgTypeNameWithPtr[MAX_STR_LEN];
  strcpy(returnMarshallMesgTypeNameWithPtr,"void");
  if (bop->isConstructor()) {
    strcpy(returnMarshallMesgTypeName,"");
    strcpy(returnMarshallMesgTypeNameWithPtr,"");
  } else if (bop->isReturnMessageNeeded()) { // either a retval, or a retarg
    sprintf(returnMarshallMesgTypeName, "M%sM%d", classname, 
 	    bop->getReturnMessageNumber());
    sprintf(returnMarshallMesgTypeNameWithPtr, "M%sM%d *", classname, 
 	    bop->getReturnMessageNumber());
  } else {
    // they're already void.
  }

  // Marshall, return messages
  // We always have a marshalling message since charmxi expects one
  spew(*MESGH, MESGOP20, marshallMesgTypeName);
  spew(*SKELCI, SKELCI20, marshallMesgTypeName);
  spew(*STUBCI, STUBCI20, marshallMesgTypeName);
  char mesghPass2[MAX_STR_LEN];
  strcpy(mesghPass2,"");
  if (bop->isReturnMessageNeeded())
    spew(mesghPass2, MESGOP20, returnMarshallMesgTypeName);

  // Output the method's return type and name.
  spew(*SKELH, SKELHDR30,
       returnMarshallMesgTypeNameWithPtr, skelmethodname, marshallMesgTypeName);
  if (bop->isConstructor())
    spew(*SKELC, SKELCEE35,
	 classname, skelmethodname, marshallMesgTypeName, 
	 returnMarshallMesgTypeNameWithPtr, methodname);
  else
    spew(*SKELC, SKELCEE30,
	 classname, skelmethodname, marshallMesgTypeName, 
	 returnMarshallMesgTypeNameWithPtr);
  char skelcPass2[MAX_STR_LEN];
  strcpy(skelcPass2,"");
  if (bop->isReturnMessageNeeded()) {
     // create retmsg instance
    spew(skelcPass2, SKELCEE52, returnMarshallMesgTypeName);
    // declare helper function
    spew(*STUBH, STUBHDR55, returnMarshallMesgTypeNameWithPtr,
	 marshallMesgTypeName, stubmethodname);
  }

  // declare user function
  spew(*STUBH, STUBHDR30, methodReturnTypeName, stubmethodname);
  spew(*STUBC, STUBCEE30, classname, methodReturnTypeName, stubmethodname);

  if (bop->isReturnMessageNeeded()) {
    // declare retmsg
    spew(*SKELCI, SKELCI20, returnMarshallMesgTypeName);
    spew(*STUBCI, STUBCI20, returnMarshallMesgTypeName);
    // declare EP
    spew(*SKELCIREST, SKELCI40, skelmethodname, marshallMesgTypeName,
	 returnMarshallMesgTypeNameWithPtr);
    spew(*STUBCIREST, STUBCI40, stubmethodname, marshallMesgTypeName,
	 returnMarshallMesgTypeNameWithPtr);
  } else {
    // do not declare the stub ep since no stub helper function is needed.
    spew(*SKELCIREST, SKELCI40, skelmethodname, marshallMesgTypeName, "");
  }

  // FIRST PASS

  // Output the stuff for the parameters.
  char stubcPass1B[MAX_STR_LEN];
  strcpy(stubcPass1B,"");
  char stubcPass2[MAX_STR_LEN];
  strcpy(stubcPass2,"");
  BE_produce_parameters(bop, stubcPass1B, mesghPass2, skelcPass2, stubcPass2);

  // Output the closing stuff for the method, i.e. parentheses, etc.
  spew(*SKELC, SKELCEE50);
  spew(*MESGH, MESGOP40, classname);
  spew(*STUBH, STUBHDR50);
  spew(*STUBC, STUBCEE50);
  spew(*STUBC, STUBCEE60,
       classname, stubmethodname, marshallMesgTypeName);
  spew(*STUBC, stubcPass1B);  // body of the method

  if(!bop->isReturnMessageNeeded()/* && !bop->is_threaded()*/) // invoke the remote call to the skeleton
    if (bop->isConstructor()) // constructor
      spew(*STUBC, STUBCEE88, classname, marshallMesgTypeName);
    else // async. message
      spew(*STUBC, STUBCEE80, classname, skelmethodname, marshallMesgTypeName);
  else // sync message
    if (bop->isConstructor()) // constructor
      spew(*STUBC, STUBCEE88, "ERROR", "ERROR"); // constructor cannot have retval @@ but it can be threaded, right ?
    else
      spew(*STUBC, STUBCEE82, returnMarshallMesgTypeName, stubmethodname);

  // SECOND PASS
  // rest of SKELC, STUBC, MESGH; i.e. return message stuff, if needed.
  if (bop->isReturnMessageNeeded()/* || bop->is_threaded()*/) {
    spew(mesghPass2, MESGOP40, classname);
    spew(skelcPass2, SKELCEE54);
  }
  spew(*MESGH, mesghPass2); 
  spew(*SKELC, skelcPass2);
  spew(*STUBC, stubcPass2);

  spew(*SKELC, SKELCEE55);
  spew(*STUBC, STUBCEE90);
  if (bop->isReturnMessageNeeded()/* || bop->is_threaded()*/) // output the helper function for the op
    spew(*STUBC, STUBCEE92, returnMarshallMesgTypeName, marshallMesgTypeName,
	 classname, stubmethodname);

//   i = new UTL_ScopeActiveIterator(op, UTL_Scope::IK_decls);
//   op->return_type()->name()->dump(o);
//   o << " ";
//   op->local_name()->dump(o);
//   o << "(";
//   while (!(i->is_done())) {
//     d = i->item();
//     d->dump(o);
//     i->next();
//     if (!(i->is_done()))
//       o << ", ";
//   }
//   delete i;
//   o << ")";

  // @@ We will deal with exceptions later
//   if (op->exceptions() != NULL) {
//     o << " raises(";
//     ei = new UTL_ExceptlistActiveIterator(op->exceptions());
//     while (!(ei->is_done())) {
//       e = ei->item();
//       ei->next();
//       e->local_name()->dump(o);
//       if (!(ei->is_done()))
// 	o << ", ";
//     }
//     delete ei;
//     o << ")";
//   }
//   if (op->context() != NULL) {
//     o << " context(";
//     si = new UTL_StrlistActiveIterator(op->context());
//     while (!(si->is_done())) {
//       s = si->item();
//       si->next();
//       o << s->get_string();
//       if (!(si->is_done()))
// 	o << ", ";
//     }
//     delete si;
//     o << ")";
//   }
}

//----------------------------------------------------------------------
void
BE_produce_attribute(AST_Decl *d)
{
  be_attribute *a = be_attribute::narrow_from_decl(d);
  cerr << "WARNING: Attributes unimplemented as yet." << endl;
  //exit (-1);

//   cout << "NT_attr " << a->field_type()->local_name()->get_string() << " "
//        << a->local_name()->get_string() << endl;

//   o << (pd_readonly == I_TRUE ? "readonly" : "") << " attribute ";
//   AST_Field::dump(o);
}

//----------------------------------------------------------------------
void
BE_produce_interface(AST_Decl *d)
  /* An interface is basically a C++ class */
{
  UTL_ScopeActiveIterator	*i;
  AST_Interface			*m;
  ostream &o(cout);

  m = AST_Interface::narrow_from_decl(d);

  char *classname = m->local_name()->get_string();
  spew(*SKELH, SKELHDR20, classname);
  spew(*SKELC, SKELCEE20, classname);
  spew(*STUBH, STUBHDR20, classname);
  spew(*STUBC, STUBCEE20, classname);
  spew(*SKELCIREST, SKELCI30, classname);
  spew(*STUBCIREST, STUBCI30, classname);

  //  if (UTL_Scope::decls_used() > 0) {
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

//       if (!d->imported()) {
// 	cout << "count = " << count << endl;
// 	idl_global->indent()->skip_to(o);
// 	d->dump(o);
// 	o << ";\n";
//       } else {
// 	cout << "count = " << count << " IMPORTED" << endl;
//       }

      i->next();
    }
    delete i;
    //  }
  spew(*SKELH, SKELHDR40);
  spew(*STUBH, STUBHDR60);
  spew(*SKELCIREST, SKELCI50);
  spew(*STUBCIREST, STUBCI50);

}

//----------------------------------------------------------------------
void
BE_produce_module(AST_Decl *d)
{
  UTL_ScopeActiveIterator	*i;
  AST_Module			*m;
  ostream &o(cout);

  m = AST_Module::narrow_from_decl(d);
  //  if (UTL_Scope::decls_used() > 0) {
    i = new UTL_ScopeActiveIterator(m, UTL_Scope::IK_both);
    int count = 0;
    while (!(i->is_done())) {
      d = i->item();
      count++;

      switch(d->node_type()){
      case AST_Decl::NT_interface:
	//cout << "NT_interface\n";
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
    //  }
}

//----------------------------------------------------------------------
/*
  Output occurs at several levels of the hierarchy:

  File Level
    Module Level (Basically a namespace)
      Interface Level (Basically a class)
        Operation Level (Basically a method)
	  Parameter Level (Basically a list of parameters for the method)

  In each case, we output when:
  1. Entering the level
  2. Within the level
  3. Leaving the level
 */

// Create the various files that the BE outputs.  Output the
// file-level opening stuff.
void
initialize()
{
  fullNameNoExt = get_fullname_noextension(
		      idl_global->main_filename()->get_string());
  // cout << "DEBUG:" << fullNameNoExt << endl;
  fileBaseNameNoExt = get_basename_noextension(
		      idl_global->main_filename()->get_string());
  // cout << "DEBUG:" << fileBaseNameNoExt << endl;

  char filename[MAX_STR_LEN];

  sprintf(filename, "%s_messages.h", fullNameNoExt);
  MESGH = new ofstream(filename);
  spew(*MESGH, MESGOP10, fullNameNoExt);

  sprintf(filename, "%s_skeleton.h", fullNameNoExt);
  SKELH = new ofstream(filename);
  spew(*SKELH, SKELHDR10, fileBaseNameNoExt);

  sprintf(filename, "%s_skeleton.C", fullNameNoExt);
  SKELC = new ofstream(filename);
  spew(*SKELC, SKELCEE10, fileBaseNameNoExt);

  sprintf(filename, "%s_skeleton.ci", fullNameNoExt);
  SKELCI = new ofstream(filename);
  spew(*SKELCI, SKELCI10, fileBaseNameNoExt);

  sprintf(filename, "%s_skeleton.ci.rest", fullNameNoExt);
  SKELCIREST = new ofstream(filename);

  sprintf(filename, "%s_stub.h", fullNameNoExt);
  STUBH = new ofstream(filename);
  spew(*STUBH, STUBHDR10, fileBaseNameNoExt);

  sprintf(filename, "%s_stub.C", fullNameNoExt);
  STUBC = new ofstream(filename);
  spew(*STUBC, STUBCEE10, fileBaseNameNoExt);

  sprintf(filename, "%s_stub.ci", fullNameNoExt);
  STUBCI = new ofstream(filename);

  sprintf(filename, "%s_stub.ci.rest", fullNameNoExt);
  STUBCIREST = new ofstream(filename);
}

// At the top level we can have modules, interfaces, constants or type
// declarations. @@
void
BE_produce_top_level()
{
  // PROCESS the root/file level of the AST
  UTL_ScopeActiveIterator	*i;
  AST_Decl			*d;
  ostream &o(cout);

  //  if (UTL_Scope::decls_used() > 0) {
    i = new UTL_ScopeActiveIterator(idl_global->root(), UTL_Scope::IK_both);

    //    o << GTDEVEL("\n/* Declarations0: */\n");
    int count = 0;
    while (!(i->is_done())) {
      d = i->item();
      count++;

      switch(d->node_type()){
      case AST_Decl::NT_module:
	//cout << "NT_module\n";
	BE_produce_module(d);
	break;
      case AST_Decl::NT_interface:
	//cout << "NT_interface\n";
	BE_produce_interface(d);
	break;
      case AST_Decl::NT_const:
      case AST_Decl::NT_struct:
	if (!d->imported()) {
	  d->dump(o);
	  o << ";" << endl << endl;
	}
	break;
      default:
	if (!d->imported()) {
	  cerr << "LIMIT: Only Modules and Interfaces at top level\n";
	  d->dump(cout); o << ";" << endl << endl;
	}
	break;
      }

//       if (!d->imported()) {
// 	cout << "count = " << count << endl;
// 	idl_global->indent()->skip_to(o);
// 	d->dump(o);
// 	o << ";\n";
//       } else {
// 	//	cout << "count = " << count << " IMPORTED" << endl;
//       }
      i->next();
    }
    delete i;
    //  }
}

// Output the file-level closing stuff; Close the output files.
void
clean_up()
{
  spew(*MESGH, MESGOP50);
  delete MESGH;

  spew(*SKELH, SKELHDR50);
  delete SKELH;

  char cleanUpCommands[MAX_STR_LEN];
  sprintf(cleanUpCommands, "cat %s_skeleton.h >> %s_messages.h; "
	  "mv %s_messages.h %s_skeleton.h", fullNameNoExt, fullNameNoExt,
	  fullNameNoExt, fullNameNoExt);
  assert(system(cleanUpCommands) == 0);

  spew(*SKELC, SKELCEE60, fileBaseNameNoExt);
  delete SKELC;

  delete SKELCI;
  delete SKELCIREST;
  sprintf(cleanUpCommands, "cat %s_skeleton.ci.rest >> %s_skeleton.ci; "
	  "rm %s_skeleton.ci.rest", fullNameNoExt, fullNameNoExt,
	  fullNameNoExt);
  assert(system(cleanUpCommands) == 0);

  spew(*STUBH, STUBHDR70);
  delete STUBH;

  spew(*STUBC, STUBCEE95, fileBaseNameNoExt);
  delete STUBC;

  delete STUBCI;
  delete STUBCIREST;
  sprintf(cleanUpCommands, "cat %s_stub.ci.rest >> %s_stub.ci; "
	  "rm %s_stub.ci.rest", fullNameNoExt, fullNameNoExt,
	  fullNameNoExt);
  assert(system(cleanUpCommands) == 0);

  delete fullNameNoExt;
  delete fileBaseNameNoExt;
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
