/*
Portable Dynamically Linked Libraries (DLL) interface

Orion Sky Lawlor, olawlor@acm.org, 7/26/2002
*/
#include "converse.h" //For CMK_ symbols
#include "ckdll.h"
#include <stdio.h> //For fopen
#include <stdlib.h> //For system
#include <string.h>


/*#include the appropriate CkDll implementation: */

#if CMK_DLL_USE_DLOPEN  /*********** UNIX .so/dlopen Version ******/
#include "ckdll_dlopen.C"

static void deleteFile(const char *fileName) {
	unlink(fileName);
}
#define CMK_SCRATCH_PATH "/tmp"

#elif CMK_DLL_USE_WIN32 /*********** Win32 .dll/GetProcAddress Version ******/
#include "ckdll_win32.C"

static void deleteFile(const char *fileName) {
	DeleteFile(fileName);
}
#define CMK_SCRATCH_PATH ""

#else 
/********* It ain't UNIX, it ain't win32-- what *is* it? */
CkDll::CkDll(const char *name) {
	handle=0; /*DLL's not supported here.*/
}
void *CkDll::lookup(const char *name) {
	return 0;
}
CkDll::~CkDll() {
	;
}

const char *CkDll::extension=0;
static void deleteFile(const char *fileName) { }

#define CMK_SCRATCH_PATH ""
#endif

/****************************************************************
CkCppInterpreter interface:
	Call the C++ compiler on a string, then use CkDll to link the
resulting dll into the running program.

	This depends on conv-mach.h or conv-mach-opt.h setting the symbol
CMK_DLL_CC to the correct invokation of the C++ compiler to generate a shared
library.  CMK_DLL_CC will immediately be followed by the output library name,
so it should end with, e.g., "-o " on UNIX platforms.

	CMK_DLL_LINK is an optional extra link step (required on HP machines);
and CMK_DLL_INC is the compiler flag to change the #include path.
*/

/* 
Command-line compilers for various platforms (now in conv-mach.h files)
#if CMK_DLL_VIA_SUN_CC
#  define CMK_DLL_CC  "CC -G -O3 -o "
#elif CMK_DLL_VIA_SGI_CC
#  define CMK_DLL_CC  "CC -shared -64 -LANG:std -O3 -o "
#elif CMK_DLL_VIA_CXX
#  define CMK_DLL_CC  "cxx -shared -O3 -o "
#elif CMK_DLL_VIA_HP_CC
#  define CMK_DLL_CC  "CC +z -O -c -o "
#  define CMK_DLL_LINK "CC -b -o "
#else
//Default: try g++
#  define CMK_DLL_CC  "g++ -shared -O3 -o "
#endif
*/


#ifdef CMK_DLL_CC //We have a command-line dynamic-link library compiler:

#ifndef CMK_DLL_INC
#  define CMK_DLL_INC "-I" /*Assume unix-style command-line flags*/
#endif

/*Return 1 if this file exists*/
static int fileExists(const char *fileName) {
	FILE *f=fopen(fileName,"r");
	if (f==NULL) return 0;
	else {
		fclose(f);
		return 1;
	}
}


#ifdef CMK_SIGSAFE_SYSTEM
#  include "ckdll_system.C"
#else
/*No need for a signal-safe system call*/
static int CkSystem (const char *command) {
	system(command);
}
#endif

//Compile "cppCode", making available the includes at inclPath
CkCppInterpreter::CkCppInterpreter(const char *cppCode,const char *inclPath)
	:library(NULL)
{
	int verbose=0;
	int randA=CrnRand();
	int randB=CmiMyPe();

/*Write the c++ code to a temporary file:*/
	char sourceFile[256];
	sprintf(sourceFile,"%s/ckSharedLib_%d_%d_%p.%s",
		CMK_SCRATCH_PATH,randA,randB,this,"cpp");
	FILE *f=fopen(sourceFile,"w"); if (f==NULL) return;
	fputs(cppCode,f);
	fclose(f);

/*Allocate a spot for the library file:*/
	sprintf(libraryFile,"%s/ckSharedLib_%d_%d_%p%s",
		CMK_SCRATCH_PATH,randA,randB,this,CkDll::extension);
	
//Compile the .cpp file into a .dll:
	char compilerCmd[1024];
	sprintf(compilerCmd,"%s%s %s %s%s",
		CMK_DLL_CC, libraryFile, sourceFile,
		inclPath!=NULL?CMK_DLL_INC:"",inclPath!=NULL?inclPath:"");
	
	if (verbose) CmiPrintf("Executing: '%s'\n",compilerCmd);
	int compilerRet=CkSystem(compilerCmd);
	deleteFile(sourceFile);
	if (compilerRet!=0) { //!fileExists(libraryFile)) {
		CmiPrintf("Compilation error! Cmd='%s', err=%d, src='%s'\n",
			compilerCmd,compilerRet,cppCode);
		return; /*with library set to NULL*/
	}
	
#ifdef CMK_DLL_LINK
//Link the .so into a ".sop"
	// HIDEOUS HACK: playing silly games with filename:
	//    CC source -o foo.so
	//    CC foo.so -o foo.sop
	sprintf(compilerCmd,"%s%sp %s",
		CMK_DLL_LINK, libraryFile, libraryFile);
	compilerRet=CkSystem(compilerCmd);
	unlink(libraryFile);
	strcat(libraryFile,"p");
	if (compilerRet!=0) { //!fileExists(libraryFile)) {
		CmiPrintf("Link error! Cmd='%s', err=%d, src='%s'\n",
			compilerCmd,compilerRet,cppCode);
		return; 
	}
#endif
	
/*Link the library into the program: */	
	library=new CkDll(libraryFile);
}

//Remove "cppCode" from the program.
//  This invalidates any function pointers created with lookup
CkCppInterpreter::~CkCppInterpreter()
{
	if (library) {
		delete library;
		deleteFile(libraryFile);
	}
}

#endif
