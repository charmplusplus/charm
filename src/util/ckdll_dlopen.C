/*
 dlopen version of CkDll class.  
 This file can be #included whole by the configure script or ckdll.C.

 Orion Sky Lawlor, olawlor@acm.org, 9/10/2002
*/
#include "ckdll.h"
#include <unistd.h> //For unlink
#include <dlfcn.h> //for dlopen, etc.

CkDll::CkDll(const char *name) {
	handle=dlopen(name,RTLD_NOW);
}
void *CkDll::lookup(const char *name) {
	return dlsym(handle,name);
}
CkDll::~CkDll() {
	dlclose(handle);
}

const char *CkDll::extension=".so";
