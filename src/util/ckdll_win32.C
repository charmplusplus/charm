/*
  Win32/getProcAddress version of CkDll class.
  This file can be #included whole by the configure script or ckdll.C.

  WARNING: This file is untested-- it may not even compile!
  
  Orion Sky Lawlor, olawlor@acm.org, 9/10/2002
*/
#include "ckdll.h"
#include <windows.h>

CkDll::CkDll(const char *name) {
	handle=(void *)LoadLibrary(name);
}
void *CkDll::lookup(const char *name) {
	return GetProcAddress((Handle)handle,name);
}
CkDll::~CkDll() {
	UnloadLibrary((Handle)handle);
}

const char *CkDll::extension=".dll";
