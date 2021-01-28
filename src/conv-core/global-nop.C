/*
  Empty versions of Ctg routines, for use when 
 we don't want to swap globals (or can't figure 
 out how.

 Orion Sky Lawlor, olawlor@acm.org, 2003/9/22
*/
#include "converse.h"

CpvDeclare(int, CmiPICMethod);

void CtgInit(void) {
	CpvInitialize(int, CmiPICMethod);
        CpvAccess(CmiPICMethod) = CMI_PIC_NOP;
}
size_t CtgGetSize() { return 0; }
void CtgInstall(CtgGlobals g) {}
void CtgUninstall() {}
CtgGlobals CtgCreate(void * buffer) { return CtgGlobalStruct{}; }
CtgGlobals CtgCurrentGlobals() { return CtgGlobalStruct{}; }
