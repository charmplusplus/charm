/*
  Empty versions of Ctg routines, for use when 
 we don't want to swap globals (or can't figure 
 out how.

 Orion Sky Lawlor, olawlor@acm.org, 2003/9/22
*/
#include "converse.h"

CpvDeclare(int, CmiHasPIC);

void CtgInit(void) {
	CpvInitialize(int, CmiHasPIC);
        CpvAccess(CmiHasPIC) = 0;
}
void CtgInstall(CtgGlobals g) {}
CtgGlobals CtgCreate(void) {return 0;}
CtgGlobals CtgPup(pup_er p,CtgGlobals g) { return 0;}
void CtgFree(CtgGlobals g) {}
CtgGlobals CtgCurrentGlobals(void) { return 0; }
