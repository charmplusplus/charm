/**
"Standalone" version of all the Charm++ routines
needed by simple libraries.

Orion Sky Lawlor, olawlor@acm.org, 2003/8/15
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h> /*<- for va_start & friends */

#include "charm.h"
#include "charm-api.h"

/* CkPrintf/CkAbort support: */
extern "C" void CmiAbort(const char *why) {
	fprintf(stderr,"Fatal error> %s\n",why);
	abort();
}

extern "C" void CmiPrintf(const char *fmt, ...) {
        va_list p; va_start(p, fmt);
        vfprintf(stdout,fmt,p);
        va_end(p);
}
extern "C" void CmiError(const char *fmt, ...) {
        va_list p; va_start(p, fmt);
        vfprintf(stderr,fmt,p);
        va_end(p);
}

extern "C" void *CmiTmpAlloc(int size) {return malloc(size);}
extern "C" void CmiTmpFree(void *p) {free(p);}

/* Bizarre stuff for Cpv support: thankfully not needed.
#ifndef CmiMyPe
int CmiMyPe(void) {return 0;}
#endif
#ifndef CmiMyRank
int CmiMyRank(void) {return 0;}
#endif
int Cmi_mype=0;
int Cmi_numpes=1;
int Cmi_myrank=0;
int Cmi_mynodesize=1;
int Cmi_mynode=0;
int Cmi_numnodes=1;
 */

/* Mini-micro TCHARM */
#include "tcharmc.h"
CDECL void TCHARM_Migrate(void) {}

enum {tc_global_max=64};
void *tc_globals[tc_global_max];
void TCHARM_Set_global(int globalID,void *new_value,TCHARM_Pup_global_fn pup_or_NULL) {
	tc_globals[globalID]=new_value;
}
void *TCHARM_Get_global(int globalID) {
	return tc_globals[globalID];
}


/* FIXME: add real calls here: */
CDECL int TCHARM_Register(void *data,TCHARM_Pup_fn pfn) {return -1;}
CDECL void *TCHARM_Get_userdata(int id) {return 0;}

CDECL void TCHARM_Done(void) { /* fallthrough */ }
CDECL void TCHARM_Barrier(void) { /* fallthrough */ }

CDECL int TCHARM_Element(void) {return 0;}
CDECL int TCHARM_Num_elements(void) {return 1;}
CDECL double TCHARM_Wall_timer(void) {return 0.0;}





