/**
 Fallback implementation of ftests.F90,
 used when no F90 compiler is available.
*/
#include "charm++.h" /* for CkPrintf */
#include "charm-api.h" /* for FTN_NAME */

FDECL void FTN_NAME(RUN_TEST,run_test)(void) {
	CkPrintf("   ftests_fallback.C: no fortran compiler\n");
}

