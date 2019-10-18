/**
"Standalone" version of all the Charm++ routines
needed by simple libraries.  Currently only used 
for FEM_ALONE mode.

Orion Sky Lawlor, olawlor@acm.org, 2003/8/15
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h> /*<- for va_start & friends */
#include <string.h> /*<- for strlen */

#include "charm.h"
#include "charm-api.h"

/* CkPrintf/CkAbort support: */
extern "C" void CmiAbort(const char *fmt, ...) {
	fprintf(stderr, "Fatal error> ");
	va_list p; va_start(p, fmt);
	vfprintf(stderr, fmt, p);
	va_end(p);
	fprintf(stderr, "\n");
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
extern "C" void __cmi_assert(const char *errmsg) { CmiAbort(errmsg);}

/* Bizarre stuff for Cpv support: thankfully not needed by FEM.
#ifndef CmiMyPe
int CmiMyPe(void) {return 0;}
#endif
#ifndef CmiMyRank
int CmiMyRank(void) {return 0;}
#endif
 */
int _Cmi_mype=0;
int _Cmi_numpes=1;
int _Cmi_mynodesize=1;
int _Cmi_mynode=0;
int _Cmi_numnodes=1;

/* CmiWallTimer support */
double _cpu_speed_factor=0.0;
CLINKAGE double CmiTimer(void) {return 0.0;}
#ifndef CmiWallTimer
CLINKAGE double CmiWallTimer(void) {return 0.0;}
#endif

/* Mini-micro TCHARM */
#include "tcharmc.h"
CLINKAGE void TCHARM_Migrate(void) {}

enum {tc_global_max=64};
void *tc_globals[tc_global_max];
void TCHARM_Set_global(int globalID,void *new_value,TCHARM_Pup_global_fn pup_or_NULL) {
	tc_globals[globalID]=new_value;
}
void *TCHARM_Get_global(int globalID) {
	return tc_globals[globalID];
}


/* FIXME: add real calls based on MPI here: */
CLINKAGE int TCHARM_Register(void *data,TCHARM_Pup_fn pfn) {return -1;}
FORTRAN_AS_C_RETURN(int,TCHARM_REGISTER,TCHARM_Register,tcharm_register,
	(void *data,TCHARM_Pup_fn pfn),(data,pfn))

CLINKAGE void *TCHARM_Get_userdata(int id) {return 0;}

CLINKAGE void TCHARM_Done(int exitcode) { /* fallthrough */ }
FORTRAN_AS_C(TCHARM_DONE,TCHARM_Done,tcharm_done,(int *exitcode),(*exitcode))

CLINKAGE void TCHARM_Barrier(void) { /* fallthrough */ }
FORTRAN_AS_C(TCHARM_BARRIER,TCHARM_Barrier,tcharm_barrier,(void),())

CLINKAGE int TCHARM_Element(void) {return 0;}
FORTRAN_AS_C_RETURN(int,TCHARM_ELEMENT,TCHARM_Element,tcharm_element,(void),())

CLINKAGE int TCHARM_Num_elements(void) {return 1;}
FORTRAN_AS_C_RETURN(int,TCHARM_NUM_ELEMENTS,TCHARM_Num_elements,tcharm_num_elements,(void),())

CLINKAGE double TCHARM_Wall_timer(void) {return 0.0;}
FORTRAN_AS_C_RETURN(double,TCHARM_WALL_TIMER,TCHARM_Wall_timer,tcharm_wall_timer,(void),())

#define FORTRAN_TCHARM_UNIMPLEMENTED(ROUTINE,routine) \
FLINKAGE void FTN_NAME(TCHARM_##ROUTINE,tcharm_##routine)(void) \
	{ CkAbort("Sorry, standalone mode does not have TCHARM_" #ROUTINE); }

FORTRAN_TCHARM_UNIMPLEMENTED(INIT,init)
FORTRAN_TCHARM_UNIMPLEMENTED(CREATE,create)
FORTRAN_TCHARM_UNIMPLEMENTED(MIGRATE,migrate)
FORTRAN_TCHARM_UNIMPLEMENTED(SET_STACK_SIZE,set_stack_size)
FORTRAN_TCHARM_UNIMPLEMENTED(CREATE_DATA,create_data)
FORTRAN_TCHARM_UNIMPLEMENTED(MIGRATE_TO,migrate_to)
FORTRAN_TCHARM_UNIMPLEMENTED(YIELD,yeild)


/* Command-line argument handling */
static char **saved_argv=NULL;

CLINKAGE int CmiGetArgFlagDesc(char **argv,const char *arg,const char *desc) {
	int i;
	saved_argv=argv;
	for (i=0;argv[i]!=NULL;i++)
		if (0==strcmp(argv[i],arg))
		{/*We found the argument*/
			return 1;
		}
	return 0;/*Didn't find the argument*/
}

CLINKAGE int CmiGetArgIntDesc(char **argv,const char *arg,int *optDest,const char *desc) {
	int i;
	saved_argv=argv;
	for (i=0;argv[i]!=NULL;i++)
		if (0==strcmp(argv[i],arg))
		{/*We found the argument*/
			*optDest = atoi(argv[i+1]);
			return 1;
		}
	return 0;/*Didn't find the argument*/
}

FLINKAGE void FTN_NAME(TCHARM_GETARG,tcharm_getarg)(int *arg,char *dest,int destLen) {
	if (saved_argv==NULL)
		CkAbort("TCHARM_GETARG not supported in FEM_ALONE mode!\n");
	else /* actually have a saved argument-- return it */ {
		int i;
		const char *src=saved_argv[*arg];
		strcpy(dest,src);
		for (i=strlen(dest);i<destLen;i++) dest[i]=' ';
	}
}

