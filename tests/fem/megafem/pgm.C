/*
Tester for FEM framework C/F90 routines. 
*/
#include "mpi.h"
#include "fem.h"
#include "tcharm.h"
#include "ctests.h"

CDECL void RUN_Abort(int v) {
	CkError("FEM Test failed: %d\n",v);
	CkAbort("FEM Test failed.\n");
}
FDECL void FTN_NAME(RUN_ABORT,run_abort)(int *v) {
	RUN_Abort(*v);
}

int main(int argc,char **argv) {
	MPI_Init(&argc,&argv);
	FEM_Init(MPI_COMM_WORLD);
	
	CkPrintf("---- Running C++ FEM tests... -----\n");
	RUN_Test();
	
	CkPrintf("---- Running F90 FEM tests... -----\n");
	FTN_NAME(RUN_TEST,run_test)();
	
	return 0;
}

