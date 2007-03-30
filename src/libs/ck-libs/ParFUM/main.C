/**
 * \addtogroup ParFUM
*/
/*@{*/

/*
This is a compatability main routine for FEM framework programs.
It is an MPI main program that calls the old-style "init" routine
on processor 0, then runs driver on all processors.

This routine need not get called-- an MPI program may choose
to implement a main routine and call FEM_Init itself, in which
case there's no "init" or "driver" routines.

This routine is only linked in if you link with
	-language ParFUM
if you use
	-language ampi -module ParFUM
this routine is not linked.

Orion Sky Lawlor, olawlor@acm.org, 2003/7/13
*/

#include "ParFUM.h"
#include "ParFUM_internals.h"

CDECL void fem_impl_call_init(void);

FDECL void FTN_NAME(INIT,init)(void);
FDECL void FTN_NAME(DRIVER,driver)(void);

int main(int argc,char **argv) {
	MPI_Comm comm=MPI_COMM_WORLD;
	int myPart, nPart;
	int parallelMesh=0, serialMesh=0;
	MPI_Init(&argc,&argv);
	FEM_Init((FEM_Comm_t)comm);
	MPI_Comm_rank(comm,&myPart); MPI_Comm_size(comm,&nPart);
	if (CmiGetArgFlagDesc(argv,"-read","Skip init()--read mesh from files"))
	{ /* "-read" was passed: skip init & read parallel mesh from files */
		parallelMesh=FEM_Mesh_read("fem_mesh",myPart,nPart);
	}
	else
	{ /* Call init normally: */
		if (myPart==0)
		{ // Make a little mesh, and call user's "init" routine to fill it out:
			serialMesh=FEM_Mesh_allocate();
			FEM_Mesh_set_default_write(serialMesh);
#ifndef FEM_ALONE
			fem_impl_call_init();
#ifndef CMK_FORTRAN_USES_NOSCORE
			FTN_NAME(INIT,init)();
#endif
#else /* FEM_ALONE version: just call F90 init routine */
			FTN_NAME(INIT,init)();
#endif
			serialMesh=FEM_Mesh_default_write();
		}
		parallelMesh=FEM_Mesh_broadcast(serialMesh,0,(FEM_Comm_t)comm);
		if (myPart==0) FEM_Mesh_deallocate(serialMesh);
	}

	if (CmiGetArgFlagDesc(argv,"-write","Skip driver()--write mesh to files"))
	{/* "-write" was passed: skip driver & write files */
		FEM_Mesh_write(parallelMesh,"fem_mesh",myPart,nPart);
		FEM_Mesh_deallocate(parallelMesh);
	}
	else
	{ /* Call user's driver routine for the main computation */
		FEM_Mesh_set_default_read(parallelMesh);
		FEM_Mesh_set_default_write(FEM_Mesh_allocate());
#ifndef FEM_ALONE
	        driver();
#ifndef CMK_FORTRAN_USES_NOSCORE
	        FTN_NAME(DRIVER,driver)();
#endif
#else /* FEM_ALONE version: just call F90 init routine */
	        FTN_NAME(DRIVER,driver)();
#endif
		FEM_Mesh_deallocate(FEM_Mesh_default_write());
		FEM_Mesh_deallocate(FEM_Mesh_default_read());
	}
	MPI_Finalize();
	return 0;
}
/*@}*/
