/**
 * Parallel solution transfer interface.
 * 
 * Orion Sky Lawlor, olawlor@acm.org, 3/24/2003
 */
#ifndef __UIUC_CHARM_PARALLELTRANSFER_H
#define __UIUC_CHARM_PARALLELTRANSFER_H

#include "collidec.h"
#include "mpi.h"

void parallelTransfer(collide_t voxels,MPI_Comm mpi_comm,
	int valsPerTet,
	const double *srcVals,const TetMesh &srcMesh,
	double *destVals,const TetMesh &destMesh);


#endif
