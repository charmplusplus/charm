/**
 * Parallel solution transfer interface.
 *
 * Make this collective call to transfer the 
 * volumetric (tet) and node-centered (pt) data 
 * from src to dest, in parallel.
 * 
 * Orion Sky Lawlor, olawlor@acm.org, 3/24/2003
 */
#ifndef __UIUC_CHARM_PARALLELTRANSFER_H
#define __UIUC_CHARM_PARALLELTRANSFER_H

#include "collidec.h"
#include "mpi.h"

/** Transfer datatype */
typedef double xfer_t;
#define PARALLELTRANSFER_MPI_DTYPE MPI_DOUBLE
#define PARALLELTRANSFER_MPI_TAG 0xDA7A

/** Transfer this data from srcMesh to destMesh.
There are valsPerTet xfer_t's associated with each
tet of the mesh, stored in srcTet and destTet;
there are valsPerPt xfer_t's associated with each
point of the mesh, stored in srcPt and destPt; 
*/
void ParallelTransfer(collide_t voxels, MPI_Comm mpi_comm, int valsPerTet,
		      int valsPerPt, const xfer_t *srcTet, const xfer_t *srcPt,
		      const TetMesh &srcMesh, xfer_t *destTet, xfer_t *destPt,
		      const TetMesh &destMesh);

#endif
