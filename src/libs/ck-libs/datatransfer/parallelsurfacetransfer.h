/**
 * Parallel surface solution transfer interface.
 *
 * Make this collective call to transfer face- and node-centered data 
 * from src to dest, in parallel.
 * 
 * Terry L. Wilmarth, wilmarth@uiuc.edu, 4 Oct 2006
 */
#ifndef __UIUC_CHARM_PARALLELSURFACETRANSFER_H
#define __UIUC_CHARM_PARALLELSURFACETRANSFER_H

#include "collidec.h"
#include "mpi.h"

#define PARALLELTRANSFER_MPI_DTYPE MPI_DOUBLE
#define PARALLELTRANSFER_MPI_TAG 0xDA7A

/** Transfer this data from srcMesh to destMesh, where the srcMesh is
    a doubly-extruded surface mesh of prisms, and destMesh is a
    surface mesh of triangles.  There are valsPerFace double's
    associated with each prism of the mesh, stored in srcFace and
    destFace; there are valsPerPt double's associated with each point
    of the mesh, stored in srcPt and destPt;
*/
void ParallelSurfaceTransfer(collide_t voxels, MPI_Comm mpi_comm, int valsPerFace,
		      int valsPerPt, const double *srcFace, const double *srcPt,
		      const PrismMesh &srcMesh, double *destFace, double *destPt,
		      const TriangleSurfaceMesh &destMesh);

#endif
