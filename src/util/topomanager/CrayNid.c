/** \file CrayNid.c
 *  Author: Abhinav S Bhatele
 *  Date created: October 10th, 2007  
 *  
 *  This file is needed because including the cnos_mpi_os.h in a C++ leads 
 *  to a compiler error. Hence we have defined a wrapper function here which
 *  can be called from C++ files
 */

#include "topomanager_config.h"
#include <stdlib.h>
#ifndef __TPM_STANDALONE__
#include "converse.h"
#else
#include "tpm_standalone.h"
#endif

#ifndef CLINKAGE
# ifdef __cplusplus
#  define CLINKAGE extern "C"
# else
#  define CLINKAGE
# endif
#endif

#if CMK_CRAYXE || CMK_CRAYXC || CMK_CRAYEX

#if XT3_TOPOLOGY
#else	/* if it is a XT4/5 or XE */
#include <pmi.h>
#endif

CmiNodeLock  cray_lock, cray_lock2;

/** \function getXTNodeID
 *  returns nodeID corresponding to the MPI rank (possibly obtained
 *  from CmiMyNode()/CmiNodeOf(pe)) passed to it
 */
CLINKAGE int getXTNodeID(int mpirank, int nummpiranks)
{
  int nid = -1;

#if CMK_HAS_PMI_GET_NID	/* if it is a XT4/5 or XE */
  PMI_Get_nid(mpirank, &nid);
#else
#if CMK_CRAYEX
#error "Cannot get network topology information on a Cray build. Load the cray-pmi module and then rebuild"
#else
#error "Cannot get network topology information on a Cray build. Swap current module xt-mpt with xt-mpt/5.0.0 or higher and xt-asyncpe with xt-asyncpe/4.0 or higher and then rebuild"
#endif
#endif

  return nid;
}

#endif /* CMK_CRAYXE || CMK_CRAYXC || CMK_CRAYEX */

#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY

#if !CMK_HAS_RCALIB
#error "The Cray rca library is not available. Try 'module load rca' and rebuild"
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include <rca_lib.h>
#ifdef __cplusplus
}
#endif

int *pid2nid = NULL;            /* rank to node ID */
int maxX = -1;
int maxY = -1;
int maxZ = -1;
int maxNID = -1;
#if CMK_HAS_RCALIB
rca_mesh_coord_t  *rca_coords = NULL;
#endif

CLINKAGE void getDimension(int *maxnid, int *xdim, int *ydim, int *zdim);

/** \function getMeshCoord
 *  wrapper function for rca_get_meshcoord
 *  0: success,   -1: failure
 */
CLINKAGE int getMeshCoord(int nid, int *x, int *y, int *z)
{
#if CMK_HAS_RCALIB
  if (rca_coords == NULL) {
  rca_mesh_coord_t xyz;
  int ret = -1;
  ret = rca_get_meshcoord(nid, &xyz);
  if (ret == -1) return -1;
  *x = xyz.mesh_x;
  *y = xyz.mesh_y;
  *z = xyz.mesh_z;
  return ret;
  }
  else {
  *x = rca_coords[nid].mesh_x;
  *y = rca_coords[nid].mesh_y;
  *z = rca_coords[nid].mesh_z;
  return *x==-1?-1:0;
  }
#else
  CmiAbort("rca_get_meshcoord does not exist");
  return -1;
#endif
}

/** \function pidtonid
 *  finds nids for pids 1 to CmiNumPes and stores them in an array
 *  correspondingly also creates an array for nids to pids
 */
CLINKAGE void pidtonid(int numpes)
{
  CmiLock(cray_lock);
  if (pid2nid != NULL) {
      CmiUnlock(cray_lock);
      return;          /* did once already */
  }

  getDimension(&maxNID,&maxX,&maxY,&maxZ);
  
  pid2nid = (int *)malloc(sizeof(int) * numpes);

#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
  int i, nid, ret;
  CmiAssert(rca_coords == NULL);
  rca_coords = (rca_mesh_coord_t *)malloc(sizeof(rca_mesh_coord_t)*(maxNID+1));
  for (i=0; i<maxNID; i++) {
    rca_coords[i].mesh_x = rca_coords[i].mesh_y = rca_coords[i].mesh_z = -1;
  }
  for (i=0; i<numpes; i++) {
    PMI_Get_nid(CmiGetNodeGlobal(CmiNodeOf(i),CmiMyPartition()), &nid);
    pid2nid[i] = nid;
    CmiAssert(nid < maxNID);
    ret = rca_get_meshcoord(nid, &rca_coords[nid]);
    CmiAssert(ret != -1);
  }
#endif
  CmiUnlock(cray_lock);
}

/* get size and dimension for XE machine */
CLINKAGE void getDimension(int *maxnid, int *xdim, int *ydim, int *zdim)
{
  int i = 0, nid, ret;
  rca_mesh_coord_t dimsize;

  CmiLock(cray_lock2);

  if(maxNID != -1) {
	*xdim = maxX;
	*ydim = maxY;
	*zdim = maxZ;
	*maxnid = maxNID;
        CmiUnlock(cray_lock2);
	return;
  }

#if CMK_HAS_RCA_MAX_DIMENSION
  // rca_get_meshtopology(&mnid);
  rca_get_max_dimension(&dimsize);
  maxX = *xdim = dimsize.mesh_x+1;
  maxY = *ydim = dimsize.mesh_y+1;
  maxZ = *zdim = dimsize.mesh_z+1;
  maxNID = 0;

  for(i = 0; i < CmiNumNodesGlobal(); i++) {
    PMI_Get_nid(i, &nid);
    if(nid >= maxNID) maxNID = nid + 1;
  }
  *maxnid = maxNID;

#else

  *xdim = *ydim = *zdim = 0;
    /* loop until fails to find the max */ 
  do {
      int x, y, z;
      ret = getMeshCoord(i, &x, &y, &z);
      if (ret == -1) {
#if CMK_CRAY_MAXNID
          if (i<=CMK_CRAY_MAXNID) {
              i++;
              ret = 0;
              continue;
          }
#endif
          break;
      }
      if (x>*xdim) *xdim = x;
      if (y>*ydim) *ydim = y;
      if (z>*zdim) *zdim = z;
      i++;
  } while (ret == 0);
  maxNID = *maxnid = i;
  maxX = *xdim = *xdim+1;
  maxY = *ydim = *ydim+1;
  maxZ = *zdim = *zdim+1;
#endif

  CmiUnlock(cray_lock2);

  /* printf("%d %d %d %d\n", *maxnid, *xdim, *ydim, *zdim); */
}

CLINKAGE void craynid_free(void)
{
  CmiLock(cray_lock);
  free(pid2nid);
  pid2nid = NULL;
#if CMK_HAS_RCALIB
  free(rca_coords);
  rca_coords = NULL;
#endif
  CmiUnlock(cray_lock);
}

CLINKAGE void craynid_reset(void)
{
  craynid_free();
  CmiLock(cray_lock);
  maxX = -1;
  maxY = -1;
  maxZ = -1;
  maxNID = -1;
  CmiUnlock(cray_lock);
}

CLINKAGE void craynid_init(void)
{
  static int init_done = 0;
  if (!init_done) {
    cray_lock = CmiCreateLock();
    cray_lock2 = CmiCreateLock();
    init_done = 1;
  }
}

#endif /* XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY */
