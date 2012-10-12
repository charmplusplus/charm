/** \file CrayNid.c
 *  Author: Abhinav S Bhatele
 *  Date created: October 10th, 2007  
 *  
 *  This file is needed because including the cnos_mpi_os.h in a C++ leads 
 *  to a compiler error. Hence we have defined a wrapper function here which
 *  can be called from C++ files
 */

#include <stdlib.h>
#include "converse.h"

#if CMK_CRAYXT || CMK_CRAYXE

#if XT3_TOPOLOGY
#include <catamount/cnos_mpi_os.h>

#else	/* if it is a XT4/5 or XE */
#include <pmi.h>
#endif

CmiNodeLock  cray_lock, cray_lock2;

/** \function getXTNodeID
 *  returns nodeID corresponding to the MPI rank (possibly obtained
 *  from CmiMyNode()/CmiNodeOf(pe)) passed to it
 */
int getXTNodeID(int mpirank, int nummpiranks) {
  int nid = -1;

#if XT3_TOPOLOGY
  cnos_nidpid_map_t *nidpid; 
  int ierr;
  
  nidpid = (cnos_nidpid_map_t *)malloc(sizeof(cnos_nidpid_map_t) * nummpiranks);

  ierr = cnos_get_nidpid_map(&nidpid);
  nid = nidpid[mpirank].nid;
  free(nidpid); 

#elif CMK_HAS_PMI_GET_NID	/* if it is a XT4/5 or XE */
  PMI_Get_nid(mpirank, &nid);
#else
#error "Cannot get network topology information on a Cray build. Swap current module xt-mpt with xt-mpt/5.0.0 or higher and xt-asyncpe with xt-asyncpe/4.0 or higher and then rebuild"
#endif

  return nid;
}

#endif /* CMK_CRAYXT || CMK_CRAYXE */

#if XT3_TOPOLOGY || XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY

#if !CMK_HAS_RCALIB
#error "The Cray rca library is not available. Try 'module load rca' and rebuild"
#endif

#include <rca_lib.h>

/*
	#if XT3_TOPOLOGY
	#define MAXNID 2784
	#define TDIM 2

  #elif XT4_TOPOLOGY
  #define MAXNID 14000
  #define TDIM 4

  #elif XT5_TOPOLOGY
  #define MAXNID 22020
  #define TDIM 12

  #elif XE6_TOPOLOGY
    // hopper 
  #define MAXNID 6384
  #define TDIM 24
#if 0
    // titan
  #define MAXNID 9600
#define TDIM 16
    // ESS 
  #define MAXNID 4608
  #define TDIM 32
    // JYC
  #define MAXNID 97
  #define TDIM 32
#endif
  #endif
*/

int *pid2nid = NULL;            /* rank to node ID */
int maxX = -1;
int maxY = -1;
int maxZ = -1;
int maxNID = -1;
#if CMK_HAS_RCALIB
rca_mesh_coord_t  *rca_coords = NULL;
#endif

void getDimension(int *maxnid, int *xdim, int *ydim, int *zdim);

/** \function getMeshCoord
 *  wrapper function for rca_get_meshcoord
 *  0: success,   -1: failure
 */
int getMeshCoord(int nid, int *x, int *y, int *z) {
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
  CmiAbort("rca_get_meshcoord not exist");
  return -1;
#endif
}

/** \function pidtonid
 *  finds nids for pids 1 to CmiNumPes and stores them in an array
 *  correspondingly also creates an array for nids to pids
 */
void pidtonid(int numpes) {
  CmiLock(cray_lock);
  if (pid2nid != NULL) {
      CmiUnlock(cray_lock);
      return;          /* did once already */
  }

  getDimension(&maxNID,&maxX,&maxY,&maxZ);
  int numCores = CmiNumCores();
  
  pid2nid = (int *)malloc(sizeof(int) * numpes);

#if XT3_TOPOLOGY
  cnos_nidpid_map_t *nidpid; 
  int ierr, i, nid;
	int *nid2pid;
	
  nid2pid = (int*)malloc(maxNID*2*sizeof(int));
  nidpid = (cnos_nidpid_map_t *)malloc(sizeof(cnos_nidpid_map_t) * numpes);

  for(i=0; i<maxNID; i++) {
    nid2pid[2*i+0] = -1;
    nid2pid[2*i+1] = -1;
  }
      
  ierr = cnos_get_nidpid_map(&nidpid);
  for(i=0; i<numpes; i++) {
    nid = nidpid[i].nid;
    pid2nid[i] = nid;
    
    /* if the first position on the node is not filled */
    /* put it there (0) else at (1) */
    if (nid2pid[2*nid+0] == -1)
      nid2pid[2*nid+0] = i;
    else
      nid2pid[2*nid+1] = i;
  }

  /* CORRECTION FOR MPICH_RANK_REORDER_METHOD */

  int k = -1;
  for(i=0; i<maxNID; i++) {
    if(nid2pid[2*i+0] != -1) {
      nid2pid[2*i+0] = k++;
      pid2nid[k] = i;
      nid2pid[2*i+1] = k++;
      pid2nid[k] = i;
    }
  }
	free(nidpid);
	free(nid2pid);

#elif XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
  int i, nid, ret;
  CmiAssert(rca_coords == NULL);
  rca_coords = (rca_mesh_coord_t *)malloc(sizeof(rca_mesh_coord_t)*(maxNID+1));
  for (i=0; i<maxNID; i++) {
    rca_coords[i].mesh_x = rca_coords[i].mesh_y = rca_coords[i].mesh_z = -1;
  }
  for (i=0; i<numpes; i++) {
    PMI_Get_nid(CmiNodeOf(i), &nid);
    pid2nid[i] = nid;
    CmiAssert(nid < maxNID);
    ret = rca_get_meshcoord(nid, &rca_coords[nid]);
    CmiAssert(ret != -1);
  }
#endif
  CmiUnlock(cray_lock);
}

/* get size and dimension for XE machine */
void getDimension(int *maxnid, int *xdim, int *ydim, int *zdim)
{
  int i = 0, ret;
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
  maxNID = *maxnid = *xdim * *ydim * *zdim * 2;

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

void craynid_init()
{
  if (CmiMyRank()==0) {
    cray_lock = CmiCreateLock();
    cray_lock2 = CmiCreateLock();

    pidtonid(CmiNumPes());
  }
}

#endif /* XT3_TOPOLOGY || XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY */
