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

#if CMK_CRAYXT

#if XT3_TOPOLOGY
#include <catamount/cnos_mpi_os.h>
#define MAXNID 2784
#define TDIM 2

#else	/* if it is a XT4/5 */
#include <pmi.h>
#endif

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
  /* free(nidpid); */

#elif CMK_HAS_PMI_GET_NID	/* if it is a XT4/5 */
  PMI_Get_nid(mpirank, &nid);
#else
#error "Cannot get network topology information on a Cray build. Swap current module xt-mpt with xt-mpt/5.0.0 or higher and xt-asyncpe with xt-asyncpe/4.0 or higher and then rebuild"
#endif

  return nid;
}

#endif /* CMK_CRAYXT */

#if XT3_TOPOLOGY || XT4_TOPOLOGY || XT5_TOPOLOGY

#if !CMK_HAS_RCALIB
#error "The Cray rca library is not available. Try 'module load rca' and rebuild"
#endif

#include <rca_lib.h>

  #if XT4_TOPOLOGY
  #define MAXNID 14000
  #define TDIM 4

  #elif XT5_TOPOLOGY
  #define MAXNID 22020
  #define TDIM 12
  #endif

int *pid2nid;                   /* rank to node ID */
int nid2pid[MAXNID][TDIM];      /* node ID to rank */

/** \function getMeshCoord
 *  wrapper function for rca_get_meshcoord
 */
int getMeshCoord(int nid, int *x, int *y, int *z) {
  rca_mesh_coord_t xyz;
  rca_get_meshcoord(nid, &xyz);
  *x = xyz.mesh_x;
  *y = xyz.mesh_y;
  *z = xyz.mesh_z;
}

/** \function pidtonid
 *  finds nids for pids 1 to CmiNumPes and stores them in an array
 *  correspondingly also creates an array for nids to pids
 */
void pidtonid(int numpes) {
#if XT3_TOPOLOGY
  cnos_nidpid_map_t *nidpid; 
  int ierr, i, j, nid;
  
  nidpid = (cnos_nidpid_map_t *)malloc(sizeof(cnos_nidpid_map_t) * numpes);
  pid2nid = (int *)malloc(sizeof(int) * numpes);

  for(i=0; i<MAXNID; i++) {
    nid2pid[i][0] = -1;
    nid2pid[i][1] = -1;
  }
      
  ierr = cnos_get_nidpid_map(&nidpid);
  for(i=0; i<numpes; i++) {
    nid = nidpid[i].nid;
    pid2nid[i] = nid;
    
    /* if the first position on the node is not filled */
    /* put it there (0) else at (1) */
    if (nid2pid[nid][0] == -1)
      nid2pid[nid][0] = i;
    else
      nid2pid[nid][1] = i;
  }
  /* free(nidpid); */

  /* CORRECTION FOR MPICH_RANK_REORDER_METHOD */

  int k = -1;
  for(i=0; i<MAXNID; i++) {
    if(nid2pid[i][0] != -1) {
      nid2pid[i][0] = k++;
      pid2nid[k] = i;
      nid2pid[i][1] = k++;
      pid2nid[k] = i;
    }
  }
  
#elif XT4_TOPOLOGY || XT5_TOPOLOGY
  int i, l, nid;
  pid2nid = (int *)malloc(sizeof(int) * numpes);

  for(i=0; i<MAXNID; i++)
    for(l=0; l<TDIM; l++)
      nid2pid[i][l] = -1;

  for (i=0; i<numpes; i++) {
    PMI_Get_nid(i, &nid);
    pid2nid[i] = nid;

    l = 0;
    while(nid2pid[nid][l] != -1)
      l++;
    nid2pid[nid][l] = i;
  }
#endif
}

#endif /* XT3_TOPOLOGY || XT4_TOPOLOGY || XT5_TOPOLOGY */
