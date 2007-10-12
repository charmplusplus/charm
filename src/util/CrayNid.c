/** \file CrayNid.c
 *  Author: Abhinav S Bhatele
 *  Date created: October 10th, 2007  
 *  
 *  This file is needed because including the cnos_mpi_os.h in a C++ leads 
 *  to a compiler error. Hence we have defined a wrapper function here which
 *  can be called from C++ files
 */

#include "converse.h"

#if CMK_XT3

#include <catamount/cnos_mpi_os.h>
#define MAXNID 2784

int *pid2nid;
int nid2pid[MAXNID][2];  // assuming 2 ppn for cray xt3

/** \function pidtonid
 *  finds nids for pids 1 to CmiNumPes and stores them in an array
 *  correspondingly also creates an array for nids to pids
 */
void pidtonid(int numpes) {
  cnos_nidpid_map_t *nidpid; 
  int ierr, i, j, nid;
  
  nidpid = (cnos_nidpid_map_t *)malloc(sizeof(cnos_nidpid_map_t) * numpes);
  pid2nid = (int *)malloc(sizeof(int) * numpes);

  for(i=0; i<MAXNID; i++)
    for(j=0; j<2; j++)
      nid2pid[i][j] = -1;
      
  ierr = cnos_get_nidpid_map(&nidpid);
  for(i=0; i<numpes; i++) {
    nid = nidpid[i].nid;
    pid2nid[i] = nid;
    if(nid2pid[nid][0]==-1)
      nid2pid[nid][0] = i;
    else
      nid2pid[nid][1] = i;
  }
}

#endif // CMK_XT3
