#include "fem.h"
#include <assert.h>

/*
 * subroutine fem_map takes in the original mesh, as well as
 * partitioning produced by metis, and creates messages to be sent to
 * individual partitions. These messages contain information related to
 * that partition only. The connectivity matrix refers to the local
 * node numbers, and is indexed by local element numbers. Similarly, the
 * communication information, which contains the nodes to be communicated
 * to different partitions in fem_update, also refer to local node
 * numbers.
 *
 * nelems: total number of elements
 * nnodes: total number of nodes
 * ctype: type of element
 * connmat: total connectivity matrix, in row major format [nelems][esize]
 * nparts: required number of partitions
 * epart: element partitioning by metis, epart[i] contains partition for elem i
 * npart: node partitioning by metis, npart[i] contains partition for node i
 * msgs: messages produced by this routine
 */

void
fem_map(int nelems, int nnodes, int ctype, int *connmat,
        int nparts, int *epart, int *npart, ChunkMsg *msgs[])
{
  int i,j,k;
  int *pinfo = new int[nparts]; CHK(pinfo);
  int *ninfo = new int[nnodes]; CHK(ninfo);
  int esize = (ctype==FEM_TRIANGULAR) ? 3 :
              ((ctype==FEM_HEXAHEDRAL) ? 8 :
              4);
  for(i=0;i<nparts;i++) {
    msgs[i] = new ChunkMsg; CHK(msgs[i]);
    msgs[i]->nelems = 0;
    msgs[i]->nconn = esize;
  }
#if FEM_FORTRAN
  // Make everything 0-based
  for(i=0;i<nelems;i++) { epart[i]--; }
  for(i=0;i<nnodes;i++) { npart[i]--; }
  for(i=0;i<(nelems*esize);i++) { connmat[i]--; }
#endif 
  // count the number of elements assigned to each partition
  for(i=0;i<nelems;i++) { msgs[epart[i]]->nelems++; }
  for(i=0;i<nparts;i++) { 
    msgs[i]->gElemNums = new int[msgs[i]->nelems]; CHK(msgs[i]->gElemNums);
    msgs[i]->conn = new int[msgs[i]->nelems*esize]; CHK(msgs[i]->conn);
  }
  // pinfo[i] contains filled global element numbers for partition i
  for(i=0;i<nparts;i++) { pinfo[i] = 0; }
  // fill in the global element numbers
  for(i=0;i<nelems;i++) { 
    int ep = epart[i];
    msgs[ep]->gElemNums[pinfo[ep]++] = i;
  }
  for(i=0;i<nparts;i++) {
    ChunkMsg *m = msgs[i];
    m->nnodes = 0;
    // ninfo[j] is true if node j appears in partition i
    for(j=0;j<nnodes;j++) { ninfo[j] = 0; }
    // scan the connectivity matrix to see which nodes appear in a partition
    // a node appears in a partition if at least one element connected to it
    // belongs to that partition
    for(j=0;j<nelems;j++) {
      // check if element belongs to this partition
      if(epart[j]==i) {
        // make all its bordering nodes belong to this partition
        // if not already here
        for(k=0;k<esize;k++) {
          if(ninfo[connmat[j*esize+k]]==0) {
            ninfo[connmat[j*esize+k]] = 1;
            m->nnodes++;
          }
        }
      }
    }
    m->gNodeNums = new int[m->nnodes]; CHK(m->gNodeNums);
    m->primaryPart = new int[m->nnodes]; CHK(m->primaryPart);
    m->nnodes = 0;
    for(j=0;j<nnodes;j++) {
      if(ninfo[j]==1) {
        m->gNodeNums[m->nnodes] = j;
        // ninfo now contains the local index for a node
        ninfo[j] = m->nnodes++;
      } else {
        // 0 is a valid local index, so set to (-1) to denote absence
        ninfo[j] = (-1);
      }
    }
    // now fill in the connectivity matrix with local node numbers
    for(j=0;j<m->nelems;j++) {
      int telem = m->gElemNums[j];
      for(k=0;k<esize;k++) {
        int tnode = connmat[telem*esize+k];
        assert(ninfo[tnode]>=0 && ninfo[tnode]<m->nnodes);
        m->conn[j*esize+k] = ninfo[tnode];
      }
    }
  }
  // now find the communication information
  int *comm = new int[nnodes*nparts]; CHK(comm);
  // comm[i][j] contains the local index of node i in partition j
  // if i belongs to j; otherwise it contains -1
  for(i=0;i<(nnodes*nparts);i++) { comm[i] = (-1); }
  // fill comm
  for(j=0;j<nparts;j++) {
    ChunkMsg *m = msgs[j];
    for(i=0;i<(m->nnodes);i++) {
      // tnode is the i'th node of partition j
      int tnode = m->gNodeNums[i];
      comm[tnode*nparts+j] = i;
    }
  }
  // now find the primary partition
  // ninfo[i] contains the number of partitions that node i belongs to
  for(i=0;i<nnodes;i++) { ninfo[i] = 0; }
  for(i=0;i<nnodes;i++) {
    for(j=0;j<nparts;j++) {
      // if node i belongs to partition j, increment ninfo[i]
      if(comm[i*nparts+j] != (-1)) { ninfo[i]++; }
    }
  }
  // now generate a random partition number among those 
  // that the node belongs to. At the end of this loop, ninfo[i] contains
  // the primary partition number for node i
  for(i=0;i<nnodes;i++) {
    assert(ninfo[i]>=1);
    int ridx = CrnRand()%ninfo[i];
    k = (-1);
    for(j=0;j<nparts;j++) {
      if(comm[i*nparts+j] != (-1)) { 
        k++; 
        if(k==ridx) { break; }
      }
    }
    assert(j!=nparts);
    ninfo[i] = j;
  }
  // now assign the primary partition number to nodes in each of the messages
  for(i=0;i<nparts;i++) {
    ChunkMsg *m = msgs[i];
    for(j=0;j<(m->nnodes);j++) {
      m->primaryPart[j] = ninfo[m->gNodeNums[j]];
    }
  }
  for(i=0;i<nparts;i++) {
    ChunkMsg *m = msgs[i];
    m->npes = 0;
    // pinfo[j] contains number of nodes that partition i needs to 
    // communicate with partition j
    for(j=0;j<nparts;j++) { pinfo[j] = 0; }
    for(j=0;j<nparts;j++) {
      if(i==j) continue;
      for(k=0;k<(m->nnodes);k++) {
        int tnode = m->gNodeNums[k];
        // if tnode also belongs to partition j, i has to communicate with j
        if(comm[tnode*nparts+j] != (-1)) {
          if(pinfo[j]==0) { m->npes++; }
          pinfo[j]++;
        }
      }
    }
    m->numNodesPerPe = new int[m->npes]; CHK(m->numNodesPerPe);
    m->peNums = new int[m->npes]; CHK(m->peNums);
    for(j=0,k=0; j<nparts; j++) {
      if(pinfo[j]>0) {
        m->numNodesPerPe[k] = pinfo[j];
        m->peNums[k] = j;
        k++;
      }
    }
    int tcomm = 0;
    for(j=0; j<(m->npes); j++) { tcomm += m->numNodesPerPe[j]; }
    m->nodesPerPe = new int[tcomm]; CHK(m->nodesPerPe);
    int icomm = 0;
    for(j=0;j<(m->npes); j++) {
      int tpe = m->peNums[j];
      for(k=0;k<(m->nnodes);k++) {
        int tnode = m->gNodeNums[k];
        if(comm[tnode*nparts+tpe] != (-1)) { m->nodesPerPe[icomm++] = k; }
      }
    }
    assert(icomm==tcomm);
  }
  delete[] comm;

// Epilogue:
#if FEM_FORTRAN
  // Make everything 1-based again
  for(i=0;i<nparts;i++) {
    ChunkMsg *m = msgs[i];
    for(j=0;j<(m->nnodes);j++) { m->gNodeNums[j]++; }
    for(j=0;j<(m->nelems);j++) { m->gElemNums[j]++; }
    for(j=0;j<(m->nelems*m->nconn);j++) { m->conn[j]++; }
  }
#endif 
  delete[] pinfo;
  delete[] ninfo;
  return;
}
