#if CMK_SEQUENTIAL

#include <stdio.h>
#include <stdlib.h>
#ifdef CkPrintf
#undef CkPrintf
#define CkPrintf printf
#endif
#ifdef CkError
#undef CkError
#define CkError  printf
#endif
#define CrnRand  rand
#define CHK(p) do\
               {\
                 if ((p)==0)\
                 {\
                   printf("MAP>Memory Allocation failure.");\
                   exit(1);\
                 }\
               } while (0)
typedef struct _chunkMsg {
  int nnodes, nelems, npes, nconn;
  int *gNodeNums; // gNodeNums[nnodes]
  int *primaryPart; // primaryPart[nnodes]
  int *gElemNums; // gElemNums[nelems]
  int *conn; // conn[nelems][nconn]
  int *peNums; // peNums[npes]
  int *numNodesPerPe; // numNodesPerPe[npes]
  int *nodesPerPe; // nodesPerPe[npes][nodesPerPe[i]]
} ChunkMsg;
#define FEM_TRIANGULAR 1
#define FEM_HEXAHEDRAL 3
#else

#include "fem.h"

#endif

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
 * nelems: (in) total number of elements
 * nnodes: (in) total number of nodes
 * esize: (in) nodes per element
 * connmat: (in) total connectivity matrix, always row major[nelems][esize]
 * nparts: (in) required number of partitions
 * epart: (in) element partitioning by metis, 
 *        epart[i] contains partition for elem i
 * msgs: (out) messages produced by this routine
 */

void
fem_map (int nelems, int nnodes, int esize, int *connmat,
         int nparts, int *epart, ChunkMsg *msgs[])
{
  int i,j,k;
  int *pinfo = new int[nparts]; CHK(pinfo);
  int *ninfo = new int[nnodes]; CHK(ninfo);
  for (i=0;i<nparts;i++)
  {
    msgs[i] = new ChunkMsg; CHK(msgs[i]);
    msgs[i]->nelems = 0;
    msgs[i]->nconn = esize;
  }
#if FEM_FORTRAN
  // Make everything 0-based
  for (i=0;i<nelems;i++)
    epart[i]--;
  for (i=0;i<(nelems*esize);i++)
    connmat[i]--;
#endif 
  // count the number of elements assigned to each partition
  for (i=0;i<nelems;i++)
    msgs[epart[i]]->nelems++;
  for (i=0;i<nparts;i++)
  { 
    msgs[i]->gElemNums = new int[msgs[i]->nelems]; CHK(msgs[i]->gElemNums);
    msgs[i]->conn = new int[msgs[i]->nelems*esize]; CHK(msgs[i]->conn);
  }
  // pinfo[i] contains filled global element numbers for partition i
  for (i=0;i<nparts;i++)
    pinfo[i] = 0;
  // fill in the global element numbers
  for (i=0;i<nelems;i++)
  { 
    int ep = epart[i];
    msgs[ep]->gElemNums[pinfo[ep]++] = i;
  }
  for (i=0;i<nparts;i++)
  {
    ChunkMsg *m = msgs[i];
    m->nnodes = 0;
    // ninfo[j] is true if node j appears in partition i
    for (j=0;j<nnodes;j++)
      ninfo[j] = 0;
    // scan the connectivity matrix to see which nodes appear in a partition
    // a node appears in a partition if at least one element connected to it
    // belongs to that partition
    for (j=0;j<nelems;j++)
    {
      // check if element belongs to this partition
      if (epart[j]==i)
      {
        // make all its bordering nodes belong to this partition
        // if not already here
        for (k=0;k<esize;k++)
        {
          if (ninfo[connmat[j*esize+k]]==0)
          {
            ninfo[connmat[j*esize+k]] = 1;
            m->nnodes++;
          }
        }
      }
    }
    m->gNodeNums = new int[m->nnodes]; CHK(m->gNodeNums);
    m->primaryPart = new int[m->nnodes]; CHK(m->primaryPart);
    m->nnodes = 0;
    for (j=0;j<nnodes;j++)
    {
      if (ninfo[j]==1)
      {
        m->gNodeNums[m->nnodes] = j;
        // ninfo now contains the local index for a node
        ninfo[j] = m->nnodes++;
      }
      else
      {
        // 0 is a valid local index, so set to (-1) to denote absence
        ninfo[j] = (-1);
      }
    }
    // now fill in the connectivity matrix with local node numbers
    for (j=0;j<m->nelems;j++)
    {
      int telem = m->gElemNums[j];
      for (k=0;k<esize;k++)
      {
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
  for (i=0;i<(nnodes*nparts);i++)
    comm[i] = (-1);
  // fill comm
  for (j=0;j<nparts;j++)
  {
    ChunkMsg *m = msgs[j];
    for (i=0;i<(m->nnodes);i++)
    {
      // tnode is the i'th node of partition j
      int tnode = m->gNodeNums[i];
      comm[tnode*nparts+j] = i;
    }
  }
  // now find the primary partition
  // ninfo[i] contains the number of partitions that node i belongs to
  for (i=0;i<nnodes;i++)
    ninfo[i] = 0;
  for (i=0;i<nnodes;i++)
  {
    for (j=0;j<nparts;j++)
    {
      // if node i belongs to partition j, increment ninfo[i]
      if (comm[i*nparts+j] != (-1))
        ninfo[i]++;
    }
  }
  // now generate a random partition number among those 
  // that the node belongs to. At the end of this loop, ninfo[i] contains
  // the primary partition number for node i
  for (i=0;i<nnodes;i++)
  {
    assert(ninfo[i]>=1);
    int ridx = CrnRand()%ninfo[i];
    k = (-1);
    for (j=0;j<nparts;j++)
    {
      if (comm[i*nparts+j] != (-1))
      { 
        k++; 
        if (k==ridx)
          break;
      }
    }
    assert(j!=nparts);
    ninfo[i] = j;
  }
  // now assign the primary partition number to nodes in each of the messages
  for (i=0;i<nparts;i++)
  {
    ChunkMsg *m = msgs[i];
    for (j=0;j<(m->nnodes);j++)
      m->primaryPart[j] = ninfo[m->gNodeNums[j]];
  }
  for (i=0;i<nparts;i++)
  {
    ChunkMsg *m = msgs[i];
    m->npes = 0;
    // pinfo[j] contains number of nodes that partition i needs to 
    // communicate with partition j
    for (j=0;j<nparts;j++)
      pinfo[j] = 0;
    for (j=0;j<nparts;j++)
    {
      if (i==j)
        continue;
      for (k=0;k<(m->nnodes);k++)
      {
        int tnode = m->gNodeNums[k];
        // if tnode also belongs to partition j, i has to communicate with j
        if (comm[tnode*nparts+j] != (-1))
        {
          if (pinfo[j]==0)
            m->npes++;
          pinfo[j]++;
        }
      }
    }
    m->numNodesPerPe = new int[m->npes]; CHK(m->numNodesPerPe);
    m->peNums = new int[m->npes]; CHK(m->peNums);
    for (j=0,k=0; j<nparts; j++)
    {
      if (pinfo[j]>0)
      {
        m->numNodesPerPe[k] = pinfo[j];
        m->peNums[k] = j;
        k++;
      }
    }
    int tcomm = 0;
    for (j=0; j<(m->npes); j++)
      tcomm += m->numNodesPerPe[j];
    m->nodesPerPe = new int[tcomm]; CHK(m->nodesPerPe);
    int icomm = 0;
    for (j=0;j<(m->npes); j++)
    {
      int tpe = m->peNums[j];
      for (k=0;k<(m->nnodes);k++)
      {
        int tnode = m->gNodeNums[k];
        if (comm[tnode*nparts+tpe] != (-1))
          m->nodesPerPe[icomm++] = k;
      }
    }
    assert(icomm==tcomm);
  }
  delete[] comm;

// Epilogue:
#if FEM_FORTRAN
  // Make everything 1-based again
  for (i=0;i<nparts;i++)
  {
    ChunkMsg *m = msgs[i];
    for (j=0;j<(m->nnodes);j++)
      m->gNodeNums[j]++;
    for (j=0;j<(m->nelems);j++)
      m->gElemNums[j]++;
    for (j=0;j<(m->nelems*m->nconn);j++)
      m->conn[j]++;
  }
#endif 
  delete[] pinfo;
  delete[] ninfo;
  return;
}

#if MAP_MAIN

#if FEM_FORTRAN
  static int numflag=1;
#else
  static int numflag = 0;
#endif

#if MAP_GRAPH
  extern "C" void
  METIS_PartGraphKway (int* nv, int* xadj, int* adjncy, int* vwgt, int* adjwgt,
                       int* wgtflag, int* numflag, int* nparts, int* options,
                       int* edgecut, int* part);
#else
  extern "C" void
  METIS_PartMeshNodal (int* ne, int* nn, int* elmnts, int* etype, int* numflag,
                       int* nparts, int* edgecut, int* epart, int* npart);
#endif

static void
write_partitions (ChunkMsg **msgs, int nparts, int esize)
{
  int i, j, k;
  for (i=0;i<nparts;i++)
  {
    ChunkMsg *m = msgs[i];
    char filename[128];
    sprintf(filename, "meshdata.Pe%d", i);
    FILE *fp = fopen(filename, "w");
    if (fp==0)
    { 
      fprintf(stderr, "cannot open %s for writing.\n", filename);
      exit(1);
    }
    // write nodes
    fprintf(fp, "%d\n", m->nnodes);
    for (j=0;j<(m->nnodes);j++)
      fprintf(fp, "%d %d\n", m->gNodeNums[j], m->primaryPart[j]);
    // write elems
    fprintf(fp, "%d %d\n", m->nelems, esize);
    for (j=0;j<(m->nelems);j++)
    {
      fprintf(fp, "%d ", m->gElemNums[j]);
      for (k=0;k<esize;k++)
        fprintf(fp, "%d ", m->conn[j*esize+k]);
      fprintf(fp, "\n");
    }
    // write comm
    fprintf(fp, "%d\n", m->npes);
    int idx = 0;
    for (j=0; j<(m->npes); j++)
    {
      fprintf(fp, "%d %d ", m->peNums[j], m->numNodesPerPe[j]);
      for (k=0; k<(m->numNodesPerPe[j]); k++)
      {
        fprintf(fp, "%d ", m->nodesPerPe[idx++]);
      }
    }
    fclose(fp);
  }
}

#if MAP_GRAPH

class vertex
{
 private:
  int nn; // number of current neighbors
  int sn; // size of array n
  int *nbrs; // list of nbrs
 public:
  void init(int _sn)
  {
    sn = _sn;
    nbrs = new int[sn]; CHK(nbrs);
    nn = 0;
  }
  void add(int nbr)
  {
    int i;
    // check to see if nbr already exists
    // if yes, return
    for (i=0; i<nn; i++)
    {
      if (nbrs[i] == nbr)
        return;
    }
    // otherwise see if nbrs is full
    // if yes, allocate more space, and copy existing nbrs there
    // delete old space
    if (sn <= nn)
    {
      sn *= 2;
      int *tnbrs = new int[sn];
      for (i=0; i<nn; i++)
        tnbrs[i] = nbrs[i];
      delete[] nbrs;
      nbrs = tnbrs;
    }
    // add new neighbor
    nbrs[nn++] = nbr;
  }
  void destroy(void)
  {
    delete[] nbrs;
  }
  int getnn(void)
  {
    return nn;
  }
  int *getnbrs(void)
  {
    return nbrs;
  }
};

// Converts given mesh to graph (CSR) required by Metis Graph Partitioner
// ne: (in) number of elements
// esize: (in) number of nodes per element
// conn[ne][esize]: (in) connectivity matrix
// xadj[ne+1]: (out) xadj[i] is the starting index of the adjacency list of 
//             element i.
// returns adj[2m] : (out) m is the number of edges. allocated inside this
//                   function, since number of edges is not known in advance.
static int*
mesh2graph(int ne, int esize, int *conn, int *xadj)
{
  int i, j, k, l;
  vertex *v = new vertex[ne]; CHK(v);
  for (i=0;i<ne;i++)
    v[i].init(esize);
  for (i=0; i<(ne-1); i++)
  {
    for (j=0; j<esize; j++)
    {
      int n1 = conn[i*esize+j];
      for (k=i+1; k<ne; k++)
      {
        for (l=0; l<esize; l++)
        {
          int n2 = conn[k*esize+l];
          if (n1 == n2) // elements i and k share a node
          {
            v[i].add(k);
            v[k].add(i);
          }
        }
      }
    }
  }
  int m = 0;
  xadj[0] = 0;
  for (i=0;i<ne;i++)
  {
    m += v[i].getnn();
    xadj[i+1] = m;
  }
  int *adjncy = new int[m]; CHK(adjncy);
  for (k=0, i=0; i<ne; i++)
  {
    int nn = v[i].getnn();
    if (nn==0)
    { 
      fprintf(stderr, "Disconnected Mesh? Element %d has no nbrs\n", i);
      exit(1);
    }
    int *nbrs = v[i].getnbrs();
    for (j=0; j<nn; j++)
    {
      adjncy[k++] = nbrs[j];
    }
  }
  for (i=0;i<ne;i++)
    v[i].destroy();
  delete[] v;
  return adjncy;
}

#endif
static void
usage (char *pgm)
{
  fprintf(stderr, "Usage: %s <meshfile> <nparts>\n", pgm);
  exit(1);
}

int 
main (int argc, char **argv)
{
  if (argc != 3)
    usage(argv[0]);
  FILE *fp = fopen(argv[1], "r");
  if (fp==0)
  { 
    fprintf(stderr, "cannot open %s for reading.\n", argv[1]);
    exit(1);
  }
  int nparts = atoi(argv[2]);
  int nelems, nnodes, ctype;
  fscanf(fp, "%d%d%d", &nelems, &nnodes, &ctype);
#if MAP_GRAPH
  int esize = ctype;
#else
  int esize = (ctype==FEM_TRIANGULAR) ? 3 :
              ((ctype==FEM_HEXAHEDRAL) ? 8 :
              4);
#endif
  int *conn = new int[nelems*esize]; CHK(conn);
  int i, j;
  for (i=0;i<nelems;i++)
  {
    for (j=0;j<esize;j++)
      fscanf(fp, "%d", &conn[i*esize+j]);
  }
  fclose(fp);
  int ecut;
  int *epart = new int[nelems]; CHK(epart);
#if MAP_GRAPH
  int *xadj = new int[nelems+1]; CHK(xadj);
  int *adjncy = mesh2graph(nelems, esize, conn, xadj);
  int wgtflag = 0; // no weights associated with elements or edges
  int opts[5];
  opts[0] = 0; //use default values
  METIS_PartGraphKway(&nelems, xadj, adjncy, 0, 0, &wgtflag, &numflag, &nparts,
                      opts, &ecut, epart);
#else
  int *npart = new int[nnodes]; CHK(npart);
  METIS_PartMeshNodal(&nelems, &nnodes, conn, &ctype, &numflag, 
                      &nparts, &ecut, epart, npart);
  delete[] npart;
#endif
  ChunkMsg **msgs = new ChunkMsg*[nparts]; CHK(msgs);
  fem_map(nelems, nnodes, esize, conn, nparts, epart, msgs);
  delete[] epart;
  delete[] conn;
  write_partitions(msgs, nparts, esize);
  for (i=0;i<nparts;i++) 
    delete msgs[i];
  delete[] msgs;
  return 0;
}

#endif
