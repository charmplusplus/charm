#include <stdio.h>
#include <stdlib.h>
#define CHK(p) do\
               {\
                 if ((p)==0)\
                 {\
                   printf("mesh2graph>Memory Allocation failure.");\
                   exit(1);\
                 }\
               } while (0)
#include <assert.h>

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
    // see if nbrs is full
    // if yes, allocate more space, and copy existing nbrs there
    // delete old space
    if (sn <= nn)
    {
      sn *= 2;
      int *tnbrs = new int[sn];
      for (int i=0; i<nn; i++)
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
    if(i%100 == 0) printf("processing element %d\n", i);
    for (k=i+1; k<ne; k++)
    {
      int found = 0;
      for (j=0; j<esize && !found; j++)
      {
        int n1 = conn[i*esize+j];
        for (l=0; l<esize; l++)
        {
          int n2 = conn[k*esize+l];
          if (n1 == n2) // elements i and k share a node
          {
            v[i].add(k);
            v[k].add(i);
            found = 1;
            break;
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

static void
usage (char *pgm)
{
  fprintf(stderr, "Usage: %s <meshfile> <graphfile>\n", pgm);
  exit(1);
}

int 
main (int argc, char **argv)
{
  if (argc != 3)
    usage(argv[0]);
  FILE *fp = fopen(argv[1], "r");
  FILE *fo = fopen(argv[2], "w");
  if (fp==0)
  { 
    fprintf(stderr, "cannot open %s for reading.\n", argv[1]);
    exit(1);
  }
  if (fo==0)
  { 
    fprintf(stderr, "cannot open %s for writing.\n", argv[2]);
    exit(1);
  }
  int nelems, nnodes, esize;
  printf("reading mesh file...\n");
  fscanf(fp, "%d%d%d", &nelems, &nnodes, &esize);
  int *conn = new int[nelems*esize]; CHK(conn);
  int i, j;
  for (i=0;i<nelems;i++)
    for (j=0;j<esize;j++)
      fscanf(fp, "%d", &conn[i*esize+j]);
  fclose(fp);
  printf("finished reading mesh file...\n");
  int *xadj = new int[nelems+1]; CHK(xadj);
  printf("calling mesh2graph...\n");
  int *adjncy = mesh2graph(nelems, esize, conn, xadj);
  printf("mesh2graph returned...\n");
  printf("writing graph file...\n");
  fprintf(fo, "%d %d %d\n", nelems, nnodes, esize);
  for (i=0;i<nelems;i++)
  {
    for (j=0;j<esize;j++)
      fprintf(fo, "%d ", conn[i*esize+j]);
    fprintf(fo, "\n");
  }
  for(i=0;i<(nelems+1);i++)
    fprintf(fo, "%d\n", xadj[i]);
  for(i=0;i<nelems;i++)
  {
    for(j=xadj[i];j<xadj[i+1];j++)
      fprintf(fo, "%d ", adjncy[j]);
    fprintf(fo, "\n");
  }
  fclose(fo);
  printf("graph file written...\n");
  delete[] conn;
  delete[] adjncy;
  return 0;
}
