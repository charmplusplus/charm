#define CHK(p) do\
               {\
                 if ((p)==0)\
                 {\
                   printf("mesh2graph>Memory Allocation failure.");\
                   exit(1);\
                 }\
               } while (0)

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

class Mesh
{
  int _nelems, _nnodes, _esize;
  int *conn;
 public:
  Mesh() { conn = 0; _nelems = _nnodes = _esize = 0; }
  ~Mesh() { delete [] conn; }
  void load(char *file);	// Load a mesh from file
  int nelems() { return _nelems; }
  int nnodes() { return _nnodes; }
  int esize() { return _esize; }
  int node(int elem, int nnode);
};

class NList
{
  int nn; // number of current elements
  int sn; // size of array n
  int *elts; // list of elts
  static int cmp(const void *v1, const void *v2);
 public:
  NList(void) { sn = 0; nn = 0; elts = 0; }
  void init(int _sn) { sn = _sn; nn = 0; elts = new int[sn]; }
  void add(int elt);
  int found(int elt);
  void sort(void) { qsort(elts, nn, sizeof(int), cmp); };
  int getnn(void) { return nn; }
  int getelt(int n) { assert(n < nn); return elts[n]; }
  ~NList() { delete [] elts; }
};

class Nodes
{
  int nnodes;
  NList *elts;
 public:
  Nodes(int _nnodes);
  ~Nodes() { delete [] elts; }
  void add(int node, int elem);
  int nelems(int node) 
  {
    assert(node < nnodes);
    return elts[node].getnn();
  }
  int getelt(int node, int n)
  {
    assert(node < nnodes);
    return elts[node].getelt(n);
  }
};

class Graph
{
  int nelems;
  NList *nbrs;
 public:
  Graph(int elems);
  ~Graph() { delete [] nbrs; }
  void add(int elem1, int elem2);
  int elems(int elem)
  {
    assert(elem<nelems);
    return nbrs[elem].getnn();
  }
  void save(char *file, Mesh *m);
};

void Mesh::load(char *file)
{
  fprintf(stderr, "reading mesh file\n");

  FILE *fp;

  if((fp=fopen(file, "r")) == NULL) {
    perror(file);
    exit(1);
  }

  fscanf(fp, "%d%d%d", &_nelems, &_nnodes, &_esize);

  conn = new int[_nelems*_esize]; CHK(conn);
  int i, j;
  for (i=0;i<_nelems;i++)
    for (j=0;j<_esize;j++)
      fscanf(fp, "%d", &conn[i*_esize+j]);
  fclose(fp);

}

int Mesh::node(int elem, int nnode) 
{
  assert(elem < _nelems);
  assert(nnode < _esize);
  return conn[elem*_esize + nnode];
}

int NList::cmp(const void *v1, const void *v2)
{
  int *e1 = (int *) v1;
  int *e2 = (int *) v2;
  if(*e1==*e2) return 0;
  else if(*e1 < *e2) return -1;
  else return 1;
}

void NList::add(int elt) 
{
  // see if elts is full
  // if yes, allocate more space, and copy existing nbrs there
  // delete old space

  if (sn <= nn) {
    sn *= 2;
    int *telts = new int[sn];
    for (int i=0; i<nn; i++)
      telts[i] = elts[i];
    delete[] elts;
    elts = telts;
  }

  // add new neighbor
  elts[nn++] = elt;
}

int NList::found(int elt)
{
  for(int i = 0; i < nn; i++) 
  {
    if(elts[i] == elt)
      return 1;
  }
  return 0;
}

Nodes::Nodes(int _nnodes) 
{
  nnodes = _nnodes;
  elts = new NList[nnodes];

  for(int i=0; i<nnodes; i++) {
    elts[i].init(10);
  }
}

void Nodes::add(int node, int elem) 
{
  assert(node < nnodes);
  elts[node].add(elem);
}

Graph::Graph(int _nelems) 
{
  nelems = _nelems;
  nbrs = new NList[nelems];

  for(int i=0; i<nelems; i++) 
  {
    nbrs[i].init(10);
  }
}

void Graph::add(int elem1, int elem2) 
{
  assert(elem1 < nelems);
  assert(elem2 < nelems);

// eliminate duplicates

  if(!nbrs[elem1].found(elem2)) 
  {
    nbrs[elem1].add(elem2);
    nbrs[elem2].add(elem1);
  }
}

void Graph::save(char *file, Mesh *m) 
{
  FILE *fp;

  if((fp = fopen(file, "w")) == NULL) {
    perror(file);
    exit(1);
  }

  printf("writing graph file...\n");
    
  fprintf(fp, "%d %d %d\n", m->nelems(), m->nnodes(), m->esize());

  for(int i=0; i<m->nelems(); i++) {
    for(int j = 0; j < m->esize(); j++)
      fprintf(fp, "%d ", m->node(i, j));
    fprintf(fp, "\n");
  }

  int *xadj = new int[nelems+1];
  xadj[0] = 0;
  for(int i=1; i<nelems+1; i++)
    xadj[i] = xadj[i-1] + elems(i-1);

  for(int i = 0; i < nelems+1; i++)
    fprintf(fp, "%d\n", xadj[i]);

  for(int i = 0; i < nelems; i++) {
    nbrs[i].sort();
    for(int j = 0; j < nbrs[i].getnn(); j++)
      fprintf(fp, "%d ", nbrs[i].getelt(j));
    fprintf(fp, "\n");
  }

  fclose(fp);
}

void mesh2graph(Mesh *m, Graph *g)
{
  int nelems = m->nelems();
  int nnodes = m->nnodes();
  int esize = m->esize();

  Nodes nl(nnodes);

  for(int i = 0; i < nelems; i++)
    for(int j = 0; j < esize; j++) {
      nl.add(m->node(i, j), i);
    }

  // nl to graph
    
  for(int i = 0; i < nnodes; i++) {
    int nn = nl.nelems(i);
    for(int j = 0; j < nn; j++) {
      int e1 = nl.getelt(i, j);
      for(int k = j + 1; k < nn; k++) {
        int e2 = nl.getelt(i, k);
        g->add(e1, e2);
      }
    }
  }
}

int main(int argc, char **argv) 
{
  Mesh *m = new Mesh; CHK(m);

  if(argc != 3) {
    fprintf(stderr, "Usage: %s <mesh-file> <graph-file>\n", argv[0]);
    exit(0);
  }

  m->load(argv[1]);

  Graph* g = new Graph(m->nelems()); CHK(g);

  mesh2graph(m, g);

  g->save(argv[2], m);

  delete m;
  delete g;

  return 0;
}
