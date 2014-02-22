/*Charm++ Finite Element Framework:
C++ implementation file

This code implements exactly one routine: fem_partition.
This partitions a mesh's elements into n chunks, and writes
out each element's 0-based chunk number to an array.

The partitioning is done using metis.

Originally written by Karthik Mahesh, September 2000.
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <metis.h>
#include "fem_impl.h"
#include "cktimer.h"

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
  void toAdjList(int *&adjStart,int *&adjList);
};

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


void Graph::toAdjList(int *&adjStart,int *&adjList)
{
	int e,i;
	adjStart=new int[nelems+1];
	adjStart[0]=0;
	for (e=0;e<nelems;e++)
		adjStart[e+1]=adjStart[e]+elems(e);
	adjList=new int[adjStart[nelems]];
	int *adjOut=adjList;
	for (e=0;e<nelems;e++)
		for (i=nbrs[e].getnn()-1;i>=0;i--)
			*(adjOut++)=nbrs[e].getelt(i);
}


void mesh2graph(const FEM_Mesh *m, Graph *g)
{
  Nodes nl(m->node.size());

  //Build an inverse mapping, from node to list of surrounding elements
  int globalCount=0,t,e,n;
  for(t=0;t<m->elem.size();t++) 
  if (m->elem.has(t)) {
    const FEM_Elem &k=m->elem[t];
    for(e=0;e<k.size();e++,globalCount++)
      for(n=0;n<k.getNodesPer();n++)
        nl.add(k.getConn(e,n),globalCount);
  }
  
  //Convert nodelists to graph:
  // Elements become nodes of graph; nodes become edges of graph.
  // Metis can partition this graph.
  int i, j;    
  for(i = 0; i < m->node.size(); i++) {
    int nn = nl.nelems(i);
    for(j = 0; j < nn; j++) {
      int e1 = nl.getelt(i, j);
      for(int k = j + 1; k < nn; k++) {
        int e2 = nl.getelt(i, k);
        g->add(e1, e2);
      }
    }
  }
}


/*Partition this mesh's elements into n chunks,
 writing each element's 0-based chunk number to elem2chunk.
*/
void FEM_Mesh_partition(const FEM_Mesh *mesh,int nchunks,int *elem2chunk)
{
	CkThresholdTimer time("FEM Split> Building graph for metis partitioner",1.0);
	int nelems=mesh->nElems();
	if (nchunks==1) {//Metis can't handle this case (!)
		for (int i=0;i<nelems;i++) elem2chunk[i]=0;
		return;
	}
	Graph g(nelems);
	mesh2graph(mesh,&g);
	
	int *adjStart; /*Maps elem # -> start index in adjacency list*/
	int *adjList; /*Lists adjacent vertices for each element*/
	g.toAdjList(adjStart,adjList);
	int ecut,ncon=1;
	time.start("FEM Split> Calling metis partitioner");
	if (nchunks<8) /*Metis manual says recursive version is higher-quality here*/
	  METIS_PartGraphRecursive(&nelems, &ncon, adjStart, adjList, NULL, NULL, NULL,
                        &nchunks, NULL, NULL, NULL, &ecut, elem2chunk);
	else /*For many chunks, Kway is supposedly faster */
	  METIS_PartGraphKway(&nelems, &ncon, adjStart, adjList, NULL, NULL, NULL,
                        &nchunks, NULL, NULL, NULL, &ecut, elem2chunk);
	delete[] adjStart;
	delete[] adjList;
}
