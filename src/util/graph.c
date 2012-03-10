/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "converse.h"
#include "graph.h"
#include <stdlib.h>

#define printf CmiPrintf
void printPartition(Graph * g, int nodes[], int numNodes)
{
  int i;
  for (i=0; i<numNodes; i++)
    printf("\t%d", nodes[i]);
  printf("\n");

}

int intSqrt(int x)
{ 
  int y=1;
  while (y*y<x) y++;
  return y; 
}

Graph * g_initGraph(int V, int E) {
  Graph *g;

  g = (Graph *) malloc(sizeof(Graph));

  g->V = V;
  g->E = E ;
  g->vertices = (VertexRecord *) malloc(V*sizeof(VertexRecord));
  g->edges = (int *) malloc( 2* (1 + g->E) * sizeof(int));
  g->currentVertex = -1;
  g->currentEdge = 0;
  return g;
}

void g_freeGraph(Graph* g)
{
  free(g->vertices);
  g->vertices=0;
  free(g->edges);
  g->edges = 0;
  free(g);
}

void g_nextVertex(Graph *g, int v, float weight)
{
  int current;

  g->currentVertex++;
  current = g->currentVertex; 
  if(current >= g->V)
    printf("current overflow\n");
  g->vertices[current].index = v;
  g->vertices[current].weight = weight;
  g->vertices[current].firstEdge = g->currentEdge;
  g->vertices[current].numEdges = 0;
/*  printf("next vertex is: %d weight:%f\n", current, weight); */

}


void g_addEdge(Graph *g, int w, float weight)
{
  /* for now , ignore weight */
  int v, i;
  v = g->currentVertex;

/*
  CmiPrintf("addEdge: graph: %d, v=%d, w=%d, wt=%f, currentEdge=%d\n", 
	   (int) g, v, w, weight, g->currentEdge, i);
  CmiPrintf("addEdge: firstedge = %d, numEdges = %d\n",
	    g->vertices[v].firstEdge, g->vertices[v].numEdges);
*/
   i = g->vertices[v].firstEdge;
   while (i < g->currentEdge) {
     if (g->edges[i] == w) { 
       return; /* was duplicate edge */ }
     i++;
   }

/*  printf("adding (%d,%d) at %d \n", v, w, g->currentEdge);*/

  g->vertices[v].numEdges++;
  g->edges[g->currentEdge++] = w;

}

void g_finishVertex(Graph *g) {

if (g->vertices[g->currentVertex].numEdges != g->currentEdge - g->vertices[g->currentVertex].firstEdge)
 printf("Error in finishVertex\n");
  /* finish up the previous vertex's record. 
     Nothing needs to be done here in the current scheme.*/
}


Graph *generateRandomGraph(int numNodes) {
  
 Graph *g;

 int i, stride, n;

 g = (Graph *) malloc(sizeof(Graph));
 g->vertices = (VertexRecord *) malloc(numNodes*sizeof(VertexRecord));
 g->V = numNodes;
 
 g->E = 4* g->V ;
 g->edges = (int *) malloc( (1 + g->E) * sizeof(int));
 stride = intSqrt(g->V);

 n = 0;
 for (i = 0; i<g->V; i++) {
   g->vertices[i].index = i;
   g->vertices[i].firstEdge = n;
   g->vertices[i].numEdges = 4;
   g->vertices[i].weight = 1.0;

   g->edges[n++] = (i + numNodes - 1) % numNodes;
   g->edges[n++] = (i + 1) % numNodes;
   
   g->edges[n++] = (i +numNodes - stride) % numNodes;
   g->edges[n++] = (i + stride) % numNodes;
 
 }
return g;
}




void g_printGraph(Graph *g) {
  int i, j;

   CmiPrintf("%d vertices, %d edges \n", g->V, g->E); 
  for (i=0; i< g->V; i++)
    { printf("\n %d: (%d)\t", i, g->vertices[i].numEdges ); 
      for (j=0; j<g->vertices[i].numEdges; j++) 
	printf(" %d,", g->edges[g->vertices[i].firstEdge + j ]); }

 }

int g_numNeighbors(Graph *g, int node) {

  return g->vertices[node].numEdges;
}

int g_getNeighbor(Graph *g, int node, int i) {
  
  if (i >= g->vertices[node].numEdges) {
    printf("error: node %d has only %d neighbors. You asked for %d'th nbr\n",
	   node, g->vertices[node].numEdges, i);
    return 0; 
  }
  return   g->edges [ g->vertices[node].firstEdge + i];
  
}

float graph_weightof(Graph *g, int vertex) {
/*   return 1.0 ; */
  return g->vertices[vertex].weight;

}

/*@}*/
