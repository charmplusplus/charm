/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/* The data structure/ ADT for the edge-list */
#ifdef   WIN32
#include <stdio.h>
#include <stdlib.h>
#endif

#include "typedefs.h"

extern VerticesListType graph;

void * InitEdgeList(E)
int E;
{
  EdgeListType * edgesRec;

  edgesRec = (EdgeListType *) malloc(sizeof(EdgeListType));
  _MEMCHECK(edgesRec);
  edgesRec->next = 0;
  edgesRec->edges = (Edge *) malloc(E*sizeof(Edge));
  _MEMCHECK(edgesRec->edges);
  return(edgesRec);
}

void addEdge(EdgeList, v,w)
     EdgeListType * EdgeList;
     int v;
     int w;
{ int n, index;
  n = EdgeList->next;
  EdgeList->next++;

  /* printf("adding edge: (%d, %d)\n", v, w); */
  ((EdgeList->edges)[n]).node1 = v;
  (EdgeList->edges[n]).node2 = w;
   index =  graph.vertexArray[v].next++;
   graph.adjArray[ index ] = w;
   index =  graph.vertexArray[w].next++;
   graph.adjArray[ index ] = v;

   graph.vertexArray[v].degree++;
   graph.vertexArray[w].degree++;
}

void printEdges(EdgeList)
     EdgeListType * EdgeList;
{int i;
 Edge * edges;
 edges = EdgeList->edges;
 for (i=0; i< (EdgeList->next ); i++)
   {printf("%d\t%d\n", edges[i].node1, edges[i].node2);
  }
}

int edgeExists(x,y)
{
  int i, ind;
  ind = graph.vertexArray[x].adjListInd; 
  
  for(i=0; i< graph.vertexArray[x].degree; i++)
    { if (graph.adjArray[ind + i] == y) return 1;}
  
  return 0;
}
