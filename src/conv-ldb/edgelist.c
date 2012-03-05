/* The data structure/ ADT for the edge-list */
#include <stdio.h>
#include <stdlib.h>
#include "graphdefs.h"

/*extern VerticesListType graph; */

void * InitEdgeList(int E)
{
  EdgeListType * edgesRec;

  edgesRec = (EdgeListType *) malloc(sizeof(EdgeListType));
  _MEMCHECK(edgesRec);
  edgesRec->next = 0;
  edgesRec->edges = (Edge *) malloc(E*sizeof(Edge));
  _MEMCHECK(edgesRec->edges);
  return(edgesRec);
}

void addEdge(VerticesListType *graph, EdgeListType * EdgeList, int v, int w)
{
  int n, index;
  n = EdgeList->next;
  EdgeList->next++;

  /* printf("adding edge: (%d, %d)\n", v, w); */
  ((EdgeList->edges)[n]).node1 = v;
  (EdgeList->edges[n]).node2 = w;
   index =  graph->vertexArray[v].next++;
   graph->adjArray[ index ] = w;
   index =  graph->vertexArray[w].next++;
   graph->adjArray[ index ] = v;

   graph->vertexArray[v].degree++;
   graph->vertexArray[w].degree++;
}

void printEdges(EdgeListType *EdgeList)
{
 int i;
 Edge * edges;
 edges = EdgeList->edges;
 for (i=0; i< (EdgeList->next ); i++)
   {printf("%d\t%d\n", edges[i].node1, edges[i].node2);
  }
}

int edgeExists(VerticesListType *graph, int x, int y)
{
  int i, ind;
  ind = graph->vertexArray[x].adjListInd; 
  
  for(i=0; i< graph->vertexArray[x].degree; i++)
    { if (graph->adjArray[ind + i] == y) return 1;}
  
  return 0;
}

/*
  If we are adding an edge between two nodes already connected then we 
  need special changes. We open some distinct edge say x,y and 
  connect v with y and x with w
*/

void addspEdge(VerticesListType *graph, EdgeListType * EdgeList, int v, int w)
{ 
  int n, index,i,x,y,ind;
  n = EdgeList->next;
  EdgeList->next++;

  /* printf("adding edge: (%d, %d)\n", v, w); */
  /*((EdgeList->edges)[n]).node1 = v;*/
  /*(EdgeList->edges[n]).node2 = w;*/
   for (i=0;i<n-1;i++)
     if (((EdgeList->edges[i]).node1!=v) && ((EdgeList->edges[i]).node2!=w)) 
	{
	    x=(EdgeList->edges[i]).node1;
	    y=(EdgeList->edges[i]).node2;	
            (EdgeList->edges[i]).node2=w;
            (EdgeList->edges[n]).node1=v;
	    (EdgeList->edges[n]).node2=y;	

            
            ind =  graph->vertexArray[x].adjListInd;
	    for(i=0; i< graph->vertexArray[x].degree; i++)
		    { if (graph->adjArray[ind + i] == y) graph->adjArray[ind+i]=w;}

            ind =  graph->vertexArray[y].adjListInd;
	    for(i=0; i< graph->vertexArray[y].degree; i++)
		    { if (graph->adjArray[ind + i] == x) graph->adjArray[ind+i]=y;}
	
            index =  graph->vertexArray[v].next++;
	    graph->adjArray[ index ] = y;
	    index =  graph->vertexArray[w].next++;
	    graph->adjArray[ index ] = x;
	    graph->vertexArray[v].degree++;
	    graph->vertexArray[w].degree++;
            break;
	 
	}
}

