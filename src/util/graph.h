/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GRAPH_H
#define _GRAPH_H

typedef struct {
  int index;
  float weight;
  int firstEdge;
  int numEdges;
} VertexRecord;


typedef struct {
  int V, E;
  VertexRecord * vertices;
  int * edges;
  int currentVertex; /* needed during construction of graph */
  int currentEdge; /* needed during construction of graph */
} Graph;


Graph * generateRandomGraph(int numNodex);

#endif

/*@}*/
