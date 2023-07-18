#include "converse.h"

typedef struct {int node1, node2;} ConvEdge;

typedef struct {int next; 
	 ConvEdge * edges;} EdgeListType;

typedef struct {
  int degree; 
  int next; /* temporary count needed to tell where to insert the next entry */
  int adjListInd; /* where in the big array does its adj list begin */
  /*  int available;*/ /* number of connections still available. 12/2/97 */
} ConvVertex;

typedef struct {
  int numVertices;
  ConvVertex * vertexArray; /* ptr to an array of records, one for each vertex */
  int * adjArray; /* ptr to an array in which adjacency sub-arrays for each 
		     vertex are stored contiguosly */
} VerticesListType;

#define connections(graph, i) (graph->vertexArray[i].degree)

/* -----*/
typedef struct {
  int size, head, tail;
  int numElements;
  int * buf; }
 Q;

