/* Generate a random graph, given the number of nodes, the number of
   connections per node.

Output form: directory graphN/ containing files graph0 ... graphN-1
             file graphK has C, the number of connections, followed by
             a list of C vertices that K is connected to

Modified from the original: changed output format, and converted main to a parametered function
*/

#include "stdio.h"
#include "typedefs.h"

#define VMAX 2050
int V; /* no. of vertices */
int E; /* no. of edges */
int C; /* no. of connections per vertex */
int seed;

VerticesListType graph;

VerticesListType * InitVertices();

gengraph(int V, int C, int seed)
{ int i;
  EdgeListType * EdgeList;
  VerticesListType * vertices;
  extern EdgeListType * InitEdgeList();
  extern addEdge();
  char dircmd[20], dirname[10];

  for (i=0; i<seed; i++) rand();
  if ((V*C %2) != 0) printf("V*C must be even\n");
  E = V*C/2;

  initGraph(V, E);
  EdgeList = InitEdgeList(E);
  AddEdges(EdgeList, V, E); 
  /*  vertices = (VerticesListType *) InitVertices(EdgeList, V,E); */

  /* make a directory */
  sprintf(dirname, "graph%d", V);
  sprintf(dircmd, "mkdir %s", dirname);
  system(dircmd);
  
  printOut(graph); 
  diameter();
}


AddEdges(EdgeList, V, n)
   /* Add more edges to make up a total of E edges */
     EdgeListType * EdgeList;
     int V;
     int n;
{int i,j,w,x,y;
/* first add edges for a C-way spanning tree.*/
int c1;
c1 = C-1;
for (i=0; i< V/c1; i++)
  for (j=0; j<c1; j++) {
    w = c1*i + j +1; 
    /*      printf("considering (%d, %d)\n", i, w); */
      if (w < V) {
	addEdge(EdgeList,i,w);
	/* printf(" --- added.\n");*/
      }
      /*      else printf(" not added.\n"); */
  }
n -= (V-1);

 for (i=0; i<n; i++) 
   {
     do {
       do {x = rand() % V;} while (connections(x) >= C);
       do {y = rand() % V; } while ((y == x) || connections(y) >= C);
     } while (edgeExists(x,y));
      addEdge(EdgeList, x, y);
    }
}

VerticesListType * 
InitVertices(EdgeList, V,E)
     EdgeListType * EdgeList;
     int V;
     int E;
{ /* returns a structure of type VerticesListType, which contains an arry of 
     vertex records, and an array of adjacency information. See typedef. */
  /* First allocate the adjacency subarray of size E, and vertex subarray size V.
     Then count the occurences of each vertex in the Edgelist in vertex.degree.
     Then compute the real adjListInd = sum from 0 to i-1 of the previous degrees.
     Then, traverse edge list, and enter each edge in two adj lists.
     Then sort individual adj-lists, separately. */
  VerticesListType * vlist;

  vlist = (VerticesListType *) malloc(sizeof(VerticesListType));
  vlist->numVertices = V;
  vlist->vertexArray = (Vertex *) malloc(V*sizeof(Vertex));
  vlist->adjArray = (int *) malloc(2*E*sizeof(int)); 
                    /* as each edge is entered twice */
  countDegrees(EdgeList->edges, vlist->vertexArray, V, E);
  fillAdjArray(EdgeList->edges, vlist, V, E);
  sortAdjArrays(vlist);
  return(vlist);
}

countDegrees(edges, vertRecs, V, E)
     Edge * edges; /* array of Edge records */
     Vertex * vertRecs; /* array of Vertex records */
     int V;
     int E;
{ /* initialize the degrees of all vertices to 0. 
     Traverse the edge list, incrementing the degree of the 2 nodes for each edge.
     */
 int i, count;
 
 for (i=0; i<V; i++)
   { vertRecs[i].degree = 0;
     vertRecs[i].next = 0;}
 for (i=0; i<E; i++)
   {vertRecs[ edges[i].node1 ].degree++;
    vertRecs[ edges[i].node2 ].degree++;}

/* now fill adjListInd, by starting a counter at 0, and adding degrees of visited
   nodes. */
 count = 0;
 for (i=0; i<V; i++)
   { vertRecs[i].adjListInd = count;
     count += vertRecs[i].degree;
   }
}

fillAdjArray(edges, vlist, V, E)
     Edge * edges;
     VerticesListType * vlist;
     int V;
     int E;
{ /* Insert each edge <x,y> as an entry y in x's adj list, and vice versa. */
  int i, x,y;
 int * adj;
 Vertex * vertexRecs;

 adj = vlist->adjArray;
 vertexRecs = vlist->vertexArray;

  for (i=0; i<E; i++)
    { x = edges[i].node1; y = edges[i].node2;
      adj[ vertexRecs[x].adjListInd + vertexRecs[x].next ] = y;
      vertexRecs[x].next++;
      adj[ vertexRecs[y].adjListInd + vertexRecs[y].next ] = x;
      vertexRecs[y].next++;
    }
}

sortAdjArrays(vlist)
     VerticesListType * vlist;
{ /* sort each section of the array corresponding to each vertex. */
  int V, i;
  int dupcount;

  V = vlist->numVertices;
  for (i=0; i<V; i++)
    { sort(vlist->adjArray, 
	   vlist->vertexArray[i].adjListInd,
	   vlist->vertexArray[i].adjListInd + vlist->vertexArray[i].degree -1);
    }
  /* eliminate duplicates. May be should be merged with above? */
  dupcount = 0;
  for (i=0; i<V; i++)
    { int m,n, limit;
      int * adj;

      m = vlist->vertexArray[i].adjListInd;
      n = m+1;
      limit = m + vlist->vertexArray[i].degree; /* this is the first index
						   not belonging to this vertex.*/
      adj = vlist->adjArray;
      /* do this in 2 phases. First phase: m and n are exactly one away.
	 goes on until the first duplicate is found. In this phase, there
	 is no need to assign values (no shifting is going on.) */
      while ((adj[m] != adj[n]) && (n < limit)) {m++; n++;}
      /* Phase 2: */
      while (n<limit) {
	while ((adj[m] == adj[n]) && (n<limit)) 
	  { n++; dupcount++; vlist->vertexArray[i].degree--;}
	adj[m+1] = adj[n]; /*move the non duplicate back to its new position*/
	m++; n++;
      }
    }
/* printf("number of duplicate edges removed = %d\n", dupcount/2);*/
/* Here is an assignment to a global variable.... */
if ((dupcount % 2) != 0) {printf("error. duplicates not even.\n"); }
E -= dupcount/2;
}

sort(adj, fromIndex, toIndex)
     int * adj;
     int fromIndex;
     int toIndex;
{ int i,j, tmp;
  short changed;
  changed = 1;
  for (i=toIndex; ((i>fromIndex) && changed); i--)
    { changed = 0;
      for (j=fromIndex; j<i; j++)
	{ if (adj[j] > adj[j+1])
	    { /* swap */
	      changed = 1;
	      tmp = adj[j];
	      adj[j] = adj[j+1];
	      adj[j+1] = tmp;
	    }
	}
    }
}

printOut(vertices)
VerticesListType * vertices ;
{int i,j;
 int * adj;
 Vertex * vertexRecs;
 FILE *fp;
 char filename[20];
 
 adj = vertices->adjArray;
 vertexRecs = vertices->vertexArray;

 for (i=0; i<vertices->numVertices; i++)
   {
     /* Open graphN/graphi */
     sprintf(filename, "graph%d/graph%d", vertices->numVertices, i);
     fp = fopen(filename, "w");
     fprintf(fp, "%d ", vertexRecs[i].degree);
     for (j=0; j<vertexRecs[i].degree; j++)
       fprintf(fp, "%d ", adj[ vertexRecs[i].adjListInd + j ]);
     fprintf(fp, "\n");
     fclose(fp);
   }
}

initGraph()
{ int i;
  graph.numVertices = V;
  graph.vertexArray = (Vertex *) malloc(V*sizeof(Vertex));
  graph.adjArray = (int *) malloc(2*E*sizeof(int));

  for (i=0; i< V; i++) {
    graph.vertexArray[i].degree = 0;
    graph.vertexArray[i].next = i*C;
    graph.vertexArray[i].adjListInd = i*C;
  }
}


diameter()
{
  Q * makeQueue();
  int i,j, k, v, w, start;
  int distance[V];
  int histogram[V];
  Q * q;
  int dia;
  float average;

  for (i=0; i<V; i++) {
    histogram[i] = 0; 
  }
  
  dia = 0;
  average = 0.0;
  q = makeQueue();
  for (i=0; i<V; i++) {
    /* run non-weighted shortes distance algorithm for each vertex i */
    for (j=0; j<V; j++) distance[j] = -1;
    distance[i] = 0;
    enqueue(q, i);
    while (! (isEmpty(q))) {
      v = dequeue(q);
      
      start=graph.vertexArray[v].adjListInd;
      for (k=0; k< graph.vertexArray[i].degree; k++) {
	w = graph.adjArray[k+start];
	if (distance[w] == -1) {
	  distance[w] = distance[v] + 1;
	  enqueue(q, w);
	}
      }
    }
    for (j=0; j<V; j++) {
      if (distance[j] > dia) dia = distance[j];
      average += distance[j];
      histogram[ distance[j]]++;
    }
  }
  average = average / (V*V);
  printf("the diameter is: %d, average internode distance = %f\n", 
	 dia, average);
  /*   for (i = 0; i< 6; i++) printf("histo[%d] = %d\n", i, histogram[i]);*/
}

/* ------------------------------------------------- */
/* The queue ADT */

Q * makeQueue()
{
  Q *q = (Q *) malloc(sizeof(Q));
  q->size = VMAX;
  q->head = 1;
  q->tail = 0;
  q->buf = (int *) malloc(VMAX*sizeof(int));
  return q;
}

enqueue(Q * q, int i) {
  q->tail++;
  if (q->tail == q->size) q->tail = 0; /* no overflows possible */
  q->buf[q->tail] = i;
  q->numElements++;
}

int dequeue(Q *q) {
  int r;
  r = q->buf[q->head++];
  if (q->head == q->size) q->head = 0;
  q->numElements--;
  return r;
}

isEmpty(Q * q) {
  return (q->numElements == 0) ;
}
