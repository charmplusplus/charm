/* Generate a random graph, given the number of nodes, the number of
   connections per node.

dump graph if "tofile" is true
Output form: directory graphN/ containing files graph0 ... graphN-1
             file graphK has C, the number of connections, followed by
             a list of C vertices that K is connected to

Modified from the original: changed output format, and converted main to a parametered function
*/
#include <stdio.h>
#include <stdlib.h>

/* comment this out to test and change CmiPrintf to printf */
//#include "converse.h"
#include "typedefs.h"

int addEdge(EdgeListType *l,int fm,int to);
void addspEdge(EdgeListType *, int, int);
int edgeExists(int fm, int to);
static Q * makeQueue();
static int isEmpty(Q*);
static int dequeue(Q*);

#define VMAX 2050
int V; /* no. of vertices */
int E; /* no. of edges */
int C; /* no. of connections per vertex */
int seed;

VerticesListType graph;

VerticesListType * InitVertices();


/* For testing... 
main(int argc, char **argv)
{
  gengraph(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
}
*/

static void printOut(VerticesListType *vertices);
static void copyOut(VerticesListType *vertices, int *npe, int *pes,int mype);
static void readFromFile(int *npe, int *pes,int mype);
static void initGraph(void);
static void diameter(void);
static void AddEdges(EdgeListType *EdgeList, int V, int n);
int compare (const void * a, const void * b)
{
   int p=*((int *)(*((int **)b)) +1);
   int q=*((int *)(*((int **)a)) +1);
   
   return (p-q);
}
void gengraph(long int pV, int pC, int pseed, int *pes, int *npe, int tofile,int mype)
{ int i,j;
  EdgeListType * EdgeList;
  /* VerticesListType * vertices; */
  extern EdgeListType * InitEdgeList();
  char dircmd[20], dirname[10],filename[20];
  FILE *fp;

  V = pV;
  C = pC;
  seed = 100;
  
  if (CmiMyPe() == 0)
  CmiPrintf("for %d PEs, connectivity %d... \n", V, C);

  /* for (i=0; i<seed; i++) rand(); */
  //for (i=0; i<seed; i++) CrnRand();
  if ((V*C %2) != 0) { printf("V*C must be even\n");CkExit();}
  E = V*C/2;
  
  /*  vertices = (VerticesListType *) InitVertices(EdgeList, V,E); */
  
  sprintf(filename, "graph%dC%d/graph0", V,C);
  if ((fp=fopen(filename,"r"))==NULL) {
    /* make a directory */
    sprintf(dirname, "graph%dC%d", V, C);
    sprintf(dircmd, "mkdir %s", dirname);
    system(dircmd);
    if (mype==0){
     initGraph();
     EdgeList = InitEdgeList(E);
     AddEdges(EdgeList, V, E); 
     printOut(&graph);
     copyOut(&graph, npe, pes, mype);
     if (CmiMyPe()==0) diameter();

	}
  }
  else 
	{
          fclose(fp);
          readFromFile(npe,pes,mype);
        }

  /*int * adj;
  Vertex * vertexRecs;

  
  VerticesListType *vertices=&graph;
  
  adj = vertices->adjArray;
  vertexRecs = vertices->vertexArray;
  for (i=0; i<vertices->numVertices; i++)
   {
    printf("%d ", vertexRecs[i].degree);
    for (j=0; j<vertexRecs[i].degree; j++)
       printf("%d ", adj[ vertexRecs[i].adjListInd + j ]);
    printf("\n");
   }

  printf("%d\n",*npe);
  for (i=0;i<*npe;i++) printf("%d ",pes[i]);*/


}

#if 0
static void AddEdges(EdgeListType *EdgeList, int V, int n)
   /* Add more edges to make up a total of E edges */
{int i,j,w,x,y;
/* first add edges for a C-way spanning tree.*/
int c1;
if (C>1) c1 = C-1;

for (i=0; i< V/c1; i++)
  for (j=0; j<c1; j++) {

      w = c1*i + j +1; 
      /* printf("considering (%d, %d)\n", i, w);  */
      if (w < V) {
	addEdge(EdgeList,i,w);
	/* printf(" --- added.\n"); */
      }
      else /*printf(" not added.\n")*/; 
  }
n -= (V-1);

 for (i=0; i<n; i++) 
   {
     do {
       do {
	 x = CrnRand() % V;
       } while (connections(x) >= C);
       do {
	 y = CrnRand() % V;
       } while ((y== x) || connections(y) >= C);
     } while (edgeExists(x,y));
     addEdge(EdgeList, x, y);
   }
}
#endif

static void AddEdges(EdgeListType *EdgeList, int V, int n)
	/* Add more edges to make up a total of E edges */
  {	int i,j,w,x,y,k,m;
	int c1,max,maxi,temp;
	int **varr;
	int varrlen,count=0;
	int flag=0;
        
	/* first add edges for a C-way spanning tree.*/
        varr=(int **)calloc(V, sizeof(int*));
        for (i=0;i<V;i++)
            varr[i]=calloc(2, sizeof(int));
	
	if (C>1) c1 = C-1;

	for (i=0; i< V/c1; i++)
	  for (j=0; j<c1; j++) {
	      w = c1*i + j +1; 
	      if (w < V) {
		addEdge(EdgeList,i,w);
		count++;
			      }
	     	  }
	
	/*varr is array of vertices and free connection for each vertex*/
	j=0;
	for (i=0;i<V;i++)
		if(connections(i)<C)
		{
		 varr[j][0]=i;
		 varr[j][1]=C-connections(i);
		 j++;
		}
	varrlen=j;
	
      //  for (i=0;i<j;i++) {printf("%d %d\n",varr[i][0],varr[i][1]);}
	/*for all edges except last 10 , edge is formed by randomly selecting vertices from varr*/

	n -= count;

	qsort(varr, varrlen, (sizeof(int*)), compare);
	k=n;
	for (j=0;j<k;j++)
		{
			        //printf("\nUnsorted \n");
                                	//for (i=0;i<varrlen;i++) {printf("%d %d\n",varr[i][0],varr[i][1]);}
                                //  flag=0;
			  	max=0;
				maxi=0;

				/*for (i=0;i<varrlen-1;i++)
                                   for (m=i+1;m<varrlen;m++)
                                         if (varr[i][1]<varr[m][1])     
  						{
 							int temp=varr[i][1];
							varr[i][1]=varr[m][1];
							varr[m][1]=temp;
                                                        temp=varr[i][0];
                                                        varr[i][0]=varr[m][0];
                                                        varr[m][0]=temp;
						}*/

                                //qsort(varr, varrlen, (sizeof(int*)), compare);
                                //printf("\nSorted \n");
                                	//for (i=0;i<varrlen;i++) {printf("%d %d\n",varr[i][0],varr[i][1]);}
  

			       //if (varr[i][1]>max) { max=varr[i][1];maxi=i;}
                                                             
				x = 0;
				y = 1;
				if (edgeExists(varr[x][0],varr[y][0])) 
					{
					      int z=y;
 					      do {
				     			y++;
				         	 } while ((edgeExists(varr[x][0],varr[y][0]))&&(y<varrlen));
                                              if (y==varrlen)
					         { addspEdge(EdgeList,varr[x][0],varr[z][0]);y=z;}
					      else	
						 addEdge(EdgeList,varr[x][0],varr[y][0]);
					}
				else addEdge(EdgeList,varr[x][0],varr[y][0]); 
                                
				varr[x][1]=varr[x][1]-1;
	     			varr[y][1]=varr[y][1]-1;

				while ((varr[y][1]<varr[y+1][1])&&(y<varrlen-1)) {
					temp=varr[y][1];
					varr[y][1]=varr[y+1][1];
					varr[y+1][1]=temp;
					temp=varr[y][0];
					varr[y][0]=varr[y+1][0];
					varr[y+1][0]=temp;
					y=y+1;
				}

				while ((varr[x][1]<varr[x+1][1])&&(x<varrlen-1)) {
					temp=varr[x][1];
					varr[x][1]=varr[x+1][1];
					varr[x+1][1]=temp;
					temp=varr[x][0];
					varr[x][0]=varr[x+1][0];
					varr[x+1][0]=temp;
					x=x+1;
				}

						
				if (varr[x][1]==0) 
					{
					 flag=1;
					 for (i=x;i<varrlen-1;i++)
						{	
				 		 varr[i][0]=varr[i+1][0];
						 varr[i][1]=varr[i+1][1];
						}
					 varrlen--;
					}	
				if ((y>x)&&(flag))         
					{
				         if (varr[y-1][1]==0)   
							{
							  for (i=y-1;i<varrlen-1;i++)
								{	
							 	 varr[i][0]=varr[i+1][0];
								 varr[i][1]=varr[i+1][1];
								}
				 			  varrlen--;
							}
					}			        
		             	else if (varr[y][1]==0)
					{
						 for (i=y;i<varrlen-1;i++)
							{	
					 		 varr[i][0]=varr[i+1][0];
							 varr[i][1]=varr[i+1][1];
							}
						 varrlen--;
					}
			}	      
        for (i=0;i<V;i++) free(varr[i]);
        free(varr);
}


void fillAdjArray(Edge *edges, VerticesListType *vlist, int V, int E);
void sortAdjArrays(VerticesListType *vlist);
static void sort(int *adj, int fromIndex, int toIndex);
void countDegrees(Edge *edges, Vertex *vertRecs, int V, int E);

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
  _MEMCHECK(vlist);
  vlist->numVertices = V;
  vlist->vertexArray = (Vertex *) malloc(V*sizeof(Vertex));
  _MEMCHECK(vlist->vertexArray);
  vlist->adjArray = (int *) malloc(2*E*sizeof(int)); 
                    /* as each edge is entered twice */
  _MEMCHECK(vlist->adjArray);
  countDegrees(EdgeList->edges, vlist->vertexArray, V, E);
  fillAdjArray(EdgeList->edges, vlist, V, E);
  sortAdjArrays(vlist);
  return(vlist);
}

void countDegrees(Edge *edges, Vertex *vertRecs, int V, int E)
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

void fillAdjArray(Edge *edges, VerticesListType *vlist, int V, int E)
{ /* Insert each edge <x,y> as an entry y in x's adj list, and vice versa. */
  int i, x,y;
 int * adj;
 Vertex * vertexRecs;
 printf("%d %d %d\n",x,y,graph.vertexArray[x].degree);

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

void sortAdjArrays(VerticesListType *vlist)
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

static void sort(int *adj, int fromIndex, int toIndex)
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

static void copyOut(VerticesListType *vertices, int *npe, int *pes,int mype)
{ 
 int i;
 int * adj;
 Vertex * vertexRecs;
 
 adj = vertices->adjArray;
 vertexRecs = vertices->vertexArray;

 *npe = vertexRecs[CmiMyPe()].degree;
 for (i=0; i<vertexRecs[CmiMyPe()].degree; i++)
       pes[i] = adj[ vertexRecs[CmiMyPe()].adjListInd + i ];
}

static void readFromFile(int *npe, int *pes,int mype)
{ 
 int i;
 char filename[30];
 FILE *fp;
 sprintf(filename, "graph%dC%d/graph%d", V,C,mype);
 //printf("%s\n",filename);
 fp=fopen(filename,"r");
 if(fp!=NULL){
 fscanf(fp,"%d",npe);
 //printf("%d\n",*npe);
 
  for (i=0; i<*npe; i++){
	 fscanf(fp,"%d",&pes[i]);
         //printf("%d %d\n",i,pes[i]);
 }
 fclose(fp);
 }
 else {printf("Error reading files...quitting!\n");CkExit();}
}

static void printOut(VerticesListType *vertices)
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
     sprintf(filename, "graph%dC%d/graph%d", vertices->numVertices,C, i);
     fp = fopen(filename, "w");
     fprintf(fp, "%d ", vertexRecs[i].degree);
     for (j=0; j<vertexRecs[i].degree; j++)
       fprintf(fp, "%d ", adj[ vertexRecs[i].adjListInd + j ]);
     fprintf(fp, "\n");
     fclose(fp);
   }
}

static void initGraph(void)
{ int i;
  graph.numVertices = V;
  graph.vertexArray = (Vertex *) malloc(V*sizeof(Vertex));
  _MEMCHECK(graph.vertexArray);
  graph.adjArray = (int *) malloc(2*E*sizeof(int));
  _MEMCHECK(graph.adjArray);
  for (i=0; i< V; i++) {
    graph.vertexArray[i].degree = 0;
    graph.vertexArray[i].next = i*C;
    graph.vertexArray[i].adjListInd = i*C;
  }
}

static void enqueue(Q *q, int i);

static void diameter(void)
{
  Q * makeQueue();
  int i,j, k, v, w, start;
  int *distance;
  int *histogram;
  Q * q;
  int dia;
  float average;

  distance = (int *)calloc(V, sizeof(int));
  histogram = (int *)calloc(V, sizeof(int));

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
  average = average / ((long int)V*(long int)V);
  printf("the diameter is: %d, average internode distance = %f\n", 
	 dia, average);
  /*for (i = 0; i< 6; i++) printf("histo[%d] = %d\n", i, histogram[i]);*/
}

/* ------------------------------------------------- */
/* The queue ADT */

static Q * makeQueue()
{
  Q *q = (Q *) malloc(sizeof(Q));
  _MEMCHECK(q);
  q->size = VMAX;
  q->numElements = 0;
  q->head = 1;
  q->tail = 0;
  q->buf = (int *) malloc(VMAX*sizeof(int));
  _MEMCHECK(q->buf);
  return q;
}

static void enqueue(Q * q, int i) {
  q->tail++;
  if (q->tail == q->size) q->tail = 0; /* no overflows possible */
  q->buf[q->tail] = i;
  q->numElements++;
}

static int dequeue(Q *q) {
  int r;
  r = q->buf[q->head++];
  if (q->head == q->size) q->head = 0;
  q->numElements--;
  return r;
}

static int isEmpty(Q * q) {
  return (q->numElements == 0) ;
}
