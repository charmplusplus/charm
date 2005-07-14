// FEM 2D adaptivity and adjacency demo code: Reads in a Triangle mesh, 
// sets up adjacencies, performs refine and coarsen operations, physics-free.
// Questions & comments to TLW.
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include "charm++.h"
#include "fem.h"
#include "ckvector3d.h"
#include "charm-api.h"
#include "fem_mesh.h"
#include "fem_adapt.h"
#include "netfem.h"
#include "vector2d.h"

static void die(const char *str) {
  CkError("Fatal error: %s\n",str);
  CkExit();
}

void resize_nodes(void *data,int *len,int *max);
void resize_elems(void *data,int *len,int *max);
void resize_edges(void *data,int *len,int *max);

extern "C" void
init(void)
{
  CkPrintf("Init started\n");
  const char *eleName="xxx.1.ele";
  const char *nodeName="xxx.1.node";
  const char *edgeName="xxx.1.edge";
  int nPts=0; // Number of nodes
  vector2d *pts=0; // Node coordinates
  char line[1024];
  FILE *f;
  
  // Open and read the node coordinate file
  CkPrintf("Reading node coordinates from %s\n",nodeName);
  f=fopen(nodeName,"r");
  if (f==NULL) die("Can't open node file!");
  fgets(line,1024,f);
  if (1!=sscanf(line,"%d",&nPts)) die("Can't read number of points!");
  pts=new vector2d[nPts];
  for (int i=0;i<nPts;i++) {
    int ptNo;
    if (NULL==fgets(line,1024,f)) die("Can't read node input line!");
    if (3!=sscanf(line,"%d%lf%lf",&ptNo,&pts[i].x,&pts[i].y)) 
      die("Can't parse node input line!");
  }
  fclose(f);
  CkPrintf("Passing node coords to framework\n");
  
  // Register the node entity and its data arrays that will be used later. This
  // needs to be done so that the width of the data segments are set correctly 
  // and can be used later
  FEM_Register_entity(FEM_Mesh_default_write(),FEM_NODE,NULL,nPts,nPts,resize_nodes);
  vector2d *t = new vector2d[nPts];
  FEM_Register_array(FEM_Mesh_default_write(),FEM_NODE,FEM_DATA,t,FEM_DOUBLE,2);
  /*boundary value for nodes*/
  int *nodeBoundary = new int[nPts];
  FEM_Register_array(FEM_Mesh_default_write(),FEM_NODE,FEM_BOUNDARY,nodeBoundary,FEM_INT,1);
  
  int *nodeValid = new int[nPts];
  for (int i=0; i<nPts; i++) {
    nodeValid[i] = 1;
  }
  FEM_Register_array(FEM_Mesh_default_write(),FEM_NODE,FEM_VALID,nodeValid,FEM_INT,1);
  
  int nEle=0;
  int *ele=NULL;
  CkPrintf("Reading elements from %s\n",eleName);
  //Open and read the element connectivity file
  f=fopen(eleName,"r");
  if (f==NULL) die("Can't open element file!");
  fgets(line,1024,f);
  if (1!=sscanf(line,"%d",&nEle)) die("Can't read number of elements!");
  ele=new int[3*nEle];
  for (int i=0;i<nEle;i++) {
    int elNo;
    if (NULL==fgets(line,1024,f)) die("Can't read element input line!");
    if (4!=sscanf(line,"%d%d%d%d",&elNo,&ele[3*i+0],&ele[3*i+1],&ele[3*i+2])) 
      die("Can't parse element input line!");
    ele[3*i+0]--; //Fortran to C indexing
    ele[3*i+1]--; //Fortran to C indexing
    ele[3*i+2]--; //Fortran to C indexing
  }
  fclose(f);
  CkPrintf("Passing elements to framework\n");
  
  // Register the Element entity and its connectivity array. Register the
  // data arrays to set up the widths correctly at the beginning
  FEM_Register_entity(FEM_Mesh_default_write(),FEM_ELEM,NULL,nEle,nEle,resize_nodes);
  FEM_Register_array(FEM_Mesh_default_write(),FEM_ELEM,FEM_CONN,ele,FEM_INDEX_0,3);
  
  int *elemValid = new int[nEle];
  for (int i=0; i<nEle; i++) {
    elemValid[i] = 1;
  }
  FEM_Register_array(FEM_Mesh_default_write(),FEM_ELEM,FEM_VALID,elemValid,FEM_INT,1);

  // Build the ghost layer with only 1 node of adjacency required to add a 
  // ghost element.  Add ghost nodes.
  FEM_Add_ghost_layer(1, 1); 
  const static int tri2node[3]={0,1,2};
  FEM_Add_ghost_elem(0, 3, tri2node); 
  
  //open and read the .edge (edge connectivity file) needed for boundary values
  int nEdge;
  int *edgeConn;
  int *edgeBoundary;
  f=fopen(edgeName,"r");
  if (f==NULL) die("Can't open edge file!");
  fgets(line,1024,f);
  if (1!=sscanf(line,"%d",&nEdge)) die("Can't read number of elements!");
  edgeConn = new int[2*nEdge];
  edgeBoundary = new int[nEdge];
  for(int i=0;i<nEdge;i++){
    int edgeNo;
    if (NULL==fgets(line,1024,f)) die("Can't read edge input line!");
    if (4 != sscanf(line,"%d%d%d%d",&edgeNo,&edgeConn[i*2+0],&edgeConn[i*2+1],&edgeBoundary[i])){
      die("Can't parse edge input line!");
    }
    edgeConn[i*2+0]--;
    edgeConn[i*2+1]--;		
  }
  fclose(f);
  printf("Number of edges %d \n",nEdge);
  FEM_Register_entity(FEM_Mesh_default_write(),FEM_SPARSE,NULL,nEdge,nEdge,resize_edges);
  FEM_Register_array(FEM_Mesh_default_write(),FEM_SPARSE,FEM_CONN,edgeConn,FEM_INDEX_0,2);
  FEM_Register_array(FEM_Mesh_default_write(),FEM_SPARSE,FEM_BOUNDARY,edgeBoundary,FEM_INT,1);
  CkPrintf("Finished with init\n");
}

struct myGlobals {
  int nnodes, maxnodes;
  int nelems, maxelems;
  int nedges, maxedges;
  int *conn;             // Element connectivity table
  vector2d *coord;       // Undeformed coordinates of each node
  int *nodeBoundary;     // Node boundary
  int *nodeValid;        // Is the node valid?
  int *edgeConn;         // Edge connectivity table
  int *edgeBoundary;     // Edge boundary value
  int *elemValid;        // Is the node valid?
};

void pup_myGlobals(pup_er p,myGlobals *g) 
{
  FEM_Print("-------- called pup routine -------");
  pup_int(p,&g->nnodes);
  pup_int(p,&g->nelems);
  pup_int(p,&g->nedges);
  pup_int(p,&g->maxelems);
  pup_int(p,&g->maxnodes);
  pup_int(p,&g->maxedges);
  
  int nnodes=g->nnodes, nelems=g->nelems;
  if (pup_isUnpacking(p)) {
    g->coord=new vector2d[g->maxnodes];
    g->conn=new int[3*g->maxelems];
    g->nodeBoundary = new int[g->maxnodes];
    g->nodeValid = new int[g->maxnodes];
    g->edgeConn = new int[2*g->maxedges];
    g->edgeBoundary = new int[g->maxedges];
    g->elemValid = new int[g->maxelems];
  }
  pup_doubles(p,(double *)g->coord,2*nnodes);
  pup_ints(p,(int *)g->conn,3*nelems);
  pup_ints(p,(int *)g->nodeBoundary,nnodes);
  pup_ints(p,(int *)g->nodeValid,nnodes);
  pup_ints(p,(int *)g->edgeConn,2*g->nedges);
  pup_ints(p,(int *)g->edgeBoundary,g->nedges);
  pup_ints(p,(int *)g->elemValid,g->nelems);
	
  if (pup_isDeleting(p)) {
    delete[] g->coord;
    delete[] g->conn;
    delete[] g->nodeBoundary;
    delete[] g->nodeValid;
    delete[] g->edgeConn;
    delete[] g->edgeBoundary;
    delete[] g->elemValid;
  }
}

//Return the signed area of triangle i
double calcArea(myGlobals &g, int i)
{
  int n1=g.conn[3*i+0];
  int n2=g.conn[3*i+1];
  int n3=g.conn[3*i+2];
  vector2d a=g.coord[n1];
  vector2d b=g.coord[n2];
  vector2d c=g.coord[n3];
  c-=a; b-=a;
  double area=0.5*(b.x*c.y-c.x*b.y);
  return area;
}

void init_myGlobal(myGlobals *g){
  g->coord = NULL;
  g->conn = NULL;
  g->edgeConn = g->edgeBoundary = NULL;
}

void resize_nodes(void *data,int *len,int *max){
  printf("[%d] resize nodes called len %d max %d\n",FEM_My_partition(),*len,*max);
  FEM_Register_entity(FEM_Mesh_default_read(),FEM_NODE,data,*len,*max,resize_nodes);
  myGlobals *g = (myGlobals *)data;
  vector2d *coord=g->coord;
  int *nodeBoundary = g->nodeBoundary;
  int *nodeValid = g->nodeValid;
	
  g->coord=new vector2d[*max];
  g->coord[0].x = 0.9;
  g->coord[0].y = 0.8;
  g->maxnodes = *max;
  g->nodeBoundary = new int[g->maxnodes];
  g->nodeValid = new int[g->maxnodes];
	
  if(coord != NULL){
    for(int k=0;k<*len;k++){
      printf("before resize node %d ( %.6f %.6f ) \n", k, coord[k].x,
	     coord[k].y);
    }
  }	
	
  FEM_Register_array(FEM_Mesh_default_read(),FEM_NODE,FEM_DATA,(void *)g->coord,FEM_DOUBLE,2);
  FEM_Register_array(FEM_Mesh_default_read(),FEM_NODE,FEM_BOUNDARY,(void *)g->nodeBoundary,FEM_INT,1);
  FEM_Register_array(FEM_Mesh_default_read(),FEM_NODE,FEM_VALID,(void *)g->nodeValid,FEM_INT,1);
  
  for(int k=0;k<*len;k++) {
    printf("after resize node %d ( %.6f %.6f )\n", k, g->coord[k].x,
	   g->coord[k].y);
  }

  if(coord != NULL){
    delete [] coord;
    delete [] nodeBoundary;
    delete [] nodeValid;
  }
}


void resize_elems(void *data,int *len,int *max){
  printf("[%d] resize elems called len %d max %d\n",FEM_My_partition(),*len,*max);
  FEM_Register_entity(FEM_Mesh_default_read(),FEM_ELEM,data,*len,*max,resize_elems);
  myGlobals *g = (myGlobals *)data;
  int *conn=g->conn, *elemValid = g->elemValid;
  
  g->conn = new int[3*(*max)];
  g->maxelems = *max;
  g->elemValid = new int[g->maxelems];

  FEM_Register_array(FEM_Mesh_default_read(),FEM_ELEM,FEM_CONN,(void *)g->conn,FEM_INDEX_0,3);	
  CkPrintf("Connectivity array starts at %p \n",g->conn);
  FEM_Register_array(FEM_Mesh_default_read(),FEM_ELEM,FEM_VALID,(void *)g->elemValid,FEM_INT,1);	
  
  if(conn != NULL){
    delete [] conn;
    delete [] elemValid;
  }
};

void resize_edges(void *data,int *len,int *max){
  printf("[%d] resize edges called len %d max %d\n",FEM_My_partition(),*len,*max);
  FEM_Register_entity(FEM_Mesh_default_read(),FEM_SPARSE,data,*len,*max,resize_edges);
  myGlobals *g = (myGlobals *)data;
  int *conn = g->edgeConn;
  int *bound = g->edgeBoundary;
  g->maxedges = *max;	
  g->edgeConn = new int[2*(*max)];
  g->edgeBoundary = new int[(*max)];
  
  FEM_Register_array(FEM_Mesh_default_read(),FEM_SPARSE,FEM_CONN,(void *)g->edgeConn,FEM_INDEX_0,2);	
  FEM_Register_array(FEM_Mesh_default_read(),FEM_SPARSE,FEM_BOUNDARY,(void *)g->edgeBoundary,FEM_INT,1);
  if(conn != NULL){
    delete [] conn;
    delete [] bound;	
  }
}

extern "C" void
driver(void)
{
  myGlobals g;
  FEM_Register(&g,(FEM_PupFn)pup_myGlobals);
  init_myGlobal(&g);
  g.nnodes = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_NODE);
  int maxNodes = g.nnodes;
  g.maxnodes=2*maxNodes;
  resize_nodes((void *)&g,&g.nnodes,&maxNodes);

  g.nelems=FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_ELEM);
  g.maxelems=g.nelems;
  resize_elems((void *)&g,&g.nelems,&g.maxelems);
  g.nedges = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_SPARSE);
  g.maxedges = g.nedges;
  resize_edges((void *)&g,&g.nedges,&g.maxedges);
  
  FEM_Mesh *meshP = FEM_Mesh_lookup(FEM_Mesh_default_read(),"driver");
  meshP->createNodeNodeAdj();
  meshP->createElemNodeAdj();
  const static int tri2edge[]={0,1, 1,2, 2,0};
  FEM_Add_elem2face_tuples(FEM_Mesh_default_read(),0,2,3,tri2edge);
  FEM_Mesh_create_elem_elem_adjacency(FEM_Mesh_default_read());
  
  FEM_Adapt *adaptor = new FEM_Adapt(meshP, FEM_My_partition());

  // Test out adjacencies here

  // Test out basic modification operations here

  if (CkMyPe()==0)
    CkPrintf("Driver finished\n");
}
