#include "pgm.h"
#include "mpi.h"
#include "ParFUM_SA.h"

extern "C" void
init(void)
{
  CkPrintf("init started\n");
  double startTime=CmiWallTimer();
  const char *eleName="../2Dexample/meshes/mesh750.tri";
  const char *nodeName="../2Dexample/meshes/mesh750.node";
  int nPts=0; //Number of nodes
  vector2d *pts=0; //Node coordinates
  int *bounds=0; //boundary conditions

  CkPrintf("Reading node coordinates from %s\n",nodeName);
  //Open and read the node coordinate file
  {
    char line[1024];
    FILE *f=fopen(nodeName,"r");
    if (f==NULL) die("Can't open node file!");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nPts)) die("Can't read number of points!");
    pts=new vector2d[nPts];
    bounds = new int[nPts];
    for (int i=0;i<nPts;i++) {
      int ptNo;
      if (NULL==fgets(line,1024,f)) die("Can't read node input line!");
      if (4!=sscanf(line,"%d%lf%lf%d",&ptNo,&pts[i].x,&pts[i].y,&bounds[i])) 
	die("Can't parse node input line!");
    }
    fclose(f);
  }
 
  int nEle=0;
  connRec *ele=NULL;
  CkPrintf("Reading elements from %s\n",eleName);
  //Open and read the element connectivity file
  {
    char line[1024];
    FILE *f=fopen(eleName,"r");
    if (f==NULL) die("Can't open element file!");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nEle)) die("Can't read number of elements!");
    ele=new connRec[nEle];
    for (int i=0;i<nEle;i++) {
      int elNo;
      if (NULL==fgets(line,1024,f)) die("Can't read element input line!");
      if (4!=sscanf(line,"%d%d%d%d",&elNo,&ele[i][0],&ele[i][1],&ele[i][2])) 
	die("Can't parse element input line!");  
      ele[i][0]--; //Fortran to C indexing
      ele[i][1]--; //Fortran to C indexing
      ele[i][2]--; //Fortran to C indexing
    }
    fclose(f);
  }


  int fem_mesh=FEM_Mesh_default_write();

  int *ndata1 = new int[2*nPts];
  double *ndata2 = new double[3*nPts];

  CkPrintf("Passing nodes to framework\n");
  FEM_Register_entity(fem_mesh,FEM_NODE, NULL, nPts, nPts, resize_nodes);
  FEM_Register_array(fem_mesh,FEM_NODE,FEM_DATA+0,(double *)pts,FEM_DOUBLE,2);
  FEM_Register_array(fem_mesh,FEM_NODE,FEM_BOUNDARY,(int *)bounds,FEM_INT,1);
  FEM_Register_array(fem_mesh,FEM_NODE,FEM_DATA+1,(int*)ndata1,FEM_INT,2);
  FEM_Register_array(fem_mesh,FEM_NODE,FEM_DATA+2,(double*)ndata2,FEM_DOUBLE,3);

  CkPrintf("Passing elements to framework\n");
  FEM_Register_entity(fem_mesh,FEM_ELEM+0, NULL, nEle, nEle, resize_elems);
  FEM_Register_array(fem_mesh,FEM_ELEM+0,FEM_CONN,(int *)ele,FEM_INDEX_0,3);

  // add ghost layers
  const int triangleFaces[6] = {0,1,2};
  CkPrintf("Adding Ghost layers\n");
  FEM_Add_ghost_layer(1,1);
  FEM_Add_ghost_elem(0,3,triangleFaces);

  CkPrintf("Finished with init (Reading took %.f s)\n",CmiWallTimer()-startTime);

}

void init_myGlobal(myGlobals *g) {
  g->nnodes = 0;
  g->nelems = 0;
  g->nedges = 0;
  g->maxnnodes = 0;
  g->maxnelems = 0;
  g->maxnedges = 0;
  g->coord = NULL;
  g->conn = NULL;
  g->bounds = NULL;
  g->ndata1 = NULL;
  g->ndata2 = NULL;
}

void resize_nodes(void *data, int *len, int *max) {
  int fem_mesh = FEM_Mesh_default_read();
  FEM_Register_entity(fem_mesh,FEM_NODE,data,*len,*max,resize_nodes);

  myGlobals *g = (myGlobals *)data;
  vector2d *coord = g->coord;
  int *bounds = g->bounds;
  int *ndata1 = g->ndata1;
  double *ndata2 = g->ndata2;

  g->coord = new vector2d[*max];
  g->bounds = new int[*max];
  g->ndata1 = new int[2*(*max)];
  g->ndata2 = new double[3*(*max)];
  g->nnodes = *len;
  g->maxnnodes = *max;

  FEM_Register_array(fem_mesh,FEM_NODE,FEM_DATA+0,(void*)g->coord,FEM_DOUBLE,2);
  FEM_Register_array(fem_mesh,FEM_NODE,FEM_BOUNDARY,(void*)g->bounds,FEM_INT,1);
  FEM_Register_array(fem_mesh,FEM_NODE,FEM_DATA+1,(void*)g->ndata1,FEM_INT,2);
  FEM_Register_array(fem_mesh,FEM_NODE,FEM_DATA+2,(void*)g->ndata2,FEM_DOUBLE,3);

//   delete coord;
//   delete bounds;
//   delete ndata1;
//   delete ndata2;
}

void resize_elems(void *data, int *len, int *max) {
  int fem_mesh = FEM_Mesh_default_read();
  FEM_Register_entity(fem_mesh,FEM_ELEM,data,*len,*max,resize_elems);

  myGlobals *g = (myGlobals *)data;
  connRec *conn = g->conn;

  g->conn = new connRec[*max];
  g->nelems = *len;
  g->maxnelems = *max;

  FEM_Register_array(fem_mesh,FEM_ELEM+0,FEM_CONN,(void*)g->conn,FEM_INDEX_0,3);

//   if(conn!=NULL) delete conn;
}

void repeat_after_split(void *data) {
  myGlobals *g = (myGlobals *)data;
  g->nnodes = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_NODE);
  g->nelems = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_ELEM);
  //g->nelems = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_SPARSE);
}

void migrateM(myGlobals *g, int mesh) {
  CkPrintf("Starting load balance\n");
  FEM_Migrate();
  resize_nodes((void*)g,&g->nnodes,&g->maxnnodes);
  resize_elems((void*)g,&g->nelems,&g->maxnelems);
  CkPrintf("Done load balance!\n");
  return;
}

// A driver() function 
// driver() is required in all FEM programs
extern "C" void
driver(void)
{
  int nnodes,nelems,nedges,ignored;
  int i, myId=FEM_My_partition();
  myGlobals g;
  printf("partition %d is in driver\n", myId);
  FEM_Register(&g,(FEM_PupFn)pup_myGlobals);
  init_myGlobal(&g);
  int mesh=FEM_Mesh_default_read();
  // Get node data
  nnodes=FEM_Mesh_get_length(mesh,FEM_NODE); // Get number of nodes
  g.nnodes=nnodes;
  resize_nodes((void*)&g,&g.nnodes,&g.nnodes);
  // Get element data
  nelems=FEM_Mesh_get_length(mesh,FEM_ELEM+0); // Get number of elements
  g.nelems=nelems;
  resize_elems((void*)&g,&g.nelems,&g.nelems);

  int netIndex = 0;
  int rank = 0;
  MPI_Comm comm=MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&rank);
  MPI_Barrier(comm);
  ParFUM_SA_Init(mesh);

  FEM_Mesh *meshP = FEM_Mesh_lookup(FEM_Mesh_default_read(),"driver");
  MPI_Barrier(comm);

  //test load-balancing
#ifdef LOAD_BALANCE
  MPI_Barrier(comm);
  migrateM(&g,mesh);
  MPI_Barrier(comm);
#endif

  double targetArea = 0.00000049;
  double startTime = CmiWallTimer();

  //do some IDXL operations
  int *chklist  = (int*)malloc(2*sizeof(int));
  chklist[0] = 1; chklist[1] = 0;
  if(myId==1) {
    meshP->parfumSA->IdxlLockChunks(chklist, 2, 0);
    meshP->parfumSA->IdxlAddPrimary(500,0,0);
    meshP->parfumSA->IdxlUnlockChunks(chklist, 2, 0);
    meshP->parfumSA->IdxlLockChunks(chklist, 2, 0);
    meshP->parfumSA->IdxlRemovePrimary(500,0,0);
    meshP->parfumSA->IdxlUnlockChunks(chklist, 2, 0);
  }

  CkPrintf("chunk %d Waiting for Synchronization\n",rank);
  MPI_Barrier(comm);
  CkPrintf("Synchronized\n");
#ifdef SUMMARY_ON
  FEM_Print_Mesh_Summary(mesh);
#endif
  if(rank==0) {
    CkPrintf("Total time taken: %f",CmiWallTimer()-startTime);
  }
  CkExit();
}


// A PUP function to allow for migration and load balancing of mesh partitions.
// The PUP function is not needed if no migration or load balancing is desired.
void pup_myGlobals(pup_er p,myGlobals *g) 
{
  FEM_Print("----called pup globals----");
  pup_int(p,&g->nnodes);
  pup_int(p,&g->nelems);
  pup_int(p,&g->maxnnodes);
  pup_int(p,&g->maxnelems);
}
