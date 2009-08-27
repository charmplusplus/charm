#include "pgm.h"
#include "mpi.h"
#include "ckvector3d.h"
#include "charm-api.h"
#include "fem_mesh.h"
#include "fem_adapt_new.h"
#include "fem_mesh_modify.h"
#include <math.h>

extern void _registerFEMMeshModify(void);

extern "C" void
init(void)
{
  CkPrintf("init started\n");
  double startTime=CmiWallTimer();
  const char *eleName="mesh1.tri";
  const char *nodeName="mesh1.node";
  int nPts=0; //Number of nodes
  vector2d *pts=0; //Node coordinates
  int *bounds;

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
      if (4!=sscanf(line,"%d%lf%lf%d",&ptNo,&pts[i].x,&pts[i].y, &bounds[i])) 
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


  int fem_mesh=FEM_Mesh_default_write(); // Tell framework we are writing to the mesh

  CkPrintf("Passing node coords to framework\n");


  FEM_Mesh_data(fem_mesh,        // Add nodes to the current mesh
                FEM_NODE,        // We are registering nodes
                FEM_DATA+0,      // Register the point locations which are normally 
                                 // the first data elements for an FEM_NODE
                (double *)pts,   // The array of point locations
                0,               // 0 based indexing
                nPts,            // The number of points
                FEM_DOUBLE,      // Coordinates are doubles
                2);              // Points have dimension 2 (x,y)
 
  CkPrintf("Passing node bounds to framework\n");


  FEM_Mesh_data(fem_mesh,        // Add nodes to the current mesh
                FEM_NODE,        // We are registering nodes
                FEM_BOUNDARY,      // Register the point bound info 
                                 // the first data elements for an FEM_NODE
                (int *)bounds,	 // The array of point bound info
                0,               // 0 based indexing
                nPts,            // The number of points
                FEM_INT,	 // bounds are ints
                1);              // Points have dimension 1

  CkPrintf("Passing elements to framework\n");

  FEM_Mesh_data(fem_mesh,      // Add nodes to the current mesh
                FEM_ELEM+0,      // We are registering elements with type 0
                                 // The next type of element could be registered with FEM_ELEM+1
                FEM_CONN,        // Register the connectivity table for this
                                 // data elements for this type of FEM entity
                (int *)ele,      // The array of point locations
                0,               // 0 based indexing
                nEle,            // The number of elements
                FEM_INDEX_0,     // We use zero based node numbering
                3);              // Elements have degree 3, since triangles are defined 
                                 // by three nodes
 
  delete[] ele;
  delete[] pts;




  // Register values to the elements so we can keep track of them after partitioning
  int *values;
  values=new int[nEle];
  for(int i=0;i<nEle;i++)values[i]=i;

  // triangles
  FEM_Mesh_data(fem_mesh,      // Add nodes to the current mesh
                FEM_ELEM,      // We are registering elements with type 1
                FEM_DATA,   
                (int *)values,   // The array of point locations
                0,               // 0 based indexing
                nEle,            // The number of elements
                FEM_INT,         // We use zero based node numbering
                1);
  
  delete [] values;
  values=new int[nPts];
  for (int i=0; i<nPts; i++) values[i]=i;
  
  

  // triangles
  FEM_Mesh_data(fem_mesh,      // Add nodes to the current mesh
                FEM_NODE,      // We are registering elements with type 1
                FEM_DATA+1,   
                (int *)values,   // The array of point locations
                0,               // 0 based indexing
                nPts,            // The number of elements
                FEM_INT,         // We use zero based node numbering
                1);

  delete [] values; 

  //boundary conditions
  FEM_Mesh_data(fem_mesh,      // Add nodes to the current mesh
                FEM_NODE,      // We are registering elements with type 1
                FEM_BOUNDARY,   
                (int *)bounds,   // The array of point locations
                0,               // 0 based indexing
                nPts,            // The number of elements
                FEM_INT,         // We use zero based node numbering
                1);

  delete [] bounds;

  // add ghost layers

	const int triangleFaces[6] = {0,1,2};
	CkPrintf("Adding Ghost layers\n");
    FEM_Add_ghost_layer(1,1);
    FEM_Add_ghost_elem(0,3,triangleFaces);

	CkPrintf("Finished with init (Reading took %.f s)\n",CmiWallTimer()-startTime);

}


// A driver() function 
// driver() is required in all FEM programs
extern "C" void
driver(void)
{
  int *neighbors, *adjnodes, adjSz, *adjelems;
  int nnodes,nelems,nelems2,ignored;
  int i,t=0, myId=FEM_My_partition();
  myGlobals g;
  g.coord=NULL;
  g.conn=NULL;
  g.vCoord=NULL;
  g.vConn=NULL;
  
  FEM_Register(&g,(FEM_PupFn)pup_myGlobals);

  int mesh=FEM_Mesh_default_read(); // Tell framework we are reading data from the mesh
  FEM_Mesh *meshP = FEM_Mesh_lookup(mesh, "driver");
  _registerFEMMeshModify();

  printf("partition %d is in driver\n", myId);
  {
    int j, t=0;
    const int triangleFaces[6] = {0,1,1,2,2,0};
    FEM_Add_elem2face_tuples(mesh, 0, 2, 3, triangleFaces);
    FEM_Mesh_create_elem_elem_adjacency(mesh);
    FEM_Mesh_allocate_valid_attr(mesh, FEM_ELEM+0);
    FEM_Mesh_allocate_valid_attr(mesh, FEM_NODE);
    FEM_Mesh_create_node_elem_adjacency(mesh);
    FEM_Mesh_create_node_node_adjacency(mesh);
	
    FEM_Print_Mesh_Summary(mesh);
	
    int tuplesPerElem = 3;
    int elementNum = 0;
    int rank = 0;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm,&rank);
    MPI_Barrier(comm);
    FEM_REF_INIT(mesh);
    MPI_Barrier(comm);

//********************* Test mesh modification here **************************//
     
    double meanArea=0.0;
    FEM_Adapt *adaptor= meshP->getfmMM()->getfmAdaptL();
    doNetFEM(t, mesh, g);
    FEM_Adapt_Algs *adaptAlgs= meshP->getfmMM()->getfmAdaptAlgs();
    adaptAlgs->FEM_Adapt_Algs_Init(FEM_DATA+0, FEM_BOUNDARY);
    FEM_Interpolate *interp = meshP->getfmMM()->getfmInp();
    interp->FEM_SetInterpolateNodeEdgeFnPtr(interpolate);
/*    adaptor->edge_contraction(5,2);
    adaptor->edge_contraction(1,4);
    adaptor->edge_contraction(25,13);
    adaptor->edge_contraction(43,47);
    adaptor->edge_contraction(40,39);
    adaptor->edge_contraction(21,22);
    adaptor->edge_contraction(9,52);
    adaptor->edge_contraction(19,29);
    adaptor->edge_contraction(42,14);
    adaptor->edge_contraction(12,50);
    adaptor->edge_contraction(26,46);*/
    doNetFEM(t, mesh, g);
    CkPrintf("*************** CONTRACTION *************** \n");
    printQualityInfo(g, meanArea);

      
    int *nodes=NULL, *adjnodes, nNod;
    for (int c=0; c<3; c++) {  
      for (int i=0; i<g.nelems; i++) {
	if (FEM_is_valid(mesh, FEM_ELEM, i))
	  adaptAlgs->refine_element_leb(i);
      }
      doNetFEM(t, mesh, g);
      CkPrintf("*************** REFINEMENT *************** \n");
      printQualityInfo(g, meanArea);
/*      if (rank==0) {
	meshP->n2n_getAll(3, &adjnodes, &nNod);
for (int j=0; j<nNod; j++) CkPrintf("node[%d]: %d\n", 3,adjnodes[j]);

      }
      else if (rank==1) {
	meshP->n2n_getAll(17, &adjnodes, &nNod);
for (int j=0; j<nNod; j++) CkPrintf("node[%d]: %d\n", 17,adjnodes[j]);

      }

 //	  CkPrintf("%d: (%f, %f)\n", j, theCoord.x, theCoord.y);
*/	


      delete[] nodes;
      nodes = NULL;
//      for (int i=0; i<g.nnodes; i++) nodes[i]=i;
      for (int i=0; i<3; i++) { 
	adaptAlgs->FEM_mesh_smooth(meshP, NULL, g.nnodes, FEM_DATA+0);
	doNetFEM(t, mesh, g);
	CkPrintf("*************** SMOOTHING **************** \n");
       	printQualityInfo(g, meanArea);
      }
    }
    FEM_Print_Mesh_Summary(mesh);
  
    CkPrintf("Chunk %d Waiting for Synchronization\n",rank);
    MPI_Barrier(comm);
    CkPrintf("Synchronized\n");
    CkExit();
  }
  CkPrintf("Driver finished.\n");
}


// A PUP function to allow for migration and load balancing of mesh partitions.
// The PUP function is not needed if no migration or load balancing is desired.
void pup_myGlobals(pup_er p,myGlobals *g) 
{
  FEM_Print("-------- called pup routine -------");
  pup_int(p,&g->nnodes);
  pup_int(p,&g->nelems);
  int nnodes=g->nnodes, nelems=g->nelems;
  if (pup_isUnpacking(p)) {
    g->coord=new vector2d[nnodes];
    g->conn=new connRec[nelems];
    g->R_net=new vector2d[nnodes]; //Net force
    g->d=new vector2d[nnodes];//Node displacement
    g->v=new vector2d[nnodes];//Node velocity
    g->a=new vector2d[nnodes];
	g->S11=new double[nelems];
	g->S22=new double[nelems];
	g->S12=new double[nelems];
  }
  pup_doubles(p,(double *)g->coord,2*nnodes);
  pup_ints(p,(int *)g->conn,3*nelems);
  pup_doubles(p,(double *)g->R_net,2*nnodes);
  pup_doubles(p,(double *)g->d,2*nnodes);
  pup_doubles(p,(double *)g->v,2*nnodes);
  pup_doubles(p,(double *)g->a,2*nnodes);
  pup_doubles(p,(double *)g->S11,nelems);
  pup_doubles(p,(double *)g->S22,nelems);
  pup_doubles(p,(double *)g->S12,nelems);
  if (pup_isDeleting(p)) {
    delete[] g->coord;
    delete[] g->conn;
    delete[] g->R_net;
    delete[] g->d;
    delete[] g->v;
    delete[] g->a;
	delete[] g->S11;
	delete[] g->S22;
	delete[] g->S12;
  }
}

void doNetFEM(int& t, int mesh, myGlobals &g) {
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Barrier(comm);
  CkPrintf("Sending to netFem step %d.\n",t);
  rebuildArrays(mesh, g);
//  for (int i=0; i<g.nVnodes; i++)  CkPrintf("vNode[%d]: (%f, %f) \n", i, g.vCoord[i].x, g.vCoord[i].y);  
//  for (int i=0; i<g.nVelems; i++)  CkPrintf("vElem[%d]: (%d, %d, %d) \n", i, g.vConn[i][0], g.vConn[i][1], g.vConn[i][2]);
  NetFEM n=NetFEM_Begin(FEM_My_partition(),t,2,NetFEM_WRITE);
  NetFEM_Nodes(n,g.nnodes,(double *)g.coord,"Position (m)");
  NetFEM_Elements(n,g.nVelems,3,(int *)g.vConn,"Triangles");
  NetFEM_End(n);
  t++;
}

void rebuildArrays (int mesh, myGlobals &g) {
  CkPrintf("Rebuilding arrays. \n");
  delete [] g.conn;
  delete [] g.coord;
  delete [] g.vCoord;
  delete [] g.vConn;
  g.nelems=FEM_Mesh_get_length(mesh, FEM_ELEM);
  g.nnodes=FEM_Mesh_get_length(mesh, FEM_NODE);
  g.nVnodes = FEM_count_valid(mesh, FEM_NODE);
  g.nVelems = FEM_count_valid(mesh, FEM_ELEM);
  g.coord=new vector2d[g.nnodes];
  g.conn=new connRec[g.nelems];
  g.vConn = new connRec[g.nVelems];
  g.vCoord = new vector2d[g.nVnodes];

  FEM_Mesh_data(mesh, FEM_NODE, FEM_DATA, (double *)g.coord, 0, g.nnodes, FEM_DOUBLE, 2);

  int j=0;
  for (int i=0; i<g.nnodes;i++) {
    if (FEM_is_valid(mesh, FEM_NODE, i)) {
//      CkPrintf("g.coord[%d]: (%f, %f) \n", i, g.coord[i].x, g.coord[i].y);
      g.vCoord[j]=g.coord[i];
      j++;
    }
  }
  j=0;
  for (int i=0; i<g.nelems;i++) {
    FEM_Mesh_lookup(mesh, "driver")->e2n_getAll(i, (int *)g.conn[i]);
    if (FEM_is_valid(mesh, FEM_ELEM, i)) {
//      CkPrintf("g.conn[%d]: (%d, %d, %d) \n", i, g.conn[i][0], g.conn[i][1], g.conn[i][2]);
      for (int k=0; k<3; k++)
	g.vConn[j][k]=g.conn[i][k];
      j++;  
    }
  }
}

void FEM_mesh_smooth(int mesh, int *nodes, int nNodes, int attrNo)
{
  vector2d newPos, *coords, *ghostCoords,theCoord;
  int nNod, nGn, *boundVals, nodesInChunk;
  int neighbors[3], *adjnodes;
  int gIdxN;
  FEM_Mesh *meshP = FEM_Mesh_lookup(mesh, "driver");
  nodesInChunk = FEM_Mesh_get_length(mesh,FEM_NODE);
  boundVals = new int[nodesInChunk];
  nGn = FEM_Mesh_get_length(mesh, FEM_GHOST + FEM_NODE);
  coords = new vector2d[nodesInChunk+nGn];

  FEM_Mesh_data(mesh, FEM_NODE, FEM_BOUNDARY, (int*) boundVals, 0, nodesInChunk, FEM_INT, 1);    

  FEM_Mesh_data(mesh, FEM_NODE, attrNo, (double*)coords, 0, nodesInChunk, FEM_DOUBLE, 2);

  IDXL_Layout_t coord_layout = IDXL_Layout_create(IDXL_DOUBLE, 2);
  FEM_Update_ghost_field(coord_layout,-1, coords); 
  ghostCoords = &(coords[nodesInChunk]);
 // FEM_Mesh_data(FEM_Mesh_default_write(), FEM_GHOST+FEM_NODE, attrNo, (double*)ghostCoords, 0, nGn, FEM_DOUBLE, 2);
  for (int i=0; i<nNodes; i++)
  {
    newPos.x=0;
    newPos.y=0;
    CkAssert(nodes[i]<nodesInChunk);  
    if (FEM_is_valid(mesh, FEM_NODE, i) && boundVals[i]>-1) //node must be internal
    {
      meshP->n2n_getAll(i, &adjnodes, &nNod);
     // for (int j=0; j<nNod; j++) CkPrintf("node[%d]: %d\n", i,adjnodes[j]);
      
      for (int j=0; j<nNod; j++) { //for all adjacent nodes, find coords
	if (adjnodes[j]<-1) {
	  gIdxN = FEM_From_ghost_index(adjnodes[j]);
	  newPos.x += theCoord.x=ghostCoords[gIdxN].x;
	  newPos.y += theCoord.y=ghostCoords[gIdxN].y;
	}
	else {
	  newPos.x += theCoord.x=coords[adjnodes[j]].x;
	  newPos.y += theCoord.y=coords[adjnodes[j]].y;
	}     
	int rank=FEM_My_partition();
	if (rank==0 && i==3 || rank==1 && i==17) {
	  CkPrintf("node[%d]: %d\n", i,adjnodes[j]);
	  CkPrintf("%d: (%f, %f)\n", j, theCoord.x, theCoord.y);
	}
      }
      newPos.x/=nNod;
      newPos.y/=nNod;
      FEM_set_entity_coord2(mesh, FEM_NODE, nodes[i], newPos.x, newPos.y);
      delete [] adjnodes;
    }
  }
 // FEM_Update_field(coord_layout, coords);
 // FEM_Mesh_data(FEM_Mesh_default_write(), FEM_NODE, attrNo, (double*)coords, 0, nodesInChunk, FEM_DOUBLE, 2);

  if (coords) delete [] coords;
  delete [] boundVals;
}

void interpolate(FEM_Interpolate::NodalArgs args, FEM_Mesh *meshP)
{
//  CkPrintf("INTERPOLATOR!!!!!!!!!!!\n");
  int *boundVals= new int[meshP->node.realsize()];
  FEM_Mesh_dataP(meshP, FEM_NODE, FEM_BOUNDARY, (int*) boundVals, 0, meshP->node.realsize() , FEM_INT, 1);   
  CkVec<FEM_Attribute *>*attrs = (meshP->node).getAttrVec();
  for (int i=0; i<attrs->size(); i++) {
    FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
    if (a->getAttr() < FEM_ATTRIB_TAG_MAX || a->getAttr()==FEM_BOUNDARY) {
      if (a->getAttr()==FEM_BOUNDARY) {
	if (boundVals[args.nodes[1]]<0)
	  a->copyEntity(args.n, *a, args.nodes[0]);
	else
	  a->copyEntity(args.n, *a, args.nodes[1]);
      }
      else {
	FEM_DataAttribute *d = (FEM_DataAttribute *)a;
	d->interpolate(args.nodes[0], args.nodes[1], args.n, args.frac);
      }
    }
  }
}

void printQualityInfo (myGlobals &g, double &meanArea) {
  double len[3], s, area[g.nVelems], angles[g.nVelems], qArr[g.nVelems], minlen=0.0, A, B, C;
  double meanQ=0.0, meanA=0.0, f;
  int smallest=0;
  int idx=0;
  meanArea=0.0;
  for (int i=0; i<g.nelems; i++) {
  if (FEM_is_valid(FEM_Mesh_default_read(), FEM_ELEM, i)) {
    for (int j=0; j<3; j++) {
      len[j] = getDistance(g.coord[g.conn[i][j]],g.coord[g.conn[i][(j+1)%3]]);
    }  
    s = (len[0] + len[1] + len[2])/2.0;
    area[idx] = sqrt(s * (s - len[0]) * (s - len[1]) * (s - len[2]));
    f = 4.0*sqrt(3.0); //proportionality constant
    qArr[idx] = (f*area[idx])/(len[0]*len[0]+len[1]*len[1]+len[2]*len[2]);
//    CkPrintf("area[%d]: %e || ", i, area[idx]);
//    CkPrintf("g.conn: (%d, %d, %d) || ", i, g.conn[i][0], g.conn[i][1], g.conn[i][2]);
   // CkPrintf("g.coord[%d]: {(%.3f, %.3f),(%.3f, %.3f), (%.3f, %.3f)}\n", i, g.coord[g.conn[i][0]].x,g.coord[g.conn[i][0]].y, g.coord[g.conn[i][1]].x,g.coord[g.conn[i][1]].y, g.coord[g.conn[i][2]].x, g.coord[g.conn[i][2]].y);
    //CkPrintf("len: (%.4f, %.4f, %.4f) \n", len[0], len[1], len[2]);


    minlen=999.0;  
    for (int j=0; j<3; j++) { // find min length of a side
      if (len[j] < minlen) {
	smallest = j;
	minlen = len[j];
      }
    }
    C = len[smallest];
    A = len[(smallest+1)%3];
    B = len[(smallest+2)%3];
    angles[idx] = acos((C*C - A*A - B*B)/(-2*A*B));
    angles[idx] *= (180.0/pi);
    meanQ+=qArr[idx];
    meanA+=angles[idx];
    meanArea+=area[idx];
    idx++;
  }
  }
 
  insertion_sort(qArr, g.nVelems);
  insertion_sort(angles, g.nVelems);
  insertion_sort(area, g.nVelems);

  meanQ/=g.nVelems;
  meanA/=g.nVelems;
  meanArea/=g.nVelems;
  CkPrintf("------------ Quality Summary ---------------\n");
  CkPrintf(" Angle stats:\n");
  //for (int i=0; i<g.nelems; i++) CkPrintf(" angles[%d]: %f \n", i, angles[i]);
  CkPrintf("  min   :	%f\n", angles[0]);
  CkPrintf("  median:	%f\n", angles[g.nVelems/2]);
  CkPrintf("  mean  :	%f\n", meanA);
  CkPrintf("  max   :	%f\n\n", angles[g.nVelems-1]);
  
  CkPrintf(" Quality stats:\n");
  //for (int i=0; i<g.nelems; i++) CkPrintf(" qArr[%d]: %f \n", i, qArr[i]);
  CkPrintf("  min   :	%f\n", qArr[0]);
  CkPrintf("  median:	%f\n", qArr[g.nVelems/2]);
  CkPrintf("  mean  :	%f\n", meanQ);
  CkPrintf("  max   :	%f\n\n", qArr[g.nVelems-1]);

  CkPrintf(" Area stats:\n");
 // for (int i=0; i<g.nVelems; i++) CkPrintf(" Area[%d]: %f (%d)\n", i, area[i], FEM_is_valid(FEM_Mesh_default_read(), FEM_ELEM, i));
  CkPrintf("  min   :	%.5e\n", area[0]);
  CkPrintf("  median:	%.5e\n", area[g.nVelems/2]);
  CkPrintf("  mean  :	%.5e\n", meanArea);
  CkPrintf("  max   :	%.5e\n\n", area[g.nVelems-1]);

  
}

double getDistance (vector2d n1, vector2d n2) {
  return sqrt((n1.x-n2.x)*(n1.x-n2.x)+(n1.y-n2.y)*(n1.y-n2.y));
}

void insertion_sort(double *numbers, int array_size)
{
  int i, j;
  double index;

  for (i=1; i < array_size; i++)
  {
    index = numbers[i];
    j = i;
    while ((j > 0) && (numbers[j-1] > index))
    {
      numbers[j] = numbers[j-1];
      j = j - 1;
    }
    numbers[j] = index;
  }
}
