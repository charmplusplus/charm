#include "pgm.h"
#include "mpi.h"
#include "ckvector3d.h"
#include "charm-api.h"
#include "fem_mesh.h"
#include "fem_adapt_new.h"
#include "fem_mesh_modify.h"


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
                FEM_DATA+2,      // Register the point bound info 
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
                FEM_DATA+2,   
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
  FEM_Register(&g,(FEM_PupFn)pup_myGlobals);
  int mesh=FEM_Mesh_default_read(); // Tell framework we are reading data from the mesh
  FEM_Mesh *meshP = FEM_Mesh_lookup(mesh, "driver");
  _registerFEMMeshModify();

  printf("partition %d is in driver\n", myId);

   
  // Get node data
  nnodes=FEM_Mesh_get_length(mesh,FEM_NODE); // Get number of nodes
  g.nnodes=nnodes;
  g.coord=new vector2d[nnodes];
  // Get node positions
  FEM_Mesh_data(mesh, FEM_NODE, FEM_DATA+0, (double*)g.coord, 0, nnodes, FEM_DOUBLE, 2);  


  // Get element data
  nelems=FEM_Mesh_get_length(mesh,FEM_ELEM+0); // Get number of elements
  g.nelems=nelems;
  g.conn=new connRec[nelems];
  // Get connectivity for elements
  FEM_Mesh_data(mesh, FEM_ELEM+0, FEM_CONN, (int *)g.conn, 0, nelems, FEM_INDEX_0, 3);  


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
	

	int *values, *nvalues;
	double *valuesDouble, *nvaluesDouble;
	values = new int[g.nelems];
	nvalues = new int[g.nnodes];
	valuesDouble = new double[g.nelems];
	nvaluesDouble = new double[g.nnodes];

	FEM_Mesh_data(mesh, FEM_ELEM+elementNum, FEM_DATA, values, 0, g.nelems, FEM_INT, 1);
	FEM_Mesh_data(mesh, FEM_NODE, FEM_DATA+1, nvalues, 0, g.nnodes, FEM_INT, 1);

	for (int i=0; i<g.nelems; i++) valuesDouble[i]=values[i];
	for (int i=0; i<g.nnodes; i++) nvaluesDouble[i]=nvalues[i];

	// ghost data
	int ng=FEM_Mesh_get_length(mesh,FEM_GHOST+FEM_ELEM+elementNum); 
	int *valuesg, *nvaluesg;
	valuesg=new int[ng];
	int ngn = FEM_Mesh_get_length(mesh,FEM_GHOST+FEM_NODE);
	nvaluesg = new int[ngn];

	FEM_Mesh_data(mesh, FEM_GHOST+FEM_ELEM+elementNum, FEM_DATA, valuesg, 0, ng, FEM_INT, 1);
	FEM_Mesh_data(mesh, FEM_GHOST+FEM_NODE, FEM_DATA+1, nvaluesg, 0, ngn, FEM_INT, 1);

	FEM_Print_Mesh_Summary(mesh);

	int rank = 0;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm,&rank);
	
	MPI_Barrier(comm);

	FEM_REF_INIT(mesh);



/*// Remaps the nodes to correspond the original node numberings
  vector2d *coords;
  connRec *conns;
  conns=new connRec[g.nelems];
  coords=new vector2d[g.nnodes];
  for (int i=0; i<g.nnodes; i++) {
    CkPrintf("oldCoords[%d]: (%f, %f)\n", i, g.coord[i].x, g.coord[i].y);
    coords[nvalues[i]]=g.coord[i];  
    CkPrintf("node %d has val %d\n", i, nvalues[i]);
  }
  for (int i=0; i<g.nelems;i++) {
    for (int j=0; j<3; j++) {
      conns[i][j]=nvalues[g.conn[i][j]];
    }    
  }


*/
	// Test out adjacencies here
	// Print them pretty like
/*
	CkPrintf("\n  *** TESTING E2E ADJACENCIES *** \n\n");
	neighbors=new int[3];
	for (int i=0; i<g.nelems; i++) {
	  meshP->e2e_getAll(i, neighbors);
	  CkPrintf(" %d:  e2e : ", myId);
	  if (values[i]<10) CkPrintf(" ");
	  CkPrintf("[%d] => { ", values[i]);
	  for (j=0; j<3; j++) {
	    if (neighbors[j]<-1) 
	    {
	      if (valuesg[FEM_From_ghost_index(neighbors[j])]<10) CkPrintf(" ");
	      CkPrintf("(%d)*",valuesg[FEM_From_ghost_index(neighbors[j])]);
	    }
	    else if (neighbors[j]>-1) {
	      if (values[neighbors[j]]<10) CkPrintf(" ");
	      CkPrintf("(%d) ", values[neighbors[j]]);
	    }
	    else {
	      CkPrintf("(%d) ", neighbors[j]);
	    }
	    if (j<2) CkPrintf(", ");
	  }
	  CkPrintf(" }\n");
	}

	delete [] neighbors;    

	CkPrintf("\n  *** TESTING E2N ADJACENCIES *** \n\n");
	neighbors=new int[3];
	for (int i=0; i<g.nelems; i++) {
	  meshP->e2n_getAll(i, neighbors);
	  CkPrintf(" %d:  e2n : ", myId);
	  if (values[i]<10) CkPrintf(" ");
	  CkPrintf("[%d] => { ", values[i]);
	  for (j=0; j<3; j++) {
	    if (neighbors[j]<-1) 
	    {
	      if (nvaluesg[FEM_From_ghost_index(neighbors[j])]<10) CkPrintf(" ");
	      CkPrintf("(%d)*",nvaluesg[FEM_From_ghost_index(neighbors[j])]);
	    }
	    else if (neighbors[j]>-1) {
	      if (nvalues[neighbors[j]]<10) CkPrintf(" ");
	      CkPrintf("(%d) ", nvalues[neighbors[j]]);//coords[i].x, g.coord[i].y
	    }
	    else {
	      CkPrintf("(%d) ", neighbors[j]);
	    }
	    if (j<2) CkPrintf(", ");
	  }
	  CkPrintf(" }\n");
	}

      
	CkPrintf("\n  *** TESTING N2N ADJACENCIES *** \n\n");
	for (int i=0; i<g.nnodes; i++) {
	  meshP->n2n_getAll(i, &adjnodes, &adjSz);
	  CkPrintf(" %d:  n2n : ", myId);
	  if (nvalues[i]<10) CkPrintf(" ");
	  CkPrintf("[%d] => { ", nvalues[i]);
	  for (int j=0; j<adjSz; j++){
	    if (adjnodes[j]<-1) 
	    {
	      if (nvaluesg[FEM_From_ghost_index(adjnodes[j])]<10) CkPrintf(" ");
	      CkPrintf("(%d)*",nvaluesg[FEM_From_ghost_index(adjnodes[j])]);
	    }
	    else if (adjnodes[j]>-1) {
	      if (nvalues[adjnodes[j]]<10) CkPrintf(" ");
	      CkPrintf("(%d) ", nvalues[adjnodes[j]]);
	    }
	    else {
	      CkPrintf("(%d) ", adjnodes[j]);
	    }
	    if (j<adjSz-1) CkPrintf(", ");
	  }
	  CkPrintf(" }\n");
	}	
      
	delete [] adjnodes;

	CkPrintf("\n  *** TESTING N2E ADJACENCIES *** \n\n");
	for (int i=0; i<g.nnodes; i++) {
	  meshP->n2e_getAll(i, &adjelems, &adjSz);
	  CkPrintf(" %d:  n2e : ", myId);
	  if (nvalues[i]<10) CkPrintf(" ");
	  CkPrintf("[%d] => { ", nvalues[i]);
	  for (int j=0; j<adjSz; j++){
	    if (adjelems[j]<-1) 
	    {
	      if (valuesg[FEM_From_ghost_index(adjelems[j])]<10) CkPrintf(" ");
	      CkPrintf("(%d)*",valuesg[FEM_From_ghost_index(adjelems[j])]);
	    }
	    else if (adjnodes[j]>-1) {
	      if (values[adjelems[j]]<10) CkPrintf(" ");
	      CkPrintf("(%d) ", values[adjelems[j]]);
	    }
	    else {
	      CkPrintf("(%d) ", adjelems[j]);
	    }
	    if (j<adjSz-1) CkPrintf(", ");
	  }
	  CkPrintf(" }\n");
	}	
*/
//      doNetFEM(t, mesh, g);


//********************* Test mesh modification here **************************//
      FEM_Adapt *adaptor= meshP->getfmMM()->getfmAdapt();
      doNetFEM(t, mesh, g);

/*   // EDGE FLIP TESTING
      int flip[2];
      CkPrintf("Begin edge flip testifications. \n");
      
      if (myId==0) 
      {
	flip[0]=1;
	flip[1]=2;
      }	
      else if (myId==1) 
      {
	flip[0]=11;
	flip[1]=12;
      }	
      else if (myId==2) 
      {
	flip[0]=12;
	flip[1]=3;
      }	
      else //if (myId==3) 
      {
	flip[0]=1;
	flip[1]=2;
      }	

      CkPrintf("%d:Running edge_flip (%d, %d)\n",myId, flip[0],flip[1]);
      adaptor->edge_flip(flip[0],flip[1]);
      doNetFEM(t, mesh, g);
*/

      int bisect[2];

      CkPrintf("Begin edge bisect. \n");
      if (myId==0) 
      {
	bisect[0]=2;
	bisect[1]=3;
      }	
      else if (myId==1) 
      {
	bisect[0]=1;
	bisect[1]=4;
      }	
      else if (myId==2) 
      {
	bisect[0]=15;
	bisect[1]=17;
      }	
      else //if (myId==3) 
      {
	bisect[0]=1;
	bisect[1]=2;
      }	
      int newNode=0;
      //CkPrintf("%d:Running edge_bisect (%d, %d)\n",myId, bisect[0],bisect[1]);
      //if (rank==0) newNode=adaptor->edge_bisect(bisect[0],bisect[1]);
      adaptor->edge_bisect(4,22);
      adaptor->edge_bisect(13,24);
      adaptor->edge_bisect(50,47);
      adaptor->edge_bisect(37,39);
      adaptor->edge_bisect(27,28);
      adaptor->edge_bisect(6,32);
      adaptor->edge_bisect(44,52);

      doNetFEM(t, mesh, g);

      FEM_mesh_smooth(mesh, g);

      doNetFEM(t, mesh, g);

      FEM_mesh_smooth(mesh, g);


/*
      int vRemove[2];
  
      CkPrintf("Begin vertex remove. \n");
      if (myId==0) 
      {
	vRemove[0]=newNode;
	vRemove[1]=13;
      }	
      else if (myId==1) 
      {
	vRemove[0]=newNode;
	vRemove[1]=1;
      }	
      else if (myId==2) 
      {
	vRemove[0]=newNode;
	vRemove[1]=15;
      }	
      else //if (myId==3) 
      {
	vRemove[0]=newNode;
	vRemove[1]=2;
      }	

      CkPrintf("%d:Running vertex_remove (%d, %d)\n",myId, vRemove[0],vRemove[1]);
      if (rank==0) adaptor->vertex_remove(vRemove[0],vRemove[1]);*/
      doNetFEM(t, mesh, g);

  
      CkPrintf("Chunk %d Waiting for Synchronization\n",rank);
      MPI_Barrier(comm);
      CkPrintf("Synchronized\n");
      doNetFEM(t, mesh, g);
      FEM_Print_Mesh_Summary(mesh);
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
  NetFEM n=NetFEM_Begin(FEM_My_partition(),t,2,NetFEM_WRITE);
  NetFEM_Nodes(n,g.nVnodes,(double *)g.vCoord,"Position (m)");
  NetFEM_Elements(n,g.nVelems,3,(int *)g.vConn,"Triangles");
  NetFEM_End(n);
  t++;
}

void rebuildArrays (int mesh, myGlobals &g) {
  CkPrintf("Rebuilding arrays. \n");
  delete [] g.conn;
  delete [] g.coord;
  if (!g.vCoord) delete [] g.vCoord;
  if (!g.vConn) delete [] g.vConn;
  g.nelems=FEM_Mesh_get_length(mesh, FEM_ELEM);
  g.nnodes=FEM_Mesh_get_length(mesh, FEM_NODE);
  g.nVnodes = FEM_count_valid(mesh, FEM_NODE);
  g.nVelems = FEM_count_valid(mesh, FEM_ELEM);
  g.coord=new vector2d[g.nnodes];
  g.conn=new connRec[g.nelems];
  g.vConn = new connRec[g.nVelems];
  g.vCoord = new vector2d[g.nVnodes];

  FEM_Mesh_data(mesh, FEM_NODE, FEM_DATA, (double *)g.coord, 0, g.nnodes, FEM_DOUBLE, 2);
  FEM_Mesh_data(mesh, FEM_ELEM, FEM_CONN, (int *)g.conn, 0, g.nelems, FEM_INDEX_0, 3);

  int j=0;
  for (int i=0; i<g.nnodes;i++)
    if (FEM_is_valid(mesh, FEM_NODE, i))
      g.vCoord[j++]=g.coord[i];
  
  j=0;
  for (int i=0; i<g.nelems;i++)
    if (FEM_is_valid(mesh, FEM_ELEM, i)) {
      for (int k=0; k<3; k++)
	g.vConn[j][k]=g.conn[i][k];
      j++;  
    }
}

void FEM_mesh_smooth(int mesh, myGlobals &g)
{
  double *areas;
  vector2d *centroids, sum;
  int nLocEle;
  for (int i=0; i<g.nnodes; i++)
  {
    sum.x=0;
    sum.y=0;
    if (isNodeInternal(i))
    {
      getData(mesh, i, areas, centroids, nLocEle, g);      
      for (int j=0; j<nLocEle; j++) {
	sum.x += centroids[j].x;
	sum.y += centroids[j].y;
      }
      sum.x/=nLocEle;
      sum.y/=nLocEle;
      FEM_set_entity_coord2(mesh, FEM_NODE, i, sum.x, sum.y);
      CkPrintf("Sum vector for node %d: (%f,%f)\n", i, sum.x, sum.y);
    }
  }
}

int isNodeInternal(int idx)
{
  int boundVal=0;
  FEM_Mesh_data(FEM_Mesh_default_read(), FEM_NODE,FEM_DATA+2, &boundVal,idx, 1, FEM_INT, 1);
  return (boundVal>-1);
}


void getData(int mesh, int idx, double *areas, vector2d*& centroids, int& nEle, myGlobals &g)
{
  double x1, x2, x3, y1, y2, y3;
  if (!areas) delete [] areas;
  if (!centroids) delete [] centroids;
  int *adjelems;
  FEM_Mesh *meshP = FEM_Mesh_lookup(mesh, "driver");
  meshP->n2e_getAll(idx, &adjelems, &nEle);
  areas = new double[nEle];
  centroids = new vector2d[nEle];
  for (int i=0; i<nEle; i++)
  {
    x1 = g.coord[g.conn[adjelems[i]][0]].x;
    x2 = g.coord[g.conn[adjelems[i]][1]].x;
    x3 = g.coord[g.conn[adjelems[i]][2]].x;

    y1 = g.coord[g.conn[adjelems[i]][0]].y;
    y2 = g.coord[g.conn[adjelems[i]][1]].y;
    y3 = g.coord[g.conn[adjelems[i]][2]].y;

    centroids[i].x=(x1+x2+x3)/3.0;
    centroids[i].y=(y1+y2+y3)/3.0;
    areas[i]= (0.5)*(x1*(y2-y3)-y1*(x2-x3)+x2*y3-x3*y2);
  }
}

