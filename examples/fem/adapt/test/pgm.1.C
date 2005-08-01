#include "pgm.h"
#include "mpi.h"
#include "ckvector3d.h"
#include "charm-api.h"
#include "fem_mesh.h"
#include "fem_adapt_new.h"


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

  CkPrintf("Reading node coordinates from %s\n",nodeName);
  //Open and read the node coordinate file
  {
    char line[1024];
    FILE *f=fopen(nodeName,"r");
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
  int i, myId=FEM_My_partition();
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
 	int j;
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

/*  vector2d *coords;
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
  }*/
  if(0) {
	int t=0;
	NetFEM n=NetFEM_Begin(FEM_My_partition(),t,2,NetFEM_WRITE);
	NetFEM_Nodes(n,nnodes,(double *)g.coord,"Position (m)");
//	NetFEM_Scalar(n,nvaluesDouble, 1, "nodeNums");
	NetFEM_Elements(n,nelems,3,(int *)g.conn,"Triangles");
	NetFEM_Scalar(n,valuesDouble,1,"elemNums");
	NetFEM_End(n);
  }



	// Test out adjacencies here
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
	      CkPrintf("(%d)*",neighbors[j]);
	    }
	    else if (neighbors[j]>-1) {
	      if (neighbors[j]<10) CkPrintf(" ");
	      CkPrintf("(%d) ", neighbors[j]);//coords[i].x, g.coord[i].y
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
	  if (i<10) CkPrintf(" ");
	  CkPrintf("[%d] => { ", i);
	  for (int j=0; j<adjSz; j++){
	    if (adjnodes[j]<-1) 
	    {
	      if (nvaluesg[FEM_From_ghost_index(adjnodes[j])]<10) CkPrintf(" ");
	      CkPrintf("(%d)*",adjnodes[j]);
	    }
	    else if (adjnodes[j]>-1) {
	      if (adjnodes[j]<10) CkPrintf(" ");
	      CkPrintf("(%d) ", adjnodes[j]);
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
	  if (i<10) CkPrintf(" ");
	  CkPrintf("[%d] => { ", i);
	  for (int j=0; j<adjSz; j++){
	    if (adjelems[j]<-1) 
	    {
	      if (adjelems[j]<10) CkPrintf(" ");
	      CkPrintf("(%d)*",adjelems[j]);
	    }
	    else if (adjnodes[j]>-1) {
	      if (adjelems[j]<10) CkPrintf(" ");
	      CkPrintf("(%d) ", adjelems[j]);
	    }
	    else {
	      CkPrintf("(%d) ", adjelems[j]);
	    }
	    if (j<adjSz-1) CkPrintf(", ");
	  }
	  CkPrintf(" }\n");
	}	
  }

  {
/*	int rank = 0;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm,&rank);
	int adjs[2];
	if(rank == 0) {
	  adjs[0] = 20;
	  adjs[1] = 21;
	} else if(rank == 2) {
	  adjs[0] = 8;
	  adjs[1] = 16;
	} else if(rank == 1) {
	  adjs[0] = 11;
	  adjs[1] = 9;
	} else {
	  adjs[0] = 0;
	  adjs[1] = 1;
	}
	int newnode = 0;
	MPI_Barrier(comm);
	FEM_REF_INIT(mesh);
	CkPrintf("Shadow arrays have been bound\n");

	//if(rank == 0) {
	  FEM_Modify_Lock(mesh, adjs, 2, adjs, 0);
	  CkPrintf("locked chunk\n");
	  
	  newnode = FEM_add_node(mesh, adjs, 2, 0);
	  CkPrintf("newnode=%d\n", newnode);
	  FEM_Print_Mesh_Summary(mesh);

	  adjs[0] = newnode;
	  newnode = FEM_add_node(mesh, adjs, 2, 0);
	  CkPrintf("newnode=%d\n", newnode);

	  FEM_Print_Mesh_Summary(mesh);
	  int removenode = newnode;
	  FEM_remove_node(mesh, removenode);

	  FEM_Modify_Unlock(mesh);
	  CkPrintf("Unlocked chunk\n");
	  //}
	FEM_Print_Mesh_Summary(mesh);
	*/
	/*	
	CkPrintf("Marking 5 nodes and one element as invalid\n");
	FEM_set_entity_invalid(mesh, FEM_NODE, 5);
	FEM_set_entity_invalid(mesh, FEM_NODE, 6);
	FEM_set_entity_invalid(mesh, FEM_NODE, 7);
	FEM_set_entity_invalid(mesh, FEM_NODE, 8);
	FEM_set_entity_invalid(mesh, FEM_NODE, 9);	
	FEM_set_entity_invalid(mesh, FEM_ELEM, 9);
	FEM_Print_Mesh_Summary(mesh);
	
	CkPrintf("Marking 5 nodes and one element as valid again\n");
	FEM_set_entity_valid(mesh, FEM_NODE, 5);
	FEM_set_entity_valid(mesh, FEM_NODE, 6);
	FEM_set_entity_valid(mesh, FEM_NODE, 7);
	FEM_set_entity_valid(mesh, FEM_NODE, 8);
	FEM_set_entity_valid(mesh, FEM_NODE, 9);	
	FEM_set_entity_valid(mesh, FEM_ELEM, 9);
	FEM_Print_Mesh_Summary(mesh);
	*/	
	// add new nodes for a new element
	/*	int newnode1 = FEM_add_node(mesh);
	int newnode2 = FEM_add_node(mesh);
	int newnode3 = FEM_add_node(mesh);
	CkPrintf("3 new nodes\n");
	FEM_Print_Mesh_Summary(mesh);	
		
	int e1conn[3];
	e1conn[0]=newnode1;
	e1conn[1]=newnode2;
	e1conn[2]=newnode3;
	int newel1 = FEM_add_element(mesh,e1conn,3);
	CkPrintf("New Element\n");
	FEM_Print_Mesh_Summary(mesh);		
	*/
	/*
	CkPrintf("chunk %d Waiting for Synchronization\n",rank);
	MPI_Barrier(comm);
	CkPrintf("Synchronized\n");
	FEM_Print_Mesh_Summary(mesh);
	//CkExit();
	*/  
  }

  int numghosttri=FEM_Mesh_get_length(mesh,FEM_GHOST+FEM_ELEM+0); // Get number of nodes
  
  
   
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


