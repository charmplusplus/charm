#include "pgm.h"
#include "mpi.h"

#include "fem_adapt_new.h"
#include "fem_mesh_modify.h"
#include "matrix.h"

#define SUMMARY_ON

extern void _registerFEMMeshModify(void);

double getArea(double *n1_coord, double *n2_coord, double *n3_coord) {
  double area=0.0;
  double aLen, bLen, cLen, sLen, d, ds_sum;

  ds_sum = 0.0;
  for (int i=0; i<2; i++) {
    d = n1_coord[i] - n2_coord[i];
    ds_sum += d*d;
  }
  aLen = sqrt(ds_sum);
  ds_sum = 0.0;
  for (int i=0; i<2; i++) {
    d = n2_coord[i] - n3_coord[i];
    ds_sum += d*d;
  }
  bLen = sqrt(ds_sum);
  ds_sum = 0.0;
  for (int i=0; i<2; i++) {
    d = n3_coord[i] - n1_coord[i];
    ds_sum += d*d;
  }
  cLen = sqrt(ds_sum);
  sLen = (aLen+bLen+cLen)/2;
  if(sLen-aLen < 0) return 0.0;
  else if(sLen-bLen < 0) return 0.0;
  else if(sLen-cLen < 0) return 0.0; //area too small to note
  return (sqrt(sLen*(sLen-aLen)*(sLen-bLen)*(sLen-cLen)));
}

double getVolume(double *n1_coord, double *n2_coord, double *n3_coord, double *n4_coord) {
  Matrix mtx(4);
  int i;
  double vol;
  
  for (i=0; i<3; i++) {
    mtx.setElement(0, i, n1_coord[i]);
    mtx.setElement(1, i, n2_coord[i]);
    mtx.setElement(2, i, n3_coord[i]);
    mtx.setElement(3, i, n4_coord[i]);
  }
  for (i=0; i<4; i++)  
    mtx.setElement(i, 3, 1.0); // set end of row to 1

  vol = mtx.determinant() / 6.0;
  if (vol < 0.0) vol *= -1.0;

  return vol;
}

void publish_data_netfem(int i,  myGlobals g) {
  MPI_Barrier(MPI_COMM_WORLD);
  if (1) { //Publish data to the net
    int mesh=FEM_Mesh_default_read(); // Tell framework we are reading data from the mesh
    int rank = FEM_My_partition();
    g.nnodes=FEM_Mesh_get_length(mesh,FEM_NODE); // Get number of nodes
    g.coord=new vector3d[g.nnodes];
    int count = 0;
    vector3d *coord = new vector3d[g.nnodes];
    int *maptovalid = new int[g.nnodes];
    double *nodeid = new double[g.nnodes];
    // Get node positions
    FEM_Mesh_data(mesh, FEM_NODE, FEM_DATA+0, (double*)g.coord, 0, g.nnodes, FEM_DOUBLE, 3);
    for(int j=0; j<g.nnodes; j++) {
      if(FEM_is_valid(mesh,FEM_NODE+0,j)) {
	coord[count].x = g.coord[j].x;
	coord[count].y = g.coord[j].y;
	coord[count].z = g.coord[j].z;
	maptovalid[j] = count;
	nodeid[count] = j;
	count++;
      }
    }
    NetFEM n=NetFEM_Begin(rank,i,3,NetFEM_WRITE);
    NetFEM_Nodes(n,count,(double *)coord,"Position (m)");
    //NetFEM_Nodes(n,g.nnodes,(double *)g.coord,"Position (m)");
    NetFEM_Scalar(n,nodeid,1,"Node ID");

    // Get element data
    g.nelems=FEM_Mesh_get_length(mesh,FEM_ELEM+0); // Get number of elements
    g.conn=new connRec[g.nelems];
    connRec *conn = new connRec[g.nelems];
    double *elid = new double[g.nelems];
    count = 0;
    // Get connectivity for elements
    FEM_Mesh_data(mesh, FEM_ELEM+0, FEM_CONN, (int *)g.conn, 0, g.nelems, FEM_INDEX_0, 4);
    double totalVolume = 0.0;
    for(int j=0; j<g.nelems; j++) {
      if(FEM_is_valid(mesh,FEM_ELEM+0,j)) {
	conn[count][0] = maptovalid[g.conn[j][0]];
	conn[count][1] = maptovalid[g.conn[j][1]];
	conn[count][2] = maptovalid[g.conn[j][2]];
	conn[count][3] = maptovalid[g.conn[j][3]];
	elid[count] = j;
	totalVolume += getVolume(coord[conn[count][0]],coord[conn[count][1]],coord[conn[count][2]],coord[conn[count][3]]);
	if(totalVolume != totalVolume) {
	  CkPrintf("NAN\n");
	}
	count++;
      }
    }
    NetFEM_Elements(n,count,4,(int *)conn,"Tets");
    //NetFEM_Elements(n,g.nelems,4,(int *)g.conn,"Tets");
    NetFEM_Scalar(n,elid,1,"Element ID");
    NetFEM_End(n);

    double finalVolume;
    CkPrintf("Chunk[%d]: local volume: %.12f\n",rank,totalVolume);
    MPI_Reduce((void*)&totalVolume,(void*)&finalVolume,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if(rank == 0) CkPrintf("Chunk[%d]: total volume: %.12f\n",rank,finalVolume);

    delete [] g.coord;
    delete [] g.conn;
    delete [] coord;
    delete [] conn;
    delete [] maptovalid;
    delete [] nodeid;
    delete [] elid;
  }
}


extern "C" void
init(void)
{
  CkPrintf("init started\n");
  double startTime=CmiWallTimer();
  const char *eleName="mesh1.tet";//"adpmm/xxx.1.ele";//"88mesh/mesh1.tri";
  const char *nodeName="mesh1.node";//"adpmm/xxx.1.node";//"88mesh/mesh1.node";
  int nPts=0; //Number of nodes
  vector3d *pts=0; //Node coordinates
  int *bounds;

  CkPrintf("Reading node coordinates from %s\n",nodeName);
  //Open and read the node coordinate file
  {
    char line[1024];
    FILE *f=fopen(nodeName,"r");
    if (f==NULL) die("Can't open node file!");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nPts)) die("Can't read number of points!");
    pts=new vector3d[nPts];
    bounds = new int[nPts];
    for (int i=0;i<nPts;i++) {
      int ptNo;
      if (NULL==fgets(line,1024,f)) die("Can't read node input line!");
      if (5!=sscanf(line,"%d%lf%lf%lf%d",&ptNo,&pts[i].x,&pts[i].y,&pts[i].z,&bounds[i])) 
	die("Can't parse node input line!");
    }
    fclose(f);
  }
 
  int nEle=0;
  connRec *ele=NULL;
  CkPrintf("Reading elements from %s\n",eleName);
  //Open and read the element connectivity file
  {
    char line[10240];
    FILE *f=fopen(eleName,"r");
    if (f==NULL) die("Can't open element file!");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nEle)) die("Can't read number of elements!");
    ele=new connRec[nEle];
    for (int i=0;i<nEle;i++) {
      int elNo;
      if (NULL==fgets(line,10240,f)) die("Can't read element input line!");
      if (5!=sscanf(line,"%d%d%d%d%d",&elNo,&ele[i][0],&ele[i][1],&ele[i][2],&ele[i][3])) 
	die("Can't parse element input line!");  
      //ele[i][0]--; 
      //ele[i][1]--; 
      //ele[i][2]--; 
      //ele[i][3]--;
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
                3);              // Points have dimension 3 (x,y,z)
 
 
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
                4);              // Elements have degree 3, since triangles are defined 
                                 // by three nodes



  // Register values to the elements so we can keep track of them after partitioning
  double values[10240];
  for(int i=0;i<10240;i++)values[i]=i;

  // tets
  FEM_Mesh_data(fem_mesh,      // Add nodes to the current mesh
                FEM_ELEM,      // We are registering elements with type 1
                FEM_DATA,   
                (int *)values,   // The array of point locations
                0,               // 0 based indexing
                nEle,            // The number of elements
                FEM_DOUBLE,         // We use zero based node numbering
                1);
 

  //boundary conditions
  FEM_Mesh_data(fem_mesh,      // Add nodes to the current mesh
                FEM_NODE,      // We are registering elements with type 1
                FEM_BOUNDARY,   
                (int *)bounds,   // The array of point locations
                0,               // 0 based indexing
                nPts,            // The number of elements
                FEM_INT,         // We use zero based node numbering
                1);



  delete[] ele;
  delete[] pts;
  delete[] bounds;
  
  double *sizingData = new double[nEle];
  for (int i=0; i<nEle; i++) sizingData[i]=-1.0;
  FEM_Mesh_data(fem_mesh, FEM_ELEM+0, FEM_MESH_SIZING, sizingData, 0, nEle,
                  FEM_DOUBLE, 1);
  delete [] sizingData;

  // add ghost layers

  const int triangleFaces[4] = {0,1,2,3};
  CkPrintf("Adding Ghost layers\n");
  FEM_Add_ghost_layer(1,1);
  FEM_Add_ghost_elem(0,4,triangleFaces);

  CkPrintf("Finished with init (Reading took %.f s)\n",CmiWallTimer()-startTime);

}


// A driver() function 
// driver() is required in all FEM programs
extern "C" void
driver(void)
{
  int nnodes,nelems,nelems2,ignored;
  int i, myId=FEM_My_partition();
  myGlobals g;
  FEM_Register(&g,(FEM_PupFn)pup_myGlobals);
  
  _registerFEMMeshModify();

  printf("partition %d is in driver\n", myId);

  int mesh=FEM_Mesh_default_read(); // Tell framework we are reading data from the mesh
  
  // Get node data
  nnodes=FEM_Mesh_get_length(mesh,FEM_NODE); // Get number of nodes
  g.nnodes=nnodes;
  g.coord=new vector3d[nnodes];
  // Get node positions
  FEM_Mesh_data(mesh, FEM_NODE, FEM_DATA+0, (double*)g.coord, 0, nnodes, FEM_DOUBLE, 3);

  // Get element data
  nelems=FEM_Mesh_get_length(mesh,FEM_ELEM+0); // Get number of elements
  g.nelems=nelems;
  g.conn=new connRec[nelems];
  // Get connectivity for elements
  FEM_Mesh_data(mesh, FEM_ELEM+0, FEM_CONN, (int *)g.conn, 0, nelems, FEM_INDEX_0, 4);


  double* tridata =new double[nelems];
  FEM_Mesh_data(mesh, FEM_ELEM+0, FEM_DATA, (int *)tridata, 0, nelems, FEM_DOUBLE, 1);  

  int nelemsghost=FEM_Mesh_get_length(mesh,FEM_ELEM+0+FEM_GHOST); 
  double* trighostdata =new double[nelemsghost];
  FEM_Mesh_data(mesh, FEM_ELEM+0+FEM_GHOST, FEM_DATA, (int *)trighostdata, 0, nelemsghost, FEM_DOUBLE, 1);
  int nnodesghost=FEM_Mesh_get_length(mesh,FEM_NODE+0+FEM_GHOST);
  double* nodeghostdata =new double[3*nnodesghost];
  FEM_Mesh_data(mesh, FEM_NODE+0+FEM_GHOST, FEM_DATA, (int *)nodeghostdata, 0, nnodesghost, FEM_DOUBLE, 3);


  {
    const int triangleFaces[12] = {0,1,2,1,2,3,2,3,0,0,3,1};
    FEM_Add_elem2face_tuples(mesh, 0, 3, 4, triangleFaces);
    FEM_Mesh_create_elem_elem_adjacency(mesh);
    FEM_Mesh_allocate_valid_attr(mesh, FEM_ELEM+0);
    FEM_Mesh_allocate_valid_attr(mesh, FEM_NODE);
    FEM_Mesh_create_node_elem_adjacency(mesh);
    FEM_Mesh_create_node_node_adjacency(mesh);
    
    int netIndex = 0;
    publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
    FEM_Print_Mesh_Summary(mesh);
#endif
    int rank = 0;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm,&rank);
    
    MPI_Barrier(comm);
    FEM_REF_INIT(mesh);
    
    FEM_Mesh *meshP = FEM_Mesh_lookup(FEM_Mesh_default_read(),"driver");
    FEM_AdaptL *ada = meshP->getfmMM()->getfmAdaptL();
    int ret_op = -1;
    
    FEM_Adapt_Algs *adaptAlgs= meshP->getfmMM()->getfmAdaptAlgs();
    adaptAlgs->FEM_Adapt_Algs_Init(FEM_DATA+0,FEM_DATA+4);
    FEM_Interpolate *interp = meshP->getfmMM()->getfmInp();
    //interp->FEM_SetInterpolateNodeEdgeFnPtr(interpolate);
    
    MPI_Barrier(comm);
    
    //CkPrintf("Shadow arrays have been bound\n");
    /*
#ifdef SUMMARY_ON
    FEM_Print_Mesh_Summary(mesh);
#endif
    
    CkPrintf("Marking 5 nodes and one element as invalid\n");
    FEM_set_entity_invalid(mesh, FEM_NODE, 5);
    FEM_set_entity_invalid(mesh, FEM_NODE, 6);
    FEM_set_entity_invalid(mesh, FEM_NODE, 7);
    FEM_set_entity_invalid(mesh, FEM_NODE, 8);
    FEM_set_entity_invalid(mesh, FEM_NODE, 9);	
    FEM_set_entity_invalid(mesh, FEM_ELEM, 9);
    
#ifdef SUMMARY_ON
    FEM_Print_Mesh_Summary(mesh);
#endif
    
    CkPrintf("Marking 5 nodes and one element as valid again\n");
    FEM_set_entity_valid(mesh, FEM_NODE, 5);
    FEM_set_entity_valid(mesh, FEM_NODE, 6);
    FEM_set_entity_valid(mesh, FEM_NODE, 7);
    FEM_set_entity_valid(mesh, FEM_NODE, 8);
    FEM_set_entity_valid(mesh, FEM_NODE, 9);	
    FEM_set_entity_valid(mesh, FEM_ELEM, 9);
    */
#ifdef SUMMARY_ON
    FEM_Print_Mesh_Summary(mesh);
#endif
    
    int adjs[4];
    int elemid;
	if(rank == 0) {
	  adjs[0] = 0;
	  adjs[1] = 1;
	  adjs[2] = 2;
	  adjs[3] = 3;
	  elemid = 0;
	} else if(rank == 1) {
	  adjs[0] = 19;
	  adjs[1] = 5;
	  adjs[2] = 7;
	  adjs[3] = 21;
	  elemid = 21;
	} else if(rank == 2) {
	  adjs[0] = 8;
	  adjs[1] = 11;
	  adjs[2] = 6;
	  adjs[3] = 21;
	  elemid = 7;
	} else {
	  adjs[0] = 0;
	  adjs[1] = 1;
	  adjs[2] = 2;
	  adjs[3] = 3;
	  elemid = 0;
	}
	int newel1 = 0;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
	FEM_Print_n2e(mesh,adjs[0]);
	FEM_Print_n2e(mesh,adjs[1]);
	FEM_Print_n2e(mesh,adjs[2]);
	FEM_Print_n2e(mesh,adjs[3]);
	FEM_Print_n2n(mesh,adjs[0]);
	FEM_Print_n2n(mesh,adjs[1]);
	FEM_Print_n2n(mesh,adjs[2]);
	FEM_Print_n2n(mesh,adjs[3]);
	FEM_Print_e2n(mesh,elemid);
	FEM_Print_e2e(mesh,elemid);
#endif

	//FEM_Modify_Lock(mesh, adjs, 3, adjs, 0);
	if(rank == 0) {
	  FEM_remove_element(mesh, elemid, 0, 1);
	}
	//FEM_Modify_Unlock(mesh);
	MPI_Barrier(comm);
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif

	//FEM_Modify_Lock(mesh, adjs, 3, adjs, 0);
	if(rank == 0) {
	  newel1 = FEM_add_element(mesh,adjs,4,0,0);
	  CkPrintf("New Element\n");
	}
	//FEM_Modify_Unlock(mesh);
	publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
	FEM_Print_n2e(mesh,adjs[0]);
	FEM_Print_n2e(mesh,adjs[1]);
	FEM_Print_n2e(mesh,adjs[2]);
	FEM_Print_n2e(mesh,adjs[3]);
	FEM_Print_n2n(mesh,adjs[0]);
	FEM_Print_n2n(mesh,adjs[1]);
	FEM_Print_n2n(mesh,adjs[2]);
	FEM_Print_n2n(mesh,adjs[3]);
	FEM_Print_e2n(mesh,newel1);
	FEM_Print_e2e(mesh,newel1);
#endif
	/*
	if(rank==0){
	  FEM_Print_Mesh_Summary(mesh);
	  CkPrintf("%d: Removing element \n", rank);
	  
	  int nelemsghost   =FEM_Mesh_get_length(mesh,FEM_ELEM+0+FEM_GHOST); 
	  int numvalidghost =FEM_count_valid(mesh,FEM_ELEM+0+FEM_GHOST);
	  CkPrintf("nelemsghost=%d numvalidghost=%d\n", nelemsghost, numvalidghost);
	
	  for(int i=1;i<20;i++){
		if(FEM_is_valid(mesh, FEM_ELEM+FEM_GHOST, i)){
		  double data[1];
		  FEM_Mesh_data(mesh, FEM_ELEM+FEM_GHOST, FEM_DATA, (int *)data, i, 1, FEM_DOUBLE, 1);  

		  CkPrintf("%d: Eating ghost element %d with value %f\n", rank, i, data[1]);
		  int conn[3];
		  
		  FEM_Mesh_data(mesh, FEM_ELEM+FEM_GHOST, FEM_CONN, (int *)conn, i, 1, FEM_INDEX_0, 3);
		  CkPrintf("conn for element is: %d %d %d\n", conn[0], conn[1], conn[2]);
		  FEM_Modify_Lock(mesh, conn, 3, conn, 0);
		  FEM_remove_element(mesh, FEM_From_ghost_index(i), 0, 1);
		  FEM_Modify_Unlock(mesh);

		  MPI_Barrier(comm);
		  FEM_Print_Mesh_Summary(mesh);

		  FEM_Modify_Lock(mesh, conn, 3, conn, 0);
		  FEM_add_element(mesh, conn, 3, 0, rank); // add locally
		  FEM_Modify_Unlock(mesh);
		  CkPrintf("New conn for element is: %d %d %d\n", conn[0], conn[1], conn[2]);
		  
		  publish_data_netfem(netIndex,g); netIndex++;
		  FEM_Print_Mesh_Summary(mesh);
		}
		else{
		  //  CkPrintf("invalid element %d\n", i);
		}
	  }
	}
	else {
	  CkPrintf("Rank %d\n", rank);
	  for(int i=1;i<20;i++){
	    MPI_Barrier(comm);
	    FEM_Print_Mesh_Summary(mesh);

	    publish_data_netfem(netIndex,g); netIndex++;
	    FEM_Print_Mesh_Summary(mesh);
	  }
	}
	
	publish_data_netfem(netIndex,g); netIndex++;
	*/	

	/*
	CkPrintf("Starting Local edge flips on individual chunks\n");
	int flip[4];
	if(rank == 0) {
	  flip[0] = 1;
	  flip[1] = 2;
	  flip[2] = 0;
	  flip[3] = 3;
	}
	else if(rank == 1) {
	  flip[0] = 1;
	  flip[1] = 2;
	  flip[2] = 0;
	  flip[3] = 4;
	}
	else if(rank == 2) {
	  flip[0] = 13;
	  flip[1] = 14;
	  flip[2] = 15;
	  flip[3] = 7;
	}
	else {
	  flip[0] = 0;
	  flip[1] = 1;
	  flip[2] = 2;
	  flip[3] = 3;
	}
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	ret_op = ada->edge_flip(flip[0],flip[1]);
	publish_data_netfem(netIndex,g); netIndex++;
	ret_op = ada->edge_flip(flip[2],flip[3]);
	publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	
	CkPrintf("Starting shared edge flips on individual chunks\n");
	int sflip[4];
	if(rank == 0) {
	  sflip[0] = 19;
	  sflip[1] = 18;
	  sflip[2] = 1;
	  sflip[3] = -4;
	}
	else if(rank == 1) {
	  sflip[0] = 5;
	  sflip[1] = 6;
	  sflip[2] = 7;
	  sflip[3] = -5;
	}
	else if(rank == 2) {
	  sflip[0] = 11;
	  sflip[1] = 2;
	  sflip[2] = 0;
	  sflip[3] = -2;
	}
	else {
	  sflip[0] = 0;
	  sflip[1] = 1;
	  sflip[2] = 2;
	  sflip[3] = 3;
	}
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	ret_op = ada->edge_flip(sflip[0],sflip[1]);
	publish_data_netfem(netIndex,g); netIndex++;
	if(ret_op > 0) {
	  if(sflip[2]<0) sflip[2] = ret_op;
	  else if(sflip[3]<0) sflip[3] = ret_op;
	}
	ret_op = ada->edge_flip(sflip[2],sflip[3]);
	publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	
	CkPrintf("Starting Local edge bisect on individual chunks\n");
	int bisect[2];
	if(rank == 0) {
	  bisect[0] = 16;
	  bisect[1] = 21;
	}
	else if(rank == 1) {
	  bisect[0] = 5;
	  bisect[1] = 6;
	}
	else if(rank == 2) {
	  bisect[0] = 8;
	  bisect[1] = 11;
	}
	else {
	  bisect[0] = 0;
	  bisect[1] = 1;
	}
	if(rank==2) ret_op = ada->edge_bisect(bisect[0],bisect[1]);
	publish_data_netfem(netIndex,g); netIndex++;
	adaptAlgs->tests();
	MPI_Barrier(comm);
	
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	
	CkPrintf("Starting Local vertex remove on individual chunks\n");
	int vr[2];
	if(rank == 0) {
	  vr[0] = ret_op;
	  vr[1] = 6;
	}
	else if(rank == 1) {
	  vr[0] = ret_op;
	  vr[1] = 13;
	}
	else if(rank == 2) {
	  vr[0] = ret_op;
	  vr[1] = 21;
	}
	else {
	  vr[0] = ret_op;
	  vr[1] = 1;
	}
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	ret_op = ada->vertex_remove(vr[0],vr[1]);
	publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	
	
	CkPrintf("Starting shared edge bisect on individual chunks\n");
	int sbisect[2];
	if(rank == 0) {
	  sbisect[0] = 1;//4;//21;
	  sbisect[1] = 19;//20;
	}
	else if(rank == 1) {
	  sbisect[0] = 0;
	  sbisect[1] = 21;
	}
	else if(rank == 2) {
	  sbisect[0] = 20;//9;//3;
	  sbisect[1] = 8;//2;//19;
	}
	else {
	  sbisect[0] = 0;
	  sbisect[1] = 1;
	}
	
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	ret_op = ada->edge_bisect(sbisect[0],sbisect[1]);
	publish_data_netfem(netIndex,g); netIndex++;
	CkPrintf("Starting shared vertex remove on individual chunks\n");

	int svr[2];
	if(rank == 0) {
	  svr[0] = ret_op;
	  svr[1] = 19;
	}
	else if(rank == 1) {
	  svr[0] = ret_op;
	  svr[1] = 21;
	}
	else if(rank == 2) {
	  svr[0] = ret_op;
	  svr[1] = 20;
	}
	else {
	  svr[0] = ret_op;
	  svr[1] = 1;
	}
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	ret_op = ada->vertex_remove(svr[0],svr[1]);
	publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif

	CkPrintf("Starting Local edge contract on individual chunks\n");
	int contract[2];
	if(rank == 0) {
	  contract[1] = 16;
	  contract[0] = 21;
	}
	else if(rank == 1) {
	  contract[0] = 5;
	  contract[1] = 6;
	}
	else if(rank == 2) {
	  contract[0] = 19;
	  contract[1] = 17;
	}
	else {
	  contract[0] = 0;
	  contract[1] = 1;
	}
	
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	if(rank==0) ret_op = ada->edge_contraction(contract[0],contract[1]);
	publish_data_netfem(netIndex,g); netIndex++;

	//CkPrintf("Starting Local vertex split on individual chunks\n");
	/*
	int vs[3];
	if(rank == 0) {
	  vs[0] = ret_op;
	  vs[1] = 9;
	  vs[2] = 13;
	}
	else if(rank == 1) {
	  vs[0] = ret_op;
	  vs[1] = 8;
	  vs[2] = 7;
	}
	else if(rank == 2) {
	  vs[0] = ret_op;
	  vs[1] = 14;
	  vs[2] = 23;
	}
	else {
	  vs[0] = ret_op;
	  vs[1] = 2;
	  vs[2] = 3;
	}
#ifdef SUMMARY_ON
	//FEM_Print_Mesh_Summary(mesh);
#endif
	//ret_op = ada->vertex_split(vs[0],vs[1],vs[2]);
	//publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	*/
	/*
	CkPrintf("Starting shared edge contract on individual chunks\n");
	int scontract[2];
	if(rank == 0) {
	  scontract[0] = 9;
	  scontract[1] = 10;
	}
	else if(rank == 1) {
	  scontract[0] = 5;
	  scontract[1] = 6;
	}
	else if(rank == 2) {
	  scontract[0] = 11;
	  scontract[1] = 2;
	}
	else {
	  scontract[0] = 0;
	  scontract[1] = 1;
	}
	
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	ret_op = ada->edge_contraction(scontract[0],scontract[1]);
	publish_data_netfem(netIndex,g); netIndex++;
	/*
	//CkPrintf("Starting shared vertex split on individual chunks\n");
	int svs[3];
	if(rank == 0) {
	  svs[0] = ret_op;
	  svs[1] = 1;
	  svs[2] = -6;
	}
	else if(rank == 1) {
	  svs[0] = ret_op;
	  svs[1] = 7;
	  svs[2] = 7;
	}
	else if(rank == 2) {
	  svs[0] = ret_op;
	  svs[1] = 0;
	  svs[2] = -2;
	}
	else {
	  svs[0] = ret_op;
	  svs[1] = 2;
	  svs[2] = 3;
	}
#ifdef SUMMARY_ON
	//FEM_Print_Mesh_Summary(mesh);
#endif
	//ret_op = ada->vertex_split(svs[0],svs[1],svs[2]);
	//publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	*/

	/*
	CkPrintf("Starting LEB on individual chunks\n");
	int *leb_elem = (int*)malloc(1*sizeof(int));
	if(rank==0) {
	  leb_elem[0] = 2;
	}
	else if(rank==1) {
	  leb_elem[0] = 13; //4;
	}
	else if(rank==2) {
	  leb_elem[0] = 20; //26;
	}
	else if (rank == 3){
	  leb_elem[0] = 14;
	}
	else {
	  leb_elem[0] = 0;
	}

	adaptAlgs->refine_element_leb(leb_elem[0]);
	publish_data_netfem(netIndex,g); netIndex++;
	*/
	/*
	  int nEle;
	  //for(int tstep = 0; tstep < 2; tstep++) {
	  nEle = FEM_Mesh_get_length(mesh, FEM_ELEM);	
	  for (int i=0; i<nEle; i++)
	  if (FEM_is_valid(mesh, FEM_ELEM, i))
	  adaptAlgs->refine_element_leb(i);
	  publish_data_netfem(netIndex,g); netIndex++;
	  FEM_Print_Mesh_Summary(mesh);
	  //}
	  */

      
      double targetArea = 0.00004;
      
      for(int tstep = 0; tstep < 0; tstep++) {
	adaptAlgs->simple_refine(targetArea);
	publish_data_netfem(netIndex,g); netIndex++;
	adaptAlgs->tests();
	MPI_Barrier(comm);
	
	//int *nodes = new int[g.nnodes];
	//for (int i=0; i<g.nnodes; i++) nodes[i]=i;	
	//FEM_mesh_smooth(mesh, nodes, g.nnodes, FEM_DATA+0);
	//publish_data_netfem(netIndex,g); netIndex++;
	//delete [] nodes;

#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	adaptAlgs->simple_coarsen(targetArea);
	publish_data_netfem(netIndex,g); netIndex++;
	adaptAlgs->tests();
	MPI_Barrier(comm);
	
	//int *nodes = new int[g.nnodes];
	//for (int i=0; i<g.nnodes; i++) nodes[i]=i;
	//FEM_mesh_smooth(mesh, nodes, g.nnodes, FEM_DATA+0);
	//publish_data_netfem(netIndex,g); netIndex++;
	//delete [] nodes;

#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
      }

      targetArea = 0.00004;
      
      for(int tstep = 0; tstep < 0; tstep++) {
	adaptAlgs->simple_refine(targetArea);
	publish_data_netfem(netIndex,g); netIndex++;
	adaptAlgs->tests();
	MPI_Barrier(comm);
	
	//int *nodes = new int[g.nnodes];
	//for (int i=0; i<g.nnodes; i++) nodes[i]=i;	
	//FEM_mesh_smooth(mesh, nodes, g.nnodes, FEM_DATA+0);
	//publish_data_netfem(netIndex,g); netIndex++;
	//delete [] nodes;


	targetArea *= 0.5;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
      }
      targetArea /= 0.4;
      
      for(int tstep = 0; tstep < 0; tstep++) {
	adaptAlgs->simple_coarsen(targetArea);
	publish_data_netfem(netIndex,g); netIndex++;
	adaptAlgs->tests();
	MPI_Barrier(comm);
	
	//int *nodes = new int[g.nnodes];
	//for (int i=0; i<g.nnodes; i++) nodes[i]=i;
	//FEM_mesh_smooth(mesh, nodes, g.nnodes, FEM_DATA+0);
	//publish_data_netfem(netIndex,g); netIndex++;
	//delete [] nodes;

	targetArea *= 1.5;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
      }

      //wave propagation on a bar
      targetArea = 0.00004;
      double xmin = 0.00;
      double xmax = 0.1;
      double ymin = 0.00;
      double ymax = 0.01;
      for(int tstep = 0; tstep < 0; tstep++) {
	targetArea = 0.000002;
	adaptAlgs->simple_refine(targetArea, xmin, ymin, xmax, ymax);
	publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	targetArea = 0.0000014;
	adaptAlgs->simple_coarsen(targetArea, xmin, ymin, xmax, ymax);
	publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	ymin += 0.01;
	ymax += 0.01;
      }

      //crack propagation on a block
      targetArea = 0.00004;
      xmin = 0.00;
      xmax = 0.2;
      double xcrackmin = 0.09;
      double xcrackmax = 0.10;
      ymin = 0.00;
      ymax = 0.02;
      for(int tstep = 0; tstep < 0; tstep++) {
	targetArea = 0.000025;
	adaptAlgs->simple_refine(targetArea, xmin, ymin, xmax, ymax);
	publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	targetArea = 0.00005;
	adaptAlgs->simple_coarsen(targetArea, xmin, ymin, xmax, ymax);
	//publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	FEM_Print_Mesh_Summary(mesh);
#endif
	/*if(tstep > 2) {
	  targetArea = 0.000025;
	  adaptAlgs->simple_refine(targetArea, xcrackmin, ymin, xcrackmax, ymax);
	  //publish_data_netfem(netIndex,g); netIndex++;
#ifdef SUMMARY_ON
	  FEM_Print_Mesh_Summary(mesh);
#endif
	  xcrackmin -= 0.004;
	  xcrackmax += 0.004;
	}
	*/

	ymin += 0.02;
	ymax += 0.02;
      }


      CkPrintf("chunk %d Waiting for Synchronization\n",rank);
      MPI_Barrier(comm);
      CkPrintf("Synchronized\n");
#ifdef SUMMARY_ON
      FEM_Print_Mesh_Summary(mesh);
#endif
      publish_data_netfem(netIndex,g); netIndex++;
      
      CkExit();
  }
  
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
    g->coord=new vector3d[nnodes];
    g->conn=new connRec[nelems];
    //g->R_net=new vector3d[nnodes]; //Net force
    //g->d=new vector3d[nnodes];//Node displacement
    //g->v=new vector3d[nnodes];//Node velocity
    //g->a=new vector3d[nnodes];
    //g->S11=new double[nelems];
    //g->S22=new double[nelems];
    //g->S12=new double[nelems];
  }
  pup_doubles(p,(double *)g->coord,2*nnodes);
  pup_ints(p,(int *)g->conn,3*nelems);
  //pup_doubles(p,(double *)g->R_net,3*nnodes);
  //pup_doubles(p,(double *)g->d,3*nnodes);
  //pup_doubles(p,(double *)g->v,3*nnodes);
  //pup_doubles(p,(double *)g->a,3*nnodes);
  //pup_doubles(p,(double *)g->S11,nelems);
  //pup_doubles(p,(double *)g->S22,nelems);
  //pup_doubles(p,(double *)g->S12,nelems);
  if (pup_isDeleting(p)) {
    delete[] g->coord;
    delete[] g->conn;
    //delete[] g->R_net;
    //delete[] g->d;
    //delete[] g->v;
    //delete[] g->a;
    //delete[] g->S11;
    //delete[] g->S22;
    //delete[] g->S12;
  }
}


void FEM_mesh_smooth(int mesh, int *nodes, int nNodes, int attrNo)
{
  vector3d *centroids, newPos, *coords, *ghostCoords, *vGcoords;
  int nEle, nGn, *boundVals, nodesInChunk, nVg;
  int neighbors[3], *adjelems;
  int gIdxN;
  int j=0;
  double x[3], y[3];
  FEM_Mesh *meshP = FEM_Mesh_lookup(mesh, "driver");

  nodesInChunk = FEM_Mesh_get_length(mesh,FEM_NODE);
  boundVals = new int[nodesInChunk];
  nGn = FEM_Mesh_get_length(mesh, FEM_GHOST + FEM_NODE);
  coords = new vector3d[nodesInChunk+nGn];

  FEM_Mesh_data(mesh, FEM_NODE, FEM_BOUNDARY, (int*) boundVals, 0, nodesInChunk, FEM_INT, 1);    

  FEM_Mesh_data(mesh, FEM_NODE, attrNo, (double*)coords, 0, nodesInChunk, FEM_DOUBLE, 2);
  for (int i=0; i<(nodesInChunk); i++) {
    //CkPrintf(" coords[%d]: (%f, %f)\n", i, coords[i].x, coords[i].y);
  }
  IDXL_Layout_t coord_layout = IDXL_Layout_create(IDXL_DOUBLE, 2);
  FEM_Update_ghost_field(coord_layout,-1, coords); 
  ghostCoords = &(coords[nodesInChunk]);
  /*
  for (int i=0; i<nGn;i++) {
    if (FEM_is_valid(mesh, FEM_GHOST+FEM_NODE, i)) {
      CkPrintf("ghost %d is valid \n", i);	  
      // vGcoords[j]=ghostCoords[i];
      //j++;
    }
    else
      CkPrintf("ghost %d is invalid \n", i);
  }
  */
  for (int i=0; i<(nodesInChunk+nGn); i++) {
    //CkPrintf(" coords[%d]: (%f, %f)\n", i, coords[i].x, coords[i].y);
  }
//  FEM_Mesh_data(FEM_Mesh_default_write(), FEM_GHOST+FEM_NODE, attrNo, (double*)ghostCoords, 0, nGn, FEM_DOUBLE, 2);
 
  for (int i=0; i<nNodes; i++)
  {
    newPos.x=0;
    newPos.y=0;
    CkAssert(nodes[i]<nodesInChunk);    
    if (FEM_is_valid(mesh, FEM_NODE, i) && boundVals[i]>-1) //node must be internal
    {
      meshP->n2e_getAll(i, &adjelems, &nEle);
      centroids = new vector3d[nEle];
      
      for (int j=0; j<nEle; j++) { //for all adjacent elements, find centroids
	meshP->e2n_getAll(adjelems[j], neighbors);
	for (int k=0; k<3; k++) {
	  if (neighbors[k]<-1) {
	    gIdxN = FEM_From_ghost_index(neighbors[k]);
	    x[k] = ghostCoords[gIdxN].x;
	    y[k] = ghostCoords[gIdxN].y;
	  }
	  else {
	    x[k] = coords[neighbors[k]].x;
	    y[k] = coords[neighbors[k]].y;
	  }
	}     
	centroids[j].x=(x[0]+x[1]+x[2])/3.0;
	centroids[j].y=(y[0]+y[1]+y[2])/3.0;
	newPos.x += centroids[j].x;
	newPos.y += centroids[j].y;
      }
      newPos.x/=nEle;
      newPos.y/=nEle;
      FEM_set_entity_coord2(mesh, FEM_NODE, nodes[i], newPos.x, newPos.y);
      delete [] centroids;
      delete [] adjelems;
    }
  }
  delete [] coords;
  delete [] boundVals;
}

void interpolate(FEM_Interpolate::NodalArgs args, FEM_Mesh *meshP)
{
  //CkPrintf("INTERPOLATOR!!!!!!!!!!!\n");
  int length = meshP->node.realsize();
  int *boundVals= new int[length];

  FEM_Mesh_dataP(meshP, FEM_NODE, FEM_BOUNDARY, (int*) boundVals, 0, length , FEM_INT, 1);   
  CkVec<FEM_Attribute *>*attrs = (meshP->node).getAttrVec();
  for (int i=0; i<attrs->size(); i++) {
    FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
    if (a->getAttr() < FEM_ATTRIB_TAG_MAX || a->getAttr()==FEM_BOUNDARY) {
      if (a->getAttr()==FEM_BOUNDARY) {
	
	int n1_bound =boundVals[args.nodes[0]]; 
	int n2_bound =boundVals[args.nodes[1]]; 
	if (n1_bound == n2_bound && n1_bound < 0) {
	  a->copyEntity(args.n, *a, args.nodes[0]);
	} else if (n1_bound != n2_bound && n1_bound<0 && n2_bound < 0){
	  a->copyEntity(args.n, *a, args.nodes[0]); //a node which is not on the boundary
	} else  if(n1_bound < 0){
	  a->copyEntity(args.n, *a, args.nodes[1]); 
	} else {
	  a->copyEntity(args.n, *a, args.nodes[0]); 
	}
      }
      else {
	FEM_DataAttribute *d = (FEM_DataAttribute *)a;
	d->interpolate(args.nodes[0], args.nodes[1], args.n, args.frac);
      }
    }
  }
  delete boundVals;
}



