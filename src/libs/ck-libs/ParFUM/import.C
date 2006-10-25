#include "import.h"
#include <algorithm>

void ParFUM_desharing(int meshid){
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_desharing"))->lookup(meshid,"ParFUM_desharing");
	mesh->clearSharedNodes();
}


void ParFUM_deghosting(int meshid){
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_deghosting"))->lookup(meshid,"ParFUM_deghosting");
	mesh->clearGhostNodes();
	mesh->clearGhostElems();
}

void ParFUM_recreateSharedNodes(int meshid, int dim) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_size, rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &rank);
  // Shared data will be temporarily stored in the following structure
  int *sharedNodeCounts; // sharedCounts[i] = number of nodes shared with rank i
  int **sharedNodeLists; // sharedNodes[i] is the list of nodes shared with rank i
  // Initialize shared data
  sharedNodeCounts = (int *)malloc(comm_size*sizeof(int));
  sharedNodeLists = (int **)malloc(comm_size*sizeof(int *));
  for (int i=0; i<comm_size; i++) {
    sharedNodeLists[i] = NULL;
    sharedNodeCounts[i] = 0;
  }
  // Get local node count and coordinates
  int numNodes;
  int coord_msg_tag=42, sharedlist_msg_tag=43;
  double *nodeCoords;
  numNodes = FEM_Mesh_get_length(meshid,FEM_NODE);
  nodeCoords = (double *)malloc(dim*numNodes*sizeof(double));
  FEM_Mesh_data(meshid,FEM_NODE,FEM_COORD, nodeCoords, 0, numNodes,FEM_DOUBLE, dim);
  /*
  printf("Node Coords for rank %d \n",rank);
  for(int n=0;n<numNodes;n++){
    printf("%d -> ", n);
    for (int m=0; m<dim; m++) 
      printf("%.5lf ", nodeCoords[dim*n+m]);
    printf("\n");
  }
  //*/
  
  // Begin exchange of node coordinates to determine shared nodes
  // compute bounding box
  double *localBoundingBox = (double*)malloc(sizeof(double)*2*dim);
  double *allBoundingBoxes = (double*)malloc(sizeof(double)*2*dim*comm_size);
  ParFUM_findBoundingBox(numNodes, dim, nodeCoords, localBoundingBox);
  MPI_Allgather(localBoundingBox, 2*dim, MPI_DOUBLE, 
  		allBoundingBoxes, 2*dim, MPI_DOUBLE,
  		comm);

  // only exchange when bounding boxes collide
  for (int i=rank+1; i<comm_size; i++) { //send nodeCoords to rank i
    if (ParFUM_boundingBoxesCollide(dim, localBoundingBox, &allBoundingBoxes[2*dim*i])) {
      //printf("[%d] Sending %d doubles to rank %d \n",rank,dim*numNodes,i);
      MPI_Send(nodeCoords, dim*numNodes, MPI_DOUBLE, i, coord_msg_tag, comm);
    }
  }

  // Handle node coordinate-matching requests from other ranks
  for (int i=0; i<rank; i++) {
    // only recieve if the bounding boxes collide.
    if (ParFUM_boundingBoxesCollide(dim, localBoundingBox, &allBoundingBoxes[2*dim*i])) {
      std::vector<int> remoteSharedNodes, localSharedNodes;
      double *recvNodeCoords;
      MPI_Status status;
      int source, length;
      // Probe for a coordinate message from any source; extract source and msg length
      MPI_Probe(MPI_ANY_SOURCE, coord_msg_tag, comm, &status);
      source = status.MPI_SOURCE;
      length = status.MPI_LENGTH/sizeof(double);
      //printf("[%d] Receiving %d doubles from rank %d \n",rank,length,i);
      // Receive whatever data was available according to probe
      recvNodeCoords = (double *)malloc(length*sizeof(double));
      MPI_Recv((void*)recvNodeCoords, length, MPI_DOUBLE, source, 
	       coord_msg_tag, comm, &status);
      // Match coords between local nodes and received coords
      int recvNodeCount = length/dim;
      ParFUM_findMatchingCoords(dim,
				numNodes, nodeCoords,
				recvNodeCount, recvNodeCoords,
				localSharedNodes,
				remoteSharedNodes
				);

      // Copy local nodes that were shared with source into the data structure
      int *localSharedNodeList = (int *)malloc(localSharedNodes.size()*sizeof(int));
      for (int m=0; m<localSharedNodes.size(); m++) {
	localSharedNodeList[m] = localSharedNodes[m];
      }
      sharedNodeCounts[source] = localSharedNodes.size();
      sharedNodeLists[source] = localSharedNodeList;
      // do not delete localSharedNodeList as a pointer to it is stored
      // Send remote nodes that were shared with this partition to remote partition
      MPI_Send((int *)&remoteSharedNodes[0], remoteSharedNodes.size(), MPI_INT, source, 
	       sharedlist_msg_tag, comm);
      free(recvNodeCoords);
    }
  }
  for (int i=rank+1; i<comm_size; i++) {  // recv shared node lists
    if (ParFUM_boundingBoxesCollide(dim, localBoundingBox, &allBoundingBoxes[2*dim*i])) {
      int *sharedNodes;
      MPI_Status status;
      int source, length;
      // Probe for a shared node list from any source; extract source and msg length
      MPI_Probe(MPI_ANY_SOURCE, sharedlist_msg_tag, comm, &status);
      source = status.MPI_SOURCE;
      length = status.MPI_LENGTH/sizeof(int);
      // Recv the shared node list the probe revealed was available
      sharedNodes = (int *)malloc(length*sizeof(int));
      MPI_Recv((void*)sharedNodes, length, MPI_INT, source, sharedlist_msg_tag, comm, &status);
      // Store the shared node list in the data structure
      sharedNodeCounts[source] = length;
      sharedNodeLists[source] = sharedNodes;
      // don't delete sharedNodes! we kept a pointer to it!
    }
  }
  free(localBoundingBox);
  free(allBoundingBoxes);


  // IMPLEMENT ME: use sharedNodeLists and sharedNodeCounts to move shared node data 
  // to IDXL
  FEM_Mesh *mesh = (FEM_chunk::get("ParFUM_recreateSharedNodes"))->lookup(meshid,"ParFUM_recreateSharedNodes");
  IDXL_Side &shared = mesh->node.shared;
  
  for(int i=0;i<comm_size;i++){
    if(i == rank)
      continue;
    if(sharedNodeCounts[i] != 0){
      IDXL_List &list = shared.addList(i);
      for(int j=0;j<sharedNodeCounts[i];j++){
	list.push_back(sharedNodeLists[i][j]);
      }
    }
  }
  //printf("After recreating shared nodes %d \n",rank);
  //shared.print();
	
  // Clean up
  free(nodeCoords);
  free(sharedNodeCounts);
  for (int i=0; i<comm_size; i++) {
    if (sharedNodeLists[i])
      free(sharedNodeLists[i]);
  }
  free(sharedNodeLists);
}

void ParFUM_createComm(int meshid, int dim)
{
  ParFUM_desharing(meshid);
  ParFUM_deghosting(meshid);
  MPI_Barrier(MPI_COMM_WORLD);
  ParFUM_recreateSharedNodes(meshid, dim);
  MPI_Barrier(MPI_COMM_WORLD);
  ParFUM_generateGlobalNodeNumbers(meshid);
  FEM_Mesh *mesh = (FEM_chunk::get("ParFUM_recreateSharedNodes"))->lookup(meshid,"ParFUM_recreateSharedNodes");
  MPI_Barrier(MPI_COMM_WORLD);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  struct ghostdata *gdata;
  if(rank == 0){
    gdata = gatherGhosts();
  }else{
    gdata = new ghostdata;
  }
  MPI_Bcast_pup(*gdata,0,MPI_COMM_WORLD);
  makeGhosts(mesh,MPI_COMM_WORLD,0,gdata->numLayers,gdata->layers);
  MPI_Barrier(MPI_COMM_WORLD);
}

void ParFUM_import_nodes(int meshid, int numNodes, double *nodeCoords, int dim)
{
  FEM_Mesh_become_set(meshid);
  FEM_Mesh_data(meshid, FEM_NODE, FEM_COORD, nodeCoords, 0, numNodes, FEM_DOUBLE, dim);
  FEM_Mesh_become_get(meshid);
}

void ParFUM_import_elems(int meshid, int numElems, int nodesPer, int *conn, int type)
{
  FEM_Mesh_become_set(meshid);
  FEM_Mesh_data(meshid, FEM_ELEM+type, FEM_CONN, conn, 0, numElems, FEM_INDEX_0,
		nodesPer);
  FEM_Mesh_become_get(meshid);
}


/**
 * To use std::sort, we need a coordinate class that defines the <
 * operator and implements value semantics for the = operator.
 */
struct Coord {
  static int dim;
  int index;
  double* value;
  bool operator<(const Coord& other) const {
    return coordLessThan(value, other.value, dim);
  }
  bool operator==(Coord& other) {
    return coordEqual(value, other.value, dim);
  }
};

int Coord::dim=0;

/**
 * Finds all the duplicate coords between two lists in n log n time.  
 *
 * Takes definitions for the lists of both coordinates, and returns a list 
 * containing: 
 *
 * <pair_index_a1> <pair_index_b1> <pair_index_a2> <pair_index_b2>
 *
 */
void ParFUM_findMatchingCoords(int dim, 
			       int extent_a, double* a, 
			       int extent_b, double* b,
			       std::vector<int>& matches_a,
			       std::vector<int>& matches_b
			       ) {
  using namespace std;
  
  //read in the arrays
  vector<Coord> point_a(extent_a);
  for (unsigned int ii=0; ii<point_a.size(); ii++) {
    point_a[ii].index = ii;
    point_a[ii].value = a+dim*ii;
  }
  
  vector<Coord> point_b(extent_b);
  for (unsigned int ii=0; ii<point_b.size(); ii++) {
    point_b[ii].index = ii;
    point_b[ii].value = b+dim*ii;
  }
  
  //sort the lists in ascending order
  Coord::dim = dim;
  sort(point_a.begin(), point_a.end());
  sort(point_b.begin(), point_b.end());
  
  //search through both lists, looking for nodes that are equivalent.
  unsigned int cursor_a = 0;
  unsigned int cursor_b = 0;
  while((cursor_a < point_a.size()) && (cursor_b < point_b.size())) {
    int comparison = coordCompare(point_a[cursor_a].value, point_b[cursor_b].value, dim);
    if (1==comparison) {
      //a < b, so advance a to meet b
      cursor_a++;
    } else if (-1==comparison) {
      //a > b, so advance b to meet a
      cursor_b++;
    } else if (0==comparison) {
      //we have a match!  Record the original indicies.
      matches_a.push_back(point_a[cursor_a].index);
      matches_b.push_back(point_b[cursor_b].index);
      
      //now, advance for the next comparison.
      cursor_a++;
      cursor_b++;
    } else {
      //code should never get here.
      //printf("comparison = %d\n", comparison);
      assert(comparison == 0);
    }
  } 
}

void
ParFUM_findBoundingBox
/////////////////////////////////////////////////////
/**
 * Computes the bounding box for a region of points
 *
 * I ordered the index lookups this way so it would be conceptually
 * identical to how C lays out 2d arrays.
 */
(
 /// numer of points in our region
 int nPoints,
 /// are we working in 1D, 2D, or 3D ?
 int dim,
 /// array of points
 double* points,
 /**
  * output parameter 
  *
  * \pre points to a double[dim*2] region of memory
  *
  * \post contains the bounding box for our region.  The first dim
  * coordinates are the minimum extent for our region, while the second
  * dim coordinates contain the maxium for our region.
  */
 double* boundingBox
 ) {

  for(int idim=0; idim<dim; idim++) {
    double initial_value;
    if (nPoints >= 1) {
      initial_value = points[idim];
    } else {
      initial_value = FEM_BOUNDING_BOX_INVALID;
    }
    boundingBox[idim] = boundingBox[dim+idim] = initial_value;
  }

  for(int ipoint=1; ipoint<nPoints; ipoint++) {
    for (int idim=0; idim<dim; idim++) {
      double considered = points[ipoint*dim+idim];
      if (considered < boundingBox[idim]) {
	boundingBox[idim] = considered;
      } 
      if (considered > boundingBox[dim+idim]) {
	boundingBox[dim+idim] = considered;
      }
    }
  }
}
 
bool 
ParFUM_boundingBoxesCollide
///////////////////////////////////////////
/** \return true if two rectangular bounding boxes collide, false
 * otherwise.  The format for the bounding boxes is the same as that
 * returned by parFUM_findBoundingBox.
 *
 */
(
 /// number of spacial dimensions.
 int dim, 
 /// pointer to the first box
 double* box_a, 
 /// pointer to the second box
 double* box_b
 ) {

  for (int ii=0; ii<dim; ii++) {
    // return false if the extents indicate that no collision is possible
    // a_max < b_min || b_max < a_min
    // note that this operation is symmetric
    if ((box_a[ii] == FEM_BOUNDING_BOX_INVALID)
	||
	(box_b[ii] == FEM_BOUNDING_BOX_INVALID)
	||
	(box_a[ii+dim] < box_b[ii]) 
	|| 
	(box_b[ii+dim] < box_a[ii])) {
      return false;
    }
 }
  return true;
}
    
void
ParFUM_printBoundingBox
//////////////////////////////////////////////////////////
/** Prints a bounding box's extents
 *
 */
(
 int dim,
 double* box
 ) {
  printf("Lower bound: ");
  for (int ii=0; ii<dim; ii++) {
    printf("%lf ", box[ii]);
  }
  printf("\nUpper bound: ");
  
  for (int ii=dim; ii<2*dim; ii++) {
    printf("%lf ", box[ii]);
  }
  printf("\n");
}
