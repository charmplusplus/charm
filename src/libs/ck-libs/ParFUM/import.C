#include "import.h"

bool coord_leq(double *a, double* b, int dim);

void ParFUM_desharing(int meshid){
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_desharing"))->lookup(meshid,"ParFUM_desharing");
	mesh->clearSharedNodes();
}


void ParFUM_deghosting(int meshid){
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_deghosting"))->lookup(meshid,"ParFUM_deghosting");
	mesh->clearGhostNodes();
	mesh->clearGhostElems();
}

void ParFUM_recreateSharedNodes(int meshid, int dim, MPI_Comm newComm) {
  MPI_Comm comm = newComm;
  int rank, nParts;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nParts);
  // Shared data will be temporarily stored in the following structure
  int *sharedNodeCounts; // sharedCounts[i] = number of nodes shared with rank i
  int **sharedNodeLists; // sharedNodes[i] is the list of nodes shared with rank i
  // Initialize shared data
  sharedNodeCounts = (int *)malloc(nParts*sizeof(int));
  sharedNodeLists = (int **)malloc(nParts*sizeof(int *));
  for (int i=0; i<nParts; i++) {
    sharedNodeLists[i] = NULL;
    sharedNodeCounts[i] = 0;
  }
  // Get local node count and coordinates
  int numNodes;
  int coord_msg_tag=42, sharedlist_msg_tag=43;
  double *nodeCoords;
  numNodes = FEM_Mesh_get_length(meshid,FEM_NODE);
  nodeCoords = (double *)malloc(dim*numNodes*sizeof(double));

  FEM_Mesh_become_get(meshid);

  FEM_Mesh_data(meshid,FEM_NODE,FEM_COORD, nodeCoords, 0, numNodes,FEM_DOUBLE, dim);
  /*
  printf("Node Coords for rank %d \n",rank);
  for(int n=0;n<numNodes;n++){
    printf("%d -> ", n);
    for (int m-0; m<dim; m++) 
      printf("%.5lf %.5lf \n", nodeCoords[dim*n+m]);
  }
  */
  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) CkPrintf("Extracted node data...\n");

  // Begin exchange of node coordinates to determine shared nodes
  // FIX ME: compute bounding box, only exchange when bounding boxes collide
  for (int i=rank+1; i<nParts; i++) { //send nodeCoords to rank i
    //printf("[%d] Sending %d doubles to rank %d \n",rank,dim*numNodes,i);
    MPI_Send(nodeCoords, dim*numNodes, MPI_DOUBLE, i, coord_msg_tag, comm);
  }
  // Handle node coordinate-matching requests from other ranks

  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) CkPrintf("Exchanged node coords...\n");

  int *sorted_local_idxs = (int *)malloc(numNodes*sizeof(int));
  double *sorted_nodeCoords = (double *)malloc(dim*numNodes*sizeof(double));
  sortNodes(nodeCoords, sorted_nodeCoords, sorted_local_idxs, numNodes, dim);

  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) CkPrintf("Sorted node coords...\n");

  for (int i=0; i<rank; i++) {
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

    int *sorted_remote_idxs = (int *)malloc(recvNodeCount*sizeof(int));
    double *sorted_recvNodeCoords = (double *)malloc(length*sizeof(double));
    sortNodes(recvNodeCoords, sorted_recvNodeCoords, sorted_remote_idxs, recvNodeCount, dim);

    /* OLD SLOW WAY
    for (int j=0; j<numNodes; j++) {
      for (int k=0; k<recvNodeCount; k++) {
	if (coordEqual(&nodeCoords[j*dim], &recvNodeCoords[k*dim], dim)) {
	  localSharedNodes.push_back(j); 
	  remoteSharedNodes.push_back(k);
	  //printf("[%d] found local node %d to match with remote node %d \n",rank,j,k);
	  break;
	}
      }
    }
    */

    int j = 0; 
    int k = 0;
    while ((j<numNodes) && (k < recvNodeCount)) {
      if ((coordCompare(&sorted_nodeCoords[j*dim], &sorted_recvNodeCoords[k*dim], dim)) == 0) {
	localSharedNodes.push_back(sorted_local_idxs[j]); 
	remoteSharedNodes.push_back(sorted_remote_idxs[k]);
	j++; k++;
      }
      else if (coord_leq(&sorted_nodeCoords[j*dim], &sorted_recvNodeCoords[k*dim], dim)) {
	// local is less than remote
	j++;
      }
      else if (!coord_leq(&sorted_nodeCoords[j*dim], &sorted_recvNodeCoords[k*dim], dim)) { 
	// remote is less than local; remote is not present
	k++;
      }
    }
    
    /*
    if (localSharedNodes.size() > 0) {
      printf("%d has %d nodes shared with %d.\n", rank, localSharedNodes.size(), source);
    }
    */

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

  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) CkPrintf("Received node coords, send shared...\n");

  for (int i=rank+1; i<nParts; i++) {  // recv shared node lists
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

  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) CkPrintf("Received shared...\n");

  // IMPLEMENT ME: use sharedNodeLists and sharedNodeCounts to move shared node data 
  // to IDXL
  FEM_Mesh *mesh = (FEM_chunk::get("ParFUM_recreateSharedNodes"))->lookup(meshid,"ParFUM_recreateSharedNodes");
  IDXL_Side &shared = mesh->node.shared;
  
  for(int i=0;i<nParts;i++){
    if(i == rank)
      continue;
    if(sharedNodeCounts[i] != 0){
      IDXL_List &list = shared.addList(i);
      for(int j=0;j<sharedNodeCounts[i];j++){
	list.push_back(sharedNodeLists[i][j]);
      }
    }
  }

  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) CkPrintf("Recreation of shared nodes complete...\n");

  //printf("After recreating shared nodes %d \n",rank);
  //shared.print();
	
  // Clean up
  free(nodeCoords);
  free(sharedNodeCounts);
  for (int i=0; i<nParts; i++) {
    if (sharedNodeLists[i])
      free(sharedNodeLists[i]);
  }
  free(sharedNodeLists);
  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) CkPrintf("All cleaned up.\n");
}

void ParFUM_createComm(int meshid, int dim, MPI_Comm comm)
{
  int rank, comm_size;
  ParFUM_desharing(meshid);
  ParFUM_deghosting(meshid);
  //MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);
  if (rank==0) CkPrintf("Recreating shared nodes...\n");
  ParFUM_recreateSharedNodes(meshid, dim, comm);
  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) CkPrintf("Generating global node numbers...\n");
  ParFUM_generateGlobalNodeNumbers(meshid, comm);
  FEM_Mesh *mesh = (FEM_chunk::get("ParFUM_recreateSharedNodes"))->lookup(meshid,"ParFUM_recreateSharedNodes");
  //MPI_Barrier(MPI_COMM_WORLD);
  
  if (rank==0) CkPrintf("Gathering ghost data...\n");
  struct ghostdata *gdata;
  if(rank == 0){
    gdata = gatherGhosts();
  }else{
    gdata = new ghostdata;
  }
  MPI_Bcast_pup(*gdata,0,comm);
  if (rank==0) CkPrintf("Making ghosts...\n");
  makeGhosts(mesh,comm,0,gdata->numLayers,gdata->layers);
  //MPI_Barrier(MPI_COMM_WORLD);
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

void qsort(double *nodes, int *ids, int dim, int first, int last);
void sortNodes(double *nodes, double *sorted_nodes, int *sorted_ids, int numNodes, int dim)
{
  
  for(int i=0; i<numNodes; i++) {
    sorted_ids[i] = i;
    for(int j=0; j<dim; j++) {
      sorted_nodes[i*dim+j] = nodes[i*dim+j];
    }
  }
  /*
  for (int i=0; i<10; i++) {
    printf("[%d] %6.4f,%6.4f,%6.4f is at %d\n", i, sorted_nodes[i*dim], sorted_nodes[i*dim+1], sorted_nodes[i*dim+2], sorted_ids[i]);
  }
  */
  qsort(sorted_nodes, sorted_ids, dim, 0, numNodes-1);
  /*
  for (int i=0; i<20; i++) {
    printf("[%d] %6.4f,%6.4f,%6.4f is at %d\n", i, sorted_nodes[i*dim], sorted_nodes[i*dim+1], sorted_nodes[i*dim+2], sorted_ids[i]);
  }
  */
}

void merge(double *nodes, int *ids, int dim, int first, int mid, int last);
void qsort(double *nodes, int *ids, int dim, int first, int last)
{
  if (first>=last) return;
  else if (first==last-1) {
    if (!coord_leq(&(nodes[first*dim]), &(nodes[last*dim]), dim)) {
      int tmpId=ids[first];
      ids[first] = ids[last];
      ids[last] = tmpId;
      double tmpCoord;
      for (int i=0; i<dim; i++) {
	tmpCoord = nodes[first*dim+i];
	nodes[first*dim+i] = nodes[last*dim+i];
	nodes[last*dim+i] = tmpCoord;
      }
    }
  }
  else {
    qsort(nodes, ids, dim, first, first+((last-first)/2));
    qsort(nodes, ids, dim, first+((last-first)/2)+1, last);
    merge(nodes, ids, dim, first, first+((last-first)/2)+1, last);
  }
}

void merge(double *nodes, int *ids, int dim, int first, int mid, int last)
{
  int rover1=first, rover2=mid;
  double *tmpCoords = (double *)malloc(dim*(last-first+1)*sizeof(double));
  int *tmpIds = (int *)malloc((last-first+1)*sizeof(int));
  int pos = 0;
  while ((rover1<mid) && (rover2<=last)) {
    if (!coord_leq(&(nodes[rover1*dim]), &(nodes[rover2*dim]), dim)) {
      tmpIds[pos] = ids[rover2];
      for (int i=0; i<dim; i++) {
	tmpCoords[pos*dim+i] = nodes[rover2*dim+i];
      }
      rover2++;
    }
    else if (coord_leq(&(nodes[rover1*dim]), &(nodes[rover2*dim]), dim)) {
      tmpIds[pos] = ids[rover1];
      for (int i=0; i<dim; i++) {
	tmpCoords[pos*dim+i] = nodes[rover1*dim+i];
      }
      rover1++;
    }
    else {
      CkPrintf("import.C: merge: ERROR: found identical nodes on single partition!\n");
    }
    pos++;
  }

  if (rover1 < mid) {
    while (rover1 < mid) {
      tmpIds[pos] = ids[rover1];
      for (int i=0; i<dim; i++) {
	tmpCoords[pos*dim+i] = nodes[rover1*dim+i];
      }
      rover1++;
      pos++;
    }
  }
  else if (rover2 <= last) {
    while (rover2 <= last) {
      tmpIds[pos] = ids[rover2];
      for (int i=0; i<dim; i++) {
	tmpCoords[pos*dim+i] = nodes[rover2*dim+i];
      }
      rover2++;
      pos++;
    }
  }

  for (int i=first; i<=last; i++) {
    ids[i] = tmpIds[i-first];
    for (int j=0; j<dim; j++) {
      nodes[i*dim+j] = tmpCoords[(i-first)*dim+j];
    }
  }
  free(tmpCoords);
  free(tmpIds);
}

// this is a strictly ordering floating point less-than-or-equal-to operator, faster than the coord_compare
bool coord_leq(double *a, double* b, int dim) {
  int d=0;
  while (d<dim) {
    if (a[d] < b[d]) return true;
    else if (a[d] > b[d]) return false;
    else d++;
  }
  // a and b are identical
  return true;
}
