#include "import.h"
void ParFUM_desharing(int meshid){
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_desharing"))->getMesh("ParFUM_desharing");
	mesh->clearSharedNodes();
}


void ParFUM_deghosting(int meshid){
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_deghosting"))->getMesh("ParFUM_deghosting");
	mesh->clearGhostNodes();
	mesh->clearGhostElems();
}

void ParFUM_recreateSharedNodes(int meshid) {
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
  nodeCoords = (double *)malloc(3*numNodes*sizeof(double));
  FEM_Mesh_data(meshid,FEM_NODE,FEM_COORD, nodeCoords, 0, numNodes,FEM_DOUBLE, 3);

/*	printf("Node Coords for rank %d \n",rank);
	for(int n=0;n<numNodes;n++){
		printf("%d -> %.5lf %.5lf %.5lf \n",n,nodeCoords[3*n+0],nodeCoords[3*n+1],nodeCoords[3*n+2]);
	}*/
  
  // Begin exchange of node coordinates to determine shared nodes
  // FIX ME: compute bounding box, only exchange when bounding boxes collide
  for (int i=rank+1; i<comm_size; i++) { //send nodeCoords to rank i
    printf("[%d] Sending %d doubles to rank %d \n",rank,3*numNodes,i);
    MPI_Send(nodeCoords, 3*numNodes, MPI_DOUBLE, i, coord_msg_tag, comm);
  }
  // Handle node coordinate-matching requests from other ranks
  for (int i=0; i<rank; i++) {
    std::vector<int> remoteSharedNodes, localSharedNodes;
    double *recvNodeCoords;
    MPI_Status status;
    int source, length;
    // Probe for a coordinate message from any source; extract source and msg length
    MPI_Probe(MPI_ANY_SOURCE, coord_msg_tag, comm, &status);
    source = status.MPI_SOURCE;
    length = status.MPI_LENGTH/sizeof(double);
    printf("[%d] Receiving %d doubles from rank %d \n",rank,length,i);
    // Receive whatever data was available according to probe
    recvNodeCoords = (double *)malloc(length*sizeof(double));
    MPI_Recv((void*)recvNodeCoords, length, MPI_DOUBLE, source, 
	      coord_msg_tag, comm, &status);
    // Match coords between local nodes and received coords
    // FIX ME: this is the dumb super-slow brute force algorithm
    int recvNodeCount = length/3;
    for (int j=0; j<numNodes; j++) {
      for (int k=0; k<recvNodeCount; k++) {
	if (coordEqual(&nodeCoords[j*3], &recvNodeCoords[k*3])) {
	  localSharedNodes.push_back(j); 
	  remoteSharedNodes.push_back(k);
	  printf("[%d] found local node %d to match with remote node %d \n",rank,j,k);
	  break;
	}
      }
    }
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
  for (int i=rank+1; i<comm_size; i++) {  // recv shared node lists
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
  // IMPLEMENT ME: use sharedNodeLists and sharedNodeCounts to move shared node data 
  // to IDXL
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_recreateSharedNodes"))->getMesh("ParFUM_recreateSharedNodes");
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
	printf("After recreating shared nodes %d \n",rank);
	shared.print();
	
  // Clean up
  free(nodeCoords);
  free(sharedNodeCounts);
  for (int i=0; i<comm_size; i++) {
    if (sharedNodeLists[i])
      free(sharedNodeLists[i]);
  }
  free(sharedNodeLists);
}

void ParFUM_createComm(int meshid)
{
  ParFUM_desharing(meshid);
  ParFUM_deghosting(meshid);
	MPI_Barrier(MPI_COMM_WORLD);
  ParFUM_recreateSharedNodes(meshid);
	MPI_Barrier(MPI_COMM_WORLD);
	ParFUM_generateGlobalNodeNumbers(meshid);
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_recreateSharedNodes"))->getMesh("ParFUM_recreateSharedNodes");
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
