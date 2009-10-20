#include <stdlib.h>
#include "ParFUM.h"
#include "ParFUM_internals.h"
/***
 * ParFUM_generateGlobalNodeNumbers regenerates global numbers for nodes
 * for the specified mesh.
 * It is a collective call.
 * It assumes that the mesh already has an idxl list for shared nodes
 **/
void addToLists(int *listIndex,CkVec<CkVec<int> > &lists,int chunk,int node);


void ParFUM_generateGlobalNodeNumbers(int fem_mesh, MPI_Comm comm){
	int myID;
	int numberChunks;
	//MPI_Barrier(comm);
	MPI_Comm_rank(comm,&myID);
	MPI_Comm_size(comm,&numberChunks);
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_renumberMesh"))->lookup(fem_mesh,"ParFUM_renumberMesh");
	
	FEM_Node *node = &(mesh->node);
	FEM_IndexAttribute *globalAttr = (FEM_IndexAttribute *)node->lookup(FEM_GLOBALNO,"ParFUM_renumberMesh");
	AllocTable2d<int> &globalTable = globalAttr->get();
	
	int *primaryChunk = new int[node->size()];
	IDXL_Side *shared = &(node->shared);
	
	for(int i=0;i<node->size();i++){
		//initialize the global number for all nodes to -1
		globalTable[i][0]=-1;
		
		// this is the default case, means that this node is not shared
		primaryChunk[i] = -1; 
	}
	
	//mark the primaryChunk array such that if it is not shared a node is marked -1
	//else primaryChunk[i] = chunk that owns that node 
	//A node is owned by the chunk with the lowest index that owns it
	for(int j=0;j<shared->size();j++){
		//list of nodes I share with another chunk
		const IDXL_List &list = shared->getLocalList(j);
		if(list.getDest() > myID){
			//the nodes shared with this chunk are owned by me or some other chunk
			for(int k=0;k<list.size();k++){
				int sharedNode = list[k];
				if(primaryChunk[sharedNode] == -1){
					//this node is shared but I might be the owner
					primaryChunk[sharedNode] = myID;
				}
			}
		}else{
			//the nodes shared with this chunk are owned by this chunk or some other chunk
			//but not me
			for(int k=0;k<list.size();k++){
				int sharedNode = list[k];
				if(primaryChunk[sharedNode] == -1){
					primaryChunk[sharedNode] = list.getDest();
				}else{
					if(primaryChunk[sharedNode] > list.getDest()){
						primaryChunk[sharedNode] = list.getDest();
					}
				}
			}
		}
	}
	//count number of nodes owned by me
	int numberOwned = 0;
	for(int i=0;i<node->size();i++){
		if(primaryChunk[i] == -1 || primaryChunk[i] == myID){
			numberOwned++;
		}
	}
	// perform a prefix sum of nodes owned by each processor
	int globalUniqueNodes;
	MPI_Scan((void*)&numberOwned , (void* ) &globalUniqueNodes, 1, MPI_INT, MPI_SUM, comm) ;

	//assign global numbers to nodes owned by me
	int firstGlobalNumber = globalUniqueNodes - numberOwned;
	int counter = firstGlobalNumber;
	for(int i=0;i<node->size();i++){
		if(primaryChunk[i] == -1 || primaryChunk[i] == myID){
			globalTable[i][0] = counter;
			counter++;
		}
	}
	CkAssert(counter == globalUniqueNodes);
	
	// make lists of nodes whose global numbers need to be sent
	// or received from other chunks
	
	int *sendListIndex  = new int[numberChunks];
	CkVec<CkVec<int> > listsToSend; 
	
	int *recvListIndex  = new int[numberChunks];
	CkVec<CkVec<int> > listsToRecv;

	for(int j=0;j<numberChunks;j++){
		sendListIndex[j] = recvListIndex[j] = -1;
	}

	//we have to use the IDXL to create the lists of nodes whose 
	//global numbers must be sent or received. This is necessary
	//to establish the correspondence between nodes on different chunks
	for(int j=0;j<shared->size();j++){
		const IDXL_List &list = shared->getLocalList(j);
		
		if(list.getDest() > myID){
			//i might have to send global numbers to nodes on this chunk
			for(int k=0;k<list.size();k++){
				int sharedNode = list[k];
				if(primaryChunk[sharedNode] == myID){
					//i am the primary node .. so i shall have to send the global number of the nodes
					addToLists(sendListIndex,listsToSend,list.getDest(),sharedNode);
				}
			}
		}else{
			//i might have to recv global numbers from nodes on this chunk
			for(int k=0;k<list.size();k++){
				int sharedNode = list[k];
				if(primaryChunk[sharedNode] == list.getDest()){
					//list.getDest() is the primary chunk for this node.. so i shall have to recv
					//the global number from this chunk
					addToLists(recvListIndex,listsToRecv,primaryChunk[sharedNode],sharedNode);
				}
			}
		}
	}

	
	//count the number of IRecvs we are going to post
	int numberRequests=0;
	for(int j=0;j<numberChunks;j++){
		if(recvListIndex[j] != -1){
			numberRequests++;
		}
	}
	MPI_Request *req = new MPI_Request[numberRequests];
	MPI_Status *status = new MPI_Status[numberRequests];
	int **recvdGlobalNumbers = new int *[numberRequests];

	int countRequests = 0;
	for(int j=0;j<numberChunks;j++){
		if(recvListIndex[j] != -1){
			// for each chunk that sends us global numbers, post a irecv
			
			CkVec<int> &recvList = listsToRecv[recvListIndex[j]];
			recvdGlobalNumbers[countRequests] = new int[recvList.size()];
			MPI_Irecv((void*)recvdGlobalNumbers[countRequests] , recvList.size(), MPI_INT, j, 42, comm, &req[countRequests]) ;
			countRequests++;
		}
	}

	for(int j=0;j<numberChunks;j++){
		if(sendListIndex[j] != -1){
			//for each chunk to which we send global numbers, do a mpi_send
			CkVec<int> &sendList = listsToSend[sendListIndex[j]];
			int *sendGlobalNumbers = new int[sendList.size()];
			for(int k=0;k<sendList.size();k++){
				sendGlobalNumbers[k] = globalTable[sendList[k]][0];
			}
			MPI_Send((void*)sendGlobalNumbers , sendList.size(), MPI_INT, j, 42, comm); 
			delete [] sendGlobalNumbers;
		}
	}

	MPI_Waitall(numberRequests, req, status);
	//should check statuses .. but am too lazy to do that
	
	//now read the received global numbers and set them to the correct node
	countRequests = 0;
	for(int j=0;j<numberChunks;j++){
		if(recvListIndex[j] != -1){
			CkVec<int> &recvList = listsToRecv[recvListIndex[j]];
			for(int k=0;k<recvList.size();k++){
				int recvNode = recvList[k];
				int recvGlobalNumber = recvdGlobalNumbers[countRequests][k] ;
				CkAssert(globalTable[recvNode][0] == -1);
				globalTable[recvNode][0] = recvGlobalNumber;
			}
			countRequests++;
		}
	}
	
/*	shared->print(NULL);
	
	char name[100];
	sprintf(name,"global.%d",myID);
	FILE *fp = fopen(name,"w");
	for(int i=0;i<node->size();i++){
		fprintf(fp,"[%d] %d: %d \n",myID,i,globalTable[i][0]);
		CkAssert(globalTable[i][0] >= 0);
	}
	fclose(fp);*/
				
	delete []primaryChunk;
	delete [] sendListIndex;
	delete [] recvListIndex;
}

//function to add a node to the list for a particular node
void addToLists(int *listIndex,CkVec<CkVec<int> > &lists,int chunk,int node){
	if(listIndex[chunk] == -1){
		CkVec<int> tmpVec;
		int index = lists.push_back_v(tmpVec);
		listIndex[chunk] = index;		
	}
	CkVec<int> &vec = lists[listIndex[chunk]];
	vec.push_back(node);
}
