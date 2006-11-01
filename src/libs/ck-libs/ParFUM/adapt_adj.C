/* Adaptivity Adjacencies: element-to-element adjacencies for use by
   adaptivity codes only.  Adaptivity codes should keep them
   up-to-date for each mesh modification primitive.
   
   Created 11 Sept 2006 - Terry L. Wilmarth
*/
#include "adapt_adj.h"
using namespace std;

int nodeSetMap2d_tri[3][2] = {{0,1},{1,2},{2,0}};
int nodeSetMap2d_quad[4][2] = {{0,1},{1,2},{2,3},{3,0}};
int nodeSetMap3d_tet[4][3] = {{0,1,2},{1,0,3},{1,3,2},{0,2,3}};
int nodeSetMap3d_hex[6][4] = {{0,1,2,3},{1,5,6,2},{2,6,7,3},{3,7,4,0},{0,4,5,1},{5,4,6,7}};

inline void addSharedNodeData(int node,const IDXL_Rec *sharedChunkList,adjNode *adaptAdjTable){
	adaptAdjTable[node].numSharedPartitions = sharedChunkList->getShared();
  adaptAdjTable[node].sharedWithPartition = new int [sharedChunkList->getShared()];
  adaptAdjTable[node].sharedWithLocalIdx = new int [sharedChunkList->getShared()];
  for(int j=0;j<sharedChunkList->getShared();j++){
    int sharedChunk = sharedChunkList->getChk(j);
    int sharedIdx = sharedChunkList->getIdx(j);
    adaptAdjTable[node].sharedWithPartition[j] = sharedChunk;
    adaptAdjTable[node].sharedWithLocalIdx[j] = sharedIdx;
  }
}



inline void addElementNodeSetData(int elem,const int *conn,int numAdjElems,const int nodesPerElem,int nodeSetSize,int **nodeSetMap,adjNode *adaptAdjTable){
   for (int j=0; j<numAdjElems; j++) { // There is one nodeSet per neighbor element
     adjElem *e = new adjElem(nodeSetSize);
     e->nodeSetID = j;
     for (int k=0; k<nodeSetSize; k++) { // Build the nodeSet for an element pairing
       e->nodeSet[k] = conn[elem*nodesPerElem+nodeSetMap[j][k]];
     }
     // Add this element-nodeSet pair to the table at the min nodeID in the nodeSet
     e->nodeSet.quickSort();
     int minNode = e->nodeSet[0];
     e->elemID = elem;
     e->next = adaptAdjTable[minNode].adjElemList;
     adaptAdjTable[minNode].adjElemList = e;
     adaptAdjTable[minNode].adjElemCount++;
   }
}


//Look for an adjElem (rover->next) whose nodeSet matches that specified in searchForNodeSet in the link list following
//adjStart and return rover such that rover->next matches
//with the searchForNodeSet. It also checks that the elemID of the element being searched 
//does not match with that of searchForElemID.
//*found is set to 1 if match is found .. else to 0
inline adjElem *searchAdjElemInList(adjElem *adjStart,int *searchForNodeSet,int nodeSetSize,int searchForElemID,int *found){
  adjElem *rover = adjStart; // compare rover->next with adjStart
  *found = 0;
  while (rover->next != NULL) {
    if (rover->next->elemID != searchForElemID) {
      *found = 1; // found an element that is not myself, possibly a match
      for (int j=0; j<nodeSetSize; j++) {
        if (rover->next->nodeSet[j] != searchForNodeSet[j]) {
          *found = 0; // No, the nodeSets dont match
          break;
        }
      }
    }
    if (*found) {
      break; // We have found a nodeSet that matches adjStart
    }else {
      rover = rover->next; // Keep looking in adjElemList for matching nodeSet
    }
  }
  return rover;
}

/** Create Adaptivity Adjacencies for elemType; dimension inferred. */
void CreateAdaptAdjacencies(int meshid, int elemType)
{
  // Need to derive all of these from elemType;
  int myRank,numChunks;
  int numAdjElems;
  int nodeSetSize; // number of nodes shared by two adjacent elems

  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  MPI_Comm_size(MPI_COMM_WORLD,&numChunks);
  FEM_Mesh *mesh = FEM_chunk::get("CreateAdaptAdjacencies")->lookup(meshid,"CreateAdaptAdjacencies");
  FEM_Elem *elem = (FEM_Elem *)mesh->lookup(FEM_ELEM+elemType,"CreateAdaptAdjacencies");
  FEM_Node *node = (FEM_Node *)mesh->lookup(FEM_NODE,"CreateAdaptAdjacencies");
  const int numElems = elem->size();
  const int numNodes = node->size();
  const int nodesPerElem = (elem->getConn()).width();
  assert(node->getCoord()!= NULL);
  const int dim = (node->getCoord())->getWidth();
  assert(dim == 2|| dim == 3);

  
  // A nodeSet is a set of nodes that defines a pairing of two adjacent elements;
  // For example, in 2D triangle meshes, the nodeSet is the nodes of an edge between
  // two elements.
  // The nodeSetMap is an ordering of element-local node IDs that specifies all 
  // possible nodeSets for a particular element type
  int **nodeSetMap;
  guessElementShape(dim,nodesPerElem,&numAdjElems,&nodeSetSize,&nodeSetMap);
	CkAssert(nodeSetSize <= MAX_NODESET_SIZE);
  

  // Create the adaptAdj array associated with the new FEM attribute FEM_ADAPT_ADJ
  // This array will be populated by this function, then set on the FEM mesh
  adaptAdj *adaptAdjacencies;
  adaptAdjacencies = new adaptAdj[numElems*numAdjElems];

  // Init adaptAdj array to at least have -1 as partID signifying no neighbor
  for (int i=0; i<numElems*numAdjElems; i++) {
    adaptAdjacencies[i].partID = -1;
  }

  // Create an array of size equal to the number of local nodes, each entry has a
  // pair:  (shared partitions, list of elem-nodeSet pairs) Call this adaptAdjTable
  adjNode *adaptAdjTable;
  adaptAdjTable = new adjNode[numNodes];

  // Loop through shared node list and add the partition ids to adaptAdjTable.
  for(int i=0;i<numNodes;i++){
    if(node->is_valid(i)){
      const IDXL_Rec *sharedChunkList = node->shared.getRec(i);
      if(sharedChunkList != NULL){
        addSharedNodeData(i,sharedChunkList,adaptAdjTable);
      }
    }
  }

  // Pull out conn for elems of elemType
  const int *conn = (elem->getConn()).getData();
  
  for (int i=0; i<numElems; i++) { // Add each element-nodeSet pair to the table
    if(elem->is_valid(i)){
      addElementNodeSetData(i,conn,numAdjElems,nodesPerElem,nodeSetSize,nodeSetMap,adaptAdjTable);
    }
  }

  for (int i=0; i<numNodes; i++) { 
    // For each node, match up incident elements
    // Each adjacency between two elements is represented by two adjElems
    // We try to match those up
    if(node->is_valid(i)){
      CkAssert(adaptAdjTable[i].adjElemList != NULL);
      adjElem *adjStart = adaptAdjTable[i].adjElemList;
      adjElem *preStart = adjStart; //pointer before adjStart so that we can delete adjStart
                                    //Note: as long as adjStart is the first element in adjElemList
                                    //preStart = adjStart. After that preStart->next = adjStart
      while (adjStart != NULL) { //each entry in the adjElemList of a node 
        int found = 0; 
        //AdjStart represents an adjacency between two elements
        //We search for the other adjElem corresponding to that adjancency:
        //Look for an entry in adjElemList after adjStart such that 
        //the nodeset of that entry and adjStart match but they 
        //do not belong to the same element. 
        adjElem *rover = searchAdjElemInList(adjStart,adjStart->nodeSet.getVec(),nodeSetSize,adjStart->elemID,&found); 
				
        if (found) {
          // We found an adjacent element for adjStart->elemID
          
          // Set adjacency of adjStart->elemID corresponding to nodeSet to 
          // rover->next->elemID, and vice versa
          // Store adjacency info in adaptAdjacency of each one and use nodeSetID to index into 
          // adaptAdjacency
          adaptAdjacencies[numAdjElems*adjStart->elemID+adjStart->nodeSetID] = adaptAdj(myRank,rover->next->elemID,elemType);
          adaptAdjacencies[numAdjElems*rover->next->elemID+rover->next->nodeSetID] = adaptAdj(myRank,adjStart->elemID,elemType);
          // Remove both elem-nodeSet pairs from the list
          adjElem *tmp = rover->next;
          rover->next = rover->next->next;
          delete tmp;
          if (preStart == adjStart) { // adjStart was at the start of adjElemList
            adaptAdjTable[i].adjElemList = adjStart->next;
            delete adjStart; 
            adjStart = preStart = adaptAdjTable[i].adjElemList;
          }else { //adjStart was not at the start
            preStart->next = adjStart->next;
            delete adjStart;
            adjStart = preStart->next;
          }
        }else { 
          // No match for adjStart was found in adjElemList
          // It means that either adjStart is on the domain boundary 
          // or it is on the chunk boundary and its neighbor is on another VP
          // Move adjStart to next entry in adjElemList
          if (adjStart != preStart){
            preStart = preStart->next;
          }
          adjStart = adjStart->next;
        }
      }
    }
  }

  // Now all elements' local adjacencies are set; remainder in table are 
  // nodeSets shared with other chunks or nodeSets on domain boundary
	
	MPI_Barrier(MPI_COMM_WORLD);
	MSA1DREQLIST *requestTable;
	if(myRank == 0){
		requestTable = new MSA1DREQLIST(numChunks,numChunks);
	}else{
		requestTable = new MSA1DREQLIST;
	}
	MPI_Bcast_pup(*requestTable,0,MPI_COMM_WORLD);
	requestTable->enroll(numChunks);
	requestTable->sync();

//	CkVec<adjRequest *> requestVec; // This vector stores the requests made by this chunk
	
  for (int i=0; i<numNodes; i++) { 
    // For each node, examine the remaining  entries in adjElemList
    if(node->is_valid(i)){
      if (adaptAdjTable[i].adjElemList!=NULL) {
				adjElem *adjStart = adaptAdjTable[i].adjElemList;
        while (adjStart !=NULL) {
          // create and empty set, commonSharedChunks
					std::set<int> commonSharedChunks;
          for (int j=0; j<nodeSetSize; j++) {
          // look up sharedWithPartitions for node: 
          //    adaptAdjTable[i]->adjElemList->nodeset[j]
          // intersect with commonSharedChunks
						int sharedNode = adjStart->nodeSet[j];
						adjNode *sharedNodeAdj = &adaptAdjTable[sharedNode];
						std::set<int> sharedChunks(sharedNodeAdj->sharedWithPartition,
								sharedNodeAdj->sharedWithPartition+sharedNodeAdj->numSharedPartitions);
						if(j == 0){
							commonSharedChunks = sharedChunks;
						}else{
							std::set<int> tmpIntersect;
							set_difference(commonSharedChunks.begin(), commonSharedChunks.end(), 
									sharedChunks.begin(), sharedChunks.end(),inserter(tmpIntersect, tmpIntersect.begin()));
							commonSharedChunks = tmpIntersect;							
						}
          }
					// At this point commonSharedChunks contains the list of chunks with which
					// the element pointed by adjStart might be shared
					int numCommonSharedChunks = commonSharedChunks.size();
					if(numCommonSharedChunks > 0){
					//adjStart is possibly shared with these chunks. It is shared across 
					//the adjStart->nodeSet set of nodes.
						adjRequest *adjRequestList = new adjRequest[numCommonSharedChunks];
						//Translate the nodes in the nodeSet into the index in the idxl list for each chunk
						//in the commonSharedChunks
						for(int j=0;j<nodeSetSize;j++){
							int sharedNode = adjStart->nodeSet[j];
							const IDXL_Rec *recSharedNode = node->shared.getRec(sharedNode);
							int countChunk=0;
							for(std::set<int>::iterator chunkIterator = commonSharedChunks.begin();chunkIterator != commonSharedChunks.end();chunkIterator++){
								countChunk++;
								if(j == 0){
									// if this is the first node we need to initialize the adjRequestobject
									new (&adjRequestList[countChunk]) adjRequest(adjStart->elemID,myRank,adjStart->nodeSetID,elemType);
								}
								int chunk = *chunkIterator;
								int sharedNodeIdx=-1; // index of sharedNode in the idxl list of chunk
								//search for this chunk in the list of nodes shared by this node
								for(int k=0;k<recSharedNode->getShared();k++){
									if(recSharedNode->getChk(k) == chunk){
										//found the correct chunk
										sharedNodeIdx = recSharedNode->getIdx(k);
										break;
									}
								}
								CkAssert(sharedNodeIdx != -1);
								//The index of sharedNode in the index list of chunk has been found.
								//this needs to be saved in the corresponding translatedNodeSet
								adjRequestList[countChunk].translatedNodeSet[j] = sharedNodeIdx;
							}
						}
						//Now the nodeNumbers for the nodeSets that might be along chunk boundaries have
						//been translated into idxl indices between the two chunks
						//We now need to write the requests into the msa array requestTable
						//WARNING: This depends on sets getting enumerated in the same way always
						//Might not be true
						int countChunk=0;
						for(std::set<int>::iterator chunkIterator = commonSharedChunks.begin();chunkIterator != commonSharedChunks.end();chunkIterator++){
							countChunk++;
							int chunk = *chunkIterator;
							(*requestTable).accumulate(chunk,adjRequestList[countChunk]);
//							requestVec.push_back(&adjRequestList[countChunk]);
						}
						delete [] adjRequestList;
					}
					
					adjStart = adjStart->next;
        }
      }
    }
  }
	requestTable->sync();
  // look up request table, put answer in reply table 
  // receive: for local elemID, the adjacency on this set of shared indices is (remote elemID, remote partition ID, remote elem type)
  // add adjacencies to local table
  // lots of error checking :)
	
	//Look at each request that in the requestTable for this chunk
	//Put the data for the requests in our own table and then create replies
	CkVec<adjRequest> *receivedRequestVec = requestTable->get(myRank).vec;
	for(int i=0;i<receivedRequestVec->length();i++){		
		adjRequest &receivedRequest = (*receivedRequestVec)[i];
		const IDXL_List &sharedNodeList = node->shared.getList(receivedRequest.chunkID);
		CkVec<int> sharedNodes(nodeSetSize);
		//Translate all the nodes in the nodeSet of the request
		for(int j=0;j<nodeSetSize;j++){
			int sharedNode = sharedNodeList[receivedRequest.translatedNodeSet[j]];
			sharedNodes.push_back(sharedNode);
		}
		sharedNodes.quickSort();
		//We need to find the matching nodeset for the nodeset in the request
		//We look it up in the adaptAdjTable for the minimum node on this chunk
		adjNode *minNode = &adaptAdjTable[sharedNodes[0]];
		int found=0;
		//search for the nodeSet in the list of adjacencies around minNode
		adjElem *rover =  searchAdjElemInList(minNode->adjElemList,sharedNodes.getVec(),nodeSetSize,-1,&found);
		if(found){
			 //we have found a matching adjElem for the requested nodeset
			 //we shall set the adjacency correctly in the adjacency Table for the 
			 //elemID in the found adjElem. We need to send a reply to the requesting chunk
			 int matchingElemID = rover->next->elemID;
			 adaptAdj *matchingAdaptAdj = &adaptAdjacencies[matchingElemID*numAdjElems + rover->next->nodeSetID];
			 matchingAdaptAdj->partID = receivedRequest.chunkID;
			 matchingAdaptAdj->localID = receivedRequest.elemID;
			 matchingAdaptAdj->elemType = receivedRequest.elemType;

			 adjReply reply;
			 //Set requesting data in reply to that in receivedRequest
			 //Put in data from rover->next into the replyElem portion of data
			 //Write into the replyTable
		}else{
			//we have no matching nodeset for this request.. hopefully some other chunk does
			//we canignore this request
		}
	}

	//Once the replies are back
	//Loop through each reply and update the adaptAdjacencies for each element in the reply
	//
	//Register the adaptAdjacency with ParFUM
}

// Access functions

/** Look up elemID in elemType array, access edgeFaceID-th adaptAdj. */
adaptAdj *GetAdaptAdj(int elemID, int elemType, int edgeFaceID)
{
}

/** Look up elemID in elemType array, calculate edgeFaceID from vertexList (with
    GetEdgeFace below), and access edgeFaceID-th adaptAdj with GetAdaptAdj above. */
adaptAdj *GetAdaptAdj(int elemID, int elemType, int *vertexList)
{
}

/** Look up elemID in elemType array and determine the set of vertices
    associated with the edge or face represented by edgeFaceID. */
void GetVertices(int elemID, int elemType, int edgeFaceID, int *vertexList)
{
}

/** Look up elemID in elemType array and determine the edge or face ID specified by
    the set of vertices in vertexList. */
int GetEdgeFace(int elemID, int elemType, int *vertexList)
{
}

// Update functions
/** Look up elemID in elemType array and set the adjacency on edgeFaceID to nbr. */
void SetAdaptAdj(int elemID, int elemType, int edgeFaceID, adaptAdj nbr)
{
}

//given the dimensions and nodes per element guess whether the element 
// is a triangle, quad, tet or hex. At the moment these are the 4 shapes
// that are handled
  
void guessElementShape(int dim,int nodesPerElem,int *numAdjElems,int *nodeSetSize,int ***nodeSetMap){
  switch(dim){
    case 2:
          {
          //2 dimension
            switch(nodesPerElem){
              case 3:
                //Triangles
                *numAdjElems = 3;
                *nodeSetSize = 2;
                *nodeSetMap = (int **)nodeSetMap2d_tri;
                break;
              case 4:
                //quads
                *numAdjElems = 4;
                *nodeSetSize = 2;
                *nodeSetMap = (int **)nodeSetMap2d_quad;
                break;
            }
          }
          break;
    case 3:
          {
          //3 dimension
            switch(nodesPerElem){
              case 4:
                //Tetrahedra
                *numAdjElems = 4;
                *nodeSetSize = 3;
                *nodeSetMap = (int **)nodeSetMap3d_tet;
                break;
              case 6:
                //Hexahedra
                *numAdjElems = 6;
                *nodeSetSize = 4;
                *nodeSetMap = (int **)nodeSetMap3d_hex;
                break;
            }
          }
          break;
  }
}

