/* Adaptivity Adjacencies: element-to-element adjacencies for use by
   adaptivity codes only.  Adaptivity codes should keep them
   up-to-date for each mesh modification primitive.
   
   Created 11 Sept 2006 - Terry L. Wilmarth
*/
#include "ParFUM.h"
#include "ParFUM_internals.h"
using namespace std;

int nodeSetMap2d_tri[3][2] = {{0,1},{1,2},{2,0}};
int nodeSetMap2d_quad[4][2] = {{0,1},{1,2},{2,3},{3,0}};
int nodeSetMap3d_tet[4][3] = {{0,1,2},{1,0,3},{1,3,2},{0,2,3}};
int nodeSetMap3d_hex[6][4] = {{0,1,2,3},{1,5,6,2},{2,6,7,3},{3,7,4,0},{0,4,5,1},{5,4,6,7}};

int nodeSetMap2d_cohquad[2][2] = {{0,1},{2,3}}; // Cohesive for 2D triangles
int nodeSetMap3d_cohprism[2][3] = {{0,1,2},{3,4,5}}; // Cohesive for 3D tets

inline void addSharedNodeData(int node,const IDXL_Rec *sharedChunkList,
            adjNode *adaptAdjTable){
  adaptAdjTable[node].numSharedPartitions = sharedChunkList->getShared();
  adaptAdjTable[node].sharedWithPartition = 
    new int [sharedChunkList->getShared()];
  adaptAdjTable[node].sharedWithLocalIdx = 
    new int [sharedChunkList->getShared()];
  for(int j=0;j<sharedChunkList->getShared();j++){
    int sharedChunk = sharedChunkList->getChk(j);
    int sharedIdx = sharedChunkList->getIdx(j);
    adaptAdjTable[node].sharedWithPartition[j] = sharedChunk;
    adaptAdjTable[node].sharedWithLocalIdx[j] = sharedIdx;
  }
}

inline void addElementNodeSetData(int elem,const int *conn,int numAdjElems,
          const int nodesPerElem,int nodeSetSize,
              int nodeSetMap[MAX_ADJELEMS][MAX_NODESET_SIZE],
          adjNode *adaptAdjTable){
  for (int j=0; j<numAdjElems; j++) { // one nodeSet per neighbor element
    adjElem *e = new adjElem(nodeSetSize);
    e->nodeSetID = j;
    for (int k=0; k<nodeSetSize; k++) { // Build nodeSet for an element pairing
      e->nodeSet[k] = conn[elem*nodesPerElem+nodeSetMap[j][k]];
    }
    // Add this element-nodeSet pair to table at min nodeID in the nodeSet
    e->nodeSet.quickSort();
    int minNode = e->nodeSet[0];
    e->elemID = elem;
    e->next = adaptAdjTable[minNode].adjElemList->next;
    adaptAdjTable[minNode].adjElemList->next = e;
    adaptAdjTable[minNode].adjElemCount++;
  }
}


//Look for an adjElem (rover->next) whose nodeSet matches that
//specified in searchForNodeSet in the link list following adjStart
//and return rover such that rover->next matches with the
//searchForNodeSet. It also checks that the elemID of the element
//being searched does not match with that of searchForElemID.  *found
//is set to 1 if match is found .. else to 0
inline adjElem *searchAdjElemInList(adjElem *adjStart,int *searchForNodeSet,
            int nodeSetSize,int searchForElemID,
            int *found){
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

  
  // A nodeSet is a set of nodes that defines a pairing of two
  // adjacent elements; For example, in 2D triangle meshes, the
  // nodeSet is the nodes of an edge between two elements.  The
  // nodeSetMap is an ordering of element-local node IDs that
  // specifies all possible nodeSets for a particular element type
  int nodeSetMap[MAX_ADJELEMS][MAX_NODESET_SIZE];
  guessElementShape(dim,nodesPerElem,&numAdjElems,&nodeSetSize,nodeSetMap);
  CkAssert(nodeSetSize <= MAX_NODESET_SIZE);
  
  // Add the FEM_ADAPT_ADJ attribute to the elements
	// Set the correct width of the table and then get the pointer to the actual data
	FEM_DataAttribute *adaptAdjAttr = (FEM_DataAttribute *) elem->lookup(FEM_ADAPT_ADJ,"CreateAdaptAdjacencies");	
	adaptAdjAttr->setWidth(sizeof(adaptAdj)*numAdjElems);
  adaptAdj *adaptAdjacencies = (adaptAdj *)(adaptAdjAttr->getChar()).getData();

  // Init adaptAdj array to at least have -1 as partID signifying no neighbor
  for (int i=0; i<numElems*numAdjElems; i++) {
    adaptAdjacencies[i].partID = -1;
		adaptAdjacencies[i].localID = -1;
  }

  // Create an array of size equal to the number of local nodes, each
  // entry has a pair: (shared partitions, list of elem-nodeSet pairs)
  // Call this adaptAdjTable
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
  
  for (int i=0; i<numElems; i++) { // Add each element-nodeSet pair to table
    if(elem->is_valid(i)){
      addElementNodeSetData(i,conn,numAdjElems,nodesPerElem,nodeSetSize,
          nodeSetMap,adaptAdjTable);
    }
  }

  fillLocalAdaptAdjacencies(numNodes,node,adaptAdjTable,adaptAdjacencies,nodeSetSize,numAdjElems,myRank,elemType);


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

  makeAdjacencyRequests(numNodes,node,adaptAdjTable,requestTable,nodeSetSize,myRank,elemType);

  requestTable->sync();
  printf("[%d] All requests made \n",myRank);

  MSA1DREPLYLIST *replyTable;
  if(myRank == 0){
    replyTable = new MSA1DREPLYLIST(numChunks,numChunks);
  }else{
    replyTable = new MSA1DREPLYLIST;
  }
  MPI_Bcast_pup(*replyTable,0,MPI_COMM_WORLD);
  replyTable->enroll(numChunks);
  replyTable->sync();

	
	replyAdjacencyRequests(requestTable,replyTable,node,adaptAdjTable,adaptAdjacencies,nodeSetSize,numAdjElems,myRank,elemType);

  requestTable->sync();
  replyTable->sync();

  //Once the replies are back, loop through each reply and update the
  //adaptAdjacencies for each element in the reply
  CkVec<adjReply> *receivedReplyVec = replyTable->get(myRank).vec;
  for(int i=0;i< receivedReplyVec->size();i++){
    adjReply *receivedReply = &(*receivedReplyVec)[i];
    printf("[%d] Replies received for (%d,%d) (%d,%d,%d)\n",myRank,receivedReply->requestingElemID,receivedReply->requestingNodeSetID,receivedReply->replyingElem.partID,receivedReply->replyingElem.localID,receivedReply->replyingElem.elemType);
    adaptAdjacencies[receivedReply->requestingElemID*numAdjElems + receivedReply->requestingNodeSetID] = receivedReply->replyingElem;
  }

  replyTable->sync();
  dumpAdaptAdjacencies(adaptAdjacencies,numElems,numAdjElems,myRank);

/*  mesh->becomeSetting();  
  //Register the adaptAdjacency with ParFUM
  FEM_Register_array(meshid,FEM_ELEM+elemType,FEM_ADAPT_ADJ,(void *)adaptAdjacencies,FEM_BYTE,sizeof(adaptAdj)*numAdjElems);
  //do not delete adaptAdjacencies. It will be used during the rest of adaptivity
  mesh->becomeGetting();  
	*/
}

void fillLocalAdaptAdjacencies(int numNodes,FEM_Node *node,adjNode *adaptAdjTable,adaptAdj *adaptAdjacencies,int nodeSetSize,int numAdjElems,int myRank,int elemType){

  for (int i=0; i<numNodes; i++) { 
    // For each node, match up incident elements
    // Each adjacency between two elements is represented by two adjElems
    // We try to match those up
    if(node->is_valid(i) && adaptAdjTable[i].adjElemList != NULL){  
//      CkAssert(adaptAdjTable[i].adjElemList != NULL);
      adjElem *preTarget = adaptAdjTable[i].adjElemList;
      adjElem *target = adaptAdjTable[i].adjElemList->next;
      while (target != NULL) { //loop over adjElemList of a node
        int found = 0; 
        //target represents an adjacency between two elements
        //We search for the other adjElem corresponding to that adjancency:
        //Look for an entry in adjElemList after target such that 
        //the nodeset of that entry and that of target match but they 
        //do not belong to the same element. 
        adjElem *rover = searchAdjElemInList(preTarget,
               target->nodeSet.getVec(),
               nodeSetSize,target->elemID,
               &found); 
        
        if (found) { // We found a local element adjacent to target->elemID
          // Set adjacency of target->elemID corresponding to nodeSet to 
          // rover->next->elemID, and vice versa
          // Store adjacency info in adaptAdjacency of each one and
          // use nodeSetID to index into adaptAdjacency
          adaptAdjacencies[numAdjElems*target->elemID+target->nodeSetID] = 
      adaptAdj(myRank,rover->next->elemID,elemType);
          adaptAdjacencies[numAdjElems*rover->next->elemID+rover->next->nodeSetID] = adaptAdj(myRank,target->elemID,elemType);
          // Remove both elem-nodeSet pairs from the list
          adjElem *tmp = rover->next;
          rover->next = rover->next->next;
          delete tmp;
          tmp = target;
          preTarget->next = target->next;
          target = target->next;
          delete tmp;
        }else { // No match for target was found in adjElemList
          // This means that either target is on the domain boundary 
          // or it is on a partition boundary and its neighbor is on another VP
          // Move target to next entry in adjElemList
          preTarget = target;
          target = target->next;
        }
      }
    }
  }
}


void makeAdjacencyRequests(int numNodes,FEM_Node *node,adjNode *adaptAdjTable,MSA1DREQLIST *requestTable, int nodeSetSize,int myRank,int elemType){

  for (int i=0; i<numNodes; i++) { // Examine each node's remaining elements
    if(node->is_valid(i)){
      if (adaptAdjTable[i].adjElemList->next!=NULL) {
        adjElem *adjStart = adaptAdjTable[i].adjElemList->next;
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
              set_intersection(commonSharedChunks.begin(), commonSharedChunks.end(), 
             sharedChunks.begin(), sharedChunks.end(),inserter(tmpIntersect, tmpIntersect.begin()));
              commonSharedChunks = tmpIntersect;              
            }
          }
          // At this point commonSharedChunks contains the list of
          // chunks with which the element pointed by adjStart might
          // be shared
          int numCommonSharedChunks = commonSharedChunks.size();
          if(numCommonSharedChunks > 0){
      //adjStart is possibly shared with these chunks. It is
      //shared across the adjStart->nodeSet set of nodes.
            adjRequest *adjRequestList = new adjRequest[numCommonSharedChunks];
            //Translate the nodes in the nodeSet into the index in the
            //idxl list for each chunk in the commonSharedChunks
            for(int j=0;j<nodeSetSize;j++){
              int sharedNode = adjStart->nodeSet[j];
              const IDXL_Rec *recSharedNode = node->shared.getRec(sharedNode);
              int countChunk=0;
              for(std::set<int>::iterator chunkIterator = commonSharedChunks.begin();chunkIterator != commonSharedChunks.end();chunkIterator++){
                int chunk = *chunkIterator;
                // if(chunk > myRank){
                if(j == 0){
                  // if this is the first node we need to initialize
                  // the adjRequestobject
                  adjRequestList[countChunk] =  adjRequest(adjStart->elemID,myRank,adjStart->nodeSetID,elemType);
                }
                int sharedNodeIdx=-1; // index of sharedNode in the idxl list of chunk
                //search for this chunk in the list of chunks shared
                //by this node
                for(int k=0;k<recSharedNode->getShared();k++){
                  if(recSharedNode->getChk(k) == chunk){
                    //found the correct chunk
                    sharedNodeIdx = recSharedNode->getIdx(k);
                    break;
                  }
                }
                CkAssert(sharedNodeIdx != -1);
                //The index of sharedNode in the index list of chunk
                //has been found.  this needs to be saved in the
                //corresponding translatedNodeSet
                adjRequestList[countChunk].translatedNodeSet[j] = sharedNodeIdx;
                //                }
                countChunk++;
              }
            }
            //Now the nodeNumbers for the nodeSets that might be along
            //chunk boundaries have been translated into idxl indices
            //between the two chunks We now need to write the requests
            //into the msa array requestTable WARNING: This depends on
            //sets getting enumerated in the same way always Might not
            //be true
            int countChunk=0;
            for(std::set<int>::iterator chunkIterator = commonSharedChunks.begin();chunkIterator != commonSharedChunks.end();chunkIterator++){
              int chunk = *chunkIterator;
//              if(chunk > myRank){
                printf("[%d] Sending to chunk %d request (%d,%d,%d,%d) \n",myRank,chunk,adjRequestList[countChunk].elemID,adjRequestList[countChunk].chunkID,adjRequestList[countChunk].elemType,adjRequestList[countChunk].nodeSetID);
                (*requestTable).accumulate(chunk,adjRequestList[countChunk]);
//              requestVec.push_back(&adjRequestList[countChunk]);
//              }
              countChunk++;
            }
            delete [] adjRequestList;
          }
          adjStart = adjStart->next;
        }
      }
    }
  }
}

void replyAdjacencyRequests(MSA1DREQLIST *requestTable,MSA1DREPLYLIST *replyTable,FEM_Node *node,adjNode *adaptAdjTable,adaptAdj *adaptAdjacencies,int nodeSetSize,int numAdjElems,int myRank,int elemType){
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
      sharedNodes[j] = sharedNode;
    }
    sharedNodes.quickSort();
    //We need to find the matching nodeset for the nodeset in the request
    //We look it up in the adaptAdjTable for the minimum node on this chunk
    adjNode *minNode = &adaptAdjTable[sharedNodes[0]];
    int found=0;
    //search for the nodeSet in the list of adjacencies around minNode
    adjElem *rover =  searchAdjElemInList(minNode->adjElemList,sharedNodes.getVec(),nodeSetSize,-1,&found);
    printf("[%d] Received request (%d,%d,%d,%d) minNode %d found %d\n",myRank,receivedRequest.elemID,receivedRequest.chunkID,receivedRequest.elemType,receivedRequest.nodeSetID,sharedNodes[0],found);
    if(found){ //we have found a matching adjElem for the requested nodeset
       //we shall set the adjacency correctly in the adjacency Table
       //for the elemID in the found adjElem. We need to send a reply
       //to the requesting chunk
       int matchingElemID;
       matchingElemID = rover->next->elemID;
       adaptAdj *matchingAdaptAdj = &adaptAdjacencies[matchingElemID*numAdjElems + rover->next->nodeSetID];
       adjElem *tmp = rover->next;
       rover->next = tmp->next;
       delete tmp;
       CkAssert(matchingAdaptAdj->localID == -1);
       matchingAdaptAdj->partID = receivedRequest.chunkID;
       matchingAdaptAdj->localID = receivedRequest.elemID;
       matchingAdaptAdj->elemType = receivedRequest.elemType;

       //Set requesting data in reply to that in receivedRequest
       //Put in data from rover->next into the replyElem portion of data
       adjReply reply;
       reply.requestingElemID = receivedRequest.elemID;
       reply.requestingNodeSetID = receivedRequest.nodeSetID;
       reply.replyingElem.partID = myRank;
       reply.replyingElem.localID = matchingElemID;
       reply.replyingElem.elemType = elemType;
       //Write into the replyTable
       replyTable->accumulate(receivedRequest.chunkID,reply);
    }else{
      //we have no matching nodeset for this request.. hopefully some
      //other chunk does; we can ignore this request
    }
  }
}



void dumpAdaptAdjacencies(adaptAdj *adaptAdjacencies,int numElems,int numAdjElems,int myRank){
  for(int i=0;i<numElems;i++){
    printf("[%d] %d  :",myRank,i);
    for(int j=0;j<numAdjElems;j++){
      adaptAdj *entry = &adaptAdjacencies[i*numAdjElems+j];
      printf("(%d,%d,%d)",entry->partID,entry->localID,entry->elemType);
    }
    printf("\n");
  }
}

inline void findNodeSet(int meshid,int elemType,int *numAdjElems,int *nodeSetSize,int  nodeSetMap[MAX_ADJELEMS][MAX_NODESET_SIZE]){
	FEM_Mesh *mesh = FEM_chunk::get("GetAdaptAdj")->lookup(meshid,"GetAdaptAdj");
  FEM_Elem *elem = (FEM_Elem *)mesh->lookup(FEM_ELEM+elemType,"GetAdaptAdj");
  FEM_Node *node = (FEM_Node *)mesh->lookup(FEM_NODE,"GetAdaptAdj");
  const int nodesPer = (elem->getConn()).width();
  assert(node->getCoord()!= NULL);
  const int dim = (node->getCoord())->getWidth();
  assert(dim == 2|| dim == 3);
  guessElementShape(dim,nodesPer,numAdjElems,nodeSetSize,nodeSetMap);
}

void getAndDumpAdaptAdjacencies(int meshid, int numElems, int elemType, int myRank){
  int numAdjElems;
  int nodeSetSize, nodeSetMap[MAX_ADJELEMS][MAX_NODESET_SIZE]; // not used
	findNodeSet(meshid,elemType,&numAdjElems,&nodeSetSize,nodeSetMap);
  
  for(int i=0;i<numElems;i++){
    printf("[%d] %d  :",myRank,i);
    for(int j=0;j<numAdjElems;j++){
      adaptAdj *entry = GetAdaptAdj(meshid, i, elemType, j);
      printf("(%d,%d,%d)",entry->partID,entry->localID,entry->elemType);
    }
    printf("\n");
  }
}

// Access functions
inline adaptAdj *lookupAdaptAdjacencies(FEM_Mesh *mesh,int elemType,int *numAdjacencies){
  FEM_Elem *elem = (FEM_Elem *)mesh->lookup(FEM_ELEM+elemType,"lookupAdaptAdjacencies");

  FEM_DataAttribute *adaptAttr = (FEM_DataAttribute *)elem->lookup(FEM_ADAPT_ADJ,"lookupAdaptAdjacencies");
	*numAdjacencies = adaptAttr->getWidth()/sizeof(adaptAdj);
  AllocTable2d<unsigned char> &table = adaptAttr->getChar();

  return (adaptAdj  *)table.getData();
}

inline adaptAdj *lookupAdaptAdjacencies(int meshid,int elemType,int *numAdjacencies){
  FEM_Mesh *mesh = FEM_chunk::get("lookupAdaptAdjacencies")->lookup(meshid,"lookupAdaptAdjacencies");
  return lookupAdaptAdjacencies(mesh, elemType, numAdjacencies);
}

/** Look up elemID in elemType array, access edgeFaceID-th adaptAdj. */
adaptAdj *GetAdaptAdj(int meshid,int elemID, int elemType, int edgeFaceID)
{
  int numAdjacencies;
  adaptAdj *adaptAdjacencies = lookupAdaptAdjacencies(meshid, elemType,&numAdjacencies);
  return(&(adaptAdjacencies[elemID*numAdjacencies+edgeFaceID]));
}

adaptAdj *GetAdaptAdj(FEM_Mesh *meshPtr,int elemID, int elemType, int edgeFaceID)
{
  int numAdjacencies;
  adaptAdj *adaptAdjacencies = lookupAdaptAdjacencies(meshPtr, elemType,&numAdjacencies);
  return(&(adaptAdjacencies[elemID*numAdjacencies+edgeFaceID]));
}

/** Look up elemID in elemType array, calculate edgeFaceID from
    vertexList (with GetEdgeFace below), and access edgeFaceID-th
    adaptAdj with GetAdaptAdj above. */
adaptAdj *GetAdaptAdj(int meshid,int elemID, int elemType, int *vertexList)
{
  int edgeFaceID = GetEdgeFace(meshid, elemID, elemType, vertexList);
  return GetAdaptAdj(meshid, elemID, elemType, edgeFaceID);
}

/** Look up elemID in elemType array and determine the set of vertices
    associated with the edge or face represented by edgeFaceID. */
void GetVertices(int meshid,int elemID, int elemType, int edgeFaceID, int *vertexList)
{
	int numAdjacencies;
  int nodeSetSize, nodeSetMap[MAX_ADJELEMS][MAX_NODESET_SIZE];
	findNodeSet(meshid,elemType,&numAdjacencies,&nodeSetSize,nodeSetMap);

  for (int i=0; i<nodeSetSize; i++) {
    vertexList[i] = nodeSetMap[edgeFaceID][i];
  }
}

/** Look up elemID in elemType array and determine the edge or face ID
    specified by the set of vertices in vertexList. */
int GetEdgeFace(int meshid,int elemID, int elemType, int *vertexList)
{
	int numAdjacencies;
  int nodeSetSize, nodeSetMap[MAX_ADJELEMS][MAX_NODESET_SIZE];
	findNodeSet(meshid,elemType,&numAdjacencies,&nodeSetSize,nodeSetMap);

  std::set<int> vertexSet(vertexList, vertexList+nodeSetSize);
  for (int i=0; i<numAdjacencies; i++) {
    std::set<int> aNodeSet(nodeSetMap[i], nodeSetMap[i]+nodeSetSize);
    if (vertexSet == aNodeSet) return i; // CHECK: does set equivalence exist?
  }
  CkAbort("ERROR: GetEdgeFace: vertexList is not a valid nodeSet./n");
}

// Update functions
/** Look up elemID in elemType array and set the adjacency on
    edgeFaceID to nbr. */
void SetAdaptAdj(int meshID,int elemID, int elemType, int edgeFaceID, adaptAdj nbr)
{
	int numAdjacencies;
  adaptAdj *adaptAdjTable = lookupAdaptAdjacencies(meshID, elemType,&numAdjacencies);
  adaptAdjTable[elemID*numAdjacencies + edgeFaceID] = nbr;
}

/** Lookup elemID in elemType array and search for the edgeID which has originalNbr as
 * a neighbor, then replace originalNbr with newNbr
 */
void ReplaceAdaptAdj(FEM_Mesh *meshPtr,int elemID,int elemType,adaptAdj originalNbr, adaptAdj newNbr){
  int numAdjacencies;
  adaptAdj *adaptAdjTable = lookupAdaptAdjacencies(meshPtr, elemType,&numAdjacencies);
  for(int i=0;i<numAdjacencies;i++){
    if(adaptAdjTable[elemID*numAdjacencies+i] == originalNbr){
      adaptAdjTable[elemID*numAdjacencies+i] = newNbr;
      return;
    }
  }
  CkAbort("ReplaceAdaptAdj did not find the specified originalNbr");
}

void ReplaceAdaptAdj(int meshID,int elemID,int elemType,adaptAdj originalNbr, adaptAdj newNbr){
  FEM_Mesh *meshPtr = FEM_chunk::get("ReplaceAdaptAdj")->lookup(meshID,"ReplaceAdaptAdj");
  ReplaceAdaptAdj(meshPtr, elemID, elemType, originalNbr, newNbr);
}


//given the dimensions and nodes per element guess whether the element 
// is a triangle, quad, tet or hex. At the moment these are the 4 shapes
// that are handled
  
#define copyNodeSetMap(numAdjElems,nodeSetSize,nodeSetMap,srcMap) {\
  for (int i=0;i<numAdjElems;i++){ \
    for(int j=0;j<nodeSetSize;j++){ \
      nodeSetMap[i][j] = srcMap[i][j]; \
    } \
  } \
}

void guessElementShape(int dim,int nodesPerElem,int *numAdjElems,int *nodeSetSize,int nodeSetMap[MAX_ADJELEMS][MAX_NODESET_SIZE]){
  switch(dim){
  case 2:
    {
      //2 dimension
      switch(nodesPerElem){
      case 3:
  //Triangles
  *numAdjElems = 3;
  *nodeSetSize = 2;
  copyNodeSetMap(3,2,nodeSetMap,nodeSetMap2d_tri)
    break;
      case 4:
  //quads
  *numAdjElems = 4;
  *nodeSetSize = 2;
  copyNodeSetMap(4,2,nodeSetMap,nodeSetMap2d_quad)
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
  copyNodeSetMap(4,3,nodeSetMap,nodeSetMap3d_tet)
    break;
      case 6:
  //Hexahedra
  *numAdjElems = 6;
  *nodeSetSize = 4;
  copyNodeSetMap(6,4,nodeSetMap,nodeSetMap3d_hex)
    break;
      }
    }
    break;
  }
}

