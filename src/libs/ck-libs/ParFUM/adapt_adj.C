/* Adaptivity Adjacencies: element-to-element adjacencies for use by
   adaptivity codes only.  Adaptivity codes should keep them
   up-to-date for each mesh modification primitive.
   
   Created 11 Sept 2006 - Terry L. Wilmarth
*/
#include "ParFUM.h"
#include "ParFUM_internals.h"
using namespace std;

const int faceMap2d_tri[3][2] = {{0,1},{1,2},{2,0}};
const int faceMap2d_quad[4][2] = {{0,1},{1,2},{2,3},{3,0}};
const int faceMap3d_tet[4][3] = {{0,1,2},{0,3,1},{0,2,3},{1,3,2}};
const int faceMap3d_hex[6][4] = {{0,1,2,3},{1,5,6,2},{2,6,7,3},{3,7,4,0},{0,4,5,1},{5,4,6,7}};

const int edgeMap3d_tet[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
const int edgeMap3d_hex[12][2] = {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};

const int faceMap2d_cohquad[2][2] = {{0,1},{2,3}}; // Cohesive for 2D triangles
const int faceMap3d_cohprism[2][3] = {{0,1,2},{3,4,5}}; // Cohesive for 3D tets


inline void addSharedNodeData(int node, const IDXL_Rec *sharedChunkList,
            adjNode *adaptAdjTable)
{
    adaptAdjTable[node].numSharedPartitions = sharedChunkList->getShared();
    adaptAdjTable[node].sharedWithPartition = 
        new int [sharedChunkList->getShared()];
    adaptAdjTable[node].sharedWithLocalIdx = 
        new int [sharedChunkList->getShared()];
    for(int j=0; j<sharedChunkList->getShared(); j++){
        int sharedChunk = sharedChunkList->getChk(j);
        int sharedIdx = sharedChunkList->getIdx(j);
        adaptAdjTable[node].sharedWithPartition[j] = sharedChunk;
        adaptAdjTable[node].sharedWithLocalIdx[j] = sharedIdx;
    }
}


inline void addElementNodeSetData(
        const int elem, 
        const int *conn, 
        const int nodesPerElem,
        const int numFaces, 
        const int numEdges, 
        const int faceSize,
        const int faceMap[MAX_ADJELEMS][MAX_FACE_SIZE],
        const int edgeMap[MAX_EDGES][2],
        adjNode *faceTable, 
        adjNode* edgeTable)
{
    // first add face adjacencies
    for (int j=0; j<numFaces; j++) { 
        adjElem *e = new adjElem(faceSize);
        e->nodeSetID = j;
        for (int k=0; k<faceSize; k++) { 
            e->nodeSet[k] = conn[elem * nodesPerElem + faceMap[j][k]];
        }
        // Add this element-nodeSet pair to table at min nodeID in the nodeSet
        e->nodeSet.quickSort();
        int minNode = e->nodeSet[0];
        e->elemID = elem;
        e->next = faceTable[minNode].adjElemList->next;
        faceTable[minNode].adjElemList->next = e;
        faceTable[minNode].adjElemCount++;
        //printf("Adding element %d face nodeset [ ", elem);
        //for (int i=0; i<faceSize; ++i)
        //	printf("%d ", e->nodeSet[i]);
        //printf("]\n");
    }

    // then add edge adjacencies
    if (edgeTable) {
        for (int j=0; j<numEdges; j++) { 
            adjElem *e = new adjElem(2);
            e->nodeSetID = j;
            for (int k=0; k<2; k++) { 
                e->nodeSet[k] = conn[elem * nodesPerElem + edgeMap[j][k]];
            }
            // Add this element-nodeSet pair to table at min nodeID in the 
            // nodeSet
            e->nodeSet.quickSort();
            int minNode = e->nodeSet[0];
            e->elemID = elem;
            e->next = edgeTable[minNode].adjElemList->next;
            edgeTable[minNode].adjElemList->next = e;
            edgeTable[minNode].adjElemCount++;
        
            //printf("Adding element %d edge nodeset [ ", elem);
            //for (int i=0; i<2; ++i)
            //	printf("%d ", e->nodeSet[i]);
            //printf("]\n");
        }
    }
}


//Look for an adjElem (rover->next) whose nodeSet matches that
//specified in searchForNodeSet in the link list following adjStart
//and return rover such that rover->next matches with the
//searchForNodeSet. It also checks that the elemID of the element
//being searched does not match with that of searchForElemID.  *found
//is set to 1 if match is found .. else to 0
inline adjElem *searchAdjElemInList(adjElem *adjStart, int *searchForNodeSet,
            int nodeSetSize, int searchForElemID, int *found)
{
    adjElem *rover = adjStart; // compare rover->next with adjStart
    *found = 0;

    while (rover->next != NULL) {
    	if (rover->next->elemID != searchForElemID) {
            //printf("looking for a match between %d and %d...", searchForElemID, rover->next->elemID);
            *found = 1; // found an element that is not myself, 
                        // possibly a match

            if (nodeSetSize != rover->next->nodeSet.size()) {
                *found = 0; // not a match
                //printf("nope\n");
                continue; 
            }
            for (int j=0; j<nodeSetSize; j++) {
                if (rover->next->nodeSet[j] != searchForNodeSet[j]) {
                    //printf("nope\n");
                	*found = 0; // not a match
                    break;
                }
            }
        }
        if (*found) {
            //printf(" [ ");
            //for (int i=0; i<nodeSetSize; ++i)
            //	printf("%d ", searchForNodeSet[i]);
            //printf("]\n");
        	break; // We have found a nodeSet that matches adjStart
        } else {
            rover = rover->next; // Keep looking for matching nodeSet
        }
    }
    return rover;
}


/** Create Adaptivity Adjacencies for elemType; dimension inferred. */
void CreateAdaptAdjacencies(int meshid, int elemType)
{
    // Need to derive all of these from elemType;
    int myRank,numChunks;
    int faceSize;    // number of nodes shared by two adjacent elems
    int faceMapSize; // number of faces per element
    int edgeMapSize; // number of edges per element

    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
    MPI_Comm_size(MPI_COMM_WORLD,&numChunks);
    FEM_Mesh* mesh = FEM_chunk::get("CreateAdaptAdjacencies")->lookup(
            meshid,"CreateAdaptAdjacencies");
    FEM_Elem* elem = (FEM_Elem *)mesh->lookup(
            FEM_ELEM+elemType,"CreateAdaptAdjacencies");
    FEM_Node* node = (FEM_Node *)mesh->lookup(
            FEM_NODE,"CreateAdaptAdjacencies");

    const int numElems = elem->size();
    const int numNodes = node->size();
    const int nodesPerElem = (elem->getConn()).width();
    CkAssert(node->getCoord() != NULL);
    const int dim = (node->getCoord())->getWidth();
    CkAssert(dim == 2|| dim == 3);

    // A nodeSet is a set of nodes that defines a pairing of two
    // adjacent elements; For example, in 2D triangle meshes, the
    // nodeSet is the nodes of an edge between two elements.
    // These adjacency relationships are defined by face and edge maps that
    // define which set of nodes constitute faces and edges for each element
    // type.
    int faceMap[MAX_ADJELEMS][MAX_FACE_SIZE];
    int edgeMap[MAX_EDGES][2];

    guessElementShape(dim, nodesPerElem, 
            &faceSize, &faceMapSize, &edgeMapSize, 
            faceMap, edgeMap);
    CkAssert(faceMapSize <= MAX_ADJELEMS);
    CkAssert(edgeMapSize <= MAX_EDGES);
    CkAssert(faceSize <= MAX_FACE_SIZE);

    // Add the FEM_ADAPT_FACE_ADJ and FEM_ADAPT_EDGE_ADJ attributes
    FEM_DataAttribute* adaptAdjAttr = (FEM_DataAttribute*) elem->lookup(
            FEM_ADAPT_FACE_ADJ, "CreateAdaptAdjacencies");	
    adaptAdjAttr->setWidth(faceMapSize*sizeof(adaptAdj));
    adaptAdj* adaptFaceAdjacencies = 
        reinterpret_cast<adaptAdj*>((adaptAdjAttr->getChar()).getData());

    FEM_DataAttribute* adaptAdjEdgeAttr = (FEM_DataAttribute*) elem->lookup(
            FEM_ADAPT_EDGE_ADJ, "CreateAdaptAdjacencies");	
    adaptAdjEdgeAttr->setWidth(edgeMapSize*sizeof(CkVec<adaptAdj>*));
    CkVec<adaptAdj>** adaptEdgeAdjacencies = edgeMapSize == 0 ?
        NULL :
        reinterpret_cast<CkVec<adaptAdj>**>(
                (adaptAdjEdgeAttr->getChar()).getData());

    // Initialize adaptAdj arrays
    for (int i=0; i<numElems; i++) {
        for (int j=0; j<faceMapSize; ++j) {
            adaptFaceAdjacencies[i*faceMapSize + j].partID = -1;
            adaptFaceAdjacencies[i*faceMapSize + j].localID = -1;
        }
        if (adaptEdgeAdjacencies) {
            for (int j=0; j<edgeMapSize; ++j) {
                adaptEdgeAdjacencies[i*edgeMapSize + j] = new CkVec<adaptAdj>;
                assert(adaptEdgeAdjacencies[i*edgeMapSize + j] != NULL);
            }
        }
    }

    // Create an array of size equal to the number of local nodes, each
    // entry has a pair: (shared partitions, list of elem-nodeSet pairs)
    // make one array for face adjacencies and one for edge adjacencies

    adjNode *faceTable;
    adjNode *edgeTable;
    faceTable = new adjNode[numNodes];
    edgeTable = adaptEdgeAdjacencies ? new adjNode[numNodes] : NULL;

    // Loop through the shared node list and add shared partition ids
    for (int i=0; i<numNodes; i++) {
        if (node->is_valid(i)) {
            const IDXL_Rec *sharedChunkList = node->shared.getRec(i);
            if (sharedChunkList != NULL) {
                addSharedNodeData(i, sharedChunkList, faceTable);
                if (edgeTable)
                    addSharedNodeData(i, sharedChunkList, edgeTable);
            } else {
                faceTable[i].sharedWithPartition = NULL;
                faceTable[i].sharedWithLocalIdx= NULL;
                if (edgeTable) {
                    edgeTable[i].sharedWithPartition = NULL;
                    edgeTable[i].sharedWithLocalIdx= NULL;
                }
            }
        } else {
            faceTable[i].sharedWithPartition = NULL;
            faceTable[i].sharedWithLocalIdx= NULL;
            if (edgeTable) {
                edgeTable[i].sharedWithPartition = NULL;
                edgeTable[i].sharedWithLocalIdx= NULL;
            }
        }
    }

    // add local node adjacency info via connectivity
    const int *conn = (elem->getConn()).getData();
    for (int i=0; i<numElems; i++) {
        if(elem->is_valid(i)){
            addElementNodeSetData(i, conn, nodesPerElem, faceMapSize, 
                    edgeMapSize, faceSize, faceMap, edgeMap, faceTable, 
                    edgeTable);
        }
    }

    fillLocalAdaptAdjacencies(
            numNodes,
            node,
            faceTable,
            edgeTable,
            faceMapSize,
            edgeMapSize,
            adaptFaceAdjacencies,
            adaptEdgeAdjacencies,
            myRank,
            elemType);

    // Now all elements' local adjacencies are set; remainder in table are 
    // nodeSets shared with other chunks or nodeSets on domain boundary
    // We handle face nodeSets first, then do the edges.

    MPI_Barrier(MPI_COMM_WORLD);
    MSA1DREQLIST *requestTable;
    if (myRank == 0) {
        requestTable = new MSA1DREQLIST(numChunks,numChunks);
    } else {
        requestTable = new MSA1DREQLIST;
    }
    MPI_Bcast_pup(*requestTable,0,MPI_COMM_WORLD);
    requestTable->enroll(numChunks);
    MSA1DREQLIST::Accum requestTableAcc = requestTable->getInitialAccum();

    makeAdjacencyRequests(
            numNodes,
            node,
            faceTable,
            requestTableAcc,
            faceSize,
            myRank,
            elemType);

    MSA1DREQLIST::Read reqTableRead = requestTableAcc.syncToRead();
    //printf("[%d] All face requests made \n",myRank);

    MSA1DREPLYLIST *replyTable;
    if (myRank == 0) {
        replyTable = new MSA1DREPLYLIST(numChunks,numChunks);
    } else {
        replyTable = new MSA1DREPLYLIST;
    }
    MPI_Bcast_pup(*replyTable,0,MPI_COMM_WORLD);
    replyTable->enroll(numChunks);
    MSA1DREPLYLIST::Accum replyAcc = replyTable->getInitialAccum();

    replyAdjacencyRequests(
            reqTableRead.get(myRank).vec,
            replyAcc,
            node,
            faceTable,
            adaptFaceAdjacencies,
            adaptEdgeAdjacencies,
            faceSize,
            faceMapSize,
            myRank,
            elemType,
            false);

    reqTableRead.syncDone();
    replyAcc.syncDone();

//    // Once the replies are back, loop through each reply and update the
//    // adjacencies for each element in the reply
//    CkVec<adjReply> *receivedReplyVec = replyTable->get(myRank).vec;
//    for(int i=0;i< receivedReplyVec->size();i++){
//        adjReply *receivedReply = &(*receivedReplyVec)[i];
//        printf("[%d] Replies received for (%d,%d) (%d,%d,%d)\n",
//                myRank,receivedReply->requestingElemID,
//                receivedReply->requestingNodeSetID,
//                receivedReply->replyingElem.partID,
//                receivedReply->replyingElem.localID,
//                receivedReply->replyingElem.elemType);
//        adaptFaceAdjacencies[receivedReply->requestingElemID*faceMapSize + 
//            receivedReply->requestingNodeSetID] = receivedReply->replyingElem;
//    }
//    replyTable->sync();
    delete requestTable;
    delete replyTable;

    if (adaptEdgeAdjacencies != NULL) {
	    MSA1DREQLIST *edgeRequestTable;
	    MSA1DREPLYLIST *edgeReplyTable;

        // do the same thing for the edges
        if (myRank == 0) {
            edgeRequestTable = new MSA1DREQLIST(numChunks,numChunks);
        } else {
            edgeRequestTable = new MSA1DREQLIST;
        }
        MPI_Bcast_pup(*edgeRequestTable,0,MPI_COMM_WORLD);
        edgeRequestTable->enroll(numChunks);
	MSA1DREQLIST::Accum edgeRequestTableAcc = requestTable->getInitialAccum();

        makeAdjacencyRequests(
                numNodes,
                node,
                edgeTable,
                edgeRequestTableAcc,
                2,
                myRank,
                elemType);

	MSA1DREQLIST::Read edgeReqRead = edgeRequestTableAcc.syncToRead();
        //printf("[%d] All edge requests made \n",myRank);

        if (myRank == 0) {
            edgeReplyTable = new MSA1DREPLYLIST(numChunks,numChunks);
        } else {
            edgeReplyTable = new MSA1DREPLYLIST;
        }
        MPI_Bcast_pup(*edgeReplyTable,0,MPI_COMM_WORLD);
        edgeReplyTable->enroll(numChunks);
	MSA1DREPLYLIST::Accum edgeReplyAcc = edgeReplyTable->getInitialAccum();

        replyAdjacencyRequests(
                edgeReqRead.get(myRank).vec,
                edgeReplyAcc,
                node,
                edgeTable,
                adaptFaceAdjacencies,
                adaptEdgeAdjacencies,
                2,
                edgeMapSize,
                myRank,
                elemType,
                true);

        edgeReqRead.syncDone();
        edgeReplyAcc.syncDone();

//        // Once the replies are back, loop through each reply and update the
//        // adjacencies for each element in the reply
//        CkVec<adjReply> *receivedReplyVec = edgeReplyTable->get(myRank).vec;
//        for(int i=0;i< receivedReplyVec->size();i++){
//            adjReply *receivedReply = &(*receivedReplyVec)[i];
//            printf("[%d] Replies received for (%d,%d) (%d,%d,%d)\n",
//                    myRank,receivedReply->requestingElemID,
//                    receivedReply->requestingNodeSetID,
//                    receivedReply->replyingElem.partID,
//                    receivedReply->replyingElem.localID,
//                    receivedReply->replyingElem.elemType);
//            adaptEdgeAdjacencies[receivedReply->requestingElemID*edgeMapSize + 
//                receivedReply->requestingNodeSetID]->push_back(
//                        receivedReply->replyingElem);
//        }
//        edgeReplyTable->sync();

        delete edgeRequestTable;
        delete edgeReplyTable;
    }

    for (int i=0; i<numNodes; ++i) {
        delete[] faceTable[i].sharedWithPartition;
        delete[] faceTable[i].sharedWithLocalIdx;
        adjElem* e = faceTable[i].adjElemList->next;
        adjElem* dummy;
        	while (e != NULL) {
            dummy = e;
            e = e->next;
            delete dummy;
        }

        if (edgeTable) {
            delete[] edgeTable[i].sharedWithPartition;
            delete[] edgeTable[i].sharedWithLocalIdx;
            e = edgeTable[i].adjElemList->next;
            while (e != NULL) {
                dummy = e;
                e = e->next;
                delete dummy;
            }
        }
    }
    delete[] faceTable;
    delete[] edgeTable;
    
}


void fillLocalAdaptAdjacencies(
        const int numNodes, 
        FEM_Node* node, 
        adjNode* faceTable, 
        adjNode* edgeTable,
        const int faceMapSize,
        const int edgeMapSize,
        adaptAdj* adaptFaceAdjacencies,
        CkVec<adaptAdj>** adaptEdgeAdjacencies,
        const int myRank, 
        const int elemType)
{
	// start with face adjacencies
    adjNode* adaptAdjTable = faceTable;
    for (int i=0; i<numNodes; i++) {
		if (!adaptAdjTable)
			break;
		// For each node, match up incident elements
		// Each adjacency between two elements is represented by two 
		// adjElems. We try to match those up
		if (node->is_valid(i) && adaptAdjTable[i].adjElemList != NULL) {
			// CkAssert(adaptAdjTable[i].adjElemList != NULL);
			adjElem *preTarget = adaptAdjTable[i].adjElemList;
			adjElem *target = adaptAdjTable[i].adjElemList->next;
			while (target != NULL) { //loop over adjElemList of a node
				int found = 0;
				// target represents an adjacency between two elements
				// We search for the other adjElem corresponding to that 
				// adjancency:
				// Look for an entry in adjElemList after target such that 
				// the nodeset of that entry and that of target match but 
				// they do not belong to the same element. 
				adjElem *rover = searchAdjElemInList(preTarget,
						target->nodeSet.getVec(), target->nodeSet.size(),
						target->elemID, &found);

				if (found) {
					// We found a local element adjacent to target->elemID
					// Set adjacency of target->elemID corresponding to 
					// nodeSet to rover->next->elemID, and vice versa
					// Store adjacency info in adaptAdjacency of each one 
					// and use nodeSetID to index into adaptAdjacency

					adaptFaceAdjacencies[faceMapSize*target->elemID +
					target->nodeSetID] = adaptAdj(myRank, rover->next->elemID,
							elemType);
					adaptFaceAdjacencies[
					faceMapSize*rover->next->elemID +
					rover->next->nodeSetID] = adaptAdj(myRank, target->elemID,
							elemType);

					// Remove both elem-nodeSet pairs from the list
					adjElem *tmp = rover->next;
					rover->next = rover->next->next;
					delete tmp;
					tmp = target;
					preTarget->next = target->next;
					target = target->next;
					delete tmp;
				} else {
					// No match for target was found in adjElemList
					// This means that either target is on the domain 
					// boundary or it is on a partition boundary and its 
					// neighbor is on another VP. Move target to next 
					// entry in adjElemList
					preTarget = target;
					target = target->next;
				}
			}
		}
	}
  
  // now handle edge adjacencies
	adaptAdjTable = edgeTable;
	for (int i=0; i<numNodes; i++) {
		if (!adaptAdjTable)
			break;
		// For each node, match up incident elements
		// Each adjacency between two elements is represented by two 
		// adjElems. We try to match those up
		if (node->is_valid(i) && adaptAdjTable[i].adjElemList != NULL) {
			adjElem *preTarget = adaptAdjTable[i].adjElemList;
			adjElem *target = adaptAdjTable[i].adjElemList->next;
			//			printf("adjElemList for node %d [ ", i);
			//			while (preTarget != NULL) {
			//				printf("%d ", preTarget->elemID);
			//				preTarget = preTarget->next;
			//			}
			//			printf("]\n");

			preTarget = adaptAdjTable[i].adjElemList;
			target = preTarget->next;
			while (target != NULL) { //loop over adjElemList of a node
				//if (target->elemID >= preTarget->elemID) {
				//	target = target->next;
				//	continue;
				//}
				int found = 0;
				// target represents an adjacency between two elements
				// We search for the other adjElem corresponding to that 
				// adjancency:
				// Look for an entry in adjElemList after target such that 
				// the nodeset of that entry and that of target match but 
				// they do not belong to the same element. 
				adjElem
				        *rover =
				                searchAdjElemInList(preTarget, target->nodeSet.getVec(), target->nodeSet.size(), target->elemID, &found);

				while (found) {
					// We found a local element adjacent to target->elemID
					// Set adjacency of target->elemID corresponding to 
					// nodeSet to rover->next->elemID, and vice versa
					// Store adjacency info in adaptAdjacency of each one 
					// and use nodeSetID to index into adaptAdjacency

					//printf("adding adj for elem %d position %d (%d %d %d)\n", target->elemID, target->nodeSetID, myRank, rover->next->elemID, elemType);

					adaptEdgeAdjacencies[edgeMapSize*target->elemID +
					target->nodeSetID]->push_back(adaptAdj(myRank, rover->next->elemID, elemType));
					//adaptEdgeAdjacencies[
					//edgeMapSize*rover->next->elemID +
					//rover->next->nodeSetID]->push_back(adaptAdj(myRank,
					//		target->elemID, elemType));

					rover
					        = searchAdjElemInList(rover->next, target->nodeSet.getVec(), target->nodeSet.size(), target->elemID, &found);
				}
				target = target->next;
			}
		}
	}
}


void makeAdjacencyRequests(
        const int numNodes,
        FEM_Node *node,
        adjNode *adaptAdjTable,
        MSA1DREQLIST::Accum &requestTable, 
        const int nodeSetSize,
        const int myRank,
        const int elemType)
{
    for (int i=0; i<numNodes; i++) { 
        // Examine each node's remaining elements
        if(node->is_valid(i)){
            if (adaptAdjTable[i].adjElemList->next!=NULL) {
                adjElem *adjStart = adaptAdjTable[i].adjElemList->next;
                while (adjStart !=NULL) {
                    // create an empty set, commonSharedChunks
                    std::set<int> commonSharedChunks;
                    for (int j=0; j<nodeSetSize; j++) {
                        // look up sharedWithPartitions for node: 
                        //    adaptAdjTable[i]->adjElemList->nodeset[j]
                        // intersect with commonSharedChunks
                        int sharedNode = adjStart->nodeSet[j];
                        adjNode *sharedNodeAdj = &adaptAdjTable[sharedNode];
                        std::set<int> sharedChunks(
                                sharedNodeAdj->sharedWithPartition,
                                sharedNodeAdj->sharedWithPartition + 
                                sharedNodeAdj->numSharedPartitions);
                        if(j == 0){
                            commonSharedChunks = sharedChunks;
                        }else{
                            std::set<int> tmpIntersect;
                            set_intersection(
                                    commonSharedChunks.begin(), 
                                    commonSharedChunks.end(), 
                                    sharedChunks.begin(), 
                                    sharedChunks.end(),
                                    inserter(
                                        tmpIntersect, 
                                        tmpIntersect.begin()));
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
                        adjRequest *adjRequestList = 
                            new adjRequest[numCommonSharedChunks];
                        // Translate the nodes in the nodeSet into the index in 
                        // the idxl list for each chunk in the 
                        // commonSharedChunks
//                        CkPrintf("[%d] Creating request for NodeSet ( ", myRank);
//                        for (int j=0; j<nodeSetSize; ++j) {
//                        	CkPrintf("%d ", adjStart->nodeSet[j]);
//                        }
//                        CkPrintf(")\n");
                        for(int j=0;j<nodeSetSize;j++){
                            int sharedNode = adjStart->nodeSet[j];
                            const IDXL_Rec *recSharedNode = 
                                node->shared.getRec(sharedNode);
                            int countChunk=0;
                            for(std::set<int>::iterator chunkIterator = 
                                    commonSharedChunks.begin();
                                    chunkIterator != commonSharedChunks.end();
                                    chunkIterator++){
                                int chunk = *chunkIterator;
                                if(j == 0){
                                    // if this is the first node we need to 
                                    // initialize the adjRequestobject
                                    adjRequestList[countChunk] = 
                                        adjRequest(adjStart->elemID,
                                                myRank,
                                                adjStart->nodeSetID,
                                                elemType);
                                }

                                // index of sharedNode in the idxl list of 
                                // chunk search for this chunk in the list of 
                                // chunks shared by this node
                                int sharedNodeIdx=-1; 
                                for(int k=0;k<recSharedNode->getShared();k++){
                                    if(recSharedNode->getChk(k) == chunk){
                                        //found the correct chunk
                                        sharedNodeIdx = 
                                            recSharedNode->getIdx(k);
                                        break;
                                    }
                                }
                                CkAssert(sharedNodeIdx != -1);
                                //The index of sharedNode in the index list of
                                //chunks has been found.  This needs to be 
                                //saved in the corresponding translatedNodeSet
                                adjRequestList[countChunk].
                                    translatedNodeSet[j] = sharedNodeIdx;
                                countChunk++;
                            }
                        }

                        // Now the nodeNumbers for the nodeSets that might be 
                        // along chunk boundaries have been translated into 
                        // idxl indices between the two chunks We now need to 
                        // write the requests into the msa array requestTable 
                        int countChunk=0;
                        for(std::set<int>::iterator chunkIterator = 
                                commonSharedChunks.begin();
                                chunkIterator != commonSharedChunks.end();
                                chunkIterator++){
                            int chunk = *chunkIterator;
#if 0
                            printf("[%d] Sending to chunk %d request (%d,%d,%d,%d) \n",
				   myRank, chunk,
				   adjRequestList[countChunk].elemID,
				   adjRequestList[countChunk].chunkID,
				   adjRequestList[countChunk].elemType,
				   adjRequestList[countChunk].nodeSetID);
#endif
                            requestTable(chunk) += adjRequestList[countChunk];
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


void replyAdjacencyRequests(
	CkVec<adjRequest> *receivedRequestVec,
        MSA1DREPLYLIST::Accum &replyTable,
        FEM_Node* node,
        adjNode* adaptAdjTable,
        adaptAdj* adaptFaceAdjacencies,
        CkVec<adaptAdj>** adaptEdgeAdjacencies,
        const int nodeSetSize,
        const int numAdjElems,
        const int myRank,
        const int elemType,
        bool isEdgeRequest)
{
    // look up request table, put answer in reply table 
    // receive: for local elemID, the adjacency on this set of shared indices 
    // is (remote elemID, remote partition ID, remote elem type)
    // add adjacencies to local table
    // lots of error checking :)
	
	// TODO: adjustment of adj lists based on reply tables was unnecessary
	// Can we get rid of reply table altogether?

    //Look at each request that in the requestTable for this chunk
    //Put the data for the requests in our own table and then create replies
    for (int i=0;i<receivedRequestVec->length();i++) {    
        adjRequest &receivedRequest = (*receivedRequestVec)[i];
        const IDXL_List &sharedNodeList = 
            node->shared.getList(receivedRequest.chunkID);
        CkVec<int> sharedNodes(nodeSetSize);
        //Translate all the nodes in the nodeSet of the request
        for (int j=0;j<nodeSetSize;j++) {
            int sharedNode = sharedNodeList[
                receivedRequest.translatedNodeSet[j]];
            sharedNodes[j] = sharedNode;
        }
        sharedNodes.quickSort();
        //We need to find the matching nodeset for the nodeset in the request
        //We look it up in the adaptAdjTable for the minimum node on this chunk
        adjNode *minNode = &adaptAdjTable[sharedNodes[0]];
        int found=0;
        //search for the nodeSet in the list of adjacencies around minNode
        adjElem *rover =  searchAdjElemInList(
                minNode->adjElemList, 
                sharedNodes.getVec(), 
                nodeSetSize, 
                -1, 
                &found);
//        printf("[%d] Received request (%d,%d,%d,%d) minNode %d found %d\n",
//                myRank,receivedRequest.elemID,receivedRequest.chunkID,
//                receivedRequest.elemType,receivedRequest.nodeSetID,
//                sharedNodes[0],found);
        

        
        if (found) { 
            //we have found a matching adjElem for the requested nodeset
            //we shall set the adjacency correctly in the adjacency Table
            //for the elemID in the found adjElem. We need to send a reply
            //to the requesting chunk        	
            int matchingElemID = rover->next->elemID;

            if (isEdgeRequest) {
                // TODO: we cannot delete edge requests as we go because they
            	// may be needed again. Deallocate whole list at end.
        		while (found) {
//                    printf("[%d] (%d,%d,%d,%d) adds adj (%d %d %d) to elem %d edge %d\n",
//                            myRank,receivedRequest.elemID,receivedRequest.chunkID,
//                            receivedRequest.elemType,receivedRequest.nodeSetID,
//                            receivedRequest.chunkID, receivedRequest.elemID,
//                            receivedRequest.elemType, matchingElemID, rover->next->nodeSetID);
                    CkVec<adaptAdj>* matchingAdaptAdj = adaptEdgeAdjacencies[
                        matchingElemID*numAdjElems + rover->next->nodeSetID];
                    matchingAdaptAdj->push_back(adaptAdj(
                                receivedRequest.chunkID,
                                receivedRequest.elemID, 
                                receivedRequest.elemType));

                    rover = searchAdjElemInList(
        	                rover->next, 
        	                sharedNodes.getVec(), 
        	                nodeSetSize, 
        	                -1, 
        	                &found);
        			
        			if (found) {
            			matchingElemID = rover->next->elemID;
        			}
        		}
            } else {
                adaptAdj *matchingAdaptAdj = &adaptFaceAdjacencies[
                    matchingElemID*numAdjElems + rover->next->nodeSetID];
                CkAssert(matchingAdaptAdj->localID == -1);
                matchingAdaptAdj->partID = receivedRequest.chunkID;
                matchingAdaptAdj->localID = receivedRequest.elemID;
                matchingAdaptAdj->elemType = receivedRequest.elemType;

                adjElem *tmp = rover->next;
                rover->next = tmp->next;
                delete tmp;
            }
            
            //Set requesting data in reply to that in receivedRequest
            //Put in data from rover->next into the replyElem portion of data
            adjReply reply;
            reply.requestingElemID = receivedRequest.elemID;
            reply.requestingNodeSetID = receivedRequest.nodeSetID;
            reply.replyingElem.partID = myRank;
            reply.replyingElem.localID = matchingElemID;
            reply.replyingElem.elemType = elemType;
            //Write into the replyTable
            replyTable(receivedRequest.chunkID) += reply;
        } else {
            //we have no matching nodeset for this request.. hopefully some
            //other chunk does; we can ignore this request
        }
    }
}


inline void findNodeSet(
        int meshid,
        int elemType,
        int* faceSize, 
        int* faceMapSize, 
        int* edgeMapSize,
        int nodeSetMap[MAX_ADJELEMS][MAX_NODESET_SIZE],
        int edgeMap[MAX_EDGES][2])
{
  FEM_Mesh *mesh = FEM_chunk::get("GetAdaptAdj")->
    lookup(meshid,"GetAdaptAdj");
  FEM_Elem *elem = (FEM_Elem *)mesh->
    lookup(FEM_ELEM+elemType,"GetAdaptAdj");
  FEM_Node *node = (FEM_Node *)mesh->
    lookup(FEM_NODE,"GetAdaptAdj");
  const int nodesPer = (elem->getConn()).width();
  assert(node->getCoord()!= NULL);
  //assert(nodesPer > 0 && nodesPer < MAX_FACE_SIZE); 
  // I fail to see a connection between nodesPer and MAX_FACE_SIZE, and
  // for a tet nodesPer==4 while MAX_FACE_SIZE==4, so this test is
  // always going to fail, so I have removed it
  assert(nodesPer > 0);
  const int dim = (node->getCoord())->getWidth();
  assert(dim == 2|| dim == 3);
  
  guessElementShape(dim, nodesPer, 
		    faceSize, faceMapSize, edgeMapSize, 
		    nodeSetMap, edgeMap);
}

void getAndDumpAdaptAdjacencies(
        const int meshid, 
        const int numElems, 
        const int elemType, 
        const int myRank)
{
  int faceSize;
  int faceMapSize, edgeMapSize;
  int nodeSetMap[MAX_ADJELEMS][MAX_NODESET_SIZE]; // not used
  int edgeMap[MAX_EDGES][2]; // not used
  
  findNodeSet(meshid, elemType, &faceSize, &faceMapSize, &edgeMapSize,
	      nodeSetMap, edgeMap);
  
  for (int i=0; i<numElems; i++) {
    printf("[%d] %d  :",myRank,i);
    for (int j=0; j<faceMapSize; j++) {
      adaptAdj *entry = getAdaptAdj(meshid, i, elemType, j);
      printf("(%d,%d,%d)", entry->partID, entry->localID, entry->elemType);
    }
    printf("\n");
  }
  
  if (edgeMapSize == 0) return;
  for (int i=0; i<numElems; i++) {
    printf("[%d] %d  :", myRank, i);
    for (int j=0; j<edgeMapSize; j++) {
      CkVec<adaptAdj>* entry = getEdgeAdaptAdj(meshid, i, elemType, j);
      for (int k=0; k<entry->size(); ++k) {
    	  printf("(%d,%d,%d)", (*entry)[k].partID, (*entry)[k].localID, 
	       (*entry)[k].elemType);
      }
      if (j < (edgeMapSize-1)) printf(" | ");
    }
    printf("\n");
  }
}

// Access functions
adaptAdj* lookupAdaptAdjacencies(
        const FEM_Mesh* const mesh,
        const int elemType,
        int* numAdjacencies)
{
    FEM_Elem *elem = (FEM_Elem*)mesh->lookup(
            FEM_ELEM+elemType, "lookupAdaptAdjacencies");

    FEM_DataAttribute* adaptAttr = (FEM_DataAttribute*)elem->lookup(
            FEM_ADAPT_FACE_ADJ, "lookupAdaptAdjacencies");
    *numAdjacencies = adaptAttr->getWidth()/sizeof(adaptAdj);
    AllocTable2d<unsigned char> &table = adaptAttr->getChar();

    return (adaptAdj*)(adaptAttr->getChar().getData());
}

adaptAdj* lookupAdaptAdjacencies(
        const int meshid,
        const int elemType,
        int* numAdjacencies)
{
    FEM_Mesh* mesh = FEM_chunk::get("lookupAdaptAdjacencies")->lookup(
            meshid, "lookupAdaptAdjacencies");
    return lookupAdaptAdjacencies(mesh, elemType, numAdjacencies);
}

CkVec<adaptAdj>** lookupEdgeAdaptAdjacencies(
        const FEM_Mesh* const mesh,
        const int elemType,
        int* numAdjacencies)
{
    FEM_Elem *elem = (FEM_Elem*)mesh->lookup(
            FEM_ELEM+elemType,"lookupAdaptAdjacencies");
    FEM_DataAttribute* adaptAdjAttr = (FEM_DataAttribute*)elem->lookup(
            FEM_ADAPT_EDGE_ADJ, "CreateAdaptAdjacencies");	
    *numAdjacencies = adaptAdjAttr->getWidth()/sizeof(CkVec<adaptAdj>**);
    CkVec<adaptAdj>** adaptAdjacencies = 
        reinterpret_cast<CkVec<adaptAdj>**>((adaptAdjAttr->getChar()).getData());
    return adaptAdjacencies;
}

CkVec<adaptAdj>** lookupEdgeAdaptAdjacencies(
        const int meshID,
        const int elemType,
        int* numAdjacencies)
{
    FEM_Mesh *mesh = FEM_chunk::get("lookupAdaptAdjacencies")->
        lookup(meshID,"lookupAdaptAdjacencies");
    return lookupEdgeAdaptAdjacencies(mesh, elemType, numAdjacencies);
}

/** Look up elemID in elemType array, access edgeFaceID-th adaptAdj. */
adaptAdj* getAdaptAdj(
        const int meshID, 
        const int localID,
        const int elemType, 
        const int faceID)
{
    int numAdjacencies;
    adaptAdj* adaptAdjacencies = 
        lookupAdaptAdjacencies(meshID, elemType, &numAdjacencies);
    return &(adaptAdjacencies[localID*numAdjacencies+faceID]);
}

adaptAdj* getAdaptAdj(
        const FEM_Mesh* const meshPtr, 
        const int localID,
        const int elemType, 
        const int faceID)
{
    int numAdjacencies;
    adaptAdj* adaptAdjacencies = 
        lookupAdaptAdjacencies(meshPtr, elemType, &numAdjacencies);
    return &(adaptAdjacencies[localID*numAdjacencies+faceID]);
}

CkVec<adaptAdj>* getEdgeAdaptAdj(
        const int meshID, 
        const int localID,
        const int elemType, 
        const int edgeID)
{
    int numAdjacencies;
    CkVec<adaptAdj>** adaptAdjacencies = 
        lookupEdgeAdaptAdjacencies(meshID, elemType, &numAdjacencies);
    return adaptAdjacencies[localID*numAdjacencies+edgeID];
}

CkVec<adaptAdj>* getEdgeAdaptAdj(
        const FEM_Mesh* const meshPtr, 
        const int localID,
        const int elemType, 
        const int edgeID)
{
    int numAdjacencies;
    CkVec<adaptAdj>** adaptAdjacencies = 
        lookupEdgeAdaptAdjacencies(meshPtr, elemType, &numAdjacencies);
    return adaptAdjacencies[localID*numAdjacencies+edgeID];
}

/** Look up elemID in elemType array, calculate edgeFaceID from
    vertexList (with GetEdgeFace below), and access edgeFaceID-th
    adaptAdj with GetAdaptAdj above. */
adaptAdj* getAdaptAdj(
        const int meshID, 
        const int localID,
        const int elemType, 
        const int* const vertexList)
{
  int faceID = getElemFace(meshID, elemType, vertexList);
  return getAdaptAdj(meshID, localID, elemType, faceID);
}

CkVec<adaptAdj>* getEdgeAdaptAdj(
        const int meshID, 
        const int localID,
        const int elemType, 
        const int* const vertexList)
{
  int edgeID = getElemEdge(meshID, elemType, vertexList);
  return getEdgeAdaptAdj(meshID, localID, elemType, edgeID);
}

adaptAdj* getFaceAdaptAdj(
        const int meshID, 
        const int localID,
        const int elemType, 
        const int* const vertexList)
{
  return getAdaptAdj(meshID, localID, elemType, vertexList);
}

adaptAdj* getFaceAdaptAdj(
        const int meshID, 
        const int localID,
        const int elemType, 
        const int faceID)
{
    return getAdaptAdj(meshID, localID, elemType, faceID);
}

adaptAdj* getFaceAdaptAdj(
        const FEM_Mesh* const meshPtr, 
        const int localID,
        const int elemType, 
        const int faceID)
{
    return getAdaptAdj(meshPtr, localID, elemType, faceID);
}


/** Remove all neighbors on the specified edge */
void clearEdgeAdjacency(
		const FEM_Mesh* const meshPtr,
		const int localID,
		const int elemType,
		const int edgeID)
{
	getEdgeAdaptAdj(meshPtr, localID, elemType, edgeID)->removeAll();
}

void clearEdgeAdjacency(
		const int meshID,
		const int localID,
		const int elemType,
		const int edgeID)
{
    FEM_Mesh *mesh = FEM_chunk::get("lookupAdaptAdjacencies")->
        lookup(meshID,"lookupAdaptAdjacencies");
	clearEdgeAdjacency(mesh, localID, elemType, edgeID);
}

/** Add a new adjacency on the specified edge */
void addEdgeAdjacency(
		const FEM_Mesh* const meshPtr,
		const int localID,
		const int elemType,
		const int edgeID,
		const adaptAdj adj)
{
  CkAssert(adj.localID != -1);
  getEdgeAdaptAdj(meshPtr, localID, elemType, edgeID)->push_back(adj);
}

void addEdgeAdjacency(
		const int meshID,
		const int localID,
		const int elemType,
		const int edgeID,
		const adaptAdj adj)
{
  CkAssert(adj.localID != -1);
  FEM_Mesh *mesh = FEM_chunk::get("lookupAdaptAdjacencies")->
    lookup(meshID,"lookupAdaptAdjacencies");
  addEdgeAdjacency(mesh, localID, elemType, edgeID, adj);
}

/** Look up elemID in elemType array and determine the edge or face ID
    specified by the set of vertices in vertexList. */
int getElemFace(
        const int meshID, 
        const int type, 
        const int* vertexList)
{
  int faceSize;
  int faceMapSize;
  int faceMap[MAX_ADJELEMS][MAX_NODESET_SIZE];
  int edgeMapSize; // not used
  int edgeMap[MAX_EDGES][2]; // not used
  findNodeSet(meshID, type, &faceSize, &faceMapSize, &edgeMapSize,
         faceMap, edgeMap);
  
  // look for vertexList in the face map
  std::set<int> vertexSet(vertexList, vertexList+faceSize);
  for (int i=0; i<faceMapSize; i++) {
    std::set<int> aNodeSet(faceMap[i], faceMap[i]+faceSize);
    if (vertexSet == aNodeSet) return i;
  }

  // uh-oh, didn't find it
  CkAbort("ERROR: GetEdgeFace: vertexList is not a valid face./n");
  return -1;
}

int getElemEdge(
        const int meshID, 
        const int type, 
        const int* vertexList)
{
  int faceSize; // not used
  int faceMapSize; // not used
  int faceMap[MAX_ADJELEMS][MAX_NODESET_SIZE]; // not used
  int edgeMapSize;
  int edgeMap[MAX_EDGES][2];
  findNodeSet(meshID, type, &faceSize, &faceMapSize, &edgeMapSize,
         faceMap, edgeMap);
  
  // look for vertexList in the edge map
  std::set<int> vertexSet(vertexList, vertexList+2);
  for (int i=0; i<edgeMapSize; i++) {
    std::set<int> aNodeSet(edgeMap[i], edgeMap[i]+2);
    if (vertexSet == aNodeSet) return i;
  }

  // uh-oh, didn't find it
  CkAbort("ERROR: GetEdgeFace: vertexList is not a valid edge./n");
  return -1;
}


// Update functions
/** Look up elemID in elemType array and set the adjacency on
    edgeFaceID to nbr. */
void setAdaptAdj(
        const int meshID, 
        const adaptAdj elem, 
        const int faceID, 
        const adaptAdj nbr)
{
  int numAdjacencies;
  adaptAdj *adaptAdjTable = lookupAdaptAdjacencies(meshID, elem.elemType,
						   &numAdjacencies);
  adaptAdjTable[elem.localID*numAdjacencies + faceID] = nbr;
}

void setAdaptAdj(
        const FEM_Mesh* meshPtr,
        const adaptAdj elem, 
        const int faceID, 
        const adaptAdj nbr)
{
  int numAdjacencies;
  adaptAdj *adaptAdjTable = lookupAdaptAdjacencies(meshPtr, elem.elemType,
						   &numAdjacencies);
  adaptAdjTable[elem.localID*numAdjacencies + faceID] = nbr;
}

void addToAdaptAdj(
        const int meshid, 
        const adaptAdj elem, 
        const int edgeID, 
        const adaptAdj nbr)
{
  CkAssert(nbr.localID != -1);
  CkVec<adaptAdj>* adjVec = getEdgeAdaptAdj(meshid, elem.localID, 
					    elem.elemType, edgeID);
  adjVec->push_back(nbr);
}

void addToAdaptAdj(
        const FEM_Mesh* meshPtr, 
        const adaptAdj elem, 
        const int edgeID, 
        const adaptAdj nbr)
{
  CkAssert(nbr.localID != -1);
  CkVec<adaptAdj>* adjVec = getEdgeAdaptAdj(meshPtr, elem.localID, 
					    elem.elemType, edgeID);
  adjVec->push_back(nbr);
}

void removeFromAdaptAdj(
        const int meshid, 
        const adaptAdj elem, 
        const int edgeID, 
        const adaptAdj nbr)
{
    CkVec<adaptAdj>* adjVec = getEdgeAdaptAdj(meshid, elem.localID, 
            elem.elemType, edgeID);
    for (int i=0; i<adjVec->size(); ++i) {
        if ((*adjVec)[i] == nbr) {
            adjVec->remove(i);
            return;
        }
    }
    CkAbort("removeFromAdaptAdj did not find the specified nbr");
}


void copyAdaptAdj(
		const int meshid, 
		const adaptAdj* const srcElem, 
		const adaptAdj* const destElem)
{
    FEM_Mesh* meshPtr = FEM_chunk::get("ReplaceAdaptAdj")->lookup(
            meshid,"ReplaceAdaptAdj");
    copyAdaptAdj(meshPtr, srcElem, destElem);
}


void copyAdaptAdj(
		const FEM_Mesh* const meshPtr, 
		const adaptAdj* const srcElem, 
		const adaptAdj* const destElem)
{
    int numAdjacencies;
    adaptAdj* adaptAdjTable = lookupAdaptAdjacencies(meshPtr, srcElem->elemType,
            &numAdjacencies);
    memcpy(&adaptAdjTable[destElem->localID*numAdjacencies],
    		&adaptAdjTable[srcElem->localID*numAdjacencies],
    		numAdjacencies*sizeof(adaptAdj));
}


void copyEdgeAdaptAdj(
		const int meshid, 
		const adaptAdj* const srcElem, 
		const adaptAdj* const destElem)
{
    FEM_Mesh* meshPtr = FEM_chunk::get("ReplaceAdaptAdj")->lookup(
            meshid,"ReplaceAdaptAdj");
    copyEdgeAdaptAdj(meshPtr, srcElem, destElem);
}


void copyEdgeAdaptAdj(
		const FEM_Mesh* const meshPtr, 
		const adaptAdj* const srcElem, 
		const adaptAdj* const destElem)
{
    int nAdj;
    CkVec<adaptAdj>** adaptAdjTable = lookupEdgeAdaptAdjacencies(meshPtr, 
    		srcElem->elemType, &nAdj);
    
    CkVec<adaptAdj>** srcTable = &adaptAdjTable[srcElem->localID*nAdj];
    CkVec<adaptAdj>** dstTable = &adaptAdjTable[destElem->localID*nAdj];
    for (int i=0; i<nAdj; ++i) 
        *dstTable[i] = *srcTable[i]; // let CkVec operator= do the work
}



/** Lookup elemID in elemType array and search for the face which has 
 *  originalNbr as a neighbor, then replace originalNbr with newNbr
 */
void replaceAdaptAdj(
        const FEM_Mesh* const meshPtr, 
        const adaptAdj elem, 
        const adaptAdj originalNbr, 
        const adaptAdj newNbr)
{
  int numAdjacencies;
  adaptAdj *adaptAdjTable = lookupAdaptAdjacencies(meshPtr, elem.elemType,
						   &numAdjacencies);
  for(int i=0;i<numAdjacencies;i++){
    if(adaptAdjTable[elem.localID*numAdjacencies+i] == originalNbr){
      adaptAdjTable[elem.localID*numAdjacencies+i] = newNbr;
      return;
    }
  }
  CkAbort("replaceAdaptAdj did not find the specified originalNbr");
}


void replaceAdaptAdj(
        const int meshID, 
        const adaptAdj elem, 
        const adaptAdj originalNbr, 
        const adaptAdj newNbr)
{
  FEM_Mesh* meshPtr = FEM_chunk::get("ReplaceAdaptAdj")->lookup(
    meshID,"ReplaceAdaptAdj");
  replaceAdaptAdj(meshPtr, elem, originalNbr, newNbr);
}


void replaceAdaptAdjOnEdge(
        const FEM_Mesh* const meshPtr, 
        const adaptAdj elem, 
        const adaptAdj originalNbr,
        const adaptAdj newNbr,
        const int edgeID)
{
  int numAdjacencies;
  CkAssert(newNbr.localID != -1);
  CkVec<adaptAdj>** adaptAdjTable = lookupEdgeAdaptAdjacencies(
    meshPtr, elem.elemType, &numAdjacencies);
  CkVec<adaptAdj>* innerTable = 
    adaptAdjTable[elem.localID*numAdjacencies + edgeID];
  for (int n=0; n<innerTable->size(); ++n) {
    if((*innerTable)[n] == originalNbr){
      (*innerTable)[n] = newNbr;
      return;
    }
  }
  CkAbort("replaceAdaptAdjOnEdge did not find the specified originalNbr");
}


void replaceAdaptAdjOnEdge(
        const int meshID, 
        const adaptAdj elem, 
        const adaptAdj originalNbr, 
        const adaptAdj newNbr,
        const int edgeID)
{
  CkAssert(newNbr.localID != -1);
  FEM_Mesh* meshPtr = FEM_chunk::get("ReplaceAdaptAdj")->lookup(
    meshID,"ReplaceAdaptAdj");
  replaceAdaptAdjOnEdge(meshPtr, elem, originalNbr, newNbr, edgeID);
}


#define copyNodeMap(numEntries,entrySize,map,srcMap) \
do { \
  for (int i=0;i<numEntries;i++){ \
    for(int j=0;j<entrySize;j++){ \
      map[i][j] = srcMap[i][j]; \
    } \
  } \
} while (0) 


// Given the dimensions and nodes per element guess whether the element 
// is a triangle, quad, tet or hex. At the moment these are the 4 shapes
// that are handled  
void guessElementShape(
        const int dim, 
        const int nodesPerElem, 
        int* faceSize, 
        int *faceMapSize, 
        int* edgeMapSize,
        int faceMap[MAX_ADJELEMS][MAX_FACE_SIZE], 
        int edgeMap[MAX_EDGES][2])
{
    switch(dim){
        case 2:
            {
                //2 dimension
                switch (nodesPerElem) {
                    case 3:
                        //Triangles
                        *faceSize = 2;
                        *faceMapSize = 3;
                        *edgeMapSize = 0;
                        copyNodeMap(3,2,faceMap,faceMap2d_tri);
                        break;
                    case 4:
                        //quads
                        *faceSize = 2;
                        *faceMapSize = 4;
                        *edgeMapSize = 0;
                        copyNodeMap(4,2,faceMap,faceMap2d_quad);
                        break;
                    default:
                        CkPrintf("Unknown element type\n");
                        CkAssert(false);
                }
            }
            break;
        case 3:
            {
                //3 dimension
                switch(nodesPerElem){
                    case 4:
                        //Tetrahedra
                        *faceSize = 3;
                        *faceMapSize = 4;
                        *edgeMapSize = 6;
                        copyNodeMap(4,3,faceMap,faceMap3d_tet);
                        copyNodeMap(6,2,edgeMap,edgeMap3d_tet);
                        break;
                    case 8:
                        //Hexahedra
                        *faceSize = 4;
                        *faceMapSize = 6;
                        *edgeMapSize = 12;
                        copyNodeMap(6,4,faceMap,faceMap3d_hex);
                        copyNodeMap(12,2,edgeMap,edgeMap3d_hex);
                        break;
                    default:
                        CkPrintf("Unknown element type\n");
                        CkAssert(false);
                }
            }
            break;
        default:
            CkPrintf("Unknown element type\n");
            CkAssert(false);
    }
}

