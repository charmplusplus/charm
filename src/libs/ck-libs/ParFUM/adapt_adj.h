/* Adaptivity Adjacencies: element-to-element adjacencies for use by
   adaptivity codes only.  Adaptivity codes should keep these
   up-to-date for each mesh modification primitive.

   Created 11 Sept 2006 - Terry L. Wilmarth
   
   Format of adjacency information: (partitionID, localID on partition, type)
   
   2D Adjacencies: Adjacencies are across edges
   3D Adjacencies: Adjacencies can be across edges and faces

   Numberings:
   
   TRIANGLES: Given nodes 0, 1, 2, the edges 0, 1, and 2 of a triangle are:
   (0, 1), (1, 2) and (2, 0), in that order.
                                                     // Avoid '\' line splicing
              0                                                               *
             / \                                                              *
            /   \                                                             *
          2/     \0                                                           *
          /       \                                                           *
         /         \                                                          *
        2-----------1                                                         *
              1                                                               *
   
   QUADS: Given nodes 0, 1, 2, 3, the edges 0, 1, 2 and 3 of a quad are:
   (0, 1), (1, 2), (2, 3) and (3, 0), in that order.
   
   3D Adjacencies: 
   
   TETS: Given nodes 0, 1, 2, 3, 
   The faces (0-3) of a tetrahedra are:
   (0, 1, 2), (0, 3, 1), (0, 2, 3), and (1,3,2), in that order
   The edges (0-5) are:
   (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), in that order

              0                                                               *
             /|\                                                              *
            / | \                                                             *
          1/  0  \2       Back Face: 2                                        *
          / 0 | 1 \       Bottom Face: 3                                      *
         /....|....\                                                          *
        2-_   |   _-3     Back Edge: 5                                        *
           -_ | _-                                                            *
          3  -1-  4                                                           *

   
   HEXES: Given nodes 0, 1, 2, 3, 4, 5, 6, 7, 
   The faces 0, 1, 2, 3, 4, 5 of a hex are: 
   (0, 1, 2, 3), (1, 5, 6, 2), 
   (2, 6, 7, 3), (3, 7, 4, 0), 
   (0, 4, 5, 1), (5, 4, 6, 7) in that order
   The edges (0-11) are: 
   (0, 1), (0, 3), (0, 4), (1, 2), 
   (1, 5), (2, 3), (2, 6), (3, 7), 
   (4, 5), (4, 7), (5, 6), (6, 7), in that order
*/

#ifndef __ADAPT_ADJ_H__
#define __ADAPT_ADJ_H__

// NOTE: review for mixed and cohesive element handling
#include <set>
#include <algorithm>

#define MAX_ADJELEMS 6
#define MAX_FACE_SIZE 4
#define MAX_EDGES 12

#define MAX_NODESET_SIZE 4

// Each instance of adaptAdj represents an element to 
// element adjacency
class adaptAdj{
    public:
        int partID;   // partition ID
        int localID;  // local entity ID on partition partID
        int elemType; // element type (tri, quad, tet, hex, etc.)
        adaptAdj():partID(-1),localID(-1),elemType(-1){};
        adaptAdj(int _partID,int _localID,int _elemType) : 
            partID(_partID), 
            localID(_localID), 
            elemType(_elemType){};
        inline adaptAdj &operator=(const adaptAdj &rhs){
            partID = rhs.partID;
            localID = rhs.localID;
            elemType = rhs.elemType;
            return *this;
        }
        inline bool operator==(const adaptAdj &rhs) const{
            return (partID==rhs.partID && 
                    localID==rhs.localID && 
                    elemType==rhs.elemType);
        }
        inline bool operator!=(const adaptAdj &rhs) const{
            return (partID!=rhs.partID ||
                    localID!=rhs.localID || 
                    elemType!=rhs.elemType);
        }
        void pup(PUP::er &p){
            p | partID;
            p | localID;
            p | elemType;
        }
};

// Each adjElem describes an adjacency by enumerating
// the nodes that form the "edge" shared by two 
// adjacent elements
class adjElem { // list entry for an element incident on a node
    public:
        int elemID; // local element id
        int nodeSetID; // which nodeSet in nodeSetMap does this refer to
        CkVec<int> nodeSet; //local node ids
        adjElem *next;
        adjElem(int nodeSetSize):
            nodeSet(nodeSetSize){};
};

class adjNode { // struct to store each node's adjacency info
    public:	
        int *sharedWithPartition; // array of partition IDs on which there 
                                  // is a corresponding shared node; 
                                  // this is NULL if this is not a shared node
        int *sharedWithLocalIdx;  // local Idx in idxl list with the 
                                  // corresponding chunk in sharedWithPartition
        int numSharedPartitions;
        int adjElemCount;         // number of entries in adjElemList (below)
        // max length of adjElemList is 2*nodal degree
	//cppcheck-suppress unsafeClassCanLeak
        adjElem *adjElemList;     // list of elems incident on this node
        adjNode() { 
            sharedWithPartition = NULL;
            adjElemList = new adjElem(0); // Create a dummy head node in the 
                                          // list
            adjElemList->elemID = -1;
            adjElemList->next = NULL;
            adjElemCount = 0; 
            numSharedPartitions=0;
        }
        ~adjNode() { 
            //delete [] sharedWithPartition; 
            //delete [] sharedWithLocalIdx;
        }
};

class adjRequest{
 public:
  int elemID,chunkID,elemType,nodeSetID;
  int translatedNodeSet[MAX_NODESET_SIZE];
 adjRequest():
  elemID(-1), 
    chunkID(-1), 
    elemType(-1){};
 adjRequest(int _elemID,int _chunkID,int _nodeSetID,int _elemType ): 
  elemID(_elemID),
    chunkID(_chunkID),
    elemType(_elemType),
    nodeSetID(_nodeSetID) {};
  adjRequest(const adjRequest &rhs){
    *this = rhs;
  }
  inline adjRequest& operator=(const adjRequest &rhs) {
    elemID = rhs.elemID;
    chunkID = rhs.chunkID;
    elemType = rhs.elemType;
    nodeSetID = rhs.nodeSetID;
    memcpy(&translatedNodeSet[0],&(rhs.translatedNodeSet[0]),
	   MAX_NODESET_SIZE*sizeof(int));
    return *this;
  }
  void pup(PUP::er &p){
    p | elemID;
    p | chunkID;
    p | elemType;
    p | nodeSetID;
    p(translatedNodeSet,MAX_NODESET_SIZE);
  }
};

class adjReply {
 public:
  int requestingElemID,requestingNodeSetID;
  adaptAdj replyingElem;
 adjReply(): 
  requestingElemID(-1),
    requestingNodeSetID(-1), 
    replyingElem(){};
  adjReply(const adjReply &rhs){
    *this = rhs;
  }
  inline adjReply& operator=(const adjReply &rhs){
    requestingElemID = rhs.requestingElemID;
    requestingNodeSetID = rhs.requestingNodeSetID;
    replyingElem = rhs.replyingElem;
    return *this;
  }
  void pup(PUP::er &p){
    p | requestingElemID;
    p | requestingNodeSetID;
    replyingElem.pup(p);
  }
};


typedef ElemList<adjRequest> AdjRequestList;
typedef MSA::MSA1D<AdjRequestList, DefaultListEntry<AdjRequestList,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DREQLIST;

typedef ElemList<adjReply> AdjReplyList;
typedef MSA::MSA1D<AdjReplyList, DefaultListEntry<AdjReplyList,true>, MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DREPLYLIST;

/** Create Adaptivity Adjacencies for elemType; dimension inferred. */
void CreateAdaptAdjacencies(int meshid, int elemType);

// raw FEM attribute array accessors
adaptAdj* lookupAdaptAdjacencies(
        const FEM_Mesh* const mesh,
        const int elemType,
        int* numAdjacencies);
adaptAdj* lookupAdaptAdjacencies(
        const int meshid,
        const int elemType,
        int* numAdjacencies);
CkVec<adaptAdj>** lookupEdgeAdaptAdjacencies(
        const FEM_Mesh* const mesh,
        const int elemType,
        int* numAdjacencies);
CkVec<adaptAdj>** lookupEdgeAdaptAdjacencies(
        const int meshID,
        const int elemType,
        int* numAdjacencies);

// Access functions

// 2D accessors
adaptAdj *getAdaptAdj(
        const int meshID, 
        const int localID, 
        const int elemType,
        const int edgeID);
adaptAdj *getAdaptAdj(
        const FEM_Mesh* const meshPtr, 
        const int localID,
        const int elemType, 
        const int faceID);
adaptAdj *getAdaptAdj(
        const int meshID, 
        const int localID,
        const int elemType, 
        const int* const vertexList);

// 3D accessors
CkVec<adaptAdj>* getEdgeAdaptAdj(
        const int meshid, 
        const int localID,
        const int elemType,
        const int edgeID);
CkVec<adaptAdj>* getEdgeAdaptAdj(
        const FEM_Mesh* const meshPtr, 
        const int localID,
        const int elemType,
        int edgeID);
CkVec<adaptAdj>* getEdgeAdaptAdj(
        const int meshID, 
        const int localID,
        const int elemType, 
        const int* const vertexList);

adaptAdj* getFaceAdaptAdj(
        const int meshID,
        const int localID,
        const int elemType, 
        int faceID);
adaptAdj* getFaceAdaptAdj(
        const FEM_Mesh* const meshPtr,
        const int localID,
        const int elemType,
        int faceID);
adaptAdj* getFaceAdaptAdj(
        const int meshID, 
        const int localID,
        const int elemType, 
        const int* const vertexList);

// Adjacency manipulation functions
void clearEdgeAdjacency(
		const FEM_Mesh* const meshPtr,
		const int localID,
		const int elemType,
		const int edgeID);
void clearEdgeAdjacency(
		const int meshID,
		const int localID,
		const int elemType,
		const int edgeID);
void addEdgeAdjacency(
		const FEM_Mesh* const meshPtr,
		const int localID,
		const int elemType,
		const int edgeID,
		const adaptAdj adj);
void addEdgeAdjacency(
		const int meshID,
		const int localID,
		const int elemType,
		const int edgeID,
		const adaptAdj adj);

/** Look up elemID in elemType array and determine the set of vertices
    associated with the edge or face represented by edgeFaceID. */
void GetVertices(int meshid, adaptAdj elem, int edgeFaceID, int *vertexList);

/** Look up elemID in elemType array and determine the edge or face ID
    specified by the set of vertices in vertexList. */
int getElemFace(
        const int meshID, 
        const int type, 
        const int* vertexList);
int getElemEdge(
        const int meshID, 
        const int type, 
        const int* vertexList);

// Update functions
/** 2D or 3D (faces): Look up elemID in elemType array and set the adjacency on
    edgeFaceID to nbr. */
void setAdaptAdj(
        const int meshid, 
        const adaptAdj elem, 
        const int faceID, 
        const adaptAdj nbr);

void setAdaptAdj(
        const FEM_Mesh* meshPtr, 
        const adaptAdj elem, 
        const int faceID, 
        const adaptAdj nbr);

/** 3D: Look up elemID in elemType array and add nbr to the adjacency on
    edgeID. */
void addToAdaptAdj(
        const int meshid, 
        const adaptAdj elem, 
        const int edgeID, 
        const adaptAdj nbr);
void addToAdaptAdj(
        const FEM_Mesh* meshPtr, 
        const adaptAdj elem, 
        const int edgeID, 
        const adaptAdj nbr);
/** 3D: Look up elemID in elemType array and remove nbr from the adjacency on
    edgeID. */
void removeFromAdaptAdj(
        const int meshid, 
        const adaptAdj elem, 
        const int edgeID, 
        const adaptAdj nbr);

/** Copy adjacency information for one element to another.
*/
void copyAdaptAdj(
		const int meshid, 
		const adaptAdj* const srcElem, 
		const adaptAdj* const destElem);
void copyAdaptAdj(
		const FEM_Mesh* const meshPtr, 
		const adaptAdj* const srcElem, 
		const adaptAdj* const destElem);

void copyEdgeAdaptAdj(
		const int meshid, 
		const adaptAdj* const srcElem, 
		const adaptAdj* const destElem);
void copyEdgeAdaptAdj(
		const FEM_Mesh* const meshPtr, 
		const adaptAdj* const srcElem, 
		const adaptAdj* const destElem);


/** Substitute an old neighbor with a new neighbor, assumes 2D or 3D-face 
    neighbor */
void replaceAdaptAdj(
        const int meshID, 
        const adaptAdj elem, 
        const adaptAdj originalNbr, 
        const adaptAdj newNbr);
void replaceAdaptAdj(
        const FEM_Mesh* const meshPtr, 
        const adaptAdj elem, 
        const adaptAdj originalNbr,
        const adaptAdj newNbr);
/** 3D edge neighbors: Substitution operation needs to know edgeID to reduce 
    search space. */
void replaceAdaptAdjOnEdge(
        const int meshID, 
        const adaptAdj elem, 
        const adaptAdj originalNbr, 
        const adaptAdj newNbr,
        const int edgeID);
void replaceAdaptAdjOnEdge(
        const FEM_Mesh* const meshPtr, 
        const adaptAdj elem, 
        const adaptAdj originalNbr, 
        const adaptAdj newNbr,
        const int edgeID);

/** Given the dimensions and nodes per element guess whether the element 
    is a triangle, quad, tet or hex. At the moment these are the 4 shapes
    that are handled */
void guessElementShape(
        const int dim,
        const int nodesPerElem,
        int* faceSize, 
        int *faceMapSize, 
        int* edgeMapSize,
        int faceMap[MAX_ADJELEMS][MAX_FACE_SIZE],
        int edgeMap[MAX_EDGES][2]);
void getAndDumpAdaptAdjacencies(
        const int meshid, 
        const int numElems, 
        const int elemType, 
        const int myRank);

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
        const int elemType);

void makeAdjacencyRequests(
        const int numNodes, 
        FEM_Node* node, 
        adjNode* adaptAdjTable,
        MSA1DREQLIST::Accum &requestTable, 
        const int nodeSetSize, 
        const int myRank,
        int elemType);
void replyAdjacencyRequests(
        CkVec<adjRequest> *requests, 
        MSA1DREPLYLIST::Accum &replyTable,
        FEM_Node* node, 
        adjNode* adaptAdjTable, 
        adaptAdj* adaptFaceAdjacencies, 
        CkVec<adaptAdj>** adaptEdgeAdjacencies, 
        const int nodeSetSize,
        const int numAdjElems, 
        const int myRank, 
        const int elemType,
        bool isEdgeRequest);

#endif
