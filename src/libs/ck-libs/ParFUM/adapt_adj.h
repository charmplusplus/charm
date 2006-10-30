/* Adaptivity Adjacencies: element-to-element adjacencies for use by
   adaptivity codes only.  Adaptivity codes should keep these
   up-to-date for each mesh modification primitive.

   Created 11 Sept 2006 - Terry L. Wilmarth
   
   Format of adjacency information:  (partition ID, local ID on partition, type)
   
   2D Adjacencies: 
   
   TRIANGLES: Given nodes 0, 1, 2, the edges 0, 1, and 2 of a triangle are:
   (0, 1), (1, 2) and (2, 0), in that order.
   
   QUADS: Given nodes 0, 1, 2, 3, the edges 0, 1, 2 and 3 of a quad are:
   (0, 1), (1, 2), (2, 3) and (3, 0), in that order.
   
   3D Adjacencies: 
   
   TETS: Given nodes 0, 1, 2, 3, the faces 0, 1, 2 and 3 of a tetrahedra are:
   (0, 1, 2), (1, 0, 3), (1, 3, 2), and (0, 2, 3), in that order
   
   HEXES: Given nodes 0, 1, 2, 3, 4, 5, 6, 7, the faces 0, 1, 2, 3, 4, 5
   of a hex are: (0, 1, 2, 3), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
   (0, 4, 5, 1), (5, 4, 6, 7) in that order
   
*/
#include "ParFUM.h"
#include "ParFUM_internals.h"
#include <set>
#include <algorithm>

#define MAX_NODESET_SIZE 6

// Each instance of adaptAdj represents an element to 
// element adjacency
class adaptAdj{
public:
  int partID;   // partition ID
  int localID;  // local entity ID on partition partID
  int elemType; // element type (tri, quad, tet, hex, etc.)
	adaptAdj():partID(-1),localID(-1),elemType(-1){};
	adaptAdj(int _partID,int _localID,int _elemType) : partID(_partID), localID(_localID), elemType(_elemType){};
};

// Each adjElem describes an adjacency by enumerating
// the nodes that form the "edge" shared by two 
// adjacent elements
class adjElem { // list entry for an element incident on a node
public:
  int elemID; // local element id
	int nodeSetID; // which nodeSet in nodeSetMap does this nodeSet refer to
	CkVec<int> nodeSet; //local node ids
  adjElem *next;
	adjElem(int nodeSetSize) : nodeSet(nodeSetSize){};
};

class adjNode { // struct to store each node's adjacency info
public:	
  int *sharedWithPartition; // array of partition IDs on which there is a corresponding
                            // shared node; this is NULL if this is not a shared node
	int *sharedWithLocalIdx;  // local Idx in idxl list with the corresponding chunk in sharedWithPartition
	int numSharedPartitions;
  int adjElemCount;         // number of entries in adjElemList (below)
														// max length of adjElemList is 2*nodal degree
  adjElem *adjElemList;     // list of elems incident on this node
  adjNode() { sharedWithPartition = NULL; adjElemList = NULL; adjElemCount = 0; numSharedPartitions=0;}
	~adjNode(){ delete [] sharedWithPartition; delete [] sharedWithLocalIdx;}
};

class adjRequest{
public:
	int elemID,chunkID,elemType,nodeSetID;
	int translatedNodeSet[MAX_NODESET_SIZE];
	adjRequest(): elemID(-1),chunkID(-1),elemType(-1){};
	adjRequest(int _elemID,int _chunkID,int _nodeSetID,int _elemType ): elemID(_elemID),chunkID(_chunkID),nodeSetID(_nodeSetID), elemType(_elemType){};
	adjRequest(const adjRequest &rhs){
		*this = rhs;
	}
	inline adjRequest& operator=(const adjRequest &rhs){
		elemID = rhs.elemID;
		chunkID = rhs.chunkID;
		elemType = rhs.elemType;
		memcpy(&translatedNodeSet[0],&(rhs.translatedNodeSet[0]),MAX_NODESET_SIZE*sizeof(int));
		return *this;
	}
	inline bool operator==(const adjRequest &rhs){
		return chunkID == rhs.chunkID && elemID == rhs.elemID && nodeSetID == rhs.nodeSetID;
	}
	inline bool operator<=(const adjRequest &rhs){
		if(chunkID < rhs.chunkID){ return true;}
		if(chunkID > rhs.chunkID){ return false;}

		if(elemID < rhs.elemID){ return true;}
		if(elemID > rhs.elemID){ return false;}
		
		if(nodeSetID < rhs.nodeSetID){ return true;}
		if(nodeSetID > rhs.nodeSetID){ return false;}

		return true;
	}
	inline bool operator>=(const adjRequest &rhs){
		if(*this == rhs){return true;}
		if(*this <= rhs){return false;}else{return true;}
	}
	virtual void pup(PUP::er &p){
	 p | elemID;
	 p | chunkID;
	 p | elemType;
	 p | nodeSetID;
	 p(translatedNodeSet,MAX_NODESET_SIZE);
	}
};

typedef ElemList<adjRequest> AdjRequestList;
typedef MSA1D<AdjRequestList, DefaultListEntry<AdjRequestList,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DREQLIST;

/** Create Adaptivity Adjacencies for elemType; dimension inferred. */
void CreateAdaptAdjacencies(int meshid, int elemType);

// Access functions

/** Look up elemID in elemType array, access edgeFaceID-th adaptAdj. */
adaptAdj *GetAdaptAdj(int meshid, int elemID, int elemType, int edgeFaceID);
/** Look up elemID in elemType array, calculate edgeFaceID from vertexList (with
    GetEdgeFace below), and access edgeFaceID-th adaptAdj with GetAdaptAdj above. */
adaptAdj *GetAdaptAdj(int meshid, int elemID, int elemType, int *vertexList);

/** Look up elemID in elemType array and determine the set of vertices
    associated with the edge or face represented by edgeFaceID. */
void GetVertices(int meshid, int elemID, int elemType, int edgeFaceID, int *vertexList);
/** Look up elemID in elemType array and determine the edge or face ID specified by
    the set of vertices in vertexList. */
int GetEdgeFace(int meshid, int elemID, int elemType, int *vertexList);

// Update functions
/** Look up elemID in elemType array and set the adjacency on edgeFaceID to nbr. */
void SetAdaptAdj(int meshid, int elemID, int elemType, int edgeFaceID, adaptAdj nbr);


/**given the dimensions and nodes per element guess whether the element 
 is a triangle, quad, tet or hex. At the moment these are the 4 shapes
 that are handled */
void guessElementShape(int dim,int nodesPerElem,int *numAdjElems,int *nodeSetSize,int ***nodeSetMap);

