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

typedef struct adaptAdjStruct {
  int partID;   // partition ID
  int localID;  // local entity ID on partition partID
  int elemType; // element type (tri, quad, tet, hex, etc.)
} adaptAdj;

class adjElem { // list entry for an element incident on a node
public:
  int elemID; // local element id
  int nodeSet[4]; // quad faces is probably the max we will ever deal with
  adjElem *next;
};

class adjNode { // struct to store each node's adjacency info
public:	
  int *sharedWithPartition; // array of partition IDs on which there is a corresponding
                            // shared node; this is NULL if this is not a shared node
  int adjElemCount;         // number of entries in adjElemList (below)
  adjElem *adjElemList;     // list of elems incident on this node
  adjNode() { sharedWithPartition = NULL; adjElemList = NULL; adjElemCount = 0; }
};

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
