/* Adaptivity Adjacencies: element-to-element adjacencies for use by
   adaptivity codes only.  Adaptivity codes should keep them
   up-to-date for each mesh modification primitive.
   
   Created 11 Sept 2006 - Terry L. Wilmarth
*/
#include "ParFUM.h"
#include "ParFUM_internals.h"
#include "adapt_adj.h"

int nodeSetMap2d_tri[3][2] = {{0,1},{1,2},{2,0}};
int nodeSetMap2d_quad[4][2] = {{0,1},{1,2},{2,3},{3,0}};
int nodeSetMap3d_tet[4][3] = {{0,1,2},{1,0,3},{1,3,2},{0,2,3}};
int nodeSetMap3d_hex[6][4] = {{0,1,2,3},{1,5,6,2},{2,6,7,3},{3,7,4,0},{0,4,5,1},{5,4,6,7}};


//given the dimensions and nodes per element guess whether the element 
// is a triangle, quad, tet or hex. At the moment these are the 4 shapes
// that are handled
  
inline void guessElementShape(int dim,int nodesPerElem,int *numAdjElems,int *nodeSetSize,int ***nodeSetMap){
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


/** Create Adaptivity Adjacencies for elemType; dimension inferred. */
void CreateAdaptAdjacencies(int meshid, int elemType)
{
  // Need to derive all of these from elemType;
  
  int numElems, numNodes;
  int nodesPerElem;
  int numAdjElems;
  int nodeSetSize; // number of nodes shared by two adjacent elems
  int dim;

  FEM_Mesh *mesh = FEM_chunk::get("CreateAdaptAdjacencies")->lookup(meshid,"CreateAdaptAdjacencies");
  FEM_Elem *elem = (FEM_Elem *)mesh->lookup(FEM_ELEM+elemType,"CreateAdaptAdjacencies");
  FEM_Node *node = (FEM_Node *)mesh->lookup(FEM_NODE,"CreateAdaptAdjacencies");
  numElems = elem->size();
  numNodes = node->size();
  nodesPerElem = (elem->getConn()).width();
  assert(node->getCoord()!= NULL);
  dim = (node->getCoord())->getWidth();
  assert(dim == 2|| dim == 3);

  
  // A nodeSet is a set of nodes that defines a pairing of two adjacent elements;
  // For example, in 2D triangle meshes, the nodeSet is the nodes of an edge between
  // two elements.
  // The nodeSetMap is an ordering of element-local node IDs that specifies all 
  // possible nodeSets for a particular element type
  int **nodeSetMap;
  guessElementShape(dim,nodesPerElem,&numAdjElems,&nodeSetSize,&nodeSetMap);
  

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
  // DO THIS! add a valid check
  
  // Pull out conn for elems of elemType
  int *conn;
  conn = (elem->setConn()).getData();
  
  for (int i=0; i<numElems; i++) { // Add each element-nodeSet pair to the table
    // ADD is_valid test for this element! DO THIS!
    for (int j=0; j<numAdjElems; j++) { // There is one nodeSet per neighbor element
      adjElem *e = new adjElem(nodeSetSize);
      e->nodeSetID = j;
      for (int k=0; k<nodeSetSize; k++) { // Build the nodeSet for an element pairing
        e->nodeSet[k] = conn[i*nodesPerElem+nodeSetMap[j][k]];
      }
      // Add this element-nodeSet pair to the table at the min nodeID in the nodeSet
      e->nodeSet.quickSort();
      int minNode = e->nodeSet[0];
      e->elemID = i;
      e->next = adaptAdjTable[minNode].adjElemList;
      adaptAdjTable[minNode].adjElemList = e;
      adaptAdjTable[minNode].adjElemCount++;
    }
  }

  for (int i=0; i<numNodes; i++) { 
    // For each node, match up incident elements
    //DO THIS! do a valid test or adjElemList = NULL
    adjElem *adjStart = adaptAdjTable[i].adjElemList;
    adjElem *rover = adjStart; // compare rover->next with adjStart
    adjElem *preStart = adjStart; //pointer before adjStart so that we can delete adjStart
                                  //Note: as long as adjStart is the first element in adjElemList
                                  //preStart = adjStart. After that preStart->next = adjStart
    while (adjStart) { //each entry in the adjElemList of a node 
      int found = 0; 
      while (rover->next) {
        if (rover->next->elemID != adjStart->elemID) {
          found = 1; // found an element that is not myself, possibly a match
          for (int j=0; j<nodeSetSize; j++) {
            if (rover->next->nodeSet[j] != adjStart->nodeSet[j]) {
              found = 0; // No, the nodeSets dont match
              break;
            }
          }
        }
        if (found) {
          break; // We have found a nodeSet that matches adjStart
        }else {
          rover = rover->next; // Keep looking in adjElemList for matching nodeSet
        }
      }
      if (found) {
        // We found an adjacent element for adjStart->elemID
        
        // Set adjacency of adjStart->elemID corresponding to nodeSet to 
        // rover->next->elemID, and vice versa
        // DO THIS!
        // Store it in adaptAdjacency of each one and use nodeSetID to index into 
        // adaptAdjacency
  
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
      rover = adjStart;
    }
  }

  // Now all elements' local adjacencies are set; remainder in table are 
  // nodeSets shared with other chunks or nodeSets on domain boundary

  //   For each node with remaining element-nodeSet entries:
  //     Determine if ALL nodes in nodeSet are shared with one other partition
  //     If not, remove the pair (element has no adjacency, on domain boundary)
  //     If all nodes shared, find the one partition P with all the nodes
  //       Get indexesOfNodeSet, send it to P with elemID, elemType and local partID
  //       Increment a counter of sends
  // Receive counter amount of nbr info
  //   With the indexesOfNodeSet and remotePartID, develop a new nodeSet
  //   Look up nodeSet under the min node, find matching nodeSet with unused localElem
  //   Add (remoteElemID, remoteElemType, remotePartID) to localElems adjacency
  //   Mark localElem as used
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
