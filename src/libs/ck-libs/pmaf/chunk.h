/** \mainpage 
  Parallel Mesh Adaptivity Framework - 3D

  Created by: Terry L. Wilmarth
*/
#ifndef CHUNK_H
#define CHUNK_H

#include <vector.h>
#include "charm++.h"
#include "ref.h"
#include "node.h"
#include "element.h"
#include "messages.h"
//#include "tcharm.h"
//#include "charm-api.h"
#include "PMAF.decl.h"

// ------------------------ Global Read-only Data ---------------------------
extern CProxy_chunk mesh;
CtvExtern(chunk *, _refineChunk);

// The user inherits from this class to receive mesh updates.
// This code needs to be reworked for PMAF3D. 
class refineClient {
public:
  virtual ~refineClient() {}
  // This tetrahedron of our chunk is being split at a central point.
  // Client's responsibilities: need to figure this out
  virtual void splinter(int triNo, double x, double y, double z) =0;

  // These 2 tetrahedra of our chunk are collapsing along an edge
  // defined by two points.  Client's responsibilities: need to figure
  // this out
  virtual void collapse(int triNo1, int triNo2, int pt1, int pt2) =0;

  // This point has improved its location to [x,y,z].
  // Client's responsibilities: need to figure this out
  virtual void movePoint(int pt, double x, double y, double z) =0;
};

class refineResults; //Used by refinement API to store intermediate results

typedef struct prioLockStruct {
  int holder;
  double prio;
  prioLockStruct *next;
} *prioLockRequests;
// ---------------------------- Chare Arrays -------------------------------
class chunk : public ArrayElement1D {
  // Data fields for this chunk's array index and counts of elements
  // and nodes located on this chunk
  int cid, numElements, numNodes, numChunks;

  // current sizes of arrays allocated for the mesh
  int sizeElements, sizeNodes;

  // information about mesh status
  int additions, coordsRecvd;
  
  // debug_counter is used to print successive snapshots of the chunk
  // and match them up to other chunk snapshots; refineInProgress
  // flags that the refinement loop is active; modified flags that a
  // target volume for some element on this chunk has been modified
  int debug_counter, refineInProgress, coarsenInProgress, modified;

  int accessLock, adjustLock;
 public:
  int lock, lockHolder, lockCount;
  double lockPrio;
  double smoothness;
  prioLockRequests lockList;
  // the chunk's components, left public for sanity's sake
  std::vector<element> theElements;
  std::vector<node> theNodes;
  std::vector<std::vector<int> > theSurface;

  // client to report refinement split information to
  refineClient *theClient;
  refineResults *refineResultsStorage;

  // Basic constructor
  chunk(int nChunks);
  chunk(CkMigrateMessage *) { };
  
  // entry methods
  void refineElement(int idx, double volume);
  void refineElement(int idx);
  void refiningElements();	
  void coarsenElement(int idx, double volume);
  void coarseningElements();	
  void improveMesh();
  void relocatePoints();
  void flippingElements();	  

  intMsg *lockChunk(int lh, double prio);
  void unlockChunk(int lh);
  int lockLocalChunk(int lh, double prio);
  void unlockLocalChunk(int lh);
  void removeLock(int lh);
  void insertLock(int lh, double prio);

  // mesh debug I/O
  void print();
  void out_print();
  
  // entries to node data
  nodeMsg *getNode(int n);
  void updateNodeCoord(nodeMsg *);
  void relocationVote(nodeVoteMsg *);
  
  // entries to element data
  doubleMsg *getVolume(intMsg *im);
  void setTargetVolume(doubleMsg *);
  void resetTargetVolume(doubleMsg *);
  elemRef findNeighbor(nodeRef nr1, nodeRef nr2, nodeRef nr3, int lidx);
  refMsg *findRemoteNeighbor(threeNodeMsg *);
  intMsg *checkFace(int idx, elemRef face);
  intMsg *checkFace(int idx, node n1, node n2, node n3, elemRef nbr);
  intMsg *lockLF(int idx, node n1, node n2, node n3, node n4, 
		 elemRef requester, double prio);
  splitResponse *splitLF(int idx,node in1, node in2, node in3, node in4,
		       elemRef requester);
  LEsplitResult *LEsplit(LEsplitMsg *);
  lockResult *lockArc(lockArcMsg *lm);
  void unlockArc1(int idx, int prio, elemRef parentRef, elemRef destRef, node aNode, 
		 node bNode);
  void unlockArc2(int idx, int prio, elemRef parentRef, elemRef destRef, node aNode, 
		 node bNode);
  void updateFace(int idx, int rcid, int ridx);
  void updateFace(int idx, elemRef oldElem, elemRef newElem);
  flip23response *flip23remote(flip23request *);
  flip32response *chunk::flip32remote(flip32request *fr);
  flip32response *chunk::remove32element(flip32request *fr);

  // local methods
  void debug_print(int c); // prints a snapshot of the chunk to file

  // helpers
  void splitAll(nodeRef le1, nodeRef le2, nodeRef mid, int elemId);
  nodeRef findNode(node n);

  // surface maintenance
  int nodeOnSurface(int n);
  int edgeOnSurface(int n1, int n2);
  int faceOnSurface(int n1, int n2, int n2);
  void updateFace(int n1, int n2, int n3, int oldNode, int newNode);
  void addFace(int n1, int n2, int n3);
  void removeFace(int n1, int n2, int n3);
  void simpleAddFace(int n1, int n2, int n3);
  void simpleUpdateFace(int n1, int n2, int n3, int newNode);
  void simpleRemoveFace(int n1, int n2, int n3);
  void printSurface();
  void printEdgeLists();

  // mesh access control methods
  void getAccessLock();
  void forcedGetAccessLock();
  void releaseAccessLock();
  void getAdjustLock();
  void releaseAdjustLock();

  // these methods allow for run-time additions/modifications to the chunk
  void allocMesh(int nEl);
  void adjustMesh();
  nodeRef addNode(node& n);
  elemRef addElement(nodeRef& nr1, nodeRef& nr2, nodeRef& nr3, nodeRef& nr4);
  void removeNode(intMsg *);
  void removeElement(intMsg *);

  // FEM interface methods
  // newMesh specifies the elements and connectivity for a chunk
  void newMesh(int nEl, int nGhost,const int *conn_,const int *gid_, 
	       int *surface, int nSurFaces, int idxOffset);
  // updateNodeCoords sets node coordinates to new values
  void updateNodeCoords(int nNode, double *coord, int nEl, int nFx,int *fixed);
  // refine sets target volumes of elements and starts refining; rest similar
  void deriveFaces();
  void refine(double *desiredVolume, refineClient *client);
  void coarsen(double *desiredVolume, refineClient *client);
  void improve(refineClient *client);

  // entries to get data in mesh in stand-alone mode
  void newMesh(meshMsg *);	
  void updateNodeCoords(coordMsg *);
  void refine();
  void start();
  void improve();
  void finalizeImprovements();
  void checkRefine();
};

#endif
