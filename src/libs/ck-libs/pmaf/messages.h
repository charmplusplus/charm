// Parallel Mesh Adaptivity Framework - 3D
// Created by: Terry L. Wilmarth
#include "charm++.h"

// ------------------------------ Messages ----------------------------------
// nodeMsg: coordinates of a node
class nodeMsg : public CMessage_nodeMsg {
 public:
  int idx;
  double coord[3];
};

// nodeMsg: coordinates of a node
class nodeVoteMsg : public CMessage_nodeVoteMsg {
 public:
  double oldCoord[3];
  double newCoord[3];
};

// faceMsg: passes face coordinates
class faceMsg : public CMessage_faceMsg {
 public:
  int idx;
  double nodeCoords[3][3];
};

// updateMsg: update a reference for a node or edge in element idx
class updateMsg : public CMessage_updateMsg {
 public:
  int idx;
  objRef oldval, newval;
};

// refMsg: for returning refs from entry methods
class refMsg : public CMessage_refMsg {
 public:
  int idx, cid;
};

// intMsg: used to send parameterless messages to an element anInt
class intMsg : public CMessage_intMsg {
 public:
  int anInt;
};

// doubleMsg: used to send a double to an element idx
class doubleMsg : public CMessage_doubleMsg {
 public:
  int idx;
  double aDouble;
};

class meshMsg : public CMessage_meshMsg {
 public:
  int numElements, numGhosts, idxOffset, numSurFaces;
  int *conn;
  int *gid;
  int *surface;
};

class coordMsg : public CMessage_coordMsg {
 public:
  int numNodes, numElements, numFixed;
  double *coords;
  int *fixedNodes;
};

class threeNodeMsg : public CMessage_threeNodeMsg {
 public:
  double coords[3][3];
};

class splitResponse : public CMessage_splitResponse {
 public:
  int success, ance, bnce;
  double newNode[3];
};

class flip23request : public CMessage_flip23request {
 public:
  node a, b, c, d;
  elemRef acd, bcd, requester;
  int requestee;
};

class flip23response : public CMessage_flip23response {
 public:
  node e;
  elemRef abe, acde, requester;
  int requestee;
};

class flip32request : public CMessage_flip32request {
 public:
  node a, b, d, e;
  elemRef abe, bce, bcde, requester;
  int requestee;
};

class flip32response : public CMessage_flip32response {
 public:
  node c;
  elemRef acd, bcd, bce;
};

class LEsplitMsg : public CMessage_LEsplitMsg {
 public:
  int idx;
  double targetVol;
  elemRef root, parent, newRootElem, newElem, targetElem;
  nodeRef newNodeRef;
  node newNode, a, b;
};

class LEsplitResult : public CMessage_LEsplitResult {
 public:
  elemRef newElem1, newElem2;
  int status;
};

class lockMsg : public CMessage_lockMsg {
 public:
  int idx;
};

class lockResult : public CMessage_lockResult {
 public:
  int result;
};

class lockArcMsg : public CMessage_lockArcMsg {
 public:
  int idx;
  double prio;
  elemRef prioRef, parentRef, destRef;
  node a, b;
};
