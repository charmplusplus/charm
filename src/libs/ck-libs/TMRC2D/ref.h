#ifndef REF_H
#define REF_H
#include "charm++.h"

class objRef { // a reference to a piece of data that may be remotely located
 public:
  int cid, idx;
  objRef() { cid = -1;  idx = -1; }
  objRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void set(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void reset() { cid = -1; idx = -1; }
  int isNull() { return ((cid == -1) && (idx == -1)); }
  bool operator==(const objRef& o) const { return((cid == o.cid) && (idx == o.idx)); }
  bool operator>(const objRef& o) const { return((idx > o.idx) || 
					  ((idx == o.idx) && (cid > o.cid))); }
  bool operator<(const objRef& o) const { return((idx < o.idx) || 
					  ((idx == o.idx) && (cid < o.cid))); }
  objRef& operator=(const objRef& o) { cid=o.cid; idx=o.idx; return *this; }
  void sanityCheck() {
    if (isNull()) CkAbort("REFINE2D objRef is unexpectedly null");
    else if ((cid < -1) || (idx < -1)) 
      CkAbort("REFINE2D objRef has an insane value");
  }
};

class node;
class elemRef;
class edgeRef;

class nodeRef : public objRef {
 public:
  nodeRef() { }
  nodeRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  node get();
  void setBorder();
  void update(node& m);
  void reportPos(node& m);
  void remove();
  int lock();
  void unlock();
  int safeToMove(node& m, elemRef& E0, edgeRef& e0, edgeRef& e1, 
		 nodeRef& n1, nodeRef& n2, nodeRef& n3);
};

class edgeRef : public objRef {
 public:
  edgeRef() { }
  edgeRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void update(nodeRef& oldval, nodeRef& newval);
  void update(elemRef& oldval, elemRef& newval);
  elemRef get(elemRef& m);
  nodeRef get(nodeRef nr);
  nodeRef getNot(nodeRef nr);
  elemRef getNot(elemRef er);
  void midpoint(node& result);
  int setPending();
  void unsetPending();
  int isPending();
  void remove();
  int split(nodeRef *m, edgeRef *e_prime, nodeRef othernode, elemRef eRef);
  void checkPending(elemRef e);
  void checkPending(elemRef e, elemRef ne);
};

class elemRef : public objRef {
 public:
  elemRef() { }
  elemRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void update(edgeRef& oldval, edgeRef& newval);
  edgeRef getEdge(edgeRef eR, nodeRef nR);
  int isLongestEdge(edgeRef& e);
  double getArea();
  void setTargetArea(double ta);
  void resetTargetArea(double ta);
  void update(edgeRef& e0, edgeRef& e1, edgeRef& e2);
  nodeRef getOpnode(const edgeRef& e);
  void remove();
  void collapseHelp(edgeRef er, nodeRef nr1, nodeRef nr2);
};

static elemRef nullRef(-1,-1);

#endif
