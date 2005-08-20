#ifndef REF_H
#define REF_H
#include <charm++.h>
#include "charm-api.h"
#include "ckvector3d.h"
#include "pup_mpi.h"
#include "tcharm.h"
#include "fem.h"
#include "fem_mesh.h"

//#define FLIPTEST
//#define FLIPPREVENT

class objRef { // a reference to a piece of data that may be remotely located
 public:
  int cid, idx;
  objRef() { cid = -1;  idx = -1; }
  objRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void set(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void reset() { cid = -1; idx = -1; }
  void pup(PUP::er &p) { p(cid); p(idx); }
  int isNull() { return ((cid == -1) && (idx == -1)); }
  bool operator==(const objRef& o) const { 
    return((cid == o.cid) && (idx == o.idx)); }
  bool operator>(const objRef& o) const { 
    return((idx > o.idx) || ((idx == o.idx) && (cid > o.cid))); }
  bool operator<(const objRef& o) const { 
    return((idx < o.idx) || ((idx == o.idx) && (cid < o.cid))); }
  objRef& operator=(const objRef& o) { cid=o.cid;  idx=o.idx;  return *this; }
  void sanityCheck() { CkAssert((cid >= -1) && (idx >= -1)); }
};

class node;
class elemRef;
class edgeRef : public objRef {
 public:
  edgeRef() { }
  edgeRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void update(elemRef& oldval, elemRef& newval, int b);
  elemRef getNbr(elemRef m);
  void remove();
  int split(int *m, edgeRef *e_prime, int oIdx, int fIdx,
	    elemRef requester, int *local, int *first, int *nullNbr);
  void collapse(elemRef requester, int kIdx, int dIdx, elemRef kNbr, 
		elemRef dNbr, edgeRef kEdge, edgeRef dEdge, node newN, 
		double frac);
  int flipPrevent(elemRef requester, int kIdx, int dIdx, elemRef kNbr, 
	       elemRef dNbr, edgeRef kEdge, edgeRef dEdge, node newN);
  void resetEdge();
  int isPending(elemRef e);
  int getBoundary();
  void checkPending(elemRef e);
  void checkPending(elemRef e, elemRef ne);
};

class elemRef : public objRef {
 public:
  elemRef() { }
  elemRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void update(edgeRef& oldval, edgeRef& newval);
  double getArea();
  void setTargetArea(double ta);
  void resetTargetArea(double ta);
  void remove();
  //void collapseHelp(edgeRef er, nodeRef nr1, nodeRef nr2);
};

static elemRef nullRef(-1,-1);

#endif
