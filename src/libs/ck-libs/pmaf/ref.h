// Reference class for PMAF3D Framework
// Created by: Terry L. Wilmarth
#ifndef REF_H
#define REF_H
#include "charm.h"

class node;

class objRef { // a reference to a piece of data that may be remotely located
 public:
  int cid, idx;
  objRef() { cid = -1;  idx = -1; }
  objRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void set(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void reset() { cid = -1; idx = -1; }
  bool operator==(const objRef& o) const { return((cid == o.cid) && (idx == o.idx)); }
  bool operator>(const objRef& o) const { return((idx > o.idx) || 
                 ((idx == o.idx) && (cid > o.cid))); }
  bool operator<(const objRef& o) const { return((idx < o.idx) || 
                 ((idx == o.idx) && (cid < o.cid))); }
  objRef& operator=(const objRef& o) { cid=o.cid; idx=o.idx; return *this; }
  void pup(PUP::er &p) { p|cid;  p|idx; }
};

class node;

class nodeRef : public objRef {
 public:
  nodeRef() { cid = -1;  idx = -1; }
  nodeRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  node get();
  void update(node& m);
  void pup(PUP::er &p) { p|cid;  p|idx; }
};

class elemRef : public objRef {
 public:
  elemRef() { cid = -1;  idx = -1; }
  elemRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  double getVolume();
  void setTargetVolume(double ta);
  void resetTargetVolume(double ta);
  void pup(PUP::er &p) { p|cid;  p|idx; }
};

#endif
