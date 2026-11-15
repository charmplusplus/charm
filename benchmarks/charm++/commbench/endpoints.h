// The five endpoint kinds share one implementation: entry methods hand the
// event to the Side and its Case, and the Link methods send to the peer
// through the kind's own proxy. Included once, by driver.C, before
// commbench.def.h.
#ifndef COMMBENCH_ENDPOINTS_H
#define COMMBENCH_ENDPOINTS_H

#include "commbench.h"

template <class CBase, class Proxy, class Index, bool kNodeLevel>
class EP : public CBase, public Link {
 public:
  Side side;
  Proxy peer;

  EP() {}
  EP(CkMigrateMessage *m) : CBase(m) {}

  void setup(int method, int size, int role, int myPe, int peerPe, bool validate, Proxy p,
             CkFuture ready) {
    peer = p;
    side.trackPes = kNodeLevel || familyOf(method) == F_NODE;
    side.begin(method, size, role, myPe, peerPe, validate, this, ready);
  }
  void run(int iters, int warmup, CkFuture done) { side.startRun(iters, warmup, done); }
  void teardown(CkFuture done) { side.end(done); }
  void recvHandles(CkNcpyBuffer s, CkNcpyBuffer r) { side.gotHandles(s, r); }

  void recvMarshal(int n, char *d) { side.c->onMarshal(n, d); }
  void recvMsg(CommMsg *m) { side.c->onMsg(m); }
  void recvNocopy(int n, char *d) { side.c->onNocopy(n, d); }
  void recvPost(int n, char *d, CkNcpyBufferPost *post) { side.c->onPostMatch(n, post); }
  void recvPost(int n, char *d) { side.c->onPost(n, d); }
  void notify(CmiInt8 seq, CmiInt8 n) { side.c->onNotify(seq, n); }
  void notifyHandle(CmiInt8 seq, CmiInt8 n, CkNcpyBuffer h) { side.c->onNotifyHandle(seq, n, h); }
  void getDone(CkDataMsg *m) {
    delete m;
    side.c->onGetDone();
  }
  void putDone(CkDataMsg *m) {
    delete m;
    side.c->onPutDone();
  }

  void sendMarshal(int n, const char *d) override { peer.recvMarshal(n, d); }
  void sendMsg(CommMsg *m) override { peer.recvMsg(m); }
  void sendNocopy(int n, const CkNcpyBuffer &b) override { peer.recvNocopy(n, b); }
  void sendPost(int n, const CkNcpyBuffer &b) override { peer.recvPost(n, b); }
  void sendNotify(CmiInt8 seq, CmiInt8 n) override { peer.notify(seq, n); }
  void sendNotifyHandle(CmiInt8 seq, CmiInt8 n, const CkNcpyBuffer &h) override {
    peer.notifyHandle(seq, n, h);
  }
  void sendHandles(const CkNcpyBuffer &s, const CkNcpyBuffer &r) override {
    peer.recvHandles(s, r);
  }
  CkCallback getDoneCallback() override {
    return CkCallback(this, Index::getDone((CkDataMsg *)nullptr));
  }
  CkCallback putDoneCallback() override {
    return CkCallback(this, Index::putDone((CkDataMsg *)nullptr));
  }
};

class ChareEP : public EP<CBase_ChareEP, CProxy_ChareEP, CkIndex_ChareEP, false> {
 public:
  // charmxi's code for a nocopypost entry reads thisIndex, which a singleton
  // chare lacks; the index only matters for broadcasts, so 0 serves.
  int thisIndex = 0;

  explicit ChareEP(CkFuture created) {
    ResultMsg *r = new ResultMsg;
    r->cid = thishandle;
    CkSendToFuture(created, r);
  }
};

class GroupEP : public EP<CBase_GroupEP, CProxyElement_GroupEP, CkIndex_GroupEP, false> {};

class NodeGroupEP
    : public EP<CBase_NodeGroupEP, CProxyElement_NodeGroupEP, CkIndex_NodeGroupEP, true> {};

class ArrayEP : public EP<CBase_ArrayEP, CProxyElement_ArrayEP, CkIndex_ArrayEP, false> {
 public:
  ArrayEP() {}
  ArrayEP(CkMigrateMessage *m) : EP(m) {}
};

class HandlerEP
    : public EP<CBase_HandlerEP, CProxyElement_HandlerEP, CkIndex_HandlerEP, false> {
 public:
  HandlerEP() { registerHandlerSide(&side); }
};

// Element i of the ArrayEP lives on PE i.
class PeMap : public CkArrayMap {
 public:
  PeMap() {}
  int procNum(int, const CkArrayIndex &idx) override {
    return ((const int *)idx.data())[0] % CkNumPes();
  }
};

#endif
