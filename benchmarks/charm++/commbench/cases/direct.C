// get.<reg>.<dereg> and put.<reg>.<dereg> through the CkNcpyBuffer direct
// API, <reg> in {reg, unreg, prereg}, <dereg> in {dereg, nodereg}. Each half
// round trip has two steps:
//   get: a notify to the peer, then the peer's get; the half ends when the get
//        completes on the peer (its destination callback).
//   put: a put to the peer; its local completion (source callback) sends a
//        notify, and the half ends when the notify arrives.
// nodereg: both sides create their send and receive handles at setup and
// exchange them once; the notify is 16 B.
// dereg: a handle is good for one transfer, since the runtime deregisters
// both buffers when it completes. Every notify therefore carries a handle
// created for the next transfer: the sender's send buffer (get), or its
// receive buffer for the peer's next put (put). A's first put uses B's
// receive handle, sent at setup.
#include "../commbench.h"

namespace {

class DirectCase : public Case {
  CkNcpyBuffer sendH, recvH;
  CmiInt8 seq = 0;
  bool recvOutstanding = false;  // put.*.dereg: recvH was sent and not yet used

  bool isGet() const { return familyOf(s.method) == F_GET; }
  bool dereg() const { return kMethods[s.method].deregMode == CK_BUFFER_DEREG; }
  CkNcpyBuffer handle(char *buf, CkCallback cb) {
    return CkNcpyBuffer(buf, s.size, cb, kMethods[s.method].regMode, kMethods[s.method].deregMode);
  }
  static CkCallback ignore() { return CkCallback(CkCallback::ignore); }

 public:
  using Case::Case;
  bool needsHandles() const override { return !(isGet() && dereg()); }

  void setup() override {
    s.auxNote = rdmaPathNote(s.peerPe);
    if (!dereg()) {
      sendH = handle(s.sendBuf, isGet() ? ignore() : s.link->putDoneCallback());
      recvH = handle(s.recvBuf, isGet() ? s.link->getDoneCallback() : ignore());
      s.link->sendHandles(sendH, recvH);
    } else if (!isGet()) {
      // Only B's receive handle is needed: A puts first, and its notify
      // carries the handle for B's put. A sends empty handles so that both
      // sides wait for each other's setup.
      if (s.role == ROLE_B) {
        recvH = handle(s.recvBuf, ignore());
        recvOutstanding = true;
      }
      s.link->sendHandles(CkNcpyBuffer(), s.role == ROLE_B ? recvH : CkNcpyBuffer());
    }
  }

  void ping() override {
    if (isGet()) {
      if (dereg()) {
        sendH = handle(s.sendBuf, ignore());
        s.link->sendNotifyHandle(++seq, s.size, sendH);
      } else {
        s.link->sendNotify(++seq, s.size);
      }
    } else {
      if (dereg()) sendH = handle(s.sendBuf, s.link->putDoneCallback());
      sendH.put(s.peerRecv);
    }
  }

  void onNotify(CmiInt8, CmiInt8 n) override {
    checkSize(n);
    if (isGet())
      recvH.get(s.peerSend);
    else
      s.halfArrived(s.recvBuf, n);
  }

  void onNotifyHandle(CmiInt8, CmiInt8 n, CkNcpyBuffer &h) override {
    checkSize(n);
    if (isGet()) {
      s.peerSend = h;
      recvH = handle(s.recvBuf, s.link->getDoneCallback());
      recvH.get(s.peerSend);
    } else {
      s.peerRecv = h;
      recvOutstanding = false;  // the peer's put into our recvH has completed
      s.halfArrived(s.recvBuf, n);
    }
  }

  void onGetDone() override { s.halfArrived(s.recvBuf, s.size); }

  // Local completion of the put: the source buffer is free, but the data may
  // not have landed. The notify must travel behind the put on the network;
  // with +ipc a peer process on this host would otherwise get it through the
  // shared-memory pool, ahead of the data (conv-rdma.cpp, CommRputLocalHandler).
  void onPutDone() override {
    CmiIpcBeginNetworkOnly();
    if (dereg()) {
      recvH = handle(s.recvBuf, ignore());
      recvOutstanding = true;
      s.link->sendNotifyHandle(++seq, s.size, recvH);
    } else {
      s.link->sendNotify(++seq, s.size);
    }
    CmiIpcEndNetworkOnly();
  }

  void teardown() override {
    if (!dereg()) {
      sendH.deregisterMem();
      recvH.deregisterMem();
    } else if (recvOutstanding) {
      recvH.deregisterMem();
    }
  }

 private:
  void checkSize(CmiInt8 n) {
    if (n != s.size)
      CkAbort("commbench %s: notify for %lld bytes, %d expected\n", kMethods[s.method].name,
              (long long)n, s.size);
  }
};

}  // namespace

Case *makeDirectCase(Side &s) { return new DirectCase(s); }
