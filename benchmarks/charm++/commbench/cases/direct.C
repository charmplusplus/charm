// get.<reg>.<dereg> and put.<reg>.<dereg> through the CkNcpyBuffer direct
// API, <reg> in {reg, unreg, prereg}, <dereg> in {dereg, nodereg}. Each half
// round trip has two steps:
//   get: a notify to the peer, then the peer's get; the half ends when the get
//        completes on the peer (its receive handle's callback).
//   put: a put to the peer; the half ends when the runtime's destination
//        acknowledgement runs the peer's receive handle callback.
// nodereg: both sides create their send and receive handles at setup and
// exchange them once; the get notify is 16 B.
// dereg: a handle is good for one transfer, since the runtime deregisters
// both buffers when it completes.
//   get: each notify carries a handle created for that get.
//   put: the receive handles are exchanged at setup; when a put has landed,
//        the receiver sends the sender a new receive handle in a message of
//        its own (the put's data path cannot carry it), and the half ends
//        when both the data and the peer's new handle are in.
#include "../commbench.h"

namespace {

class DirectCase : public Case {
  CkNcpyBuffer sendH, recvH;
  CmiInt8 seq = 0;
  // put.*.dereg
  bool recvOutstanding = false;  // recvH was sent to the peer and not yet used
  bool dataIn = false;           // a put into recvH has landed
  bool peerHandleFresh = false;  // s.peerRecv has not been used yet

  bool isGet() const { return familyOf(s.method) == F_GET; }
  bool dereg() const { return kMethods[s.method].deregMode == CK_BUFFER_DEREG; }
  CkNcpyBuffer handle(char *buf, CkCallback cb) {
    return CkNcpyBuffer(buf, s.size, cb, kMethods[s.method].regMode, kMethods[s.method].deregMode);
  }
  static CkCallback ignore() { return CkCallback(CkCallback::ignore); }

  void newRecvHandle() {
    recvH = handle(s.recvBuf, s.link->destDoneCallback());
    recvOutstanding = true;
  }
  // put.*.dereg: the half is complete once the data and the peer's next
  // receive handle have both arrived.
  void putHalfIfReady() {
    if (!dataIn || !peerHandleFresh) return;
    dataIn = false;
    s.halfArrived(s.recvBuf, s.size);
  }

 public:
  using Case::Case;
  bool needsHandles() const override { return !(isGet() && dereg()); }

  void setup() override {
    s.auxNote = rdmaPathNote(s.peerPe);
    if (isGet() && dereg()) return;
    if (isGet()) {
      sendH = handle(s.sendBuf, ignore());
      recvH = handle(s.recvBuf, s.link->destDoneCallback());
    } else {
      if (!dereg()) sendH = handle(s.sendBuf, ignore());
      newRecvHandle();
      peerHandleFresh = true;  // the peer's receive handle, exchanged below
    }
    s.link->sendHandles(sendH, recvH);
  }

  void ping() override {
    if (isGet()) {
      if (dereg()) {
        sendH = handle(s.sendBuf, ignore());
        s.link->sendNotifyHandle(++seq, s.size, sendH);
      } else {
        s.link->sendNotify(++seq, s.size);
      }
      return;
    }
    if (dereg()) {
      if (!peerHandleFresh) CkAbort("commbench put.dereg: no fresh handle from the peer\n");
      peerHandleFresh = false;
      sendH = handle(s.sendBuf, ignore());
    }
    sendH.put(s.peerRecv);
  }

  void onNotify(CmiInt8, CmiInt8 n) override {
    checkSize(n);
    if (!isGet()) unexpected("notify");
    recvH.get(s.peerSend);
  }

  void onNotifyHandle(CmiInt8, CmiInt8 n, CkNcpyBuffer &h) override {
    checkSize(n);
    if (isGet()) {
      s.peerSend = h;
      recvH = handle(s.recvBuf, s.link->destDoneCallback());
      recvH.get(s.peerSend);
    } else {
      s.peerRecv = h;
      peerHandleFresh = true;
      putHalfIfReady();
    }
  }

  void onDestDone() override {
    if (isGet() || !dereg()) {
      s.halfArrived(s.recvBuf, s.size);
      return;
    }
    // The peer's put used recvH; hand it the next one before looking at the
    // data, so that its following put is not held up.
    recvOutstanding = false;
    newRecvHandle();
    s.link->sendNotifyHandle(++seq, s.size, recvH);
    dataIn = true;
    putHalfIfReady();
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
