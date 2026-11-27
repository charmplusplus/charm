// Converse-level cases, run by the HandlerEP group. Setup, run and teardown
// arrive as entry methods; the timed traffic uses Converse handlers only.
//   sync:       CmiSyncSendAndFree; the receiver sends the message back.
//   node:       CmiSyncNodeSendAndFree to the peer's node; any PE of that
//               node may run the handler, and the record lists which did.
//   persistent.send, persistent.recv: CmiSyncSendAndFree inside
//               CmiUsePersistentHandle, over a channel set up by the sender
//               (CmiCreatePersistent) or by the receiver
//               (CmiCreateReceiverPersistent; the PersistentReq travels to the
//               sender in a message, which calls CmiRegisterReceivePersistent).
//               A channel within one process forwards to the plain send path.
//               The record's aux.persistent_path says which path carried it:
//               local_forward, rma_put or inline_message.
//   rget, rput: CmiNcpyBuffer rdmaGet / rdmaPut between buffers registered at
//               setup. rget: a notify, then the peer's get. rput: the put,
//               then a notify sent from the initiator's ack handler; Converse
//               has no destination callback.
// Every message carries a 16 B commbench header after the Converse header;
// the size of a case counts only the payload after it.
#include <cstring>
#include <mutex>
#include <sstream>

#include "commbench.h"
#include "persistent.h"

namespace {

struct CvHdr {
  char core[CmiMsgHeaderSizeBytes];
  CmiInt4 sidePe;  // the PE of the Side this message is for
  CmiInt4 pad;
  CmiInt8 n;       // payload bytes
};
const int kHdr = sizeof(CvHdr);

CpvStaticDeclare(int, hData);
CpvStaticDeclare(int, hGetNotify);
CpvStaticDeclare(int, hGetDone);
CpvStaticDeclare(int, hPutNotify);
CpvStaticDeclare(int, hPersistReq);

// HandlerEP sides of this process, by rank. A node-level handler may run on
// another PE of the node than the Side it serves.
Side **g_sides;
// persistent.recv: the request received for each Side, by rank.
PersistentReq *g_reqs;

Side &sideOf(int pe) {
  if (CmiNodeOf(pe) != CmiMyNode())
    CmiAbort("commbench: message for PE %d reached node %d\n", pe, CmiMyNode());
  return *g_sides[CmiRankOf(pe)];
}

// Tags in the ref pointer of the rget/rput handles; the ack handler uses
// them to tell commbench's operations from Charm++'s.
char kGetTag, kPutTag;

char *newMsg(int handler, int destSidePe, CmiInt8 n, CmiInt8 payload) {
  char *m = (char *)CmiAlloc(kHdr + payload);
  CvHdr *h = (CvHdr *)m;
  h->sidePe = destSidePe;
  h->pad = 0;
  h->n = n;
  CmiSetHandler(m, handler);
  return m;
}

class ConvCase : public Case {
 public:
  char *held = nullptr;  // sync/node/persistent: the message, while on this side
  PersistentHandle ph = nullptr;
  CkNcpyBuffer sendH, recvH;

  using Case::Case;
  bool needsHandles() const override {
    int f = familyOf(s.method);
    return f == F_RGET || f == F_RPUT || f == F_PERSIST_RECV;
  }
  // persistent.recv: the sender's end of the channel, from the peer's request.
  void adoptRequest() {
    if (!ph) ph = CmiRegisterReceivePersistent(g_reqs[CmiRankOf(s.myPe)]);
  }

  void setup() override;
  void ping() override;
  void teardown() override;
};

void directAck(void *ack);

void ConvCase::setup() {
  switch (familyOf(s.method)) {
    case F_PERSIST_SEND:
    case F_PERSIST_RECV: {
      // As persist-comm.cpp chooses per send: a peer process reached by the
      // IPC pool, or a backend without RMA, gets the payload inline in a
      // message instead of a put.
      int node = CmiNodeOf(s.peerPe);
      const char *path = node == CmiMyNode() ? "local_forward"
                         : (CmiUseCopyBasedRDMA || CmiIpcReaches(node)) ? "inline_message"
                                                                        : "rma_put";
      std::ostringstream note;
      note << "\"persistent_path\":\"" << path << '"';
      s.auxNote = note.str();
      s.persistentPath = path;
      // inline_message: the notification carries the whole message
      // (PersistentDataMsg in persist-comm.cpp: header, two handles, 4 ints).
      if (!strcmp(path, "inline_message"))
        s.msgBytes = CmiMsgHeaderSizeBytes + 2 * sizeof(void *) + 4 * sizeof(int) + kHdr + s.size;
      if (familyOf(s.method) == F_PERSIST_SEND) {
        ph = CmiCreatePersistent(s.peerPe, kHdr + s.size);
      } else {
        // This side receives on a channel it sets up; the peer sends on it.
        char *m = newMsg(CpvAccess(hPersistReq), s.peerPe, 0, sizeof(PersistentReq));
        *(PersistentReq *)(m + kHdr) = CmiCreateReceiverPersistent(kHdr + s.size);
        CmiSyncSendAndFree(s.peerPe, kHdr + sizeof(PersistentReq), m);
        if (s.haveHandles) adoptRequest();
      }
    }
      // fall through
    case F_SYNC:
    case F_NODE:
      if (s.msgBytes < 0) s.msgBytes = kHdr + s.size;
      if (s.role == ROLE_A) held = newMsg(CpvAccess(hData), s.peerPe, s.size, s.size);
      break;
    case F_RGET:
    case F_RPUT: {
      // The ack handler is process-wide; ours forwards Charm++'s operations.
      static std::once_flag installed;
      std::call_once(installed, [] { CmiSetDirectNcpyAckHandler(directAck); });
      s.auxNote = rdmaPathNote(s.peerPe);
      CkCallback ignore(CkCallback::ignore);
      sendH = CkNcpyBuffer(s.sendBuf, s.size, ignore, CK_BUFFER_REG, CK_BUFFER_NODEREG);
      recvH = CkNcpyBuffer(s.recvBuf, s.size, ignore, CK_BUFFER_REG, CK_BUFFER_NODEREG);
      sendH.setRef(&kPutTag);
      recvH.setRef(&kGetTag);
      s.link->sendHandles(sendH, recvH);
      break;
    }
  }
}

void ConvCase::ping() {
  switch (familyOf(s.method)) {
    case F_SYNC:
    case F_NODE:
    case F_PERSIST_SEND:
    case F_PERSIST_RECV: {
      char *m = held;
      held = nullptr;
      if (!m) CmiAbort("commbench %s: ping without the message\n", kMethods[s.method].name);
      CvHdr *h = (CvHdr *)m;
      h->sidePe = s.peerPe;
      s.fillOutgoing(m + kHdr);
      if (familyOf(s.method) == F_NODE) {
        CmiSyncNodeSendAndFree(CmiNodeOf(s.peerPe), kHdr + s.size, m);
      } else if (familyOf(s.method) != F_SYNC) {
        CmiUsePersistentHandle(&ph, 1);
        CmiSyncSendAndFree(s.peerPe, kHdr + s.size, m);
        CmiUsePersistentHandle(nullptr, 0);
      } else {
        CmiSyncSendAndFree(s.peerPe, kHdr + s.size, m);
      }
      break;
    }
    case F_RGET:
      CmiSyncSendAndFree(s.peerPe, kHdr, newMsg(CpvAccess(hGetNotify), s.peerPe, s.size, 0));
      break;
    case F_RPUT:
      static_cast<CmiNcpyBuffer &>(sendH).rdmaPut(s.peerRecv, 0, nullptr, nullptr);
      break;
  }
}

void ConvCase::teardown() {
  if (held) CmiFree(held);
  held = nullptr;
  if (ph) CmiDestroyPersistent(ph);
  ph = nullptr;
  if (familyOf(s.method) == F_RGET || familyOf(s.method) == F_RPUT) {
    sendH.deregisterMem();
    recvH.deregisterMem();
  }
}

void dataHandler(char *m) {
  CvHdr *h = (CvHdr *)m;
  Side &s = sideOf(h->sidePe);
  ((ConvCase *)s.c)->held = m;
  s.halfArrived(m + kHdr, h->n);
}

// rget: the peer's data is ready; fetch it.
void getNotifyHandler(char *m) {
  CvHdr *h = (CvHdr *)m;
  Side &s = sideOf(h->sidePe);
  CmiInt8 n = h->n;
  CmiFree(m);
  if (n != s.size) CmiAbort("commbench rget: notify for %lld bytes, %d expected\n", (long long)n, s.size);
  static_cast<CmiNcpyBuffer &>(((ConvCase *)s.c)->recvH).rdmaGet(s.peerSend, 0, nullptr, nullptr);
}

// rget completed on a PE other than the Side's; delivered by CmiPushPE.
void getDoneHandler(char *m) {
  Side &s = sideOf(((CvHdr *)m)->sidePe);
  CmiFree(m);
  s.halfArrived(s.recvBuf, s.size);
}

// rput: the peer's put into this side's buffer has completed.
void putNotifyHandler(char *m) {
  CvHdr *h = (CvHdr *)m;
  Side &s = sideOf(h->sidePe);
  CmiInt8 n = h->n;
  CmiFree(m);
  s.halfArrived(s.recvBuf, n);
}

// persistent.recv: the peer's request for the channel this side sends on.
void persistReqHandler(char *m) {
  Side &s = sideOf(((CvHdr *)m)->sidePe);
  g_reqs[CmiRankOf(s.myPe)] = *(PersistentReq *)(m + kHdr);
  CmiFree(m);
  if (s.c) ((ConvCase *)s.c)->adoptRequest();
  s.peerReady();
}

// Runs once per operation on the initiator's process: the destination for a
// get, the source for a put. The put's notify is sent from here, so it takes
// the same channel as the put (see the +ipc ordering in conv-rdma.cpp).
void directAck(void *ack) {
  NcpyOperationInfo *info = (NcpyOperationInfo *)ack;
  if (info->refPtr == &kGetTag) {
    int pe = info->destPe;
    if (CmiMyPe() == pe) {
      Side &s = sideOf(pe);
      s.halfArrived(s.recvBuf, s.size);
    } else {
      CmiPushPE(CmiRankOf(pe), newMsg(CpvAccess(hGetDone), pe, 0, 0));
    }
  } else if (info->refPtr == &kPutTag) {
    CmiInt8 n = (CmiInt8)std::min(info->srcSize, info->destSize);
    CmiSyncSendAndFree(info->destPe, kHdr, newMsg(CpvAccess(hPutNotify), info->destPe, n, 0));
  } else {
    CkRdmaDirectAckHandler(ack);
  }
}

}  // namespace

void converseInitNode() {
  g_sides = new Side *[CkMyNodeSize()]();
  g_reqs = new PersistentReq[CkMyNodeSize()]();
}

void converseInitProc() {
  CpvInitialize(int, hData);
  CpvInitialize(int, hGetNotify);
  CpvInitialize(int, hGetDone);
  CpvInitialize(int, hPutNotify);
  CpvInitialize(int, hPersistReq);
  CpvAccess(hData) = CmiRegisterHandler((CmiHandler)dataHandler);
  CpvAccess(hGetNotify) = CmiRegisterHandler((CmiHandler)getNotifyHandler);
  CpvAccess(hGetDone) = CmiRegisterHandler((CmiHandler)getDoneHandler);
  CpvAccess(hPutNotify) = CmiRegisterHandler((CmiHandler)putNotifyHandler);
  CpvAccess(hPersistReq) = CmiRegisterHandler((CmiHandler)persistReqHandler);
}

void registerHandlerSide(Side *s) { g_sides[CkMyRank()] = s; }

Case *makeConverseCase(Side &s) { return new ConvCase(s); }
