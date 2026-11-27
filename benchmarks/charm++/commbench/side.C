// The run loop of one end of a case, payload patterns and per-PE buffers.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <strings.h>

#include "build_info.h"
#include "commbench.h"

const char *const kEntityName[E_COUNT] = {"handler", "chare", "group", "nodegroup",
                                          "array"};

#define ZC(f, x, r) {x, f, r, CK_BUFFER_DEREG}
#define DIRECT(f, x, r, d) {x, f, r, d}
const MethodInfo kMethods[] = {
    {"marshal", F_MARSHAL, CK_BUFFER_REG, CK_BUFFER_DEREG},
    {"msg", F_MSG, CK_BUFFER_REG, CK_BUFFER_DEREG},
    ZC(F_NOCOPY, "nocopy.reg", CK_BUFFER_REG),
    ZC(F_NOCOPY, "nocopy.unreg", CK_BUFFER_UNREG),
    ZC(F_NOCOPY, "nocopy.prereg", CK_BUFFER_PREREG),
    ZC(F_POST, "post.reg", CK_BUFFER_REG),
    ZC(F_POST, "post.unreg", CK_BUFFER_UNREG),
    ZC(F_POST, "post.prereg", CK_BUFFER_PREREG),
    DIRECT(F_GET, "get.reg.nodereg", CK_BUFFER_REG, CK_BUFFER_NODEREG),
    DIRECT(F_GET, "get.reg.dereg", CK_BUFFER_REG, CK_BUFFER_DEREG),
    DIRECT(F_GET, "get.unreg.nodereg", CK_BUFFER_UNREG, CK_BUFFER_NODEREG),
    DIRECT(F_GET, "get.unreg.dereg", CK_BUFFER_UNREG, CK_BUFFER_DEREG),
    DIRECT(F_GET, "get.prereg.nodereg", CK_BUFFER_PREREG, CK_BUFFER_NODEREG),
    DIRECT(F_GET, "get.prereg.dereg", CK_BUFFER_PREREG, CK_BUFFER_DEREG),
    DIRECT(F_PUT, "put.reg.nodereg", CK_BUFFER_REG, CK_BUFFER_NODEREG),
    DIRECT(F_PUT, "put.reg.dereg", CK_BUFFER_REG, CK_BUFFER_DEREG),
    DIRECT(F_PUT, "put.unreg.nodereg", CK_BUFFER_UNREG, CK_BUFFER_NODEREG),
    DIRECT(F_PUT, "put.unreg.dereg", CK_BUFFER_UNREG, CK_BUFFER_DEREG),
    DIRECT(F_PUT, "put.prereg.nodereg", CK_BUFFER_PREREG, CK_BUFFER_NODEREG),
    DIRECT(F_PUT, "put.prereg.dereg", CK_BUFFER_PREREG, CK_BUFFER_DEREG),
    {"sync", F_SYNC, CK_BUFFER_REG, CK_BUFFER_DEREG},
    {"node", F_NODE, CK_BUFFER_REG, CK_BUFFER_DEREG},
    {"rget", F_RGET, CK_BUFFER_REG, CK_BUFFER_NODEREG},
    {"rput", F_RPUT, CK_BUFFER_REG, CK_BUFFER_NODEREG},
    {"persistent.send", F_PERSIST_SEND, CK_BUFFER_REG, CK_BUFFER_DEREG},
    {"persistent.recv", F_PERSIST_RECV, CK_BUFFER_REG, CK_BUFFER_DEREG},
};
#undef ZC
#undef DIRECT
const int kNumMethods = sizeof(kMethods) / sizeof(kMethods[0]);

const char *protocolOf(int m, bool sameNode) {
  bool dereg = kMethods[m].deregMode == CK_BUFFER_DEREG;
  switch (familyOf(m)) {
    case F_MARSHAL: return "marshal";
    case F_MSG: return "msg-reuse";
    case F_NOCOPY: return "rendezvous-api";
    case F_POST: return "rendezvous-post-match";
    case F_GET: return dereg ? "notify(handle)+get" : "notify+get";
    case F_PUT: return dereg ? "put+destcb+handle" : "put+destcb";
    case F_SYNC: return "sync-send-reuse";
    case F_NODE: return "node-send-reuse";
    case F_RGET: return "notify+rget";
    case F_RPUT: return "rput+notify";
    case F_PERSIST_SEND: return sameNode ? "persistent-send-local" : "persistent-send";
    case F_PERSIST_RECV: return sameNode ? "persistent-recv-local" : "persistent-recv";
  }
  return "?";
}

void programCopies(int m, int *sender, int *receiver) {
  int f = familyOf(m);
  // Methods whose payload travels in a message the runtime allocates: the
  // sender fills it, the receiver finds the payload inside it.
  bool inMsg = f == F_MSG || f == F_SYNC || f == F_NODE || f == F_PERSIST_SEND || f == F_PERSIST_RECV;
  // marshal packs from sendBuf itself; nocopy lands in a fresh message.
  bool landsInMsg = inMsg || f == F_MARSHAL || f == F_NOCOPY;
  *sender = g_sourceUser && inMsg ? 1 : 0;
  *receiver = g_deliverUser && landsInMsg ? 1 : 0;
}

void fillPattern(char *buf, size_t n, int role, size_t size) {
  for (size_t i = 0; i < n; i++) buf[i] = (char)patternByte(i, role, size);
}

void fillInverse(char *buf, size_t n, int role, size_t size) {
  for (size_t i = 0; i < n; i++) buf[i] = (char)~patternByte(i, role, size);
}

namespace {
struct PeBuf {
  char *send = nullptr, *recv = nullptr, *psend = nullptr, *precv = nullptr;
};
std::vector<PeBuf> *g_bufs;  // one per rank of this process
std::mutex g_bufLock;

char *allocPlain() {
  void *p = nullptr;
  if (posix_memalign(&p, 4096, (size_t)g_maxSize) != 0)
    CkAbort("commbench: cannot allocate %d bytes\n", g_maxSize);
  return (char *)p;
}

// Writes every page, so that no timed copy into the buffer takes page faults.
char *touched(char *p) {
  memset(p, 0, (size_t)g_maxSize);
  return p;
}
}  // namespace

void sideInitNode() { g_bufs = new std::vector<PeBuf>(CkMyNodeSize()); }

void peBuffers(int pe, bool prereg, char **send, char **recv) {
  std::lock_guard<std::mutex> lock(g_bufLock);
  PeBuf &b = (*g_bufs)[CmiRankOf(pe)];
  if (prereg) {
    if (!b.psend) {
      b.psend = touched((char *)CkRdmaAlloc(g_maxSize));
      b.precv = touched((char *)CkRdmaAlloc(g_maxSize));
    }
    *send = b.psend;
    *recv = b.precv;
  } else {
    if (!b.send) {
      b.send = touched(allocPlain());
      b.recv = touched(allocPlain());
    }
    *send = b.send;
    *recv = b.recv;
  }
}

std::string rdmaPathNote(int peerPe) {
  const char *p = CmiNodeOf(peerPe) == CmiMyNode() ? "local" : CmiUseCopyBasedRDMA ? "message" : "rma";
  std::ostringstream o;
  o << "\"rdma_path\":\"" << p << '"';
  return o.str();
}

// A string into a fixed-size field of a message, cut to fit.
template <size_t N>
static void copyField(char (&dst)[N], const std::string &s) {
  size_t n = std::min(s.size(), N - 1);
  memcpy(dst, s.data(), n);
  dst[n] = '\0';
}

void Case::unexpected(const char *what) {
  CkAbort("commbench: %s arrived for method %s on PE %d\n", what, kMethods[s.method].name,
          CkMyPe());
}

static Case *makeCase(Side &s) {
  switch (familyOf(s.method)) {
    case F_MARSHAL:
    case F_MSG: return makePlainCase(s);
    case F_NOCOPY:
    case F_POST: return makeZerocopyCase(s);
    case F_GET:
    case F_PUT: return makeDirectCase(s);
    default: return makeConverseCase(s);
  }
}

void Side::begin(int method_, int size_, int role_, int myPe_, int peerPe_, bool validate_,
                 Link *link_, CkFuture ready_) {
  method = method_;
  size = size_;
  role = role_;
  myPe = myPe_;
  peerPe = peerPe_;
  validate = validate_;
  link = link_;
  ready = ready_;
  readySent = false;
  checked = mismatches = 0;
  firstBad = -1;
  error.clear();
  recvPes.clear();
  auxNote.clear();
  pings = arrivals = sendCopies = recvCopies = 0;
  msgBytes = -1;
  persistentPath = nullptr;
  peBuffers(myPe, isPreregMethod(method), &sendBuf, &recvBuf);
  if (validate) {
    fillPattern(sendBuf, size, role, size);
    fillInverse(recvBuf, size, 1 - role, size);
  }
  c = makeCase(*this);
  c->setup();
  std::ostringstream aux;
  aux << auxNote;
  if (CmiIpcEnabled())
    aux << (auxNote.empty() ? "" : ",") << "\"ipc_reaches\":"
        << (CmiIpcReaches(CmiNodeOf(peerPe)) ? "true" : "false");
  aux << (aux.tellp() > 0 ? "," : "") << copiesNote();
  auxNote = aux.str();
  setupDone = true;
  maybeReady();
}

void Side::gotHandles(const CkNcpyBuffer &s, const CkNcpyBuffer &r) {
  peerSend = s;
  peerRecv = r;
  haveHandles = true;
  maybeReady();
}

void Side::peerReady() {
  haveHandles = true;
  maybeReady();
}

void Side::maybeReady() {
  if (!setupDone || readySent || (c->needsHandles() && !haveHandles)) return;
  readySent = true;
  CkSendToFuture(ready, new ResultMsg);
}

void Side::startRun(int iters_, int warmup_, CkFuture f) {
  iters = iters_;
  warmup = warmup_;
  done = 0;
  runDone = f;
  t0 = CmiWallTimer();
  ping();
}

void Side::ping() {
  pings++;
  c->ping();
}

void Side::fillOutgoing(char *dst) {
  if (g_sourceUser) {
    memcpy(dst, sendBuf, (size_t)size);
    sendCopies++;
  } else if (validate) {
    fillPattern(dst, size, role, size);
  }
}

void Side::halfArrived(const char *d, CmiInt8 n) {
  if (trackPes) recvPes.insert(CkMyPe());
  arrivals++;
  if (g_deliverUser && d != recvBuf) {
    memcpy(recvBuf, d, (size_t)std::min<CmiInt8>(n, g_maxSize));
    recvCopies++;
    d = recvBuf;
  }
  if (validate) check(d, n);
  if (role == ROLE_B) {
    ping();
    return;
  }
  ++done;
  if (done == warmup) t0 = CmiWallTimer();
  if (done == warmup + iters) {
    ResultMsg *r = new ResultMsg;
    r->elapsed = CmiWallTimer() - t0;
    CkSendToFuture(runDone, r);
  } else {
    ping();
  }
}

void Side::check(const char *d, CmiInt8 n) {
  if (n != size) {
    if (error.empty())
      error = "received " + std::to_string(n) + " bytes, expected " + std::to_string(size);
    mismatches++;
    return;
  }
  const int from = 1 - role;
  for (CmiInt8 i = 0; i < n; i++) {
    unsigned char want = patternByte(i, from, size);
    if ((unsigned char)d[i] != want) {
      if (firstBad < 0) {
        firstBad = i;
        badExpected = want;
        badGot = (unsigned char)d[i];
      }
      mismatches++;
    }
  }
  checked += n;
}

void Side::end(CkFuture f) {
  c->teardown();
  delete c;
  c = nullptr;
  setupDone = false;
  haveHandles = false;
  int wantSend, wantRecv;
  programCopies(method, &wantSend, &wantRecv);
  if (sendCopies != wantSend * pings || recvCopies != wantRecv * arrivals) {
    std::ostringstream o;
    o << (error.empty() ? "" : "; ") << "copy count: " << pings << " sends with " << sendCopies
      << " copies, " << arrivals << " arrivals with " << recvCopies << " copies";
    error += o.str();
  }
  ResultMsg *r = new ResultMsg;
  r->checked = checked;
  r->mismatches = mismatches;
  r->ok = (mismatches == 0 && error.empty());
  std::ostringstream e;
  e << error;
  if (firstBad >= 0)
    e << (error.empty() ? "" : "; ") << mismatches << " bad bytes, first at offset " << firstBad
      << ": expected 0x" << std::hex << std::setfill('0') << std::setw(2) << badExpected
      << " got 0x" << std::setw(2) << badGot;
  copyField(r->error, e.str());
  copyField(r->aux, auxNote);
  for (int pe : recvPes) {
    if (r->npes == 32) break;
    r->pes[r->npes++] = pe;
  }
  CkSendToFuture(f, r);
}

// The runtime's own payload copies per one-way message, estimated from
// reading the code; README.md, "Copies", gives the reasoning and citations.
namespace {
struct RtCopies {
  int n;             // -1: unknown
  std::string path;  // which path the estimate assumes
};

// Largest Converse message LCI sends eagerly: the packet payload (packet size
// less a 64 B cache line) less the 12 B of tag and remote completion.
long lciEagerMax() {
  long pkt = COMMBENCH_LCI_PACKET_SIZE;
  if (const char *e = getenv("LCI_ATTR_PACKET_SIZE")) pkt = atol(e);
  return pkt > 0 ? pkt - 64 - 12 : -1;
}

bool envTrue(const char *name) {
  const char *e = getenv(name);
  return e && (!strcmp(e, "1") || !strcasecmp(e, "true") || !strcasecmp(e, "on"));
}

// A message of `bytes` (headers included) from this PE to peerPe.
RtCopies messageCopies(int peerPe, CmiInt8 bytes) {
  int node = CmiNodeOf(peerPe);
  if (node == CmiMyNode()) return {0, "local_pointer"};
  if (CmiIpcReaches(node) && bytes <= (CmiInt8)CmiRecommendedIpcBlockCutoff()) return {1, "ipc_pool"};
  if (COMMBENCH_LCI_WITH_SHM && CmiPhysicalNodeID(peerPe) == CmiPhysicalNodeID(CmiMyPe()))
    return {-1, "lci_shm"};
  long eager = lciEagerMax();
  if (eager < 0) return {-1, "unknown_backend"};
  if (bytes <= eager) return {envTrue("LCI_ATTR_ZERO_COPY_AM") ? 0 : 1, "lci_eager"};
  return {0, "lci_rendezvous"};
}

// A zerocopy transfer (nocopy, post, get, put, rget, rput) to or from peerPe.
RtCopies zerocopyCopies(int peerPe) {
  if (CmiNodeOf(peerPe) == CmiMyNode()) return {1, "memcpy"};
  if (CmiUseCopyBasedRDMA) return {-1, "copy_based_rdma"};
  return {0, "rdma"};
}
}  // namespace

std::string Side::copiesNote() const {
  int snd, rcv;
  programCopies(method, &snd, &rcv);
  RtCopies rt{-1, "?"};
  switch (familyOf(method)) {
    case F_MARSHAL:
      rt = messageCopies(peerPe, msgBytes);
      rt.path = "pack+" + rt.path;
      if (rt.n >= 0) rt.n += 1;
      break;
    case F_MSG:
    case F_SYNC:
    case F_NODE: rt = messageCopies(peerPe, msgBytes); break;
    case F_PERSIST_SEND:
    case F_PERSIST_RECV:
      if (persistentPath && !strcmp(persistentPath, "rma_put")) {
        rt = {0, "rma_put"};
      } else if (persistentPath && !strcmp(persistentPath, "inline_message")) {
        rt = messageCopies(peerPe, msgBytes);  // the notification message
        rt.path = "inline_message+" + rt.path;
        if (rt.n >= 0) rt.n += 2;
      } else {
        rt = messageCopies(peerPe, msgBytes);
      }
      break;
    default: rt = zerocopyCopies(peerPe); break;
  }
  std::ostringstream o;
  o << "\"copies_sender\":" << snd << ",\"copies_receiver\":" << rcv << ",\"copies_runtime_est\":";
  if (rt.n < 0)
    o << "null";
  else
    o << rt.n;
  o << ",\"copies_runtime_path\":\"" << rt.path << '"';
  if (msgBytes >= 0) o << ",\"msg_bytes\":" << msgBytes;
  return o.str();
}
