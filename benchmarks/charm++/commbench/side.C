// The run loop of one end of a case, payload patterns and per-PE buffers.
#include <cstdlib>
#include <cstring>
#include <mutex>

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
    case F_PUT: return dereg ? "put+notify(handle)" : "put+notify";
    case F_SYNC: return "sync-send-reuse";
    case F_NODE: return "node-send-reuse";
    case F_RGET: return "notify+rget";
    case F_RPUT: return "rput+notify";
    case F_PERSIST_SEND: return sameNode ? "persistent-send-local" : "persistent-send";
    case F_PERSIST_RECV: return sameNode ? "persistent-recv-local" : "persistent-recv";
  }
  return "?";
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
}  // namespace

void sideInitNode() { g_bufs = new std::vector<PeBuf>(CkMyNodeSize()); }

void peBuffers(int pe, bool prereg, char **send, char **recv) {
  std::lock_guard<std::mutex> lock(g_bufLock);
  PeBuf &b = (*g_bufs)[CmiRankOf(pe)];
  if (prereg) {
    if (!b.psend) {
      b.psend = (char *)CkRdmaAlloc(g_maxSize);
      b.precv = (char *)CkRdmaAlloc(g_maxSize);
    }
    *send = b.psend;
    *recv = b.precv;
  } else {
    if (!b.send) {
      b.send = allocPlain();
      b.recv = allocPlain();
    }
    *send = b.send;
    *recv = b.recv;
  }
}

std::string rdmaPathNote(int peerPe) {
  const char *p = CmiNodeOf(peerPe) == CmiMyNode() ? "local" : CmiUseCopyBasedRDMA ? "message" : "rma";
  return std::string("\"rdma_path\":\"") + p + "\"";
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
  peBuffers(myPe, isPreregMethod(method), &sendBuf, &recvBuf);
  if (validate) {
    fillPattern(sendBuf, size, role, size);
    fillInverse(recvBuf, size, 1 - role, size);
  }
  c = makeCase(*this);
  c->setup();
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
  c->ping();
}

void Side::halfArrived(const char *d, CmiInt8 n) {
  if (trackPes) recvPes.insert(CkMyPe());
  if (validate) check(d, n);
  if (role == ROLE_B) {
    c->ping();
    return;
  }
  ++done;
  if (done == warmup) t0 = CmiWallTimer();
  if (done == warmup + iters) {
    ResultMsg *r = new ResultMsg;
    r->elapsed = CmiWallTimer() - t0;
    CkSendToFuture(runDone, r);
  } else {
    c->ping();
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
  ResultMsg *r = new ResultMsg;
  r->checked = checked;
  r->mismatches = mismatches;
  r->ok = (mismatches == 0 && error.empty());
  std::string e = error;
  if (firstBad >= 0) {
    char buf[160];
    snprintf(buf, sizeof buf, "%s%lld bad bytes, first at offset %lld: expected 0x%02x got 0x%02x",
             e.empty() ? "" : "; ", (long long)mismatches, (long long)firstBad, badExpected,
             badGot);
    e += buf;
  }
  snprintf(r->error, sizeof r->error, "%s", e.c_str());
  snprintf(r->aux, sizeof r->aux, "%s", auxNote.c_str());
  for (int pe : recvPes) {
    if (r->npes == 32) break;
    r->pes[r->npes++] = pe;
  }
  CkSendToFuture(f, r);
}
