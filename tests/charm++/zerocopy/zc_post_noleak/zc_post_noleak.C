// Sends Entry Method Post API messages, point to point and as broadcasts, and
// checks that no memory registration is left behind for the posted buffers,
// whatever registration and de-registration modes the receiver gives them.
// The runtime registers a posted buffer for each message, in a handle the
// receiver never sees, so the registration must be released after the message
// even with CK_BUFFER_NODEREG.
//
// A phase ends when every message has run its regular entry method and every
// sender has had its source callback: the runtime releases the posted buffers
// before the former and the sent buffers before the latter, so the counts are
// final then. Completion is counted rather than detected with CkStartQD, which
// does not complete after a Post API broadcast on processes of several PEs.
//
// The count of registrations each process holds comes from reconverse's
// CmiRegisteredBuffers (reconverse#332). It is declared weak, so the test also
// builds against a reconverse without it and then only checks the transfers;
// only an ELF platform resolves a weak reference to nothing at link time. The
// count is meaningful with two or more processes, where transfers are RDMA;
// within one process they are copies and nothing is registered.
#include "zc_post_noleak.decl.h"
#include <cstring>

#if defined(__ELF__)
long CmiRegisteredBuffers(void) __attribute__((weak));
static long registeredBuffers() { return CmiRegisteredBuffers ? CmiRegisteredBuffers() : -1; }
#else
static long registeredBuffers() { return -1; }
#endif

CProxy_Main mainProxy;
CProxy_Peer peers;
int numMsgs;

static const int bufSize = 64 * 1024;

// The posted buffer's modes in each phase. CK_BUFFER_PREREG posts a buffer
// from CkRdmaAlloc, as the manual's example does.
struct Variant {
  const char *name;
  unsigned short regMode, deregMode;
};
static const Variant variants[] = {
    {"REG/DEREG", CK_BUFFER_REG, CK_BUFFER_DEREG},
    {"REG/NODEREG", CK_BUFFER_REG, CK_BUFFER_NODEREG},
    {"UNREG/DEREG", CK_BUFFER_UNREG, CK_BUFFER_DEREG},
    {"UNREG/NODEREG", CK_BUFFER_UNREG, CK_BUFFER_NODEREG},
    {"PREREG/NODEREG", CK_BUFFER_PREREG, CK_BUFFER_NODEREG},
};
static const int numVariants = sizeof(variants) / sizeof(variants[0]);

static char pattern(int pe, int i) { return (char)(pe * 31 + i); }

class Main : public CBase_Main {
  int phase = 0;  // p2p phases 0..numVariants-1, then broadcast phases
  int round = 0, reports = 0, roundParts = 0;
  bool counted = false;
  char *sendBuf;

  int variant() const { return phase % numVariants; }
  bool isBcast() const { return phase >= numVariants; }

  void startPhase() {
    round = 0;
    if (!isBcast()) {
      peers.startP2p(variant());
    } else {
      sendBcast();
    }
  }

  void sendBcast() {
    roundParts = 0;
    CkCallback cb(CkIndex_Main::bcastSourceDone(NULL), thisProxy);
    peers.bcast(CkSendBuffer(sendBuf, cb), bufSize, variant(), round);
  }

  // A broadcast round is over when every PE has run the regular entry method
  // and the source callback has come back.
  void bcastPartDone() {
    if (++roundParts < 2) return;
    if (++round < numMsgs) {
      sendBcast();
      return;
    }
    report();
  }

  void report() {
    reports = 0;
    peers.report();
  }

public:
  Main(CkArgMsg *m) {
    numMsgs = (m->argc > 1) ? atoi(m->argv[1]) : 100;
    delete m;
    mainProxy = thisProxy;
    sendBuf = new char[bufSize];
    for (int i = 0; i < bufSize; i++) sendBuf[i] = pattern(CkMyPe(), i);
    CkPrintf("zc_post_noleak: %d messages per phase on %d PEs in %d processes\n",
             numMsgs, CkNumPes(), CkNumNodes());
    peers = CProxy_Peer::ckNew();
    startPhase();
  }

  void p2pDone() { report(); }

  void bcastRoundDone() { bcastPartDone(); }

  void bcastSourceDone(CkDataMsg *m) {
    delete m;
    bcastPartDone();
  }

  void count(int node, long registered) {
    const char *what = isBcast() ? "broadcast" : "p2p";
    if (registered > 0)
      CkAbort("zc_post_noleak: %s with posted %s: process %d still holds %ld "
              "memory registrations\n", what, variants[variant()].name, node, registered);
    if (registered == 0) counted = true;
    if (++reports < CkNumNodes()) return;
    CkPrintf("zc_post_noleak: %s with posted %s passed\n", what, variants[variant()].name);
    if (++phase < 2 * numVariants) {
      startPhase();
      return;
    }
    CkPrintf("zc_post_noleak: %s\n", counted ? "no registration left"
             : "registrations not counted (no CmiRegisteredBuffers); transfers checked");
    CkExit();
  }
};

class Peer : public CBase_Peer {
  char *sendBuf, *postBuf, *preregBuf;
  int variant = 0, sent = 0, acked = 0, received = 0, sourcesDone = 0, posts = 0;

  int partner() const { return (CkMyPe() + CkNumPes() / 2) % CkNumPes(); }

  // Posts the buffer for one message with this variant's modes. One message is
  // in flight to each PE at a time, so the buffer can be reused.
  void post(int n, CkNcpyBufferPost *ncpyPost) {
    const Variant &v = variants[variant];
    char *buf = v.regMode == CK_BUFFER_PREREG ? preregBuf : postBuf;
    memset(buf, 0, n);
    ncpyPost[0].regMode = v.regMode;
    ncpyPost[0].deregMode = v.deregMode;
    const int tag = posts++;
    CkMatchBuffer(ncpyPost, 0, tag);
    CkPostBuffer(buf, n, tag);
  }

  void check(const char *what, const char *data, int n, int from) {
    for (int i = 0; i < n; i++)
      if (data[i] != pattern(from, i))
        CkAbort("zc_post_noleak: %s with posted %s: PE %d got byte %d wrong from PE %d\n",
                what, variants[variant].name, CkMyPe(), i, from);
  }

  void sendNext() {
    if (sent == numMsgs) return;
    CkCallback cb(CkIndex_Peer::sourceDone(NULL), thisProxy[CkMyPe()]);
    thisProxy[partner()].p2p(CkSendBuffer(sendBuf, cb), bufSize, CkMyPe(), variant, sent++);
  }

  // A partner's first message of a phase can arrive before this PE's
  // startP2p, so the counters are reset when a phase ends, not when it starts.
  void maybeDone() {
    if (received < numMsgs || acked < numMsgs || sourcesDone < numMsgs) return;
    sent = acked = received = sourcesDone = 0;
    contribute(CkCallback(CkReductionTarget(Main, p2pDone), mainProxy));
  }

public:
  Peer() {
    sendBuf = new char[bufSize];
    postBuf = new char[bufSize];
    preregBuf = (char *)CkRdmaAlloc(bufSize);
    for (int i = 0; i < bufSize; i++) sendBuf[i] = pattern(CkMyPe(), i);
  }

  void startP2p(int v) {
    variant = v;
    sendNext();
  }

  void p2p(char *data, int n, int from, int v, int round, CkNcpyBufferPost *ncpyPost) {
    variant = v;
    post(n, ncpyPost);
  }

  void p2p(char *data, int n, int from, int v, int round) {
    check("p2p", data, n, from);
    received++;
    thisProxy[from].p2pAck();
    maybeDone();
  }

  void p2pAck() {
    acked++;
    sendNext();
    maybeDone();
  }

  void sourceDone(CkDataMsg *m) {
    delete m;
    sourcesDone++;
    maybeDone();
  }

  void bcast(char *data, int n, int v, int round, CkNcpyBufferPost *ncpyPost) {
    variant = v;
    post(n, ncpyPost);
  }

  void bcast(char *data, int n, int v, int round) {
    check("broadcast", data, n, 0);
    contribute(CkCallback(CkReductionTarget(Main, bcastRoundDone), mainProxy));
  }

  void report() {
    if (CkMyRank() == 0) mainProxy.count(CkMyNode(), registeredBuffers());
  }
};

#include "zc_post_noleak.def.h"
