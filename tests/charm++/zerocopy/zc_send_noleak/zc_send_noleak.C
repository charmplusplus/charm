// Sends zerocopy (nocopy) messages back and forth between the first and the
// last PE and checks that the memory in use (CmiMemoryUsage) of each process
// does not grow with the number of messages. Each message also carries a
// 100 KB marshalled pad, so a message the runtime fails to free costs about
// 128 KB of message pool, and 1000 round trips would leak over 100 MB per
// process. The check guards the receiver of an RDMA transfer, which needs the
// two PEs in different processes; within one process the transfer is a memcpy.
#include "zc_send_noleak.decl.h"

CProxy_Main mainProxy;
int numIters;

static const int bufSize = 64;
static const int padSize = 100 * 1024;
static const int warmupIters = 50;
static const long maxGrowth = 32L * 1024 * 1024;

class Main : public CBase_Main {
  CProxy_Peer peers;

public:
  Main(CkArgMsg *m) {
    numIters = (m->argc > 1) ? atoi(m->argv[1]) : 1000;
    delete m;
    if (numIters <= warmupIters)
      CkAbort("zc_send_noleak: the iteration count must exceed %d\n", warmupIters);
    mainProxy = thisProxy;
    CkPrintf("zc_send_noleak: %d iterations on %d PEs in %d processes\n",
             numIters, CkNumPes(), CkNumNodes());
    peers = CProxy_Peer::ckNew();
    peers[0].startP2p();
  }

  void check(const char *what, long growth) {
    CkPrintf("zc_send_noleak: %s: memory in use grew by %ld KB over %d messages\n",
             what, growth / 1024, numIters - warmupIters);
    if (growth > maxGrowth)
      CkAbort("zc_send_noleak: %s: memory in use grew by %ld KB (limit %ld KB); "
              "zerocopy messages are not freed\n", what, growth / 1024, maxGrowth / 1024);
  }

  void p2pDone(long growth) {
    check("p2p send", growth);
    CkExit();
  }
};

class Peer : public CBase_Peer {
  char buf[bufSize];
  std::vector<char> pad;
  CmiUInt8 base = 0;
  long growth = 0;

  int partner() const { return CkMyPe() == 0 ? CkNumPes() - 1 : 0; }

  // Records the memory in use after the warmup and its growth since then.
  void sample(int iter) {
    if (iter == warmupIters)
      base = CmiMemoryUsage();
    else if (iter == numIters)
      growth = (long)CmiMemoryUsage() - (long)base;
  }

public:
  Peer() : pad(padSize, 7) {
    for (int i = 0; i < bufSize; i++) buf[i] = (char)(i + CkMyPe());
  }

  void startP2p() {
    thisProxy[partner()].p2p(1, CkSendBuffer(buf), bufSize, pad.data(), padSize);
  }

  // The reply is sent only after this entry method runs, which is after the
  // transfer into the receiver's copy completed, so buf is free to reuse.
  void p2p(int iter, char *data, int n, char *, int) {
    const int from = partner();
    if (n != bufSize || data[1] != (char)(1 + from))
      CkAbort("zc_send_noleak: p2p message %d holds the wrong bytes\n", iter);
    sample(iter);
    if (CkMyPe() == 0 && iter == numIters) {
      thisProxy.reportP2p();
      return;
    }
    // PE 0 counts the round trips; the last PE echoes.
    const int next = (CkMyPe() == 0) ? iter + 1 : iter;
    thisProxy[from].p2p(next, CkSendBuffer(buf), bufSize, pad.data(), padSize);
  }

  void reportP2p() {
    contribute(sizeof(long), &growth, CkReduction::max_long,
               CkCallback(CkReductionTarget(Main, p2pDone), mainProxy));
  }
};

#include "zc_send_noleak.def.h"
