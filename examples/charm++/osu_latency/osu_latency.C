// osu_latency.C
#include "osu_latency.decl.h"
#include <cstdlib>
#include <cstring>

static const int DEFAULT_MIN = 0;
static const int DEFAULT_MAX = 1<<22; // 4 MiB
static const int DEFAULT_ITERS = 10000;
static const int DEFAULT_SKIP  = 1000;

CProxy_Main mainProxy;

class Endpoint;
class Main;

class LatencyMsg : public CMessage_LatencyMsg {
  public:
    int size;
    char* data;
};

class Main : public CBase_Main {
  CProxy_Endpoint sender, receiver;
  int minSize, maxSize, iters, skip;
  int curSize;
  int ready_count;

 public:
  Main(CkArgMsg* m) {
    // Parse arguments
    minSize = DEFAULT_MIN; maxSize = DEFAULT_MAX; iters = DEFAULT_ITERS; skip = DEFAULT_SKIP;
    ready_count = 0;
    for (int i=1; i<m->argc; ++i) {
      if (!strcmp(m->argv[i], "-m") && i+1<m->argc) minSize = atoi(m->argv[++i]);
      else if (!strcmp(m->argv[i], "-M") && i+1<m->argc) maxSize = atoi(m->argv[++i]);
      else if (!strcmp(m->argv[i], "-i") && i+1<m->argc) iters   = atoi(m->argv[++i]);
      else if (!strcmp(m->argv[i], "-s") && i+1<m->argc) skip    = atoi(m->argv[++i]);
    }
    delete m;

    // Create endpoints - handle case where we only have 1 PE
    if (CkNumPes() < 2) {
      CkPrintf("Error: Need at least 2 PEs to run latency test\n");
      CkExit();
      return;
    }
    sender   = CProxy_Endpoint::ckNew(thisProxy, iters, skip, 0);
    receiver = CProxy_Endpoint::ckNew(thisProxy, iters, skip, 1);
    sender.setPeer(receiver);
    receiver.setPeer(sender);
    mainProxy = thisProxy;

    // Header like OMB
    CkPrintf("# OSU-style Latency (Charm++)\n# Size          Latency (us)\n");

    curSize = minSize;
  }

  void ready() {
    ready_count++;
    if (ready_count == 2) {
      nextSize();
    }
  }

  void nextSize() {
    if (curSize > maxSize) {
      finish();
      return;
    }
    sender.start(curSize);
  }

  void doneOne(double seconds) {
    // Compute latency in microseconds (round-trip / 2)
    double latency_us = (seconds / (2.0 * double(iters))) * 1.0e6;
    CkPrintf("%-12d    %.2f\n", curSize, latency_us);
    curSize = (curSize == 0) ? 1 : curSize * 2;
    nextSize();
  }

  void finish() {
    CkExit();
  }
};

class Endpoint : public CBase_Endpoint {
  CProxy_Endpoint peer;
  int size, iters, skip;
  int iter;
  double t0;
  bool is_sender;

 public:
  Endpoint(CProxy_Main m, int iters_, int skip_) : 
              size(0), iters(iters_), skip(skip_),
              iter(0), t0(0.0), is_sender(false) {}

  void setPeer(CProxy_Endpoint p) { 
    peer = p;
    is_sender = (CkMyPe() == 0);
    mainProxy.ready();
  }

  void start(int size_) {
    size = size_; 
    iter = 0; 
    t0 = 0.0;
    
    if (is_sender) {
      // Start the ping-pong
      sendPing();
    }
  }

  void sendPing() {
    // Start timer at end of warmup
    if (iter == skip) {
      t0 = CkWallTimer();
    }
    
    LatencyMsg* m = new (size) LatencyMsg;
    m->size = size;
    // touch payload to avoid lazy effects
    if (size > 0) {
      memset(m->data, iter % 256, size);
    }
    peer.ping(m);
  }

  void ping(LatencyMsg* m) {
    // Receiver gets ping and sends pong back
    if (!is_sender) {
      LatencyMsg* reply = new (m->size) LatencyMsg;
      reply->size = m->size;
      if (m->size > 0) {
        memcpy(reply->data, m->data, m->size);
      }
      peer.pong(reply);
    }
    delete m;
  }

  void pong(LatencyMsg* m) {
    // Sender gets pong back, completes one iteration
    if (is_sender) {
      iter++;
      
      // After warmups + measured iterations, stop and report
      if (iter == skip + iters) {
        double t = CkWallTimer() - t0;
        mainProxy.doneOne(t);
        delete m;
        return;
      }
      
      // Otherwise send next ping
      delete m;
      sendPing();
    }
  }
};

#include "osu_latency.def.h"
