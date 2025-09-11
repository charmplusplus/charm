// osu_bw.C
#include "osu_bw.decl.h"
#include <cstdlib>
#include <cstring>

static const int DEFAULT_MIN = 1;
static const int DEFAULT_MAX = 1<<22; // 4 MiB
static const int DEFAULT_ITERS = 1000;
static const int DEFAULT_SKIP  = 100;
static const int DEFAULT_WIN   = 64;

class Endpoint;
class Main;

class DataMsg : public CMessage_DataMsg {
  public:
    int size;
    char* data;
};

class Main : public CBase_Main {
  CProxy_Endpoint sender, receiver;
  int minSize, maxSize, iters, skip, win;
  int curSize;
  double tMeasured;
  int pendingReports;
  int ready_count;

 public:
  Main(CkArgMsg* m) {
    // Parse arguments
    minSize = DEFAULT_MIN; maxSize = DEFAULT_MAX; iters = DEFAULT_ITERS; skip = DEFAULT_SKIP; win = DEFAULT_WIN;
    ready_count = 0;
    for (int i=1; i<m->argc; ++i) {
      if (!strcmp(m->argv[i], "-m") && i+1<m->argc) minSize = atoi(m->argv[++i]);
      else if (!strcmp(m->argv[i], "-M") && i+1<m->argc) maxSize = atoi(m->argv[++i]);
      else if (!strcmp(m->argv[i], "-i") && i+1<m->argc) iters   = atoi(m->argv[++i]);
      else if (!strcmp(m->argv[i], "-s") && i+1<m->argc) skip    = atoi(m->argv[++i]);
      else if (!strcmp(m->argv[i], "-w") && i+1<m->argc) win     = atoi(m->argv[++i]);
    }
    delete m;

    // Create endpoints - handle case where we only have 1 PE
    if (CkNumPes() < 2) {
      CkPrintf("Error: Need at least 2 PEs to run bandwidth test\n");
      CkExit();
      return;
    }
    sender   = CProxy_Endpoint::ckNew(thisProxy, iters, win, skip, 0);
    receiver = CProxy_Endpoint::ckNew(thisProxy, iters, win, skip, 1);
    sender.setPeer(receiver);
    receiver.setPeer(sender);

    // Header like OMB
    CkPrintf("# OSU-style Bandwidth (Charm++)\n# Size       MB/s (MB=1e6)\n");

    curSize = minSize;
    pendingReports = 0;
  }

  void ready()
  {
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
    pendingReports = 1;
    sender.start(curSize);
  }

  void doneOne(double seconds) {
    // Compute bandwidth in MB/s (decimal)
    double bytes = double(curSize) * double(iters) * double(win);
    double mbps  = bytes / seconds / 1.0e6;
    CkPrintf("%-10d %.2f\n", curSize, mbps);
    curSize = (curSize < 1024 ? curSize*2 : curSize + 1024);
    nextSize();
  }

  void finish() {
    CkExit();
  }
};

class Endpoint : public CBase_Endpoint {
  CProxy_Endpoint peer;
  CProxy_Main mainProxy;
  int size, iters, window, skip;
  int iter, inFlightRecv, recvInIter;
  double t0;

 public:
  Endpoint(CProxy_Main m, int iters_, int window_, int skip_) : 
              mainProxy(m), size(0), iters(iters_), window(window_), skip(skip_),
              iter(0), inFlightRecv(0), recvInIter(0), t0(0.0) {}

  void setPeer(CProxy_Endpoint p) { 
    peer = p;
    mainProxy.ready();
  }

  void start(int size_) {
    size = size_; iter = 0; recvInIter = 0; inFlightRecv = 0; t0 = 0.0;
    // Warmups + measured
    // Kick off first window - but only from sender (PE 0)
    if (CkMyPe() == 0) {
      //CkPrintf("Starting bandwidth test: size=%d, iters=%d, window=%d, skip=%d\n", 
      //         size, iters, window, skip);
      sendWindow();
    } else if (CkMyPe() == 1) {
      CkPrintf("Receiver ready on PE %d\n", CkMyPe());
    }
  }

  void sendWindow() {
    // Start timer at end of warmup
    if (iter == skip) t0 = CkWallTimer();
    for (int w = 0; w < window; ++w) {
      DataMsg* m = new (size) DataMsg;
      //DataMsg* m = (DataMsg*)CkAllocMsg(DataMsg, sizeof(DataMsg) + size);
      m->size = size;
      // touch payload to avoid lazy effects
      if (size > 0) memset(m->data, w, size);
      peer.recv(m);
    }
    // Wait for ack from receiver to proceed to next window/iter
  }

  void recv(DataMsg* m) {
    // Receiver counts messages and acks per window
    recvInIter++;
    //CkPrintf("Received message of size %d on PE %d, %d, %d\n", m->size, CkMyPe(), recvInIter, window);
    if (recvInIter == window) {
      recvInIter = 0;
      peer.ack();
    }
    delete m;
  }

  void ack() {
    // Sender advances iteration
    iter++;
    // After warmups + measured iterations, stop and report
    if (iter == skip + iters) {
      double t = CkWallTimer() - t0;
      if (CkMyPe() == 0) {
        //CkPrintf("Test completed, reporting results\n");
        mainProxy.doneOne(t);
      }
      return;
    }
    // Otherwise send next window
    if (CkMyPe() == 0) sendWindow();
  }
};

#include "osu_bw.def.h"

