// Acceptance test for GPU-direct device-to-device messaging -- GPU migration
// plan stage 9.2.
//
// Self-checking (bcastred house style: every check is a CkEnforce, success is
// one PASS line, any failure aborts). Backend-neutral by construction, so the
// same source is the acceptance run on an NVIDIA and on an AMD machine.
//
// What it covers, and why each piece is worth a check:
//
//   1. All three transfer modes, and that the ones the topology makes
//      reachable were actually taken. findTransferModeDevice() picks MEMCPY
//      (same process), IPC (same physical node, different process) or RDMA
//      (different physical nodes); each of those is a completely separate code
//      path in CkRdmaDeviceIssueRgets/CkRdmaDeviceOnSender. Senders tally the
//      mode they used and Main asserts the tally against CmiNumNodes() and
//      CmiNumPhysicalNodes() -- so a run that quietly degraded to a single
//      path (the classic way this breaks) fails instead of passing.
//
//   2. Payload correctness against the *sender's* pattern. Every buffer is
//      filled with a value derived from the sender's index, the iteration and
//      which of the two buffers it is. A receiver that gets a stale buffer,
//      its own buffer, or its neighbour's other buffer sees a value that does
//      not match and aborts. The values are small integers scaled by powers of
//      ten, so the comparison is exact and needs no tolerance.
//
//   3. Two device buffers per message. numops > 1 drives the multi-op
//      bookkeeping in DeviceRdmaInfo (the n_ops/counter pair that decides when
//      the real entry method finally runs) rather than the degenerate
//      single-buffer case.
//
//   4. Source callbacks. Each CkDeviceBuffer carries a completion callback,
//      counted per element; the total must be exactly one per buffer sent.
//      This is what tells the sender its buffer is reusable, and it is
//      delivered from a different place in each of the three modes.
//
//   5. Quiescence bracketing. Iterations are separated by CkWaitQD(), so the
//      QdCreate/QdProcess pairs the D2D path adds -- including the ones around
//      loopback_bridge, which bounces an inter-node completion to the
//      destination PE -- have to balance. If they over-count, QD never fires
//      and the run hangs; if they under-count, QD fires before the data lands
//      and check 2 fails.
//
//   6. Chare arrays, groups and nodegroups, which reach the device path
//      through different proxy machinery.
//
// Flags:  -s <doubles per buffer, default 1024>   -i <iterations, default 20>
//         -e <ring elements, default 2*PEs>       -v (verbose)
//
// Run (reconverse, two Frontier nodes, to reach the RDMA path):
//   srun -N2 -n2 -c56 --gpus-per-node=8 ./d2dtest +pe 4

#include "d2dtest.decl.h"
#include "hapi.h"
#include "conv-rdmadevice.h"

#include <cmath>
#include <cstdlib>
#include <unistd.h>

// CkDeviceBufferPost names its stream field differently in the two runtimes:
// the reconverse half of ckrdmadevice.h carries hapi_stream, the classic half
// still spells it cuda_stream. Only the name differs.
#if CMK_RECONVERSE
#define D2D_POST_STREAM(p) ((p).hapi_stream)
#else
#define D2D_POST_STREAM(p) ((p).cuda_stream)
#endif

extern hapiError_t invokeFillKernel(double* d, int n, double base, int period,
                                    hapiStream_t stream);

#define D2D_PERIOD 7

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Ring ringProxy;
/* readonly */ CProxy_RingGroup groupProxy;
/* readonly */ CProxy_RingNodeGroup nodegroupProxy;
/* readonly */ int blockSize;
/* readonly */ int nIters;
/* readonly */ int nElems;
/* readonly */ int verbose;

// Tally slots, summed across every sender/receiver in the run.
enum {
  T_RECV = 0,     // messages whose payload was verified
  T_VALUES,       // individual doubles compared
  T_CALLBACK,     // source (buffer-reusable) callbacks delivered
  T_MEMCPY,       // sends that took the same-process path
  T_IPC,          // sends that took the same-physical-node path
  T_RDMA,         // sends that took the inter-node path
  T_REPORTERS,
  T_WIDTH
};

// Distinct, exactly representable value for (id, iter, buf, j).
static inline double d2dBase(int id, int iter, int buf) {
  return (double)id * 1000000.0 + (double)iter * 1000.0 + (double)buf * 10.0;
}

// One sender/receiver's device state. Two send buffers and two receive
// buffers so a message can carry two nocopydevice parameters.
struct Endpoint {
  double* d_send[2];
  double* d_recv[2];
  double* h_check;
  hapiStream_t stream;
  int tally[T_WIDTH];

  Endpoint() {
    for (int i = 0; i < T_WIDTH; i++) tally[i] = 0;
    hapiCheck(hapiStreamCreate(&stream));
    for (int b = 0; b < 2; b++) {
      hapiCheck(hapiMalloc((void**)&d_send[b], sizeof(double) * blockSize));
      hapiCheck(hapiMalloc((void**)&d_recv[b], sizeof(double) * blockSize));
    }
    hapiCheck(hapiMallocHost((void**)&h_check, sizeof(double) * blockSize));
  }

  ~Endpoint() {
    for (int b = 0; b < 2; b++) {
      hapiCheck(hapiFree(d_send[b]));
      hapiCheck(hapiFree(d_recv[b]));
    }
    hapiCheck(hapiFreeHost(h_check));
    hapiCheck(hapiStreamDestroy(stream));
  }

  // Refill the send buffers so every iteration carries a new pattern; a
  // receiver that verifies a stale buffer is therefore caught.
  void fill(int id, int iter, int nbufs) {
    for (int b = 0; b < nbufs; b++) {
      hapiCheck(invokeFillKernel(d_send[b], blockSize, d2dBase(id, iter, b),
                                 D2D_PERIOD, stream));
    }
    hapiCheck(hapiStreamSynchronize(stream));
  }

  // Check that d_recv[buf] holds exactly what sender 'srcId' put in its
  // buffer 'buf' on iteration 'iter'.
  void verify(int srcId, int iter, int buf) {
    hapiCheck(hapiMemcpyAsync(h_check, d_recv[buf], sizeof(double) * blockSize,
          hapiMemcpyDeviceToHost, stream));
    hapiCheck(hapiStreamSynchronize(stream));

    const double base = d2dBase(srcId, iter, buf);
    for (int j = 0; j < blockSize; j++) {
      const double expect = base + (double)(j % D2D_PERIOD);
      if (h_check[j] != expect) {
        CkAbort("d2dtest: PE %d buffer %d from source %d iter %d: index %d is "
                "%.1f, expected %.1f", CkMyPe(), buf, srcId, iter, j,
                h_check[j], expect);
      }
      tally[T_VALUES]++;
    }
    tally[T_RECV]++;
  }

  // Record which of the three code paths this send will take.
  void countMode(int destPe) {
    CmiNcpyModeDevice m = findTransferModeDevice(CkMyPe(), destPe);
    if (m == CmiNcpyModeDevice::MEMCPY)   tally[T_MEMCPY]++;
    else if (m == CmiNcpyModeDevice::IPC) tally[T_IPC]++;
    else                                  tally[T_RDMA]++;
  }
};

// ---------------------------------------------------------------------------
// Chare array: the main workhorse, two device buffers per message
// ---------------------------------------------------------------------------
class Ring : public CBase_Ring {
  Endpoint* ep;

public:
  Ring() : ep(new Endpoint()) {}
  ~Ring() { delete ep; }

  void send(int iter) {
    ep->fill(thisIndex, iter, 2);
    const int dst = (thisIndex + 1) % nElems;
    // Ask the location manager where the neighbour actually is, rather than
    // assuming the default mapping; the mode tally has to describe the path
    // the message really takes. Nothing here migrates, so the cached location
    // and the home PE agree; fall back to the home PE before the cache is
    // populated.
    CkLocMgr* locMgr = ringProxy.ckLocMgr();
    const CkArrayIndex1D dstIdx(dst);
    int dstPe = locMgr->whichPe(dstIdx);
    if (dstPe < 0) dstPe = locMgr->homePe(dstIdx);
    ep->countMode(dstPe);
    ringProxy[dst].recv(thisIndex, iter, blockSize,
        CkDeviceBuffer(ep->d_send[0],
          CkCallback(CkIndex_Ring::reuse(), thisProxy[thisIndex]), ep->stream),
        CkDeviceBuffer(ep->d_send[1],
          CkCallback(CkIndex_Ring::reuse(), thisProxy[thisIndex]), ep->stream));
  }

  // Post entry method: hand the runtime the destination buffers and the
  // stream the transfers should run on.
  void recv(int& srcIdx, int& iter, int& size, double*& a, double*& b,
            CkDeviceBufferPost* post) {
    a = ep->d_recv[0];
    b = ep->d_recv[1];
    D2D_POST_STREAM(post[0]) = ep->stream;
    D2D_POST_STREAM(post[1]) = ep->stream;
  }

  void recv(int srcIdx, int iter, int size, double* a, double* b) {
    CkEnforce(size == blockSize);
    ep->verify(srcIdx, iter, 0);
    ep->verify(srcIdx, iter, 1);
    if (verbose) {
      CkPrintf("[d2dtest] array %d on PE %d verified iter %d from %d\n",
               thisIndex, CkMyPe(), iter, srcIdx);
    }
  }

  void reuse() { ep->tally[T_CALLBACK]++; }

  void collect() {
    ep->tally[T_REPORTERS] = 1;
    contribute(sizeof(ep->tally), ep->tally, CkReduction::sum_int,
               CkCallback(CkIndex_Main::tally(NULL), mainProxy));
  }
};

// ---------------------------------------------------------------------------
// Group: one buffer per message, PE p -> PE p+1
// ---------------------------------------------------------------------------
class RingGroup : public CBase_RingGroup {
  Endpoint* ep;

public:
  RingGroup() : ep(new Endpoint()) {}
  ~RingGroup() { delete ep; }

  void send(int iter) {
    ep->fill(CkMyPe(), iter, 1);
    const int dst = (CkMyPe() + 1) % CkNumPes();
    ep->countMode(dst);
    groupProxy[dst].recv(CkMyPe(), iter, blockSize,
        CkDeviceBuffer(ep->d_send[0],
          CkCallback(CkIndex_RingGroup::reuse(), thisProxy[CkMyPe()]), ep->stream));
  }

  void recv(int& srcPe, int& iter, int& size, double*& a,
            CkDeviceBufferPost* post) {
    a = ep->d_recv[0];
    D2D_POST_STREAM(post[0]) = ep->stream;
  }

  void recv(int srcPe, int iter, int size, double* a) {
    CkEnforce(size == blockSize);
    ep->verify(srcPe, iter, 0);
  }

  void reuse() { ep->tally[T_CALLBACK]++; }

  void collect() {
    ep->tally[T_REPORTERS] = 1;
    contribute(sizeof(ep->tally), ep->tally, CkReduction::sum_int,
               CkCallback(CkIndex_Main::tally(NULL), mainProxy));
  }
};

// ---------------------------------------------------------------------------
// Nodegroup: one buffer per message, node n -> node n+1
// ---------------------------------------------------------------------------
class RingNodeGroup : public CBase_RingNodeGroup {
  Endpoint* ep;

public:
  RingNodeGroup() : ep(new Endpoint()) {}
  ~RingNodeGroup() { delete ep; }

  void send(int iter) {
    ep->fill(CkMyNode(), iter, 1);
    const int dst = (CkMyNode() + 1) % CkNumNodes();
    ep->countMode(CmiNodeFirst(dst));
    nodegroupProxy[dst].recv(CkMyNode(), iter, blockSize,
        CkDeviceBuffer(ep->d_send[0],
          CkCallback(CkIndex_RingNodeGroup::reuse(), thisProxy[CkMyNode()]),
          ep->stream));
  }

  void recv(int& srcNode, int& iter, int& size, double*& a,
            CkDeviceBufferPost* post) {
    a = ep->d_recv[0];
    D2D_POST_STREAM(post[0]) = ep->stream;
  }

  void recv(int srcNode, int iter, int size, double* a) {
    CkEnforce(size == blockSize);
    ep->verify(srcNode, iter, 0);
  }

  void reuse() { ep->tally[T_CALLBACK]++; }

  void collect() {
    ep->tally[T_REPORTERS] = 1;
    contribute(sizeof(ep->tally), ep->tally, CkReduction::sum_int,
               CkCallback(CkIndex_Main::tally(NULL), mainProxy));
  }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
class Main : public CBase_Main {
  double startTime;
  int total[T_WIDTH];
  int phase;
  int expectSends;      // messages the just-finished phase should have sent
  int buffersPerMsg;
  CthThread waiting;

public:
  Main(CkArgMsg* m) {
    mainProxy = thisProxy;
    blockSize = 1024;
    nIters = 20;
    nElems = 0;
    verbose = 0;
    phase = 0;
    waiting = NULL;

    int c;
    while ((c = getopt(m->argc, m->argv, "s:i:e:v")) != -1) {
      switch (c) {
        case 's': blockSize = atoi(optarg); break;
        case 'i': nIters = atoi(optarg); break;
        case 'e': nElems = atoi(optarg); break;
        case 'v': verbose = 1; break;
        default: CkAbort("d2dtest: unknown command line argument");
      }
    }
    delete m;

    if (nElems <= 0) nElems = 2 * CkNumPes();
    CkEnforce(blockSize > 0);
    CkEnforce(nIters > 0);
    // A ring needs at least two distinct endpoints, otherwise every element
    // would send to itself and nothing crosses a PE.
    CkEnforce(nElems >= 2);
    CkEnforce(CkNumPes() >= 2);

    CkPrintf("[d2dtest] %d PEs, %d processes, %d physical nodes | "
             "%d ring elements, %d doubles/buffer, %d iters\n",
             CkNumPes(), CkNumNodes(), CmiNumPhysicalNodes(),
             nElems, blockSize, nIters);

    ringProxy = CProxy_Ring::ckNew(nElems);
    groupProxy = CProxy_RingGroup::ckNew();
    nodegroupProxy = CProxy_RingNodeGroup::ckNew();

    thisProxy.run();
  }

  // Register this thread as the one waiting for the next collect() reduction.
  // Done before the reduction is launched, not after, so there is no window in
  // which tally() could arrive with no thread to wake.
  void startTally() {
    for (int i = 0; i < T_WIDTH; i++) total[i] = 0;
    waiting = CthSelf();
  }

  void awaitTally() { CthSuspend(); }

  void tally(CkReductionMsg* m) {
    CkEnforce(m->getSize() == (int)sizeof(total));
    memcpy(total, m->getData(), sizeof(total));
    delete m;
    CthAwaken(waiting);
  }

  // Common assertions for one phase: nsenders endpoints each sent nIters
  // messages of nbufs buffers.
  void checkPhase(const char* what, int nsenders, int nbufs) {
    const int msgs = nsenders * nIters;
    CkEnforce(total[T_RECV] == msgs * nbufs);
    CkEnforce(total[T_VALUES] == msgs * nbufs * blockSize);
    CkEnforce(total[T_CALLBACK] == msgs * nbufs);

    CkEnforce(total[T_REPORTERS] == nsenders);

    const int modes = total[T_MEMCPY] + total[T_IPC] + total[T_RDMA];
    CkEnforce(modes == msgs);

    // The modes actually taken have to match what the job layout allows.
    if (CkNumNodes() == 1) {
      CkEnforce(total[T_IPC] == 0);
      CkEnforce(total[T_RDMA] == 0);
      CkEnforce(total[T_MEMCPY] == msgs);
    } else {
      // With more than one process, a ring must cross a process boundary.
      CkEnforce(total[T_IPC] + total[T_RDMA] > 0);
    }
    if (CmiNumPhysicalNodes() == 1) {
      CkEnforce(total[T_RDMA] == 0);
    } else {
      // With more than one physical node, a ring must cross one.
      CkEnforce(total[T_RDMA] > 0);
    }

    CkPrintf("[d2dtest] %-9s ok: %d msgs (%d buffers), modes memcpy=%d ipc=%d rdma=%d\n",
             what, msgs, msgs * nbufs, total[T_MEMCPY], total[T_IPC], total[T_RDMA]);
  }

  void run() {
    startTime = CkWallTimer();

    // ---- 1. chare array, two device buffers per message ----
    for (int it = 0; it < nIters; it++) {
      for (int e = 0; e < nElems; e++) ringProxy[e].send(it);
      CkWaitQD();
    }
    startTally();
    ringProxy.collect();
    awaitTally();
    checkPhase("array", nElems, 2);
    const int arrayValues = total[T_VALUES];

    // ---- 2. group ----
    for (int it = 0; it < nIters; it++) {
      groupProxy.send(it);
      CkWaitQD();
    }
    startTally();
    groupProxy.collect();
    awaitTally();
    checkPhase("group", CkNumPes(), 1);
    const int groupValues = total[T_VALUES];

    // ---- 3. nodegroup ----
    int nodeValues = 0;
    if (CkNumNodes() >= 2) {
      for (int it = 0; it < nIters; it++) {
        nodegroupProxy.send(it);
        CkWaitQD();
      }
      startTally();
      nodegroupProxy.collect();
      awaitTally();
      checkPhase("nodegroup", CkNumNodes(), 1);
      nodeValues = total[T_VALUES];
    } else {
      CkPrintf("[d2dtest] nodegroup skipped: single process\n");
    }

    CkPrintf("d2dtest PASS: %d PEs, %d processes, %d physical nodes, "
             "%lld doubles verified, %.3f s\n",
             CkNumPes(), CkNumNodes(), CmiNumPhysicalNodes(),
             (long long)arrayValues + groupValues + nodeValues,
             CkWallTimer() - startTime);
    CkExit();
  }
};

#include "d2dtest.def.h"
