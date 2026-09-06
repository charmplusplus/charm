#include "deferred_send.decl.h"
#include "hapi.h"
#include <vector>

CProxy_Main mainProxy;
extern void delayedFill(int*, int*, int, unsigned long long, cudaStream_t);
static const int count = 4096;

class Main : public CBase_Main {
  CProxy_Endpoint endpoints;
  int readyCount = 0;
  bool sent = false, received = false, progressed = false;
 public:
  Main(CkArgMsg* m) {
    delete m;
    mainProxy = thisProxy;
    int peer = -1;
    for (int pe = 1; pe < CkNumPes(); ++pe)
      if (!CmiPeOnSamePhysicalNode(0, pe)) { peer = pe; break; }
    if (peer < 0)
      CkAbort("deferred_send requires PEs on two physical hosts (network device path)");
    endpoints = CProxy_Endpoint::ckNew();
    endpoints[0].insert(0);
    endpoints[1].insert(peer);
    endpoints.doneInserting();
  }
  void ready() { if (++readyCount == 2) endpoints[0].start(); }
  void senderDone() { CkAssert(!sent); sent = true; }
  void receiverDone() { CkAssert(!received); received = true; }
  void progress() { CkAssert(!progressed); progressed = true; }
  void quiescent() {
    if (!sent || !received || !progressed)
      CkAbort("Quiescence reported before the deferred send completed");
    CkPrintf("TEST PASSED: deferred multi-stream send, payloads, callbacks, and QD\n");
    CkExit();
  }
};

class Endpoint : public CBase_Endpoint {
  int *a, *b, *c;
  cudaStream_t fast, slow;
  cudaEvent_t produced;
  int completions = 0;
  unsigned long long cycles;
 public:
  Endpoint() {
    int device, clockKHz;
    hapiCheck(cudaGetDevice(&device));
    hapiCheck(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, device));
    // About half a second: long enough to observe an unfinished producer
    // immediately after returning from the proxy, below display watchdog limits.
    cycles = static_cast<unsigned long long>(clockKHz) * 500;
    hapiCheck(cudaStreamCreateWithFlags(&fast, cudaStreamNonBlocking));
    hapiCheck(cudaStreamCreateWithFlags(&slow, cudaStreamNonBlocking));
    hapiCheck(cudaEventCreateWithFlags(&produced, cudaEventDisableTiming));
    hapiCheck(cudaMalloc(&a, count * sizeof(int)));
    hapiCheck(cudaMalloc(&b, count * sizeof(int)));
    hapiCheck(cudaMalloc(&c, count * sizeof(int)));
    // Warm up module loading before the nonblocking assertion.
    delayedFill(a, c, count, 0, fast);
    hapiCheck(cudaMemsetAsync(a, 0, count * sizeof(int), fast));
    hapiCheck(cudaMemsetAsync(c, 0, count * sizeof(int), fast));
    hapiCheck(cudaMemsetAsync(b, 0, count * sizeof(int), slow));
    hapiCheck(cudaStreamSynchronize(fast));
    hapiCheck(cudaStreamSynchronize(slow));
    mainProxy.ready();
  }
  void start() {
    delayedFill(a, c, count, cycles / 5, fast);
    delayedFill(b, nullptr, count, cycles, slow);
    hapiCheck(cudaEventRecord(produced, slow));
    int tags[] = {17, 29, 43};
    const CkCallback cb(CkIndex_Endpoint::sent(), thisProxy[thisIndex]);
    // Temporary proxy, stack-local descriptors, two distinct producing
    // streams, and two buffers sharing a stream must all survive deferral.
    thisProxy[1].receive(count, CkDeviceBuffer(a, cb, fast),
        CkDeviceBuffer(b, cb, slow), CkDeviceBuffer(c, cb, fast), 3, tags);
    tags[0] = tags[1] = tags[2] = -1;
    if (cudaEventQuery(produced) != cudaErrorNotReady)
      CkAbort("Device send waited for its producer instead of returning asynchronously");
    thisProxy[thisIndex].progress();
    CkStartQD(CkCallback(CkIndex_Main::quiescent(), mainProxy));
  }
  void progress() { mainProxy.progress(); }
  void sent() {
    if (++completions > 3) CkAbort("Duplicate source completion callback");
    if (completions == 3) mainProxy.senderDone();
  }
  void receive(int& n, int*& x, int*& y, int*& z, int& ntags, int* tags,
               CkDeviceBufferPost* post) {
    CkAssert(n == count);
    x = a; y = b; z = c;
    for (int i = 0; i < 3; ++i) post[i].hapi_stream = fast;
  }
  void receive(int n, int* x, int* y, int* z, int ntags, int* tags) {
    if (ntags != 3 || tags[0] != 17 || tags[1] != 29 || tags[2] != 43)
      CkAbort("Deferred send did not retain its host arguments");
    std::vector<int> host(n);
    int* buffers[] = {x, y, z};
    const int expected[] = {11, 11, 33};
    for (int j = 0; j < 3; ++j) {
      hapiCheck(cudaMemcpy(host.data(), buffers[j], n * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < n; ++i)
        if (host[i] != expected[j]) CkAbort("RDMA read an unfinished producer buffer");
    }
    mainProxy.receiverDone();
  }
};

#include "deferred_send.def.h"
