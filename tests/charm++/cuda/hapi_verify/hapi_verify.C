// hapi_verify: self-checking acceptance test for the HAPI portable core
// (GPU series stage 9.1). Exercises, per PE:
//   1. device mapping sanity (hapiMyDevice vs the visible device count)
//   2. the pinned-host pool / buddy allocator (interleaved alloc/free of
//      varied sizes, each buffer pattern-filled and verified before free)
//   3. async completion: kernels writing device buffers, async copies back
//      into pool memory, completions delivered via hapiAddCallback
// Prints exactly one "HAPI_VERIFY PASS" line and exits 0 on success; any
// failure trips a CkEnforce/abort. Requires a GPU at runtime; see
// doc/reconverse-merge-plan.md for the GPU-series validation protocol.
//
// Flags: -n <pool buffers, default 64>  -k <kernels in flight, default 32>
//        -w <watchdog seconds, default 60; 0 disables>

#include "hapi_verify.decl.h"
#include "hapi.h"
#include <vector>
#include <cstring>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nBufs;
/*readonly*/ int nKernels;
/*readonly*/ int wdSecs;

extern "C" void launchFill(int* dbuf, int n, int val, cudaStream_t stream);

static void watchdogAbort(void* /*unused*/, double /*curT*/) {
  CkAbort("hapi_verify: watchdog fired -- async completions stalled");
}

class Main : public CBase_Main {
 public:
  Main(CkArgMsg* m) {
    nBufs = 64; nKernels = 32; wdSecs = 60;
    CmiGetArgInt(m->argv, "-n", &nBufs);
    CmiGetArgInt(m->argv, "-k", &nKernels);
    CmiGetArgInt(m->argv, "-w", &wdSecs);
    delete m;
    CkEnforce(nBufs >= 4 && nKernels >= 1);
    mainProxy = thisProxy;
    CkPrintf("hapi_verify: %d PEs, %d pool buffers, %d kernels in flight per PE\n",
             CkNumPes(), nBufs, nKernels);
    CProxy_Tester tester(CProxy_Tester::ckNew());
    tester.run();
  }

  void done(int nFail) {
    CkEnforce(nFail == 0);
    CkPrintf("HAPI_VERIFY PASS: %d PEs x (%d pool buffers, %d async kernels)\n",
             CkNumPes(), nBufs, nKernels);
    CkExit();
  }
};

class Tester : public CBase_Tester {
  int completions;
  int elemsPerBuf;
  std::vector<int*> devBufs;      // device buffers, one per in-flight kernel
  std::vector<int*> hostBufs;     // pinned pool buffers receiving the copies
  hapiStream_t stream;

 public:
  Tester() : completions(0), elemsPerBuf(1024) {}

  void run() {
    if (wdSecs > 0 && CkMyPe() == 0)
      CcdCallFnAfter(watchdogAbort, nullptr, wdSecs * 1000.0);

    // --- 1. device mapping sanity -------------------------------------
    int devCount = 0;
    hapiCheck(cudaGetDeviceCount(&devCount));
    CkEnforce(devCount >= 1);
    uint64_t myDev = hapiMyDevice();
    CkEnforce((int)myDev < devCount);
    int current = -1;
    hapiCheck(cudaGetDevice(&current));
    CkEnforce(current == (int)myDev);

    // --- 2. allocator torture (CPU-verifiable, synchronous) -----------
    // Interleaved alloc/free of varied sizes; every buffer carries a
    // deterministic pattern verified before its free.
    {
      std::vector<char*> bufs(nBufs, nullptr);
      std::vector<size_t> sizes(nBufs);
      for (int i = 0; i < nBufs; i++) {
        sizes[i] = ((i * 2654435761u) % (64 * 1024)) + 8;
        hapiCheck(hapiPoolMalloc((void**)&bufs[i], sizes[i]));
        CkEnforce(bufs[i] != nullptr);
        memset(bufs[i], (i + CkMyPe()) & 0xFF, sizes[i]);
      }
      // free the odd-indexed half, then reallocate it at different sizes
      for (int i = 1; i < nBufs; i += 2) {
        verifyPattern(bufs[i], sizes[i], (i + CkMyPe()) & 0xFF);
        hapiCheck(hapiPoolFree(bufs[i]));
        bufs[i] = nullptr;
      }
      for (int i = 1; i < nBufs; i += 2) {
        sizes[i] = ((i * 40503u) % (16 * 1024)) + 8;
        hapiCheck(hapiPoolMalloc((void**)&bufs[i], sizes[i]));
        CkEnforce(bufs[i] != nullptr);
        memset(bufs[i], (i * 3 + CkMyPe()) & 0xFF, sizes[i]);
      }
      // the even half must still hold its original pattern (no overlap)
      for (int i = 0; i < nBufs; i += 2)
        verifyPattern(bufs[i], sizes[i], (i + CkMyPe()) & 0xFF);
      for (int i = 1; i < nBufs; i += 2)
        verifyPattern(bufs[i], sizes[i], (i * 3 + CkMyPe()) & 0xFF);
      for (int i = 0; i < nBufs; i++) hapiCheck(hapiPoolFree(bufs[i]));
    }

    // --- 3. async kernels + pool copies + hapiAddCallback -------------
    // hapiGetStream draws from the pool hapiCreateStreams fills; calling it
    // first is required (Ritvik's test has an explicit phase for this --
    // found when this assertion fired on classic, defect #5 of this test).
    hapiCreateStreams();
    stream = hapiGetStream();
    CkEnforce(stream != nullptr);
    devBufs.resize(nKernels);
    hostBufs.resize(nKernels);
    const size_t bytes = elemsPerBuf * sizeof(int);
    for (int k = 0; k < nKernels; k++) {
      hapiCheck(cudaMalloc((void**)&devBufs[k], bytes));
      hapiCheck(hapiPoolMalloc((void**)&hostBufs[k], bytes));
      launchFill(devBufs[k], elemsPerBuf, kernelSeed(k), stream);
      hapiCheck(cudaMemcpyAsync(hostBufs[k], devBufs[k], bytes,
                                cudaMemcpyDeviceToHost, stream));
      hapiAddCallback(stream,
                      CkCallback(CkIndex_Tester::kernelDone(), thisProxy[CkMyPe()]));
    }
  }

  void kernelDone() {
    completions++;
    CkEnforce(completions <= nKernels);
    if (completions == nKernels) {
      int nFail = 0;
      for (int k = 0; k < nKernels; k++) {
        for (int i = 0; i < elemsPerBuf; i++)
          if (hostBufs[k][i] != kernelSeed(k) + i) nFail++;
        hapiCheck(cudaFree(devBufs[k]));
        hapiCheck(hapiPoolFree(hostBufs[k]));
      }
      CkPrintf("[PE %d] device %d: allocator + %d async completions ok, %d bad elements\n",
               CkMyPe(), (int)hapiMyDevice(), completions, nFail);
      contribute(sizeof(int), &nFail, CkReduction::sum_int,
                 CkCallback(CkReductionTarget(Main, done), mainProxy));
    }
  }

 private:
  int kernelSeed(int k) const { return (CkMyPe() << 16) + k * 101; }

  void verifyPattern(char* buf, size_t size, int byte) {
    for (size_t j = 0; j < size; j++)
      CkEnforce((unsigned char)buf[j] == (unsigned char)byte);
  }
};

#include "hapi_verify.def.h"
