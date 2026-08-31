// Acceptance test for the HAPI portable core -- GPU migration plan stage 9.1.
//
// Self-checking (bcastred house style: every check is a CkEnforce, success is
// one PASS line, any failure aborts). Nothing here is timing- or
// hardware-specific, so the same binary is the acceptance run on an NVIDIA and
// on an AMD machine; the point of stage 9.1 is that those two produce the same
// verdict from one source.
//
// What it covers, and why each piece is worth a check:
//
//   1. The portability header itself. This file is compiled by charmc -- a
//      plain host compiler, not nvcc/hipcc -- and calls hapiMalloc,
//      hapiMemcpyAsync, hapiStreamCreate and friends directly. That is the
//      configuration hapi_portable.h exists to support (and the one its
//      __HIP_PLATFORM_AMD__ fallback was added for): if the header stops being
//      self-sufficient for user code, this test stops compiling. Only the
//      kernel launch itself lives in hapitest.cu.
//
//   2. Per-PE device mapping. hapiMapping() assigns each PE a device and calls
//      hapiSetDevice; hapiMyDevice() reports the result. The checks assert
//      that the reported device is the one actually current for the thread,
//      that it is in range, that the node half of the id is this PE's physical
//      node, that repeated calls agree, and -- under the default round-robin
//      mapping -- that the device index matches the runtime's own formula.
//      A cross-PE pass then asserts the mapping actually spreads: the classic
//      regression here is every PE silently landing on device 0, which every
//      per-PE check above would happily accept.
//
//   3. The pinned-host allocation API, in all three of its spellings:
//      hapiMallocHost(ptr, size), the pooled overload
//      hapiMallocHost(ptr, size, pool), and the hapiMallocHost_Pool alias
//      (and their hapiFreeHost counterparts). The pooled overloads are what
//      external users -- ChaNGa's allocatePinnedHostMemory and
//      freePinnedHostMemory -- actually call, and they can only exist while
//      hapi_portable.h keeps those names overloadable, so a regression there
//      breaks this file at the preprocessor.
//
//   4. The buddy allocator. Exercised directly (it is a plain class in
//      libhybridapi, and buddy_allocator.h is installed) rather than through
//      the D2D path, which does not arrive until stage 9.2. The checks pin
//      down the properties the D2D path will depend on: power-of-two rounding,
//      non-overlapping blocks, and -- the property that gives the allocator
//      its name -- that freeing everything coalesces back to a single whole
//      region, in any free order.
//
//   5. N asynchronous callbacks. Each worker issues nKernels independent
//      H2D/kernel/D2H chains and hangs a hapiAddCallback off each. Completions
//      are counted per element and summed; the result of every chain is
//      verified against a closed form, out of a buffer pre-filled with a
//      sentinel, so a callback that fires before its copy lands is a failure
//      rather than a silent pass. Finally quiescence is started while the GPU
//      work is still outstanding: hapiAddCallback brackets each chain with
//      QdCreate/QdProcess, so QD firing before the last callback is caught.
//
// Flags:  -c <chares, default 2*PEs>   -k <chains per chare, default 8>
//         -n <doubles per chain, default 1024>   -v (verbose)
//         -M (skip the exact device-mapping formula -- use this when running
//             with +gpumap block or +gpumap none, which the formula does not
//             model)
//
// Run (reconverse, single Frontier node):
//   srun -N1 -n1 -c56 --gpus-per-node=8 --network=single_node_vni \
//        ./hapitest +pe 8

#include "hapitest.decl.h"
#include "hapi.h"
#include "buddy_allocator.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_PeCheck peCheckProxy;
/*readonly*/ CProxy_Worker workerProxy;
/*readonly*/ int nChares;
/*readonly*/ int nKernels;
/*readonly*/ int nSlotElems;
/*readonly*/ int verbose;
/*readonly*/ int checkMapFormula;

extern void hapitestLaunch(hapiStream_t stream, const double* in, double* out,
                           int n, double addend);

// Size of the process-wide stream pool, published by the rank that created it.
// A plain static is process-scoped, which is exactly the scope of the pool.
static int g_streamPoolSize = 0;

// Per-PE report gathered by Main: see PeCheck::check().
enum { R_PE = 0, R_NODE, R_DEV, R_VISIBLE, R_LOGICAL_NODE, R_CHECKS, R_WIDTH };

static size_t roundPow2(size_t v) {
  size_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

// The value chain j of chare c writes into slot element i.
//
// Every quantity here is an exact binary fraction and every product and sum
// stays well inside 2^53, so the kernel's in*2+addend and the host's are
// bit-identical whether or not the compiler contracts them into an FMA. The
// result check below can therefore be an exact comparison. (The Makefile also
// turns contraction off, so the property does not silently depend on defaults.)
static inline double slotIn(int chare, int chain, int i) {
  return (double)i + 0.5 * (double)chain + 0.25 * (double)chare;
}
static inline double slotAddend(int chare, int chain) {
  return 1000.0 * (double)chare + (double)chain;
}

// ---------------------------------------------------------------------------
// 4. Buddy allocator
// ---------------------------------------------------------------------------
static void runBuddyChecks(int& checks) {
  const size_t commSize = (size_t)1 << 20;  // 1 MB communication region
  const size_t lbSize = (size_t)1 << 19;    // 512 KB load-balancing region

  buddy::allocator a(commSize + lbSize, commSize);
  uint8_t* base = a.base_ptr;
  CkEnforce(base != NULL);
  CkEnforce(a.get_free_size() == commSize);
  CkEnforce(a.get_lb_free_size() == lbSize);
  checks += 3;

  // Round-tripping one block at a time. The allocator rounds a request up to a
  // power of two (floored at its 4-byte minimum), and a lone free must return
  // the region to pristine.
  const size_t reqs[] = {1, 4, 5, 1000, 4096, 65536};
  for (size_t r : reqs) {
    const size_t want = roundPow2(r < 4 ? 4 : r);
    void* p = a.malloc(r, true);
    CkEnforce(p != NULL);
    CkEnforce((uint8_t*)p >= base && (uint8_t*)p + want <= base + commSize);
    CkEnforce(a.get_free_size() == commSize - want);
    a.free(p);
    CkEnforce(a.get_free_size() == commSize);
    checks += 4;
  }

  // Many live blocks at once: they must be distinct, non-overlapping and
  // in-region, and freeing all of them must coalesce back to one whole region
  // no matter what order the frees arrive in. Order 0 is allocation order,
  // 1 is reverse, 2 is a deterministic scramble.
  const int kBlocks = 64;
  const size_t blk = 4096;
  for (int order = 0; order < 3; order++) {
    std::vector<uint8_t*> ps;
    for (int i = 0; i < kBlocks; i++) {
      void* p = a.malloc(blk, true);
      CkEnforce(p != NULL);
      ps.push_back((uint8_t*)p);
    }
    CkEnforce(a.get_free_size() == commSize - (size_t)kBlocks * blk);
    checks += kBlocks + 1;

    std::vector<uint8_t*> sorted(ps);
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < kBlocks; i++) {
      CkEnforce(sorted[i] >= base && sorted[i] + blk <= base + commSize);
      if (i > 0) CkEnforce(sorted[i - 1] + blk <= sorted[i]);
    }
    checks++;

    if (order == 1) std::reverse(ps.begin(), ps.end());
    if (order == 2) {
      for (int i = 0; i + 1 < kBlocks; i += 2)
        std::swap(ps[i], ps[(i * 7 + 3) % kBlocks]);
    }
    for (uint8_t* p : ps) a.free(p);
    CkEnforce(a.get_free_size() == commSize);
    checks++;
  }

  // The whole region as one block, then given back.
  void* whole = a.malloc(commSize, true);
  CkEnforce(whole != NULL);
  CkEnforce(a.get_free_size() == 0);
  a.free(whole);
  CkEnforce(a.get_free_size() == commSize);
  checks += 3;

  // A request larger than the region is refused rather than served.
  CkEnforce(a.malloc(commSize + 1, true) == NULL);
  checks++;
  //
  // Note: there is deliberately no "allocate until exhausted, then ask for one
  // more" case here. That path runs allocator::malloc's bucket scan off the end
  // of the bucket array (`buckets[bucket].empty() && bucket < bucket_count`
  // tests emptiness before the bound), which is a pre-existing read past the
  // end -- it predates stage 9.1 and is not this test's to assert on.

  // The load-balancing sub-region is a separate bump allocator above the
  // communication region, with its own accounting.
  void* l1 = a.malloc(1024, false);
  void* l2 = a.malloc(2048, false);
  void* l3 = a.malloc(512, false);
  CkEnforce(l1 != NULL && l2 != NULL && l3 != NULL);
  for (void* p : {l1, l2, l3}) {
    CkEnforce((uint8_t*)p >= base + commSize);
    CkEnforce((uint8_t*)p < base + commSize + lbSize);
  }
  CkEnforce(a.get_lb_free_size() == lbSize - (1024 + 2048 + 512));
  checks += 3;

  // Free out of order -- middle, first, last -- so both the merge-with-neighbour
  // and the give-back-to-the-bump-pointer paths run.
  a.free(l2);
  a.free(l1);
  a.free(l3);
  CkEnforce(a.get_lb_free_size() == lbSize);
  CkEnforce(a.malloc(lbSize + 1, false) == NULL);
  checks += 2;

  // No LB region at all -- which is the runtime's own default, not an exotic
  // case: GPUManager::lb_buffer_size stays 0 unless +gpulbbuffer is passed, so
  // create_comm_buffer() asks for total == comm. free() routes to the LB path
  // by comparing against lb_base_ptr unconditionally, so that member has to be
  // meaningful here too.
  {
    buddy::allocator b(commSize, commSize);
    CkEnforce(b.get_free_size() == commSize);
    CkEnforce(b.get_lb_free_size() == 0);
    void* p1 = b.malloc(4096, true);
    void* p2 = b.malloc(8192, true);
    CkEnforce(p1 != NULL && p2 != NULL);
    b.free(p1);
    b.free(p2);
    CkEnforce(b.get_free_size() == commSize);
    checks += 5;
  }
}

// ---------------------------------------------------------------------------
// Per-PE checks: device mapping (2) plus the runtime stream pool, then (3).
// ---------------------------------------------------------------------------
class PeCheck : public CBase_PeCheck {
public:
  PeCheck() {}

  // hapiCreateStreams() appends to a pool held in the process-wide GPUManager,
  // so exactly one PE per process creates it; the reduction that follows is
  // what lets every other PE then draw from it.
  void createStreams() {
    if (CmiMyRank() == 0) {
      int n = hapiCreateStreams();
      CkEnforce(n > 0);
      g_streamPoolSize = n;
    }
    contribute(CkCallback(CkReductionTarget(Main, streamsReady), mainProxy));
  }

  void check() {
    int checks = 0;

    // ---- 2. device mapping, per PE ----
    int visible = -1;
    hapiCheck(hapiGetDeviceCount(&visible));
    CkEnforce(visible > 0);

    const uint64_t mine = hapiMyDevice();
    const int dev = (int)(mine & 0xffffffffu);
    const int node = (int)(mine >> 32);

    CkEnforce(dev >= 0 && dev < visible);
    CkEnforce(node == CmiPhysicalNodeID(CkMyPe()));

    // The device HAPI says this PE owns must be the device the thread is
    // actually set to -- this is what catches my_device drifting away from the
    // hapiSetDevice that hapiMapping performed.
    int current = -1;
    hapiCheck(hapiGetDevice(&current));
    CkEnforce(current == dev);

    // The mapping is fixed at startup, so it must not move under us.
    CkEnforce(hapiMyDevice() == mine);
    checks += 4;

    if (checkMapFormula) {
      // Reproduce hapiMapping()'s round-robin arithmetic exactly. Anything that
      // changes the device_count derivation or the local-index formula without
      // meaning to lands here.
      const int procsPerPnode = CmiNumNodes() / CmiNumPhysicalNodes();
      CkEnforce(procsPerPnode >= 1);
      const int nodeRankLocal = CmiNodeOf(CkMyPe()) % procsPerPnode;
      const int nodeSize = CmiNodeSize(CmiMyNode());
      int dc = visible / procsPerPnode;
      if (dc > nodeSize) dc = nodeSize;
      if (dc == 0) dc = 1;
      const int expect = (dc * nodeRankLocal + (CmiMyRank() % dc)) % visible;
      CkEnforce(dev == expect);
      checks++;
    }

    // ---- runtime stream pool ----
    // The pool lives in the process-wide GPUManager, so a PE that did not
    // create it must still be served from it.
    hapiStream_t s1 = hapiGetStream();
    hapiStream_t s2 = hapiGetStream();
    CkEnforce(s1 != NULL);
    CkEnforce(s2 != NULL);
    if (g_streamPoolSize > 1) CkEnforce(s1 != s2);  // handed out round-robin
    checks += 3;

    // ---- 3. pinned host memory: every spelling of the public API ----
    // hapiMallocHost/hapiFreeHost carry pooled overloads that external HAPI
    // users call directly -- ChaNGa's allocatePinnedHostMemory and
    // freePinnedHostMemory are hapiMallocHost(ptr, size, pool) and
    // hapiFreeHost(ptr, pool). Those overloads can only exist if the portable
    // header leaves the names overloadable, so this block is as much a
    // compile-time assertion as a runtime one: if hapiMallocHost goes back to
    // being a function-like macro, the 3-argument call below does not even
    // preprocess ("macro passed 3 arguments, but takes just 2").
    {
      const size_t nbytes = 4096;
      void* h = NULL;

      hapiCheck(hapiMallocHost(&h, nbytes));              // plain, 2 arguments
      CkEnforce(h != NULL);
      memset(h, 0xA5, nbytes);                            // must be real host memory
      CkEnforce(*((unsigned char*)h) == 0xA5);
      hapiCheck(hapiFreeHost(h));

      h = NULL;
      hapiCheck(hapiMallocHost(&h, nbytes, false));       // pooled overload, unpooled
      CkEnforce(h != NULL);
      hapiCheck(hapiFreeHost(h, false));

      h = NULL;
      hapiCheck(hapiMallocHost(&h, nbytes, true));        // pooled overload, pooled
      CkEnforce(h != NULL);
      memset(h, 0x5A, nbytes);
      CkEnforce(*((unsigned char*)h + nbytes - 1) == 0x5A);
      hapiCheck(hapiFreeHost(h, true));

      h = NULL;
      hapiCheck(hapiMallocHost_Pool(&h, nbytes, true));   // explicit-name alias
      CkEnforce(h != NULL);
      hapiCheck(hapiFreeHost_Pool(h, true));

      checks += 7;
    }

    // ---- 4. buddy allocator ----
    runBuddyChecks(checks);

    if (verbose)
      CkPrintf("[hapitest] PE %d: node %d device %d of %d, %d checks\n",
               CkMyPe(), node, dev, visible, checks);

    int report[R_WIDTH];
    report[R_PE] = CkMyPe();
    report[R_NODE] = node;
    report[R_DEV] = dev;
    report[R_VISIBLE] = visible;
    report[R_LOGICAL_NODE] = CmiNodeOf(CkMyPe());
    report[R_CHECKS] = checks;
    contribute(sizeof(report), report, CkReduction::concat,
               CkCallback(CkIndex_Main::peReport(NULL), mainProxy));
  }
};

// ---------------------------------------------------------------------------
// 5. N asynchronous callbacks
// ---------------------------------------------------------------------------
class Worker : public CBase_Worker {
  hapiStream_t stream;
  double* hostIn;
  double* hostOut;
  double* devIn;
  double* devOut;
  int completed;
  int homePe;
  size_t bytes;

public:
  Worker() : completed(0), homePe(CkMyPe()) {
    bytes = (size_t)nKernels * (size_t)nSlotElems * sizeof(double);
    hapiCheck(hapiStreamCreate(&stream));
    // Pinned host memory out of HAPI's pool -- the pool is the allocator under
    // test here; the transfers below are what make it matter.
    hapiCheck(hapiPoolMalloc((void**)&hostIn, bytes));
    hapiCheck(hapiPoolMalloc((void**)&hostOut, bytes));
    CkEnforce(hostIn != NULL && hostOut != NULL);
    hapiCheck(hapiMalloc((void**)&devIn, bytes));
    hapiCheck(hapiMalloc((void**)&devOut, bytes));
  }

  ~Worker() {
    hapiCheck(hapiFree(devIn));
    hapiCheck(hapiFree(devOut));
    hapiCheck(hapiPoolFree(hostIn));
    hapiCheck(hapiPoolFree(hostOut));
    hapiCheck(hapiStreamDestroy(stream));
  }

  void run() {
    for (int j = 0; j < nKernels; j++)
      for (int i = 0; i < nSlotElems; i++)
        hostIn[j * nSlotElems + i] = slotIn(thisIndex, j, i);

    // Sentinel: if a callback is delivered before its device-to-host copy has
    // landed, the check in kernelDone() sees this and fails, instead of
    // happening to read a plausible value.
    for (int k = 0; k < nKernels * nSlotElems; k++) hostOut[k] = -1.0;

    // All nKernels chains are issued back to back, so they are outstanding
    // together; being on one stream makes their completion order deterministic,
    // which is what lets kernelDone() know which chain it is finishing.
    for (int j = 0; j < nKernels; j++) {
      const size_t off = (size_t)j * (size_t)nSlotElems;
      const size_t sz = (size_t)nSlotElems * sizeof(double);
      hapiCheck(hapiMemcpyAsync(devIn + off, hostIn + off, sz,
                                hapiMemcpyHostToDevice, stream));
      hapitestLaunch(stream, devIn + off, devOut + off, nSlotElems,
                     slotAddend(thisIndex, j));
      hapiCheck(hapiMemcpyAsync(hostOut + off, devOut + off, sz,
                                hapiMemcpyDeviceToHost, stream));
      hapiAddCallback(stream,
                      CkCallback(CkIndex_Worker::kernelDone(),
                                 CkArrayIndex1D(thisIndex), thisArrayID));
    }
  }

  void kernelDone() {
    // Chains complete in issue order on a single stream, so this is chain
    // number `completed`. Enforcing that ordering is itself part of the test.
    CkEnforce(completed < nKernels);
    CkEnforce(CkMyPe() == homePe);  // callback runs where it was registered

    const int j = completed;
    const size_t off = (size_t)j * (size_t)nSlotElems;
    const double addend = slotAddend(thisIndex, j);
    for (int i = 0; i < nSlotElems; i++) {
      const double want = slotIn(thisIndex, j, i) * 2.0 + addend;
      CkEnforce(hostOut[off + i] == want);
    }
    completed++;

    if (completed == nKernels) {
      long long mine = completed;
      contribute(sizeof(long long), &mine, CkReduction::sum_long_long,
                 CkCallback(CkReductionTarget(Main, workDone), mainProxy));
    }
  }
};

// ---------------------------------------------------------------------------
class Main : public CBase_Main {
  double startTime;
  long long totalCompletions;
  long long totalChecks;
  bool workReported;

public:
  Main(CkArgMsg* m) : totalCompletions(0), totalChecks(0), workReported(false) {
    mainProxy = thisProxy;
    nChares = 2 * CkNumPes();
    nKernels = 8;
    nSlotElems = 1024;
    verbose = 0;
    checkMapFormula = 1;

    int c;
    while ((c = getopt(m->argc, m->argv, "c:k:n:vM")) != -1) {
      switch (c) {
        case 'c': nChares = atoi(optarg); break;
        case 'k': nKernels = atoi(optarg); break;
        case 'n': nSlotElems = atoi(optarg); break;
        case 'v': verbose = 1; break;
        case 'M': checkMapFormula = 0; break;
        default:
          CkPrintf("Usage: %s [-c chares] [-k chains] [-n elems] [-v] [-M]\n",
                   m->argv[0]);
          CkExit(1);
      }
    }
    delete m;

    CkEnforce(nChares >= 1);
    CkEnforce(nKernels >= 1);
    CkEnforce(nSlotElems >= 1);

    CkPrintf("[hapitest] PEs %d, logical nodes %d, physical nodes %d\n",
             CkNumPes(), CmiNumNodes(), CmiNumPhysicalNodes());
    CkPrintf("[hapitest] chares %d, chains/chare %d, doubles/chain %d\n",
             nChares, nKernels, nSlotElems);

    startTime = CkWallTimer();
    peCheckProxy = CProxy_PeCheck::ckNew();
    peCheckProxy.createStreams();
  }

  void streamsReady() { peCheckProxy.check(); }

  void peReport(CkReductionMsg* msg) {
    const int* d = (const int*)msg->getData();
    const int n = msg->getSize() / (int)sizeof(int);
    CkEnforce(n == R_WIDTH * CkNumPes());

    // Cross-PE mapping properties. The per-PE checks all pass if every PE is
    // pinned to device 0; these are the ones that do not.
    std::vector<int> pes(CkNumPes(), -1);
    for (int r = 0; r < CkNumPes(); r++) {
      const int* e = d + (size_t)r * R_WIDTH;
      const int pe = e[R_PE];
      CkEnforce(pe >= 0 && pe < CkNumPes());
      CkEnforce(pes[pe] == -1);  // exactly one report per PE
      pes[pe] = r;
      totalChecks += e[R_CHECKS];
    }

    // Group by physical node, then by process within it.
    std::vector<int> nodes;
    for (int r = 0; r < CkNumPes(); r++) nodes.push_back(d[(size_t)r * R_WIDTH + R_NODE]);
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

    for (int node : nodes) {
      int visible = -1, pesHere = 0;
      std::vector<int> devs;
      for (int r = 0; r < CkNumPes(); r++) {
        const int* e = d + (size_t)r * R_WIDTH;
        if (e[R_NODE] != node) continue;
        pesHere++;
        // Every PE on a physical node must see the same devices.
        if (visible < 0) visible = e[R_VISIBLE];
        CkEnforce(e[R_VISIBLE] == visible);
        CkEnforce(e[R_DEV] >= 0 && e[R_DEV] < visible);
        devs.push_back(e[R_DEV]);
      }
      std::vector<int> uniq(devs);
      std::sort(uniq.begin(), uniq.end());
      uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

      // The mapping must spread. With V devices and P PEs on the node it
      // should reach min(V, P) of them; falling short means PEs are piling
      // onto one device.
      const int expectDistinct = std::min(visible, pesHere);
      CkEnforce((int)uniq.size() == expectDistinct);

      // ...and spread evenly: per-device PE counts differ by at most one.
      int lo = pesHere, hi = 0;
      for (int dv : uniq) {
        int cnt = (int)std::count(devs.begin(), devs.end(), dv);
        lo = std::min(lo, cnt);
        hi = std::max(hi, cnt);
      }
      CkEnforce(hi - lo <= 1);

      if (verbose)
        CkPrintf("[hapitest] physical node %d: %d PEs over %d of %d devices\n",
                 node, pesHere, (int)uniq.size(), visible);
    }
    delete msg;

    workerProxy = CProxy_Worker::ckNew(nChares);
    workerProxy.run();
    // Started while the GPU work is still in flight on purpose: hapiAddCallback
    // brackets each chain with QdCreate/QdProcess, so quiescence must not be
    // reached until the last callback has been delivered.
    CkStartQD(CkCallback(CkIndex_Main::quiesced(), mainProxy));
  }

  void workDone(long long completions) {
    CkEnforce(completions == (long long)nChares * (long long)nKernels);
    totalCompletions = completions;
    workReported = true;
  }

  void quiesced() {
    CkEnforce(workReported);
    CkEnforce(totalCompletions == (long long)nChares * (long long)nKernels);
    CkPrintf("hapitest PASS: %d PEs, %lld per-PE checks, %lld GPU callbacks, "
             "%.3f s\n",
             CkNumPes(), totalChecks, totalCompletions,
             CkWallTimer() - startTime);
    CkExit(0);
  }
};

#include "hapitest.def.h"
