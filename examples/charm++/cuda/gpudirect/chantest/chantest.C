// chantest: two array elements exchange a device buffer every iteration through
// persistent channels (CkDeviceChannel), each verifying what it received, and
// with -l they swap PEs every lb_period iterations (each element calls
// migrateMe, which takes the runtime's migration path: stand-down, pack, unpack)
// WITHOUT re-exchanging the channels: the receiver reopens its channel on the
// new buffer in its pup, and the sender's next put re-resolves. This is the migration test of PERSISTENT_COMM_PLAN.md
// step 3. Run with 2 PEs; device buffers come from the pool under +gpupool.
//
// -a moves through the balancer's path instead (CkLocRec::recvMigrate, as a
// load balancer's decision does): under +LBAsync the move is deferred to a
// quiet moment and the element's Stable device buffer (PUP::DeviceLife::Stable,
// a ramp written once at construction) is shipped at the decision while a
// Mutable one (the iteration number) goes at the pack; both are verified after
// every arrival. -x rewrites the Stable buffer after the decision: the runtime
// must abort naming it.
#include "chantest.decl.h"
#include "hapi.h"

#define ERROR_TOLERANCE 1e-6

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Elem elem_proxy;
/* readonly */ int block_size;
/* readonly */ int n_iters;
/* readonly */ int lb_period;
/* readonly */ int async_path;
/* readonly */ int corrupt_stable;

extern void invokeFillKernel(double*, int, double, hapiStream_t);
extern void invokeRampKernel(double*, int, double, hapiStream_t);
#define STABLE_BASE 7000.0

struct Container {
  double* h_remote_data;
  double* d_local_data;
  double* d_remote_data;
  // Two buffers that travel by pup_buffer_device (rebound into the landing
  // at the new home): a Stable ramp and the Mutable iteration number.
  double* d_stable;
  double* d_iter;
  hapiStream_t stream;

  Container() : h_remote_data(nullptr), d_local_data(nullptr), d_remote_data(nullptr),
                d_stable(nullptr), d_iter(nullptr), stream(nullptr) {}

  ~Container() {
    if (h_remote_data) hapiCheck(hapiFreeHost(h_remote_data));
    if (d_local_data) {
      if (CkDevicePoolOn()) { CkDeviceFree(d_local_data); CkDeviceFree(d_remote_data); }
      else { hapiCheck(hapiFreeRaw(d_local_data)); hapiCheck(hapiFreeRaw(d_remote_data)); }
    }
    // Either an allocation of ours or a pointer into the runtime's landing.
    if (d_stable) hapiFreeMigratable(d_stable);
    if (d_iter) hapiFreeMigratable(d_iter);
    if (stream) hapiCheck(hapiStreamDestroy(stream));
  }

  void init() {
    hapiCheck(hapiMallocHost(&h_remote_data, sizeof(double) * block_size));
    if (CkDevicePoolOn()) {
      d_local_data = (double*)CkDeviceMalloc(sizeof(double) * block_size);
      d_remote_data = (double*)CkDeviceMalloc(sizeof(double) * block_size);
    } else {
      hapiCheck(hapiMallocRaw(&d_local_data, sizeof(double) * block_size));
      hapiCheck(hapiMallocRaw(&d_remote_data, sizeof(double) * block_size));
    }
    hapiCheck(hapiStreamCreate(&stream));
  }

  // The two pup_buffer_device buffers: allocated at construction only; a
  // migration rebinds them.
  void initMigratable() {
    if (CkDevicePoolOn()) {
      d_stable = (double*)CkDeviceMalloc(sizeof(double) * block_size);
      d_iter = (double*)CkDeviceMalloc(sizeof(double) * block_size);
    } else {
      hapiCheck(hapiMallocRaw(&d_stable, sizeof(double) * block_size));
      hapiCheck(hapiMallocRaw(&d_iter, sizeof(double) * block_size));
    }
    invokeRampKernel(d_stable, block_size, STABLE_BASE, stream);
    invokeFillKernel(d_iter, block_size, -1.0, stream);
  }
  void setIter(int iter) { invokeFillKernel(d_iter, block_size, (double)iter, stream); }
  void corruptStable() { invokeFillKernel(d_stable, block_size, 0.0, stream); }

  // What the rebound buffers hold at the new home.
  bool verifyMigratable(int iter, const char*& what, int& bad_index, double& got) {
    hapiCheck(hapiMemcpyAsync(h_remote_data, d_stable, sizeof(double) * block_size,
                              hapiMemcpyDeviceToHost, stream));
    hapiCheck(hapiStreamSynchronize(stream));
    for (int i = 0; i < block_size; i++)
      if (fabs(h_remote_data[i] - (STABLE_BASE + i)) > ERROR_TOLERANCE) { what = "stable"; bad_index = i; got = h_remote_data[i]; return false; }
    hapiCheck(hapiMemcpyAsync(h_remote_data, d_iter, sizeof(double) * block_size,
                              hapiMemcpyDeviceToHost, stream));
    hapiCheck(hapiStreamSynchronize(stream));
    for (int i = 0; i < block_size; i++)
      if (fabs(h_remote_data[i] - (double)iter) > ERROR_TOLERANCE) { what = "iter"; bad_index = i; got = h_remote_data[i]; return false; }
    return true;
  }

  // The receive buffer is application state and migrates with the element: a
  // put whose bytes landed before the pack is in it at the new home.
  void pup(PUP::er& p) {
    if (p.isUnpacking()) init();
    if (p.isPacking()) {
      hapiCheck(hapiMemcpyAsync(h_remote_data, d_remote_data, sizeof(double) * block_size,
                                hapiMemcpyDeviceToHost, stream));
      hapiCheck(hapiStreamSynchronize(stream));
    }
    PUParray(p, h_remote_data, block_size);
    if (p.isUnpacking()) {
      hapiCheck(hapiMemcpyAsync(d_remote_data, h_remote_data, sizeof(double) * block_size,
                                hapiMemcpyHostToDevice, stream));
      hapiCheck(hapiStreamSynchronize(stream));
    }
    // The device copies (pack, or the decision-time snapshot) wait for this
    // stream's work on these buffers.
    p.pup_device_order((void*)stream);
    p.pup_buffer_device(d_stable, block_size, PUP::DeviceLife::Stable);
    p.pup_buffer_device(d_iter, block_size);
  }

  void fill(double val) { invokeFillKernel(d_local_data, block_size, val, stream); }

  bool verify(double val, int& bad_index, double& got) {
    hapiCheck(hapiMemcpyAsync(h_remote_data, d_remote_data, sizeof(double) * block_size,
                              hapiMemcpyDeviceToHost, stream));
    hapiCheck(hapiStreamSynchronize(stream));
    for (int i = 0; i < block_size; i++) {
      if (fabs(h_remote_data[i] - val) > ERROR_TOLERANCE) { bad_index = i; got = h_remote_data[i]; return false; }
    }
    return true;
  }
};

class Main : public CBase_Main {
  int iter;
  int migrations;
  double start_time;
  bool lb_test;

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    block_size = 4096;
    n_iters = 20;
    lb_period = 3;
    lb_test = false;
    async_path = 0;
    corrupt_stable = 0;
    int c;
    while ((c = getopt(m->argc, m->argv, "s:i:lp:ax")) != -1) {
      switch (c) {
        case 's': block_size = atoi(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'l': lb_test = true; break;
        case 'p': lb_period = atoi(optarg); break;
        case 'a': async_path = 1; break;
        case 'x': corrupt_stable = 1; break;
        default: CkAbort("usage: chantest [-s block] [-i iters] [-l] [-p lb_period] [-a] [-x]");
      }
    }
    delete m;
    if (CkNumPes() != 2) CkAbort("chantest: run with 2 PEs");
    if (corrupt_stable && !async_path) CkAbort("chantest: -x needs -a (and +LBAsync)");
    CkPrintf("[chantest] block %d doubles, %d iterations, LB %s (period %d, %s path%s), %d process(es), %d physical node(s)\n",
             block_size, n_iters, lb_test ? "on" : "off", lb_period,
             async_path ? "balancer" : "migrateMe", corrupt_stable ? ", stable corrupted after the decision" : "",
             CkNumNodes(), CmiNumPhysicalNodes());
    elem_proxy = CProxy_Elem::ckNew(2);
    iter = 0;
    migrations = 0;
    start_time = CkWallTimer();
    elem_proxy.setup();
  }

  void ready() { elem_proxy.step(iter); }

  void iterDone(int migrated) {
    migrations += migrated;
    iter++;
    if (iter >= n_iters) {
      CkPrintf("[chantest] PASS: %d iterations verified, %d element migrations, %.3f s\n",
               n_iters, migrations, CkWallTimer() - start_time);
      CkExit();
      return;
    }
    if (lb_test && (iter % lb_period) == 0) elem_proxy.doLB();
    else elem_proxy.step(iter);
  }

  void lbDone(int migrated) {
    migrations += migrated;
    elem_proxy.step(iter);
  }
};

class Elem : public CBase_Elem {
  Container container;
  CkDeviceChannel my_ch;     // my receive buffer, exported to the peer
  CkDeviceChannel peer_ch;   // the peer's receive buffer, written by my puts
  int peer;
  int cur_iter;
  int home_pe;   // the PE this element was on at the last step: counts migrations

public:
  Elem() {
    peer = 1 - thisIndex;
    cur_iter = -1;
    home_pe = CkMyPe();
    container.init();
    container.initMigratable();
  }

  Elem(CkMigrateMessage* m) {}

  void pup(PUP::er& p) {
    p|peer;
    p|cur_iter;
    p|home_pe;
    p|container;   // allocates the buffers on unpack
    p|my_ch;
    p|peer_ch;
    // The receiver's channel lives on at the new home, on the new buffer.
    if (p.isUnpacking()) my_ch.reopen(container.d_remote_data);
  }

  void setup() {
    my_ch = CkDeviceChannel(container.d_remote_data, sizeof(double) * block_size,
                            CkCallback(CkIndex_Elem::arrived(NULL), thisProxy[thisIndex]));
    my_ch.open();
    thisProxy[peer].takeChannel(my_ch);
  }

  void takeChannel(CkDeviceChannel ch) {
    peer_ch = ch;
    contribute(CkCallback(CkReductionTarget(Main, ready), main_proxy));
  }

  static double valueOf(int iter, int from) { return 1000.0 * (from + 1) + iter; }

  void step(int iter) {
    cur_iter = iter;
    container.fill(valueOf(iter, thisIndex));
    container.setIter(iter);
    // The iteration travels as the put's tag: the peer's arrival can run before
    // its own step(iter) message, so its counter is not what to verify against.
    peer_ch.put(container.d_local_data, sizeof(double) * block_size, container.stream,
                CkCallback(CkCallback::ignore), (uint64_t)iter);
  }

  void arrived(CkDataMsg* m) {
    const CkDeviceChannelEvent* e = my_ch.event(m);
    delete m;
    if (e == NULL) return;   // a resolve, answered by the runtime
    int bad = -1; double got = 0.0;
    const int iter = (int)e->tag;
    const double want = valueOf(iter, peer);
    if (!container.verify(want, bad, got)) {
      CkAbort("[chantest] element %d on PE %d: iteration %d put %llu (%zu bytes) wrong at index %d: expected %.1f, got %.1f\n",
              thisIndex, CkMyPe(), iter, (unsigned long long)e->seq, e->bytes, bad, want, got);
    }
    my_ch.release();
    int migrated = (home_pe != CkMyPe()) ? 1 : 0;
    home_pe = CkMyPe();
    contribute(sizeof(int), &migrated, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, iterDone), main_proxy));
  }

  // Swap with the peer's PE. Nothing after the call: the element may already be
  // packed when it returns, or it is deferred and goes once its puts are done.
  // -a: the balancer's path (deferred under +LBAsync, where the Stable buffer
  // is snapshotted at the decision); -x then breaks the Stable promise.
  void doLB() {
    if (!async_path) { migrateMe(1 - CkMyPe()); return; }
    getCkLocRec()->recvMigrate(1 - CkMyPe());
    if (corrupt_stable) {
      // After the decision's snapshot has been taken (its copies run on the
      // runtime's stream: let them finish so the rewrite is seen only by the
      // pack), before the pack.
      hapiCheck(hapiDeviceSynchronize());
      container.corruptStable();
    }
  }

  void ckJustMigrated() {
    ArrayElement::ckJustMigrated();
    {
      const char* what = ""; int bad = -1; double got = 0.0;
      if (!container.verifyMigratable(cur_iter, what, bad, got))
        CkAbort("[chantest] element %d on PE %d after migration: %s buffer wrong at index %d: got %.1f (iteration %d)\n",
                thisIndex, CkMyPe(), what, bad, got, cur_iter);
    }
    int migrated = (home_pe != CkMyPe()) ? 1 : 0;
    home_pe = CkMyPe();
    contribute(sizeof(int), &migrated, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, lbDone), main_proxy));
  }
};

#include "chantest.def.h"
