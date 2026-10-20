// chantest: two array elements exchange a device buffer every iteration through
// persistent channels (CkDeviceChannel), each verifying what it received, and
// with -l they swap PEs every lb_period iterations (each element calls
// migrateMe, which takes the runtime's migration path: stand-down, pack, unpack)
// WITHOUT re-exchanging the channels: the receiver reopens its channel on the
// new buffer in its pup, and the sender's next put re-resolves. This is the migration test of PERSISTENT_COMM_PLAN.md
// step 3. Run with 2 PEs; device buffers come from the pool under +gpupool.
#include "chantest.decl.h"
#include "hapi.h"

#define ERROR_TOLERANCE 1e-6

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Elem elem_proxy;
/* readonly */ int block_size;
/* readonly */ int n_iters;
/* readonly */ int lb_period;

extern void invokeFillKernel(double*, int, double, cudaStream_t);

struct Container {
  double* h_remote_data;
  double* d_local_data;
  double* d_remote_data;
  cudaStream_t stream;

  Container() : h_remote_data(nullptr), d_local_data(nullptr), d_remote_data(nullptr), stream(nullptr) {}

  ~Container() {
    if (h_remote_data) hapiCheck(cudaFreeHost(h_remote_data));
    if (d_local_data) {
      if (CkDevicePoolOn()) { CkDeviceFree(d_local_data); CkDeviceFree(d_remote_data); }
      else { hapiCheck(cudaFree(d_local_data)); hapiCheck(cudaFree(d_remote_data)); }
    }
    if (stream) hapiCheck(cudaStreamDestroy(stream));
  }

  void init() {
    hapiCheck(cudaMallocHost(&h_remote_data, sizeof(double) * block_size));
    if (CkDevicePoolOn()) {
      d_local_data = (double*)CkDeviceMalloc(sizeof(double) * block_size);
      d_remote_data = (double*)CkDeviceMalloc(sizeof(double) * block_size);
    } else {
      hapiCheck(cudaMalloc(&d_local_data, sizeof(double) * block_size));
      hapiCheck(cudaMalloc(&d_remote_data, sizeof(double) * block_size));
    }
    hapiCheck(cudaStreamCreate(&stream));
  }

  // The receive buffer is application state and migrates with the element: a
  // put whose bytes landed before the pack is in it at the new home.
  void pup(PUP::er& p) {
    if (p.isUnpacking()) init();
    if (p.isPacking()) {
      hapiCheck(cudaMemcpyAsync(h_remote_data, d_remote_data, sizeof(double) * block_size,
                                cudaMemcpyDeviceToHost, stream));
      hapiCheck(cudaStreamSynchronize(stream));
    }
    PUParray(p, h_remote_data, block_size);
    if (p.isUnpacking()) {
      hapiCheck(cudaMemcpyAsync(d_remote_data, h_remote_data, sizeof(double) * block_size,
                                cudaMemcpyHostToDevice, stream));
      hapiCheck(cudaStreamSynchronize(stream));
    }
  }

  void fill(double val) { invokeFillKernel(d_local_data, block_size, val, stream); }

  bool verify(double val, int& bad_index, double& got) {
    hapiCheck(cudaMemcpyAsync(h_remote_data, d_remote_data, sizeof(double) * block_size,
                              cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaStreamSynchronize(stream));
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
    int c;
    while ((c = getopt(m->argc, m->argv, "s:i:lp:")) != -1) {
      switch (c) {
        case 's': block_size = atoi(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'l': lb_test = true; break;
        case 'p': lb_period = atoi(optarg); break;
        default: CkAbort("usage: chantest [-s block] [-i iters] [-l] [-p lb_period]");
      }
    }
    delete m;
    if (CkNumPes() != 2) CkAbort("chantest: run with 2 PEs");
    CkPrintf("[chantest] block %d doubles, %d iterations, LB %s (period %d), %d process(es), %d physical node(s)\n",
             block_size, n_iters, lb_test ? "on" : "off", lb_period, CkNumNodes(), CmiNumPhysicalNodes());
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
  void doLB() { migrateMe(1 - CkMyPe()); }

  void ckJustMigrated() {
    ArrayElement::ckJustMigrated();
    int migrated = (home_pe != CkMyPe()) ? 1 : 0;
    home_pe = CkMyPe();
    contribute(sizeof(int), &migrated, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, lbDone), main_proxy));
  }
};

#include "chantest.def.h"
