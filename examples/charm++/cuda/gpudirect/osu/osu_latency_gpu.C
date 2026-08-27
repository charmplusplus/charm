// Device-buffer ping-pong in the shape of the OSU latency benchmark.
//
// This exists to locate the size at which the direct CUDA IPC transport starts
// beating the staged one for intra-node, cross-process GPU messages. Run the
// sweep once per CHARM_GPU_IPC_THRESHOLD setting -- unset for all-staged, 0 for
// all-direct -- and the crossover is where the two curves meet. See
// scripts/ipc_crossover.sh, which drives exactly that.
//
// The stock examples/charm++/osu_latency measures host messages and never
// touches device memory, so it cannot see this path at all.
//
// Placement is the entire experiment. Both endpoints must sit in different
// processes on the same physical host: same process and every send resolves to
// a plain device-to-device copy, different hosts and it resolves to inter-node
// RDMA. Either way the run would report a number for a path nobody asked
// about, so the placement is checked at startup and the benchmark refuses to
// run rather than quietly measuring the wrong thing.
//
// No CkDeviceBuffer here carries a completion callback, and that is deliberate
// rather than an oversight: a ping-pong acknowledges itself. The pong cannot
// arrive before the peer has finished reading the ping, so by the time this
// side reuses its send buffer the previous read has provably completed. An
// application whose protocol does not already provide that ordering must attach
// the callback -- the direct transport hands the receiver the sender's live
// buffer and nothing else stands between the two.

#include "osu_latency_gpu.decl.h"
#include "hapi.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// CkAbort's message does not reach the terminal on every build -- the
// reconverse layer prints a backtrace and drops it -- and for these checks the
// explanation IS the output. Print it first, then abort.
static void refuse(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  CkPrintf("\nosu_latency_gpu: %s\n\n", buf);
  fflush(stdout);
  CkAbort("%s", buf);
}

/*readonly*/ CProxy_Main main_proxy;
/*readonly*/ CProxy_Endpoint endpoint_proxy;
/*readonly*/ int max_bytes;
/*readonly*/ bool validate;

// Sizes at or below this get the small-message iteration counts.
static const int SMALL_MESSAGE_LIMIT = 8192;

class PairMap : public CkArrayMap {
  int pes[2];

public:
  PairMap(int pe0, int pe1) {
    pes[0] = pe0;
    pes[1] = pe1;
  }
  PairMap(CkMigrateMessage* m) : CkArrayMap(m) {}

  int registerArray(const CkArrayIndex&, CkArrayID) { return 0; }
  int procNum(int, const CkArrayIndex& idx) { return pes[idx.data()[0] & 1]; }
};

class Main : public CBase_Main {
  int min_bytes, cur_bytes;
  int small_iters, small_skip, large_iters, large_skip;
  int peer_pe;

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;

    min_bytes = 8;
    max_bytes = 4 << 20;
    small_iters = 1000;
    small_skip = 100;
    large_iters = 100;
    large_skip = 10;
    peer_pe = -1;
    validate = false;

    for (int i = 1; i < m->argc; i++) {
      if (!strcmp(m->argv[i], "-s") && i + 1 < m->argc) {
        min_bytes = atoi(m->argv[++i]);
      } else if (!strcmp(m->argv[i], "-e") && i + 1 < m->argc) {
        max_bytes = atoi(m->argv[++i]);
      } else if (!strcmp(m->argv[i], "-i") && i + 1 < m->argc) {
        small_iters = large_iters = atoi(m->argv[++i]);
      } else if (!strcmp(m->argv[i], "-w") && i + 1 < m->argc) {
        small_skip = large_skip = atoi(m->argv[++i]);
      } else if (!strcmp(m->argv[i], "-p") && i + 1 < m->argc) {
        peer_pe = atoi(m->argv[++i]);
      } else if (!strcmp(m->argv[i], "-v")) {
        validate = true;
      } else if (m->argv[i][0] == '+') {
        // A runtime flag the runtime did not consume (a launcher can append
        // +pemap after the application's own arguments). Not ours to reject.
      } else {
        refuse("unknown argument '%s'. Usage: -s <min bytes> "
                "-e <max bytes> -i <iters> -w <warmup iters> -p <peer PE> [-v]",
                m->argv[i]);
      }
    }
    delete m;

    if (peer_pe < 0) {
      if (CkNumNodes() < 2) {
        refuse("this measures the cross-process, same-host "
                "device path, so it needs at least two processes. Launch two "
                "(one host, e.g. ++p 2 ++ppn 1) or name a peer PE with -p.");
      }
      peer_pe = CkNodeFirst(1);
    }
    if (peer_pe <= 0 || peer_pe >= CkNumPes()) {
      refuse("peer PE %d is out of range (this job has %d "
              "PEs), and it cannot be PE 0, which hosts the other endpoint.",
              peer_pe, CkNumPes());
    }
    // The two checks the whole measurement rests on.
    if (CmiNodeOf(0) == CmiNodeOf(peer_pe)) {
      refuse("peer PE %d is in the same process as PE 0, so "
              "every send would resolve to a same-process device-to-device "
              "copy -- not the cross-process path this benchmark exists to "
              "measure. Launch more processes, or pick a PE in another one.",
              peer_pe);
    }
    if (!CmiPeOnSamePhysicalNode(0, peer_pe)) {
      refuse("peer PE %d is on a different host from PE 0, "
              "so every send would resolve to inter-node RDMA rather than CUDA "
              "IPC. Run both processes on one host.",
              peer_pe);
    }

    CkArrayOptions opts(2);
    opts.setMap(CProxy_PairMap::ckNew(0, peer_pe));
    endpoint_proxy = CProxy_Endpoint::ckNew(opts);

    CkPrintf("# OSU-style GPU latency test (device buffers, cross-process)\n");
    CkPrintf("# PE 0 (process %d) <-> PE %d (process %d)\n",
             CmiNodeOf(0), peer_pe, CmiNodeOf(peer_pe));
    CkPrintf("# %-14s%s\n", "Size", "Latency (us)");
    fflush(stdout);

    cur_bytes = min_bytes;
    startNext();
  }

  // Both endpoints have allocated and are ready to be timed. Waiting for this
  // rather than letting PE 0 start on its own keeps the first message of each
  // size from racing the peer's buffer setup.
  void allReady() { endpoint_proxy[0].launch(); }

  void doneOne(double latency_us) {
    CkPrintf("%-16d%.2f\n", cur_bytes, latency_us);
    fflush(stdout);
    cur_bytes *= 2;
    startNext();
  }

  void startNext() {
    if (cur_bytes > max_bytes) {
      CkExit();
      return;
    }
    const bool small = (cur_bytes <= SMALL_MESSAGE_LIMIT);
    endpoint_proxy.start(cur_bytes, small ? small_iters : large_iters,
                         small ? small_skip : large_skip);
  }
};

class Endpoint : public CBase_Endpoint {
  char* d_send;
  char* d_recv;
  char* h_buf;
  hapiStream_t stream;

  int cur_bytes, n_iters, n_skip, iter;
  int verified_bytes;  // last size validated, so ordering does not matter
  double t_start;

public:
  Endpoint() {
    hapiCheck(hapiStreamCreate(&stream));
    hapiCheck(hapiMalloc(&d_send, (size_t)max_bytes));
    hapiCheck(hapiMalloc(&d_recv, (size_t)max_bytes));
    hapiCheck(hapiMallocHost(&h_buf, (size_t)max_bytes));

    // A pattern keyed on both the offset and the sending element, so a
    // validation failure distinguishes "wrong bytes" from "right bytes, wrong
    // base" -- which is the mistake an offset-carrying transport can make.
    fillPattern(h_buf, max_bytes, thisIndex);
    hapiCheck(hapiMemcpyAsync(d_send, h_buf, (size_t)max_bytes,
                              hapiMemcpyHostToDevice, stream));
    hapiCheck(hapiStreamSynchronize(stream));

    cur_bytes = 0;
    n_iters = n_skip = iter = 0;
    verified_bytes = -1;
    t_start = 0.0;
  }

  ~Endpoint() {
    hapiCheck(hapiFree(d_send));
    hapiCheck(hapiFree(d_recv));
    hapiCheck(hapiFreeHost(h_buf));
    hapiCheck(hapiStreamDestroy(stream));
  }

  static void fillPattern(char* buf, int n, int who) {
    for (int i = 0; i < n; i++) buf[i] = (char)((i + who * 7) & 0x7f);
  }

  void start(int bytes, int iters, int skip) {
    cur_bytes = bytes;
    n_iters = iters;
    n_skip = skip;
    iter = 0;
    contribute(CkCallback(CkReductionTarget(Main, allReady), main_proxy));
  }

  void launch() {
    // Start the clock once the warm-up iterations are behind us, so first-touch
    // costs -- an IPC handle opened for the first time, a comm buffer block
    // faulted in -- do not land in the reported number.
    if (iter == n_skip) t_start = CkWallTimer();
    thisProxy[1].ping(cur_bytes,
                      CkDeviceBuffer(d_send, (size_t)cur_bytes, stream));
  }

  void ping(int& n, char*& buf, CkDeviceBufferPost* post) {
    buf = d_recv;
    post[0].hapi_stream = stream;
  }

  void ping(int n, char* buf) {
    if (validate && n != verified_bytes) {
      verified_bytes = n;
      verify(n, 0);
    }
    thisProxy[0].pong(n, CkDeviceBuffer(d_send, (size_t)n, stream));
  }

  void pong(int& n, char*& buf, CkDeviceBufferPost* post) {
    buf = d_recv;
    post[0].hapi_stream = stream;
  }

  void pong(int n, char* buf) {
    if (validate && n != verified_bytes) {
      verified_bytes = n;
      verify(n, 1);
    }
    if (++iter < n_iters + n_skip) {
      launch();
    } else {
      const double elapsed = CkWallTimer() - t_start;
      main_proxy.doneOne(elapsed * 1e6 / (2.0 * n_iters));
    }
  }

private:
  void verify(int n, int sender) {
    hapiCheck(hapiMemcpyAsync(h_buf, d_recv, (size_t)n, hapiMemcpyDeviceToHost,
                              stream));
    hapiCheck(hapiStreamSynchronize(stream));
    for (int i = 0; i < n; i++) {
      const char expected = (char)((i + sender * 7) & 0x7f);
      if (h_buf[i] != expected) {
        refuse("validation failed on PE %d at byte %d of a "
                "%d-byte transfer from element %d: expected %d, got %d.",
                CkMyPe(), i, n, sender, (int)expected, (int)h_buf[i]);
      }
    }
    // The send buffer was clobbered by the readback; put it back.
    fillPattern(h_buf, max_bytes, thisIndex);
    hapiCheck(hapiMemcpyAsync(d_send, h_buf, (size_t)max_bytes,
                              hapiMemcpyHostToDevice, stream));
    hapiCheck(hapiStreamSynchronize(stream));
  }
};

#include "osu_latency_gpu.def.h"
