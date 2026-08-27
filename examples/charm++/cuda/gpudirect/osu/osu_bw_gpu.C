// Windowed device-buffer streaming in the shape of the OSU bandwidth benchmark.
//
// The companion to osu_latency_gpu: latency locates the crossover from the
// fixed-cost side, bandwidth from the per-byte side. Staging spends two device
// copies per message against direct's one, so this is where that shows up --
// asymptotically the staged curve should sit near half the direct one, and the
// size at which the gap opens up is the same crossover the latency sweep finds.
//
// Same placement requirement, same reasons, same startup check as the latency
// benchmark. Same reason for carrying no completion callbacks, too: the sender
// waits for the receiver's ack before reusing the window, and the ack cannot be
// sent until every receive in the window has landed.

#include "osu_bw_gpu.decl.h"
#include "hapi.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// See the note in osu_latency_gpu.C: CkAbort's message does not survive on
// every build, and here the explanation is the output.
static void refuse(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  CkPrintf("\nosu_bw_gpu: %s\n\n", buf);
  fflush(stdout);
  CkAbort("%s", buf);
}

/*readonly*/ CProxy_Main main_proxy;
/*readonly*/ CProxy_Endpoint endpoint_proxy;
/*readonly*/ int max_bytes;
/*readonly*/ int window;
/*readonly*/ bool validate;

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
    window = 32;
    small_iters = 100;
    small_skip = 10;
    large_iters = 20;
    large_skip = 2;
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
      } else if (!strcmp(m->argv[i], "-W") && i + 1 < m->argc) {
        window = atoi(m->argv[++i]);
      } else if (!strcmp(m->argv[i], "-p") && i + 1 < m->argc) {
        peer_pe = atoi(m->argv[++i]);
      } else if (!strcmp(m->argv[i], "-v")) {
        validate = true;
      } else if (m->argv[i][0] == '+') {
        // See osu_latency_gpu.C: a launcher may append runtime flags after the
        // application's own arguments.
      } else {
        refuse("unknown argument '%s'. Usage: -s <min bytes> "
                "-e <max bytes> -i <iters> -w <warmup iters> -W <window> "
                "-p <peer PE> [-v]",
                m->argv[i]);
      }
    }
    delete m;

    if (window < 1) refuse("window must be at least 1");

    if (peer_pe < 0) {
      if (CkNumNodes() < 2) {
        refuse("this measures the cross-process, same-host device "
                "path, so it needs at least two processes. Launch two (one "
                "host, e.g. ++p 2 ++ppn 1) or name a peer PE with -p.");
      }
      peer_pe = CkNodeFirst(1);
    }
    if (peer_pe <= 0 || peer_pe >= CkNumPes()) {
      refuse("peer PE %d is out of range (this job has %d PEs), "
              "and it cannot be PE 0, which hosts the other endpoint.",
              peer_pe, CkNumPes());
    }
    if (CmiNodeOf(0) == CmiNodeOf(peer_pe)) {
      refuse("peer PE %d is in the same process as PE 0, so every "
              "send would resolve to a same-process device-to-device copy -- "
              "not the cross-process path this benchmark exists to measure.",
              peer_pe);
    }
    if (!CmiPeOnSamePhysicalNode(0, peer_pe)) {
      refuse("peer PE %d is on a different host from PE 0, so "
              "every send would resolve to inter-node RDMA rather than CUDA "
              "IPC. Run both processes on one host.",
              peer_pe);
    }

    CkArrayOptions opts(2);
    opts.setMap(CProxy_PairMap::ckNew(0, peer_pe));
    endpoint_proxy = CProxy_Endpoint::ckNew(opts);

    CkPrintf("# OSU-style GPU bandwidth test (device buffers, cross-process)\n");
    CkPrintf("# PE 0 (process %d) <-> PE %d (process %d), window %d\n",
             CmiNodeOf(0), peer_pe, CmiNodeOf(peer_pe), window);
    CkPrintf("# %-14s%s\n", "Size", "Bandwidth (MB/s)");
    fflush(stdout);

    cur_bytes = min_bytes;
    startNext();
  }

  void allReady() { endpoint_proxy[0].launch(); }

  void doneOne(double bandwidth_mbps) {
    CkPrintf("%-16d%.2f\n", cur_bytes, bandwidth_mbps);
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
  char* d_recv;  // window * cur_bytes on the receiving element
  char* h_buf;
  hapiStream_t stream;
  size_t recv_capacity;

  int cur_bytes, n_iters, n_skip, iter, recv_count;
  int verified_bytes;
  double t_start;

public:
  Endpoint() {
    hapiCheck(hapiStreamCreate(&stream));
    hapiCheck(hapiMalloc(&d_send, (size_t)max_bytes));
    hapiCheck(hapiMallocHost(&h_buf, (size_t)max_bytes));
    d_recv = NULL;
    recv_capacity = 0;

    fillPattern(h_buf, max_bytes, thisIndex);
    hapiCheck(hapiMemcpyAsync(d_send, h_buf, (size_t)max_bytes,
                              hapiMemcpyHostToDevice, stream));
    hapiCheck(hapiStreamSynchronize(stream));

    cur_bytes = 0;
    n_iters = n_skip = iter = recv_count = 0;
    verified_bytes = -1;
    t_start = 0.0;
  }

  ~Endpoint() {
    hapiCheck(hapiFree(d_send));
    if (d_recv) hapiCheck(hapiFree(d_recv));
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
    recv_count = 0;

    // One landing slot per outstanding send. Sized per size rather than once at
    // the maximum, which at a 32-deep window would reserve 32x the largest
    // message up front. Only ever grows, so a sweep pays this at most once per
    // size.
    const size_t needed = (size_t)window * (size_t)bytes;
    if (needed > recv_capacity) {
      if (d_recv) hapiCheck(hapiFree(d_recv));
      hapiCheck(hapiMalloc(&d_recv, needed));
      recv_capacity = needed;
    }

    // Both sides must have their slots before the first window flies, so the
    // reduction is a real barrier here, not a convenience.
    contribute(CkCallback(CkReductionTarget(Main, allReady), main_proxy));
  }

  void launch() {
    if (iter == n_skip) t_start = CkWallTimer();
    for (int slot = 0; slot < window; slot++) {
      thisProxy[1].recvChunk(slot, cur_bytes,
                             CkDeviceBuffer(d_send, (size_t)cur_bytes, stream));
    }
  }

  void recvChunk(int& slot, int& n, char*& buf, CkDeviceBufferPost* post) {
    buf = d_recv + (size_t)slot * (size_t)n;
    post[0].hapi_stream = stream;
  }

  void recvChunk(int slot, int n, char* buf) {
    if (validate && n != verified_bytes) {
      verified_bytes = n;
      verify(slot, n, 0);
    }
    if (++recv_count == window) {
      recv_count = 0;
      thisProxy[0].ack();
    }
  }

  void ack() {
    if (++iter < n_iters + n_skip) {
      launch();
    } else {
      const double elapsed = CkWallTimer() - t_start;
      const double bytes = (double)cur_bytes * (double)window * (double)n_iters;
      main_proxy.doneOne(bytes / elapsed / (1024.0 * 1024.0));
    }
  }

private:
  void verify(int slot, int n, int sender) {
    hapiCheck(hapiMemcpyAsync(h_buf, d_recv + (size_t)slot * (size_t)n,
                              (size_t)n, hapiMemcpyDeviceToHost, stream));
    hapiCheck(hapiStreamSynchronize(stream));
    for (int i = 0; i < n; i++) {
      const char expected = (char)((i + sender * 7) & 0x7f);
      if (h_buf[i] != expected) {
        refuse("validation failed on PE %d at byte %d of a %d-byte "
                "transfer from element %d (slot %d): expected %d, got %d.",
                CkMyPe(), i, n, sender, slot, (int)expected, (int)h_buf[i]);
      }
    }
    fillPattern(h_buf, max_bytes, thisIndex);
    hapiCheck(hapiMemcpyAsync(d_send, h_buf, (size_t)max_bytes,
                              hapiMemcpyHostToDevice, stream));
    hapiCheck(hapiStreamSynchronize(stream));
  }
};

#include "osu_bw_gpu.def.h"
