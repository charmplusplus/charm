// channel: the per-message device zero-copy path (a nocopydevice entry) against a
// persistent channel (CkDeviceChannel) between two PEs, in whichever transport the
// two PEs call for: a stream memcpy in one process (+ppn 2), a device-to-device
// copy across processes on one node (two processes), or RDMA across nodes.
//
// Per message size, four measurements:
//   mode 0  ping-pong latency, zero-copy entries (one-way time = round trip / 2)
//   mode 1  ping-pong latency, channel puts with release() on each arrival
//   mode 2  one-directional rate, zero-copy: the next send leaves when the
//           source-done callback of the previous one fires (one in flight)
//   mode 3  one-directional rate, channel: the next put is issued when the
//           previous one's source-done fires; it queues until the receiver's
//           credit arrives (one buffer, one in flight)
// and for each, the sender's time inside the send or put call (issue) and the
// receiver's time inside its arrival handler, as microseconds per message. That
// pair is the host cost a channel saves per message, which is what decides the
// application work in PERSISTENT_COMM_PLAN.md.
//
// Block is a 1D chare array with one element per PE (element i on PE i), as the
// applications' receivers are; PE 0 and the last PE exchange. Device buffers
// come from the pool under +gpupool.
#include "channel.decl.h"
#include "hapi.h"
#include <vector>
#include <cmath>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ size_t max_size;
/* readonly */ int n_iters;
/* readonly */ int n_iters_large;
/* readonly */ int warmup_iters;
/* readonly */ int n_rate;
/* readonly */ int n_rate_large;
/* readonly */ int use_get;   // -g: the channel modes pull with CkDevicePersistent::get instead of put

#define LARGE_MESSAGE_SIZE (256 * 1024)

static const char* mode_name[4] = {"zc-pingpong", "ch-pingpong", "zc-rate", "ch-rate"};

class Main : public CBase_Main {
  std::vector<size_t> sizes;
  size_t si;
  int mode;
  double start_time;
  double res_us[4], res_issue[4], res_recv[4];
  int mask;   // -m: bit k runs mode k (default all four)

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    max_size = 4 * 1024 * 1024;
    n_iters = 200;
    n_iters_large = 40;
    warmup_iters = 10;
    n_rate = 400;
    n_rate_large = 40;
    size_t min_size = 1024;
    mask = 15;
    use_get = 0;

    int c;
    while ((c = getopt(m->argc, m->argv, "s:x:i:l:w:r:R:m:g")) != -1) {
      switch (c) {
        case 's': min_size = atol(optarg); break;
        case 'x': max_size = atol(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'l': n_iters_large = atoi(optarg); break;
        case 'w': warmup_iters = atoi(optarg); break;
        case 'r': n_rate = atoi(optarg); break;
        case 'R': n_rate_large = atoi(optarg); break;
        case 'm': mask = atoi(optarg); break;
        case 'g': use_get = 1; break;
        default: CkAbort("usage: channel [-s min] [-x max] [-i iters] [-l iters_large] [-w warmup] [-r rate_n] [-R rate_n_large]");
      }
    }
    delete m;

    if (CkNumPes() < 2) CkAbort("channel: run with at least 2 PEs; PE 0 and the last PE are the pair");
    for (size_t s = min_size; s <= max_size; s *= 8) sizes.push_back(s);
    if (sizes.empty() || sizes.back() != max_size) sizes.push_back(max_size);

    if (use_get) CkPrintf("# channel path: receiver pull (-g): the sender announces, the receiver gets\n");
    CkPrintf("# channel vs zero-copy entries: PE 0 <-> PE %d of %d, %d process(es), %d physical node(s); "
             "sizes %zu..%zu B; pingpong iters %d (%d large) + %d warmup; rate messages %d (%d large)\n",
             CkNumPes() - 1, CkNumPes(), CkNumNodes(), CmiNumPhysicalNodes(), sizes.front(), sizes.back(),
             n_iters, n_iters_large, warmup_iters, n_rate, n_rate_large);
    block_proxy = CProxy_Block::ckNew(CkNumPes());
    start_time = CkWallTimer();
    block_proxy.init();
  }

  // Channel modes first: a stall in the zero-copy baseline then still leaves
  // the channel numbers in the log.
  static int order(int k) { static const int seq[4] = {1, 3, 0, 2}; return seq[k]; }
  int ki;

  void ready() {
    si = 0;
    ki = 0;
    for (int m = 0; m < 4; m++) res_us[m] = res_issue[m] = res_recv[m] = 0.0;
    startNext();
  }

  // The next selected mode of this size, or the size's line and the next size.
  void startNext() {
    while (ki < 4 && !(mask & (1 << order(ki)))) ki++;
    if (ki < 4) {
      mode = order(ki);
      block_proxy.start(mode, sizes[si]);
      return;
    }
    CkPrintf("%8zu B | one-way latency: zc %8.1f us, channel %8.1f us | one-in-flight rate: "
             "zc %8.0f msg/s (issue %5.1f us), channel %8.0f msg/s (issue %5.1f us, recv handler %5.1f us)\n",
             sizes[si], res_us[0], res_us[1], res_us[2], res_issue[2],
             res_us[3], res_issue[3], res_recv[3]);
    if (++si < sizes.size()) {
      ki = 0;
      startNext();
    } else {
      CkPrintf("Elapsed: %.3f s\n", CkWallTimer() - start_time);
      CkExit();
    }
  }

  void result(int m, size_t size, double us, double issue_us, double recv_us) {
    (void)size;
    res_us[m] = us; res_issue[m] = issue_us; res_recv[m] = recv_us;
    ki++;
    startNext();
  }
};

class Block : public CBase_Block {
  int peer;
  char* d_send;
  char* d_recv;
  cudaStream_t stream;
  CkDeviceChannel my_ch;     // my receive buffer, exported to the peer
  CkDeviceChannel peer_ch;   // the peer's receive buffer, written by my puts
  // The pull variant: my send buffer exported, the peer's send buffer to get from,
  // my receive buffer as the local endpoint of the get.
  CkDevicePersistent send_ep, peer_send_ep, recv_ep;
  int handshakes;

  int mode;
  size_t size;
  int iter, n_sent, received;
  double t0, t_issue, t_recv;
  std::vector<double> times;

  int itersFor(size_t s) const { return s > LARGE_MESSAGE_SIZE ? n_iters_large : n_iters; }
  int rateFor(size_t s) const { return s > LARGE_MESSAGE_SIZE ? n_rate_large : n_rate; }

public:
  bool participant() const { return thisIndex == 0 || thisIndex == CkNumPes() - 1; }

  Block() : d_send(nullptr), d_recv(nullptr) {
    peer = (thisIndex == 0) ? CkNumPes() - 1 : 0;
    if (CkMyPe() != thisIndex) CkAbort("channel: element %d is on PE %d; the array map must place element i on PE i\n", thisIndex, CkMyPe());
  }

  void init() {
    // Only PE 0 and the last PE exchange; the others (the rest of a process,
    // as an application has) just take part in the reductions.
    if (!participant()) {
      contribute(CkCallback(CkReductionTarget(Main, ready), main_proxy));
      return;
    }
    if (CkDevicePoolOn()) {
      d_send = (char*)CkDeviceMalloc(max_size);
      d_recv = (char*)CkDeviceMalloc(max_size);
    } else {
      hapiCheck(cudaMalloc(&d_send, max_size));
      hapiCheck(cudaMalloc(&d_recv, max_size));
    }
    hapiCheck(cudaStreamCreate(&stream));
    hapiCheck(cudaMemset(d_send, 'a' + CkMyPe(), max_size));
    hapiCheck(cudaMemset(d_recv, 0, max_size));
    hapiCheck(cudaDeviceSynchronize());

    my_ch = CkDeviceChannel(d_recv, max_size, CkCallback(CkIndex_Block::arrived(NULL), thisProxy[thisIndex]));
    my_ch.open();
    handshakes = 0;
    thisProxy[peer].takeChannel(my_ch);
    send_ep = CkDevicePersistent(d_send, max_size);
    send_ep.open();
    recv_ep = CkDevicePersistent(d_recv, max_size, stream);
    thisProxy[peer].takeSendEp(send_ep);
  }

  void takeChannel(CkDeviceChannel ch) {
    peer_ch = ch;
    if (++handshakes == 2) contribute(CkCallback(CkReductionTarget(Main, ready), main_proxy));
  }
  void takeSendEp(CkDevicePersistent ep) {
    peer_send_ep = ep;
    if (++handshakes == 2) contribute(CkCallback(CkReductionTarget(Main, ready), main_proxy));
  }

  // Pull variant: the sender's announcement; the receiver gets `n` bytes from
  // the sender's buffer into its own and continues when the get completes.
  void pullReady(size_t n) {
    double t = CkWallTimer();
    recv_ep.cnt = n;
    peer_send_ep.cnt = n;
    recv_ep.hapi_stream = stream;
    recv_ep.cb = CkCallback(CkIndex_Block::pulled(), thisProxy[thisIndex]);
    recv_ep.cb_msg = nullptr;
    recv_ep.get(peer_send_ep);
    t_recv += CkWallTimer() - t;
  }
  void pulled() {
    thisProxy[peer].sent();   // the sender's source is free again (what the put's source-done says)
    if (mode == 1) pingpongStep();
    else rateStep();
  }

  void start(int mode_, size_t size_) {
    if (!participant()) return;
    mode = mode_;
    size = size_;
    iter = n_sent = received = 0;
    t_issue = t_recv = 0.0;
    times.clear();
    if (thisIndex != 0) return;
    t0 = CkWallTimer();
    if (mode == 0 || mode == 2) sendZC();
    else sendCh();
  }

  // One zero-copy entry-method send of `size` bytes; the source may be reused
  // when sent() fires.
  void sendZC() {
    double t = CkWallTimer();
    thisProxy[peer].recvZC(size, CkDeviceBuffer(d_send, size,
        CkCallback(CkIndex_Block::sent(), thisProxy[thisIndex]), stream));
    t_issue += CkWallTimer() - t;
  }

  // One channel put of `size` bytes; queues behind the credit if the previous
  // put was not yet released by the receiver.
  void sendCh() {
    double t = CkWallTimer();
    if (use_get) thisProxy[peer].pullReady(size);
    else peer_ch.put(d_send, size, stream, CkCallback(CkIndex_Block::sent(), thisProxy[thisIndex]));
    t_issue += CkWallTimer() - t;
  }

  // Zero-copy receive: the post names the destination, the body runs when the
  // bytes are in place.
  void recvZC(size_t& n, char*& data, CkDeviceBufferPost* post) {
    data = d_recv;
    post[0].hapi_stream = stream;
  }
  // The zero-copy receive's host work (admission, copy issue, completion, ack)
  // runs inside the runtime before this body; CHARM_ZC_STATS=1 prints it per
  // process at exit ([zc-host] recv-issue). Nothing here to time.
  void recvZC(size_t n, char* data) {
    if (mode == 0) pingpongStep();
    else rateStep();
  }

  // Channel receive: the put's event, the credit back, then the step.
  void arrived(CkDataMsg* m) {
    double t = CkWallTimer();
    const CkDeviceChannelEvent* e = my_ch.event(m);
    delete m;
    if (e == NULL) return;   // a resolve, answered by the runtime
    my_ch.release();
    t_recv += CkWallTimer() - t;
    if (mode == 1) pingpongStep();
    else rateStep();
  }

  void pingpongStep() {
    if (thisIndex != 0) {
      // pong
      if (mode == 0) sendZC(); else sendCh();
      return;
    }
    // PE 0: a round trip completed
    if (iter >= warmup_iters) times.push_back((CkWallTimer() - t0) / 2.0);
    iter++;
    if (iter < warmup_iters + itersFor(size)) {
      t0 = CkWallTimer();
      if (mode == 0) sendZC(); else sendCh();
    } else {
      double sum = 0.0;
      for (double x : times) sum += x;
      const double mean = sum / times.size();
      const int n = iter;
      main_proxy.result(mode, size, mean * 1e6, t_issue / n * 1e6, t_recv / n * 1e6);
    }
  }

  // Sender-side completion: the source may be reused, so the next message of a
  // rate test leaves now (a put may queue behind the credit).
  void sent() {
    if (thisIndex != 0 || (mode != 2 && mode != 3)) return;
    n_sent++;
    if (n_sent < rateFor(size)) {
      if (mode == 2) sendZC(); else sendCh();
    }
  }

  void rateStep() {
    // PE 1 counts arrivals; the last one reports the receiver's handler cost.
    received++;
    if (received == rateFor(size)) {
      thisProxy[0].rateDone(t_recv / received * 1e6);
    }
  }

  void rateDone(double recv_us) {
    const double elapsed = CkWallTimer() - t0;
    const int n = rateFor(size);
    main_proxy.result(mode, size, n / elapsed, t_issue / n * 1e6, recv_us);
  }
};

#include "channel.def.h"
