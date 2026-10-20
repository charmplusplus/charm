// sendcost: what one converse message costs between two PEs, in the layout the
// channel's credit uses. PE 0 and the last PE are the pair (same process with
// +ppn 2, two processes on one node, or two nodes).
//   mode 0  burst: PE 0 calls CmiSyncSendAndFree n_burst times (64 B payload)
//           and reports the mean time per call; the receiver reports the mean
//           handler cost.
//   mode 1  converse ping-pong (bare CmiSyncSendAndFree both ways): one-way time.
//   mode 2  entry-method ping-pong (a plain Charm++ entry both ways): one-way time.
//   mode 3  burst of entry-method sends: mean time per call.
#include "sendcost.decl.h"
#include <vector>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Peer peer_proxy;
/* readonly */ int n_burst;
/* readonly */ int n_pingpong;

static int conv_handler;
struct ConvMsg {
  char core[CmiMsgHeaderSizeBytes];
  int from;
  int seq;
  int mode;
  char pad[64];
};
static double g_recv_t = 0.0;
static int g_recv_n = 0;
static int g_last_seq = 0;
static int g_burst_expected = 0;

class Main : public CBase_Main {
  double res[4][3];
  int mode;
public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    n_burst = 2000;
    n_pingpong = 500;
    int c;
    while ((c = getopt(m->argc, m->argv, "b:p:")) != -1) {
      switch (c) {
        case 'b': n_burst = atoi(optarg); break;
        case 'p': n_pingpong = atoi(optarg); break;
      }
    }
    delete m;
    if (CkNumPes() < 2) CkAbort("sendcost: at least 2 PEs");
    CkPrintf("# sendcost: PE 0 <-> PE %d of %d, %d process(es), %d physical node(s); burst %d, pingpong %d\n",
             CkNumPes() - 1, CkNumPes(), CkNumNodes(), CmiNumPhysicalNodes(), n_burst, n_pingpong);
    peer_proxy = CProxy_Peer::ckNew();
    peer_proxy.init();
  }
  void ready() { mode = 0; peer_proxy.start(mode); }
  void result(int m, double a, double b, double c) {
    res[m][0] = a; res[m][1] = b; res[m][2] = c;
    if (++mode < 4) { peer_proxy.start(mode); return; }
    CkPrintf("converse burst   : send call %6.2f us/msg, receiver handler %6.2f us/msg, rate %8.0f msg/s\n",
             res[0][0], res[0][1], res[0][2]);
    CkPrintf("converse pingpong: one-way %6.2f us\n", res[1][0]);
    CkPrintf("entry    pingpong: one-way %6.2f us\n", res[2][0]);
    CkPrintf("entry    burst   : send call %6.2f us/msg, rate %8.0f msg/s\n", res[3][0], res[3][2]);
    CkExit();
  }
};

extern "C" void sendcost_conv_handler(void* vm) {
  ConvMsg* m = (ConvMsg*)vm;
  Peer* p = peer_proxy.ckLocalBranch();
  // forward declared below via a free function
  extern void sendcost_on_conv(Peer*, ConvMsg*);
  sendcost_on_conv(p, m);
}

class Peer : public CBase_Peer {
public:
  int peer, mode, seq, received;
  double t0, t_issue;
  std::vector<double> times;
  bool participant() const { return CkMyPe() == 0 || CkMyPe() == CkNumPes() - 1; }
  Peer() { peer = (CkMyPe() == 0) ? CkNumPes() - 1 : 0; }
  void init() { contribute(CkCallback(CkReductionTarget(Main, ready), main_proxy)); }
  void start(int mode_) {
    mode = mode_; seq = 0; received = 0; t_issue = 0.0; times.clear();
    g_recv_t = 0.0; g_recv_n = 0; g_last_seq = 0;
    if (!participant() || CkMyPe() != 0) return;
    t0 = CkWallTimer();
    if (mode == 0) {
      for (int i = 0; i < n_burst; i++) {
        double t = CkWallTimer();
        sendConv(i, 0);
        t_issue += CkWallTimer() - t;
      }
      // the receiver reports when it has them all
    } else if (mode == 1) {
      sendConv(0, 1);
    } else if (mode == 2) {
      thisProxy[peer].ping(0);
    } else {
      for (int i = 0; i < n_burst; i++) {
        double t = CkWallTimer();
        thisProxy[peer].count();
        t_issue += CkWallTimer() - t;
      }
    }
  }
  void sendConv(int s, int md) {
    ConvMsg* m = (ConvMsg*)CmiAlloc(sizeof(ConvMsg));
    CmiSetHandler(m, conv_handler);
    m->from = CkMyPe(); m->seq = s; m->mode = md;
    CmiSyncSendAndFree(peer, sizeof(ConvMsg), (char*)m);
  }
  void onConv(ConvMsg* m) {
    if (m->mode == 0) {
      // burst receiver: count, time the handler, report at the end
      double t = CkWallTimer();
      received++;
      g_recv_t += CkWallTimer() - t;
      if (received == n_burst) {
        thisProxy[0].burstDone(g_recv_t / received * 1e6);
      }
    } else {
      // ping-pong
      if (CkMyPe() != 0) { sendConv(m->seq, 1); return; }
      times.push_back((CkWallTimer() - t0) / 2.0);
      if (++seq < n_pingpong) { t0 = CkWallTimer(); sendConv(seq, 1); }
      else report(1);
    }
    CmiFree(m);
  }
  void burstDone(double recv_us) {
    const double elapsed = CkWallTimer() - t0;
    main_proxy.result(mode, t_issue / n_burst * 1e6, recv_us, n_burst / elapsed);
  }
  void ping(int s) { thisProxy[0].pong(s); }
  void pong(int s) {
    times.push_back((CkWallTimer() - t0) / 2.0);
    if (++seq < n_pingpong) { t0 = CkWallTimer(); thisProxy[peer].ping(seq); }
    else report(2);
  }
  void count() {
    received++;
    if (received == n_burst) thisProxy[0].burstDone(0.0);
  }
  void report(int m) {
    double s = 0.0; int n = 0;
    for (size_t i = times.size() / 5; i < times.size(); i++) { s += times[i]; n++; }   // drop the first fifth
    main_proxy.result(m, s / n * 1e6, 0.0, 0.0);
  }
};

void sendcost_on_conv(Peer* p, ConvMsg* m) { p->onConv(m); }

static void sendcost_init(void) {
  conv_handler = CmiRegisterHandler((CmiHandler)sendcost_conv_handler);
}

#include "sendcost.def.h"
