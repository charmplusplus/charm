// chantest: two array elements exchange a device buffer every iteration through
// device channels (CkChannel), each verifying what it received. With -l
// they swap PEs every lb_period iterations (each element calls migrateMe)
// WITHOUT exchanging the channels again: the receiver reopens its channel on the
// new buffer in its pup, and the sender's next put re-resolves.
//
// With -a the puts run free while the receivers move: each element puts again
// from the source-done of its previous put (one in flight, no barrier), and
// migrates itself from its arrival handler every lb_period arrivals while its
// peer keeps putting -- the puts a sender makes between acking a closing and
// learning the receiver's new home are staged.
//
// Every run ends by quiescence detection: PASS is printed once the runtime
// finds nothing left in flight, so the channel's protocol messages and the
// notes and credits it passes through shared memory must all be counted.
// With -q every iteration ends that way, and each element then checks it has
// taken the iteration's put: quiescence reached with an arrival not yet
// delivered fails. With -w S element 0 keeps the first put it receives for S
// seconds before releasing it, so the peer's next put waits for the credit
// long enough for the watchdog to probe its owner (two quiet 10 s ticks;
// S >= 35). With -x element 0 also writes into its own channel through a
// second copy, which must abort: one sender per channel. With -r each sender
// attaches before its first put (with -x, the attach must abort).
//
// Every run ends by destroying the channels: each receiver calls destroy(),
// and once every callback has run, each sender's copy must know its channel
// was destroyed. With -e element 0 then puts once more, which must abort.
// With -c element 0 copies its sender's channel into a vector at setup and
// puts through the copy; at the end it puts through the original, which must
// abort: the copy took the channel over.
//
// With -d N each PE also hosts N filler elements that keep its scheduler queue
// deep (each message spins -u microseconds, 1000 by default, and sends the
// next): the channel protocol's messages wait behind them, as they would
// behind a busy application. With -a it measures how long moves are held;
// set CHARM_CHANNEL_STATS=1 for the per-PE wait times at exit.
//
// Run with 2 PEs: +p2 in one process (copies on the device), or two processes
// on one host (copies through the IPC mapping), or two hosts (the receiver
// reads with RDMA).
#include <atomic>
#include <vector>
#include "chantest.decl.h"
#include "hapi.h"

#define ERROR_TOLERANCE 1e-6

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Elem elem_proxy;
/* readonly */ int block_size;
/* readonly */ int n_iters;
/* readonly */ bool async_moves;
/* readonly */ int lb_period;
/* readonly */ bool qd_steps;
/* readonly */ int hold_secs;
/* readonly */ bool two_senders;
/* readonly */ bool attach_first;
/* readonly */ bool put_after_destroy;
/* readonly */ bool copy_sender;
/* readonly */ int fillers;
/* readonly */ int fill_us;
/* readonly */ CProxy_Filler filler_proxy;

// -d: the fillers of this process stop resending once told.
static std::atomic<int> fill_stop{0};

extern void invokeFillKernel(double*, int, double, hapiStream_t);

struct Container {
  double* h_remote_data;
  double* d_local_data;
  double* d_remote_data;
  hapiStream_t stream;

  Container() : h_remote_data(nullptr), d_local_data(nullptr), d_remote_data(nullptr), stream(nullptr) {}

  ~Container() {
    if (h_remote_data) hapiCheck(hapiFreeHost(h_remote_data));
    if (d_local_data) hapiCheck(hapiFree(d_local_data));
    if (d_remote_data) hapiCheck(hapiFree(d_remote_data));
    if (stream) hapiCheck(hapiStreamDestroy(stream));
  }

  void init() {
    hapiCheck(hapiMallocHost((void**)&h_remote_data, sizeof(double) * block_size));
    hapiCheck(hapiMalloc((void**)&d_local_data, sizeof(double) * block_size));
    hapiCheck(hapiMalloc((void**)&d_remote_data, sizeof(double) * block_size));
    hapiCheck(hapiStreamCreate(&stream));
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
  bool lb_test;
  double start_time;
  char pass_line[256];

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    block_size = 4096;
    n_iters = 20;
    lb_period = 3;
    lb_test = false;
    async_moves = false;
    qd_steps = false;
    hold_secs = 0;
    two_senders = false;
    attach_first = false;
    put_after_destroy = false;
    copy_sender = false;
    fillers = 0;
    fill_us = 1000;
    int c;
    while ((c = getopt(m->argc, m->argv, "s:i:lp:aqw:xrecd:u:")) != -1) {
      switch (c) {
        case 's': block_size = atoi(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'l': lb_test = true; break;
        case 'p': lb_period = atoi(optarg); break;
        case 'a': async_moves = true; lb_test = true; break;
        case 'q': qd_steps = true; break;
        case 'w': hold_secs = atoi(optarg); break;
        case 'x': two_senders = true; break;
        case 'r': attach_first = true; break;
        case 'e': put_after_destroy = true; break;
        case 'c': copy_sender = true; break;
        case 'd': fillers = atoi(optarg); break;
        case 'u': fill_us = atoi(optarg); break;
        default: CkAbort("usage: chantest [-s block] [-i iters] [-l] [-p lb_period] [-a] [-q] [-w secs] [-x] [-r] [-e] [-c] "
                         "[-d fillers] [-u us]");
      }
    }
    delete m;
    if (CkNumPes() != 2) CkAbort("chantest: run with 2 PEs");
    if ((qd_steps || hold_secs > 0 || two_senders || copy_sender) && lb_test)
      CkAbort("chantest: -q, -w, -x and -c run without migration (no -l or -a)");
    if (copy_sender && put_after_destroy) CkAbort("chantest: -c and -e each end with their own abort");
    if (fillers > 0 && qd_steps) CkAbort("chantest: -q needs quiescence each iteration, which -d's fillers prevent");
    if (two_senders) CkPrintf("[chantest] -x: element 0 writes into its own channel too; expect the abort naming two senders\n");
    if (put_after_destroy) CkPrintf("[chantest] -e: element 0 puts after the destroy; expect the abort\n");
    if (copy_sender)
      CkPrintf("[chantest] -c: element 0 puts through a copy of its sender's channel, then through the original; "
               "expect the abort saying the copy holds the channel\n");
    CkPrintf("[chantest] block %d doubles, %d iterations, migration %s (every %d), %d process(es), %d host(s)%s, "
             "%d filler(s) per PE of %d us\n",
             block_size, n_iters, async_moves ? "async" : lb_test ? "on" : "off", lb_period, CkNumNodes(),
             CmiNumPhysicalNodes(), attach_first ? ", attach first" : "", fillers, fill_us);
    // Created before the elements, so that the fillers' messages are queued
    // ahead of the channel's from the start.
    if (fillers > 0) {
      filler_proxy = CProxy_Filler::ckNew(fillers * CkNumPes());
      filler_proxy.work();
    }
    elem_proxy = CProxy_Elem::ckNew(2);
    iter = 0;
    migrations = 0;
    start_time = CkWallTimer();
    elem_proxy.setup();
  }

  void ready() {
    if (async_moves) elem_proxy.runAsync();
    else startStep();
  }

  void startStep() {
    elem_proxy.step(iter);
    if (qd_steps) CkStartQD(CkCallback(CkIndex_Main::stepQuiet(), thisProxy));
  }

  // -q: nothing is in flight, so every put of this iteration has been taken.
  void stepQuiet() { elem_proxy.checkStep(iter); }

  // The end: the fillers stop, both receivers destroy their channels, each
  // sender's copy must then know, and PASS once quiescence is reached.
  void finish() {
    if (fillers > 0) filler_proxy.stop();
    elem_proxy.teardown();
  }
  void tornDown() { elem_proxy.checkDetached(); }
  // Per sender copy: 1 if told of the destroy, 100 if it still holds a
  // resolution. A copy that moved after its last put holds none (it learns of
  // the destroy at its next attach); one still holding a resolution after the
  // callbacks may still map the freed buffer.
  void detached(int sum) {
    const int told = sum % 100, held = sum / 100;
    if (held != 0) CkAbort("[chantest] FAIL: after both destroys, %d sender copies still hold a resolution\n", held);
    if (!async_moves && told != 2) CkAbort("[chantest] FAIL: after both destroys, %d of 2 senders were told\n", told);
    CkPrintf("[chantest] both channels destroyed; no sender holds a resolution; %d of 2 senders told\n", told);
    if (put_after_destroy) { elem_proxy[0].putAfterDestroy(); return; }
    if (copy_sender) { elem_proxy[0].putOnOriginal(); return; }
    CkStartQD(CkCallback(CkIndex_Main::quiet(), thisProxy));
  }
  void quiet() {
    CkPrintf("%s, quiescence reached\n", pass_line);
    CkExit();
  }

  void asyncDone(int migrated) {
    if (migrated == 0) CkAbort("[chantest] FAIL: -a given, but no element migrated");
    snprintf(pass_line, sizeof(pass_line), "[chantest] PASS: %d puts each way verified, %d element migrations (async), %.3f s",
             n_iters, migrated, CkWallTimer() - start_time);
    finish();
  }

  void iterDone(int migrated) {
    migrations += migrated;
    iter++;
    if (iter >= n_iters) {
      if (lb_test && migrations == 0) CkAbort("[chantest] FAIL: -l given, but no element migrated");
      snprintf(pass_line, sizeof(pass_line), "[chantest] PASS: %d iterations verified%s, %d element migrations, %.3f s",
               n_iters, qd_steps ? " (each ended by quiescence)" : "", migrations, CkWallTimer() - start_time);
      finish();
      return;
    }
    if (lb_test && (iter % lb_period) == 0) elem_proxy.doLB();
    else startStep();
  }

  void lbDone(int migrated) {
    migrations += migrated;
    startStep();
  }
};

class Elem : public CBase_Elem {
  Container container;
  CkChannel my_ch;     // my receive buffer, exported for the peer
  CkChannel peer_ch;   // the peer's receive buffer, written by my puts
  CkChannel own_ch;    // -x, element 0: a second copy of its own channel (never migrates)
  std::vector<CkChannel> kept;   // -c, element 0: holds the copy of peer_ch that puts (never migrates)
  bool second_sender = false;
  int takes = 0;
  int peer;
  int home_pe;   // the PE this element was on at the last step: counts migrations
  // -a: puts issued, source-dones seen, puts received, moves made
  int n_put = 0, n_done = 0, n_recv = 0, moves = 0;
  bool finished = false;

public:
  Elem() {
    peer = 1 - thisIndex;
    home_pe = CkMyPe();
    container.init();
  }

  Elem(CkMigrateMessage* m) {}

  void pup(PUP::er& p) {
    p|peer;
    p|home_pe;
    p|n_put; p|n_done; p|n_recv; p|moves; p|finished;
    p|container;   // allocates the buffers on unpack
    p|my_ch;
    p|peer_ch;
    // The receiver's channel lives on at the new home, on the new buffer.
    if (p.isUnpacking()) my_ch.reopen(container.d_remote_data);
  }

  void setup() {
    int dev = 0;
    hapiCheck(hapiGetDevice(&dev));
    my_ch = CkChannel(container.d_remote_data, sizeof(double) * block_size, CkMem::Device, dev,
                      CkCallback(CkIndex_Elem::arrived(NULL), thisProxy[thisIndex]));
    thisProxy[peer].takeChannel(my_ch);
    if (two_senders && thisIndex == 0) thisProxy[thisIndex].takeOwnCopy(my_ch);
  }

  void takeChannel(CkChannel ch) {
    peer_ch = ch;
    if (attach_first) peer_ch.attach();
    if (copy_sender && thisIndex == 0) kept.push_back(peer_ch);   // the copy takes the channel over
    took();
  }
  void takeOwnCopy(CkChannel ch) {
    own_ch = ch;
    second_sender = true;
    if (attach_first) own_ch.attach();
    took();
  }
  void took() {
    if (++takes < ((two_senders && thisIndex == 0) ? 2 : 1)) return;
    contribute(CkCallback(CkReductionTarget(Main, ready), main_proxy));
  }

  static double valueOf(int iter, int from) { return 1000.0 * (from + 1) + iter; }
  // The copy that holds the peer's channel: -c's, or peer_ch.
  CkChannel& sendCh() { return kept.empty() ? peer_ch : kept.back(); }

  void step(int iter) {
    container.fill(valueOf(iter, thisIndex));
    // The iteration travels as the put's tag: the peer's arrival can run before
    // its own step(iter) message, so its counter is not what to verify against.
    CkPut(sendCh(), container.d_local_data, sizeof(double) * block_size,
          CkPutOpts().stream(container.stream).tag(iter));
    if (second_sender)
      CkPut(own_ch, container.d_local_data, sizeof(double) * block_size,
            CkPutOpts().stream(container.stream).tag(iter));
  }

  // The end of every run: destroy my channel; the buffer stays until the callback.
  void teardown() {
    my_ch.destroy(CkCallback(CkIndex_Elem::destroyed(), thisProxy[thisIndex]));
  }
  void destroyed() {
    contribute(CkCallback(CkReductionTarget(Main, tornDown), main_proxy));
  }
  // Every destroy's callback has run: my copy of the peer's channel must have
  // been told (the destroy closes the sender's resolution before completing).
  void checkDetached() {
    int v = (sendCh().isDestroyed() ? 1 : 0) + (sendCh().isAttached() ? 100 : 0);
    contribute(sizeof(int), &v, CkReduction::sum_int, CkCallback(CkReductionTarget(Main, detached), main_proxy));
  }
  // -e: must abort.
  void putAfterDestroy() {
    CkPut(peer_ch, container.d_local_data, sizeof(double) * block_size, CkPutOpts().stream(container.stream));
    CkAbort("[chantest] FAIL: a put after the destroy did not abort\n");
  }
  // -c: peer_ch was copied into kept; a put on it must abort.
  void putOnOriginal() {
    CkPut(peer_ch, container.d_local_data, sizeof(double) * block_size, CkPutOpts().stream(container.stream));
    CkAbort("[chantest] FAIL: a put on a copied channel's original did not abort\n");
  }

  // -q: quiescence after step(iter): the put of iteration iter must have been taken.
  void checkStep(int iter) {
    if (n_recv != iter + 1)
      CkAbort("[chantest] FAIL: quiescence reached before element %d on PE %d took iteration %d's put (%d taken)\n",
              thisIndex, CkMyPe(), iter, n_recv);
    int migrated = 0;
    contribute(sizeof(int), &migrated, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, iterDone), main_proxy));
  }

  // -w: the release held back from arrived().
  static void releaseLater(void* arg, double) { elem_proxy[(int)(intptr_t)arg].lateRelease(); }
  void lateRelease() {
    CkPrintf("[chantest] element %d releases the first put after %d s\n", thisIndex, hold_secs);
    my_ch.release();
  }

  void arrived(CkDataMsg* m) {
    const CkChannelEvent* e = my_ch.event(m);
    delete m;
    if (e == NULL) return;   // an attach or an announce, handled by the runtime
    int bad = -1; double got = 0.0;
    const int iter = (int)e->tag;
    const double want = valueOf(iter, peer);
    // -x: both senders write this buffer, so its contents say nothing; the
    // runtime must abort at the second sender's resolve.
    if (!two_senders && !container.verify(want, bad, got)) {
      CkAbort("[chantest] element %d on PE %d: iteration %d put %llu (%zu bytes) wrong at index %d: expected %.1f, got %.1f\n",
              thisIndex, CkMyPe(), iter, (unsigned long long)e->seq, e->bytes, bad, want, got);
    }
    if (hold_secs > 0 && thisIndex == 0 && n_recv == 0 && !async_moves)
      CcdCallFnAfter(releaseLater, (void*)(intptr_t)thisIndex, 1000.0 * hold_secs);
    else
      my_ch.release();
    if (async_moves) {
      n_recv++;
      if (n_recv == n_iters) { checkDone(); return; }
      // Nothing after the call (see doLB); the peer keeps putting meanwhile.
      if (n_recv % lb_period == 0) migrateMe(1 - CkMyPe());
      return;
    }
    n_recv++;
    if (qd_steps) return;   // the iteration ends by quiescence (checkStep)
    int migrated = (home_pe != CkMyPe()) ? 1 : 0;
    home_pe = CkMyPe();
    contribute(sizeof(int), &migrated, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, iterDone), main_proxy));
  }

  // Swap with the peer's PE. Nothing after the call: the element may already be
  // packed when it returns, or the move is deferred until its channels' senders
  // have acknowledged the closing.
  void doLB() { migrateMe(1 - CkMyPe()); }

  void runAsync() { putNext(); }
  void putNext() {
    n_put++;
    container.fill(valueOf(n_put, thisIndex));
    CkPut(peer_ch, container.d_local_data, sizeof(double) * block_size,
          CkPutOpts().stream(container.stream).done(CkCallback(CkIndex_Elem::sent(), thisProxy[thisIndex])).tag(n_put));
  }
  // The source of the previous put is done: the next put may refill it.
  void sent() {
    n_done++;
    if (n_put < n_iters) putNext();
    else checkDone();
  }
  void checkDone() {
    if (finished || n_done < n_iters || n_recv < n_iters) return;
    finished = true;
    contribute(sizeof(int), &moves, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, asyncDone), main_proxy));
  }

  void ckJustMigrated() {
    ArrayElement::ckJustMigrated();
    int migrated = (home_pe != CkMyPe()) ? 1 : 0;
    home_pe = CkMyPe();
    if (async_moves) { moves += migrated; return; }
    contribute(sizeof(int), &migrated, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, lbDone), main_proxy));
  }
};

// -d: keeps its PE's scheduler queue deep. One message per filler is in the
// queue at any time, each spinning fill_us microseconds.
class Filler : public CBase_Filler {
public:
  Filler() {}
  Filler(CkMigrateMessage* m) {}
  void work() {
    const double until = CkWallTimer() + 1e-6 * fill_us;
    while (CkWallTimer() < until) {}
    if (!fill_stop.load(std::memory_order_relaxed)) thisProxy[thisIndex].work();
  }
  void stop() { fill_stop.store(1, std::memory_order_relaxed); }
};

#include "chantest.def.h"
