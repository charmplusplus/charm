/*
 * Round trips between two array elements, each message matched by iteration
 * number, three ways: plain entry methods (no matching), an SDAG when, and a
 * coroutine CK_WHEN site (ckcoro.h). Element 0 sends refnum 0; element 1
 * sends each message back with the same refnum; element 0 answers refnum r
 * with r+1. Prints the time per round trip (two matched messages).
 *
 *   ./pingpong plain|sdag|coro [iterations] [payload bytes] +pe 1   (both elements on one PE)
 *   ./pingpong ... +pe 2                                            (one element per PE)
 */
#include <cstring>
#include <limits>
#include "pingpong.decl.h"
#include "ckcoro.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int niter;
/*readonly*/ int payload;

class PingMsg : public CMessage_PingMsg {
 public:
  char *data;
};

static PingMsg *make_msg(int ref) {
  PingMsg *m = new (payload) PingMsg;
  CkSetRefNum(m, ref);
  return m;
}

class Main : public CBase_Main {
  double t0;
  const char *mode;
 public:
  Main(CkArgMsg *m) {
    mainProxy = thisProxy;
    niter = 50000;
    payload = 8;
    mode = m->argc > 1 ? m->argv[1] : "coro";
    if (m->argc > 2) niter = atoi(m->argv[2]);
    if (m->argc > 3) payload = atoi(m->argv[3]);
    // The iteration is the message's reference number, a CMK_REFNUM_TYPE
    // (unsigned short by default): it wraps, which is harmless here because
    // only one message per element is ever in flight.
    CkPrintf("pingpong %s: %d round trips, %d-byte payload, %d PEs\n", mode, niter, payload, CkNumPes());
    t0 = CkWallTimer();
    if (strcmp(mode, "plain") == 0) {
      CProxy_Plain a = CProxy_Plain::ckNew(2);
      a[0].start();
    } else if (strcmp(mode, "sdag") == 0) {
      CProxy_Sdag a = CProxy_Sdag::ckNew(2);
      a.run();
    } else if (strcmp(mode, "coro") == 0) {
      CProxy_Coro a = CProxy_Coro::ckNew(2);
      a.run();
    } else {
      CkAbort("mode must be plain, sdag or coro");
    }
    delete m;
  }
  // t: element 0's time from its first send to its last receive.
  void done(double t) {
    CkPrintf("pingpong %s: %.3f us per round trip, %.3f us per message\n", mode,
             1e6 * t / niter, 1e6 * t / niter / 2);
    CkExit();
  }
};

class Plain : public CBase_Plain {
  double t0;
  int iter = 0;
 public:
  Plain() {}
  void start() {
    t0 = CkWallTimer();
    thisProxy[1].recv(make_msg(0));
  }
  void recv(PingMsg *m) {
    int r = CkGetRefNum(m);
    delete m;
    if (thisIndex == 1) {
      thisProxy[0].recv(make_msg(r));
    } else if (++iter < niter) {
      thisProxy[1].recv(make_msg(iter));
    } else {
      mainProxy.done(CkWallTimer() - t0);
    }
  }
};

class Sdag : public CBase_Sdag {
  Sdag_SDAG_CODE
  int iter;
  double t0;
 public:
  Sdag() {}
  void begin() {
    if (thisIndex == 0) {
      t0 = CkWallTimer();
      thisProxy[1].recv(make_msg(0));
    }
  }
  void bounce(PingMsg *m) {
    int r = CkGetRefNum(m);
    delete m;
    if (thisIndex == 1) thisProxy[0].recv(make_msg(r));
    else if (iter + 1 < niter) thisProxy[1].recv(make_msg(iter + 1));
  }
  void finish() {
    if (thisIndex == 0) mainProxy.done(CkWallTimer() - t0);
  }
};

class Coro : public CBase_Coro {
  ck::coro::state co;
  int iter;
  double t0;
  PingMsg *m;
 public:
  Coro() {}
  void recv(PingMsg *msg) { co.deliver(CkIndex_Coro::recv(0), msg); }
  ck::coroutine run() {
    CK_COROUTINE(co, CkIndex_Coro::run());
    if (thisIndex == 0) {
      t0 = CkWallTimer();
      thisProxy[1].recv(make_msg(0));
    }
    for (iter = 0; iter < niter; iter++) {
      CK_WHEN(CkIndex_Coro::recv(0), iter, m);
      {
        int r = CkGetRefNum(m);
        delete m;
        if (thisIndex == 1) thisProxy[0].recv(make_msg(r));
        else if (iter + 1 < niter) thisProxy[1].recv(make_msg(iter + 1));
      }
    }
    if (thisIndex == 0) mainProxy.done(CkWallTimer() - t0);
    CK_COROUTINE_END();
  }
};

#include "pingpong.def.h"
