// Zerocopy entry methods (nocopypost and nocopy) on a singleton chare and on
// a mainchare.
//
// Usage: ./chare_post (any number of PEs)
//
// Each transfer sends a small or a large buffer; transfer ids 0-7 select the
// size (id % 2) and the byte pattern. Main sends to a Receiver chare on the
// last PE:
//   0, 1  a nocopypost entry method
//   2, 3  a nocopypost parameter of an SDAG when
// The Receiver sends back to Main:
//   4, 5  a nocopypost entry method of the mainchare
//   6, 7  a nocopy entry method of the mainchare
// Every receive checks that the bytes arrived in the posted buffer (for
// nocopypost) and checks every byte. Main exits after all 8 receives are
// checked and all 8 source buffers are released.

#include <vector>
#include "chare_post.decl.h"

/*readonly*/ CProxy_Main mainProxy;

static const int kNumIds = 8;
static int sizeOf(int which) { return which % 2 ? 1 << 20 : 1000; }
static char pattern(int which, int i) { return (char)(i * 7 + which * 13 + 3); }

static void fill(std::vector<char> &v, int which) {
  v.resize(sizeOf(which));
  for (int i = 0; i < (int)v.size(); i++) v[i] = pattern(which, i);
}

static void check(int which, const char *d, int n) {
  if (which < 0 || which >= kNumIds || n != sizeOf(which))
    CkAbort("chare_post: transfer %d arrived with n=%d", which, n);
  for (int i = 0; i < n; i++)
    if (d[i] != pattern(which, i))
      CkAbort("chare_post: transfer %d byte %d is %d, expected %d", which, i, d[i],
              pattern(which, i));
}

class Main : public CBase_Main {
  std::vector<char> src[4];
  std::vector<char> dst[kNumIds];
  int pending = 2 * kNumIds;  // one source release and one checked receive per id

public:
  Main(CkArgMsg *m) {
    delete m;
    mainProxy = thisProxy;
    CProxy_Receiver r = CProxy_Receiver::ckNew(CkNumPes() - 1);
    r.run();
    CkCallback cb(CkIndex_Main::sourceDone(NULL), thisProxy);
    for (int w = 0; w < 4; w++) {
      fill(src[w], w);
      if (w < 2)
        r.recv(w, CkSendBuffer(src[w].data(), cb), sizeOf(w));
      else
        r.sdagRecv(w, CkSendBuffer(src[w].data(), cb), sizeOf(w));
    }
    r.sendToMain();
  }

  void sourceDone(CkDataMsg *m) {
    delete m;
    finishOne();
  }

  void received(int which) { finishOne(); }

  // Post entry: name the buffer each transfer lands in.
  void mainRecv(int &which, char *&d, int &n, CkNcpyBufferPost *ncpyPost) {
    if (which < 4 || which > 5 || n != sizeOf(which))
      CkAbort("chare_post: mainRecv post entry got which=%d n=%d", which, n);
    dst[which].assign(n, 0);
    d = dst[which].data();
  }

  void mainRecv(int which, char *d, int n) {
    if (d != dst[which].data())
      CkAbort("chare_post: transfer %d did not arrive in the posted buffer", which);
    check(which, d, n);
    finishOne();
  }

  void mainRecvNocopy(int which, char *d, int n) {
    if (which < 6 || which > 7)
      CkAbort("chare_post: mainRecvNocopy got which=%d", which);
    check(which, d, n);
    finishOne();
  }

  void finishOne() {
    if (--pending == 0) {
      CkPrintf("All tests passed\n");
      CkExit();
    }
  }
};

class Receiver : public CBase_Receiver {
  Receiver_SDAG_CODE
  int k;
  std::vector<char> dst[4];
  std::vector<char> src[kNumIds];

public:
  Receiver() {}

  // Post entries: name the buffer each transfer lands in.
  void recv(int &which, char *&d, int &n, CkNcpyBufferPost *ncpyPost) { post(which, d, n, 0); }
  void sdagRecv(int &which, char *&d, int &n, CkNcpyBufferPost *ncpyPost) { post(which, d, n, 2); }

  void post(int which, char *&d, int n, int first) {
    if (which < first || which > first + 1 || n != sizeOf(which))
      CkAbort("chare_post: post entry got which=%d n=%d", which, n);
    dst[which].assign(n, 0);
    d = dst[which].data();
  }

  void recv(int which, char *d, int n) {
    if (d != dst[which].data())
      CkAbort("chare_post: transfer %d did not arrive in the posted buffer", which);
    check(which, d, n);
    mainProxy.received(which);
  }

  void sendToMain() {
    CkCallback cb(CkIndex_Main::sourceDone(NULL), mainProxy);
    for (int w = 4; w < kNumIds; w++) {
      fill(src[w], w);
      if (w < 6)
        mainProxy.mainRecv(w, CkSendBuffer(src[w].data(), cb), sizeOf(w));
      else
        mainProxy.mainRecvNocopy(w, CkSendBuffer(src[w].data(), cb), sizeOf(w));
    }
  }
};

#include "chare_post.def.h"
