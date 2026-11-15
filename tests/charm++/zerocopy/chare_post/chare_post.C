// Zerocopy post entry methods (nocopypost) on a singleton chare.
//
// Usage: ./chare_post (any number of PEs)
//
// Main sends two buffers, a small and a large one, to a chare on the last PE.
// The chare's post entry points each one at a buffer of its own, and the
// regular entry checks every byte that arrived. Main exits after both
// receives are checked and both source buffers are released.

#include <vector>
#include "chare_post.decl.h"

/*readonly*/ CProxy_Main mainProxy;

static const int kSizes[2] = {1000, 1 << 20};

static char pattern(int which, int i) { return (char)(i * 7 + which * 13 + 3); }

class Main : public CBase_Main {
  std::vector<char> src[2];
  int pending = 4;  // two source releases and two checked receives

public:
  Main(CkArgMsg *m) {
    delete m;
    mainProxy = thisProxy;
    CProxy_Receiver r = CProxy_Receiver::ckNew(CkNumPes() - 1);
    CkCallback cb(CkIndex_Main::sourceDone(NULL), thisProxy);
    for (int w = 0; w < 2; w++) {
      src[w].resize(kSizes[w]);
      for (int i = 0; i < kSizes[w]; i++) src[w][i] = pattern(w, i);
      r.recv(w, CkSendBuffer(src[w].data(), cb), kSizes[w]);
    }
  }

  void sourceDone(CkDataMsg *m) {
    delete m;
    finishOne();
  }

  void received(int which) { finishOne(); }

  void finishOne() {
    if (--pending == 0) {
      CkPrintf("All tests passed\n");
      CkExit();
    }
  }
};

class Receiver : public CBase_Receiver {
  std::vector<char> dst[2];

public:
  Receiver() {}

  // Post entry: name the buffer each transfer lands in.
  void recv(int &which, char *&d, int &n, CkNcpyBufferPost *ncpyPost) {
    if (which < 0 || which > 1 || n != kSizes[which])
      CkAbort("chare_post: post entry got which=%d n=%d", which, n);
    dst[which].assign(n, 0);
    d = dst[which].data();
  }

  void recv(int which, char *d, int n) {
    if (d != dst[which].data())
      CkAbort("chare_post: buffer %d did not arrive in the posted buffer", which);
    for (int i = 0; i < n; i++)
      if (d[i] != pattern(which, i))
        CkAbort("chare_post: buffer %d byte %d is %d, expected %d", which, i, d[i],
                pattern(which, i));
    mainProxy.received(which);
  }
};

#include "chare_post.def.h"
