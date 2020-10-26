#include <ck/dht.h>
#include "shuffle.decl.h"

#define THRESHOLD 3

struct Main : public CBase_Main {
  Main(CkArgMsg *m) {
    int n = atoi(m->argv[1]);
    ck::dht<int, int> dht;
    CProxy_Shuffler shufflers = CProxy_Shuffler::ckNew(n, dht, n);
    shufflers.shuffle(CkCallback(CkCallback::ckExit));
  }
};

struct Shuffler : public CBase_Shuffler {
  Shuffler(int n_, const ck::dht<int, int> &dht_) : n(n_), dht(dht_){};

  void shuffle(CkCallback cb) {
    // Store a value at our index in the DHT
    dht.insert(thisIndex, 2 * (thisIndex + 1));
    // Request a value from our rightmost neighbor
    // (wrapping back to 0)
    auto neighbor = (thisIndex + 1) % n;
    auto f = dht.request(neighbor);
    // Wait for, then print the received value
    CkPrintf("[%d] DHT[%d] = %d\n", (int)thisIndex, (int)neighbor, f.get());
    // Contribute to the reduction
    contribute(cb);
  }

private:
  ck::dht<int, int> dht;
  int n;
};

#include "shuffle.def.h"
