#include "fib.decl.h"

#define THRESHOLD 3

struct Main : public CBase_Main {
  Main(CkArgMsg *m) { thisProxy.run(atoi(m->argv[1])); }

  void run(int n) {
    ck::future<int> f;
    CProxy_Fib::ckNew(n, f);
    CkPrintf("Fibonacci number is: %d\n", f.get());
    f.release();
    CkExit();
  }
};

struct Fib : public CBase_Fib {
  Fib(int n, const ck::future<int> &prev_) : prev(prev_) { thisProxy.calc(n); }

  int seqFib(int n) { return (n < 2) ? n : seqFib(n - 1) + seqFib(n - 2); }

  void calc(int n) {
    if (n < THRESHOLD) {
      prev.set(seqFib(n));
    } else {
      std::vector<ck::future<int>> pending(2);
      CProxy_Fib::ckNew(n - 1, pending[0]);
      CProxy_Fib::ckNew(n - 2, pending[1]);
      int sum = 0;
      if (n % 2 == 0) {
        // even n's use wait any
        while (pending.size()) {
          // wait for any of the futures in pending to become ready
          auto pair = ck::wait_any(pending.begin(), pending.end());
          // add the received value to the sum
          sum += pair.first;
          // then release and erase the fulfilled future
          pair.second->release();
          pending.erase(pair.second);
        }
      } else {
        // odds use wait all
        auto values = ck::wait_all(pending.begin(), pending.end());
        for (int i = 0; i < pending.size(); i++) {
          sum += values[i];
          pending[i].release();
        }
      }
      // set the parent's value to the sum
      prev.set(sum);
    }
    delete this;
  }

private:
  ck::future<int> prev;
};

#include "fib.def.h"
