#include "fib.decl.h"

#define THRESHOLD 12

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
      ck::future<int> f1, f2;
      CProxy_Fib::ckNew(n - 1, f1);
      CProxy_Fib::ckNew(n - 2, f2);
      prev.set(f1.get() + f2.get());
      f1.release();
      f2.release();
    }
    delete this;
  }

private:
  ck::future<int> prev;
};

#include "fib.def.h"
