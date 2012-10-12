#include "fib.decl.h"

#define THRESHOLD 3

struct Main : public CBase_Main {
  Main(CkArgMsg* m) { CProxy_Fib::ckNew(atoi(m->argv[1]), true, CProxy_Fib()); }
};

struct Fib : public CBase_Fib {
  Fib_SDAG_CODE

  CProxy_Fib parent; bool isRoot;

  Fib(int n, bool isRoot_, CProxy_Fib parent_)
    : parent(parent_), isRoot(isRoot_) {
    calc(n);
  }

  int seqFib(int n) { return (n < 2) ? n : seqFib(n - 1) + seqFib(n - 2); }

  void respond(int val) {
    if (!isRoot) {
      parent.response(val);
      delete this;
    } else {
      CkPrintf("Fibonacci number is: %d\n", val);
      CkExit();
    }
  }
};

#include "fib.def.h"

