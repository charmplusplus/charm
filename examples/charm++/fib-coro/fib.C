#include "fib.decl.h"
#include "ckcoro.h"

#define THRESHOLD 3

struct Main : public CBase_Main {
  Main(CkArgMsg* m) { CProxy_Fib::ckNew(atoi(m->argv[1]), true, CProxy_Fib()); }
};

struct Fib : public CBase_Fib {
  Fib_CORO_CODE

  CProxy_Fib parent; bool isRoot;
  int val, val2;   // received at waits: members, as values live across waits

  Fib(int n, bool isRoot_, CProxy_Fib parent_)
    : parent(parent_), isRoot(isRoot_) {
    thisProxy.calc(n);
  }

  ck::coroutine calc(int n) {
    CK_COROUTINE();
    if (n < THRESHOLD) respond(seqFib(n));
    else {
      CProxy_Fib::ckNew(n - 1, false, thisProxy);
      CProxy_Fib::ckNew(n - 2, false, thisProxy);
      CK_WHEN(inbox::response(), val);    // when response(int val)
      CK_WHEN(inbox::response(), val2);   //   when response(int val2)
      respond(val + val2);
    }
    CK_COROUTINE_END();
  }

  int seqFib(int n) { return (n < 2) ? n : seqFib(n - 1) + seqFib(n - 2); }

  void respond(int val) {
    if (!isRoot) {
      parent.response(val);
      // The SDAG version deletes the chare here, inside the when body; a
      // coroutine body must not, so the chare is destroyed when the entry
      // method running on it returns (after calc() has finished).
      ckDestroyLater();
    } else {
      CkPrintf("Fibonacci number is: %d\n", val);
      CkExit();
    }
  }
};

#include "fib.def.h"
