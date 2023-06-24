#include <cstdint>
using fibonacci_t = std::uint64_t;

#include "fib.decl.h"

#define THRESHOLD 12

struct Main : public CBase_Main {
  Main(CkArgMsg *m) {
    if (m->argc > 1) thisProxy.run(atoi(m->argv[1]));
    else thisProxy.run(24);
  }

  void run(int n) {
    ck::future<fibonacci_t> f;
    CProxy_Fib::ckNew(n, f);
    ckout << "Fibonacci number is: " << f.get() << endl;
    f.release();
    CkExit();
  }
};

struct Fib : public CBase_Fib {
  Fib(int n, const ck::future<fibonacci_t> &prev_) : prev(prev_) { thisProxy.calc(n); }

  int seqFib(int n) { return (n < 2) ? n : seqFib(n - 1) + seqFib(n - 2); }

  void calc(int n) {
    if (n < THRESHOLD) {
      prev.set(seqFib(n));
    } else {
      std::vector<ck::future<fibonacci_t>> pending(2);
      CProxy_Fib::ckNew(n - 1, pending[0]);
      CProxy_Fib::ckNew(n - 2, pending[1]);
      fibonacci_t sum = 0;
      if (n % 2 == 0) {
        // even n's use wait any
        while (!pending.empty()) {
          // wait for any of the futures in pending to become ready
          auto pair = ck::wait_any(pending.begin(), pending.end());
          // add the received value to the sum
          sum += pair.first;
          // then release and erase the fulfilled future
          pair.second->release();
          pending.erase(pair.second);
        }
      } else if (n % 3 == 0) {
        // n's that are divisible by three (but not two) use wait_some
        auto pair = ck::wait_some(pending.begin(), pending.end());
        for (const auto& value : pair.first) {
          sum += value;
        }
        // in a real program, we could go off and do something else...
        // but we'll just wait on whatever's still outstanding
        for (auto it = pair.second; it < pending.end(); it++) {
          sum += it->get();
        }
      } else {
        // odd n's (that are not divisible by three) use wait all
        auto values = ck::wait_all(pending.begin(), pending.end());
        for (const auto& value : values) {
          sum += value;
        }
      }
      /* release any futures remaining in the vector `pending`.
       *
       * note #1: the wait_any and wait_some routes can remove
       *          futures from the vector.
       *
       * note #2, futures must be released or memory will be
       *          leaked and expired future ids will be 
       *          unavailable for reuse.
       */
      for (auto& f : pending) {
        f.release();
      }
      // set the parent's value to the sum
      prev.set(sum);
    }
    delete this;
  }

private:
  ck::future<fibonacci_t> prev;
};

#include "fib.def.h"
