// A [threaded, reductiontarget] entry method must run on its own thread and
// may block. The phases cover the target kinds charmxi generates separately:
//   1. a mainchare target: waits for quiescence and on a future;
//   2. array elements reached by a broadcast reduction: each waits on a future;
//   3. a [nokeep] target with a marshalled array: the array lives in the
//      CkReductionMsg, which must survive the thread's suspend;
//   4. a group target; 5. a templated array target.
// Each target checks it is not on the thread its object was constructed on
// (the PE's scheduler thread), then blocks.
#include "threaded_redn.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

#define CHECK_OWN_THREAD(who)                                                   \
  if (CthSelf() == ctorThread)                                                  \
  CkAbort("%s: threaded reduction target ran on the scheduler's thread", who)

static int sumBelow(int n) { return n * (n - 1) / 2; }

class Main : public CBase_Main {
  CProxy_Worker workers;
  CProxy_Counter counters;
  CProxy_Templ<int> templs;
  CthThread ctorThread;  // the PE's own thread, which the target must not be on

 public:
  Main(CkArgMsg *m) {
    nElements = 2 * CkNumPes();
    delete m;
    mainProxy = thisProxy;
    ctorThread = CthSelf();
    workers = CProxy_Worker::ckNew(nElements);
    counters = CProxy_Counter::ckNew();
    templs = CProxy_Templ<int>::ckNew(nElements);
    workers.contributeValue();
  }

  void sumDone(int sum) {
    CHECK_OWN_THREAD("Main::sumDone");
    int expected = sumBelow(nElements);
    if (sum != expected) CkAbort("wrong reduction sum: %d, expected %d", sum, expected);
    CkWaitQD();  // blocking: only legal on a thread
    CkFuture f = CkCreateFuture();
    workers[nElements - 1].fillFuture(f);
    CkReductionMsg *reply = (CkReductionMsg *)CkWaitFuture(f);  // blocking again
    int fromWorker = *(int *)reply->getData();
    delete reply;
    CkReleaseFuture(f);
    if (fromWorker != nElements - 1) CkAbort("future carried %d, expected %d", fromWorker, nElements - 1);
    CkPrintf("sumDone ran on a thread, waited for quiescence and a future (sum %d)\n", sum);
    workers.contributeBcast();
  }

  void bcastDone(int n) {
    if (n != nElements) CkAbort("bcastDone: %d of %d elements", n, nElements);
    CkPrintf("array: broadcast target waited on a future in all %d elements\n", n);
    workers.contributeVec();
  }

  void vecDone(int n) {
    if (n != nElements) CkAbort("vecDone: %d of %d elements", n, nElements);
    CkPrintf("array: nokeep target kept its marshalled array across a suspend in all %d elements\n", n);
    counters.contributeValue();
  }

  void groupDone(int n) {
    if (n != CkNumPes()) CkAbort("groupDone: %d of %d PEs", n, CkNumPes());
    CkPrintf("group: target waited for quiescence on all %d PEs\n", n);
    templs.contributeValue();
  }

  void templDone(int n) {
    if (n != nElements) CkAbort("templDone: %d of %d elements", n, nElements);
    CkPrintf("templated array: target waited for quiescence in all %d elements\n", n);
    workers.finish();
  }

  void allDone() {
    CkPrintf("All done\n");
    CkExit();
  }
};

// Counts one per caller into a Main target.
#define REPORT_ONE(target)                                                      \
  do {                                                                          \
    int one = 1;                                                                \
    contribute(sizeof(int), &one, CkReduction::sum_int,                         \
               CkCallback(CkReductionTarget(Main, target), mainProxy));         \
  } while (0)

class Worker : public CBase_Worker {
  CthThread ctorThread;

 public:
  Worker() : ctorThread(CthSelf()) {}
  void contributeValue() {
    int v = thisIndex;
    contribute(sizeof(int), &v, CkReduction::sum_int, CkCallback(CkReductionTarget(Main, sumDone), mainProxy));
  }
  void contributeBcast() {
    int v = thisIndex;
    contribute(sizeof(int), &v, CkReduction::sum_int, CkCallback(CkReductionTarget(Worker, bcastSum), thisProxy));
  }
  void bcastSum(int sum) {
    CHECK_OWN_THREAD("Worker::bcastSum");
    if (sum != sumBelow(nElements)) CkAbort("Worker::bcastSum: wrong sum %d", sum);
    int next = (thisIndex + 1) % nElements;
    CkFuture f = CkCreateFuture();
    thisProxy[next].fillFuture(f);
    CkReductionMsg *reply = (CkReductionMsg *)CkWaitFuture(f);
    int fromNext = *(int *)reply->getData();
    delete reply;
    CkReleaseFuture(f);
    if (fromNext != next) CkAbort("Worker::bcastSum: future carried %d, expected %d", fromNext, next);
    REPORT_ONE(bcastDone);
  }
  void contributeVec() {
    int v[3] = {thisIndex, 1, 2 * thisIndex};
    contribute(sizeof(v), v, CkReduction::sum_int, CkCallback(CkReductionTarget(Worker, vecSum), thisProxy));
  }
  void vecSum(int n, int *v) {
    CHECK_OWN_THREAD("Worker::vecSum");
    int s = sumBelow(nElements);
    if (n != 3 || v[0] != s || v[1] != nElements || v[2] != 2 * s) CkAbort("Worker::vecSum: wrong vector");
    CkWaitQD();
    if (v[0] != s || v[1] != nElements || v[2] != 2 * s)
      CkAbort("Worker::vecSum: vector changed across a suspend");
    REPORT_ONE(vecDone);
  }
  void fillFuture(CkFuture f) {
    int v = thisIndex;
    CkReductionMsg *msg = CkReductionMsg::buildNew(sizeof(int), &v);
    CkSendToFuture(f, msg);
  }
  void finish() { contribute(CkCallback(CkReductionTarget(Main, allDone), mainProxy)); }
};

class Counter : public CBase_Counter {
  CthThread ctorThread;

 public:
  Counter() : ctorThread(CthSelf()) {}
  void contributeValue() {
    int v = CkMyPe();
    contribute(sizeof(int), &v, CkReduction::sum_int, CkCallback(CkReductionTarget(Counter, sumDone), thisProxy));
  }
  void sumDone(int sum) {
    CHECK_OWN_THREAD("Counter::sumDone");
    if (sum != sumBelow(CkNumPes())) CkAbort("Counter::sumDone: wrong sum %d", sum);
    CkWaitQD();
    REPORT_ONE(groupDone);
  }
};

template <typename U>
class Templ : public CBase_Templ<U> {
  CthThread ctorThread;

 public:
  Templ() : ctorThread(CthSelf()) {}
  void contributeValue() {
    U v = this->thisIndex;
    this->contribute(sizeof(U), &v, CkReduction::sum_int,
                     CkCallback(CkReductionTarget(Templ<U>, sumDone), this->thisProxy));
  }
  void sumDone(U sum) {
    CHECK_OWN_THREAD("Templ::sumDone");
    if (sum != sumBelow(nElements)) CkAbort("Templ::sumDone: wrong sum");
    CkWaitQD();
    int one = 1;
    this->contribute(sizeof(int), &one, CkReduction::sum_int,
                     CkCallback(CkReductionTarget(Main, templDone), mainProxy));
  }
};

#define CK_TEMPLATES_ONLY
#include "threaded_redn.def.h"
#undef CK_TEMPLATES_ONLY
#include "threaded_redn.def.h"
