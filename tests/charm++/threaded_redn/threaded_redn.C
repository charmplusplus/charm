// A [threaded, reductiontarget] entry method must run on its own thread and
// may block: it waits for quiescence and on a future before the program ends.
#include "threaded_redn.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

class Main : public CBase_Main {
  CProxy_Worker workers;
  CthThread schedulerThread;  // the PE's own thread, which the target must not be on

 public:
  Main(CkArgMsg *m) {
    nElements = 2 * CkNumPes();
    delete m;
    mainProxy = thisProxy;
    schedulerThread = CthSelf();
    workers = CProxy_Worker::ckNew(nElements);
    workers.contributeValue();
  }

  void sumDone(int sum) {
    if (CthSelf() == schedulerThread)
      CkAbort("threaded reduction target ran on the scheduler's thread");
    int expected = nElements * (nElements - 1) / 2;
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
    workers.finish();
  }

  void allDone() {
    CkPrintf("All done\n");
    CkExit();
  }
};

class Worker : public CBase_Worker {
 public:
  Worker() {}
  void contributeValue() {
    int v = thisIndex;
    contribute(sizeof(int), &v, CkReduction::sum_int, CkCallback(CkReductionTarget(Main, sumDone), mainProxy));
  }
  void fillFuture(CkFuture f) {
    int v = thisIndex;
    CkReductionMsg *msg = CkReductionMsg::buildNew(sizeof(int), &v);
    CkSendToFuture(f, msg);
  }
  void finish() { contribute(CkCallback(CkReductionTarget(Main, allDone), mainProxy)); }
};

#include "threaded_redn.def.h"
