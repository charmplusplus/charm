#include <stdio.h>
#include "AmpiInterop.h"
#include "hello.decl.h"

void exm_mpi_fn(void* in, void* out);

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    // Process command-line arguments
    nElements = 2;
    if (m->argc > 1) nElements=atoi(m->argv[1]);
    delete m;

    // Init the AMPI chare group
    AmpiInteropInit();

    // Init the Hello chare array
    CkPrintf("[%d] In Charm++: running on %d PEs with %d chares\n",
             CkMyPe(), CkNumPes(), nElements);
    mainProxy = thisProxy;
    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);
    arr[0].sayHi(0);
  };

  void done() {
    CkPrintf("[%d] In Charm++: done\n", CkMyPe());
  };
};

/*array [1D]*/
class Hello : public CBase_Hello {
  int myDat, resDat;

public:
  Hello() {
    myDat = thisIndex;
    CkPrintf("[%d] In Charm++: created Chare idx %d\n", CkMyPe(), thisIndex);
  }

  Hello(CkMigrateMessage *m) {}

  void sayHi(int n) {
    CkPrintf("[%d] In Charm++: hello[%d] from Chare idx %d\n", CkMyPe(), n, thisIndex);
    MpiCallData mpiCall;
    mpiCall.fn  = exm_mpi_fn;
    mpiCall.in  = &myDat;
    mpiCall.out = &resDat;
    mpiCall.cb  = CkCallback(CkIndex_Hello::doneLibCall(), thisProxy[thisIndex]);
    ampiInteropProxy[CkMyPe()].callMpiFn(CkMyPe(), thisIndex, mpiCall);

    // Pass the hello on
    if (thisIndex < nElements-1) {
      thisProxy[thisIndex+1].sayHi(thisIndex+1);
    }
  }

  void doneLibCall() {
    if (thisIndex == nElements-1) {
      ampiInteropProxy.finish();
      mainProxy.done();
    }
  }
};

#include "hello.def.h"
