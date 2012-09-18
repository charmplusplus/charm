#include "threaded_ring.h"

Main::Main(CkArgMsg*)
{
    nElems = 10;
    CProxy_Ring A = CProxy_Ring::ckNew(nElems);
    CkCallback* cb = new CkCallback(CkCallback::ckExit);
    A.ckSetReductionClient(cb);
    A.run();
}

void Ring::run()
{
    CkPrintf("[%d]: nElems = %d\n", thisIndex, nElems);
    nElems = 10;
    for (int i=0; i<nElems; ++i) {
        CkPrintf("[%d] iteration %d\n", thisIndex, i);
        thisProxy((thisIndex+1) % nElems).getData();
        waitFor();
    }
    CkPrintf("[%d] done\n", thisIndex);
    contribute(0, 0, CkReduction::concat);
}

void Ring::getData()
{
    CkPrintf("[%d]::getData()\n", thisIndex);
    if (threadWaiting) {
        CthAwaken(t);
        threadWaiting = false;
    } else {
        dataHere++;
    }
}

void Ring::waitFor()
{
    CkPrintf("[%d]::waitFor()\n", thisIndex);
    if (dataHere) {
        dataHere--;
        return;
    }

    t = CthSelf();
    threadWaiting = true;
    CthSuspend();
}

#include "threaded_ring.def.h"
