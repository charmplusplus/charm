#include <stdio.h>
#include <thread>

#include "atsync.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;
/*readonly*/ int iters;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    // Process command-line arguments
    nElements = 5;
    if (m->argc > 1)
      nElements = atoi(m->argv[1]);
    iters = 10;
    if (m->argc > 2)
      iters = atoi(m->argv[2]);
    delete m;

    // Start the computation
    CkPrintf("Running atsync on %d processors for %d elements\n", CkNumPes(), nElements);

    mainProxy = thisProxy;
    CProxy_atsync arr = CProxy_atsync::ckNew();
    for (int i = 0; i < nElements; i++)
    {
      arr[i].insert();
    }
    arr.doneInserting();

    arr[0].start();
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
};

class atsync : public CBase_atsync
{
private:
  int counter;
  int originalPE;

public:
  atsync()
  {
    CkPrintf("atsync (%d) created on %d\n", thisIndex, CkMyPe());
    counter = 0;
    originalPE = CkMyPe();
    usesAtSync = true;
  }

  atsync(CkMigrateMessage* m) {}

  void ResumeFromSync()
  {
    CkPrintf("[%d] %d resume from sync, iter %d\n", CkMyPe(), thisIndex, counter);
    if (thisIndex == 0)
      thisProxy[thisIndex].start();
  }

  void pup(PUP::er& p)
  {
    p | counter;
    p | originalPE;
  }

  void start()
  {
    counter++;
    const int value = thisIndex;
    if (value + 1 < nElements)
    {
      CkPrintf("[%d] Sending to %d from element %d\n", CkMyPe(), thisIndex + 1,
               thisIndex);
      thisProxy[thisIndex + 1].start();
    }

    if (counter < iters)
    {
      AtSync();
    }
    else
    {
      mainProxy.done();
    }
  }
};

#include "atsync.def.h"
