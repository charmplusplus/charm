#include <stdio.h>
#include <stdlib.h>
#include "migrate.decl.h"

// Migrates elements of a 2D chare array whose index is NOT compressible:
// the array is created with no bounds and no initial size, and every element
// is inserted dynamically from PE 0. Without bounds there is no index
// compressor, so element ids come from the per-PE id counter and the
// index<->id map in CkLocMgr, which is the path this test exercises.
//
// Each element's SayHi(count) sends ackPlease() to the next element in
// row-major order; ackPlease() sends SayHi(count-1) back. Every
// MIGRATION_PERIOD steps an element migrates to the next PE. The program
// ends by quiescence detection once all counts reach 0.

#define MIGRATION_PERIOD 5
/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Hello arrProxy;
/*readonly*/ int nX;
/*readonly*/ int nY;

class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    nX = 4;
    nY = 3;
    int count = 40;
    if (m->argc > 1) nX = atoi(m->argv[1]);
    if (m->argc > 2) nY = atoi(m->argv[2]);
    if (m->argc > 3) count = atoi(m->argv[3]);
    delete m;

    CkPrintf("Running unbounded_migration on %d PEs for %d x %d elements, count %d\n",
             CkNumPes(), nX, nY, count);
    mainProxy = thisProxy;

    CkArrayOptions opts;  // no bounds, no initial size
    arrProxy = CProxy_Hello::ckNew(opts);
    for (int x = 0; x < nX; ++x)
      for (int y = 0; y < nY; ++y)
        arrProxy(x, y).insert();
    arrProxy.doneInserting();

    arrProxy.SayHi(count);
    CkStartQD(CkCallback(CkIndex_Main::done(), thisProxy));
  }

  void done()
  {
    CkPrintf("All done\n");
    CkExit();
  }
};

class Hello : public CBase_Hello
{
public:
  Hello()
  {
    CkPrintf("Hello[%d,%d] on PE %d: created.\n", thisIndex.x, thisIndex.y, CkMyPe());
  }

  Hello(CkMigrateMessage* m) {}

  void pup(PUP::er& p) { CBase_Hello::pup(p); }

  void ckJustMigrated()
  {
    CBase_Hello::ckJustMigrated();
    CkPrintf("Hello[%d,%d] migrated to PE %d\n", thisIndex.x, thisIndex.y, CkMyPe());
  }

  void SayHi(int count)
  {
    if (count > 0)
    {
      int lin = (thisIndex.x * nY + thisIndex.y + 1) % (nX * nY);
      thisProxy(lin / nY, lin % nY).ackPlease(thisIndex.x, thisIndex.y, count - 1);
    }
    if (count % MIGRATION_PERIOD == 1)
      migrateMe((CkMyPe() + 1) % CkNumPes());
  }

  void ackPlease(int px, int py, int c) { thisProxy(px, py).SayHi(c); }
};

#include "migrate.def.h"
