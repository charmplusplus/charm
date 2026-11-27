#include "MigrateCtorEntry.decl.h"

CProxy_Main mainProxy;

struct Main : public CBase_Main
{
  Main(CkArgMsg* m)
  {
    mainProxy = thisProxy;
    CProxy_Cell cells = CProxy_Cell::ckNew(4);
    cells.work();
  }

  void done()
  {
    ckout << "MigrateCtorEntry: test passed" << endl;
    CkExit();
  }
};

struct Cell : public CBase_Cell
{
  bool migrated = false;
  int origPe;

  Cell() : origPe(CkMyPe()) {}
  Cell(CkMigrateMessage* m) {}

  void pup(PUP::er& p)
  {
    CBase_Cell::pup(p);
    p | migrated;
    p | origPe;
  }

  void ckJustMigrated()
  {
    CBase_Cell::ckJustMigrated();
    thisProxy[thisIndex].work();
  }

  void work()
  {
    if (!migrated && CkNumPes() > 1)
    {
      migrated = true;
      migrateMe((CkMyPe() + 1) % CkNumPes());
      return;
    }
    if (migrated && CkMyPe() == origPe)
      CkAbort("element did not migrate");
    contribute(CkCallback(CkReductionTarget(Main, done), mainProxy));
  }
};

#include "MigrateCtorEntry.def.h"
