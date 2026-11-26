// A complete Charm++ program with a mainchare. Under CharmLibInit it runs
// from the mainchare's constructor to CkExit(code): an array broadcast, a
// reduction, then CkExit(code) on the last PE. This file must not include
// mpi-interoperate.h, which redefines CkExit to end an interop phase.
#include <stdlib.h>
#include "exitcode.decl.h"

CProxy_ExitMain exitMain;
CProxy_ExitGroup exitGroup;
int exitNumElems;

class ExitMain : public CBase_ExitMain {
public:
  ExitMain(CkArgMsg *m) {
    code = m->argc > 1 ? atoi(m->argv[1]) : 0;
    delete m;
    exitNumElems = 4 * CkNumPes();
    exitMain = thisProxy;
    exitGroup = CProxy_ExitGroup::ckNew();
    CProxy_ExitElem elems = CProxy_ExitElem::ckNew(exitNumElems);
    elems.work();
  }
  void done(long sum) {
    long want = (long)exitNumElems * (exitNumElems + 1) / 2;
    if (sum != want)
      CkAbort("exitcode: reduction sum %ld, expected %ld", sum, want);
    CkPrintf("exitcode: reduction sum %ld correct on %d PEs; PE %d calls "
             "CkExit(%d)\n", sum, CkNumPes(), CkNumPes() - 1, code);
    exitGroup[CkNumPes() - 1].finish(code);
  }
private:
  int code;
};

class ExitElem : public CBase_ExitElem {
public:
  ExitElem() {}
  ExitElem(CkMigrateMessage *) {}
  void work() {
    long v = thisIndex + 1;
    contribute(sizeof(v), &v, CkReduction::sum_long,
               CkCallback(CkReductionTarget(ExitMain, done), exitMain));
  }
};

class ExitGroup : public CBase_ExitGroup {
public:
  ExitGroup() {}
  void finish(int code) { CkExit(code); }
};

#include "exitcode.def.h"
