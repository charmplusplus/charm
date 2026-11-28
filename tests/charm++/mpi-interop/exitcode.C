// A complete Charm++ program with a mainchare, run inside an MPI job under
// CharmLibInit (mpi_exitcode.cpp) or CharmMpiMain (mpi_main.cpp, or charmc
// -mpi-main). It runs from the mainchare's constructor to CkExit(code): an
// array broadcast, a reduction, then CkExit(code) on the last PE.
// Usage: <program> <code> [mpi|abort]. With "mpi", every process first calls
// MPI_Allreduce from a nodegroup entry method; with "abort", the last PE
// calls CkAbort instead of CkExit. This file must not include
// mpi-interoperate.h, which redefines CkExit to end an interop phase.
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "exitcode.decl.h"

CProxy_ExitMain exitMain;
CProxy_ExitGroup exitGroup;
CProxy_ExitNode exitNode;
int exitNumElems;

class ExitMain : public CBase_ExitMain {
public:
  ExitMain(CkArgMsg *m) {
    code = m->argc > 1 ? atoi(m->argv[1]) : 0;
    useMpi = m->argc > 2 && strcmp(m->argv[2], "mpi") == 0;
    doAbort = m->argc > 2 && strcmp(m->argv[2], "abort") == 0;
    delete m;
    exitNumElems = 4 * CkNumPes();
    exitMain = thisProxy;
    exitGroup = CProxy_ExitGroup::ckNew();
    exitNode = CProxy_ExitNode::ckNew();
    CProxy_ExitElem elems = CProxy_ExitElem::ckNew(exitNumElems);
    elems.work();
  }
  void done(long sum) {
    long want = (long)exitNumElems * (exitNumElems + 1) / 2;
    if (sum != want)
      CkAbort("exitcode: reduction sum %ld, expected %ld", sum, want);
    CkPrintf("exitcode: reduction sum %ld correct on %d PEs\n", sum,
             CkNumPes());
    if (useMpi)
      exitNode.allreduce();
    else
      finish();
  }
  // One per process, each with the sum MPI_Allreduce gave that process.
  void mpiDone(int sum) {
    if (sum != CkNumNodes())
      CkAbort("exitcode: MPI_Allreduce sum %d, expected %d", sum, CkNumNodes());
    if (++mpiReplies < CkNumNodes()) return;
    CkPrintf("exitcode: MPI_Allreduce sum %d correct on %d processes\n", sum,
             CkNumNodes());
    finish();
  }
private:
  void finish() {
    if (doAbort) {
      exitGroup[CkNumPes() - 1].abortHere();
      return;
    }
    CkPrintf("exitcode: PE %d calls CkExit(%d)\n", CkNumPes() - 1, code);
    exitGroup[CkNumPes() - 1].finish(code);
  }
  int code;
  bool useMpi, doAbort;
  int mpiReplies = 0;
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
  void finish(int code) {
    CkExit(code);
    // Reached under CharmLibInit only: there CkExit returns to its caller.
    CkPrintf("exitcode: CkExit returned to its caller\n");
  }
  void abortHere() {
    CkAbort("exitcode: deliberate abort on PE %d (process %d of %d)", CkMyPe(),
            CkMyNode(), CkNumNodes());
  }
};

class ExitNode : public CBase_ExitNode {
public:
  ExitNode() {}
  // Runs on one PE of every process; blocks in MPI until all have called it.
  void allreduce() {
    int one = 1, sum = 0;
    MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    exitMain.mpiDone(sum);
  }
};

#include "exitcode.def.h"
