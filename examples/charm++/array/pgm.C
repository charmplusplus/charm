#include "pgm.decl.h"

/*readonly*/ CProxy_Main mainProxy;

struct Main : CBase_Main {
  int numRecv;

  Main(CkArgMsg* msg)
    : numRecv(0)
  {
    delete msg;

    mainProxy = thisProxy;

    const int X = 3;
    const int Y = 3;
    const int Z = 3;
    CProxy_ArrayA a1 = CProxy_ArrayA::ckNew(X);
    CProxy_ArrayB b1 = CProxy_ArrayB::ckNew(X, Y);
    CProxy_ArrayC c1 = CProxy_ArrayC::ckNew(X, Y, Z);

    a1.e();
    b1.e();
    c1.e();
  }

  void finished() {
    if (++numRecv == 3) CkExit();
  }
};

struct ArrayA : CBase_ArrayA {
  ArrayA() {
    CkPrintf("ArrayA: created element %d\n", thisIndex);
  }
  ArrayA(CkMigrateMessage*) { }
  void e() { contribute(CkCallback(CkReductionTarget(Main, finished), mainProxy)); }
};

struct ArrayB : CBase_ArrayB {
  ArrayB() {
    CkPrintf("ArrayB: created element (%d,%d)\n", thisIndex.x, thisIndex.y);
  }
  ArrayB(CkMigrateMessage*) { }
  void e() { contribute(CkCallback(CkReductionTarget(Main, finished), mainProxy)); }
};

struct ArrayC : CBase_ArrayC {
  ArrayC() {
    CkPrintf("ArrayB: created element (%d,%d,%d)\n", thisIndex.x, thisIndex.y, thisIndex.z);
  }
  ArrayC(CkMigrateMessage*) { }
  void e() { contribute(CkCallback(CkReductionTarget(Main, finished), mainProxy)); }
};

struct ArrayD : CBase_ArrayD {
  ArrayD() {}
  ArrayD(CkMigrateMessage*) { }
  void e() { }
};

struct ArrayE : CBase_ArrayE {
  ArrayE() {}
  ArrayE(CkMigrateMessage*) { }
  void e() { }
};

struct ArrayF : CBase_ArrayF {
  ArrayF() {}
  ArrayF(CkMigrateMessage*) { }
  void e() { }
};


#include "pgm.def.h"
