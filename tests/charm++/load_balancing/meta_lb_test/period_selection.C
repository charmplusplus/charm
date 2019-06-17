// This program is designed to artificially test MetaBalancer period
// selection. It creates a chare array which calls AtSync() repeatedly and
// specifies its own custom load so we can control what MetaBalancer sees.

// Every 10 iterations, each array element sets its load to the value of
// CkMyPe(), which will cause load to look imbalanced to MetaBalancer. This
// should cause MetaBalancer to trigger LB. After migration, each element sets
// its load back to balanced until the next imbalanced iteration occurs.

// Because of this we expect MetaBalancer to trigger load balancing once every
// 10 iterations (with some delay as MetaBalancer determines trends).

// NOTE: This test must be run with RotateLB to ensure that when MetaBalancer
// triggers LB, every chare is migrated, and the number of chares per PE remains
// constant.

#include "period_selection.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_TestArray arrayProxy;

#define MAX_ITER 100

class Main : public CBase_Main {
private:
  int iteration;
  int migrations;
public:
  Main(CkArgMsg* msg) : iteration(0), migrations(0) {
    delete msg;
    arrayProxy = CProxy_TestArray::ckNew(CkNumPes() * 8);
    arrayProxy.balance(iteration);
  }

  void resume() {
    CkStartQD(CkCallback(CkIndex_Main::next(), mainProxy));
  }

  void next() {
    iteration++;
    if (iteration < MAX_ITER) {
      arrayProxy.balance(iteration);
    } else {
      if (CkNumPes() > 1 && migrations != 10) {
        CkAbort("Did not do expected number of migrations!\n");
      } else if (CkNumPes() == 1 && migrations > 0) {
        CkAbort("Should not be any migration with one PE!\n");
      }
      CkExit();
    }
  }

  void migrated() {
    migrations++;
    CkPrintf("On iteration %i, migrations done: %i\n", iteration, migrations);
  }
};

class TestArray : public CBase_TestArray {
private:
  int load;
public:
  TestArray() : load(10) {
    usesAtSync = true;
    usesAutoMeasure = false;
  }
  TestArray(CkMigrateMessage* msg) { delete msg; }

  void balance(int iteration) {
    // Cause artificial imbalance every 10 iterations
    // This should trigger MetaBalancer to run the balancer
    if (iteration % 10 == 0) {
      load = CkMyPe();
    }
    AtSync();
  }

  void ResumeFromSync() {
    contribute(CkCallback(CkReductionTarget(Main, resume), mainProxy));
  }

  void resetLoad() {
    load = 10;
  }

  // This is called by the RTS when AtSync is called and ready to do LB
  virtual void UserSetLBLoad() {
    setObjTime(load);
  }

  virtual void pup(PUP::er& p) {
    p | load;

    if (p.isUnpacking()) {
      // When migration occurs, set all loads to balanced to make it appear
      // as though load balancing worked. MetaLB will not trigger again until
      // load becomes imbalanced.
      resetLoad();
      // Inform main that a migration occured
      contribute(CkCallback(CkReductionTarget(Main, migrated), mainProxy));
    }
  }
};

#include "period_selection.def.h"
