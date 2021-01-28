// This program is designed to artificially test MetaBalancer period
// selection. It creates a chare array which calls AtSync() repeatedly and
// specifies its own custom load so we can control what MetaBalancer sees.

// Every ITER_MOD iterations, each array element sets its load to the value of
// CkMyPe(), which will cause load to look imbalanced to MetaBalancer. This
// should cause MetaBalancer to trigger LB. After migration, each element sets
// its load back to balanced until the next imbalanced iteration occurs.

// Because of this we expect MetaBalancer to trigger load balancing once every
// ITER_MOD iterations (with some delay as MetaBalancer determines trends).

// NOTE: This test must be run with RotateLB to ensure that when MetaBalancer
// triggers LB, every chare is migrated, and the number of chares per PE remains
// constant.

#include "period_selection.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_TestArray arrayProxy;

#define MAX_ITER 100
#define ITER_MOD 10
#define DEFAULT_LOAD 32
#define OBJS_PER_PE 8

class Main : public CBase_Main {
private:
  int iteration;
  int migrations;
public:
  Main(CkArgMsg* msg) : iteration(0), migrations(0) {
    delete msg;
    // Test currently requires an equal number of objects on every PE
    // TODO: Make a test that works with some empty PEs
    arrayProxy = CProxy_TestArray::ckNew(CkNumPes() * OBJS_PER_PE);
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
      int expected_migrations = MAX_ITER / ITER_MOD;
      if (MAX_ITER % ITER_MOD == 0) {
        expected_migrations--;
      }
      if (CkNumPes() == 1) {
        expected_migrations = 0;
      }
      if (migrations != expected_migrations) {
        CkAbort("Did not do expected number of migrations!\n");
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
  TestArray() : load(DEFAULT_LOAD) {
    usesAtSync = true;
    usesAutoMeasure = false;
  }
  TestArray(CkMigrateMessage* msg) { delete msg; }

  void balance(int iteration) {
    // Cause artificial imbalance every ITER_MOD iterations
    // This should trigger MetaBalancer to run the balancer
    if (iteration && iteration % ITER_MOD == 0) {
      load = CkMyPe();
    }
    AtSync();
  }

  void ResumeFromSync() {
    contribute(CkCallback(CkReductionTarget(Main, resume), mainProxy));
  }

  void resetLoad() {
    load = DEFAULT_LOAD;
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
