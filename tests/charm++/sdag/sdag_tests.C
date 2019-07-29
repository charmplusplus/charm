#include "sdag_tests.decl.h"

#include <random>

#define MIGRATION_ELEMS 64
#define REFNUM_ELEMS 8

#define FOR_ITERATIONS 16
#define REFNUM_ITERATIONS 20

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_BasicTest basicTestProxy;
/* readonly */ CProxy_RefnumTest refnumTestProxy;
/* readonly */ CProxy_MigrationTest migrationTestProxy;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    delete m;
    mainProxy = thisProxy;
    basicTestProxy = CProxy_BasicTest::ckNew(CkNumPes() - 1);
    refnumTestProxy = CProxy_RefnumTest::ckNew(REFNUM_ELEMS, REFNUM_ELEMS);
    migrationTestProxy = CProxy_MigrationTest::ckNew(MIGRATION_ELEMS);

    thisProxy.doBasicTest();
  }

  void doBasicTest() {
    CkPrintf("Running basic SDAG tests\n");
    for (int i = 0; i < FOR_ITERATIONS / 3; i++) {
      basicTestProxy.fizz();
    }
    for (int i = 0; i < FOR_ITERATIONS / 5; i++) {
      basicTestProxy.buzz();
    }
    basicTestProxy.call1();
    basicTestProxy.call2();
    basicTestProxy.run();
  }

  void basicTestDone() {
    CkPrintf("Basic SDAG tests complete\n");
    thisProxy.doRefnumTest();
  }

  void doRefnumTest() {
    CkPrintf("Running refnum SDAG tests\n");
    refnumTestProxy.run(REFNUM_ITERATIONS);
  }

  void refnumTestDone() {
    CkPrintf("Refnum SDAG tests complete\n");
    thisProxy.doMigrationTest();
  }

  void doMigrationTest() {
    CkPrintf("Running SDAG migration tests\n");
    if (CkNumPes() > 1) {
      migrationTestProxy.call1();
      
      for (int i = 0; i < MIGRATION_ELEMS; i++) {
        migrationTestProxy[i].call2();
        migrationTestProxy[i].call3();
      }

      migrationTestProxy.run();
      CkStartQD(CkCallback(CkIndex_MigrationTest::doMigration(), migrationTestProxy));
    } else {
      thisProxy.migrationTestDone();
    }
  }

  void migrationTestDone() {
    CkPrintf("SDAG migration tests complete\n");
    CkExit();
  }
};

class BasicTest : public CBase_BasicTest {
private:
  BasicTest_SDAG_CODE

  int for_index;
  bool do_while, unlocked1, unlocked2, unlocked3;
public:
  BasicTest()
      : do_while(true), unlocked1(false), unlocked2(false), unlocked3(false) {
    CkPrintf("BasicTest created on PE %i\n", CkMyPe());
  }
};

class RefnumTest : public CBase_RefnumTest {
private:
  RefnumTest_SDAG_CODE
  int array_size;
  int refnum_sum, expected_refnum_sum;
  int marshall_sum, expected_marshall_sum;
  int redn_sum, expected_redn_sum;
  int cb_sum, expected_cb_sum;
  int ignored_sum, expected_ignored_sum;
  int index;
public:
  RefnumTest(int s) : array_size(s) {}
};

class TestMessage : public CMessage_TestMessage {
public:
  int* data;
  int size;
  int pe;
  TestMessage(int s, int p) : size(s), pe(p) {
    for (int i = 0; i < size - 1; i++) {
      data[i] = i;
    }
    data[size - 1] = pe;
  }
  bool validate() {
    bool result = true;
    for (int i = 0; i < size - 1; i++) {
      if (data[i] != i) {
        result = false;
        break;
      }
    }
    return result && data[size - 1] == pe;
  }
};

class MigrationTest : public CBase_MigrationTest {
private:
  MigrationTest_SDAG_CODE
  int data;
  bool qd_reached;

public:
  MigrationTest() : data(thisIndex), qd_reached(false) {}
  MigrationTest(CkMigrateMessage* msg) {}

  void pup(PUP::er& p) {
    p | data;
    p | qd_reached;
  }

  void migrateToRandomPE() {
    std::default_random_engine generator;
    // CkNumPes() - 2 because if we hit ourself we go to last PE
    std::uniform_int_distribution<int> distribution(0, CkNumPes() - 2);

    int migrateToPe = distribution(generator);
    if (migrateToPe == CkMyPe()) migrateToPe = CkNumPes() - 1;
    migrateMe(migrateToPe);
  }

  void ckJustMigrated() {
    contribute(CkCallback(CkReductionTarget(MigrationTest, migrationDone), thisProxy));
    TestMessage* msg = new (thisIndex+1) TestMessage(thisIndex, CkMyPe());
    thisProxy[thisIndex].sentAfter(msg);
  }
};

#include "sdag_tests.def.h"
#include "basic.def.h"
#include "refnum.def.h"
#include "migration.def.h"
