// Tests of coroutine entry methods (ckcoro.h). The migration test is the
// coroutine version of tests/charm++/sdag/sdag_tests.C's MigrationTest: its
// case statements are CK_WHEN_ANY sites, its conjunctions are consecutive
// CK_WHEN sites, and its final overlap over stray messages is a check that
// the inbox is empty when the coroutine ends.
#include <random>
#include "ckcoro_tests.decl.h"
#include "ckcoro.h"

#define MIGRATION_ELEMS 64

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_MigrationTest migrationTestProxy;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    delete m;
    mainProxy = thisProxy;
    migrationTestProxy = CProxy_MigrationTest::ckNew(MIGRATION_ELEMS);
    thisProxy.doMigrationTest();
  }

  void doMigrationTest() {
    CkPrintf("Running coroutine migration tests\n");
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
    CkPrintf("Coroutine migration tests complete\n");
    CkExit();
  }
};

class TestMessage : public CMessage_TestMessage {
public:
  int* data;
  int size;
  int pe;
  TestMessage(int s, int p) : size(s), pe(p) {
    for (int i = 0; i < size - 1; i++) data[i] = i;
    data[size - 1] = pe;
  }
  bool validate() {
    for (int i = 0; i < size - 1; i++)
      if (data[i] != i) return false;
    return data[size - 1] == pe;
  }
};

class MigrationTest : public CBase_MigrationTest {
private:
  ck::coro::state co;
  int data;
  bool qd_reached;
  // Scratch, assigned at sites and used after them.
  TestMessage *msg1, *msg2;
  int which;

public:
  MigrationTest() : data(thisIndex), qd_reached(false), msg1(nullptr), msg2(nullptr), which(-1) {}
  MigrationTest(CkMigrateMessage* msg) : CBase_MigrationTest(msg) {}

  void pup(PUP::er& p) {
    p | data;
    p | qd_reached;
    p | co;
    if (p.isUnpacking()) { msg1 = msg2 = nullptr; which = -1; }
  }

  // Inbox entries.
  void sentBefore(TestMessage* m) { co.deliver(CkIndex_MigrationTest::sentBefore(0), m); }
  void sentAfter(TestMessage* m) { co.deliver(CkIndex_MigrationTest::sentAfter(0), m); }
  void call1() { co.deliver(CkIndex_MigrationTest::call1()); }
  void call2() { co.deliver(CkIndex_MigrationTest::call2()); }
  void call3() { co.deliver(CkIndex_MigrationTest::call3()); }
  void response1() { co.deliver(CkIndex_MigrationTest::response1()); }
  void response2() { co.deliver(CkIndex_MigrationTest::response2()); }
  void response3() { co.deliver(CkIndex_MigrationTest::response3()); }
  void doMigration() { co.deliver(CkIndex_MigrationTest::doMigration()); }
  void migrationDone() { co.deliver(CkIndex_MigrationTest::migrationDone()); }

  // migrateMe sends ckEmigrate to the element itself, so the migration
  // happens while run() is suspended at the next site; this entry is the
  // same request from outside the coroutine.
  void migrateNow() { migrateToRandomPE(); }

  void sendBefore() {
    TestMessage* msg = new (thisIndex+1) TestMessage(thisIndex, CkMyPe());
    thisProxy[thisIndex].sentBefore(msg);
  }

  void checkPair() {
    CkAssert(msg1->pe != msg2->pe);
    CkAssert(msg1->pe >= 0 && msg1->pe < CkNumPes());
    CkAssert(msg2->pe >= 0 && msg2->pe < CkNumPes());
    CkAssert(msg1->validate());
    CkAssert(msg2->validate());
    delete msg1;
    delete msg2;
    msg1 = msg2 = nullptr;
  }

  void respond(int w) {
    if (w == 0) thisProxy[thisIndex].response1();
    else if (w == 1) thisProxy[thisIndex].response2();
    else thisProxy[thisIndex].response3();
  }

  ck::coroutine run() {
    CK_COROUTINE(co, CkIndex_MigrationTest::run());
    if (CkNumPes() > 1) {
      sendBefore();

      // Synchronized migration: QD calls doMigration, nothing else happens
      // until migrationDone is called.
      CK_WHEN(CkIndex_MigrationTest::doMigration(), ck::coro::noref);
      qd_reached = true;
      if (thisIndex == 0) CkPrintf("Testing synchronized migration\n");
      migrateToRandomPE();  // deferred: the element moves while run() waits below

      // Called after all elements arrive on their new PEs
      CK_WHEN(CkIndex_MigrationTest::migrationDone(), ck::coro::noref);
      CkAssert(qd_reached);

      // Both TestMessages (an SDAG conjunction: two consecutive sites)
      CK_WHEN(CkIndex_MigrationTest::sentBefore(0), ck::coro::noref, msg1);
      CK_WHEN(CkIndex_MigrationTest::sentAfter(0), ck::coro::noref, msg2);
      checkPair();

      // One of the calls, which was pupped as a buffered message (SDAG case)
      CK_WHEN_ANY(which, CK_ALT(CkIndex_MigrationTest::call1(), ck::coro::noref),
                         CK_ALT(CkIndex_MigrationTest::call2(), ck::coro::noref),
                         CK_ALT(CkIndex_MigrationTest::call3(), ck::coro::noref));
      respond(which);

      // Anytime migration: the element migrates without waiting for QD and
      // this coroutine keeps receiving whenever messages arrive.
      if (thisIndex == 0) CkPrintf("Testing anytime migration\n");
      sendBefore();
      thisProxy[thisIndex].migrateNow();

      CK_WHEN(CkIndex_MigrationTest::sentBefore(0), ck::coro::noref, msg1);
      CK_WHEN(CkIndex_MigrationTest::sentAfter(0), ck::coro::noref, msg2);
      checkPair();

      CK_WHEN_ANY(which, CK_ALT(CkIndex_MigrationTest::call1(), ck::coro::noref),
                         CK_ALT(CkIndex_MigrationTest::call2(), ck::coro::noref),
                         CK_ALT(CkIndex_MigrationTest::call3(), ck::coro::noref));
      respond(which);

      CK_WHEN(CkIndex_MigrationTest::migrationDone(), ck::coro::noref);

      CK_WHEN_ANY(which, CK_ALT(CkIndex_MigrationTest::call1(), ck::coro::noref),
                         CK_ALT(CkIndex_MigrationTest::call2(), ck::coro::noref),
                         CK_ALT(CkIndex_MigrationTest::call3(), ck::coro::noref));
      respond(which);

      // All three responses
      CK_WHEN(CkIndex_MigrationTest::response1(), ck::coro::noref);
      CK_WHEN(CkIndex_MigrationTest::response2(), ck::coro::noref);
      CK_WHEN(CkIndex_MigrationTest::response3(), ck::coro::noref);
    }
    // Every message was consumed (the SDAG test's final overlap).
    CkAssert(co.num_buffered() == 0);
    contribute(CkCallback(CkReductionTarget(Main, migrationTestDone), mainProxy));
    CK_END();
  }

  void migrateToRandomPE() {
    std::default_random_engine generator;
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

#include "ckcoro_tests.def.h"
#include "migration.def.h"
