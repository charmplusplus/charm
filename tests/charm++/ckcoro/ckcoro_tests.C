// Tests of coroutine entry methods (ckcoro.h). The migration test is the
// coroutine version of tests/charm++/sdag/sdag_tests.C's MigrationTest: its
// case statements are CK_WHEN_ANY sites, its conjunctions are consecutive
// CK_WHEN sites, and its final overlap over stray messages is a check that
// the inbox is empty when the coroutine ends. The marshall test checks
// marshalled inbox entries: matched by the first argument, delivered by
// broadcast, and reached through a reduction.
#include <numeric>
#include <random>
#include "ckcoro_tests.decl.h"
#include "ckcoro.h"

#define MIGRATION_ELEMS 64
#define MARSHALL_ELEMS 8
#define MARSHALL_ITERS 10

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
    thisProxy.doMarshallTest();
  }

  void doMarshallTest() {
    CkPrintf("Running marshalled inbox tests\n");
    CProxy_MarshallTest a = CProxy_MarshallTest::ckNew(MARSHALL_ELEMS);
    a.tag(7, 2.5);  // buffered: no coroutine is running yet
    a.run();
  }

  void marshallTestDone() {
    CkPrintf("Marshalled inbox tests complete\n");
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
  MigrationTest_CORO_CODE
private:
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
    if (p.isUnpacking()) { msg1 = msg2 = nullptr; which = -1; }
  }

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
    CK_COROUTINE();
    if (CkNumPes() > 1) {
      sendBefore();

      // Synchronized migration: QD calls doMigration, nothing else happens
      // until migrationDone is called.
      CK_WHEN(inbox::doMigration());
      qd_reached = true;
      if (thisIndex == 0) CkPrintf("Testing synchronized migration\n");
      migrateToRandomPE();  // deferred: the element moves while run() waits below

      // Called after all elements arrive on their new PEs
      CK_WHEN(inbox::migrationDone());
      CkAssert(qd_reached);

      // Both TestMessages (an SDAG conjunction: two consecutive sites)
      CK_WHEN(inbox::sentBefore(), msg1);
      CK_WHEN(inbox::sentAfter(), msg2);
      checkPair();

      // One of the calls, which was pupped as a buffered message (SDAG case)
      CK_WHEN_ANY(which, CK_ALT(inbox::call1()), CK_ALT(inbox::call2()), CK_ALT(inbox::call3()));
      respond(which);

      // Anytime migration: the element migrates without waiting for QD and
      // this coroutine keeps receiving whenever messages arrive.
      if (thisIndex == 0) CkPrintf("Testing anytime migration\n");
      sendBefore();
      thisProxy[thisIndex].migrateNow();

      CK_WHEN(inbox::sentBefore(), msg1);
      CK_WHEN(inbox::sentAfter(), msg2);
      checkPair();

      CK_WHEN_ANY(which, CK_ALT(inbox::call1()), CK_ALT(inbox::call2()), CK_ALT(inbox::call3()));
      respond(which);

      CK_WHEN(inbox::migrationDone());

      CK_WHEN_ANY(which, CK_ALT(inbox::call1()), CK_ALT(inbox::call2()), CK_ALT(inbox::call3()));
      respond(which);

      // All three responses
      CK_WHEN(inbox::response1());
      CK_WHEN(inbox::response2());
      CK_WHEN(inbox::response3());
    }
    // Every message was consumed (the SDAG test's final overlap).
    CkAssert(__coro.num_buffered() == 0);
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

// Marshalled inbox entries. Each element sends its right neighbour one
// message per iteration, tagged with the iteration, and waits for its own;
// the broadcast tag() arrived before run() and waits in the inbox; the
// reduction total() reaches the inbox through the reduction wrapper.
class MarshallTest : public CBase_MarshallTest {
  MarshallTest_CORO_CODE
  int iter;
  // Scratch, assigned at sites and used after them: a when stores every
  // parameter of the entry, in order, the reference number included.
  int t, i;
  std::vector<int> values;
  double x;
  int sum;

  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro marshall test: %s", what);
  }

public:
  MarshallTest() : iter(0), t(0), i(0), x(0), sum(0) {}
  MarshallTest(CkMigrateMessage* m) : CBase_MarshallTest(m) {}

  void pup(PUP::er& p) {
    p | iter;
    if (p.isUnpacking()) { values.clear(); t = i = 0; x = 0; sum = 0; }
  }

  ck::coroutine run() {
    CK_COROUTINE();
    CK_WHEN(inbox::tag(7), t, x);
    check(t == 7 && x == 2.5, "broadcast tag(7, 2.5) not received");
    // Everything ahead of time, so iteration k's message is buffered when
    // iteration k-1 is waited for.
    for (iter = 0; iter < MARSHALL_ITERS; iter++) {
      std::vector<int> v(iter + 1);
      std::iota(v.begin(), v.end(), thisIndex);
      thisProxy[(thisIndex + 1) % MARSHALL_ELEMS].recv(iter, v);
    }
    for (iter = MARSHALL_ITERS - 1; iter >= 0; iter--) {
      CK_WHEN(inbox::recv(iter), i, values);
      check(i == iter, "recv matched by the wrong iteration");
      check((int)values.size() == iter + 1, "recv: wrong vector length");
      check(values[0] == (thisIndex + MARSHALL_ELEMS - 1) % MARSHALL_ELEMS, "recv: wrong sender");
      check(values.back() == values[0] + iter, "recv: wrong vector contents");
    }
    contribute(sizeof(int), &thisIndex, CkReduction::sum_int,
               CkCallback(CkReductionTarget(MarshallTest, total), thisProxy));
    CK_WHEN(inbox::total(), sum);
    check(sum == MARSHALL_ELEMS * (MARSHALL_ELEMS - 1) / 2, "total: wrong reduction result");
    check(__coro.num_buffered() == 0, "messages left in the inbox");
    contribute(CkCallback(CkReductionTarget(Main, marshallTestDone), mainProxy));
    CK_END();
  }
};

#include "ckcoro_tests.def.h"
#include "migration.def.h"
