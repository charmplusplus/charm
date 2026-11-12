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
#define REFNUM_ELEMS 8
#define REFNUM_ITERATIONS 100
#define FOR_ITERATIONS 16
#define MARSHALL_ELEMS 8
#define MARSHALL_ITERS 10
#define OVERLAP_ELEMS 6
#define OVERLAP_ROUNDS 4
#define OVERLAP_ITEMS 5
#define OWNER_ROUNDS 3
#define OWNER_ITEMS 7

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_MigrationTest migrationTestProxy;
/* readonly */ CProxy_BasicTest basicTestProxy;
/* readonly */ CProxy_RefnumTest refnumTestProxy;

class Main : public CBase_Main {
  Main_CORO_CODE
  // The owner tests' proxies and results: pupped, the checkpoint restores them.
  CProxy_GroupTest group;
  CProxy_NodeGroupTest ngroup;
  CProxy_ChareTest chare;
  std::string chkptDir;
  int gsum, nsum, csum;

  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro owner test: %s", what);
  }

public:
  Main(CkArgMsg* m) {
    // -chkpt DIR: checkpoint to DIR during the owner tests; +restart DIR resumes there.
    for (int i = 1; i + 1 < m->argc; i++)
      if (strcmp(m->argv[i], "-chkpt") == 0) chkptDir = m->argv[i + 1];
    delete m;
    mainProxy = thisProxy;
    basicTestProxy = CProxy_BasicTest::ckNew(1);
    refnumTestProxy = CProxy_RefnumTest::ckNew(REFNUM_ELEMS, REFNUM_ELEMS);
    migrationTestProxy = CProxy_MigrationTest::ckNew(MIGRATION_ELEMS);
    thisProxy.doBasicTest();
  }

  void doBasicTest() {
    CkPrintf("Running basic coroutine tests\n");
    for (int i = 0; i < FOR_ITERATIONS / 3; i++) basicTestProxy.fizz();
    for (int i = 0; i < FOR_ITERATIONS / 5; i++) basicTestProxy.buzz();
    basicTestProxy.call1();
    basicTestProxy.call2();
    basicTestProxy.run();
  }

  void basicTestDone() {
    CkPrintf("Basic coroutine tests complete\n");
    thisProxy.doRefnumTest();
  }

  void doRefnumTest() {
    CkPrintf("Running refnum coroutine tests\n");
    refnumTestProxy.run(REFNUM_ITERATIONS);
  }

  void refnumTestDone() {
    CkPrintf("Refnum coroutine tests complete\n");
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
    thisProxy.doOverlapTest();
  }

  void doOverlapTest() {
    CkPrintf("Running overlap tests\n");
    CProxy_OverlapTest a = CProxy_OverlapTest::ckNew(OVERLAP_ELEMS);
    std::vector<int> seeds(OVERLAP_ROUNDS);
    for (int r = 0; r < OVERLAP_ROUNDS; r++) seeds[r] = 1000 + 7 * r;
    double weights[OVERLAP_ITEMS];
    for (int i = 0; i < OVERLAP_ITEMS; i++) weights[i] = 0.5 * i;
    a.run(OVERLAP_ROUNDS, seeds, OVERLAP_ITEMS, weights);
  }

  void overlapTestDone() {
    CkPrintf("Overlap tests complete\n");
    thisProxy.runOwners(OWNER_ROUNDS);
  }

  Main(CkMigrateMessage* m) : CBase_Main(m) {}
  void pup(PUP::er& p) {
    p | group; p | ngroup; p | chare; p | chkptDir;
    p | gsum; p | nsum; p | csum;
  }

  // The mainchare's own coroutine entry: starts the three owners, waits for
  // all of them to be ready (a conjunction of two reduction targets and a
  // plain inbox entry), checkpoints with every owner's coroutine suspended,
  // resumes them and collects their results. rounds is re-supplied by a
  // restart.
  ck::coroutine runOwners(int rounds) {
    CK_COROUTINE();
    CkPrintf("Running owner tests (group, nodegroup, chare, mainchare)\n");
    ngroup = CProxy_NodeGroupTest::ckNew();
    group = CProxy_GroupTest::ckNew();
    chare = CProxy_ChareTest::ckNew(OWNER_ITEMS);
    for (int i = OWNER_ITEMS - 1; i >= 0; i--) chare.item(i, i * i);
    group.run(rounds, ngroup);
    ngroup.run(rounds);
    chare.run();
    CK_WHEN_ALL(CK_ALT(inbox::groupReady()), CK_ALT(inbox::nodeGroupReady()),
                CK_ALT(inbox::chareReady()));
    if (!chkptDir.empty()) {
      CkPrintf("Checkpoint to %s with the owners' coroutines suspended\n", chkptDir.c_str());
      CkStartCheckpoint(chkptDir.c_str(), CkCallback(CkIndex_Main::checkpointed(), thisProxy));
      // After the checkpoint in this run, and the first message of a run
      // restarted from it.
      CK_WHEN(inbox::checkpointed());
      CkPrintf("Resuming after the checkpoint on %d PEs\n", CkNumPes());
    }
    group.resume();
    ngroup.resume();
    chare.resume();
    CK_WHEN_ALL(CK_ALT(inbox::groupDone(), gsum), CK_ALT(inbox::nodeGroupDone(), nsum),
                CK_ALT(inbox::chareDone(), csum));
    {
      int p = CkNumPes();
      int expected = rounds * p * (p - 1) / 2;   // every PE receives from its predecessor
      check(gsum == expected, "group sum");
      check(nsum == expected, "nodegroup sum");  // the same tokens, by node
      int items = 0;
      for (int i = 0; i < OWNER_ITEMS; i++) items += i * i;
      check(csum == items, "chare sum");
    }
    CkPrintf("Owner tests complete\n");
    CkExit();
    CK_COROUTINE_END();
  }
};

class GroupTest : public CBase_GroupTest {
  GroupTest_CORO_CODE
  int round, from, sum;
  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro owner test (group): %s", what);
  }

public:
  GroupTest() : round(0), from(-1), sum(0) {}
  GroupTest(CkMigrateMessage* m) : CBase_GroupTest(m) {}
  void pup(PUP::er& p) { p | round; p | from; p | sum; }

  ck::coroutine run(int rounds, CProxy_NodeGroupTest ng) {
    CK_COROUTINE();
    for (round = 0; round < rounds; round++) {
      thisProxy[(CkMyPe() + 1) % CkNumPes()].token(round, CkMyPe());
      ng[CkMyNode()].token(round, CkMyPe());
      CK_WHEN(inbox::token(round), ck::coro::skip, from);
      check(from == (CkMyPe() + CkNumPes() - 1) % CkNumPes(), "token from the wrong PE");
      sum += from;
    }
    contribute(CkCallback(CkReductionTarget(Main, groupReady), mainProxy));
    CK_WHEN(inbox::resume());
    contribute(sizeof(int), &sum, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, groupDone), mainProxy));
    CK_COROUTINE_END();
  }
};

class NodeGroupTest : public CBase_NodeGroupTest {
  NodeGroupTest_CORO_CODE
  int round, count, from, sum;

public:
  NodeGroupTest() : round(0), count(0), from(-1), sum(0) {}
  NodeGroupTest(CkMigrateMessage* m) : CBase_NodeGroupTest(m) {}
  void pup(PUP::er& p) { p | round; p | count; p | from; p | sum; }

  ck::coroutine run(int rounds) {
    CK_COROUTINE();
    for (round = 0; round < rounds; round++) {
      for (count = 0; count < CkNodeSize(CkMyNode()); count++) {
        CK_WHEN(inbox::token(round), ck::coro::skip, from);
        sum += from;
      }
    }
    contribute(CkCallback(CkReductionTarget(Main, nodeGroupReady), mainProxy));
    CK_WHEN(inbox::resume());
    contribute(sizeof(int), &sum, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, nodeGroupDone), mainProxy));
    CK_COROUTINE_END();
  }
};

class ChareTest : public CBase_ChareTest {
  ChareTest_CORO_CODE
  int n, i, v, sum;

public:
  ChareTest(int n_) : n(n_), i(0), v(0), sum(0) {}
  ChareTest(CkMigrateMessage* m) : CBase_ChareTest(m) {}
  void pup(PUP::er& p) { p | n; p | i; p | v; p | sum; }

  ck::coroutine run() {
    CK_COROUTINE();
    for (i = 0; i < n; i++) {
      CK_WHEN(inbox::item(i), ck::coro::skip, v);
      sum += v;
    }
    mainProxy.chareReady();
    CK_WHEN(inbox::resume());
    mainProxy.chareDone(sum);
    ckDestroyLater();   // destroyed once the entry method running this body returns
    CK_COROUTINE_END();
  }
};

class BasicTest : public CBase_BasicTest {
  BasicTest_CORO_CODE
  int for_index, which;
  bool do_while, unlocked1, unlocked2, unlocked3;

  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro basic test: %s", what);
  }
  // overlap { when unlock1() serial { unlocked1 = true; } ... }
  ck::coro::task<void> waitUnlock1() { CK_CORO_TASK(); CK_WHEN(inbox::unlock1()); unlocked1 = true; CK_CORO_TASK_END(); }
  ck::coro::task<void> waitUnlock2() { CK_CORO_TASK(); CK_WHEN(inbox::unlock2()); unlocked2 = true; CK_CORO_TASK_END(); }
  ck::coro::task<void> waitUnlock3() { CK_CORO_TASK(); CK_WHEN(inbox::unlock3()); unlocked3 = true; CK_CORO_TASK_END(); }
  // The final overlap over stray messages: each arm aborts if reached.
  ck::coro::task<void> extra(ck::coro::want w, const char* name) {
    CK_CORO_TASK();
    CK_WHEN(w);
    ::CkAbort("ckcoro basic test: extra %s", name);
    CK_CORO_TASK_END();
  }

public:
  BasicTest() : for_index(0), which(-1), do_while(true), unlocked1(false), unlocked2(false), unlocked3(false) {
    CkPrintf("BasicTest created on PE %i\n", CkMyPe());
  }
  BasicTest(CkMigrateMessage* m) : CBase_BasicTest(m) {}
  void pup(PUP::er& p) { p | for_index; p | which; p | do_while; p | unlocked1; p | unlocked2; p | unlocked3; }

  ck::coroutine run() {
    CK_COROUTINE();
    thisProxy.unlock1();

    // FizzBuzz: for loops and if/else; Main sent the fizz and buzz messages.
    CkPrintf("Testing coroutine for loops, and if/else\n");
    for (for_index = 1; for_index <= FOR_ITERATIONS; for_index++) {
      if (for_index % 3 == 0 && for_index % 5 == 0) {
        CK_WHEN_ALL(CK_ALT(inbox::fizz()), CK_ALT(inbox::buzz()));   // when fizz(), buzz()
        CkPrintf("FizzBuzz\n");
      } else if (for_index % 3 == 0) {
        CK_WHEN(inbox::fizz());
        CkPrintf("Fizz\n");
      } else if (for_index % 5 == 0) {
        CK_WHEN(inbox::buzz());
        CkPrintf("Buzz\n");
      } else {
        CkPrintf("%i\n", for_index);
      }
    }
    check(for_index == FOR_ITERATIONS + 1, "for loop did not run to the end");
    thisProxy.unlock2();

    // While loops and case statements. The arms with never() can never be
    // taken; the case must not get stuck on them.
    CkPrintf("Testing coroutine while loops and case statements\n");
    while (do_while) {
      CK_WHEN_ANY(which,
                  CK_ALL(CK_ALT(inbox::call1()), CK_ALT(inbox::never())),
                  CK_ALL(CK_ALT(inbox::call2()), CK_ALT(inbox::never())),
                  CK_ALT(inbox::call1()),
                  CK_ALT(inbox::call2()),
                  CK_ALL(CK_ALT(inbox::call1()), CK_ALT(inbox::call2())),
                  CK_ALL(CK_ALT(inbox::response1()), CK_ALT(inbox::response2())),
                  CK_ALT(inbox::responseBoth()));
      if (which == 0 || which == 1) ::CkAbort("Received never()\n");
      else if (which == 2) thisProxy.response1();
      else if (which == 3) thisProxy.response2();
      else if (which == 4) thisProxy.responseBoth();
      else do_while = false;
    }
    check(do_while == false, "while loop did not end");
    thisProxy.unlock3();

    // Overlap: all three unlock messages, in any order.
    CkPrintf("Testing coroutine overlap statements\n");
    CK_OVERLAP(waitUnlock1(), waitUnlock2(), waitUnlock3());
    check(unlocked1 && unlocked2 && unlocked3, "overlap missed an unlock");
    mainProxy.basicTestDone();

    // Any unhandled message is a failure.
    CK_OVERLAP(extra(inbox::fizz(), "fizz"), extra(inbox::buzz(), "buzz"),
               extra(inbox::call1(), "call1"), extra(inbox::call2(), "call2"),
               extra(inbox::response1(), "response1"), extra(inbox::response2(), "response2"),
               extra(inbox::responseBoth(), "responseBoth"), extra(inbox::unlock1(), "unlock1"),
               extra(inbox::unlock2(), "unlock2"), extra(inbox::unlock3(), "unlock3"),
               extra(inbox::never(), "never"));
    CK_COROUTINE_END();
  }
};

class RefnumTest : public CBase_RefnumTest {
  RefnumTest_CORO_CODE
  int array_size;
  int refnum_sum, expected_refnum_sum;
  int marshall_sum, expected_marshall_sum;
  int redn_sum, expected_redn_sum;
  int cb_sum, expected_cb_sum;
  int ignored_sum, expected_ignored_sum;
  int index, got;   // got: the value a task returns through CK_CALL_TO

  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro refnum test: %s", what);
  }
  // case { when recvNum[index](int n) {...} when recvData[index](CkDataMsg* m) {...} }
  ck::coro::task<void> receive(int idx) {
    CK_CORO_TASK();
    int which;
    int n;
    CkDataMsg* m;
    CK_WHEN_ANY(which, CK_ALT(inbox::recvNum(idx), n), CK_ALT(inbox::recvData(idx), m));
    if (which == 0) {
      refnum_sum += idx;
      if (n > 0) marshall_sum += n;
      else redn_sum += n;
    } else {
      refnum_sum += idx;
      cb_sum += ((int*)(m->getData()))[0];
      delete m;
    }
    CK_CORO_TASK_END();
  }
  // forall [index] (0:messages, 2) { if (index & (index - 1)) case {...} else serial {...} }
  ck::coro::task<void> evenIndex(int idx) {
    CK_CORO_TASK();
    if (idx & (idx - 1)) {
      CK_CALL(receive(idx));
    } else {
      ignored_sum += idx;
    }
    CK_CORO_TASK_END();
  }
  // A helper that returns a value: co_return after CK_CORO_TASK_END.
  ck::coro::task<int> receiveIndex(int idx) {
    CK_CORO_TASK();
    CK_CALL(receive(idx));
    CK_CORO_TASK_END();
    co_return idx;
  }
  ck::coro::task<void> missed(ck::coro::want w, const char* name) {
    CK_CORO_TASK();
    CK_WHEN(w);
    ::CkAbort("ckcoro refnum test: missed a %s message", name);
    CK_CORO_TASK_END();
  }

public:
  RefnumTest(int s) : array_size(s) {}
  RefnumTest(CkMigrateMessage* m) : CBase_RefnumTest(m) {}
  void pup(PUP::er& p) {
    p | array_size; p | refnum_sum; p | expected_refnum_sum; p | marshall_sum; p | expected_marshall_sum;
    p | redn_sum; p | expected_redn_sum; p | cb_sum; p | expected_cb_sum; p | ignored_sum;
    p | expected_ignored_sum; p | index; p | got;
  }

  ck::coroutine run(int messages) {
    CK_COROUTINE();
    {
      // The first third of the messages are marshalled, the second third
      // come through reductions, the last third through callbacks.
      int marshall_start = 0;
      int redn_start = messages / 3;
      int cb_start = 2 * messages / 3;

      refnum_sum = 0;
      expected_refnum_sum = (messages * (messages + 1)) / 2;

      marshall_sum = 0;
      expected_marshall_sum = (redn_start * (redn_start - 1)) / 2;
      for (int i = marshall_start; i < redn_start; i++) thisProxy[thisIndex].recvNum(i);

      // A reduction sends -1 as the data.
      redn_sum = 0;
      expected_redn_sum = -1 * array_size * (cb_start - redn_start);
      for (int i = redn_start; i < cb_start; i++) {
        CkCallback cb(CkReductionTarget(RefnumTest, recvNum), thisProxy);
        cb.setRefNum(i);
        int data = -1;
        contribute(sizeof(int), &data, CkReduction::sum_int, cb);
      }

      // The data messages contain our own index.
      cb_sum = 0;
      expected_cb_sum = thisIndex * (messages - cb_start + 1);
      for (int i = cb_start; i <= messages; i++) {
        CkCallback cb(CkIndex_RefnumTest::recvData(NULL), thisProxy[thisIndex]);
        cb.setRefNum(i);
        int data = thisIndex;
        cb.send(sizeof(int), &data);
      }

      // The first forall ignores powers of two.
      ignored_sum = 0;
      expected_ignored_sum = 0;
      for (int i = 2; i < messages; i *= 2) expected_ignored_sum += i;
    }

    // Even reference numbers, powers of two skipped: a positive stride and
    // a branch on the index inside the forall body.
    CK_FORALL_STRIDE(idx, 0, messages, 2, evenIndex(idx));

    // The powers of two, with a plain for loop.
    for (index = 1; index < messages; index *= 2) {
      CK_CALL_TO(got, receiveIndex(index));
      check(got == index, "CK_CALL_TO: wrong value returned");
    }

    // The odd ones except 1: a negative stride that does not land on the end.
    CK_FORALL_STRIDE(idx, messages + (messages % 2 - 1), 2, -2, receive(idx));

    // A single reference number (0 is in no loop above).
    CK_WHEN(inbox::recvNum(0), ck::coro::skip);

    check(refnum_sum == expected_refnum_sum, "refnum sum");
    check(marshall_sum == expected_marshall_sum, "marshalled sum");
    check(redn_sum == expected_redn_sum, "reduction sum");
    check(cb_sum == expected_cb_sum, "callback sum");
    check(ignored_sum == expected_ignored_sum, "ignored sum");
    contribute(CkCallback(CkReductionTarget(Main, refnumTestDone), mainProxy));

    // Every message was received exactly once above: these never trigger.
    CK_OVERLAP(missed(inbox::recvNum(), "recvNum"), missed(inbox::recvData(), "recvData"));
    CK_COROUTINE_END();
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
  bool finished;   // the test is over: ckJustMigrated contributes no more
  // Scratch, assigned at sites and used after them.
  TestMessage *msg1, *msg2;
  int which;

public:
  MigrationTest() : data(thisIndex), qd_reached(false), finished(false), msg1(nullptr), msg2(nullptr), which(-1) {}
  MigrationTest(CkMigrateMessage* msg) : CBase_MigrationTest(msg) {}

  void pup(PUP::er& p) {
    p | data;
    p | qd_reached;
    p | finished;
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
    finished = true;
    contribute(CkCallback(CkReductionTarget(Main, migrationTestDone), mainProxy));
    CK_COROUTINE_END();
  }

  void migrateToRandomPE() {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, CkNumPes() - 2);
    int migrateToPe = distribution(generator);
    if (migrateToPe == CkMyPe()) migrateToPe = CkNumPes() - 1;
    migrateMe(migrateToPe);
  }

  // Not once the test is over: the owner tests take a checkpoint after it,
  // and the restart restores the elements through the migration path, where
  // this contribution would have no reduction to join.
  void ckJustMigrated() {
    if (finished) return;
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
  // Array parameters: a pointer out borrows the message until the next
  // wait; a container out copies.
  int n, n2;
  double* darr;
  double* darr2;   // a conjunction's second array, borrowed alongside the first
  std::vector<double> dvec;
  std::vector<int> tail;

  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro marshall test: %s", what);
  }

public:
  MarshallTest() : iter(0), t(0), i(0), x(0), sum(0) {}
  MarshallTest(CkMigrateMessage* m) : CBase_MarshallTest(m) {}

  void pup(PUP::er& p) {
    p | iter;
    if (p.isUnpacking()) { values.clear(); t = i = 0; x = 0; sum = 0; n = 0; darr = nullptr; dvec.clear(); tail.clear(); }
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

    // Array parameters, five rounds sent ahead of time (round r, iteration
    // k has reference number 100 r + k): received into a pointer, into a
    // vector with the other parameters skipped, with the block form, and
    // two rounds at once by a conjunction with both arrays borrowed.
    for (iter = 0; iter < 4 * 100 + MARSHALL_ITERS; iter++) {
      if (iter % 100 >= MARSHALL_ITERS) continue;
      std::vector<double> d(iter % 100 + 2);
      for (size_t k = 0; k < d.size(); k++) d[k] = thisIndex * 100 + iter + k;
      std::vector<int> tl = {iter, (int)d.size()};
      thisProxy[(thisIndex + 1) % MARSHALL_ELEMS].arr(iter, (int)d.size(), d.data(), tl);
    }
    for (iter = 0; iter < MARSHALL_ITERS; iter++) {
      CK_WHEN(inbox::arr(iter), i, n, darr, tail);
      check(i == iter && n == iter + 2, "arr: wrong iteration or length");
      check(darr[0] == (double)((thisIndex + MARSHALL_ELEMS - 1) % MARSHALL_ELEMS * 100 + iter), "arr: wrong data[0]");
      check(darr[n - 1] == darr[0] + n - 1, "arr: wrong data[n-1]");
      check(tail.size() == 2 && tail[0] == iter && tail[1] == n, "arr: wrong tail");
    }
    for (iter = 100; iter < 100 + MARSHALL_ITERS; iter++) {
      CK_WHEN(inbox::arr(iter), ck::coro::skip, n, dvec, ck::coro::skip);
      check(n == iter % 100 + 2 && (int)dvec.size() == n, "arr: wrong vector length");
      check(dvec[0] == (double)((thisIndex + MARSHALL_ELEMS - 1) % MARSHALL_ELEMS * 100 + iter), "arr: wrong vector data[0]");
      check(dvec[n - 1] == dvec[0] + n - 1, "arr: wrong vector data[n-1]");
    }
    for (iter = 200; iter < 200 + MARSHALL_ITERS; iter++) {
      CK_WHEN_BLOCK(inbox::arr(iter), it, len, data, tl) {
        check(it == iter && len == iter % 100 + 2, "arr block: wrong iteration or length");
        check(data[0] == (double)((thisIndex + MARSHALL_ELEMS - 1) % MARSHALL_ELEMS * 100 + iter), "arr block: wrong data[0]");
        check(data[len - 1] == data[0] + len - 1, "arr block: wrong data[n-1]");
        check(tl.size() == 2 && tl[0] == iter && tl[1] == len, "arr block: wrong tail");
      }
    }
    for (iter = 300; iter < 300 + MARSHALL_ITERS; iter++) {
      CK_WHEN_ALL(CK_ALT(inbox::arr(iter), i, n, darr, tail),
                  CK_ALT(inbox::arr(iter + 100), ck::coro::skip, n2, darr2, ck::coro::skip));
      check(i == iter && n == iter % 100 + 2 && n2 == n, "arr all: wrong iteration or lengths");
      check(darr[0] == (double)((thisIndex + MARSHALL_ELEMS - 1) % MARSHALL_ELEMS * 100 + iter), "arr all: wrong data[0]");
      check(darr2[0] == darr[0] + 100, "arr all: wrong second data[0]");
      check(darr[n - 1] == darr[0] + n - 1 && darr2[n - 1] == darr2[0] + n - 1, "arr all: wrong data[n-1]");
    }
    // A reduction target with an array parameter, reached through the
    // reduction wrapper: the arguments are packed into a message.
    {
      double part[3] = {1.0 * thisIndex, 2.0 * thisIndex, 1.0};
      contribute(sizeof(part), part, CkReduction::sum_double,
                 CkCallback(CkReductionTarget(MarshallTest, arrTotal), thisProxy));
    }
    CK_WHEN(inbox::arrTotal(), n, dvec);
    check(n == 3 && dvec.size() == 3, "arrTotal: wrong length");
    check(dvec[0] == sum && dvec[1] == 2 * sum && dvec[2] == MARSHALL_ELEMS, "arrTotal: wrong sums");
    // The block form on a message entry's twin: the broadcast tag again.
    thisProxy[thisIndex].tag(9, 4.5);
    CK_WHEN_BLOCK(inbox::tag(9), t9, x9) {
      check(t9 == 9 && x9 == 4.5, "tag block: wrong values");
    }
    check(__coro.num_buffered() == 0, "messages left in the inbox");
    contribute(CkCallback(CkReductionTarget(Main, marshallTestDone), mainProxy));
    CK_COROUTINE_END();
  }
};

// Each round, the left neighbour sends b, then a, then the items; the
// overlap waits for a and b in either order, the forall for the items by
// index. The entry's parameters are re-supplied by the restart; weights
// points into the entry's message, which the state keeps.
class OverlapTest : public CBase_OverlapTest {
  OverlapTest_CORO_CODE
  int round;                 // live across waits
  // Written by the children and read by the parent after the overlap: a
  // finished child is not run again after a migration, so its results are
  // pupped members like any value live across a wait.
  int gotA, gotB;
  double itemSum;
  std::string order;         // 'a'/'b' per round, in completion order

  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro overlap test: %s", what);
  }
  int left() const { return (thisIndex + OVERLAP_ELEMS - 1) % OVERLAP_ELEMS; }

public:
  OverlapTest() : round(0), gotA(0), gotB(0), itemSum(0) {}
  OverlapTest(CkMigrateMessage* m) : CBase_OverlapTest(m) {}
  void pup(PUP::er& p) { p | round; p | order; p | gotA; p | gotB; p | itemSum; }

  ck::coro::task<void> waitA() {
    CK_CORO_TASK();
    CK_WHEN(inbox::a(round), ck::coro::skip, gotA);
    order += 'a';
    CK_CORO_TASK_END();
  }
  ck::coro::task<void> waitB() {
    CK_CORO_TASK();
    CK_WHEN(inbox::b(round), ck::coro::skip, gotB);
    order += 'b';
    CK_CORO_TASK_END();
  }
  // Matched by round and index together: after a migration, a neighbour's
  // messages for the next round may be delivered before this round's.
  ck::coro::task<void> waitItem(int i, double w) {
    CK_CORO_TASK();
    CK_WHEN_BLOCK(inbox::item(round * OVERLAP_ITEMS + i), tag, v) {
      check(tag == round * OVERLAP_ITEMS + i, "item: wrong tag");
      itemSum += w * v;
    }
    CK_CORO_TASK_END();
  }

  ck::coroutine run(int rounds, std::vector<int> seeds, int nitems, double* weights) {
    CK_COROUTINE();
    check(rounds == OVERLAP_ROUNDS && (int)seeds.size() == rounds && nitems == OVERLAP_ITEMS,
          "run: wrong parameters");
    for (round = 0; round < rounds; round++) {
      {  // a block: locals with initializers may not be in scope at a site
        int right = (thisIndex + 1) % OVERLAP_ELEMS;
        int seed = seeds[round] + thisIndex;
        thisProxy[right].b(round, seed + 1);
        thisProxy[right].a(round, seed);
        for (int i = nitems - 1; i >= 0; i--) thisProxy[right].item(round * nitems + i, seed * 10 + i);
      }
      // Migrate in the middle (the move happens at the next wait): the
      // record with its children is packed and restarted on the new PE.
      if (round == 1 && CkNumPes() > 1) migrateMe((CkMyPe() + 1) % CkNumPes());
      CK_OVERLAP(waitA(), waitB());
      check(gotA == seeds[round] + left() && gotB == gotA + 1, "overlap: wrong a or b");
      if (round == 2 && CkNumPes() > 1) migrateMe((CkMyPe() + 1) % CkNumPes());
      itemSum = 0;
      CK_FORALL(i, 0, nitems, waitItem(i, weights[i]));
      {
        double expect = 0;
        for (int i = 0; i < nitems; i++) expect += weights[i] * ((seeds[round] + left()) * 10 + i);
        check(itemSum == expect, "forall: wrong item sum");
      }
    }
    check((int)order.size() == 2 * rounds, "overlap: a child ran twice or not at all");
    check(__coro.num_buffered() == 0, "overlap: messages left in the inbox");
    contribute(CkCallback(CkReductionTarget(Main, overlapTestDone), mainProxy));
    CK_COROUTINE_END();
  }
};

#include "ckcoro_tests.def.h"
#include "basic.def.h"
#include "refnum.def.h"
#include "migration.def.h"
#include "owners.def.h"
