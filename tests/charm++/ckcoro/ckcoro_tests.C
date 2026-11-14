// Tests of coroutine entry methods (ckcoro.h). The migration test is the
// coroutine version of tests/charm++/sdag/sdag_tests.C's MigrationTest: its
// case statements are CK_WHEN_ANY sites, its conjunctions CK_WHEN_ALL sites
// (both messages stay in the pupped inbox until both have arrived, so the
// element may migrate in between), and its final overlap over stray
// messages is a check that
// the inbox is empty when the coroutine ends. The marshall test checks
// marshalled inbox entries: matched by the first argument, delivered by
// broadcast, reached through a reduction, and called directly from inside
// a reduction's delivery; and a coroutine entry taking a message with a
// custom pack.
#include <atomic>
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
#define PACK_N 5
#define OVERLAP_ELEMS 6
#define OVERLAP_ROUNDS 4
#define OVERLAP_ITEMS 5
#define OWNER_ROUNDS 3
#define OWNER_ITEMS 7
#define FUTURE_ELEMS 6
#define FUTURE_ROUNDS 4

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_MigrationTest migrationTestProxy;
/* readonly */ CProxy_BasicTest basicTestProxy;
/* readonly */ CProxy_RefnumTest refnumTestProxy;

// A message with a custom pack and unpack, in the manual's form: the unpacked
// message owns a heap array, pack frees the message and unpack builds a new
// one. live counts unpacked messages in the process, to check that the
// coroutine state deletes the entry's message (with its destructor) once.
class PackMsg : public CMessage_PackMsg {
public:
  static std::atomic<int> live;
  int n;
  int* data;
  explicit PackMsg(int n_) : n(n_), data(new int[n_]) { live++; }
  ~PackMsg() { delete[] data; live--; }
  static void* pack(PackMsg* m) {
    int* buf = (int*)CkAllocBuffer(m, (m->n + 1) * sizeof(int));
    buf[0] = m->n;
    std::copy(m->data, m->data + m->n, buf + 1);
    delete m;
    return buf;
  }
  static PackMsg* unpack(void* in) {
    int* buf = (int*)in;
    PackMsg* m = new (CkAllocBuffer(in, sizeof(PackMsg))) PackMsg(buf[0]);
    std::copy(buf + 1, buf + 1 + buf[0], m->data);
    CkFreeMsg(in);
    return m;
  }
};
std::atomic<int> PackMsg::live{0};

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
    // Under +ck_fakemigrate_pup the elements migrate at every wait, and this
    // test counts its own migrations in ckJustMigrated: skipped.
    if (_ckCoroFakeMigratePup) CkPrintf("(skipped under +ck_fakemigrate_pup)\n");
    if (CkNumPes() > 1 && !_ckCoroFakeMigratePup) {
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
    CkStartQD(CkCallback(CkIndex_Main::marshallLeakCheck(), thisProxy));
  }

  void marshallLeakCheck() {
    // Every packRun() coroutine has ended and the state deleted its message
    // with the destructor: none is alive in this process.
    if (CkNumNodes() == 1 && PackMsg::live != 0)
      ::CkAbort("ckcoro marshall test: %d PackMsg message(s) leaked or freed without the destructor",
                (int)PackMsg::live);
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
    thisProxy.doFutureTest();
  }

  void doFutureTest() {
    CkPrintf("Running future tests\n");
    CProxy_FutureTest a = CProxy_FutureTest::ckNew(FUTURE_ELEMS);
    a.run(FUTURE_ROUNDS);
  }

  void futureTestDone() {
    CkPrintf("Future tests complete\n");
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
    CK_MIGRATABLE {
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
    }
  }
};

class GroupTest : public CBase_GroupTest {
  GroupTest_CORO_CODE
  int round, from, sum;
  ck::coro::future total, reply;
  CkReductionMsg* red = nullptr;   // received at a wait and deleted before the next
  CkDataMsg* data = nullptr;
  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro owner test (group): %s", what);
  }

public:
  GroupTest() : round(0), from(-1), sum(0) {}
  GroupTest(CkMigrateMessage* m) : CBase_GroupTest(m) {}
  void pup(PUP::er& p) { p | round; p | from; p | sum; p | total; p | reply; }

  void ask(CkCallback cb, int v) { cb.send(CkDataMsg::buildNew(sizeof(int), &v)); }

  ck::coroutine run(int rounds, CProxy_NodeGroupTest ng) {
    CK_MIGRATABLE {
      for (round = 0; round < rounds; round++) {
        thisProxy[(CkMyPe() + 1) % CkNumPes()].token(round, CkMyPe());
        ng[CkMyNode()].token(round, CkMyPe());
        CK_WHEN(inbox::token(round), ck::coro::skip, from);
        check(from == (CkMyPe() + CkNumPes() - 1) % CkNumPes(), "token from the wrong PE");
        sum += from;
      }
      // Futures on a group: a reduction over the branches that every branch
      // waits for, and an answer from the next PE's branch.
      contribute(sizeof(int), &sum, CkReduction::sum_int, CK_CORO_FUTURE_ALL(total));
      CK_WHEN(total, red);
      check(*(int*)red->getData() == rounds * CkNumPes() * (CkNumPes() - 1) / 2,
            "group sum through a future");
      delete red;
      thisProxy[(CkMyPe() + 1) % CkNumPes()].ask(CK_CORO_FUTURE(reply), 10 * CkMyPe());
      CK_WHEN(reply, data);
      check(*(int*)data->getData() == 10 * CkMyPe(), "answer through a future");
      delete data;
      contribute(sizeof(int), &sum, CkReduction::sum_int,
                 CkCallback(CkReductionTarget(NodeGroupTest, groupSum), ng));
      contribute(CkCallback(CkReductionTarget(Main, groupReady), mainProxy));
      CK_WHEN(inbox::resume());
      contribute(sizeof(int), &sum, CkReduction::sum_int,
                 CkCallback(CkReductionTarget(Main, groupDone), mainProxy));
    }
  }
};

class NodeGroupTest : public CBase_NodeGroupTest {
  NodeGroupTest_CORO_CODE
  int round, count, from, sum, groupTotal;
  ck::coro::future total, own;
  CkReductionMsg* red = nullptr;   // received at a wait and deleted before the next
  CkDataMsg* data = nullptr;
  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro owner test (nodegroup): %s", what);
  }

public:
  NodeGroupTest() : round(0), count(0), from(-1), sum(0), groupTotal(-1) {}
  NodeGroupTest(CkMigrateMessage* m) : CBase_NodeGroupTest(m) {}
  void pup(PUP::er& p) {
    p | round; p | count; p | from; p | sum; p | groupTotal; p | total; p | own;
  }

  ck::coroutine run(int rounds) {
    CK_MIGRATABLE {
      for (round = 0; round < rounds; round++) {
        for (count = 0; count < CkNodeSize(CkMyNode()); count++) {
          CK_WHEN(inbox::token(round), ck::coro::skip, from);
          sum += from;
        }
      }
      // The group reduction, delivered to every branch through the reduction
      // wrapper of a nodegroup inbox entry (under the node lock).
      CK_WHEN(inbox::groupSum(), groupTotal);
      check(groupTotal == rounds * CkNumPes() * (CkNumPes() - 1) / 2, "group sum by reduction");
      // Futures on a nodegroup: a reduction over the nodes that every branch
      // waits for, and a value sent to the branch's own future.
      contribute(sizeof(int), &sum, CkReduction::sum_int, CK_CORO_FUTURE_ALL(total));
      CK_WHEN(total, red);
      check(*(int*)red->getData() == rounds * CkNumPes() * (CkNumPes() - 1) / 2,
            "nodegroup sum through a future");
      delete red;
      {
        int v = 100 + CkMyNode();
        CK_CORO_FUTURE(own).send(CkDataMsg::buildNew(sizeof(int), &v));
      }
      CK_WHEN(own, data);
      check(*(int*)data->getData() == 100 + CkMyNode(), "value through a nodegroup's future");
      delete data;
      contribute(CkCallback(CkReductionTarget(Main, nodeGroupReady), mainProxy));
      CK_WHEN(inbox::resume());
      contribute(sizeof(int), &sum, CkReduction::sum_int,
                 CkCallback(CkReductionTarget(Main, nodeGroupDone), mainProxy));
    }
  }
};

class ChareTest : public CBase_ChareTest {
  ChareTest_CORO_CODE
  int n, i, v, sum;
  ck::coro::future own, all;
  CkDataMsg* data = nullptr;   // received at a wait and deleted before the next
  CkDataMsg* data2 = nullptr;
  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro owner test (chare): %s", what);
  }

public:
  ChareTest(int n_) : n(n_), i(0), v(0), sum(0) {}
  ChareTest(CkMigrateMessage* m) : CBase_ChareTest(m) {}
  void pup(PUP::er& p) { p | n; p | i; p | v; p | sum; p | own; p | all; }

  ck::coroutine run() {
    CK_MIGRATABLE {
      for (i = 0; i < n; i++) {
        CK_WHEN(inbox::item(i), ck::coro::skip, v);
        sum += v;
      }
      // Futures on a singleton chare: both kinds call the chare itself.
      CK_CORO_FUTURE(own).send(CkDataMsg::buildNew(sizeof(int), &sum));
      CK_CORO_FUTURE_ALL(all).send(CkDataMsg::buildNew(sizeof(int), &n));
      CK_WHEN_ALL(CK_ALT(own, data), CK_ALT(all, data2));
      check(*(int*)data->getData() == sum && *(int*)data2->getData() == n,
            "values through a chare's futures");
      delete data;
      delete data2;
      mainProxy.chareReady();
      CK_WHEN(inbox::resume());
      mainProxy.chareDone(sum);
      ckDestroyLater();   // destroyed once the entry method running this body returns
      ckDestroyLater();   // a second call does nothing more
    }
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
  ck::coro::task<void> waitUnlock1() { CK_MIGRATABLE { CK_WHEN(inbox::unlock1()); unlocked1 = true; } }
  ck::coro::task<void> waitUnlock2() { CK_MIGRATABLE { CK_WHEN(inbox::unlock2()); unlocked2 = true; } }
  ck::coro::task<void> waitUnlock3() { CK_MIGRATABLE { CK_WHEN(inbox::unlock3()); unlocked3 = true; } }
  // The final overlap over stray messages: each arm aborts if reached.
  ck::coro::task<void> extra(ck::coro::want w, const char* name) {
    CK_MIGRATABLE {
      CK_WHEN(w);
      ::CkAbort("ckcoro basic test: extra %s", name);
    }
  }

public:
  BasicTest() : for_index(0), which(-1), do_while(true), unlocked1(false), unlocked2(false), unlocked3(false) {
    CkPrintf("BasicTest created on PE %i\n", CkMyPe());
  }
  BasicTest(CkMigrateMessage* m) : CBase_BasicTest(m) {}
  void pup(PUP::er& p) { p | for_index; p | which; p | do_while; p | unlocked1; p | unlocked2; p | unlocked3; }

  ck::coroutine run() {
    CK_MIGRATABLE {
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
    }
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
    CK_MIGRATABLE {
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
    }
  }
  // forall [index] (0:messages, 2) { if (index & (index - 1)) case {...} else serial {...} }
  ck::coro::task<void> evenIndex(int idx) {
    CK_MIGRATABLE {
      if (idx & (idx - 1)) {
        CK_CALL(receive(idx));
      } else {
        ignored_sum += idx;
      }
    }
  }
  // A helper that returns a value: co_return inside its CK_MIGRATABLE block.
  ck::coro::task<int> receiveIndex(int idx) {
    CK_MIGRATABLE {
      CK_CALL(receive(idx));
      co_return idx;
    }
  }
  ck::coro::task<void> missed(ck::coro::want w, const char* name) {
    CK_MIGRATABLE {
      CK_WHEN(w);
      ::CkAbort("ckcoro refnum test: missed a %s message", name);
    }
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
    CK_MIGRATABLE {
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
    }
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
    CK_MIGRATABLE {
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

        // Both TestMessages (the SDAG conjunction when sentBefore(m1), sentAfter(m2))
        CK_WHEN_ALL(CK_ALT(inbox::sentBefore(), msg1), CK_ALT(inbox::sentAfter(), msg2));
        checkPair();

        // One of the calls, which was pupped as a buffered message (SDAG case)
        CK_WHEN_ANY(which, CK_ALT(inbox::call1()), CK_ALT(inbox::call2()), CK_ALT(inbox::call3()));
        respond(which);

        // Anytime migration: the element migrates without waiting for QD and
        // this coroutine keeps receiving whenever messages arrive.
        if (thisIndex == 0) CkPrintf("Testing anytime migration\n");
        sendBefore();
        thisProxy[thisIndex].migrateNow();

        CK_WHEN_ALL(CK_ALT(inbox::sentBefore(), msg1), CK_ALT(inbox::sentAfter(), msg2));
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
    }
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
// reduction total() reaches the inbox through the reduction wrapper. Then
// each element sends its right neighbour a PackMsg, whose coroutine reads
// it across three waits.
class MarshallTest : public CBase_MarshallTest {
  MarshallTest_CORO_CODE
  int iter;
  int pIter;   // the packRun() coroutine's, which may overlap the end of run()
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
  int left() const { return (thisIndex + MARSHALL_ELEMS - 1) % MARSHALL_ELEMS; }

public:
  MarshallTest() : iter(0), pIter(0), t(0), i(0), x(0), sum(0) {}
  MarshallTest(CkMigrateMessage* m) : CBase_MarshallTest(m) {}

  void pup(PUP::er& p) {
    p | iter;
    p | pIter;
    p | sum;   // checked again at arrTotal, several waits later
    if (p.isUnpacking()) { values.clear(); t = i = 0; x = 0; n = 0; darr = nullptr; dvec.clear(); tail.clear(); }
  }

  ck::coroutine run() {
    CK_MIGRATABLE {
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

      // Inbox entries called directly while the coroutine runs inside a
      // reduction's delivery are matched by their first argument, not by
      // the reduction's reference number. From the second iteration on,
      // the first call is inside the previous reduction's delivery (both
      // tick waits are met from the inbox without suspending).
      for (iter = 0; iter < 3; iter++) {
        tick(2 * iter, 0.5 * iter);
        contribute(sizeof(int), &thisIndex, CkReduction::sum_int,
                   CkCallback(CkReductionTarget(MarshallTest, total), thisProxy));
        CK_WHEN(inbox::total(), sum);
        tick(2 * iter + 1, 0.5 * iter + 0.25);
        CK_WHEN(inbox::tick(2 * iter + 1), ck::coro::skip, x);
        check(x == 0.5 * iter + 0.25, "tick: wrong value inside the reduction's delivery");
        CK_WHEN(inbox::tick(2 * iter), ck::coro::skip, x);
        check(x == 0.5 * iter, "tick: wrong value");
      }

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
      // Every run() has checked its inbox before any packRun() sends.
      contribute(CkCallback(CkReductionTarget(MarshallTest, runsDone), thisProxy));
      CK_WHEN(inbox::runsDone());
      {  // crosses to another PE when there is one: packed and unpacked
        PackMsg* pm = new PackMsg(PACK_N);
        for (int k = 0; k < PACK_N; k++) pm->data[k] = thisIndex * 1000 + k;
        thisProxy[(thisIndex + 1) % MARSHALL_ELEMS].packRun(pm);
      }
    }
  }

  // The entry's message belongs to the coroutine state: it is read after
  // every wait (after a restart it is a copy) and never deleted here.
  ck::coroutine packRun(PackMsg* m) {
    CK_MIGRATABLE {
      check(m->n == PACK_N && m->data[PACK_N - 1] == left() * 1000 + PACK_N - 1, "packed: wrong message");
      for (pIter = 0; pIter < 3; pIter++) {
        thisProxy[(thisIndex + 1) % MARSHALL_ELEMS].step(pIter);
        CK_WHEN(inbox::step(pIter), ck::coro::skip);
        check(m->n == PACK_N && m->data[0] == left() * 1000 && m->data[PACK_N - 1] == left() * 1000 + PACK_N - 1,
              "packed: the entry's message changed across a wait");
      }
      contribute(CkCallback(CkReductionTarget(Main, marshallTestDone), mainProxy));
    }
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
    CK_MIGRATABLE {
      CK_WHEN(inbox::a(round), ck::coro::skip, gotA);
      order += 'a';
    }
  }
  ck::coro::task<void> waitB() {
    CK_MIGRATABLE {
      CK_WHEN(inbox::b(round), ck::coro::skip, gotB);
      order += 'b';
    }
  }
  // Matched by round and index together: after a migration, a neighbour's
  // messages for the next round may be delivered before this round's.
  ck::coro::task<void> waitItem(int i, double w) {
    CK_MIGRATABLE {
      CK_WHEN_BLOCK(inbox::item(round * OVERLAP_ITEMS + i), tag, v) {
        check(tag == round * OVERLAP_ITEMS + i, "item: wrong tag");
        itemSum += w * v;
      }
    }
  }

  ck::coroutine run(int rounds, std::vector<int> seeds, int nitems, double* weights) {
    CK_MIGRATABLE {
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
    }
  }
};

// Callbacks awaited directly: each round a sum reduction (CK_CORO_FUTURE_ALL, a
// callback to every element); a sum and a max awaited together; a callback
// to this element (CK_CORO_FUTURE) handed to the left neighbour, which sends a
// CkDataMsg to it; and a reduction inside a CK_CALL task. At the end every
// element awaits the quiescence that element 0 starts. An own future's id
// comes round again after kOwnFuture futures, which is fine once the
// earlier message has been taken. The futures and the round are pupped
// members: the waits survive +ck_fakemigrate, the pup mode, and the
// restart that follows.
class FutureTest : public CBase_FutureTest {
  FutureTest_CORO_CODE
  int round;
  ck::coro::future fsum, fa, fb, fpeer, fqd, ftask, fwrap, fquiet, fgo;
  CMK_REFNUM_TYPE wrapId = 0;
  CkReductionMsg* red;   // received at a wait and deleted before the next
  CkReductionMsg* red2;
  int total;

  static void check(bool ok, const char* what) {
    if (!ok) ::CkAbort("ckcoro future test: %s", what);
  }
  int left() const { return (thisIndex + FUTURE_ELEMS - 1) % FUTURE_ELEMS; }
  static int sumOver(int r) { return FUTURE_ELEMS * (FUTURE_ELEMS - 1) / 2 + FUTURE_ELEMS * r; }

  // A task that waits on a future of its own and returns the reduced sum.
  ck::coro::task<int> allSum(int v) {
    CK_MIGRATABLE {
      contribute(sizeof(int), &v, CkReduction::sum_int, CK_CORO_FUTURE_ALL(ftask));
      CK_WHEN(ftask, red);
      {
        int s = *(int*)red->getData();
        delete red;
        co_return s;
      }
    }
  }

public:
  FutureTest() : round(0), red(nullptr), red2(nullptr), total(0) {}
  FutureTest(CkMigrateMessage* m) : CBase_FutureTest(m), red(nullptr), red2(nullptr) {}
  void pup(PUP::er& p) {
    p | round; p | fsum; p | fa; p | fb; p | fpeer; p | fqd; p | ftask; p | fwrap; p | wrapId;
    p | fquiet; p | fgo;
    p | total;
  }

  void ask(CkCallback cb, int value) { cb.send(CkDataMsg::buildNew(sizeof(int), &value)); }

  ck::coroutine run(int rounds) {
    CK_MIGRATABLE {
      for (round = 0; round < rounds; round++) {
        {
          int v = thisIndex + round;
          contribute(sizeof(int), &v, CkReduction::sum_int, CK_CORO_FUTURE_ALL(fsum));
        }
        CK_WHEN(fsum, red);
        CK_LABEL("checkSum") {
          check(*(int*)red->getData() == sumOver(round), "sum awaited through a future");
          delete red;
        }

        // Two reductions in flight, awaited together.
        {
          int v = thisIndex + round;
          contribute(sizeof(int), &v, CkReduction::sum_int, CK_CORO_FUTURE_ALL(fa));
          contribute(sizeof(int), &v, CkReduction::max_int, CK_CORO_FUTURE_ALL(fb));
        }
        CK_WHEN_ALL(CK_ALT(fa, red), CK_ALT(fb, red2));
        check(*(int*)red->getData() == sumOver(round), "sum awaited with a max");
        check(*(int*)red2->getData() == FUTURE_ELEMS - 1 + round, "max awaited with a sum");
        delete red;
        delete red2;

        // A callback handed to another element; the block form.
        thisProxy[left()].ask(CK_CORO_FUTURE(fpeer), 100 * round + thisIndex);
        CK_WHEN_BLOCK(fpeer, m) {
          CkDataMsg* d = static_cast<CkDataMsg*>(m);
          check(*(int*)d->getData() == 100 * round + thisIndex, "value sent to a future's callback");
          delete d;
        }

        CK_CALL_TO(total, allSum(thisIndex * round));
        check(total == round * FUTURE_ELEMS * (FUTURE_ELEMS - 1) / 2, "sum returned by a task");
      }
      // The id of an own future comes round again (with a 16-bit refnum).
      {
        CkCallback cb = CK_CORO_FUTURE(fwrap);
        wrapId = fwrap.id;
        cb.send();
      }
      CK_WHEN(fwrap);
      if (ck::coro::state::kOwnFuture <= (1ul << 16)) {
        for (unsigned long i = 1; i < (unsigned long)ck::coro::state::kOwnFuture; i++)
          (void)CK_CORO_FUTURE(fwrap);
        {
          CkCallback cb = CK_CORO_FUTURE(fwrap);
          check(fwrap.id == wrapId, "own future ids wrap");
          cb.send();
        }
        CK_WHEN(fwrap);
      }
      // A CK_CORO_FUTURE_ALL message that arrives before the element makes
      // the future: element 0 sends on its callback at once, and the last
      // element makes the future only after quiescence, when the message is
      // already in its inbox under the id.
      if (thisIndex == FUTURE_ELEMS - 1) {
        CkStartQD(CK_CORO_FUTURE(fquiet));
        CK_WHEN(fquiet);
      }
      {
        CkCallback go = CK_CORO_FUTURE_ALL(fgo);
        if (thisIndex == 0) go.send();
      }
      CK_WHEN(fgo);
      // Quiescence awaited by every element, its message discarded: each
      // makes the future, element 0 starts the detection with it.
      {
        CkCallback qd = CK_CORO_FUTURE_ALL(fqd);
        if (thisIndex == 0) CkStartQD(qd);
      }
      CK_WHEN(fqd);
      check(__coro.num_buffered() == 0, "deliveries left in the inbox");
      contribute(CkCallback(CkReductionTarget(Main, futureTestDone), mainProxy));
    }
  }
};

#include "ckcoro_tests.def.h"
#include "basic.def.h"
#include "refnum.def.h"
#include "migration.def.h"
#include "owners.def.h"
