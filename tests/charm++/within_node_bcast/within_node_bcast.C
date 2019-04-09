#include "within_node_bcast.decl.h"

CProxy_TestGroup gProxy;
CProxy_TestNodeGroup ngProxy;

class TestMessage : public CMessage_TestMessage {
public:
  int number;
  std::atomic<int> counter;
  CkCallback cb;

  TestMessage(int i, CkCallback c) : number(i), cb(c), counter(0) {}
};

class Main : public CBase_Main {
private:
  int test_num;
  int result_count;
  int expected_result;
public:
  Main(CkArgMsg* msg) : test_num(0), result_count(0), expected_result(0) {
    delete msg;

    ngProxy = CProxy_TestNodeGroup::ckNew();
    gProxy = CProxy_TestGroup::ckNew();

    thisProxy.runTests();
  }

  void runTests() {
    CkPrintf("Starting test %i\n", test_num);

    result_count = 0;
    expected_result = 0;

    CkCallback cb(CkReductionTarget(Main, testDone), thisProxy);
    TestMessage* msg = new TestMessage(test_num, cb);

    switch (test_num) {
      case 0:
        ngProxy.recv(msg);
        break;
      case 1:
        ngProxy.recvCopy(msg);
        break;
      default:
        CkPrintf("Tests complete!\n");
        CkExit();
    }
  }

  void testDone(int result) {
    if (result_count == 0) {
      result_count++;
      expected_result = result;
    } else if (result == expected_result) {
      CkPrintf("Test %i PASSED!\n", test_num);
      test_num++;
      thisProxy.runTests();
    } else {
      CkPrintf("The results do not match: %i != %i\n", result, expected_result);
      CkAbort("Failed expectations!\n");
    }
  }
};

class TestNodeGroup : public CBase_TestNodeGroup {
public:
  TestNodeGroup() {
    CkPrintf("TestNodeGroup created on node %i\n", CkMyNode());
  }

  void recv(TestMessage* msg) {
    // Here the expectation is that the counter is incremented once by each PE
    // and starts at 0. So my nodes sum will be n * (n - 1) / 2.
    int myExpectation = (CmiMyNodeSize() * (CmiMyNodeSize() - 1)) / 2;
    contribute(sizeof(int), &myExpectation, CkReduction::sum_int, msg->cb);

    CkBroadcastWithinNode(CkIndex_TestGroup::recv(NULL), msg, gProxy);
  }

  void recvCopy(TestMessage* msg) {
    // Here, there will be one copy of the message per PE, so each PE will
    // contribute msg->number, and my sum is therefor my size * msg->number.
    msg->counter.store(msg->number);
    int myExpectation = msg->number * CkMyNodeSize();
    contribute(sizeof(int), &myExpectation, CkReduction::sum_int, msg->cb);

    CkBroadcastWithinNode(CkIndex_TestGroup::recvCopy(NULL), msg, gProxy);
  }
};

class TestGroup : public CBase_TestGroup {
public:
  TestGroup() {
    CkPrintf("TestGroup created on PE %i\n", CkMyPe());
  }

  // Marked [nokeep], so I share this message with every PE on my node, and
  // should not delete it.
  void recv(TestMessage* msg) {
    int val = msg->counter.fetch_add(1);
    CkPrintf("PE %i received message #%i, with receiver counter %i\n",
        CkMyPe(), msg->number, val);
    contribute(sizeof(int), &val, CkReduction::sum_int, msg->cb);
  }

  // Not marked [nokeep], so I get my own copy of the message and I'm in charge
  // of deallocation.
  void recvCopy(TestMessage* msg) {
    int val = msg->counter.fetch_add(1);
    CkPrintf("PE %i received a copy of message #%i with receiver counter %i\n",
        CkMyPe(), msg->number, val);

    // The value of what I received should always be the message number since
    // I get my own copy of the message.
    CkAssert(val == msg->number);

    contribute(sizeof(int), &val, CkReduction::sum_int, msg->cb);
    delete msg;
  }
};

#include "within_node_bcast.def.h"
