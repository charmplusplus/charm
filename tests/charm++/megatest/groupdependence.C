#include "groupdependence.h"

class tester : public CBase_tester {
  int counter;
  int testid;
  public:

  // Constructor
  tester() {
    CkAssert(CkMyPe() == 0);
    counter = 0;
    testid = 1;
  }

  // Method to drive tests
  void next_test() {

    switch(testid) {
      // Test 1 : Ensure that a group creation call respects group dependence
      case 1: {
        CkEntryOptions optsA, optsB;
        // Create GroupA
        optsA.setPriority(20);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Create GroupB
        // Set groupAID as a group dependency in Group B's constructor
        optsB.setGroupDepID(groupAID);
        optsB.setPriority(-1);
        CkGroupID groupBID = CProxy_groupB::ckNew(groupAID, &optsB);
        break;
      }
      // Test 2 : Ensure that a group's regular entry method respects group dependence
      case 2: {
        CkEntryOptions optsA, optsB;
        // Create GroupB
        CkGroupID groupBID = CProxy_groupB::ckNew();

        // Create Group A
        optsA.setPriority(20);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in Group B's regular entry method
        optsB.setPriority(-1);
        optsB.setGroupDepID(groupAID);
        ((CProxy_groupB)groupBID).test_em_regular(groupAID, &optsB);
        break;
      }

      // Test 3 : Ensure that a group's inline entry method respects group dependence
      case 3: {
        CkEntryOptions optsA, optsB;
        // Create GroupB
        CkGroupID groupBID = CProxy_groupB::ckNew();

        // Create Group A
        optsA.setPriority(20);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in Group B's inline entry method
        optsB.setPriority(-1);
        optsB.setGroupDepID(groupAID);
        ((CProxy_groupB)groupBID).test_em_inline(groupAID, &optsB);
        break;
      }

      // Test 4 : Ensure that a chare creation call respects group dependence
      case 4: {
        CkEntryOptions optsA, opts;
        // Create GroupA
        optsA.setPriority(20);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Create ChareA
        // Set groupAID as a group dependency in Chare A's constructor
        opts.setGroupDepID(groupAID);
        opts.setPriority(-1);
        CProxy_chareA::ckNew(groupAID, CK_PE_ANY, &opts);
        break;
      }

      // Test 5 : Ensure that a chare's regular entry method respects group dependence
      case 5: {
        CkEntryOptions optsA, opts;
        // Create chareA
        CProxy_chareA cProxy = CProxy_chareA::ckNew();

        // Create groupA
        optsA.setPriority(20);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in chareA's regular entry method
        opts.setPriority(-1);
        opts.setGroupDepID(groupAID);
        cProxy.test_em_regular(groupAID, &opts);
        break;
      }

      // Test 6 : Ensure that a chare's inline entry method respects group dependence
      case 6: {
        CkEntryOptions optsA, opts;
        // Create chareA
        CProxy_chareA cProxy = CProxy_chareA::ckNew();

        // Create groupA
        optsA.setPriority(20);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in chareA's inline entry method
        opts.setPriority(-1);
        opts.setGroupDepID(groupAID);
        cProxy.test_em_inline(groupAID, &opts);
        break;
      }

      // Test 7 : Ensure that a chare array creation call respects group dependence
      case 7: {
        CkEntryOptions optsA, opts;

        // Create Group A
        optsA.setPriority(-1);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in arrayA's constructor
        opts.setPriority(20);
        opts.setGroupDepID(groupAID);

        int size = CkNumPes()*8;
        CProxy_arrayA arrProxy = CProxy_arrayA::ckNew(size, groupAID, size, &opts);
        break;
      }

      // Test 8: Ensure that a chare array's regular entry method respects group dependence
      case 8: {
        CkEntryOptions optsA, opts;

        // Create arrayA
        int size = CkNumPes()*8;
        CProxy_arrayA arrProxy = CProxy_arrayA::ckNew(size, size);

        optsA.setPriority(-1);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in arrayA's regular entry method
        opts.setPriority(20);
        opts.setGroupDepID(groupAID);
        arrProxy.test_em_regular(groupAID, &opts);
        break;
      }

      // Test 9: Ensure that a chare array's inline entry method respects group dependence
      case 9: {
        CkEntryOptions optsA, opts;

        int size = CkNumPes();
        // Create arrayA
        CProxy_arrayA arrProxy = CProxy_arrayA::ckNew(size, size);

        // Create groupA
        optsA.setPriority(-1);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in arrayA's inline entry method
        opts.setPriority(20);
        opts.setGroupDepID(groupAID);
        arrProxy.test_em_inline(groupAID, &opts);
        break;
      }

      // Test 10: Ensure that a nodegroup's creation call respects group dependence
      case 10: {
        CkEntryOptions optsA, opts;

        // Create groupA
        optsA.setPriority(-1);
        CProxy_groupA groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in nodeGroupA's constructor
        opts.setPriority(20);
        opts.setGroupDepID(groupAID);
        CProxy_nodeGroupA nodegroupID = CProxy_nodeGroupA::ckNew(groupAID, &opts);
        break;
      }

      // Test 11: Ensure that a nodegroup's regular entry method respects group depenedence
      case 11: {
        CkEntryOptions optsA, opts;

        // Create nodeGroupA
        CProxy_nodeGroupA nodegroupID = CProxy_nodeGroupA::ckNew();

        // Create groupA
        optsA.setPriority(-1);
        CProxy_groupA groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in nodeGroupA's regular entry method
        opts.setPriority(20);
        opts.setGroupDepID(groupAID);
        nodegroupID.test_em_regular(groupAID, &opts);
        break;
      }

      // Test 12: Ensure that a nodegroup's inline entry method respects group dependence
      case 12: {
        CkEntryOptions optsA, opts;

        // Create nodeGroupA
        CProxy_nodeGroupA nodegroupID = CProxy_nodeGroupA::ckNew();

        // Create groupA
        optsA.setPriority(-1);
        CProxy_groupA groupAID = CProxy_groupA::ckNew(&optsA);

        // Set groupAID as a group dependency in nodeGroupA's inline entry method
        opts.setPriority(20);
        opts.setGroupDepID(groupAID);
        nodegroupID.test_em_inline(groupAID, &opts);
        break;
      }

      // Test 13: Ensure that fixed sized message respects group dependence
      case 13: {
        CkEntryOptions optsA, optsB;

        // Create groupA
        optsA.setPriority(-1);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Create groupB
        optsB.setPriority(-1);
        CkGroupID groupBID = CProxy_groupB::ckNew(&optsB);

        // Fixed sized message
        fixedMessage *msg = new (0, GroupDepNum{2}) fixedMessage(groupAID, groupBID);

        // Set groupAID & groupBID as a group dependency in the fixed sized message
        UsrToEnv(msg)->setGroupDep(groupAID);
        UsrToEnv(msg)->setGroupDep(groupBID, 1);

        ((CProxy_tester)testObj).recv_fixedmessage(msg);
        break;
      }

      // Test 14: Ensure that variable sized message respects group dependence
      case 14: {
        CkEntryOptions optsA, optsB;

        // Create groupA
        optsA.setPriority(-1);
        CkGroupID groupAID = CProxy_groupA::ckNew(&optsA);

        // Create groupB
        optsB.setPriority(-1);
        CkGroupID groupBID = CProxy_groupB::ckNew(&optsB);

        // Variable sized message
        varMessage *msg = new (20, 10, 0, GroupDepNum{2}) varMessage(groupAID, groupBID);

        // Set groupAIB & groupBID as a group dependency in the variable sized message
        UsrToEnv(msg)->setGroupDep(groupAID);
        UsrToEnv(msg)->setGroupDep(groupBID, 1);

        ((CProxy_tester)testObj).recv_varmessage(msg);
        break;
      }

      default:
        megatest_finish();
        break;
    }
  }

  // Method to indicate completion of a test and advance to the next test
  void complete_test(int num) {
    counter++;
    if(counter == num) {
      counter = 0;
      testid++;
      next_test();
    }
  }

  // Entry method to receive fixed sized messages
  void recv_fixedmessage(fixedMessage *msg) {
    groupA *groupALocal = (msg->aProxy).ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Tester's fixed message entry method called before Group A \n");

    groupB *groupBLocal = (msg->bProxy).ckLocalBranch();
    if(groupBLocal->called == false)
      CkAbort("Tester's fixed message entry method called before Group A \n");

    // check preset values
    CkAssert(msg->a = 20);
    CkAssert(msg->b = 12.88);
    complete_test(1);
  }

  // Entry method to receive variable sized messages
  void recv_varmessage(varMessage *msg) {
    groupA *groupALocal = (msg->aProxy).ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Tester's variable message entry method called before Group A \n");

    groupB *groupBLocal = (msg->bProxy).ckLocalBranch();
    if(groupBLocal->called == false)
      CkAbort("Tester's variable message entry method called before Group A \n");

    // check preset values
    for(int i=0;i<20;i++) {
      CkAssert(msg->a[i]==i);
      if(i<10) {
        CkAssert(std::fabs(msg->b[i] - 1.3*i) < 1.0e-6);
      }
    }
    complete_test(1);
  }
};

MEGATEST_REGISTER_TEST(groupdependence,"nbhat4",0);
#include "groupdependence.def.h"
