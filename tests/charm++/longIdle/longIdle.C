#include "longIdle.decl.h"

CProxy_main mProxy;
CProxy_testGroup gProxy;

void longIdleUserFn(void *dummy);

class main : public CBase_main {
  public:

  int counter;
  int testId;

  main(CkArgMsg *msg) {

    delete msg;

#if !CMK_ERROR_CHECKING
    CkPrintf("Cannot test LONG_IDLE when CMK_ERROR_CHECKING is disabled!\n");
    CkExit();
#endif

    mProxy = thisProxy;
    counter = 0;

    gProxy = CProxy_testGroup::ckNew();

    testId = 1; // firstTest

    CkPrintf("Testing Long Idle on all Pes\n");
    gProxy.testLongIdle1(testId); // tests on all Pes
  }

  void done() {
    switch(testId) {
      case 1:  CkPrintf("Test complete\n");
               testId++;

               CkPrintf("Testing Long Idle on a specific PE(%d)\n", CkNumPes() - 1);
               // test on a specific PE
               gProxy[CkNumPes() - 1].testLongIdle1(testId);
               break;

      case 2:  CkPrintf("Test complete\n");
               testId++;

               CkPrintf("Testing Long Idle on a specific PE(%d) using CcdCallOnConditionOnPE\n", CkNumPes() - 1);
               // test on a specific PE using CcdCallOnConditionOnPE
               gProxy[CkNumPes() - 1].testLongIdle2(testId);
               break;

      case 3:  CkPrintf("Test complete\n");
               testId++;
               CkPrintf("Testing Long Idle persistent on all PEs\n");
               // test on all PEs persistently
               gProxy.testLongIdle3(testId);
               break;

      case 4:  CkPrintf("Test complete\n");
               testId++;
               CkPrintf("Testing Long Idle persistent on a specific PE(%d)\n", CkNumPes() - 1);
               gProxy[CkNumPes() - 1].testLongIdle3(testId);
               break;

      case 5:  CkPrintf("Test complete\n");
               testId++;
               CkPrintf("Testing Long Idle persistent on a specific PE(%d) using CcdCallOnConditionKeepOnPE\n", CkNumPes() - 1);
               gProxy[CkNumPes() - 1].testLongIdle4(testId); // tests on all Pes
               break;

      case 6:  CkPrintf("Test complete\n");
               CkPrintf("All tests complete\n");
               CkExit();
               break;

      default:
               CmiAbort("main::done : invalid testId\n");
               break;

    }
  }
};


class testGroup : public CBase_testGroup {

  int idx;
  int testId;
  int counter;
  CkCallback cb;

  public:
  testGroup() {
    counter = 0;
    cb = CkCallback(CkReductionTarget(main, done), mProxy);
  }

  void testLongIdle1(int testId_) {
    testId = testId_;
    // Test one time call
    idx = CcdCallOnCondition(CcdPROCESSOR_LONG_IDLE, (CcdCondFn) longIdleUserFn, NULL);
  }

  void testLongIdle2(int testId_) {
    testId = testId_;
    // Test one time call
    idx = CcdCallOnConditionOnPE(CcdPROCESSOR_LONG_IDLE, (CcdCondFn) longIdleUserFn, NULL, CkMyPe());
  }

  void testLongIdle3(int testId_) {
    testId = testId_;
    // Test persistent call
    idx = CcdCallOnConditionKeep(CcdPROCESSOR_LONG_IDLE, (CcdCondFn) longIdleUserFn, NULL);
  }

  void testLongIdle4(int testId_) {
    testId = testId_;
    // Test persistent call
    idx = CcdCallOnConditionKeepOnPE(CcdPROCESSOR_LONG_IDLE, (CcdCondFn) longIdleUserFn, NULL, CkMyPe());
  }

  void reduceToCompletion() {
    switch(testId) {
      case 1:
              CcdCancelCallOnCondition(CcdPROCESSOR_LONG_IDLE, idx);
              contribute(cb);
              break;

      case 2:
              CcdCancelCallOnCondition(CcdPROCESSOR_LONG_IDLE, idx);
              mProxy.done();
              break;

      case 3:
              CcdCancelCallOnCondition(CcdPROCESSOR_LONG_IDLE, idx);
              mProxy.done();
              break;

      case 4:
              if(++counter == 3) {
                CcdCancelCallOnConditionKeep(CcdPROCESSOR_LONG_IDLE, idx);
                counter = 0;
                contribute(cb);
              }
              break;

      case 5:
              if(++counter == 3) {
                CcdCancelCallOnConditionKeep(CcdPROCESSOR_LONG_IDLE, idx);
                counter = 0;
                mProxy.done();
              }
              break;

      case 6:
              if(++counter == 3) {
                CcdCancelCallOnConditionKeep(CcdPROCESSOR_LONG_IDLE, idx);
                counter = 0;
                mProxy.done();
              }
              break;

      default:
              CmiAbort("testGroup:reduceToCompletion: invalid testId\n");
              break;
    }
  }
};

void longIdleUserFn(void *dummy) {
  gProxy.ckLocalBranch()->reduceToCompletion();
}

#include "longIdle.def.h"
