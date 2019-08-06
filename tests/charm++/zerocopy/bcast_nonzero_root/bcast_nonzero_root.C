#include "bcast_nonzero_root.decl.h"
#define NUM_ELEMENTS_PER_PE 1
#define SIZE 129
#define DEBUG(x) //x

CProxy_arr arrProxy;
CProxy_grp grpProxy;
CProxy_nodegrp ngProxy;
CProxy_tester testerProxy;
CProxy_main mProxy;

void assignValuesToIndex(int *arr, int size);
void verifyValuesWithIndex(int *arr, int size);

class main : public CBase_main {
  int testIndex;
  public:
    main(CkArgMsg *m) {
      delete m;

      // Create a chare array
      arrProxy = CProxy_arr::ckNew(CkNumPes() * NUM_ELEMENTS_PER_PE);

      // Create a group
      grpProxy = CProxy_grp::ckNew();

      // Create a nodegroup
      ngProxy = CProxy_nodegrp::ckNew();

      // Create the tester chare
      testerProxy = CProxy_tester::ckNew();

      mProxy = thisProxy;

      testIndex = -1; // will be incremented to 0 in beginTest
      beginTest();
    }

    void beginTest() {
      // testIndex represents the source PE and iterates from 0 to CkNumPes() - 1
      testIndex++;
      if(testIndex == CkNumPes()) {
        CkPrintf("[%d][%d][%d] Testing with %d source pes completed\n", CkMyPe(), CkMyNode(), CkMyRank(), CkNumPes());
        CkExit();
      } else {
        testerProxy[testIndex].beginTest();
      }
    }
};

class tester : public CBase_tester {
  int *srcBuffer;
  int counter;

  int testId;

  public:
    tester() {}

    void beginTest() {
      counter = 0;
      testId = 0;

      srcBuffer = new int[SIZE];
      assignValuesToIndex(srcBuffer, SIZE);

      DEBUG(CkPrintf("[%d][%d][%d] Broadcasting buffers to send only methods\n", CkMyPe(), CkMyNode(), CkMyRank());)

      // Test bcast sends
      arrProxy.recv_zerocopy(CkSendBuffer(srcBuffer), SIZE, thisIndex);
      grpProxy.recv_zerocopy(CkSendBuffer(srcBuffer), SIZE, thisIndex);
      ngProxy.recv_zerocopy(CkSendBuffer(srcBuffer), SIZE, thisIndex);
    }

    void bcastDone() {
      if(++counter == 3) {
        counter = 0;
        testId++;

        if(testId == 1) {

          DEBUG(CkPrintf("[%d][%d][%d] Broadcasting buffers to post methods\n", CkMyPe(), CkMyNode(), CkMyRank());)
          // Test bcast sends
          arrProxy.recv_zerocopy_post(CkSendBuffer(srcBuffer), SIZE, thisIndex);
          grpProxy.recv_zerocopy_post(CkSendBuffer(srcBuffer), SIZE, thisIndex);
          ngProxy.recv_zerocopy_post(CkSendBuffer(srcBuffer), SIZE, thisIndex);

        } else if (testId == 2) {

          CkPrintf("[%d][%d][%d] Testing with source pe:%d completed\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);

          if(thisIndex == CkNumPes() - 1)
            delete [] srcBuffer; // free buffer as the last test completed

          // move on to next pe test
          mProxy.beginTest();

        } else {
          CmiAbort("Test error! Invalid testId!\n");
        }
      }
    }
};


class grp : public CBase_grp {
  int *destBuffer;
  public:
    grp() {
      destBuffer = new int[SIZE];
    }

    void recv_zerocopy(int *buffer, size_t size, int testIndex) {
      DEBUG(CkPrintf("[%d][%d][%d] Group Send API: Entry Method\n", CkMyPe(), CkMyNode(), CkMyRank());)
      verifyValuesWithIndex(buffer, SIZE);
      CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), testerProxy[testIndex]);
      contribute(doneCb);
    }

    void recv_zerocopy_post(int *&buffer, size_t &size, int testIndex, CkNcpyBufferPost *ncpyPost) {
      DEBUG(CkPrintf("[%d][%d][%d] Group Post API: Post Entry Method\n", CkMyPe(), CkMyNode(), CkMyRank());)
      buffer = destBuffer;
    }

    void recv_zerocopy_post(int *buffer, size_t size, int testIndex) {
      DEBUG(CkPrintf("[%d][%d][%d] Group Post API: Regular Entry Method\n", CkMyPe(), CkMyNode(), CkMyRank());)

      verifyValuesWithIndex(destBuffer, SIZE);

      if(testIndex == CkNumPes() - 1)
        delete [] destBuffer; // free buffer as the last test completed

      CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), testerProxy[testIndex]);
      contribute(doneCb);
    }
};


class arr : public CBase_arr {
  int *destBuffer;
  public:
    arr() {
      destBuffer = new int[SIZE];
    }

    void recv_zerocopy(int *buffer, size_t size, int testIndex) {
      DEBUG(CkPrintf("[%d][%d][%d] Array Send API: Entry Method\n", CkMyPe(), CkMyNode(), CkMyRank());)
      verifyValuesWithIndex(buffer, SIZE);
      CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), testerProxy[testIndex]);
      contribute(doneCb);
    }

    void recv_zerocopy_post(int *&buffer, size_t &size, int testIndex, CkNcpyBufferPost *ncpyPost) {
      DEBUG(CkPrintf("[%d][%d][%d] Array Post API: Post Entry Method\n", CkMyPe(), CkMyNode(), CkMyRank());)
      buffer = destBuffer;
    }

    void recv_zerocopy_post(int *buffer, size_t size, int testIndex) {
      DEBUG(CkPrintf("[%d][%d][%d] Array Post API: Regular Entry Method\n", CkMyPe(), CkMyNode(), CkMyRank());)
      verifyValuesWithIndex(destBuffer, SIZE);

      CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), testerProxy[testIndex]);
      contribute(doneCb);

      if(testIndex == CkNumPes() - 1)
        delete [] destBuffer; // free buffer as the last test completed
    }
};


class nodegrp : public CBase_nodegrp {
  int *destBuffer;
  public:
    nodegrp() {
      destBuffer = new int[SIZE];
    }

    void recv_zerocopy(int *buffer, size_t size, int testIndex) {
      DEBUG(CkPrintf("[%d][%d][%d] Nodegroup Send API: Entry Method\n", CkMyPe(), CkMyNode(), CkMyRank());)
      verifyValuesWithIndex(buffer, SIZE);
      CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), testerProxy[testIndex]);
      contribute(doneCb);
    }

    void recv_zerocopy_post(int *&buffer, size_t &size, int testIndex, CkNcpyBufferPost *ncpyPost) {
      DEBUG(CkPrintf("[%d][%d][%d] Nodegroup Post API: Post Entry Method\n", CkMyPe(), CkMyNode(), CkMyRank());)
      buffer = destBuffer;
    }

    void recv_zerocopy_post(int *buffer, size_t size, int testIndex) {
      DEBUG(CkPrintf("[%d][%d][%d] Nodegroup Post API: Regular Entry Method\n", CkMyPe(), CkMyNode(), CkMyRank());)
      verifyValuesWithIndex(destBuffer, SIZE);

      CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), testerProxy[testIndex]);
      contribute(doneCb);

      if(testIndex == CkNumPes() - 1)
        delete [] destBuffer; // free buffer as the last test completed
    }
};

// Util methods
void assignValuesToIndex(int *arr, int size){
  for(int i=0; i<size; i++)
     arr[i] = i;
}

void verifyValuesWithIndex(int *arr, int size){
  for(int i=0; i<size; i++)
     CkAssert(arr[i] == i);
}

#include "bcast_nonzero_root.def.h"
