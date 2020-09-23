#include "zc_post_delayed.decl.h"
#define SIZE 2000
#define NUM_ELEMENTS_PER_PE 2
#define CONSTANT 188

CProxy_arr arrProxy;
CProxy_grp grpProxy;
CProxy_nodegrp ngProxy;
CProxy_tester chareProxy;

void assignValuesToIndex(int *arr, int size);
void assignValuesToConstant(int *arr, int size, int constantVal);
void verifyValuesWithConstant(int *arr, int size, int constantVal);
void verifyValuesWithIndex(int *arr, int size, int startIndex);

class main : public CBase_main {
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
      chareProxy = CProxy_tester::ckNew();
    }
};

class tester : public CBase_tester {
  int *srcBuffer1;
  int *srcBuffer2;
  int counter;
  public:
    tester() {

      counter = 0;

      srcBuffer1 = new int[SIZE];
      assignValuesToConstant(srcBuffer1, SIZE, CONSTANT);

      srcBuffer2 = new int[SIZE];
      assignValuesToConstant(srcBuffer2, SIZE, CONSTANT);

      arrProxy.recv_zerocopy(CkSendBuffer(srcBuffer1), SIZE, true);
      grpProxy.recv_zerocopy(CkSendBuffer(srcBuffer1), SIZE, CkSendBuffer(srcBuffer2), SIZE, true);
      ngProxy.recv_zerocopy(CkSendBuffer(srcBuffer1), SIZE, true);

      // Test p2p sends
      //arrProxy[9].recv_zerocopy(CkSendBuffer(srcBuffer1), SIZE, false);
      //grpProxy[CkNumPes() - 1].recv_zerocopy(CkSendBuffer(srcBuffer1), SIZE, CkSendBuffer(srcBuffer2), SIZE, false);
      //ngProxy[CkNumNodes() - 1].recv_zerocopy(CkSendBuffer(srcBuffer1), SIZE, false);
    }

    void callP2pReadyToPost() {
      if(++counter == 3) {
        counter = 0;
        arrProxy[9].readyToPost();
        grpProxy[CkNumPes() - 1].readyToPost();
        ngProxy[CkNumNodes() - 1].readyToPost();
      }
    }

    void p2pDone() {
      if(++counter == 3) {
        counter = 0;
        //CkPrintf("[%d][%d][%d] All tests have successfully completed\n", CkMyPe(), CkMyNode(), CkMyRank());
        //CkExit();

        // Test bcast sends
        arrProxy.recv_zerocopy(CkSendBuffer(srcBuffer1), SIZE, true);
        grpProxy.recv_zerocopy(CkSendBuffer(srcBuffer1), SIZE, CkSendBuffer(srcBuffer2), SIZE, true);
        ngProxy.recv_zerocopy(CkSendBuffer(srcBuffer1), SIZE, true);
      }
    }

    void callBcastReadyToPost() {
      if(++counter == 1) {
        counter = 0;
        //arrProxy.readyToPost();
        //grpProxy.readyToPost();
        ngProxy.readyToPost();
      }
    }

    void bcastDone() {
      if(++counter == 3) {
        counter = 0;
        delete [] srcBuffer1;
        delete [] srcBuffer2;
        CkPrintf("[%d][%d][%d] All tests have successfully completed\n", CkMyPe(), CkMyNode(), CkMyRank());
        CkExit();
      }
    }
};

class arr : public CBase_arr {
  int *destBuffer;
  int tag;
  void *dummy;
  public:
    arr() {
      //CkPrintf("[%d][%d][%d][%d] ************* array constructor\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      destBuffer = new int[SIZE];
      assignValuesToIndex(destBuffer, SIZE); // Initial values
    }

    void recv_zerocopy(int *&buffer, size_t &size, bool isBcast, CkNcpyBufferPost *ncpyPost) {
      CkPrintf("[%d][%d][%d][%d] =========== recv_zerocopy post em\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      tag = CkPostBufferLater(ncpyPost, 0);

      thisProxy[thisIndex].readyToPost();
      //if(isBcast) {
      //  CkCallback doneCb = CkCallback(CkReductionTarget(tester, callBcastReadyToPost), chareProxy);
      //  contribute(doneCb);
      //} else {
      //  chareProxy.callP2pReadyToPost();
      //}
    }

    void readyToPost() {
      CkPrintf("[%d][%d][%d][%d] ########## readyToPost\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      CkPostBuffer(destBuffer, (size_t) SIZE, tag);
    }

    void recv_zerocopy(int *buffer, size_t size, bool isBcast) {
      CkPrintf("[%d][%d][%d][%d] ^^^^^^^^^^^ recv_zerocopy regular em\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      verifyValuesWithConstant(destBuffer, SIZE, CONSTANT);

      if(isBcast) {
        CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), chareProxy);
        contribute(doneCb);
        delete [] destBuffer;
      } else {
        assignValuesToIndex(destBuffer, SIZE); // Reset values
        chareProxy.p2pDone();
      }
    }
};

class grp : public CBase_grp {
  int *destBuffer1;
  int *destBuffer2;
  int tag1, tag2;
  public:
    grp() {
      //CkPrintf("[%d][%d][%d][%d] ************* group constructor\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      destBuffer1 = new int[SIZE];
      assignValuesToIndex(destBuffer1, SIZE);

      destBuffer2 = new int[SIZE];
      assignValuesToIndex(destBuffer2, SIZE);
    }

    void recv_zerocopy(int *&buffer1, size_t &size1, int *&buffer2, size_t &size2, bool isBcast, CkNcpyBufferPost *ncpyPost) {
      CkPrintf("[%d][%d][%d][%d] =========== recv_zerocopy post em\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      tag1 = CkPostBufferLater(ncpyPost, 0);
      tag2 = CkPostBufferLater(ncpyPost, 1);

      thisProxy[thisIndex].readyToPost();
      //if(isBcast) {
      //  thisProxy[thisIndex].readyToPost();
      //  //CkCallback doneCb = CkCallback(CkReductionTarget(tester, callBcastReadyToPost), chareProxy);
      //  //contribute(doneCb);
      //} else {
      //  chareProxy.callP2pReadyToPost();
      //}
    }

    void readyToPost() {
      CkPrintf("[%d][%d][%d][%d] ########## readyToPost\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      CkPostBuffer(destBuffer1, (size_t) SIZE, tag1);
      CkPostBuffer(destBuffer2, (size_t) SIZE, tag2);
    }

    void recv_zerocopy(int *buffer1, size_t size1, int *buffer2, size_t size2, bool isBcast ) {
      CkPrintf("[%d][%d][%d][%d] ^^^^^^^^^^^ recv_zerocopy regular em\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      verifyValuesWithConstant(destBuffer1, SIZE, CONSTANT);
      verifyValuesWithConstant(destBuffer2, SIZE, CONSTANT);

      if(isBcast) {
        CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), chareProxy);
        contribute(doneCb);
        delete [] destBuffer1;
        delete [] destBuffer2;
      } else {
        assignValuesToIndex(destBuffer1, SIZE); // Reset values
        assignValuesToIndex(destBuffer2, SIZE); // Reset values
        chareProxy.p2pDone();
      }
    }
};

class nodegrp : public CBase_nodegrp {
  int *destBuffer;
  int tag;
  public:
    nodegrp() {
      tag = -20;
      CkPrintf("[%d][%d][%d][%d] ************* nodegroup constructor\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      destBuffer = new int[SIZE];
      assignValuesToIndex(destBuffer, SIZE);
    }

    void recv_zerocopy(int *&buffer, size_t &size, bool isBcast, CkNcpyBufferPost *ncpyPost) {
      CkPrintf("[%d][%d][%d][%d] =========== recv_zerocopy post em\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      tag = CkPostBufferLater(ncpyPost, 0);

      thisProxy[thisIndex].readyToPost();
      //if(isBcast) {
      //  CkCallback doneCb = CkCallback(CkReductionTarget(tester, callBcastReadyToPost), chareProxy);
      //  contribute(doneCb);
      //} else {
      //  chareProxy.callP2pReadyToPost();
      //}
    }

    void readyToPost() {
      CkPrintf("[%d][%d][%d][%d] ########## readyToPost\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      CkPostBuffer(destBuffer, (size_t) SIZE, tag);
    }

    void recv_zerocopy(int *buffer, size_t size, bool isBcast) {
      CkPrintf("[%d][%d][%d][%d] ^^^^^^^^^^^ recv_zerocopy regular em\n", CkMyPe(), CkMyNode(), CkMyRank(), thisIndex);
      verifyValuesWithConstant(destBuffer, SIZE, CONSTANT);

      if(isBcast) {
        CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), chareProxy);
        contribute(doneCb);
        delete [] destBuffer;
      } else {
        assignValuesToIndex(destBuffer, SIZE); // Reset values
        chareProxy.p2pDone();
      }
    }
};

// Util methods
void assignValuesToIndex(int *arr, int size){
  for(int i=0; i<size; i++)
     arr[i] = i;
}

void assignValuesToConstant(int *arr, int size, int constantVal){
  for(int i=0; i<size; i++)
     arr[i] = constantVal;
}

void verifyValuesWithConstant(int *arr, int size, int constantVal){
  for(int i=0; i<size; i++)
     CkAssert(arr[i] == constantVal);
}

void verifyValuesWithIndex(int *arr, int size, int startIndex){
  for(int i=startIndex; i<size; i++)
     CkAssert(arr[i] == i);
}

#include "zc_post_delayed.def.h"
