#include "zc_post_modify_size.decl.h"
#define SIZE 2000
#define NUM_ELEMENTS_PER_PE 10
#define CONSTANT 188

CProxy_arr1 arrProxy;
CProxy_grp1 grpProxy;
CProxy_nodegrp1 ngProxy;
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
      arrProxy = CProxy_arr1::ckNew(CkNumPes() * NUM_ELEMENTS_PER_PE);

      // Create a group
      grpProxy = CProxy_grp1::ckNew();

      // Create a nodegroup
      ngProxy = CProxy_nodegrp1::ckNew();

      // Create the tester chare
      chareProxy = CProxy_tester::ckNew();
    }
};

class tester : public CBase_tester {
  int *srcBuffer;
  int counter;
  public:
    tester() {

      counter = 0;

      srcBuffer = new int[SIZE];
      assignValuesToConstant(srcBuffer, SIZE, CONSTANT);

      // Test p2p sends
      arrProxy[9].recv_zerocopy(CkSendBuffer(srcBuffer), SIZE, false);
      grpProxy[CkNumPes() - 1].recv_zerocopy(CkSendBuffer(srcBuffer), SIZE, false);
      ngProxy[CkNumNodes() - 1].recv_zerocopy(CkSendBuffer(srcBuffer), SIZE, false);
    }

    void p2pDone() {
      if(++counter == 3) {
        counter = 0;

        // Test bcast sends
        arrProxy.recv_zerocopy(CkSendBuffer(srcBuffer), SIZE, true);
        grpProxy.recv_zerocopy(CkSendBuffer(srcBuffer), SIZE, true);
        ngProxy.recv_zerocopy(CkSendBuffer(srcBuffer), SIZE, true);
      }
    }

    void bcastDone() {
      if(++counter == 3) {
        counter = 0;
        delete [] srcBuffer;
        CkPrintf("[%d][%d][%d] All tests have successfully completed\n", CkMyPe(), CkMyNode(), CkMyRank());
        CkExit();
      }
    }
};

class arr1 : public CBase_arr1 {
  int *destBuffer;
  public:
    arr1() {
      destBuffer = new int[SIZE];
      assignValuesToIndex(destBuffer, SIZE);
    }

    void recv_zerocopy(int *&buffer, size_t &size, bool isBcast, CkNcpyBufferPost *ncpyPost) {
      buffer = destBuffer;
      if(isBcast) {
        size = SIZE/2;
      } else {
        size = SIZE/4;
      }
    }

    void recv_zerocopy(int *buffer, size_t size, bool isBcast) {
      if(isBcast) {
        verifyValuesWithConstant(destBuffer, SIZE/2, CONSTANT);
        verifyValuesWithIndex(destBuffer, SIZE, SIZE/2);

        CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), chareProxy);
        contribute(doneCb);

        delete [] destBuffer;
      } else {
        verifyValuesWithConstant(destBuffer, SIZE/4, CONSTANT);
        verifyValuesWithIndex(destBuffer, SIZE, SIZE/4);

        chareProxy.p2pDone();
      }
    }
};

class grp1 : public CBase_grp1 {
  int *destBuffer;
  public:
    grp1() {
      destBuffer = new int[SIZE];
      assignValuesToIndex(destBuffer, SIZE);
    }

    void recv_zerocopy(int *&buffer, size_t &size, bool isBcast, CkNcpyBufferPost *ncpyPost) {
      buffer = destBuffer;
      if(isBcast) {
        size = SIZE/2;
      } else {
        size = SIZE/4;
      }
    }

    void recv_zerocopy(int *buffer, size_t size, bool isBcast) {
      if(isBcast) {
        verifyValuesWithConstant(destBuffer, SIZE/2, CONSTANT);
        verifyValuesWithIndex(destBuffer, SIZE, SIZE/2);

        CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), chareProxy);
        contribute(doneCb);

        delete [] destBuffer;
      } else {
        verifyValuesWithConstant(destBuffer, SIZE/4, CONSTANT);
        verifyValuesWithIndex(destBuffer, SIZE, SIZE/4);

        chareProxy.p2pDone();
      }
    }
};

class nodegrp1 : public CBase_nodegrp1 {
  int *destBuffer;
  public:
    nodegrp1() {
      destBuffer = new int[SIZE/2];
      assignValuesToIndex(destBuffer, SIZE/2);
    }
    void recv_zerocopy(int *&buffer, size_t &size, bool isBcast, CkNcpyBufferPost *ncpyPost) {
      buffer = destBuffer;
      if(isBcast) {
        size = SIZE/2;
      } else {
        size = SIZE/4;
      }
    }

    void recv_zerocopy(int *buffer, size_t size, bool isBcast) {
      if(isBcast) {
        verifyValuesWithConstant(destBuffer, SIZE/2, CONSTANT);

        CkCallback doneCb = CkCallback(CkReductionTarget(tester, bcastDone), chareProxy);
        contribute(doneCb);

        delete [] destBuffer;
      } else {
        verifyValuesWithConstant(destBuffer, SIZE/4, CONSTANT);
        verifyValuesWithIndex(destBuffer, SIZE/2, SIZE/4);

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

#include "zc_post_modify_size.def.h"
