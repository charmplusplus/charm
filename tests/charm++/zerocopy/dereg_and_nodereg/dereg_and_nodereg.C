#include "entry_method_api.decl.h"

#define CONSTANT 557

int numElements;
CProxy_Main mProxy;

void assignValuesToIndex(int *arr, int size);
void assignValuesToConstant(int *arr, int size, int constantVal);
void verifyValuesWithConstant(int *arr, int size, int constantVal);
void verifyValuesWithIndex(int *arr, int size, int startIndex=0);

class Main : public CBase_Main {
  int testIndex;
  CProxy_testArr arr1;
  public:
    Main(CkArgMsg *m) {
      if(m->argc !=2 ) {
        CkAbort("Usage: ./zerocopy_with_qd <array size>, where <array size> is even\n");
      }
      numElements = atoi(m->argv[1]);
      if(numElements % 2 != 0) {
        CkAbort("<array size> argument is not even\n");
      }
      delete m;

      testIndex = 0;

      mProxy = thisProxy;

      arr1 = CProxy_testArr::ckNew(numElements);
      arr1.test1();
    };

    void testDone() {
      testIndex++;
      switch(testIndex) {
        case 1:
                CkPrintf("[%d][%d][%d] Test 1 (ZC EM Send API - CK_BUFFER_DEREG and CK_BUFFER_NODEREG) Completed\n", CkMyPe(), CkMyNode(), CkMyRank());
                arr1.test2();
                break;
        case 2:
                CkPrintf("[%d][%d][%d] Test 2 (ZC EM Post API - CK_BUFFER_DEREG and CK_BUFFER_NODEREG) Completed\n", CkMyPe(), CkMyNode(), CkMyRank());
                arr1[0].test3();
                break;
        case 3:
                CkPrintf("[%d][%d][%d] Test 3 (ZC EM Bcast Send API - CK_BUFFER_DEREG and CK_BUFFER_NODEREG) Completed\n", CkMyPe(), CkMyNode(), CkMyRank());
                arr1[0].test4();
                break;
        case 4:
                CkPrintf("[%d][%d][%d] Test 4 (ZC EM Bcast Post API - CK_BUFFER_DEREG and CK_BUFFER_NODEREG) Completed\n", CkMyPe(), CkMyNode(), CkMyRank());
                CkExit();
                break;
        default:
                CmiAbort("Invalid testIndex value\n");
                break;
      }
    }
};

class testArr : public CBase_testArr {

  int destIndex, counter;
  int *buff, *recvBuff1, *recvBuff2, *recvBuff3;

  static int size;
  CkCallback sourceDoneCb1, sourceDoneCb2, sourceDoneCb3, testDoneCb;

  bool isBcast;

  public:

    testArr() {
      destIndex = numElements - 1 - thisIndex;
      buff = new int[size]; // Source 1
      sourceDoneCb1 = CkCallback(CkIndex_testArr::sourceDone(NULL),thisProxy[thisIndex]);
      sourceDoneCb2 = sourceDoneCb3 = sourceDoneCb1;
      testDoneCb = CkCallback(CkIndex_Main::testDone(), mProxy);
      counter = 0;
      isBcast = false;
    }

    void test1() {
      if(thisIndex < numElements/2) {
        assignValuesToConstant(buff, size, CONSTANT);

        sourceDoneCb1.setRefNum(1);
        sourceDoneCb2.setRefNum(2);
        sourceDoneCb3.setRefNum(3);

        // Send the buffer buff to destIndex
        thisProxy[destIndex].recvBuffer1(CkSendBuffer(buff, sourceDoneCb1),
                                         CkSendBuffer(buff, sourceDoneCb2, CK_BUFFER_REG, CK_BUFFER_DEREG),
                                         CkSendBuffer(buff, sourceDoneCb3, CK_BUFFER_REG, CK_BUFFER_NODEREG), size);
      }
    }

    // Executed on indices < numElements/2
    void sourceDone(CkDataMsg *msg) {
      counter++;
      CkNcpyBuffer *src = (CkNcpyBuffer *)(msg->data);

      int refNum = CkGetRefNum(msg);

      switch(refNum) {
        case 1:   // Do not de-register as it is de-registered by the RTS
                  CkAssert(src->regMode == CK_BUFFER_REG && src->deregMode == CK_BUFFER_DEREG);
                  break;
        case 2:   // Do not de-register as it is de-registered by the RTS
                  CkAssert(src->regMode == CK_BUFFER_REG && src->deregMode == CK_BUFFER_DEREG);
                  break;
        case 3:   // De-register as it is not de-registered by the RTS (because of CK_BUFFER_NODEREG)
                  CkAssert(src->regMode == CK_BUFFER_REG && src->deregMode == CK_BUFFER_NODEREG);
                  src->deregisterMem();
                  break;
        default:
                  CmiAbort("Invalid Callback Refnum\n");
                  break;
      }

      if(counter == 3 && !isBcast) { // do not contribute to the reduction for bcast operations as this array element is also a bcast recipient
        counter = 0;
        // contribute to a reduction
        contribute(testDoneCb);
      }
      delete msg;
    }

    // executed on half of the array elements
    void recvBuffer1(int *arr1, int *arr2, int *arr3, int length) {
      verifyValuesWithConstant(arr1, length, CONSTANT);
      verifyValuesWithConstant(arr2, length, CONSTANT);
      verifyValuesWithConstant(arr3, length, CONSTANT);

      contribute(testDoneCb);
    }

    void test2() {
      if(thisIndex < numElements/2) {
        assignValuesToIndex(buff, size);

        // Send the buffer buff to destIndex
        thisProxy[destIndex].recvBuffer2(CkSendBuffer(buff, sourceDoneCb1),
                                         CkSendBuffer(buff, sourceDoneCb2, CK_BUFFER_REG, CK_BUFFER_DEREG),
                                         CkSendBuffer(buff, sourceDoneCb3, CK_BUFFER_REG, CK_BUFFER_NODEREG), size);
      }
    }

    void recvBuffer2(int *&arr1, int *&arr2, int *&arr3, int &length, CkNcpyBufferPost *ncpyPost) {
      recvBuff1 = new int[length];
      recvBuff2 = new int[length];
      recvBuff3 = new int[length];

      arr1 = recvBuff1;
      arr2 = recvBuff2;
      arr3 = recvBuff3;

      // Do not modify deregMode of ncpyPost[0] (default is CK_BUFFER_DEREG)
      ncpyPost[1].deregMode = CK_BUFFER_DEREG;
      ncpyPost[2].deregMode = CK_BUFFER_NODEREG;
    }

    // executed on half of the array elements
    void recvBuffer2(int *arr1, int *arr2, int *arr3, int length) {
      verifyValuesWithIndex(recvBuff1, length);
      verifyValuesWithIndex(recvBuff2, length);
      verifyValuesWithIndex(recvBuff3, length);

      contribute(testDoneCb);
    }

    // only executed on Arr Index 0
    void test3() {
      isBcast = true;

      assignValuesToConstant(buff, size, CONSTANT);

      // Broadcast the buffer buff to the array proxy
      thisProxy.recvBuffer1(CkSendBuffer(buff, sourceDoneCb1),
                            CkSendBuffer(buff, sourceDoneCb2, CK_BUFFER_REG, CK_BUFFER_DEREG),
                            CkSendBuffer(buff, sourceDoneCb3, CK_BUFFER_REG, CK_BUFFER_NODEREG), size);
    }

    // only executed on Arr Index 0
    void test4() {
      isBcast = true;

      assignValuesToIndex(buff, size);

      // Broadcast the buffer buff to the array proxy
      thisProxy.recvBuffer2(CkSendBuffer(buff, sourceDoneCb1),
                            CkSendBuffer(buff, sourceDoneCb2, CK_BUFFER_REG, CK_BUFFER_DEREG),
                            CkSendBuffer(buff, sourceDoneCb3, CK_BUFFER_REG, CK_BUFFER_NODEREG), size);
    }

};

int testArr::size = 200;

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
  for(int i=0; i<size; i++) {
     CkAssert(arr[i] == constantVal);
  }
}

void verifyValuesWithIndex(int *arr, int size, int startIndex){
  for(int i=startIndex; i<size; i++)
     CkAssert(arr[i] == i);
}

#include "entry_method_api.def.h"
