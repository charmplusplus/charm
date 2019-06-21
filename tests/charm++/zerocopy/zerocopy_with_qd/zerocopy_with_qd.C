#include "zerocopy_with_qd.decl.h"

#define DEBUG(x) //x

int arr_size;
int vec_size;

int num_arr1[2000000];
std::vector<int> num_vec1;

int numElements;
CProxy_Main mProxy;

class Main : public CBase_Main {
  int testIndex;
  int srcCompletedCounter;
  int destCompletedCounter;
  CProxy_testArr arr1;
  bool reductionCompleted;

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

      arr_size = 2000000;
      vec_size = 2000000;
      mProxy = thisProxy;

      for(int i=0; i<arr_size; i++) num_arr1[i] = i;
      for(int i=0; i<vec_size; i++) num_vec1.push_back(i);

      reductionCompleted = false;

      srcCompletedCounter = 0;
      destCompletedCounter = 0;
      mProxy = thisProxy;

      testIndex = 1;

      arr1 = CProxy_testArr::ckNew(numElements);

      // Start QD
      CkStartQD(CkCallback(CkIndex_Main::qdReached(), mProxy));
    };

    void done() {
      CkPrintf("[%d][%d][%d] Reduction completed\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
      reductionCompleted = true;
    }

    void qdReached() {

      switch(testIndex) {
        case 1 :  // RO Bcast QD reached
                  CkAssert(reductionCompleted == true);
                  CkPrintf("[%d][%d][%d] Test 1: QD has been reached for RO Variable Bcast\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
                  // Begin Direct API Test
                  testIndex++;

                  arr1.testDirectApi();

                  // Start QD again for next test
                  CkStartQD(CkCallback(CkIndex_Main::qdReached(), mProxy));
                  break;

        case 2 :  // Direct API QD reached

                  CkAssert(srcCompletedCounter == destCompletedCounter);
                  CkAssert(srcCompletedCounter == 3*numElements/2);
                  CkPrintf("[%d][%d][%d] Test 2: QD has been reached for Direct API\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
                  CkExit();
                  break;

        default: // Invalid
                 CmiAbort("Test Index Invalid\n");
                 break;
      }
    }

    void zcSrcCompleted(CkDataMsg *m) {
      srcCompletedCounter++;
      DEBUG(CkPrintf("[%d][%d][%d] srcCompleted:%d, completed:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), srcCompletedCounter, CkGetRefNum(m));)
    }

    void zcDestCompleted(CkDataMsg *m) {
      destCompletedCounter++;
      DEBUG(CkPrintf("[%d][%d][%d] destCompleted:%d, completed:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), destCompletedCounter, CkGetRefNum(m));)
    }
};

class testArr : public CBase_testArr {
  int destIndex, size1, size2, size3;
  char *buff1, *buff2, *buff3;
  public:
    testArr() {
      DEBUG(CkPrintf("[%d][%d][%d] testArr element create %d \n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex);)
      destIndex = numElements - 1 - thisIndex;

      size1 = 2001;
      size2 = 67;
      size3 = 4578;

      buff1 = new char[size1];
      buff2 = new char[size2];
      buff3 = new char[size3];

      for(int i=0; i<arr_size; i++) CkAssert(num_arr1[i] == i);
      for(int i=0; i<vec_size; i++) CkAssert(num_vec1[i] == i);

      CkCallback cb(CkReductionTarget(Main, done), mProxy);
      contribute(cb);
    }

    void testDirectApi() {
      if(thisIndex < numElements/2) {
        CkCallback srcCompletionCb(CkIndex_Main::zcSrcCompleted(NULL),
                                   mProxy);
        // Create CkNcpyBuffer objects  and send it to the other side
        srcCompletionCb.setRefNum(thisIndex);
        CkNcpyBuffer src1(buff1, size1, srcCompletionCb);
        CkNcpyBuffer src2(buff2, size2, srcCompletionCb);
        CkNcpyBuffer src3(buff3, size3, srcCompletionCb);

        thisProxy[destIndex].recvBufferInfo(src1, src2, src3);
      }
    }

    // executed on half of the array elements
    void recvBufferInfo(CkNcpyBuffer src1, CkNcpyBuffer src2, CkNcpyBuffer src3) {
      // Create CkNcpyBuffer objects to serve as destinations and perform get on the data
        CkCallback destCompletionCb(CkIndex_Main::zcDestCompleted(NULL),
                                   mProxy);
        destCompletionCb.setRefNum(thisIndex);
        // Create CkNcpyBuffer objects  and send it to the other side
        CkNcpyBuffer dest1(buff1, size1, destCompletionCb);
        dest1.get(src1);

        CkNcpyBuffer dest2(buff2, size2, destCompletionCb);
        dest2.get(src2);

        CkNcpyBuffer dest3(buff3, size3, destCompletionCb);
        dest3.get(src3);

        DEBUG(CkPrintf("[%d][%d][%d] Completed launching Gets %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex);)
    }
};
#include "zerocopy_with_qd.def.h"
