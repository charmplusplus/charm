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
      reductionCompleted = true;
    }

    void qdReached() {

      switch(testIndex) {
        case 1 :  // RO Bcast QD reached
                  CkAssert(reductionCompleted == true);
                  CkPrintf("[%d][%d][%d] Test 1: QD has been reached for RO Variable Bcast\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
                  // Begin Direct API Test
                  testIndex++;

                  reductionCompleted = false;


                  arr1.testDirectApi();

                  // Start QD again for next test
                  CkStartQD(CkCallback(CkIndex_Main::qdReached(), mProxy));
                  break;

        case 2 :  // Direct API QD reached

                  CkAssert(srcCompletedCounter == destCompletedCounter);
                  CkAssert(srcCompletedCounter == 3*numElements/2);
                  CkPrintf("[%d][%d][%d] Test 2: QD has been reached for Direct API\n", CmiMyPe(), CmiMyNode(), CmiMyRank());


                  // Reset callback counters
                  srcCompletedCounter = destCompletedCounter = 0;

                  testIndex++;
                  arr1.testEmP2pSendApi();

                  CkStartQD(CkCallback(CkIndex_Main::qdReached(), mProxy));
                  break;

        case 3 :  // EM Send API QD reached
                  CkAssert(srcCompletedCounter == 3*numElements/2);
                  CkAssert(reductionCompleted == true);
                  CkPrintf("[%d][%d][%d] Test 3: QD has been reached for EM Send API\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
                  testIndex++;

                  reductionCompleted = false;
                  // Reset callback counters
                  srcCompletedCounter = destCompletedCounter = 0;

                  arr1.testEmP2pPostApi();

                  CkStartQD(CkCallback(CkIndex_Main::qdReached(), mProxy));
                  break;

        case 4 :  // EM Post API QD reached
                  CkAssert(srcCompletedCounter == 3*numElements/2);
                  CkAssert(reductionCompleted == true);
                  CkPrintf("[%d][%d][%d] Test 4: QD has been reached for EM Post API\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
                  CkExit();
                  break;

        default:  // Invalid
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

  CkCallback reductionCb, srcCompletionCb, destCompletionCb;

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

      reductionCb = CkCallback(CkReductionTarget(Main, done), mProxy);

      srcCompletionCb = CkCallback(CkIndex_Main::zcSrcCompleted(NULL), mProxy);

      destCompletionCb = CkCallback(CkIndex_Main::zcDestCompleted(NULL), mProxy);

      // Perform a reduction across all chare array elements to ensure completion of
      // RO transfer and constructor execution
      contribute(reductionCb);
    }

    void testDirectApi() {
      if(thisIndex < numElements/2) {

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

    void testEmP2pSendApi() {
      if(thisIndex < numElements/2) {
        // Create CkNcpyBuffer objects  and send it to the other side
        srcCompletionCb.setRefNum(thisIndex);

        thisProxy[destIndex].recvEmSendApiBuffer(CkSendBuffer(buff1, srcCompletionCb), size1,
                                                 CkSendBuffer(buff2, srcCompletionCb), size2,
                                                 CkSendBuffer(buff3, srcCompletionCb), size3);

        DEBUG(CkPrintf("[%d][%d][%d] Completed sending nocopy buffers %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex);)
        // Perform a reduction across all chare array elements to ensure that EM Send API
        // sends have been completed by elements with indices < numElements/2
        contribute(reductionCb);
      }
    }

    void recvEmSendApiBuffer(char *buff1, int size1, char *buff2, int size2, char *buff3, int size3) {
      DEBUG(CkPrintf("[%d][%d][%d] Received nocopy buffers %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex);)
      // Perform a reduction across all chare array elements to ensure that EM Send API has been received
      // by elements with indices > numElements/2
      contribute(reductionCb);
    }

    void testEmP2pPostApi() {
      if(thisIndex < numElements/2) {
        // Create CkNcpyBuffer objects  and send it to the other side
        srcCompletionCb.setRefNum(thisIndex);

        thisProxy[destIndex].recvEmPostApiBuffer(CkSendBuffer(buff1, srcCompletionCb), size1,
                                                 CkSendBuffer(buff2, srcCompletionCb), size2,
                                                 CkSendBuffer(buff3, srcCompletionCb), size3);

        DEBUG(CkPrintf("[%d][%d][%d] Completed sending nocopypost buffers %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex);)
        // Perform a reduction across all chare array elements to ensure that EM Send API
        // sends have been completed by elements with indices < numElements/2
        contribute(reductionCb);
      }
    }

    void recvEmPostApiBuffer(char *&buff1, int &size1, char *&buff2, int &size2, char *&buff3, int &size3, CkNcpyBufferPost *ncpyPost) {
      // use member variable buffers (buff1, buff2, buff3) as recipient buffers
      buff1 = this->buff1;
      buff2 = this->buff2;
      buff3 = this->buff3;
    }

    void recvEmPostApiBuffer(char *buff1, int size1, char *buff2, int size2, char *buff3, int size3) {
      DEBUG(CkPrintf("[%d][%d][%d] Received nocopypost buffers %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex);)
      // Perform a reduction across all chare array elements to ensure that EM Send API has been received
      // by elements with indices > numElements/2
      contribute(reductionCb);
    }



};
#include "zerocopy_with_qd.def.h"
