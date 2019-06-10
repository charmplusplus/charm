#include "zerocopy_with_qd.decl.h"

#define DEBUG(x) //x

int numElements;
CProxy_Main mProxy;

class Main : public CBase_Main {
  int srcCompletedCounter;
  int destCompletedCounter;
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

      srcCompletedCounter = 0;
      destCompletedCounter = 0;
      mProxy = thisProxy;

      CProxy_testArr arr1 = CProxy_testArr::ckNew(numElements);
      CkStartQD(CkCallback(CkIndex_Main::qdReached(), mProxy));
    };

    void qdReached() {
      CkPrintf("[%d][%d][%d] Quiescence has been reached srcCompleted:%d, destCompleted:%d, 3(numElements/2)=%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), srcCompletedCounter, destCompletedCounter, 3*numElements/2);
      CkAssert(srcCompletedCounter == destCompletedCounter);
      CkAssert(srcCompletedCounter == 3*numElements/2);
      CkExit();
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
