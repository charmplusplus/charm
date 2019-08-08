#include "simpleZeroCopy.decl.h"
#include <assert.h>

//Set DEBUG(x) to x to see the debug messages
//#define DEBUG(x) x
#define DEBUG(x)
#define LBPERIOD_ITER 5
#define TOTAL_ITER    40

int numElements;

//Main chare
class Main : public CBase_Main{
  public:
    Main(CkArgMsg *m){
      if(m->argc!=2){
        ckout<<"Usage: zerocopy <numelements>"<<endl;
        CkExit(1);
      }
      numElements = atoi(m->argv[1]);
      delete m;
      if(numElements%2 != 0){
        ckout<<"Argument <numelements> should be even"<<endl;
        CkExit(1);
      }

      CProxy_RRMap rrMap = CProxy_RRMap::ckNew();
      CkArrayOptions opts(numElements);
      opts.setMap(rrMap);
      CProxy_zerocopyObject zerocopyObj = CProxy_zerocopyObject::ckNew(opts);
      zerocopyObj.testZeroCopy(thisProxy);
    }

    void done(){
      CkPrintf("sdagRun: completed\nAll sending completed and result validated\n");
      CkExit();
    }
};

template<class T>
void assignValues(T *&arr, int size, int iter){
  for(int i=0; i<size; i++)
    arr[i] = i + iter;
}

template<class T>
void verifyValues(T *arr, int size, int iter){
  for(int i=0; i<size; i++)
    CkAssert(arr[i] == i + iter);
}

//zerocopy object chare
class zerocopyObject : public CBase_zerocopyObject{
  int *iArr1, *iArr1Recv;
  int iSize1, dSize2;
  int destIndex, iter, num, j, index;
  int sdagZeroCopySentCounter, sdagZeroCopyRecvCounter;
  CkCallback sdagCb, compReductionCb, lbReductionCb;
  int idx_sdagZeroCopySent;;
  CProxy_Main mainProxy;

  public:
    zerocopyObject_SDAG_CODE
    zerocopyObject(){
      usesAtSync = true;
      destIndex = numElements - 1 - thisIndex;
      DEBUG(CkPrintf("[%d]  me - %d, my neighbour- %d \n", CkMyNode(), thisIndex, destIndex);)

      //counter for tracking sdagRecv send completions
      sdagZeroCopySentCounter = 0;

      //counter for tracking sdagRecv completions
      sdagZeroCopyRecvCounter = 0;

      iSize1 = 210;

      iArr1 = new int[iSize1]; // source array

      // Allocate a large block of memory for receiving data for each iteration
      iArr1Recv = new int[iSize1 * TOTAL_ITER]; // Parts of this array are posted for each iteration

      iter = 0;
      num = 1;
      j = 0;
      idx_sdagZeroCopySent = CkIndex_zerocopyObject::sdagZeroCopySent(NULL);
      sdagCb = CkCallback(idx_sdagZeroCopySent, thisProxy[thisIndex]);
      lbReductionCb = CkCallback(CkReductionTarget(zerocopyObject, BarrierDone), thisProxy);
      compReductionCb = CkCallback(CkReductionTarget(Main, done), mainProxy);
    }

    void pup(PUP::er &p){
      p|iter;
      p|destIndex;
      p|num;
      p|iSize1;
      p|dSize2;
      p|sdagZeroCopySentCounter;
      p|sdagZeroCopyRecvCounter;
      p|mainProxy;
      p|sdagCb;
      p|lbReductionCb;
      p|compReductionCb;

      // sdagRun only uses iArr1 and dArr2
      // other others needn't be pupped/unpupped
      if (p.isUnpacking()){
        iArr1 = new int[iSize1];
        iArr1Recv = new int[iSize1 * TOTAL_ITER];
        j=0;

        CkAssert(sdagZeroCopyRecvCounter == 0);
        CkAssert(sdagZeroCopySentCounter == 0);
      }
      p(iArr1, iSize1);
      p(iArr1Recv, iSize1 * TOTAL_ITER);

      if(p.isPacking()) {
        delete [] iArr1;
        delete [] iArr1Recv;
        CkAssert(sdagZeroCopyRecvCounter == 0);
        CkAssert(sdagZeroCopySentCounter == 0);
      }
    }

    ~zerocopyObject() {}

    zerocopyObject(CkMigrateMessage *m){}

    void sdagZeroCopySent(CkDataMsg *m){
      // increment on completing the send of an zerocopy parameter in sdagRecv
      sdagZeroCopySentCounter++;
      CkAssert(sdagZeroCopySentCounter == 1);

      // Get access to the array information sent via zerocopy
      CkNcpyBuffer *src = (CkNcpyBuffer *)(m->data);

      void *ptr = (void *)(src->ptr); // do not free pointer as it is used in the next iteration as well

      delete m;

      // check that all sends and recvs have completed and then advance
      if(sdagZeroCopySentCounter == num && sdagZeroCopyRecvCounter == num)
        nextStep();
    }

    void testZeroCopy(CProxy_Main mProxy){
      mainProxy = mProxy;

      thisProxy[thisIndex].sdagRun();
    }

    void nextStep() {
      // reset the completion counters
      sdagZeroCopyRecvCounter = 0;
      sdagZeroCopySentCounter = 0;

      if(thisIndex == 0)
          CkPrintf("sdagRun: Iteration %d completed\n", iter);

      if(iter < TOTAL_ITER)
        thisProxy[thisIndex].sdagRun();
      else
        contribute(compReductionCb);
    }

    void sdagRecv(int index, int &n1, int *& ptr1, CkNcpyBufferPost *ncpyPost) {
      int *recvBuffer = iArr1Recv + (index - 1)*n1;
      ptr1 = recvBuffer;

      CkAssert(n1 == iSize1);

      //CkAssert(index == iter);
      // NOTE: The same arrays are used to receive the data for all the 'num' sdag iterations and
      // the 'TOTAL_ITER' application iterations. This is entirely for the purpose of demonstration
      // and results in the same array being overwritten. It is important to note that messages can
      // be out of order and this could cause correctness issues in real applications if the receiver
      // doesn't receive the arrays correctly.
    }
};

#include "simpleZeroCopy.def.h"
