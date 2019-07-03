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
void compareArray(T *&aArr, T *&bArr, int size, int startIdx=0){
  for(int i=0; i<size; i++)
    assert(aArr[i] == bArr[i+startIdx]);
}

template<class T>
void copyArray(T *&dest, T *&src, int size){
  memcpy(dest,src,size*sizeof(T));
}


template<class T>
void allocateAndCopyArray(T *&dest, T *&src, int size){
  dest = new T[size];
  copyArray(dest, src, size);
}

template<class T>
void assignValues(T *&arr, int size){
  arr = new T[size];
  for(int i=0; i<size; i++)
    arr[i] = i;
}

void assignCharValues(char *&arr, int size){
  arr = new char[size];
  for(int i=0; i<size; i++)
     arr[i] = (char)(rand() % 125 + 1);
}

//zerocopy object chare
class zerocopyObject : public CBase_zerocopyObject{
  int *iArr1,*iArr1copy;
  int iSize1, dSize2;
  int destIndex, iter, num, j;
  int sdagZeroCopySentCounter, sdagZeroCopyRecvCounter;
  CkCallback sdagCb;
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

//      iArr1 = NULL;
      iSize1 = 210;


      assignValues(iArr1, iSize1);

      // copy arrays for comparison
      allocateAndCopyArray(iArr1copy, iArr1, iSize1);

      iter = 1;
      num = 1;
      j = 0;
      idx_sdagZeroCopySent = CkIndex_zerocopyObject::sdagZeroCopySent(NULL);
      sdagCb = CkCallback(idx_sdagZeroCopySent, thisProxy[thisIndex]);
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

      // sdagRun only uses iArr1 and dArr2
      // other others needn't be pupped/unpupped
      if (p.isUnpacking()){
        iArr1 = new int[iSize1];
        iArr1copy = new int[iSize1];
        CmiPrintf("[%d][%d][%d][%d] Allocating %p in unpacker\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex, iArr1);
        j=0;
      }
      p(iArr1, iSize1);
      p(iArr1copy, iSize1);

      if(p.isPacking()) {
        //delete [] dArr2;
        //delete [] iArr1;
      }
    }

    ~zerocopyObject() {
     CmiPrintf("[%d][%d][%d][%d] Freeing %p in destructor and counter1=%d, counter2=%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex, iArr1, sdagZeroCopySentCounter, sdagZeroCopyRecvCounter);
      // delete everytime after migration as they are pupped to be used for sdagRun
      //delete [] dArr2;
      delete [] iArr1;
      //delete [] iArr1copy;
      //delete [] dArr2copy;
    }

    zerocopyObject(CkMigrateMessage *m){}

    void sdagZeroCopySent(CkDataMsg *m){
      // increment on completing the send of an zerocopy parameter in sdagRecv
      sdagZeroCopySentCounter++;

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
      CmiPrintf("[%d][%d][%d][%d] **************************** All sends and recvs of Iter %d completed ***************************\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex, iter);
      // reset the completion counters
      sdagZeroCopyRecvCounter = 0;
      sdagZeroCopySentCounter = 0;

      if(thisIndex == 0)
          CkPrintf("sdagRun: Iteration %d completed\n", iter);

      //increase iteration and continue
      iter++;

      //load balance
      if(iter % LBPERIOD_ITER == 0)
        AtSync();
      else if(iter<= TOTAL_ITER)
        thisProxy[thisIndex].sdagRun();
      else {
        CkCallback reductionCb(CkReductionTarget(Main, done), mainProxy);
        contribute(reductionCb);
      }
    }

    void sdagRecv(int iter, int &n1, int *& ptr1, CkNcpyBufferPost *ncpyPost) {
      ptr1 = iArr1;

      CkAssert(n1 == iSize1);
      // NOTE: The same arrays are used to receive the data for all the 'num' sdag iterations and
      // the 'TOTAL_ITER' application iterations. This is entirely for the purpose of demonstration
      // and results in the same array being overwritten. It is important to note that messages can
      // be out of order and this could cause correctness issues in real applications if the receiver
      // doesn't receive the arrays correctly.
      CmiPrintf("[%d][%d][%d][%d] ===== application posting %p in RECV with iter %d=====\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex, iArr1, iter);
    }

    void ResumeFromSync() {
      thisProxy[thisIndex].sdagRun();
    }
};

#include "simpleZeroCopy.def.h"
