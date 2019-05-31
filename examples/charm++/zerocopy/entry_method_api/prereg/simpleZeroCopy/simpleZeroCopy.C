#include "simpleZeroCopy.decl.h"
#include <assert.h>

//Set DEBUG(x) to x to see the debug messages
//#define DEBUG(x) x
#define DEBUG(x)
#define LBPERIOD_ITER 5
#define MAX_ITER 40

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

    Main(CkMigrateMessage *m){}
};

template<class T>
void compareArray(T *&aArr, T *&bArr, int size, int startIdx=0){
  for(int i=0; i<size; i++)
    assert(aArr[i] == bArr[i+startIdx]);
}

template<class T>
void copyArray(T *&dest, T *&src, int size){
  if(dest != NULL)
    CkRdmaFree(dest);
  // Allocate using CkRdmaAlloc instead of new
  dest = (T *)CkRdmaAlloc(sizeof(T) * size);
  memcpy(dest,src,size*sizeof(T));
}

template<class T>
void assignValues(T *&arr, int size){
  //arr = new T[size];
  arr = (T *)CkRdmaAlloc(sizeof(T) * size);
  for(int i=0; i<size; i++)
     arr[i] = rand() % 100 + 1;
}

void assignCharValues(char *&arr, int size){
  //arr = new char[size];
  arr = (char *)CkRdmaAlloc(sizeof(char) * size);
  for(int i=0; i<size; i++)
     arr[i] = (char)(rand() % 125 + 1);
}

//zerocopy object chare
class zerocopyObject : public CBase_zerocopyObject{
  int *iArr1, *iArr2;
  double *dArr1, *dArr2;
  char *cArr1;
  int iSize1, iSize2, dSize1, dSize2, cSize1, iOffset1, cOffset1;
  int destIndex, iter, num, j;
  int mixedZeroCopySentCounter, sdagZeroCopySentCounter, sdagZeroCopyRecvCounter;
  bool firstMigrationPending;
  CkCallback cb, sdagCb;
  int idx_zerocopySent, idx_sdagZeroCopySent;;
  CProxy_Main mainProxy;

  public:
    zerocopyObject_SDAG_CODE
    zerocopyObject(){
      usesAtSync = true;
      destIndex = numElements - 1 - thisIndex;
      DEBUG(CkPrintf("[%d]  me - %d, my neighbour- %d \n", CkMyNode(), thisIndex, destIndex);)
      //counter for tracking mixedSend completions to initiate sdagRun
      mixedZeroCopySentCounter = 0;

      //counter for tracking sdagRecv send completions
      sdagZeroCopySentCounter = 0;

      //counter for tracking sdagRecv completions
      sdagZeroCopyRecvCounter = 0;
      iArr1 = NULL;
      iArr2 = NULL;
      dArr1 = NULL;
      dArr2 = NULL;
      cArr1 = NULL;
      iter = 1;
      num = 4;
      j = 0;
      firstMigrationPending = true;
      idx_zerocopySent = CkIndex_zerocopyObject::zerocopySent(NULL);
      idx_sdagZeroCopySent = CkIndex_zerocopyObject::sdagZeroCopySent(NULL);
      cb = CkCallback(idx_zerocopySent, thisProxy[thisIndex]);
      sdagCb = CkCallback(idx_sdagZeroCopySent, thisProxy[thisIndex]);
    }

    void pup(PUP::er &p){
      p|iter;
      p|destIndex;
      p|cb;
      p|num;
      p|iSize1;
      p|dSize2;
      p|mixedZeroCopySentCounter;
      p|sdagZeroCopySentCounter;
      p|sdagZeroCopyRecvCounter;
      p|mainProxy;
      p|sdagCb;

      // sdagRun only uses iArr1 and dArr2
      // other others needn't be pupped/unpupped
      if (p.isUnpacking()){
        iArr1 = (int *)CkRdmaAlloc(iSize1 * sizeof(int));
        dArr2 = (double *)CkRdmaAlloc(dSize2 * sizeof(double));
        j=0;
        firstMigrationPending = false;
      }
      p(iArr1, iSize1);
      p(dArr2, dSize2);
    }

    ~zerocopyObject() {
      if(firstMigrationPending) {
        // delete on first migration on all chares
        CkRdmaFree(cArr1);

        if(thisIndex < numElements/2) {
          // delete on first migration on the first set of chares
          // as it is deleted in the callback on the other set
          CkRdmaFree(iArr2);
          CkRdmaFree(dArr1);
        }

      }
      // delete everytime after migration as they are pupped to be used for sdagRun
      CkRdmaFree(dArr2);
      CkRdmaFree(iArr1);
    }

    zerocopyObject(CkMigrateMessage *m){}

    void zerocopySent(CkDataMsg *m){
      // Get access to the array information sent via zerocopy
      CkNcpyBuffer *src = (CkNcpyBuffer *)(m->data);
      //free((void *)(src->ptr));
      CkRdmaFree((void *)(src->ptr));
      delete m;

      if(++mixedZeroCopySentCounter == 2)
        thisProxy[thisIndex].sdagRun();
    }

    void sdagZeroCopySent(CkDataMsg *m){
      delete m;
      // increment on completing the send of an zerocopy parameter in sdagRecv
      sdagZeroCopySentCounter++;

      // check that all sends and recvs have completed and then advance
      if(sdagZeroCopySentCounter == 2*num && sdagZeroCopyRecvCounter == num)
        nextStep();
    }

    void testZeroCopy(CProxy_Main mProxy){
      iSize1 = 210;
      iSize2 = 11;
      dSize1 = 4700;
      dSize2 = 79;
      cSize1 = 32;

      iOffset1 = 100;
      cOffset1 = 2;

      mainProxy = mProxy;
      if(thisIndex < numElements/2){
        assignValues(iArr1, iSize1);
        assignValues(iArr2, iSize2);
        assignValues(dArr1, dSize1);
        assignValues(dArr2, dSize2);
        assignCharValues(cArr1, cSize1);
        thisProxy[destIndex].send(iSize1, iArr1, dSize1, dArr1, cSize1, cArr1);
      }
    }

    void send(int n1, int *ptr1, int n2, double *ptr2, int n3, char *ptr3){
      if(thisIndex < numElements/2){
        compareArray(ptr1, iArr1, n1);
        compareArray(ptr2, dArr1, n2);
        compareArray(ptr3, cArr1, n3);
        DEBUG(ckout<<"["<<CkMyPe()<<"] "<<thisIndex<<"->"<<destIndex<<": Regular send completed"<<endl;)
        if(thisIndex == 0)
          CkPrintf("send: completed\n");
        // cannot use PREREG mode for offset buffers
        thisProxy[destIndex].zerocopySend(iSize1-iOffset1, CkSendBuffer(iArr1+iOffset1), dSize1, CkSendBuffer(dArr1, CK_BUFFER_PREREG), cSize1-cOffset1, CkSendBuffer(cArr1 + cOffset1));
      }
      else{
        thisProxy[destIndex].send(n1, ptr1, n2, ptr2, n3, ptr3);
      }
    }

    void zerocopySend(int n1, int *ptr1, int n2, double *ptr2, int n3, char *ptr3){
      if(thisIndex < numElements/2){
        compareArray(ptr1, iArr1, n1, iOffset1);
        compareArray(ptr2, dArr1, n2);
        compareArray(ptr3, cArr1, n3, cOffset1);
        DEBUG(ckout<<"["<<CkMyPe()<<"] "<<thisIndex<<"->"<<destIndex<<": ZeroCopy send completed"<<endl;)
        if(thisIndex == 0)
          CkPrintf("zerocopySend: completed\n");
        thisProxy[destIndex].mixedSend(iSize1, iArr1, dSize1, CkSendBuffer(dArr1, CK_BUFFER_PREREG), iSize2, CkSendBuffer(iArr2, CK_BUFFER_PREREG), dSize2, dArr2);
      }
      else{
        copyArray(iArr1, ptr1, n1);
        copyArray(dArr1, ptr2, n2);
        copyArray(cArr1, ptr3, n3);
        thisProxy[destIndex].zerocopySend(n1, CkSendBuffer(iArr1, CK_BUFFER_PREREG), n2, CkSendBuffer(dArr1, CK_BUFFER_PREREG), n3, CkSendBuffer(cArr1, CK_BUFFER_PREREG));
      }
    }

    void mixedSend(int n1, int *ptr1, int n2, double *ptr2, int n3, int *ptr3, int n4, double *ptr4){
      if(thisIndex < numElements/2){
        compareArray(ptr1, iArr1, n1);
        compareArray(ptr2, dArr1, n2);
        compareArray(ptr3, iArr2, n3);
        compareArray(ptr4, dArr2, n4);
        DEBUG(ckout<<"["<<CkMyPe()<<"] "<<thisIndex<<"->"<<destIndex<<": Mixed send completed "<<endl;)
        if(thisIndex == 0)
          CkPrintf("mixedSend: completed\n");
        thisProxy[thisIndex].sdagRun();
      }
      else{
        copyArray(iArr1, ptr1, n1);
        copyArray(dArr1, ptr2, n2);
        copyArray(iArr2, ptr3, n3);
        copyArray(dArr2, ptr4, n4);
        thisProxy[destIndex].mixedSend(n1, iArr1, n2, CkSendBuffer(dArr1, cb, CK_BUFFER_PREREG), n3, CkSendBuffer(iArr2, cb, CK_BUFFER_PREREG), n4, dArr2);
      }
    }

    void nextStep() {
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
      else if(iter <= MAX_ITER)
        thisProxy[thisIndex].sdagRun();
      else {
        CkCallback reductionCb(CkReductionTarget(Main, done), mainProxy);
        contribute(reductionCb);
      }
    }

    void ResumeFromSync() {
      thisProxy[thisIndex].sdagRun();
    }
};

#include "simpleZeroCopy.def.h"
