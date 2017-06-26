#include "simpleRdma.decl.h"
#include <assert.h>

//Set DEBUG(x) to x to see the debug messages
//#define DEBUG(x) x
#define DEBUG(x)
#define LBPERIOD_ITER 5

int numElements;

//Main chare
class Main : public CBase_Main{
  public:
    Main(CkArgMsg *m){
      if(m->argc!=2){
        ckout<<"Usage: rdma <numelements>"<<endl;
        CkExit();
      }
      numElements = atoi(m->argv[1]);
      delete m;
      if(numElements%2 != 0){
        ckout<<"Argument <numelements> should be even"<<endl;
        CkExit();
      }

      CProxy_RRMap rrMap = CProxy_RRMap::ckNew();
      CkArrayOptions opts(numElements);
      opts.setMap(rrMap);
      CProxy_rdmaObject rdmaObj = CProxy_rdmaObject::ckNew(opts);
      rdmaObj.testRdma(thisProxy);
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
    free(dest);
  dest = new T[size];
  memcpy(dest,src,size*sizeof(T));
}

template<class T>
void assignValues(T *&arr, int size){
  arr = new T[size];
  for(int i=0; i<size; i++)
     arr[i] = rand() % 100 + 1;
}

void assignCharValues(char *&arr, int size){
  arr = new char[size];
  for(int i=0; i<size; i++)
     arr[i] = (char)(rand() % 125 + 1);
}

//rdma object chare
class rdmaObject : public CBase_rdmaObject{
  int *iArr1, *iArr2;
  double *dArr1, *dArr2;
  char *cArr1;
  int iSize1, iSize2, dSize1, dSize2, cSize1, iOffset1, cOffset1;
  int destIndex, iter, num, j, counter;
  bool firstMigrationPending;
  CkCallback cb;
  int idx_rdmaSent;
  CProxy_Main mainProxy;

  public:
    rdmaObject_SDAG_CODE
    rdmaObject(){
      usesAtSync = true;
      destIndex = numElements - 1 - thisIndex;
      DEBUG(CkPrintf("[%d]  me - %d, my neighbour- %d \n", CkMyNode(), thisIndex, destIndex);)
      counter = 0;
      iArr1 = NULL;
      iArr2 = NULL;
      dArr1 = NULL;
      dArr2 = NULL;
      cArr1 = NULL;
      iter = 1;
      num = 4;
      j = 0;
      firstMigrationPending = true;
      idx_rdmaSent = CkIndex_rdmaObject::rdmaSent(NULL);
    }

    void pup(PUP::er &p){
      p|iter;
      p|destIndex;
      p|cb;
      p|num;
      p|iSize1;
      p|dSize2;
      p|counter;
      p|mainProxy;

      // sdagRun only uses iArr1 and dArr2
      // other others needn't be pupped/unpupped
      if (p.isUnpacking()){
        iArr1 = new int[iSize1];
        dArr2 = new double[dSize2];
        j=0;
        firstMigrationPending = false;
      }
      p(iArr1, iSize1);
      p(dArr2, dSize2);
    }

    ~rdmaObject() {
      if(firstMigrationPending) {
        // delete on first migration on all chares
        delete [] cArr1;

        if(thisIndex < numElements/2) {
          // delete on first migration on the first set of chares
          // as it is deleted in the callback on the other set
          delete [] iArr2;
          delete [] dArr1;
        }

      }
      // delete everytime after migration as they are pupped to be used for sdagRun
      delete [] dArr2;
      delete [] iArr1;
    }

    rdmaObject(CkMigrateMessage *m){}

    void rdmaSent(CkDataMsg *m){
      //to get access to the array sent via rdma
      void *ptr = *((void **)(m->data));
      free(ptr);
      delete m;

      if(++counter == 2)
        thisProxy[thisIndex].sdagRun();
    }

    void testRdma(CProxy_Main mProxy){
      iSize1 = 210;
      iSize2 = 11;
      dSize1 = 4700;
      dSize2 = 79;
      cSize1 = 32;

      iOffset1 = 3;
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
      cb = CkCallback(idx_rdmaSent, thisProxy[thisIndex]);
    }

    void send(int n1, int *ptr1, int n2, double *ptr2, int n3, char *ptr3){
      if(thisIndex < numElements/2){
        compareArray(ptr1, iArr1, n1);
        compareArray(ptr2, dArr1, n2);
        compareArray(ptr3, cArr1, n3);
        DEBUG(ckout<<"["<<CkMyPe()<<"] "<<thisIndex<<"->"<<destIndex<<": Regular send completed"<<endl;)
        if(thisIndex == 0)
          CkPrintf("send: completed\n");
        thisProxy[destIndex].rdmaSend(iSize1-iOffset1, rdma(iArr1+iOffset1), dSize1, rdma(dArr1), cSize1-cOffset1, rdma(cArr1 + cOffset1));
      }
      else{
        thisProxy[destIndex].send(n1, ptr1, n2, ptr2, n3, ptr3);
      }
    }

    void rdmaSend(int n1, int *ptr1, int n2, double *ptr2, int n3, char *ptr3){
      if(thisIndex < numElements/2){
        compareArray(ptr1, iArr1, n1, iOffset1);
        compareArray(ptr2, dArr1, n2);
        compareArray(ptr3, cArr1, n3, cOffset1);
        DEBUG(ckout<<"["<<CkMyPe()<<"] "<<thisIndex<<"->"<<destIndex<<": Rdma send completed"<<endl;)
        if(thisIndex == 0)
          CkPrintf("rdmaSend: completed\n");
        thisProxy[destIndex].mixedSend(iSize1, iArr1, dSize1, rdma(dArr1), iSize2, rdma(iArr2), dSize2, dArr2);
      }
      else{
        copyArray(iArr1, ptr1, n1);
        copyArray(dArr1, ptr2, n2);
        copyArray(cArr1, ptr3, n3);
        thisProxy[destIndex].rdmaSend(n1, rdma(iArr1), n2, rdma(dArr1), n3, rdma(cArr1));
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
        thisProxy[destIndex].mixedSend(n1, iArr1, n2, rdma(dArr1, cb), n3, rdma(iArr2, cb), n4, dArr2);
      }
    }

    void ResumeFromSync() {
      thisProxy[thisIndex].sdagRun();
    }
};

#include "simpleRdma.def.h"
