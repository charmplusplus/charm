#include "nodegroupTest.decl.h"
#include <assert.h>

int numElements;
CProxy_Main mProxy;

//Main chare
class Main : public CBase_Main{
  public:
    Main(CkArgMsg *m) {
      if(CkNumNodes() % 2 != 0){
        ckout<<"Run it on even number of processes"<<endl;
        CkExit(1);
      }
      numElements = CkNumNodes();
      mProxy = thisProxy;
      delete m;

      CProxy_zerocopyObject zerocopyObj = CProxy_zerocopyObject::ckNew();
      zerocopyObj.testZeroCopy();
    }

    void done(){
      CkPrintf("All sending completed and result validated\n");
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
  arr = (T *)CkRdmaAlloc(sizeof(T) * size);
  for(int i=0; i<size; i++)
     arr[i] = rand() % 100 + 1;
}

//zerocopy object chare
class zerocopyObject : public CBase_zerocopyObject{
  int *iArr1, *iArr1copy;
  int iSize1;
  int destIndex;

  CkCallback reductionCb;

  public:
    zerocopyObject(){
      destIndex = numElements - 1 - CkMyNode();
      int idx_maindone = CkIndex_Main::done();
      reductionCb = CkCallback(idx_maindone, mProxy);

      iSize1 = 210;

      if(CkMyNode() < numElements/2){
        assignValues(iArr1, iSize1);
        // copy arrays for comparison
        allocateAndCopyArray(iArr1copy, iArr1, iSize1);
      } else {
        iArr1 = (int *)CkRdmaAlloc(sizeof(int) * iSize1);
      }
    }

    zerocopyObject(CkMigrateMessage *m){}

    void testZeroCopy(){
      if(CkMyNode() < numElements/2){
        thisProxy[destIndex].zerocopySend(iSize1, CkSendBuffer(iArr1, CK_BUFFER_PREREG));
      }
    }

    void zerocopySend(int &n1, int *& ptr1, CkNcpyBufferPost *ncpyPost) {
      CkAssert(iArr1 != NULL);
      ptr1 = iArr1;

      ncpyPost[0].mode = CK_BUFFER_PREREG;
    }

    void zerocopySend(int n1, int *ptr1){
      if(CkMyNode() < numElements/2){
        compareArray(iArr1, iArr1copy, n1);
        if(CkMyNode() == 0)
          CkPrintf("zerocopySend: completed\n");
        //contribute to reduction to signal completion
        contribute(reductionCb);
      }
      else{
        int idx_zerocopySent = CkIndex_zerocopyObject::zerocopySent(NULL);
        CkCallback cb = CkCallback(idx_zerocopySent, thisProxy[CkMyNode()]);
        thisProxy[destIndex].zerocopySend(n1, CkSendBuffer(iArr1, cb, CK_BUFFER_PREREG));
      }
    }

    void zerocopySent(CkDataMsg *msg) {
      CkNcpyBuffer *src = (CkNcpyBuffer *)(msg->data);
      src->deregisterMem();
      CkRdmaFree((void *)src->ptr);
      delete msg;

      //contribute to reduction to signal completion
      contribute(reductionCb);
    }
};

#include "nodegroupTest.def.h"
