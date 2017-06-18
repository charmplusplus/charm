#include "simpleRdma.decl.h"
#include <assert.h>

//Set DEBUG(x) to x to see the debug messages
//#define DEBUG(x) x
#define DEBUG(x)

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
      rdmaObj.testRdma();
      CkStartQD(CkCallback(CkIndex_Main::done(), thisProxy));
    }

    void done(){
      ckout<<"Sending completed and result validated"<<endl;
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
  int destIndex;
  int counter;
  CkCallback cb;
  int idx_rdmaSent;

  public:
    rdmaObject(){
      destIndex = numElements - 1 - thisIndex;
      DEBUG(CkPrintf("[%d]  me - %d, my neighbour- %d \n", CkMyNode(), thisIndex, destIndex);)
      counter=0;
      iArr1 = NULL;
      iArr2 = NULL;
      dArr1 = NULL;
      dArr2 = NULL;
      cArr1 = NULL;
      idx_rdmaSent = CkIndex_rdmaObject::rdmaSent(NULL);
    }

    rdmaObject(CkMigrateMessage *m){}

    void rdmaSent(CkDataMsg *m){
      //to get access to the array sent via rdma
      void *ptr = *((void **)(m->data));
      free(ptr);
      delete m;
    }

    void testRdma(){
      if(thisIndex < numElements/2){
        iSize1 = 2100;
        iSize2 = 11;
        dSize1 = 4700;
        dSize2 = 79;
        cSize1 = 32;

        iOffset1 = 3;
        cOffset1 = 2;

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
      }
      else{
        copyArray(iArr1, ptr1, n1);
        copyArray(dArr1, ptr2, n2);
        copyArray(iArr2, ptr3, n3);
        copyArray(dArr2, ptr4, n4);
        thisProxy[destIndex].mixedSend(n1, iArr1, n2, rdma(dArr1,cb), n3, rdma(iArr2,cb), n4, dArr2);
      }
    }
};

#include "simpleRdma.def.h"
