#include "simple_put.decl.h"
#include <assert.h>

CProxy_main mainProxy;
class main : public CBase_main
{
  CProxy_Ping1 arr1;
  int count;
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m)
  {
    if(CkNumPes()>2) {
      CkPrintf("Run this program on 1 or 2 processors only.\n");
      CkExit(1);
    }
    if(m->argc !=2 ) {
      CkPrintf("Usage: ./simple_put <array size>\n");
      CkExit(1);
    }
    int size = atoi(m->argv[1]);
    mainProxy = thisProxy;
    delete m;
    arr1 = CProxy_Ping1::ckNew(size, 2);
    count = 0;
    arr1[1].start();
  };

  void maindone(){
    count++;
    if(count == 2) {
      CkExit();
    }
  };
};

template<class T>
void compareArray(T *&aArr, T *&bArr, int size){
  for(int i=0; i<size; i++)
    assert(aArr[i] == bArr[i]);
}

template<class T>
void assignValues(T *&arr, int size){
  // Allocation through CkRdmaAlloc
  arr = (T *)CkRdmaAlloc(size * sizeof(T));
  for(int i=0; i<size; i++)
     arr[i] = rand() % 100 + 1;
}

void assignCharValues(char *&arr, int size){
  // Allocation through CkRdmaAlloc
  arr = (char *)CkRdmaAlloc(size * sizeof(char));
  for(int i=0; i<size; i++)
     arr[i] = (char)(rand() % 125 + 1);
}

class Ping1 : public CBase_Ping1
{
  int *iArr1;
  char *cArr1;
  double *dArr1;
  int size;
  int otherIndex, cbCounter, valCounter;
  CkCallback cb;

public:
  Ping1(int size)
  {
    this->size = size;

    if(thisIndex == 0) {
      // original arrays that contains data
      assignValues(iArr1, size);
      assignValues(dArr1, size);
      assignCharValues(cArr1, size);
      // Set PUT Sender callback
      cb = CkCallback(CkIndex_Ping1::putSenderDone(NULL), thisProxy[thisIndex]);
    } else {
      iArr1 = (int *)CkRdmaAlloc(size * sizeof(int));
      cArr1 = (char *)CkRdmaAlloc(size * sizeof(char));
      dArr1 = (double *)CkRdmaAlloc(size * sizeof(double));
      // Set PUT Receiver callback
      cb = CkCallback(CkIndex_Ping1::putReceiverDone(NULL), thisProxy[thisIndex]);
    }

    otherIndex = (thisIndex + 1) % 2;
    cbCounter = 0;
    valCounter = 0;
  }
  Ping1(CkMigrateMessage *m) {}

  // Executed on Index 1
  void start()
  {
    CkAssert(thisIndex == 1);
    CkNcpyBuffer myDest1(iArr1, size*sizeof(int), cb, CK_BUFFER_PREREG);
    CkNcpyBuffer myDest2(dArr1, size*sizeof(double), cb, CK_BUFFER_PREREG);
    CkNcpyBuffer myDest3(cArr1, size*sizeof(char), cb, CK_BUFFER_PREREG);

    // Send my destinations to Index 0; Index 0 performs Puts into these destinations
    thisProxy[otherIndex].recvNcpyInfo(myDest1, myDest2, myDest3);
  }

  // Executed on Index 0 (which calls put)
  void putSenderDone(CkDataMsg *m){
    CkAssert(thisIndex == 0);
    cbCounter++;

    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *src = (CkNcpyBuffer *)(m->data);
    src->deregisterMem(); // in PREREG mode, actual de-registration is not performed
    // the above API call is only for demonstration and testing
    delete m;

    if(cbCounter == 3) {
      CkPrintf("[%d][%d][%d] Put Source Done\n", thisIndex, CkMyPe(), CkMyNode());
      sendValidationData();
    }
  }

  // Executed on Index 1 (which receives data from put)
  void putReceiverDone(CkDataMsg *m){
    CkAssert(thisIndex == 1);
    cbCounter++;

    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *dest = (CkNcpyBuffer *)(m->data);
    dest->deregisterMem(); // in PREREG mode, actual de-registration is not performed
    // the above API call is only for demonstration and testing
    delete m;

    if(cbCounter == 3) {
      CkPrintf("[%d][%d][%d] Put Destination Done\n", thisIndex, CkMyPe(), CkMyNode());
      thisProxy[otherIndex].sendValidationData();
    }
  }

  // Executed on Index 0
  void sendValidationData() {
    CkAssert(thisIndex == 0);
    valCounter++;
    if(valCounter == 2) {
      thisProxy[otherIndex].validatePutData(iArr1, dArr1, cArr1, size);
      CkRdmaFree(iArr1);
      CkRdmaFree(dArr1);
      CkRdmaFree(cArr1);
      mainProxy.maindone();
    }
  }

  // Executed on Index 0
  void recvNcpyInfo(CkNcpyBuffer dest1, CkNcpyBuffer dest2, CkNcpyBuffer dest3)
  {
    CkAssert(thisIndex == 0);
    // Create nocopy sources for me to Put into
    CkNcpyBuffer mySrc1(iArr1, size*sizeof(int), cb, CK_BUFFER_PREREG);
    CkNcpyBuffer mySrc2(dArr1, size*sizeof(double), cb, CK_BUFFER_PREREG);
    CkNcpyBuffer mySrc3(cArr1, size*sizeof(char), cb, CK_BUFFER_PREREG);

    // Perform Puts from my sources into Index 1's destinations
    mySrc1.put(dest1);
    mySrc2.put(dest2);
    mySrc3.put(dest3);
  }

  // Executed on Index 1
  void validatePutData(int *iArr2, double *dArr2, char *cArr2, int size)
  {
    CkAssert(thisIndex == 1);
    compareArray(iArr1, iArr2, size);
    compareArray(dArr1, dArr2, size);
    compareArray(cArr1, cArr2, size);
    CkPrintf("[%d][%d][%d] Put Validated! \n", thisIndex, CkMyPe(), CkMyNode());
    CkRdmaFree(iArr1);
    CkRdmaFree(dArr1);
    CkRdmaFree(cArr1);
    mainProxy.maindone();
  }

};

#include "simple_put.def.h"
