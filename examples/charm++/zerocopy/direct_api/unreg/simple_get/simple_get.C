#include "simple_get.decl.h"
#include <assert.h>

#define ACK_DEBUG(x)  //CkPrintf x

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
      CkPrintf("Usage: ./simple_get <array size>\n");
      CkExit(1);
    }
    int size = atoi(m->argv[1]);
    mainProxy = thisProxy;
    delete m;
    arr1 = CProxy_Ping1::ckNew(size, 2);
    count = 0;
    arr1[0].start();
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
  arr = new T[size];
  for(int i=0; i<size; i++)
     arr[i] = rand() % 100 + 1;
}

void assignCharValues(char *&arr, int size){
  arr = new char[size];
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
      // Set GET Sender callback
      cb = CkCallback(CkIndex_Ping1::getSenderDone(NULL), thisProxy[thisIndex]);
    } else {
      iArr1 = new int[size];
      cArr1 = new char[size];
      dArr1 = new double[size];
      // Set GET Receiver callback
      cb = CkCallback(CkIndex_Ping1::getReceiverDone(NULL), thisProxy[thisIndex]);
    }

    otherIndex = (thisIndex + 1) % 2;
    cbCounter = 0;
    valCounter = 0;
  }
  Ping1(CkMigrateMessage *m) {}

  // Executed on Index 0
  void start()
  {
    CkAssert(thisIndex == 0);

    // arbitrary pointer pointing to valCounter
    const void *refPtr = &valCounter;

    cb.setRefNum(1);
    CkNcpyBuffer mySrc1(iArr1, size*sizeof(int), cb, CK_BUFFER_UNREG);
    ACK_DEBUG(("[%d][%d][%d] Setting source Ref: Buffer Ptr: %p, Reference Ptr: %p\n", thisIndex, CkMyPe(), CkMyNode(), iArr1, refPtr));
    mySrc1.setRef(refPtr);

    cb.setRefNum(2);
    CkNcpyBuffer mySrc2(dArr1, size*sizeof(double), cb, CK_BUFFER_UNREG);
    ACK_DEBUG(("[%d][%d][%d] Setting source Ref: Buffer Ptr: %p, Reference Ptr: %p\n", thisIndex, CkMyPe(), CkMyNode(), dArr1, refPtr));
    mySrc2.setRef(refPtr);

    cb.setRefNum(3);
    CkNcpyBuffer mySrc3(cArr1, size*sizeof(char), cb, CK_BUFFER_UNREG);
    ACK_DEBUG(("[%d][%d][%d] Setting source Ref: Buffer Ptr: %p, Reference Ptr: %p\n", thisIndex, CkMyPe(), CkMyNode(), cArr1, refPtr));
    mySrc3.setRef(refPtr);

    // Send my sources to Index 1; Index 1 performs Gets from these sources
    thisProxy[otherIndex].recvNcpyInfo(mySrc1, mySrc2, mySrc3);
  }

  // Executed on Index 0
  void getSenderDone(CkDataMsg *m){
    CkAssert(thisIndex == 0);
    cbCounter++;

    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *src = (CkNcpyBuffer *)(m->data);
    ACK_DEBUG(("[%d][%d][%d] In source callback : Buffer Ptr: %p, Reference Ptr: %p and refnum is %d\n", thisIndex, CkMyPe(), CkMyNode(), src->ptr, src->ref, CkGetRefNum(m)));

    int refNum = CkGetRefNum(m);

    void *srcPointer;
    switch(refNum) {
      case 1 : srcPointer = iArr1; break;
      case 2 : srcPointer = dArr1; break;
      case 3 : srcPointer = cArr1; break;
      default: CkAbort("Invalid value of cbCounter\n"); break;
    }

    // Verify that source pointer is equal to the buffer pointer returned
    CkAssert(srcPointer == src->ptr);

    // Verify that reference pointer is equal to the reference pointer returned
    CkAssert(&valCounter == src->ref);

    // Deregister the source object
    src->deregisterMem();

    if(cbCounter == 3) {
      CkPrintf("[%d][%d][%d] Get Source Done\n", thisIndex, CkMyPe(), CkMyNode());
      sendValidationData();
    }
    delete m;
  }

  // Executed on Index 1 (which receives data from get)
  void getReceiverDone(CkDataMsg *m){
    CkAssert(thisIndex == 1);
    cbCounter++;

    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *dest = (CkNcpyBuffer *)(m->data);
    ACK_DEBUG(("[%d][%d][%d] In destination callback : Buffer Ptr: %p, Reference Ptr: %p and refnum is %d\n", thisIndex, CkMyPe(), CkMyNode(), dest->ptr, dest->ref, CkGetRefNum(m)));

    int refNum = CkGetRefNum(m);

    void *destPointer;
    switch(refNum) {
      case 1 : destPointer = iArr1; break;
      case 2 : destPointer = dArr1; break;
      case 3 : destPointer = cArr1; break;
      default: CkAbort("Invalid value of cbCounter\n"); break;
    }

    // Verify that destination pointer is equal to the buffer pointer returned
    CkAssert(destPointer == dest->ptr);

    // Verify that reference pointer is equal to the reference pointer returned
    CkAssert(&valCounter == dest->ref);

    // Deregister the destination object
    dest->deregisterMem();

    if(cbCounter == 3) {
      CkPrintf("[%d][%d][%d] Get Destination Done\n", thisIndex, CkMyPe(), CkMyNode());
      thisProxy[otherIndex].sendValidationData();
    }
    delete m;
  }

  // Executed on Index 0
  void sendValidationData() {
    CkAssert(thisIndex == 0);
    valCounter++;
    if(valCounter == 2) {
      thisProxy[otherIndex].validateGetData(iArr1, dArr1, cArr1, size);
      delete [] iArr1;
      delete [] dArr1;
      delete [] cArr1;
      mainProxy.maindone();
    }
  }

  // Executed on Index 1
  void recvNcpyInfo(CkNcpyBuffer src1, CkNcpyBuffer src2, CkNcpyBuffer src3)
  {
    CkAssert(thisIndex == 1);

    // arbitrary pointer pointing to valCounter
    const void *refPtr = &valCounter;

    // Create nocopy destination for me to Get into
    cb.setRefNum(1);
    CkNcpyBuffer myDest1(iArr1, size*sizeof(int), cb, CK_BUFFER_UNREG);
    ACK_DEBUG(("[%d][%d][%d] Setting destination Ref: Buffer Ptr: %p, Reference Ptr: %p\n", thisIndex, CkMyPe(), CkMyNode(), iArr1, refPtr));
    myDest1.setRef(refPtr);

    cb.setRefNum(2);
    CkNcpyBuffer myDest2(dArr1, size*sizeof(double), cb, CK_BUFFER_UNREG);
    ACK_DEBUG(("[%d][%d][%d] Setting destination Ref: Buffer Ptr: %p, Reference Ptr: %p\n", thisIndex, CkMyPe(), CkMyNode(), dArr1, refPtr));
    myDest2.setRef(refPtr);

    cb.setRefNum(3);
    CkNcpyBuffer myDest3(cArr1, size*sizeof(char), cb, CK_BUFFER_UNREG);
    ACK_DEBUG(("[%d][%d][%d] Setting destination Ref: Buffer Ptr: %p, Reference Ptr: %p\n", thisIndex, CkMyPe(), CkMyNode(), cArr1, refPtr));
    myDest3.setRef(refPtr);

    // Perform Get from Index 0's sources into my destinations
    myDest1.get(src1);
    myDest2.get(src2);
    myDest3.get(src3);
  }

  // Executed on Index 1
  void validateGetData(int *iArr2, double *dArr2, char *cArr2, int size)
  {
    CkAssert(thisIndex == 1);
    compareArray(iArr1, iArr2, size);
    compareArray(dArr1, dArr2, size);
    compareArray(cArr1, cArr2, size);
    CkPrintf("[%d][%d][%d] Get Validated! \n", thisIndex, CkMyPe(), CkMyNode());
    delete [] iArr1;
    delete [] dArr1;
    delete [] cArr1;
    mainProxy.maindone();
  }

};

#include "simple_get.def.h"
