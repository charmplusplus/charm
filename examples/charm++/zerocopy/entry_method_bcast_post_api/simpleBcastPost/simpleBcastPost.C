#include "simpleBcastPost.decl.h"

#define DEBUG(x) //x

CProxy_Main mainProxy;
int bufferSize;

template<class T>
void assignValues(T *&arr, int size){
  for(int i=0; i<size; i++)
     arr[i] = i;
}

template<class T>
void checkArrValues(T *&arr, int size){
  for(int i=0; i<size; i++)
     CkAssert(arr[i] == i);
}

class Main : public CBase_Main{
  int size;
  int counter;
  public:
  Main(CkArgMsg *m) {
    // Create an array of size received in arguments
    if(m->argc > 2) {
      // print error message
      CkAbort("Usage: ./simpleBcastPost <array-size>");
    } else if(m->argc == 2 ) {
      size = atoi(m->argv[1]);
    } else {
      size = CkNumPes() * 3; // default with 10 chare array elements per pe
    }

    delete m;

    counter = 0;
    mainProxy = thisProxy;

    // allocate a large array
    bufferSize = 5000;
    int *buffer = new int[bufferSize];
    assignValues(buffer, bufferSize);

    // Create a chare array
    CProxy_zcArray arrProxy = CProxy_zcArray::ckNew(size);

    // Create a group
    CProxy_zcGroup grpProxy = CProxy_zcGroup::ckNew();

    // Create a nodegroup
    CProxy_zcNodegroup ngrpProxy = CProxy_zcNodegroup::ckNew();

    // create a callback method
    int idx_zerocopySent = CkIndex_Main::zerocopySent(NULL);
    CkCallback cb = CkCallback(idx_zerocopySent, thisProxy);

    CkCallback doneCb = CkCallback(CkReductionTarget(Main, done), thisProxy);

    CkPrintf("[%d][%d][%d] Broadcasting source buffer %p with size %d\n", CkMyPe(), CkMyNode(), CmiMyRank(), buffer, bufferSize);

    // invoking bcast on chare array
    arrProxy.recvLargeArray(CkSendBuffer(buffer, cb), bufferSize, doneCb);

    // invoking bcast on group
    grpProxy.recvLargeArray(CkSendBuffer(buffer, cb), bufferSize, doneCb);

    // invoking bcast on nodegroup
    ngrpProxy.recvLargeArray(CkSendBuffer(buffer, cb), bufferSize, doneCb);
  }

  void zerocopySent(CkDataMsg *m) {
    CkPrintf("[%d][%d][%d] Source callback invoked\n", CkMyPe(), CkMyNode(), CmiMyRank());
    done();
    delete m;
  }

  void done() {
    // Wait for 3 reductions to complete: Chare Array, Group, Nodegroup and
    // 3 more calls from zerocopySent callback method on completion of sending the buffer
    if(++counter == 6) {
      CkPrintf("[%d][%d][%d] All operations completed\n", CkMyPe(), CkMyNode(), CmiMyRank());
      CkExit();
    }
  }
};

class zcArray : public CBase_zcArray {
  int *myBuffer;
  public:
  zcArray() {
    myBuffer = new int[bufferSize];
    DEBUG(CkPrintf("[%d][%d][%d][%d] Array element: constructed and allocated buffer is %p\n", CkMyPe(), CkMyNode(), CmiMyRank(), thisIndex, myBuffer);)
  }

  void recvLargeArray(int *&ptr1, int &n1, CkCallback doneCb, CkNcpyBufferPost) {
    DEBUG(CkPrintf("[%d][%d][%d][%d] Array element: recvLargeArray Post \n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex);)
    ptr1 = myBuffer;
    CkAssert(n1 == bufferSize);
    DEBUG(CkPrintf("[%d][%d][%d][%d] Array element: recvLargeArray Post done posted buffer is %p and size is %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex, myBuffer, bufferSize);)
  }

  void recvLargeArray(int *ptr1, int n1, CkCallback doneCb) {
    DEBUG(CkPrintf("[%d][%d][%d][%d] Array element: recvLargeArray Regular \n", CmiMyPe(), CmiMyNode(), CmiMyRank(), thisIndex);)
    checkArrValues(ptr1, n1);
    contribute(doneCb);
  }
};

class zcGroup : public CBase_zcGroup {
  int *myBuffer;
  public:
  zcGroup() {
    myBuffer = new int[bufferSize];
    DEBUG(CkPrintf("[%d][%d][%d] Group: constructed and allocated buffer is %p\n", CkMyPe(), CkMyNode(), CmiMyRank(), myBuffer);)
  }

  void recvLargeArray(int *&ptr1, int &n1, CkCallback doneCb, CkNcpyBufferPost) {
    DEBUG(CkPrintf("[%d][%d][%d] Group: recvLargeArray Post \n", CmiMyPe(), CmiMyNode(), CmiMyRank());)
    ptr1 = myBuffer;
    CkAssert(n1 == bufferSize);
    DEBUG(CkPrintf("[%d][%d][%d] Group: recvLargeArray Post done posted buffer is %p and size is %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), myBuffer, bufferSize);)
  }

  void recvLargeArray(int *ptr1, int n1, CkCallback doneCb) {
    DEBUG(CkPrintf("[%d][%d][%d] Group: recvLargeArray Regular \n", CmiMyPe(), CmiMyNode(), CmiMyRank());)
    checkArrValues(ptr1, n1);
    contribute(doneCb);
  }
};

class zcNodegroup : public CBase_zcNodegroup {
  int *myBuffer;
  public:
  zcNodegroup() {
    myBuffer = new int[bufferSize];
    DEBUG(CkPrintf("[%d][%d][%d] Nodegroup: constructed and allocated buffer is %p\n", CkMyPe(), CkMyNode(), CmiMyRank(), myBuffer);)
  }

  void recvLargeArray(int *&ptr1, int &n1, CkCallback doneCb, CkNcpyBufferPost) {
    DEBUG(CkPrintf("[%d][%d][%d] Nodegroup: recvLargeArray Post \n", CmiMyPe(), CmiMyNode(), CmiMyRank());)
    ptr1 = myBuffer;
    CkAssert(n1 == bufferSize);
    DEBUG(CkPrintf("[%d][%d][%d] Nodegroup: recvLargeArray Post done posted buffer is %p and size is %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), myBuffer, bufferSize);)
  }

  void recvLargeArray(int *ptr1, int n1, CkCallback doneCb) {
    DEBUG(CkPrintf("[%d][%d][%d] Nodegroup: recvLargeArray Regular \n", CmiMyPe(), CmiMyNode(), CmiMyRank());)
    checkArrValues(ptr1, n1);
    checkArrValues(myBuffer, n1);
    contribute(doneCb);
  }
};

#include "simpleBcastPost.def.h"
