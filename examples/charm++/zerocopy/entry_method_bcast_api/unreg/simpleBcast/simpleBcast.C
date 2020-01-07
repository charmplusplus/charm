#include "simpleBcast.decl.h"

#define DEBUG(x) //x

CProxy_Main mainProxy;

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
      CkAbort("Usage: ./simpleBcast <array-size>");
    } else if(m->argc == 2 ) {
      size = atoi(m->argv[1]);
    } else {
      size = CkNumPes() * 10; // default with 10 chare array elements per pe
    }

    counter = 0;
    mainProxy = thisProxy;

    // Create a chare array
    CProxy_zcArray arrProxy = CProxy_zcArray::ckNew(size);

    // Create a group
    CProxy_zcGroup grpProxy = CProxy_zcGroup::ckNew();

    // Create a nodegroup
    CProxy_zcNodegroup ngrpProxy = CProxy_zcNodegroup::ckNew();

    // allocate a large array
    int bufferSize = 2000000;
    int *buffer = new int[bufferSize];
    assignValues(buffer, bufferSize);

    // create a callback method
    int idx_zerocopySent = CkIndex_Main::zerocopySent(NULL);
    CkCallback cb = CkCallback(idx_zerocopySent, thisProxy);

    CkCallback doneCb = CkCallback(CkReductionTarget(Main, done), thisProxy);

    // invoking bcast on chare array
    arrProxy.recvLargeArray(CkSendBuffer(buffer, cb, CK_BUFFER_UNREG), bufferSize, doneCb);

    // invoking bcast on group
    grpProxy.recvLargeArray(CkSendBuffer(buffer, cb, CK_BUFFER_UNREG), bufferSize, doneCb);

    // invoking bcast on nodegroup
    ngrpProxy.recvLargeArray(CkSendBuffer(buffer, cb, CK_BUFFER_UNREG), bufferSize, doneCb);
  }

  void zerocopySent(CkDataMsg *m) {
    CkPrintf("[%d][%d][%d] Source callback invoked\n", CkMyPe(), CkMyNode(), CmiMyRank());
    done();
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
  public:
  zcArray() {}

  void recvLargeArray(int *ptr1, int n1, CkCallback doneCb) {
    DEBUG(CkPrintf("[%d][%d] APP Chare array element received large array %p\n", thisIndex, CkMyPe(), ptr1);)
    checkArrValues(ptr1, n1);

    contribute(doneCb);
  }
};

class zcGroup : public CBase_zcGroup {
  public:
  zcGroup() {}

  void recvLargeArray(int *ptr1, int n1, CkCallback doneCb) {
    DEBUG(CkPrintf("[%d][%d] APP Group element received large array %p\n", thisIndex, CkMyPe(), ptr1);)
    checkArrValues(ptr1, n1);

    contribute(doneCb);
  }
};

class zcNodegroup : public CBase_zcNodegroup {
  public:
  zcNodegroup() {}

  void recvLargeArray(int *ptr1, int n1, CkCallback doneCb) {
    DEBUG(CkPrintf("[%d][%d] APP Nodegroup element received large array %p\n", thisIndex, CkMyPe(), ptr1);)
    checkArrValues(ptr1, n1);

    contribute(doneCb);
  }
};

#include "simpleBcast.def.h"
