#include "simple_get.decl.h"
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

class Ping1 : public CBase_Ping1
{
  int *iArr;
  int size;
  int otherIndex;
  CkCallback cb;

public:
  Ping1(int size)
  {
    this->size = size;

    iArr = new int[size];

    if(thisIndex == 0) {
      // assign values
      for(int i=0; i<size; i++)
        iArr[i] = i;

      // Set GET Sender callback
      cb = CkCallback(CkIndex_Ping1::getSenderDone(NULL), thisProxy[thisIndex]);

    } else {
      // Set GET Receiver callback
      cb = CkCallback(CkIndex_Ping1::getReceiverDone(NULL), thisProxy[thisIndex]);
    }

    otherIndex = (thisIndex + 1) % 2;
  }

  Ping1(CkMigrateMessage *m) {}

  // Executed on Index 0
  void start()
  {
    CkAssert(thisIndex == 0);
    CkNcpyBuffer mySrc(iArr, size*sizeof(int), cb);

    // Send my source to Index 1; Index 1 performs Get from this source
    thisProxy[otherIndex].recvNcpyInfo(mySrc);
  }

  // Executed on Index 0
  void getSenderDone(CkDataMsg *m){
    CkAssert(thisIndex == 0);

    CkPrintf("[%d][%d][%d] Inside Source callback     : Get is complete - ", thisIndex, CkMyPe(), CkMyNode());
    CkPrintf("De-registering and freeing source data\n");

    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *src = (CkNcpyBuffer *)(m->data);
    src->deregisterMem();

    delete m;

    delete [] iArr;

    mainProxy.maindone();
  }

  // Executed on Index 1 (which receives data from get)
  void getReceiverDone(CkDataMsg *m){
    CkAssert(thisIndex == 1);

    CkPrintf("[%d][%d][%d] Inside Destination callback: Get is complete - ", thisIndex, CkMyPe(), CkMyNode());
    CkPrintf("De-registering, validating and freeing received data\n");
    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *dest = (CkNcpyBuffer *)(m->data);
    dest->deregisterMem();

    delete m;

    validateData();
    delete [] iArr;
    mainProxy.maindone();
  }

  void validateData() {
    for(int i =0; i<size; i++)
      CkAssert(iArr[i] == i);
  }

  // Executed on Index 1
  void recvNcpyInfo(CkNcpyBuffer src)
  {
    CkAssert(thisIndex == 1);

    // Create nocopy destination for me to Get into
    CkNcpyBuffer myDest(iArr, size*sizeof(int), cb);

    // Perform Get from Index 0's sources into my destinations
    // CkNcpyCallbackMode::CB_INVOKE_NONINLINE is set to invoke receiver callback only in non-inline transfers (async RDMA call)
    CkNcpyStatus status = myDest.get(src, CkNcpyCallbackMode::CB_INVOKE_NONINLINE);

    if(status == CkNcpyStatus::complete) {
      CkPrintf("[%d][%d][%d] Inline                     : Get is complete - ", thisIndex, CkMyPe(), CkMyNode());
      CkPrintf("De-registering, validating and freeing received data\n");

      // de-register, validate and free data as get has completed
      myDest.deregisterMem();
      validateData();
      delete [] iArr;
      mainProxy.maindone();
    }
    else if(status == CkNcpyStatus::incomplete) {
      CkPrintf("[%d][%d][%d] Async RDMA call in progress: Get is still incomplete on the destination\n", thisIndex, CkMyPe(), CkMyNode());
      // async call in progress, reset of the operations in receiver callback
    }
  }
};

#include "simple_get.def.h"
