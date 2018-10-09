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

      // Set PUT Sender callback
      cb = CkCallback(CkIndex_Ping1::putSenderDone(NULL), thisProxy[thisIndex]);
    } else {
      // Set PUT Receiver callback
      cb = CkCallback(CkIndex_Ping1::putReceiverDone(NULL), thisProxy[thisIndex]);
    }

    otherIndex = (thisIndex + 1) % 2;
  }
  Ping1(CkMigrateMessage *m) {}

  // Executed on Index 1
  void start()
  {
    CkAssert(thisIndex == 1);
    CkNcpyBuffer myDest(iArr, size*sizeof(int), cb); // not using any mode uses CK_BUFFER_REG

    // Send my destination to Index 0; Index 0 performs Put into this destination
    thisProxy[otherIndex].recvNcpyInfo(myDest);
  }

  // Executed on Index 0 (which calls put)
  void putSenderDone(CkDataMsg *m){
    CkAssert(thisIndex == 0);

    CkPrintf("[%d][%d][%d] From inside source callback: Put is complete - ", thisIndex, CkMyPe(), CkMyNode());
    CkPrintf("De-registering and freeing source data\n");

    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *src = (CkNcpyBuffer *)(m->data);
    src->deregisterMem();

    delete m;

    mainProxy.maindone();
  }

  // Executed on Index 1 (which receives data from put)
  void putReceiverDone(CkDataMsg *m){
    CkAssert(thisIndex == 1);

    CkPrintf("[%d][%d][%d] Inside Destination callback: Put is complete - ", thisIndex, CkMyPe(), CkMyNode());
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

  // Executed on Index 0
  void recvNcpyInfo(CkNcpyBuffer dest)
  {
    CkAssert(thisIndex == 0);
    // Create nocopy source for me to Put into
    CkNcpyBuffer mySrc(iArr, size*sizeof(int), cb, CK_BUFFER_REG);

    // Perform Put from my source into Index 1's destinations
    CkNcpyStatus status = mySrc.put(dest);
    if(status == CkNcpyStatus::complete) {
      CkPrintf("[%d][%d][%d] Inline                     : Put is complete - \n", thisIndex, CkMyPe(), CkMyNode());
    }
    else if(status == CkNcpyStatus::incomplete) {
      CkPrintf("[%d][%d][%d] Async RDMA call in progress: Put is still incomplete on the source\n", thisIndex, CkMyPe(), CkMyNode());
      // async call in progress, reset of the operations in source callback
    }
  }

};

#include "simple_put.def.h"
