#include "zcpy_immediate.decl.h"

CProxy_main mainProxy;
CProxy_NG1 ngProxy;
int numElements;

class main : public CBase_main
{
  CProxy_Array1 arr1;
  int count;
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m)
  {
    if(m->argc !=2 ) {
      CkPrintf("Usage: ./zcpy_immediate <numelements>\n");
      CkExit(1);
    }
    numElements = atoi(m->argv[1]);
    delete m;
    if(numElements % 2 != 0){
      CkPrintf("Argument <numelements> should be even");
      CkExit(1);
    }
    mainProxy = thisProxy;
    ngProxy = CProxy_NG1::ckNew();
    int size = 2000;
    arr1 = CProxy_Array1::ckNew(size, numElements);
    count = 0;
    arr1.start();
  };

  void maindone(){
    count++;
    if(count == numElements*2) {
      CkPrintf("[%d][%d][%d] All Rgets have completed\n", CkMyPe(), CkMyNode(), CmiMyRank());
      CkExit();
    }
  };
};

void assignCharValues(char *&arr, int size){
  arr = new char[size];
  for(int i=0; i<size; i++)
     arr[i] = (char)(rand() % 125 + 1);
}

class Array1 : public CBase_Array1
{
  char *cArr1, *cArr2;
  int size;
  CkCallback srcCb, destCb;
  int destIndex;

public:
  Array1(int size)
  {
    this->size = size;

    if(thisIndex < numElements/2) {
      assignCharValues(cArr1, size); // cArr1 is the source for Get
      cArr2 = new char[size]; // cArr2 is the destination for Put
    } else {
      cArr1 = new char[size]; // cArr1 is the destination for Get
      assignCharValues(cArr2, size); // cArr2 is the source for Put
    }

    srcCb = CkCallback(CkIndex_NG1::senderDone(NULL), ngProxy[CkMyNode()]);
    destCb = CkCallback(CkIndex_NG1::receiverDone(NULL), ngProxy[CkMyNode()]);

    destIndex = numElements - 1 - thisIndex;
  }

  void start()
  {
    if(thisIndex < numElements/2) {
      CkNcpyBuffer mySrc(cArr1, size*sizeof(char), srcCb, CK_BUFFER_REG);
      CkNcpyBuffer myDest(cArr2, size*sizeof(char), destCb, CK_BUFFER_UNREG);

      // Send my source & my destination to destIndex
      // Index 1 performs get from my source and performs put into my destination
      thisProxy[destIndex].recvNcpyInfo(mySrc, myDest);
    }
  }

  // Executed on Index 1
  void recvNcpyInfo(CkNcpyBuffer otherSrc, CkNcpyBuffer otherDest)
  {
    // Create nocopy destination for me to Get into
    CkNcpyBuffer myDest(cArr1, size*sizeof(char), destCb, CK_BUFFER_UNREG);
    // Create nocopy source for me to Put from
    CkNcpyBuffer mySrc(cArr2, size*sizeof(char), srcCb, CK_BUFFER_REG);

    // Perform Get from other source into my destinations
    myDest.get(otherSrc);

    // Perform Put from my source into other destinations
    mySrc.put(otherDest);
  }
};

class NG1 : public CBase_NG1 {
  public:
  NG1() {}

  void senderDone(CkDataMsg *m){
    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *src = (CkNcpyBuffer *)(m->data);
    src->deregisterMem();
    delete m;

    mainProxy.maindone();
  }

  // Executed on Index 1 (which receives data from get)
  void receiverDone(CkDataMsg *m){
    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *dest = (CkNcpyBuffer *)(m->data);
    dest->deregisterMem();
    delete m;

    mainProxy.maindone();
  }
};

#include "zcpy_immediate.def.h"
