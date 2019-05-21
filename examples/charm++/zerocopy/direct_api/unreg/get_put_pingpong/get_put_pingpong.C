#include "simple_direct.decl.h"
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
      CkPrintf("Usage: ./simple_direct <array size>\n");
      CkExit(1);
    }
    int size = atoi(m->argv[1]);
    mainProxy = thisProxy;
    arr1 = CProxy_Ping1::ckNew(size, 2);
    count = 0;
    arr1[0].start();
  };

  void maindone(){
    count++;
    if(count == 2) {
      CkPrintf("[%d][%d] Result validated! \n", CkMyPe(), CkMyNode());
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
  int *iArr1, *iArr2;
  char *cArr1, *cArr2;
  double *dArr1, *dArr2;
  int size;
  int otherIndex, recvCbCounter, sendCbCounter;
  CkCallback sendCb, recvCb;
  CkNcpyBuffer otherDest1, otherDest2, otherDest3;

public:
  Ping1(int size)
  {
    this->size = size;

    // original arrays that contains data
    assignValues(iArr1, size);
    assignValues(dArr1, size);
    assignCharValues(cArr1, size);

    sendCb = CkCallback(CkIndex_Ping1::senderCallback(NULL), thisProxy[thisIndex]);
    recvCb = CkCallback(CkIndex_Ping1::receiverCallback(NULL), thisProxy[thisIndex]);

    otherIndex = (thisIndex + 1) % 2;
    sendCbCounter = 0;
    recvCbCounter = 0;
  }
  Ping1(CkMigrateMessage *m) {}

  // Executed on Index 0
  void start()
  {
    CkAssert(thisIndex == 0);
    CkNcpyBuffer mySrc1(iArr1, size*sizeof(int), sendCb, CK_BUFFER_UNREG);
    CkNcpyBuffer mySrc2(dArr1, size*sizeof(double), sendCb, CK_BUFFER_UNREG);
    CkNcpyBuffer mySrc3(cArr1, size*sizeof(char), sendCb, CK_BUFFER_UNREG);

    iArr2 = new int[size];
    dArr2 = new double[size];
    cArr2 = new char[size];

    CkNcpyBuffer myDest1(iArr2, size*sizeof(int), recvCb, CK_BUFFER_UNREG);
    CkNcpyBuffer myDest2(dArr2, size*sizeof(double), recvCb, CK_BUFFER_UNREG);
    CkNcpyBuffer myDest3(cArr2, size*sizeof(char), recvCb, CK_BUFFER_UNREG);

    thisProxy[otherIndex].recvNcpyInfo(mySrc1, mySrc2, mySrc3, myDest1, myDest2, myDest3);
  }

  void senderCallback(CkDataMsg *m){
    sendCbCounter++;

    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *src = (CkNcpyBuffer *)(m->data);
    src->deregisterMem();
    delete m;

    if(sendCbCounter == 3) {

      if(thisIndex == 1){
        delete [] iArr1;
        delete [] dArr1;
        delete [] cArr1;
        mainProxy.maindone();
      }
    }
  }

  void receiverCallback(CkDataMsg *m){
    recvCbCounter++;

    // Cast m->data as (CkNcpyBuffer *)
    CkNcpyBuffer *dest = (CkNcpyBuffer *)(m->data);
    dest->deregisterMem();
    delete m;

    if(recvCbCounter == 3) {

      if(thisIndex == 1){
        CkPrintf("[%d][%d][%d] Get call completed\n", thisIndex, CkMyPe(), CkMyNode());

        // Create a nocopy sources for me to Put from into destinations received
        CkNcpyBuffer mySrc1(iArr1, sizeof(int)*size, sendCb, CK_BUFFER_UNREG);
        CkNcpyBuffer mySrc2(dArr1, sizeof(double)*size, sendCb, CK_BUFFER_UNREG);
        CkNcpyBuffer mySrc3(cArr1, sizeof(char)*size, sendCb, CK_BUFFER_UNREG);

        // Index 1 Putting to 0
        mySrc1.put(otherDest1);
        mySrc2.put(otherDest2);
        mySrc3.put(otherDest3);

      } else {
        CkPrintf("[%d][%d][%d] Put call completed\n", thisIndex, CkMyPe(), CkMyNode());

        compareArray(iArr1, iArr2, size);
        compareArray(dArr1, dArr2, size);
        compareArray(cArr1, cArr2, size);

        // All arrays can be deleted at this point. But they are not as the program is exiting.
        mainProxy.maindone();
      }
    }
  }

  // Executed on Index 1
  void recvNcpyInfo(CkNcpyBuffer src1, CkNcpyBuffer src2, CkNcpyBuffer src3, CkNcpyBuffer dest1, CkNcpyBuffer dest2, CkNcpyBuffer dest3)
  {
    CkAssert(thisIndex == 1);
    otherDest1 = dest1;
    otherDest2 = dest2;
    otherDest3 = dest3;

    // Create nocopy destinations for me to Get from sources received
    CkNcpyBuffer myDest1(iArr1, size*sizeof(int), recvCb, CK_BUFFER_UNREG);
    CkNcpyBuffer myDest2(dArr1, size*sizeof(double), recvCb, CK_BUFFER_UNREG);
    CkNcpyBuffer myDest3(cArr1, size*sizeof(char), recvCb, CK_BUFFER_UNREG);

    // Index 1 Getting from 0
    myDest1.get(src1);
    myDest2.get(src2);
    myDest3.get(src3);
  }
};

#include "simple_direct.def.h"
