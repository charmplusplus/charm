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
    delete m;
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
  CkNcpyBuffer myDest1, myDest2, myDest3;
  CkNcpyBuffer mySrc1, mySrc2, mySrc3;
  CkNcpyBuffer otherDest1, otherDest2, otherDest3;

public:
  Ping1(int size)
  {
    this->size = size;

    // original arrays that contains data
    assignValues(iArr1, size);
    assignValues(dArr1, size);
    assignCharValues(cArr1, size);

    sendCb = CkCallback(CkIndex_Ping1::senderCallback(), thisProxy[thisIndex]);
    recvCb = CkCallback(CkIndex_Ping1::receiverCallback(), thisProxy[thisIndex]);

    otherIndex = (thisIndex + 1) % 2;
    sendCbCounter = 0;
    recvCbCounter = 0;
  }
  Ping1(CkMigrateMessage *m) {}

  // Executed on Index 0
  void start()
  {
    CkAssert(thisIndex == 0);
    mySrc1 = CkNcpyBuffer(iArr1, size*sizeof(int), sendCb);
    mySrc2 = CkNcpyBuffer(dArr1, size*sizeof(double), sendCb);
    mySrc3 = CkNcpyBuffer(cArr1, size*sizeof(char), sendCb);

    iArr2 = new int[size];
    dArr2 = new double[size];
    cArr2 = new char[size];

    myDest1 = CkNcpyBuffer(iArr2, size*sizeof(int), recvCb);
    myDest2 = CkNcpyBuffer(dArr2, size*sizeof(double), recvCb);
    myDest3 = CkNcpyBuffer(cArr2, size*sizeof(char), recvCb);

    thisProxy[otherIndex].recvNcpyInfo(mySrc1, mySrc2, mySrc3, myDest1, myDest2, myDest3);
  }

  void senderCallback(){
    sendCbCounter++;
    if(sendCbCounter == 3) {
      // Release Resources for my sources
      mySrc1.deregisterMem();
      mySrc2.deregisterMem();
      mySrc3.deregisterMem();

      if(thisIndex == 1){
        delete [] iArr1;
        delete [] dArr1;
        delete [] cArr1;
        mainProxy.maindone();
      }
    }
  }

  void receiverCallback(){
    recvCbCounter++;
    if(recvCbCounter == 3) {

      // Release Resources for my destinations
      myDest1.deregisterMem();
      myDest2.deregisterMem();
      myDest3.deregisterMem();

      if(thisIndex == 1){
        CkPrintf("[%d][%d][%d] Get call completed\n", thisIndex, CkMyPe(), CkMyNode());

        // Create a nocopy sources for me to Put from into destinations received
        mySrc1 = CkNcpyBuffer(iArr1, sizeof(int)*size, sendCb);
        mySrc2 = CkNcpyBuffer(dArr1, sizeof(double)*size, sendCb);
        mySrc3 = CkNcpyBuffer(cArr1, sizeof(char)*size, sendCb);

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
    myDest1 = CkNcpyBuffer(iArr1, size*sizeof(int), recvCb);
    myDest2 = CkNcpyBuffer(dArr1, size*sizeof(double), recvCb);
    myDest3 = CkNcpyBuffer(cArr1, size*sizeof(char), recvCb);

    // Index 1 Getting from 0
    myDest1.get(src1);
    myDest2.get(src2);
    myDest3.get(src3);
  }
};

#include "simple_direct.def.h"
