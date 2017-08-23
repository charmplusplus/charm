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
      CkAbort("Run this program on 1 or 2 processors only.\n");
    }
    if(m->argc !=2 ) {
      CkAbort("Usage: ./simple_direct <array size>\n");
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
  CkNcpyTarget myTar1, myTar2, myTar3;
  CkNcpySource mySrc1, mySrc2, mySrc3;
  CkNcpyTarget otherTar1, otherTar2, otherTar3;

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
    mySrc1 = CkNcpySource(iArr1, size*sizeof(int), sendCb);
    mySrc2 = CkNcpySource(dArr1, size*sizeof(double), sendCb);
    mySrc3 = CkNcpySource(cArr1, size*sizeof(char), sendCb);

    iArr2 = new int[size];
    dArr2 = new double[size];
    cArr2 = new char[size];

    myTar1 = CkNcpyTarget(iArr2, size*sizeof(int), recvCb);
    myTar2 = CkNcpyTarget(dArr2, size*sizeof(double), recvCb);
    myTar3 = CkNcpyTarget(cArr2, size*sizeof(char), recvCb);

    thisProxy[otherIndex].recvNcpyInfo(mySrc1, mySrc2, mySrc3, myTar1, myTar2, myTar3);
  }

  void senderCallback(CkDataMsg *m){
    sendCbCounter++;
    if(sendCbCounter == 3) {
      // Release Resources for my sources
      mySrc1.releaseResource();
      mySrc2.releaseResource();
      mySrc3.releaseResource();

      if(thisIndex == 1){
        delete [] iArr1;
        delete [] dArr1;
        delete [] cArr1;
        mainProxy.maindone();
      }
    }
    delete m;
  }

  void receiverCallback(CkDataMsg *m){
    recvCbCounter++;
    if(recvCbCounter == 3) {

      // Release Resources for my targets
      myTar1.releaseResource();
      myTar2.releaseResource();
      myTar3.releaseResource();

      if(thisIndex == 1){
        CkPrintf("[%d][%d][%d] Rget call completed\n", thisIndex, CkMyPe(), CkMyNode());

        // Create a nocopy sources for me to Rput from into targets received
        mySrc1 = CkNcpySource(iArr1, sizeof(int)*size, sendCb);
        mySrc2 = CkNcpySource(dArr1, sizeof(double)*size, sendCb);
        mySrc3 = CkNcpySource(cArr1, sizeof(char)*size, sendCb);

        // Index 1 Rputting to 0
        mySrc1.rput(otherTar1);
        mySrc2.rput(otherTar2);
        mySrc3.rput(otherTar3);

      } else {
        CkPrintf("[%d][%d][%d] Rput call completed\n", thisIndex, CkMyPe(), CkMyNode());

        compareArray(iArr1, iArr2, size);
        compareArray(dArr1, dArr2, size);
        compareArray(cArr1, cArr2, size);

        delete [] iArr1, iArr2, dArr1, dArr2, cArr1, cArr2;
        mainProxy.maindone();
      }
    }
    delete m;
  }

  // Executed on Index 1
  void recvNcpyInfo(CkNcpySource src1, CkNcpySource src2, CkNcpySource src3, CkNcpyTarget tar1, CkNcpyTarget tar2, CkNcpyTarget tar3)
  {
    CkAssert(thisIndex == 1);
    otherTar1 = tar1;
    otherTar2 = tar2;
    otherTar3 = tar3;

    // Create nocopy targets for me to Rget from sources received
    myTar1 = CkNcpyTarget(iArr1, size*sizeof(int), recvCb);
    myTar2 = CkNcpyTarget(dArr1, size*sizeof(double), recvCb);
    myTar3 = CkNcpyTarget(cArr1, size*sizeof(char), recvCb);

    // Index 1 Rgetting from 0
    myTar1.rget(src1);
    myTar2.rget(src2);
    myTar3.rget(src3);
  }
};

#include "simple_direct.def.h"
