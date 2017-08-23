#include "simple_rget.decl.h"
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
      CkAbort("Usage: ./simple_rget <array size>\n");
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
  CkNcpyTarget myTar1, myTar2, myTar3;
  CkNcpySource mySrc1, mySrc2, mySrc3;

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
    mySrc1 = CkNcpySource(iArr1, size*sizeof(int), cb);
    mySrc2 = CkNcpySource(dArr1, size*sizeof(double), cb);
    mySrc3 = CkNcpySource(cArr1, size*sizeof(char), cb);

    // Send my sources to Index 1; Index 1 performs Rgets from these sources
    thisProxy[otherIndex].recvNcpyInfo(mySrc1, mySrc2, mySrc3);
  }

  // Executed on Index 0
  void getSenderDone(CkDataMsg *m){
    CkAssert(thisIndex == 0);
    cbCounter++;
    if(cbCounter == 3) {
      // Release Resources for my sources
      mySrc1.releaseResource();
      mySrc2.releaseResource();
      mySrc3.releaseResource();
      CkPrintf("[%d][%d][%d] Rget Source Done\n", thisIndex, CkMyPe(), CkMyNode());
      sendValidationData();
    }
    delete m;
  }

  // Executed on Index 1 (which receives data from rget)
  void getReceiverDone(CkDataMsg *m){
    CkAssert(thisIndex == 1);
    cbCounter++;
    if(cbCounter == 3) {
      // Release Resources for my targets
      myTar1.releaseResource();
      myTar2.releaseResource();
      myTar3.releaseResource();
      CkPrintf("[%d][%d][%d] Rget Target Done\n", thisIndex, CkMyPe(), CkMyNode());
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
      delete [] iArr1, dArr1, cArr1;
      mainProxy.maindone();
    }
  }

  // Executed on Index 1
  void recvNcpyInfo(CkNcpySource src1, CkNcpySource src2, CkNcpySource src3)
  {
    CkAssert(thisIndex == 1);
    // Create nocopy target for me to Rget into
    myTar1 = CkNcpyTarget(iArr1, size*sizeof(int), cb);
    myTar2 = CkNcpyTarget(dArr1, size*sizeof(double), cb);
    myTar3 = CkNcpyTarget(cArr1, size*sizeof(char), cb);

    // Perform Rget from Index 0's sources into my targets
    myTar1.rget(src1);
    myTar2.rget(src2);
    myTar3.rget(src3);
  }

  // Executed on Index 1
  void validateGetData(int *iArr2, double *dArr2, char *cArr2, int size)
  {
    CkAssert(thisIndex == 1);
    compareArray(iArr1, iArr2, size);
    compareArray(dArr1, dArr2, size);
    compareArray(cArr1, cArr2, size);
    CkPrintf("[%d][%d][%d] Rget Validated! \n", thisIndex, CkMyPe(), CkMyNode());
    delete [] iArr1, dArr1, cArr1;
    mainProxy.maindone();
  }

};

#include "simple_rget.def.h"
