#include "simple_rput.decl.h"
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
      CkAbort("Usage: ./simple_rput <array size>\n");
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
  CkNcpyDestination myDest1, myDest2, myDest3;
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
      // Set PUT Sender callback
      cb = CkCallback(CkIndex_Ping1::putSenderDone(NULL), thisProxy[thisIndex]);
    } else {
      iArr1 = new int[size];
      cArr1 = new char[size];
      dArr1 = new double[size];
      // Set PUT Receiver callback
      cb = CkCallback(CkIndex_Ping1::putReceiverDone(NULL), thisProxy[thisIndex]);
    }

    otherIndex = (thisIndex + 1) % 2;
    cbCounter = 0;
    valCounter = 0;
  }
  Ping1(CkMigrateMessage *m) {}

  // Executed on Index 1
  void start()
  {
    CkAssert(thisIndex == 1);
    myDest1 = CkNcpyDestination(iArr1, size*sizeof(int), cb);
    myDest2 = CkNcpyDestination(dArr1, size*sizeof(double), cb);
    myDest3 = CkNcpyDestination(cArr1, size*sizeof(char), cb);

    // Send my destinations to Index 0; Index 0 performs Rputs into these destinations
    thisProxy[otherIndex].recvNcpyInfo(myDest1, myDest2, myDest3);
  }

  // Executed on Index 0 (which calls rput)
  void putSenderDone(CkDataMsg *m){
    CkAssert(thisIndex == 0);
    cbCounter++;
    if(cbCounter == 3) {
      // Release Resources for my sources
      mySrc1.releaseResource();
      mySrc2.releaseResource();
      mySrc3.releaseResource();
      CkPrintf("[%d][%d][%d] Rput Source Done\n", thisIndex, CkMyPe(), CkMyNode());
      sendValidationData();
    }
    delete m;
  }

  // Executed on Index 1 (which receives data from rput)
  void putReceiverDone(CkDataMsg *m){
    CkAssert(thisIndex == 1);
    cbCounter++;
    if(cbCounter == 3) {
      // Release Resources for my destinations
      myDest1.releaseResource();
      myDest2.releaseResource();
      myDest3.releaseResource();
      CkPrintf("[%d][%d][%d] Rput Destination Done\n", thisIndex, CkMyPe(), CkMyNode());
      thisProxy[otherIndex].sendValidationData();
    }
    delete m;
  }

  // Executed on Index 0
  void sendValidationData() {
    CkAssert(thisIndex == 0);
    valCounter++;
    if(valCounter == 2) {
      thisProxy[otherIndex].validatePutData(iArr1, dArr1, cArr1, size);
      delete [] iArr1;
      delete [] dArr1;
      delete [] cArr1;
      mainProxy.maindone();
    }
  }

  // Executed on Index 0
  void recvNcpyInfo(CkNcpyDestination dest1, CkNcpyDestination dest2, CkNcpyDestination dest3)
  {
    CkAssert(thisIndex == 0);
    // Create nocopy sources for me to Rput into
    mySrc1 = CkNcpySource(iArr1, size*sizeof(int), cb);
    mySrc2 = CkNcpySource(dArr1, size*sizeof(double), cb);
    mySrc3 = CkNcpySource(cArr1, size*sizeof(char), cb);

    // Perform Rputs from my sources into Index 1's destinations
    mySrc1.rput(dest1);
    mySrc2.rput(dest2);
    mySrc3.rput(dest3);
  }

  // Executed on Index 1
  void validatePutData(int *iArr2, double *dArr2, char *cArr2, int size)
  {
    CkAssert(thisIndex == 1);
    compareArray(iArr1, iArr2, size);
    compareArray(dArr1, dArr2, size);
    compareArray(cArr1, cArr2, size);
    CkPrintf("[%d][%d][%d] Rput Validated! \n", thisIndex, CkMyPe(), CkMyNode());
    delete [] iArr1;
    delete [] dArr1;
    delete [] cArr1;
    mainProxy.maindone();
  }

};

#include "simple_rput.def.h"
