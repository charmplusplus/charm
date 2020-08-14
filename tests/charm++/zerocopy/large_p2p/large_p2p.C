#include "large_p2p.decl.h"
#include <assert.h>

CProxy_main mainProxy;
class main : public CBase_main
{
  CProxy_Ping1 arr1;
  size_t count;
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m)
  {
    if(CkNumPes()>2) {
      CkPrintf("Run this program on 1 or 2 processors only.\n");
      CkExit(1);
    }
    if(m->argc !=2 ) {
      CkPrintf("Usage: ./large_p2p <array size>\n");
      CkExit(1);
    }
    size_t size = atoi(m->argv[1]);
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


void verifyCharValues(char *&arr, size_t size, const char constant){
  for(size_t i=0; i<size; i++)
    assert(arr[i] == constant);
}

void assignCharValues(char *&arr, size_t size, const char constant){
  arr = new char[size];
  for(size_t i=0; i<size; i++)
     arr[i] = constant;
}

class Ping1 : public CBase_Ping1
{
  char *cArr1;
  size_t size;
  size_t otherIndex, counter;
  CkCallback cb;
  const char constant = 'a';

public:
  Ping1(size_t size)
  {
    this->size = size;

    counter = 0;

    if(thisIndex == 0) {
      assignCharValues(cArr1, size, constant);
      cb = CkCallback(CkIndex_Ping1::getDone(NULL), thisProxy[thisIndex]);
    } else {
      cArr1 = new char[size];
      cb = CkCallback(CkIndex_Ping1::getDone(NULL), thisProxy[otherIndex]);
    }

    otherIndex = (thisIndex + 1) % 2;
  }
  Ping1(CkMigrateMessage *m) {}

  // Executed on Index 0
  void start()
  {
    CkAssert(thisIndex == 0);
    CkNcpyBuffer mySrc(cArr1, size*sizeof(char), cb);

    // Send my source to Index 1; Index 1 performs Gets from this source
    thisProxy[otherIndex].initiateGet(mySrc);
  }

  // Executed on Index 0
  void getDone(CkDataMsg *m){
    delete m;
    if(++counter == 2) {
      counter = 0;

      CkPrintf("[%d][%d][%d] Large Get Done\n", thisIndex, CkMyPe(), CkMyNode());
      // reset buffer
      memset(cArr1, '\0', size * sizeof(char));

      cb = CkCallback(CkIndex_Ping1::putDone(NULL), thisProxy[thisIndex]);

      CkNcpyBuffer myDest(cArr1, size*sizeof(char), cb);
      thisProxy[otherIndex].initiatePut(myDest);
    }
  }

  // Executed on Index 0 and Index 1
  void putDone(CkDataMsg *m) {
    if(thisIndex == 0) {
      verifyCharValues(cArr1, size, constant);
      CkPrintf("[%d][%d][%d] Large Put Done\n", thisIndex, CkMyPe(), CkMyNode());
    }
    delete [] cArr1;
    mainProxy.maindone();
  }

  // Executed on Index 1
  void initiateGet(CkNcpyBuffer otherSrc)
  {
    CkAssert(thisIndex == 1);
    // Create nocopy destination for me to Get into
    CkNcpyBuffer myDest(cArr1, size*sizeof(char), cb);
    myDest.get(otherSrc);
  }

  // Executed on Index 1
  void initiatePut(CkNcpyBuffer otherDest) {
    CkAssert(thisIndex == 1);

    cb = CkCallback(CkIndex_Ping1::putDone(NULL), thisProxy[thisIndex]);

    // Create nocopy source for me to Put from
    CkNcpyBuffer mySrc(cArr1, size*sizeof(char), cb);
    mySrc.put(otherDest);
  }
};

#include "large_p2p.def.h"
