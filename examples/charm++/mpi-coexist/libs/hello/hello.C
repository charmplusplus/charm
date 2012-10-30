#include <stdio.h>
#include "hello.decl.h"
//header from Charm to enable Interoperation
#include "mpi-interoperate.h"

/*mainchare*/
class MainHello : public CBase_MainHello
{
public:
  MainHello(int elems)
  {
    //Process command-line arguments
    int nElements=elems;
    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    CProxy_MainHello mainHelloProxy = thisProxy;

    CProxy_Hello arr = CProxy_Hello::ckNew(mainHelloProxy, nElements, nElements);

    arr[0].SayHi(0);
  };

  MainHello(CkMigrateMessage *m) {}

  void done(void)
  {
    CkExit();
  };
};

/*array [1D]*/
class Hello : public CBase_Hello 
{
private:
  CProxy_MainHello mainProxy;
  int nElements;
public:
  Hello(CProxy_MainHello mp, int nElems)
  {
    mainProxy = mp;
    nElements = nElems;
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    CkPrintf("Hello[%d] from element %d\n",hiNo,thisIndex);
    if (thisIndex < nElements-1)
      //Pass the hello on:
      thisProxy[thisIndex+1].SayHi(hiNo+1);
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

//C++ function invoked from MPI, marks the begining of Charm++
void HelloStart(int elems)
{
  if(CkMyPe() == 0) {
    CkPrintf("HelloStart - Starting lib by calling constructor of MainHello\n");
    CProxy_MainHello mainhello = CProxy_MainHello::ckNew(elems);
  }
  StartCharmScheduler();
}

#include "hello.def.h"
