#include <stdio.h>
#include "hi.decl.h"
//header from Charm to enable Interoperation
#include "mpi-interoperate.h"

CProxy_MainHi mainHi;
/*mainchare*/
class MainHi : public CBase_MainHi
{
public:
  MainHi(CkArgMsg *m) {
    mainHi = thisProxy;
    delete m;
    thisProxy.StartHi(10);
  }

  void StartHi(int elems)
  {
    //Process command-line arguments
    int nElements=elems;
    //Start the computation

    CkPrintf("Running Hi on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    CProxy_Hi arr = CProxy_Hi::ckNew(mainHi, nElements, nElements);

    arr[0].SayHi(1);
  };

  MainHi(CkMigrateMessage *m) {}

  void done(void)
  {
    CkExit();
  };
};

/*array [1D]*/
class Hi : public CBase_Hi 
{
private:
  CProxy_MainHi mainProxy;
  int nElements;
public:
  Hi(CProxy_MainHi mp, int nElems)
  {
    mainProxy = mp;
    nElements = nElems;
  }

  Hi(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
    if (thisIndex < nElements-1)
      //Pass the hello on:
      thisProxy[thisIndex+1].SayHi(hiNo+1);
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

//C++ function invoked from MPI, marks the begining of Charm++
void HiStart(int elems)
{
  if(CkMyPe() == 0) {
    mainHi.StartHi(elems);
  }
  StartCharmScheduler();
}

#include "hi.def.h"
