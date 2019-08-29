#include <assert.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    int expectedNumPes=2;
    if(m->argc > 2) {
        nElements=atoi(m->argv[1]);
        expectedNumPes=atoi(m->argv[2]);
    }
    else if(m->argc > 1){
        nElements=atoi(m->argv[1]);
    }
    delete m;

    //Check correctness of partitions semantics
    assert(CkNumPes() == expectedNumPes);

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
             CkNumPes(),nElements);
    mainProxy = thisProxy;

    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);

    arr[0].SayHi(17);
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
};

/*array [1D]*/
class Hello : public CBase_Hello
{
public:
  Hello()
  {
    CkPrintf("[%d] Hello %d created\n", CkMyPe(), thisIndex);
  }

  Hello(CkMigrateMessage *m) {}

  void SayHi(int hiNo)
  {
    CkPrintf("[%d] Hi[%d] from element %d\n", CkMyPe(), hiNo, thisIndex);
    if (thisIndex < nElements-1)
      //Pass the hello on:
      thisProxy[thisIndex+1].SayHi(hiNo+1);
    else
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

#include "hello.def.h"
