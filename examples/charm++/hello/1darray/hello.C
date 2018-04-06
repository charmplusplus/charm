#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Hello arrProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thisProxy;

    CkArrayOptions opts;
    opts.setNumInitial(nElements);
    CkCallback initCB(CkIndex_Main::initDone(), thisProxy);
    opts.setInitCallback(initCB);
    opts.setStaticInsertion(true);
    arrProxy = CProxy_Hello::ckNew(opts);
  };

  void initDone(void) {
    CkPrintf("Main::initDone reached\n");
    arrProxy[0].SayHi(17);
  }

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
    CkPrintf("Hello %d created\n",thisIndex);
  }

  Hello(CkMigrateMessage *m) {}

  void SayHi(int hiNo)
  {
    printf("Memusage: %lu\n", CmiMemoryUsage());
    CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
    if (thisIndex < nElements-1)
      //Pass the hello on:
      thisProxy[thisIndex+1].SayHi(hiNo+1);
    else
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

#include "hello.def.h"
