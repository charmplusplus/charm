#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

#define MAXSTEPS 1000000

/*mainchare*/
class Main : public CBase_Main
{
    double startTime;
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

    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);

    arr[0].SayHi(17,17);
    startTime = CmiWallTimer();
  };

  void done(void)
  {
    CkPrintf("All done, time per message = %gus\n", (CmiWallTimer() - startTime)*1e6/MAXSTEPS/nElements);
    CkExit();
  };
};

/*array [1D]*/
class Hello : public CBase_Hello
{
    int step;
public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex);
    step = 0;
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo, int hi1)
  {
      //CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
      if (thisIndex < nElements-1)
          //Pass the hello on:
          thisProxy[thisIndex+1].SayHi(hiNo+1, 0);
      else {
          step ++;
          if(step == MAXSTEPS)
              //We've been around MAXSTEP times -- we're done.
              mainProxy.done();
          else
              thisProxy[0].SayHi(17,17);
      }
  }
};
    
#include "hello.def.h"
