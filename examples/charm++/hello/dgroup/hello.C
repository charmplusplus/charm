#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors\n",
	     CkNumPes());
    if(CkNumPes()<2) {
      CkAbort("This program should be run on at least 2 processors.\n");
    }
    mainProxy = thisProxy;

    CProxy_Other::ckNew(1);
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
};

class Other : public CBase_Other
{
  public:
    Other(void)
    {
      CProxy_Hello grp = CProxy_Hello::ckNew();
      grp[0].SayHi(17);
    }
};

/*group*/
class Hello : public CBase_Hello
{
public:
  Hello()
  {
    CkPrintf("Hello %d created\n",CkMyPe());
  }

  void SayHi(int hiNo)
  {
    int ind=CkMyPe();
    CkPrintf("Hi[%d] from element %d\n",hiNo,ind);
    if (ind+1<CkNumPes())
      //Pass the hello on:
      thisProxy[ind+1].SayHi(hiNo+1);
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

#include "hello.def.h"
