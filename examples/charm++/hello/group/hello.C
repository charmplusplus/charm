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
    mainProxy = thisProxy;

    CProxy_Hello grp = CProxy_Hello::ckNew();

    grp[0].SayHi(17);
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
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
