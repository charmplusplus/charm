#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;
/*readonly*/ CProxy_Hello arr; 

int currentIndex; 

extern void kernelSetup(); 

void kernelReturn() {
  printf("Kernel returned\n"); 
  arr[currentIndex].SendHi(); 
  //  sendHi(); 
}

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

    arr = CProxy_Hello::ckNew(nElements);

    currentIndex = 0; 

    arr[0].SayHi();
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
    CkPrintf("Hello %d created\n",thisIndex);
  }

  Hello(CkMigrateMessage *m) {}

  void SayHi()
  {
    currentIndex = thisIndex; 
    CkPrintf("Hi from element %d\n", thisIndex);
    if (thisIndex < nElements-1)
      kernelSetup(); 
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }

  void SendHi() {
    //Pass the hello on:
    CkPrintf("Executing sendHi\n"); 
    thisProxy[thisIndex+1].SayHi();
  }

};

#include "hello.def.h"
