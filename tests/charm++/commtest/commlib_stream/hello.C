
#include <stdio.h>
#include "hello.decl.h"
#include "StreamingStrategy.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

#define TEST_HI 4001

/*mainchare*/
class Main : public Chare
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
    mainProxy = thishandle;

    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);

    StreamingStrategy *strat=new StreamingStrategy(1,10);

    strat->enableShortArrayMessagePacking();
    //strat->disableIdleFlush();
    
    ComlibInstanceHandle cinst = CkGetComlibInstance();
    cinst.setStrategy(strat); 
    
    startTime=CkWallTimer();
    CkPrintf("Starting ring...\n");
    arr.SayHi(TEST_HI);
  };

  void done(void)
  {
    CkPrintf("All done: %d elements in %f seconds\n", nElements,
    	CkWallTimer()-startTime);
    CkExit();
  };
};

/*array [1D]*/
class Hello : public CBase_Hello 
{
public:
  Hello()
  {

  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
      static int recv_count = 0;

      CkAssert(hiNo >= TEST_HI);

      // CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
      CProxy_Hello array_proxy = thisProxy;
      ComlibDelegateProxy(&array_proxy);
      
      if (thisIndex < nElements-1)
          //Pass the hello on:
          array_proxy[thisIndex+1].SayHi(hiNo+1);
      else if(recv_count == nElements-1)
          //We've been around once-- we're done.
          mainProxy.done();    
      else
          recv_count ++;
  }
};

#include "hello.def.h"
