
#include <stdio.h>
#include "hello.decl.h"
#include "StreamingStrategy.h"
#include "MeshStreamingStrategy.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

#define TEST_HI 4001
#define MAX_COUNT 500

ComlibInstanceHandle ss_inst;
ComlibInstanceHandle mss_inst;
ComlibInstanceHandle samp_inst;

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
      
      StreamingStrategy *strat=new StreamingStrategy(1,400);
      MeshStreamingStrategy *mstrat=new MeshStreamingStrategy(1,400);
      
      //strat->enableShortArrayMessagePacking();
      //strat->disableIdleFlush();
      
      mstrat->enableShortArrayMessagePacking();

      ComlibInstanceHandle cinst = CkGetComlibInstance();
      cinst.setStrategy(strat); 
      
      //ComlibInstanceHandle cinst1 = CkGetComlibInstance();

      //cinst1.setStrategy(mstrat);

      startTime=CkWallTimer();
      CkPrintf("Starting ring...\n");

      arr.start();
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
    
    void start() {
        for(int count = 0; count < 2000; count++)
            SayHi(TEST_HI, 0);
    }
    
    void SayHi(int hiNo, int hcount) {
        static int recv_count = 0;
        
        CkAssert(hiNo >= TEST_HI);
        
        // CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
        CProxy_Hello array_proxy = thisProxy;
        ComlibDelegateProxy(&array_proxy);
        
        /*
          if (thisIndex < nElements-1)
          //Pass the hello on:
          array_proxy[thisIndex+1].SayHi(hiNo+1);
          else if(recv_count == nElements-1)
          //We've been around once-- we're done.
          mainProxy.done();    
          else
          recv_count ++;
        */
        
        int next = thisIndex+1;
        if(next >= nElements)
            next = 0;
        
        if(hcount < MAX_COUNT)
            //Pass the hello on:
            array_proxy[next].SayHi(hiNo+1, hcount+1);
      else{
          mainProxy.done();    
      }
    }
};

#include "hello.def.h"
