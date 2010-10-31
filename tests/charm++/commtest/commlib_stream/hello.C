
#include <stdio.h>
#include <comlib.h>
#include "hello.decl.h"
#include "StreamingStrategy.h"
#include "MeshStreamingStrategy.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

#define TEST_HI 4001
#define MAX_COUNT 200
#define NUM_RINGS 200 

ComlibInstanceHandle ss_inst;
ComlibInstanceHandle mss_inst;
ComlibInstanceHandle samp_inst;
ComlibInstanceHandle dummy_inst;

int bucketSize = 400;

/*mainchare*/
class Main : public Chare
{
    double startTime;
    int recv_count;
public:
    Main(CkArgMsg* m)
    {
	//Process command-line arguments
	nElements=5;
	if(m->argc >1 ) nElements=atoi(m->argv[1]);
	delete m;
	
        recv_count = 0;

	//Start the computation
	CkPrintf("Running Hello on %d processors for %d elements\n",
		 CkNumPes(),nElements);
	mainProxy = thishandle;
	
	CProxy_Hello arr = CProxy_Hello::ckNew(nElements);
	
	StreamingStrategy *strat=new StreamingStrategy(1,bucketSize);
	MeshStreamingStrategy *mstrat=new MeshStreamingStrategy(1,bucketSize);
	
	//strat->enableShortArrayMessagePacking();
	//strat->disableIdleFlush();
	
//	mstrat->enableShortArrayMessagePacking();
	
//	ComlibInstanceHandle cinst = CkGetComlibInstance();
	ss_inst = ComlibRegister(strat); 
	
	//ComlibInstanceHandle cinst1 = CkGetComlibInstance();
	
	//cinst1.setStrategy(mstrat);
	
	startTime=CkWallTimer();
	CkPrintf("Starting ring...\n");
	
	arr.start();
    };
    
    void done(void)
    {
        recv_count ++;
        
        if(recv_count == CkNumPes()) {
            CkPrintf("All done: %d elements in %f seconds\n", nElements,
                     CkWallTimer()-startTime);
            CkExit();
        }
    };
};

/*array [1D]*/
class Hello : public CBase_Hello 
{
    int c;
    
public:
    Hello() {
	c = 0;
    }
    
    Hello(CkMigrateMessage *m) {}
    
    void start() {
        for(int count = 0; count < NUM_RINGS; count++)
            SayHi(TEST_HI, 0);
    }
    
    void SayHi(int hiNo, int hcount) {
        
        CkAssert(hiNo >= TEST_HI);
        
        // CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
        CProxy_Hello array_proxy = thisProxy;
        //ComlibDelegateProxy(&array_proxy);
 		ComlibAssociateProxy(ss_inst, array_proxy);
        
        int next = thisIndex+1;
        if(next >= nElements)
            next = 0;
        
        if(hcount < MAX_COUNT)
	    //Pass the hello on:
	    array_proxy[next].SayHi(hiNo+1, hcount+1);
	else{
	    c++;
	    if (c == NUM_RINGS) mainProxy.done();    
	}
    }
};

#include "hello.def.h"
