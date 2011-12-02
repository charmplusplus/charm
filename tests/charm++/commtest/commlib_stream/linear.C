
/************************
     Benchmark to demostrate the performance of streaming communication by
     having several simulteneous rings of messages among the array elements.

     - Sameer Kumar (01/20/05)

*******************************/


#include <stdio.h>
#include "hello.decl.h"
#include "StreamingStrategy.h"
#include "MeshStreamingStrategy.h"
#include "DummyStrategy.h"


/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

int NUM_MSGS;

#define TEST_HI 4001
#define MAX_COUNT 2000
int bucketSize = 100;
#define MAX_PER_ITR 2*bucketSize 

ComlibInstanceHandle ss_inst;  //basic streaming strategy
ComlibInstanceHandle mss_inst;  //mesh streaming strategy
ComlibInstanceHandle samp_inst; //streaming with short message packing
ComlibInstanceHandle dummy_inst; //streaming with short message packing

/*mainchare*/
class Main : public CBase_Main
{
    double startTime;
    int recv_count;
    int step_count;
    CProxy_Hello arr;

public:
    Main(CkArgMsg* m)
    {
        //Process command-line arguments
        nElements=5;
        if(m->argc >1 ) nElements=atoi(m->argv[1]);
        if(m->argc > 2 ) bucketSize= atoi(m->argv[2]);
        delete m;
        
        recv_count = 0;
        step_count = 0;
        
        //Start the computation
        CkPrintf("Running Hello on %d processors for %d elements with bucket %d\n",
                 CkNumPes(),nElements, bucketSize);
        mainProxy = thishandle;
        
        arr = CProxy_Hello::ckNew(nElements);
        
        StreamingStrategy *strat=new StreamingStrategy(0.1,bucketSize);
        StreamingStrategy *sstrat=new StreamingStrategy(0.1,bucketSize);

        MeshStreamingStrategy *mstrat=new 
	  MeshStreamingStrategy(1, bucketSize);
        
        sstrat->enableShortArrayMessagePacking();
        
        //mstrat->enableShortArrayMessagePacking();
        
	DummyStrategy *dstrat = new DummyStrategy();

        ss_inst = CkGetComlibInstance();
        mss_inst = CkGetComlibInstance();
	samp_inst = CkGetComlibInstance();
	dummy_inst = CkGetComlibInstance();

        ss_inst.setStrategy(strat); 
        mss_inst.setStrategy(mstrat); 
	samp_inst.setStrategy(sstrat);     
	dummy_inst.setStrategy(dstrat);

        startTime=CkWallTimer();
        CkPrintf("Starting ring...\n");
        
        arr.start();
        step_count ++;
    };

    void done(void) {
        recv_count ++;
        
        if(recv_count == CkNumPes()) {
	    char stype[256];

            if(step_count == 1)
	      sprintf(stype, "");
            else if(step_count == 2)
	      sprintf(stype, "Mesh ");
            else if(step_count == 3)
	      sprintf(stype, "SAMP ");
	    else if(step_count == 4)
	      sprintf(stype, "NO ");

            CkPrintf("%sStreaming Performance %g us/msg on %d pes and %d elements\n", 
                     stype, 
                     (CkWallTimer()-startTime)*1e6/(MAX_COUNT * MAX_PER_ITR), 
                     CkNumPes(), nElements);

            if(step_count < 4) {
                arr.start();
                step_count ++;
                recv_count = 0;
                startTime = CkWallTimer();
            }
            else {
                CkExit();
            }
        }
    };
};

/*array [1D]*/
class Hello : public CBase_Hello 
{
    int recv_count;
    int step_count;
public:
    Hello() {
        recv_count = 0;
        step_count = 0;
    }
    
    Hello(CkMigrateMessage *m) {}
    
    void start() {
        if(step_count == 0)
	    ss_inst.beginIteration();
        else if(step_count == 1)            
            mss_inst.beginIteration();
        else if(step_count == 2)
            samp_inst.beginIteration();            
        else
	    dummy_inst.beginIteration();

        step_count++;

        recv_count = 0;        
        for(int count = 0; count < MAX_PER_ITR; count++)
            SayHi(TEST_HI, 0);
    }
    
    void SayHi(int hiNo, int hcount) {
        
        CkAssert(hiNo >= TEST_HI);
        
        // CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
        CProxy_Hello array_proxy = thisProxy;
        ComlibDelegateProxy(&array_proxy);
        
        int next = thisIndex+1;
        if(next >= nElements)
            next = 0;
        
        if(hcount < MAX_COUNT)
            //Pass the hello on:
            array_proxy[next].SayHi(hiNo+1, hcount+1);
	else if(recv_count == MAX_PER_ITR-1) 
            mainProxy.done();    
	else
            recv_count ++;
    }
};

#include "hello.def.h"
