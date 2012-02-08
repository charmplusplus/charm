
/*******************
    Streaming benchmark that has a many-to-many communication 
    pattern for array objects. This benchmark should demonstrate 
    the performance of the Mesh Streaming Strategy for small bucket 
    sizes.

    - Sameer Kumar (03/03/05)

****************************/

#include <stdio.h>
#include "manytomany.decl.h"
#include "StreamingStrategy.h"
#include "MeshStreamingStrategy.h"
#include "DummyStrategy.h"

//readonly comlib instances
ComlibInstanceHandle ss_inst;  //basic streaming strategy
ComlibInstanceHandle mss_inst;  //mesh streaming strategy
ComlibInstanceHandle samp_inst; //streaming with short message packing
ComlibInstanceHandle dummy_inst; //no streaming performance

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

#define TEST_HI 4001
#define MAX_COUNT 2000  //Number of iterations
int NUM_MSGS=80;    //Number of messages sent in each iteration
#define MAX_PER_ITR NUM_MSGS

/*mainchare*/
class Main : public CBase_Main
{
  double startTime;
  int recv_count;
  int iter_count;
  CProxy_Hello arr;
  
public:
  Main(CkArgMsg* m)
  {
    //Process command-linse arguments
    nElements=5;
    
    int bucketSize = 10;
    
    if(m->argc >1 ) nElements =atoi(m->argv[1]);
    if(m->argc > 2 ) NUM_MSGS = atoi(m->argv[2]);
    if(m->argc > 3 ) bucketSize = atoi(m->argv[3]);
    delete m;
    
    recv_count = 0;
    iter_count = 0;
    
    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements with bucket %d\n",
	     CkNumPes(),nElements, bucketSize);
    mainProxy = thishandle;
    
    arr = CProxy_Hello::ckNew(nElements);
    
    StreamingStrategy *strat=new StreamingStrategy
        (0.02,bucketSize);
    
    StreamingStrategy *sstrat=new StreamingStrategy
        (0.02,bucketSize);
    
    MeshStreamingStrategy *mstrat=new MeshStreamingStrategy
        (1, bucketSize);

    DummyStrategy *dummy_strat = new DummyStrategy();
    
    sstrat->enableShortArrayMessagePacking();  
    mstrat->enableShortArrayMessagePacking();
    
    ss_inst = CkGetComlibInstance();
    ss_inst.setStrategy(strat); 
    
    mss_inst = CkGetComlibInstance();
    mss_inst.setStrategy(mstrat); 
    
    samp_inst = CkGetComlibInstance();
    samp_inst.setStrategy(sstrat);         
    
    dummy_inst = CkGetComlibInstance();
    dummy_inst.setStrategy(dummy_strat);

    startTime = CkWallTimer();
    CkPrintf("Starting many-to-many benchmark...\n");
    
    arr.start();
  };
  
  void done(void)
  {
    recv_count ++;
    
    //CkPrintf("Receiving Done\n");
    
    if(recv_count == nElements) {
      iter_count ++;
      recv_count = 0;
      
      char stype[256];
      
      if(iter_count == 1)
	sprintf(stype, "");
      else if(iter_count == 2)
	sprintf(stype, "Mesh ");
      else if(iter_count == 3)
	sprintf(stype, "SAMP ");
      else if(iter_count == 4)
	sprintf(stype, "No ");      
      
      CkPrintf("%sStreaming Performance %g us/msg on %d pes and %d elements\n", 
	       stype, 
	       (CkWallTimer()-startTime)*1e6/(MAX_COUNT * MAX_PER_ITR), 
	       CkNumPes(), nElements);

      startTime = CmiWallTimer();

      if(iter_count == 4)
	CkExit();    
      else
	arr.start();                
    }
  };
};

/*array [1D]*/
class Hello : public CBase_Hello 
{
  int recv_count;
  ComlibInstanceHandle cur_inst;
  int step_count; //Which step, streaming, mesh-stream, samp
  int cur_itr; //Which itr? A total of MAX_ITRs will be executed
public:
  Hello()
  {
    recv_count = 0;
    cur_inst = ss_inst;
    step_count = 0;
    cur_itr = 0;
  }
  
  Hello(CkMigrateMessage *m) {}
  
  void start() {
    if(step_count == 0)
      cur_inst = ss_inst;
    else if(step_count == 1)
      cur_inst = mss_inst;
    else if(step_count == 2)
      cur_inst = samp_inst;
    else
      cur_inst = dummy_inst;

    step_count ++;
    localStart();
  }
  
  void localStart(){ 
    cur_inst.beginIteration();
    CProxy_Hello array_proxy = thisProxy;

    ComlibDelegateProxy(&array_proxy);

    int hiNo = TEST_HI;
    int next = thisIndex;
    for(int count = 0; count < MAX_PER_ITR; count++) {            
      next ++;
      if(next >= nElements)
	next -= nElements;
      
      //CkPrintf("%d Sending to %d\n", CkMyPe(), next);
      array_proxy[next].SayHi(hiNo, 0);            
    }
    
    cur_inst.endIteration();
  }
  
  void SayHi(int hiNo, int hcount) {
    
    //CkPrintf("%d Message Received %d, %d\n", CkMyPe(), 
    //recv_count, NUM_MSGS);
    
    CkAssert(hiNo >= TEST_HI);
    
    recv_count ++;
    
    if(recv_count == MAX_PER_ITR) {
      cur_itr ++;
      recv_count = 0;
      
      if(cur_itr == MAX_COUNT) {
	cur_itr = 0;
	mainProxy.done();    
      }
      else
	localStart();
    }
    //cur_inst.endIteration();
  }
};

#include "manytomany.def.h"
