#include "LB_Test.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int num_chare_blocks;
/*readonly*/ int total_iterations;

class Main : public CBase_Main
{
public:
  int done_count;
  CProxy_LB_Test arrayProxy;
  double timestamp;
  double workStartTimestamp;

  Main(CkArgMsg* m) {
    CkAssert(CkMyPe() == 0);
    if (m->argc < 3) {
      CkPrintf("Usage: %s <num chare blocks/pe> <total iter>\n", m->argv[0]);
      CkAbort("Abort");
    }
    num_chare_blocks = atoi(m->argv[1]);
    total_iterations = atoi(m->argv[2]);

    timestamp = CkWallTimer();

    // store the main proxy
    mainProxy = thisProxy;

    // print info
    CkPrintf("Running on %d processors with %d chares per pe\n", 
	     CkNumPes(), num_chare_blocks*4);

    // Create new array of worker chares
    arrayProxy = CProxy_LB_Test::ckNew(num_chare_blocks*4*CkNumPes());
    arrayProxy.ckSetReductionClient(new CkCallback(CkIndex_Main::myBarrier(NULL), mainProxy));

    done_count = 0;

    // Computation is now started by every array elements' constructors
    //    reporting back to the main chare.
    // This is only done to work-around some aspects of bigsim tracing
    //    with respect to array creation and messages sent to elements.
  }

  void report_in() {
    // use done_count, but remember to reset as this is intended for
    // the exit reduction.
    done_count++;
    if (num_chare_blocks*4*CkNumPes() == done_count) {
      workStartTimestamp = CkWallTimer();
      CkPrintf("All array elements ready at %f seconds. Computation Begins\n",
	       workStartTimestamp - timestamp);
      done_count = 0;
      arrayProxy.next_iter();
    }
  }

  // Reduction callback client
  void myBarrier(CkReductionMsg *msg) {
    CkAssert(CkMyPe() == 0);
    double maxTime = *((double *)msg->getData());
    delete msg;
    arrayProxy.next_iter();
  }

  // Each worker reports back to here when it completes all work
  void report_done() {
    done_count++;
    if (num_chare_blocks*4*CkNumPes() == done_count) {
      CkPrintf("Average iteration time = %f seconds\n",
	       (CkWallTimer() - workStartTimestamp)/total_iterations);
      CkPrintf("Done after %f seconds\n", CkWallTimer() - timestamp);
      CkExit();
    }
  }
};

class LB_Test: public CBase_LB_Test {
public:
  int iteration;
  int work_factor;

  // Initialization and start of iterations. 
  // Chares starting on even-numbered processors get a single work
  // unit. 
  // Chares starting on odd-numbered processors get two work units.
  // NOTE: These work factors do not change after migration! That's
  //       the point of this example!
  LB_Test() {
    if (CkMyPe() % 2 == 0) {
      work_factor = 1;
    } else {
      work_factor = 2;
    }
    iteration = 0;
    usesAtSync = CmiTrue;
    mainProxy.report_in();
  }

  // For migration
  LB_Test(CkMigrateMessage* m) {
  }

  // Destructor
  ~LB_Test() {
  }

  // Load Balancing happens halfway into the computation
  void next_iter() {
    if (iteration < total_iterations) {
      if ((iteration == total_iterations/2) && usesAtSync) {
	/*
	CkPrintf("{%d}[%d] AtSync() called at iteration %d\n", thisIndex,
		 CkMyPe(), iteration);
	*/
	AtSync();
      } else {
	compute();
      }
    } else {
      mainProxy.report_done();
    }
  }

  void compute() {
    double a[2000], b[2000], c[2000];
    double timestamp = CkWallTimer();
    for(int j=0;j<1000*work_factor;j++){
      for(int i=0;i<2000;i++){
	a[i] = 7.0;
	b[i] = 5.0;
      }
      for(int i=0;i<2000/2;i++){
	c[i] = a[2*i]*b[2*i+1]*a[i];
	c[2*i] = a[2*i];
      }
    }
    iteration++;
    timestamp = CkWallTimer() - timestamp;
    contribute(sizeof(double), (void *)&timestamp, 
	       CkReduction::max_double);
  }
  
  void ResumeFromSync(void) { // Called by Load-balancing framework
    compute();
  }
  
  void pup(PUP::er &p)
  {
    CBase_LB_Test::pup(p);
    p | iteration;
    p | work_factor;
  }
};

#include "LB_Test.def.h"
