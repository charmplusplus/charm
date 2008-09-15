#include "LB_Test.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int num_chare_blocks;
/*readonly*/ int workWeight;
/*readonly*/ int total_iterations;

class Main : public CBase_Main
{
public:
  int report_count;
  int iter_count;
  int done_count;
  CProxy_LB_Test arrayProxy;
  double timestamp;
  double workStartTimestamp;
  double totalChareWorkTime;

  Main(CkArgMsg* m) {
    CkAssert(CkMyPe() == 0);
    if ((m->argc < 3) || (m->argc > 4)) {
      CkPrintf("Usage: %s <num chare blocks/pe> <total iter> [work weight]\n", 
	       m->argv[0]);
      CkAbort("Abort");
    }
    num_chare_blocks = atoi(m->argv[1]);
    total_iterations = atoi(m->argv[2]);
    workWeight = 1;
    if (m->argc == 4) {
      workWeight = atoi(m->argv[3]);
    }

    timestamp = CkWallTimer();
    totalChareWorkTime = 0.0;
    report_count = 0;
    iter_count = 0;
    done_count = 0;

    // store the main proxy
    mainProxy = thisProxy;

    // print info
    CkPrintf("Running on %d processors with %d chares per pe\n", 
	     CkNumPes(), num_chare_blocks*4);

    // Create new array of worker chares. The element constructors will
    // contact this object to start the computation.
    arrayProxy = CProxy_LB_Test::ckNew(num_chare_blocks*4*CkNumPes());

  }

  void report_in() {
    report_count++;
    if (num_chare_blocks*4*CkNumPes() == report_count) {
      workStartTimestamp = CkWallTimer();
      CkPrintf("All array elements ready at %f seconds. Computation Begins\n",
	       workStartTimestamp - timestamp);
      report_count = 0;
      for (int i=0; i<num_chare_blocks*4*CkNumPes(); i++) {
	arrayProxy[i].next_iter();
      }
    }
  }

  // Reduction callback client
  void iterBarrier(double chareWorkTime) {
    iter_count++;
    totalChareWorkTime += chareWorkTime;
    if (num_chare_blocks*4*CkNumPes() == iter_count) {
      iter_count = 0;
      for (int i=0; i<num_chare_blocks*4*CkNumPes(); i++) {
	arrayProxy[i].next_iter();
      }
    }
  }

  // Each worker reports back to here when it completes all work
  void report_done() {
    done_count++;
    if (num_chare_blocks*4*CkNumPes() == done_count) {
      CkPrintf("Average total chare work per iteration = %f seconds\n",
	       totalChareWorkTime/total_iterations);
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
	AtSync();
      } else {
	compute();
      }
    } else {
      mainProxy.report_done();
    }
  }

  void compute() {
    double timeStamp = CkWallTimer();
    double a[2000], b[2000], c[2000];
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
    double timeTaken = CkWallTimer() - timeStamp;
    // Sanity output
    if (((iteration == 0) || (iteration == total_iterations-1)) &&
	((thisIndex == 0) || (thisIndex == 1))) {
      CkPrintf("[%d] Array Element %d took %f seconds at iteration %d\n", 
	       CkMyPe(), thisIndex, timeTaken, iteration);
    }
    iteration++;
    mainProxy.iterBarrier(timeTaken);
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
