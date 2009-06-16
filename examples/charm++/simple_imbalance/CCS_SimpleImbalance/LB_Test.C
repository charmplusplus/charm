#include "ckcallback-ccs.h"
#include "LB_Test.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int num_chare_blocks;
/*readonly*/ int workWeight;
/*readonly*/ int total_iterations;

CkpvDeclare(bool, traceFlagSet);

// The user has to remember to run this application with +traceoff for now.
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

    // Set up application-level CCS controls
    CkPrintf("LB_Test listening for CCS handshake\n");
    CcsRegisterHandler("LB_Test_Handshake",
		       CkCallback(CkIndex_Main::ccsHandshake(NULL), 
				  thishandle));

    // Allows the traceBegin call to be made exactly once per processor.
    CkpvInitialize(boolean, traceFlagSet);
    CkpvAccess(traceFlagSet) = false;

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
    // the extra +1 is to ensure that computation will not proceed until
    // CCS is ready
    if (num_chare_blocks*4*CkNumPes() + 1 == report_count) {
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
      CkPrintf("Total work performed = %f seconds\n", totalChareWorkTime);
      CkPrintf("Average total chare work per iteration = %f seconds\n",
	       totalChareWorkTime/total_iterations);
      CkPrintf("Average iteration time = %f seconds\n",
	       (CkWallTimer() - workStartTimestamp)/total_iterations);
      CkPrintf("Done after %f seconds\n", CkWallTimer() - timestamp);
      CkExit();
    }
  }

  void ccsHandshake(CkCcsRequestMsg *m) {
    char *sendBuffer;
    CkPrintf("[%d] Handshake from Client detected.\n", CkMyPe());

    traceEnableCCS(); // happens on 1 processor

    sendBuffer = (char *)malloc(strlen("Ready")+1);
    strcpy(sendBuffer,"Ready");
    CcsSendDelayedReply(m->reply, strlen("Ready")+1, sendBuffer);

    thisProxy.report_in();

    delete m;
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
    // do this once on each PE, remember we are now in an array element.
    // the (currently valid) assumption is that each PE has at least 1 object.
    if (!CkpvAccess(traceFlagSet)) {
      if (iteration == 0) {
	traceBegin();
	CkpvAccess(traceFlagSet) = true;
      }
    }
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
    //    double a[2000], b[2000], c[2000];
    // This is to get around the tiny default stack size used by the
    // bigsim emulator on certain machines.
    double *a;
    double *b;
    double *c;
    a = new double[2000];
    b = new double[2000];
    c = new double[2000];
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
    delete [] a;
    delete [] b;
    delete [] c;
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
