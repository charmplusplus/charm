#include <string.h> // for strlen, and strcmp
#include <charm++.h>

#include "largedata.decl.h"

#define NITER 100
#define PAYLOAD (1ull<<30ull)

CProxy_main mainProxy;
int iterations;
size_t payload;

#define DEBUG(x) //x

class main : public CBase_main
{
  int niter, counter;
  char *buffer;
  CProxy_LargeDataNodeGroup ngid;
  double start_time, end_time, total_time;
  CkCallback cb;

public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg* m)
  {
    iterations = NITER;
    payload = PAYLOAD;
    if(m->argc>1)
      payload=atoi(m->argv[1]);
    if(m->argc>2)
      iterations=atoi(m->argv[2]);
    if(m->argc>3) {
      CkPrintf("Usage: large_bcast +p2 [payload] [iterations]\n Where payload (default %zd) is integer >0, iterations (default %d) is integer >0\n", (size_t)PAYLOAD, NITER);
      CmiAbort("Incorrect arguments\n");
    }
    CkPrintf("Large data transfer with payload: %zd iterations: %d\n", payload, iterations);
    delete m;

    // Initialize
    niter = 0;
    counter = 0;
    total_time = 0;
    mainProxy = thisProxy;

    // Allocate a buffer to send
    buffer = new char[payload];

    // Create a nodegroup
    ngid = CProxy_LargeDataNodeGroup::ckNew(iterations);

    // Create a callback method to pass in the Zerocopy Bcast API call
    int idx_zerocopySent = CkIndex_main::zerocopySent(NULL);
    cb = CkCallback(idx_zerocopySent, thisProxy);

    // Regular Bcast API's warmup run, which is untimed
    ngid.recv(buffer, payload);
  }

  // Invoked on main after a reduction by all the nodegroup elements
  void regular_bcast_done() {
    niter++; // An iteration of the Regular Bcast API is complete

    DEBUG(CkPrintf("[%d][%d][%d] Iteration %d: Regular Bcast API reduction complete, data received by all nodegroup elements\n", CkMyPe(), CkMyNode(), CmiMyRank(), niter);)

    if(niter != 1) {  // For all runs excluding the warmup run
      end_time = CkWallTimer();
      total_time += end_time - start_time;

      if(niter == iterations + 1) {   // 1 is added for the warmup run
        // All iterations have been completed;
        // print result
        CkPrintf("[%d][%d][%d] Regular API sending complete, Time taken per iteration after %d iterations is: %lf us\n", CkMyPe(), CkMyNode(), CmiMyRank(), iterations, 1.0e6*total_time/iterations);

        // Initialize
        niter = 0;
        total_time = 0;

        // Zerocopy Bcast API's warmup run, which is untimed
        ngid.recv_zerocopy(CkSendBuffer(buffer, cb), payload);
        return;
      }
    }
    start_time = CkWallTimer();
    ngid.recv(buffer, payload); // regular API's non-warmup run
  }




  void zerocopySent(CkDataMsg *msg) {
    niter++; // An iteration of the Zerocopy Bcast API is complete

    DEBUG(CkPrintf("[%d][%d][%d] Iteration %d: Zerocopy Bcast API callback invocation complete, data received by all nodegroup elements\n", CkMyPe(), CkMyNode(), CmiMyRank(), niter);)

    if(niter != 1) {  // For all runs excluding the warmup run
      end_time = CkWallTimer();
      total_time += end_time - start_time;

      if(niter == iterations + 1) {   // 1 is added for the warmup run
        // All iterations have been completed;
        // print result
        CkPrintf("[%d][%d][%d] Bcast API sending complete, Time taken per iteration after %d iterations is: %lf us\n", CkMyPe(), CkMyNode(), CmiMyRank(), iterations, 1.0e6*total_time/iterations);
        done();
        return;
      }
    }

    start_time = CkWallTimer();
    ngid.recv_zerocopy(CkSendBuffer(buffer, cb), payload);
    delete msg;
  }

  void done() {
    counter++;
    if(counter == CkNumNodes() + 1) {  // 1 is added for the done call from the callback
      CkPrintf("[%d][%d][%d] All recipients have received the data\n", CkMyPe(), CkMyNode(), CmiMyRank());
      CkExit();
    }
  }
};


class LargeDataNodeGroup : public CBase_LargeDataNodeGroup
{
  int niter, iterations;
  CkCallback regCb;

public:
  LargeDataNodeGroup(int _iterations) {
    niter = 0;
    iterations = _iterations;
    regCb = CkCallback(CkReductionTarget(main, regular_bcast_done), mainProxy);
  }

  void recv(char *msg, size_t size) {
    niter++;
    DEBUG(CkPrintf("[%d][%d][%d] Iteration %d: Received data through regular API\n", CkMyPe(), CkMyNode(), CmiMyRank(), niter);)
    if(niter == iterations + 1) { // 1 is added for the warmup run
      niter = 0; // Reset the value of niter as regular API is complete
    }
    contribute(regCb); // Nodegroup reduction to signal completion to the main chare
  }

  void recv_zerocopy(char *msg, size_t size) {
    niter++;
    DEBUG(CkPrintf("[%d][%d][%d] Iteration %d: Received data through zerocopy API\n", CkMyPe(), CkMyNode(), CmiMyRank(), niter);)
    if(niter == iterations + 1) { // 1 is added for the warmup run
      mainProxy.done(); // Signal mainchare to indicate that data has successfully been received
    }
  }
};


#include "largedata.def.h"
