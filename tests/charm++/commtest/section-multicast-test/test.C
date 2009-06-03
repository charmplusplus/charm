#include "test.decl.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <comlib.h>
 

CProxy_Main mainProxy;
CProxy_LUBlk luArrProxy;
ComlibInstanceHandle cinst;

#define BLKSIZE 20
#define NumCols 10
#define NumRows 10
#define COOKIE 777777
#define numIterations 10

// Test multicasts to each row of a 2d chare array.
// Element 0,0 performs a section multicast to each row.
// DirectMulticastStrategy is used
// Quiescence is used for startup


class blkMsg: public CMessage_blkMsg {
public:
  double *data;
};


class Main : public CBase_Main {
public:

  double startTime;
  int iter;

  Main(CkArgMsg* m) {
    com_debug = 0;
    iter = 0;
    mainProxy = thisProxy;
	
    CkPrintf("Testing commlib array section multicast on %d processors (%d nodes)\n",  CkNumPes(), CmiNumNodes());
	
    luArrProxy = CProxy_LUBlk::ckNew(NumRows,NumCols);

    Strategy *strategy = new DirectMulticastStrategy();	

    cinst = ComlibRegister(strategy);

    startTime = CmiWallTimer();

    CkCallback *cb = new CkCallback(CkIndex_Main::startIteration(), thisProxy);
    CkStartQD(*cb);

  }

  void startIteration() {
    CkCallback *cb = new CkCallback(CkIndex_Main::iterationCompleted(NULL), thisProxy);
    luArrProxy.ckSetReductionClient(cb);
    luArrProxy(0,0).start();
  }

  void iterationCompleted(CkReductionMsg *msg) {
    iter++;
    if(iter == numIterations){
      double endTime = CmiWallTimer();
      double duration = endTime-startTime;
      CkPrintf("Test Completed Successfully\n");
      CkPrintf("Time: %fs\n", duration);
      CkExit();
    } else {
      startIteration();
    }
 
  }

  
};



class LUBlk: public CBase_LUBlk {

public:
  LUBlk() {}
  ~LUBlk() {}
  LUBlk(CkMigrateMessage* m) {}
  void pup(PUP::er &p) {}


  // start() should be called on one element in the array
  // It will create an array section for each row in the array, and then multicast to each row section
  // The array sections will result in the recvMessage method being called once for each array element
  void start() {
    for(int i=0;i<NumRows;i++){
      CProxySection_LUBlk sect = CProxySection_LUBlk::ckNew(thisArrayID, i, i, 1, 0,NumCols-1, 1);
      ComlibAssociateProxy(cinst, sect); 
      blkMsg *msg = new (BLKSIZE) blkMsg;
      msg->data[0] = COOKIE;
      sect.recvMessage(msg);
    }
  }
  

  // Recieve the section multicast and 
  void recvMessage(blkMsg* m){
    CkAssert(m->data[0] == COOKIE);
    delete m;

    //	CkPrintf("recvMessage for %d,%d\n", thisIndex.x, thisIndex.y);
    contribute(0, 0, CkReduction::sum_int);
  }
  
};


#include "test.def.h"
