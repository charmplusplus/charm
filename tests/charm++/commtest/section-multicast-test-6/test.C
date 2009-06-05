#include "test.decl.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <comlib.h>

// Each element sends a multicast to the successor elements in the 1D array, after recieving all multicasts from its predecessor elements
 

CProxy_Main mainProxy;
CProxy_LUBlk luArrProxy;
ComlibInstanceHandle cinst_direct;
ComlibInstanceHandle cinst_ring;
ComlibInstanceHandle cinst_multiring;


#define BLKSIZE 10
#define NumElem 40
#define COOKIE 777777
#define NUM_ITER 10

#define DEBUG 0

class blkMsg: public CMessage_blkMsg {
public:
  double *data;
};


class Main : public CBase_Main {
public:

  double startTime;
  int iteration;

  Main(CkArgMsg* m) {
    iteration = 0;    
    com_debug = DEBUG;
    
    mainProxy = thisProxy;
    
    CkPrintf("Testing commlib array section multicast on %d processors (%d nodes)\n",
	     CkNumPes(), CmiNumNodes());
    
    luArrProxy = CProxy_LUBlk::ckNew(NumElem);
    
    
    Strategy *strategy_direct = new DirectMulticastStrategy();  
    cinst_direct = ComlibRegister(strategy_direct);

    Strategy *strategy_ring = new RingMulticastStrategy();  
    cinst_ring = ComlibRegister(strategy_ring);

    Strategy *strategy_multiring = new MultiRingMulticastStrategy();
    cinst_multiring = ComlibRegister(strategy_multiring);

    

    CkCallback *cb = new CkCallback(CkIndex_Main::iterationCompleted(NULL), thisProxy);
    luArrProxy.ckSetReductionClient(cb);
    startTime = CmiWallTimer();

    startIteration();
  }


  void iterationCompleted(CkReductionMsg *msg) {
    //    CkPrintf("iteration %d completed\n", iteration);
    iteration ++;
    if(iteration == NUM_ITER){
      done(); 
    } else {
      startIteration();
    }
  }



  void startIteration(){
    blkMsg *msg = new (BLKSIZE) blkMsg;
    msg->data[1] = COOKIE;
    msg->data[BLKSIZE-1] = COOKIE;
    luArrProxy.recvMessage(msg);
  }

  void done(){
    double endTime = CmiWallTimer();
    double duration = endTime-startTime;
    CkPrintf("Test Completed Successfully");
    CkPrintf("Time: %fs\n", duration);
    CkExit();
  }

 
  
};



class LUBlk: public CBase_LUBlk {
  int num_received;


public:
  LUBlk() {
    num_received = 0;
    com_debug = DEBUG;
    srand(CkMyPe());
  }

  ~LUBlk() {}
  
  LUBlk(CkMigrateMessage* m) {}
  
  void pup(PUP::er &p) {
    p | num_received;
  }
  
  
  
  // Recieve the section multicasts from previous elements in the array.
  // Once enough messages have arrived, continue on by sending a message.
  // The messages may arrive out of order.
  void recvMessage(blkMsg* m){
    //    CkPrintf("recvMessage num_received=%d for element %d\n", num_received, thisIndex);
    CkAssert(m->data[1] == COOKIE);
    CkAssert(m->data[BLKSIZE-1] == COOKIE);
    delete m;

    if(thisIndex == num_received) {
#if DEBUG
      CkPrintf("Element %d is sending\n", thisIndex);
#endif
      if(thisIndex < NumElem-1) {
	CProxySection_LUBlk sect = CProxySection_LUBlk::ckNew(thisArrayID, thisIndex+1, NumElem-1, 1);

	switch(rand() % 3){
	case 0:
	  ComlibAssociateProxy(cinst_direct, sect); 
	  break;
	case 1:
	  ComlibAssociateProxy(cinst_ring, sect); 
	  break;
	case 2:
	  ComlibAssociateProxy(cinst_multiring, sect); 
	  break;
	}

	blkMsg *msg = new (BLKSIZE) blkMsg;
	msg->data[1] = COOKIE;
	msg->data[BLKSIZE-1] = COOKIE;
	sect.recvMessage(msg);
      }
      num_received = -1;
      contribute(0, 0, CkReduction::sum_int);
    }

    num_received ++;
  }
  
};


#include "test.def.h"
