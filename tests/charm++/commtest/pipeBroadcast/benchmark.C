/*
  Benchmark to test the performance of PipeBroadcast strategy compared to Charm default
  broadcast system.

  Written By Filippo Gioachin, Aug 2004
*/

#include "benchmark.h"
#include <PipeBroadcastStrategy.h>

#define MAX_LENGTH 3
#define MAX_TYPE 9

int lengths[MAX_LENGTH] = {4, 10000, 1000000};
char *conversion[MAX_TYPE] = {"PipeBroadcast nosplit","PipeBroadcast 262000","PipeBroadcast 65500","PipeBroadcast 32700","PipeBroadcast 16300","PipeBroadcast 8096","PipeBroadcast 2048","PipeBroadcast 1024","Charm"};

CProxy_TheMain mainProxy;
CProxy_BTest arr;
ComlibInstanceHandle cinst1;
ComlibInstanceHandle cinst2;
ComlibInstanceHandle cinst3;
ComlibInstanceHandle cinst4;
ComlibInstanceHandle cinst5;
ComlibInstanceHandle cinst6;
ComlibInstanceHandle cinst7;
ComlibInstanceHandle cinst8;
int MAX_ITER;
int threshold_msgs;

TheMain::TheMain(CkArgMsg *msg) {
  threshold_msgs = 500;
  if (msg->argc>2) {
    threshold_msgs = atoi(msg->argv[2]);
  }
  MAX_ITER = 1000;
  if (msg->argc>1) {
    MAX_ITER = atoi(msg->argv[1]) + threshold_msgs;
  }
  delete msg;

  CkPrintf("Benchmark for pipeBroadcast with charm started\n");
  called = 0;
  arr = CProxy_BTest::ckNew(CmiNumPes());

  cinst1 = CkGetComlibInstance();
  mainProxy = thishandle;

  CharmStrategy *strategy = new PipeBroadcastStrategy(USE_HYPERCUBE, arr, 1024);
  cinst1.setStrategy(strategy);

  cinst2 = CkGetComlibInstance();
  strategy = new PipeBroadcastStrategy(USE_HYPERCUBE, arr, 2048);
  cinst2.setStrategy(strategy);

  cinst3 = CkGetComlibInstance();
  strategy = new PipeBroadcastStrategy(USE_HYPERCUBE, arr);
  cinst3.setStrategy(strategy);

  cinst4 = CkGetComlibInstance();
  strategy = new PipeBroadcastStrategy(USE_HYPERCUBE, arr, 16300);
  cinst4.setStrategy(strategy);

  cinst5 = CkGetComlibInstance();
  strategy = new PipeBroadcastStrategy(USE_HYPERCUBE, arr, 32700);
  cinst5.setStrategy(strategy);

  cinst6 = CkGetComlibInstance();
  strategy = new PipeBroadcastStrategy(USE_HYPERCUBE, arr, 65500);
  cinst6.setStrategy(strategy);

  cinst7 = CkGetComlibInstance();
  strategy = new PipeBroadcastStrategy(USE_HYPERCUBE, arr, 262000);
  cinst7.setStrategy(strategy);

  cinst8 = CkGetComlibInstance();
  strategy = new PipeBroadcastStrategy(USE_HYPERCUBE, arr, 100000000);
  cinst8.setStrategy(strategy);
  //CkPrintf("Main: calling send on %d elements\n",numEl);

  //cinst1.beginIteration();
  arr.sendall();
  //arr[2].send();
  //arr[0].sendall();
}

void TheMain::exit() {
  //CkPrintf("called = %d\n",called);
  if (++called >= CmiNumPes()) {
    CkPrintf("All done\n");
    CkExit();
  }
}

BTest::BTest() : cb(CkIndex_BTest::send(0), thisProxy), cball(CkIndex_BTest::sendall(), thisProxy) {
  //comm_debug=1;
  length = 0;
  type = 0;
  count = 0;
  iter = 0;
}

BTest::BTest(CkMigrateMessage *msg) {}

void BTest::sendall() {
  //ComlibManager* myman = CProxy_ComlibManager(cinst1.getComlibManagerID()).ckLocalBranch();
  cinst8.beginIteration();
  //totalTime = 0;
  if (thisIndex==0) CkPrintf("[%2d] Broadcast with length %d\n",CmiMyPe(),lengths[length]);
  send();
}

void BTest::send(CkReductionMsg *msg) {
  delete msg;
  if (thisIndex==0) CkPrintf("[%2d] totaltime with '%s' = %f\n",CmiMyPe(),conversion[type],(CmiWallTimer()-startTime)/(MAX_ITER-threshold_msgs));
  if (++type==MAX_TYPE) {
    if (++length==MAX_LENGTH) {
      mainProxy.exit();
    } else {
      type = 0;
      sendall();
      //contribute(0, &count, CkReduction::sum_int, cball);
    }
  } else {
    //totalTime=0;
    //startTime = CmiWallTimer();
    if (thisIndex==0) send();
    //contribute(0, &count, CkReduction::sum_int, cb);
  }
  
}

void BTest::send() {
  if (thisIndex) return;
  MyMess *mess = new (lengths[length],0) MyMess;
  //mess->data[0] = CkMyPe();
  //for (int i=1; i<LENGTH; ++i) mess->data[i] = i+1000;
  CProxy_BTest copy = thisArrayID;
  ComlibDelegateProxy(&copy);
  //CkPrintf("[%d-%d] sending broadcast\n",CkMyPe(),thisIndex);
  //startTime = CmiWallTimer();
  if (type>=0 && type<=7) {
    copy.receive(mess);
  } else if (type==8) {
    thisProxy.receive(mess);
  } else {
    CkPrintf("BTest: Error, type = %d\n",type);
    CkExit();
  }
}

void BTest::receive(MyMess *msg) {
  delete msg;

  if (thisIndex==0) {
    if (iter==threshold_msgs) startTime=CmiWallTimer();
    //if (iter>0) totalTime += CmiWallTimer() - startTime;
    //if (thisIndex==0) CkPrintf("Finished iteration %d (type %d)\n",iter+1,type);
    if (iter++<MAX_ITER) {
      //CkPrintf("[%2d] totaltime with '%s' = %f\n",CmiMyPe(),conversion[type],totalTime/(iter-1));
      //startTime=CmiWallTimer();
      send();
      //contribute(0, &count, CkReduction::sum_int, cb);
    } else {
      iter=0;
      //CkPrintf("[%2d] totaltime with '%s' = %f\n",CmiMyPe(),conversion[type],(CmiWallTimer()-startTime)/(MAX_ITER-threshold_msgs));
      if (type==0) cinst7.beginIteration();
      else if (type==1) cinst6.beginIteration();
      else if (type==2) cinst5.beginIteration();
      else if (type==3) cinst4.beginIteration();
      else if (type==4) cinst3.beginIteration();
      else if (type==5) cinst2.beginIteration();
      else if (type==5) cinst1.beginIteration();

      // global barrier
      contribute(0, &count, CkReduction::sum_int, cb);

      /*
      if (++type==MAX_TYPE) {
	if (++length==MAX_LENGTH) {
	  mainProxy.exit();
	} else {
	  type = 0;
	  //sendall();
	  contribute(0, &count, CkReduction::sum_int, cball);
	}
      } else {
	totalTime=0;
	//startTime = CmiWallTimer();
	send();
	//contribute(0, &count, CkReduction::sum_int, cb);
      }
      */
    }
  } else {
    if (iter++<MAX_ITER) {
      iter = 0;
      contribute(0, &count, CkReduction::sum_int, cball);
    }
  }
}


#include "benchmark.def.h"
