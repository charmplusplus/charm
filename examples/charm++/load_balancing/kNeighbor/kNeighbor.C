/** \file kNeighbor.C
 *  Author: Chao Mei (chaomei2@illinois.edu)
 *
 *  Heavily modified by Abhinav Bhatele (bhatele@illinois.edu) 2011/02/13
 */

#include "kNeighbor.decl.h"
#include <stdio.h>
#include <stdlib.h>

#define STRIDEK		1
#define CALCPERSTEP	100

#define DEBUG		0


/* readonly */ CProxy_Main mainProxy;
/* readonly */ int num_chares;
/* readonly */ int gMsgSize;
/* readonly */ int gLBFreq;

int cmpFunc(const void *a, const void *b) {
  if(*(double *)a < *(double *)b) return -1;
  if(*(double *)a > *(double *)b) return 1;
  return 0;
}

class toNeighborMsg: public CMessage_toNeighborMsg {
  public:
    int *data;
    int size;
    int fromX;
    int nID;

  public:
    toNeighborMsg() {};
    toNeighborMsg(int s): size(s) {  
    }

    void setMsgSrc(int X, int id) {
      fromX = X;
      nID = id;
    }

};

class Main: public CBase_Main {
  public:
    CProxy_Block array;

    int numSteps;
    int currentStep;
    int currentMsgSize;

    int numElemsRcvd;
    double totalTime;
    double maxTime;
    double minTime;
    double *timeRec;

    double gStarttime;

  public:
    Main(CkArgMsg *m) {
      mainProxy = thisProxy;
      CkPrintf("\nStarting kNeighbor ...\n");

      if (m->argc!=4 && m->argc!=5) {
	CkPrintf("Usage: %s <#elements> <#iterations> <msg size> [ldb freq]\n", m->argv[0]);
	delete m;
	CkExit();
      }

      num_chares = atoi(m->argv[1]);
      if(num_chares < CkNumPes()) {
	printf("Warning: #elements is forced to be equal to #pes\n");
	num_chares = CkNumPes();
      }

      numSteps = atoi(m->argv[2]);
      currentMsgSize = atoi(m->argv[3]);

      gLBFreq = 100000;
      if(m->argc==5) {
	gLBFreq = atoi(m->argv[4]);
      }

#if TURN_ON_LDB
      printf("Setting load-balancing freq to every %d steps\n", gLBFreq);
#endif
      gMsgSize = currentMsgSize;

      currentStep = -1;
      timeRec = new double[numSteps];

      array = CProxy_Block::ckNew(num_chares);
      CkCallback *cb = new CkCallback(CkIndex_Main::nextStep(NULL), thisProxy);
      array.ckSetReductionClient(cb);

      beginIteration();
    }

    void beginIteration() {
      currentStep++;
      if (currentStep == numSteps) {
	CkPrintf("kNeighbor program finished!\n\n");
	//CkCallback *cb = new CkCallback(CkIndex_Main::terminate(NULL), thisProxy);
	//array.ckSetReductionClient(cb);
	terminate(NULL);
	return;
      }

      numElemsRcvd = 0;
      totalTime = 0.0;
      maxTime = 0.0;
      minTime = 3600.0;

      //currentMsgSize = msgSize;
      if(currentStep!=0 && (currentStep % gLBFreq == 0)) {
	array.pauseForLB();
	return;
      }

      gStarttime = CkWallTimer();
	array.commWithNeighbors();
    }

    void resumeIter() {
#if DEBUG
      CkPrintf("Resume iteration at step %d\n", currentStep);
#endif
      gStarttime = CkWallTimer();
	array.commWithNeighbors();
    }

    void terminate(CkReductionMsg *msg) {
      delete msg;
      double total = 0.0;

      for (int i=0; i<numSteps; i++)
	timeRec[i] = timeRec[i]*1e6;

      qsort(timeRec, numSteps, sizeof(double), cmpFunc);
      printf("Time stats: lowest: %f, median: %f, highest: %f\n", timeRec[0], timeRec[numSteps/2], timeRec[numSteps-1]);

      int samples = 100;
      if(numSteps<=samples) samples = numSteps-1;
      for (int i=0; i<samples; i++)
	total += timeRec[i];
      total /= samples;

      CkPrintf("Average time for each %d-Neighbor iteration with msg size %d is %f (us)\n", STRIDEK, currentMsgSize, total);
      CkExit();
    }

    void nextStep(CkReductionMsg  *msg) {
      maxTime = *((double *)msg->getData());
      delete msg;
      double wholeStepTime = CkWallTimer() - gStarttime;
      timeRec[currentStep] = wholeStepTime/CALCPERSTEP;
      if(currentStep % 10 == 0)
	CkPrintf("Step %d with msg size %d finished: max=%f, total=%f\n", currentStep, currentMsgSize, maxTime/CALCPERSTEP, wholeStepTime/CALCPERSTEP);
      beginIteration();
    }

};


//no wrap around for sending messages to neighbors
class Block: public CBase_Block {
  public:
    int numNeighbors;
    int numNborsRcvd;
    int *neighbors;
    double *recvTimes;
    double startTime;

    int random;
    int curIterMsgSize;
    int internalStepCnt;
    int sum;

    toNeighborMsg **iterMsg;

  public:
    Block() {
      //srand(thisIndex.x+thisIndex.y);
      usesAtSync = true;

      numNeighbors = 2*STRIDEK;
      neighbors = new int[numNeighbors];
      recvTimes = new double[numNeighbors];
      int nidx=0;
      //setting left neighbors
      for (int i=thisIndex-STRIDEK; i<thisIndex; i++, nidx++) {
	int tmpnei = i;
	while (tmpnei<0) tmpnei += num_chares;
	neighbors[nidx] = tmpnei;
      }
      //setting right neighbors
      for (int i=thisIndex+1; i<=thisIndex+STRIDEK; i++, nidx++) {
	int tmpnei = i;
	while (tmpnei>=num_chares) tmpnei -= num_chares;
	neighbors[nidx] = tmpnei;
      }

      for (int i=0; i<numNeighbors; i++)
	recvTimes[i] = 0.0;

      iterMsg = new toNeighborMsg *[numNeighbors];
      for (int i=0; i<numNeighbors; i++)
	iterMsg[i] = NULL;

#if DEBUG
      CkPrintf("Neighbors of %d: ", thisIndex);
      for (int i=0; i<numNeighbors; i++)
	CkPrintf("%d ", neighbors[i]);
      CkPrintf("\n");
#endif

      random = thisIndex*31+73;
    }

    ~Block() {
      delete [] neighbors;
      delete [] recvTimes;
      delete [] iterMsg;
    }

    void pup(PUP::er &p){
      p(numNeighbors);
      p(numNborsRcvd);

      if(p.isUnpacking()) {
	neighbors = new int[numNeighbors];
	recvTimes = new double[numNeighbors];
      }
      PUParray(p, neighbors, numNeighbors);
      PUParray(p, recvTimes, numNeighbors);
      p(startTime);
      p(random);
      p(curIterMsgSize);
      p(internalStepCnt);
      p(sum);
      if(p.isUnpacking()) iterMsg = new toNeighborMsg *[numNeighbors];
      for(int i=0; i<numNeighbors; i++){
	CkPupMessage(p, (void **)&iterMsg[i]);
      }
    }

    Block(CkMigrateMessage *m) {}

    void pauseForLB(){
#if DEBUG
      CkPrintf("Element %d pause for LB on PE %d\n", thisIndex, CkMyPe());
#endif
      AtSync();
    }

    void ResumeFromSync(){ //Called by load-balancing framework
      CkCallback cb(CkIndex_Main::resumeIter(), mainProxy);
      contribute(0, NULL, CkReduction::sum_int, cb);
    }

    void startInternalIteration() {
#if DEBUG
      CkPrintf("[%d]: Start internal iteration \n", thisIndex);
#endif

      numNborsRcvd = 0;
      /* 1: pick a work size and do some computation */
      int N = (thisIndex * thisIndex / num_chares) * 100;
      for (int i=0; i<N; i++)
	for (int j=0; j<N; j++) {
	  sum += (thisIndex * i + j);
	}

      /* 2. send msg to K neighbors */
      int msgSize = curIterMsgSize;

      // Send msgs to neighbors
      for (int i=0; i<numNeighbors; i++) {
	//double memtimer = CkWallTimer();

	toNeighborMsg *msg = iterMsg[i];

#if DEBUG
	CkPrintf("[%d]: send msg to neighbor[%d]=%d\n", thisIndex, i, neighbors[i]);
#endif
	msg->setMsgSrc(thisIndex, i);
	//double entrytimer = CkWallTimer();
	thisProxy(neighbors[i]).recvMsgs(msg);
	//double entrylasttimer = CkWallTimer();
	//if(thisIndex==0){
	//	CkPrintf("At current step %d to neighbor %d, msg creation time: %f, entrymethod fire time: %f\n", internalStepCnt, neighbors[i], entrytimer-memtimer, entrylasttimer-entrytimer);
	//}
      }
    }

    void commWithNeighbors() {
      internalStepCnt = 0;
      curIterMsgSize = gMsgSize;
      //currently the work size is only changed every big steps (which
      //are initiated by the main proxy
      random++;

      if(iterMsg[0]==NULL) { //indicating the messages have not been created
	for(int i=0; i<numNeighbors; i++)
	  iterMsg[i] = new(curIterMsgSize/4, 0) toNeighborMsg(curIterMsgSize/4);
      }

      startTime = CkWallTimer();
      startInternalIteration();
    }

    void recvReplies(toNeighborMsg *m) {
      int fromNID = m->nID;

#if DEBUG
      CkPrintf("[%d]: receive ack from neighbor[%d]=%d\n", thisIndex, fromNID, neighbors[fromNID]);
#endif

      iterMsg[fromNID] = m;
      //recvTimes[fromNID] += (CkWallTimer() - startTime);

      //get one step time and send it back to mainProxy
      numNborsRcvd++;
      if (numNborsRcvd == numNeighbors) {
	internalStepCnt++;
	if (internalStepCnt==CALCPERSTEP) {
	  double iterCommTime = CkWallTimer() - startTime;
	  contribute(sizeof(double), &iterCommTime, CkReduction::max_double);
	  /*if(thisIndex==0){
	    for(int i=0; i<numNeighbors; i++){
	    CkPrintf("RTT time from neighbor %d (actual elem id %d): %lf\n", i, neighbors[i], recvTimes[i]);
	    }
	    }*/
	} else {
	  startInternalIteration();
	}
      }
    }

    void recvMsgs(toNeighborMsg *m) {
#if DEBUG
      CkPrintf("[%d]: recv msg from %d as its %dth neighbor\n", thisIndex, m->fromX, m->nID);
#endif

      thisProxy(m->fromX).recvReplies(m);
    }

    inline int MAX(int a, int b) {
      return (a>b)?a:b;
    }
    inline int MIN(int a, int b) {
      return (a<b)?a:b;
    }
};

#include "kNeighbor.def.h"
