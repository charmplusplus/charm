/** \file kNeighbor.C
 *  Author: Chao Mei
 *
 */

#include "kNeighbor.decl.h"
#include <stdio.h>
#include <stdlib.h>

#define STRIDEK		1
#define CALCPERSTEP	100
#define WRAPAROUND	1
#define ALLCROSSNODE	0

#define DEBUG		0
#define REUSE_ITER_MSG	0
#define TOUCH_MSGDATA	0

#define WORKSIZECNT	1
#define MSGSIZECNT	1

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
#if TOUCH_MSGDATA
      init();
#endif
    }

    void setMsgSrc(int X, int id) {
      fromX = X;
      nID = id;
    }

    void init() {
      for (int i=0; i<size; i++)
        data[i] = i;
    }

    int sum() {
      int s=0;
      for (int i=0; i<size; i++)
        s += data[i];
      return s;
    }
};

class Main: public CBase_Main {
  public:
    CProxy_Block array;

    //static int msgSizeArr[MSGSIZECNT];

    int numSteps;
    int currentStep;
    int currentMsgSize;

    int elementsRecved;
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
#if REUSE_ITER_MSG
      gMsgSize = currentMsgSize;
#endif

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
	CkPrintf("kNeighbor program finished!\n");
	//CkCallback *cb = new CkCallback(CkIndex_Main::terminate(NULL), thisProxy);
	//array.ckSetReductionClient(cb);
	//array.printSts(numSteps);
	terminate(NULL);
	return;
	//CkExit();
      }

      elementsRecved = 0;
      totalTime = 0.0;
      maxTime = 0.0;
      minTime = 3600.0;

      //int msgSize = msgSizeArr[currentStep%MSGSIZECNT];
      //int msgSize = msgSizeArr[rand()%MSGSIZECNT];
      //currentMsgSize = msgSize;
      if(currentStep!=0 && (currentStep % gLBFreq == 0)) {
	array.pauseForLB();
	return;
      }

      gStarttime = CmiWallTimer();
#if REUSE_ITER_MSG
      for (int i=0; i<num_chares; i++)
	array[i].commWithNeighbors();
#else
      for (int i=0; i<num_chares; i++)
	array[i].commWithNeighbors(currentMsgSize);
#endif	
      //array.commWithNeighbors(currentMsgSize);
    }
    
    void resumeIter() {
#if DEBUG
      CkPrintf("Resume iteration at step %d\n", currentStep);
#endif
      gStarttime = CmiWallTimer();
#if REUSE_ITER_MSG
      for (int i=0; i<num_chares; i++)
	array[i].commWithNeighbors();
#else
      for (int i=0; i<num_chares; i++)
	array[i].commWithNeighbors(currentMsgSize);
#endif
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

    void nextStep_plain(double iterTime) {
      elementsRecved++;
      totalTime += iterTime;
      maxTime = maxTime>iterTime?maxTime:iterTime;
      minTime = minTime<iterTime?minTime:iterTime;

      if (elementsRecved == num_chares) {
	  double wholeStepTime = CmiWallTimer() - gStarttime;
	  timeRec[currentStep] = wholeStepTime/CALCPERSTEP;
	  //CkPrintf("Step %d with msg size %d finished: max=%f, total=%f\n", currentStep, currentMsgSize, maxTime/CALCPERSTEP, wholeStepTime/CALCPERSTEP);

	  beginIteration();
      }
    }

    void nextStep(CkReductionMsg  *msg) {
      maxTime = *((double *)msg->getData());
      delete msg;
      double wholeStepTime = CmiWallTimer() - gStarttime;
      timeRec[currentStep] = wholeStepTime/CALCPERSTEP;
      //CkPrintf("Step %d with msg size %d finished: max=%f, total=%f\n", currentStep, currentMsgSize, maxTime/CALCPERSTEP, wholeStepTime/CALCPERSTEP);
      beginIteration();
    }

};

//int Main::msgSizeArr[MSGSIZECNT] = {16, 32, 128, 256, 512, 1024, 2048, 4096};
//int Main::msgSizeArr[MSGSIZECNT] = {10000};

//no wrap around for sending messages to neighbors
class Block: public CBase_Block {
  public:
    /** actual work size is of workSize^3 */
    static int workSizeArr[WORKSIZECNT];

    int numNeighbors;
    int neighborsRecved;
    int *neighbors;
    double *recvTimes;
    double startTime;

    int random;
    int curIterMsgSize;
    int curIterWorkSize;
    int internalStepCnt;
    int sum;

#if REUSE_ITER_MSG
    toNeighborMsg **iterMsg;
#endif

  public:
    Block() {
      //srand(thisIndex.x+thisIndex.y);
      usesAtSync = CmiTrue;

#if WRAPAROUND
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
#elif ALLCROSSNODE
      if(CkNumNodes()==1){
	if(thisIndex==0){
	  CkPrintf("This version has to run with more than 2 nodes!\n");
	  CkExit();
	}
	return;
      }
      numNeighbors = CkNumNodes()-1;
      neighbors = new int[numNeighbors];
      recvTimes = new double[numNeighbors];
      for(int i=0; i<numNeighbors; i++){
	neighbors[i] = (thisIndex+(i+1)*CmiMyNodeSize())%CkNumPes();
      }
#else
      //calculate the neighbors this element has
      numNeighbors = 0;
      numNeighbors += thisIndex - MAX(0, thisIndex-STRIDEK); //left
      numNeighbors += MIN(num_chares-1, thisIndex+STRIDEK)-thisIndex; //right
      neighbors = new int[numNeighbors];
      recvTimes = new double[numNeighbors];
      int nidx=0;
      for (int i=MAX(0, thisIndex-STRIDEK); i<thisIndex; i++, nidx++) neighbors[nidx]=i;
      for (int i=thisIndex+1; i<=MIN(num_chares-1, thisIndex+STRIDEK); i++, nidx++) neighbors[nidx] = i;
#endif

      for (int i=0; i<numNeighbors; i++)
	recvTimes[i] = 0.0;

#if REUSE_ITER_MSG
      iterMsg = new toNeighborMsg *[numNeighbors];
      for (int i=0; i<numNeighbors; i++)
	iterMsg[i] = NULL;	
#endif

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
#if REUSE_ITER_MSG
      delete [] iterMsg;
#endif
    }
    
    void pup(PUP::er &p){
      ArrayElement1D::pup(p); //pack our superclass
      p(workSizeArr, WORKSIZECNT);
      p(numNeighbors);
      p(neighborsRecved);

      if(p.isUnpacking()) {
	neighbors = new int[numNeighbors];
	recvTimes = new double[numNeighbors];
      }
      PUParray(p, neighbors, numNeighbors);
      PUParray(p, recvTimes, numNeighbors);
      p(startTime);
      p(random);
      p(curIterMsgSize);
      p(curIterWorkSize);
      p(internalStepCnt);
      p(sum);
#if REUSE_ITER_MSG
      if(p.isUnpacking()) iterMsg = new toNeighborMsg *[numNeighbors];
      for(int i=0; i<numNeighbors; i++){
	if(p.isUnpacking()) iterMsg[i] = new(curIterMsgSize/4, 0) toNeighborMsg(curIterMsgSize/4);
	CkPupMessage(p, (void **)&iterMsg[i]);
      }
#endif			
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

    void printSts(int totalSteps){
      /*for(int i=0; i<numNeighbors; i++){
	CkPrintf("Elem[%d]: avg RTT from neighbor %d (actual elem id %d): %lf\n", thisIndex, i, neighbors[i], recvTimes[i]/totalSteps);
	}*/
      contribute(0,0,CkReduction::max_int);
    }

    void startInternalIteration() {
#if DEBUG
      CkPrintf("[%d]: Start internal iteration \n", thisIndex);
#endif

      neighborsRecved = 0;
      /* 1: pick a work size and do some computation */
      int sum=0;
      int N=curIterWorkSize;
      for (int i=0; i<N; i++)
	for (int j=0; j<N; j++)
	  for (int k=0; k<N; k++)
	    sum += (thisIndex*i+thisIndex*j+k)%WORKSIZECNT;
      /* 2. send msg to K neighbors */
      int msgSize = curIterMsgSize;

      // Send msgs to neighbors
      for (int i=0; i<numNeighbors; i++) {
	//double memtimer = CmiWallTimer();

#if REUSE_ITER_MSG
	toNeighborMsg *msg = iterMsg[i];
#else
	toNeighborMsg *msg = new(msgSize/4, 0) toNeighborMsg(msgSize/4);
#endif

#if DEBUG
	CkPrintf("[%d]: send msg to neighbor[%d]=%d\n", thisIndex, i, neighbors[i]);
#endif
	msg->setMsgSrc(thisIndex, i);
	//double entrytimer = CmiWallTimer();
	thisProxy(neighbors[i]).recvMsgs(msg);
	//double entrylasttimer = CmiWallTimer();
	//if(thisIndex==0){
	//	CkPrintf("At current step %d to neighbor %d, msg creation time: %f, entrymethod fire time: %f\n", internalStepCnt, neighbors[i], entrytimer-memtimer, entrylasttimer-entrytimer);
	//}
      }
    }

    void commWithNeighbors(int msgSize) {
      internalStepCnt = 0;
      curIterMsgSize = msgSize;
      //currently the work size is only changed every big steps (which
      //are initiated by the main proxy
      curIterWorkSize = workSizeArr[random%WORKSIZECNT];
      random++;

      startTime = CmiWallTimer();
      startInternalIteration();
    }

    void commWithNeighbors() {
      internalStepCnt = 0;
      curIterMsgSize = gMsgSize;
      //currently the work size is only changed every big steps (which
      //are initiated by the main proxy
      curIterWorkSize = workSizeArr[random%WORKSIZECNT];
      random++;

#if REUSE_ITER_MSG
      if(iterMsg[0]==NULL) { //indicating the messages have not been created
	for(int i=0; i<numNeighbors; i++)
	  iterMsg[i] = new(curIterMsgSize/4, 0) toNeighborMsg(curIterMsgSize/4);
      }
#endif

      startTime = CmiWallTimer();
      startInternalIteration();
    }

    void recvReplies(toNeighborMsg *m) {
      int fromNID = m->nID;

#if DEBUG
      CkPrintf("[%d]: receive ack from neighbor[%d]=%d\n", thisIndex, fromNID, neighbors[fromNID]);
#endif

#if REUSE_ITER_MSG
      iterMsg[fromNID] = m;
#else
      delete m;
#endif
      //recvTimes[fromNID] += (CmiWallTimer() - startTime);

      //get one step time and send it back to mainProxy
      neighborsRecved++;
      if (neighborsRecved == numNeighbors) {
	internalStepCnt++;
	if (internalStepCnt==CALCPERSTEP) {
	  double iterCommTime = CmiWallTimer() - startTime;
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

#if TOUCH_MSGDATA
      sum = m->sum();
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

//int Block::workSizeArr[WORKSIZECNT] = {20, 60, 120, 180, 240};
int Block::workSizeArr[WORKSIZECNT] = {20};

#include "kNeighbor.def.h"
