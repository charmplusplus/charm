/** \file: blue.C -- Converse BlueGene Emulator Code
 *  \addtogroup  Emulator
 *  Emulator written by Gengbin Zheng, gzheng@uiuc.edu on 2/20/2001
 *

Emulator emulates a taget machine with much large number of nodes/processors.
In the programmer's view, each node consists of a number of hardware-supported
threads. Within a node, threads are divided into worker threads and
communication threads. Communication thread's only job is to check incoming
messages from the network and put the messages in either a worker's queue or a
node global queue.  Worker threads repeatedly retrieve messages from the queues
and execute the handler functions associated with the messages. Emulators
provides APIs to allow a thread to send a message to another thread. Similar to
Converse message, the header of each emulator message encodes a handler
function to be invoked at the destination. Each worker thread continuously
polls message from its queue for incoming messages. For each message, the
designated handler function associated with the message is invoked.

A low level API is described in BigSim manual. It is a set of functions
typically starts with prefix "Bg". It implements functions for setting up the
emulated machine configurations and message passing. 

In order for Charm++ to port on the emulator:
1. Cpv => Ckpv,  because emulator processor private variables
2. Some Cmi functions change to Ck version of the same functions, if there is a Ck version
2. Some Converse functions need to be manually converted to calling Bg functions (BigSim low level API)
3. Converse threads need to be specially handled (the scheduling strategy)
 
Out-of-core execution and record-replay are two relatively new features added
to the emulator.

 */

 
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include "cklists.h"
#include "queueing.h"
#include "blue.h"
#include "blue_impl.h"    	// implementation header file
//#include "blue_timing.h" 	// timing module
#include "bigsim_record.h"

#include "bigsim_ooc.h" //out-of-core module
#include "bigsim_debug.h"
#include "errno.h"

//#define  DEBUGF(x)      //CmiPrintf x;

#undef DEBUGLEVEL
#define DEBUGLEVEL 10

const double CHARM_OVERHEAD = 0.5E-6;

/* node level variables */
CpvDeclare(nodeInfo*, nodeinfo);		/* represent a bluegene node */

/* thread level variables */
CtvDeclare(threadInfo *, threadinfo);	/* represent a bluegene thread */

CpvStaticDeclare(CthThread, mainThread);

/* BG machine parameter */
CpvDeclare(BGMach, bgMach);	/* BG machine size description */
CpvDeclare(int, numNodes);        /* number of bg nodes on this PE */

/* emulator node level variables */
CpvDeclare(SimState, simState);

CpvDeclare(int, CthResumeBigSimThreadIdx);

static int arg_argc;
static char **arg_argv;

CmiNodeLock initLock;     // used for BnvInitialize

int _bgSize = 0;			// short cut of blue gene node size
int delayCheckFlag = 1;          // when enabled, only check correction 
					// messages after some interval
int programExit = 0;

static int bgstats_flag = 0;		// flag print stats at end of simulation

// for debugging log
FILE *bgDebugLog;			// for debugging

#ifdef CMK_ORIGIN2000
extern "C" int start_counters(int e0, int e1);
extern "C" int read_counters(int e0, long long *c0, int e1, long long *c1);
inline double Count2Time(long long c) { return c*5.e-7; }
#elif CMK_HAS_COUNTER_PAPI
#include <papi.h>
int *papiEvents = NULL;
int numPapiEvents;
long_long *papiValues = NULL;
CmiUInt8 *total_papi_counters = NULL;
char **papi_counters_desc = NULL;
#define MAX_TOTAL_EVENTS 10
#endif

BgTracingFn userTracingFn = NULL;

/****************************************************************************
     little utility functions
****************************************************************************/

char **BgGetArgv() { return arg_argv; }
int    BgGetArgc() { return arg_argc; }

/***************************************************************************
     Implementation of the same counter interface currently used for the
     Origin2000 using PAPI in the underlying layer.

     Because of the difference in counter numbering, it is assumed that
     the two counters desired are CYCLES and FLOPS and hence the numbers
     e0 and e1 are ignored.

     start_counters is also not implemented because PAPI does things in
     a different manner.
****************************************************************************/

#if CMK_HAS_COUNTER_PAPI
/*
CmiUInt8 total_ins = 0;
CmiUInt8 total_fps = 0;
CmiUInt8 total_l1_dcm = 0;
CmiUInt8 total_mem_rcy = 0;
*/
int init_counters()
{
    int retval = PAPI_library_init(PAPI_VER_CURRENT);

    if (retval != PAPI_VER_CURRENT) { CmiAbort("PAPI library init error!"); } 

	//a temporary
	const char *eventDesc[MAX_TOTAL_EVENTS];
	int lpapiEvents[MAX_TOTAL_EVENTS];
	
    numPapiEvents = 0;
	
#define ADD_LOW_LEVEL_EVENT(e, edesc) \
		if(PAPI_query_event(e) == PAPI_OK){ \
			if(CmiMyPe()==0) printf("PAPI Info:> Event for %s added\n", edesc); \
			eventDesc[numPapiEvents] = edesc;	\
			lpapiEvents[numPapiEvents++] = e; \
		}
		
    // PAPI high level API does not require explicit library intialization
	eventDesc[numPapiEvents] ="cycles";
    lpapiEvents[numPapiEvents++] = PAPI_TOT_CYC;
	
	
    /*if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK) {
      if (CmiMyPe()== 0) printf("PAPI_FP_INS used\n");
	  eventDesc[numPapiEvents] = "floating point instructions";	
      lpapiEvents[numPapiEvents++] = PAPI_FP_INS;	  
    } else {
      if (CmiMyPe()== 0) printf("PAPI_TOT_INS used\n");
	  eventDesc[numPapiEvents] = "total instructions";	
      lpapiEvents[numPapiEvents++] = PAPI_TOT_INS;
    }
    */
	
	//ADD_LOW_LEVEL_EVENT(PAPI_FP_INS, "floating point instructions");
	ADD_LOW_LEVEL_EVENT(PAPI_TOT_INS, "total instructions");
	//ADD_LOW_LEVEL_EVENT(PAPI_L1_DCM, "L1 cache misses");
	ADD_LOW_LEVEL_EVENT(PAPI_L2_DCM, "L2 data cache misses");	
	//ADD_LOW_LEVEL_EVENT(PAPI_MEM_RCY, "idle cycles waiting for memory reads");	
	//ADD_LOW_LEVEL_EVENT(PAPI_TLB_DM, "Data TLB misses");
	
	if(numPapiEvents == 0){
		CmiAbort("No papi events are defined!\n");
	}
	
	if(numPapiEvents >= MAX_TOTAL_EVENTS){
		CmiAbort("Exceed the pre-defined max number of papi events allowed!\n");
	}
	
	papiEvents = new int[numPapiEvents];
	papiValues = new long_long[numPapiEvents];
	total_papi_counters = new CmiUInt8[numPapiEvents];
	papi_counters_desc = new char *[numPapiEvents];
	for(int i=0; i<numPapiEvents; i++){
		papiEvents[i] = lpapiEvents[i];
		total_papi_counters[i] = 0;
		CmiPrintf("%d: %s\n", i, eventDesc[i]);
		papi_counters_desc[i] = new char[strlen(eventDesc[i])+1];
		memcpy(papi_counters_desc[i], eventDesc[i], strlen(eventDesc[i]));
		papi_counters_desc[i][strlen(eventDesc[i])] = 0;
	}
	
/*
    if (PAPI_query_event(PAPI_L1_DCM) == PAPI_OK) {
      if (CmiMyPe()== 0) printf("PAPI_L1_DCM used\n");
      papiEvents[numPapiEvents++] = PAPI_L1_DCM;   // L1 cache miss
    }
    if (PAPI_query_event(PAPI_TLB_DM) == PAPI_OK) {
      if (CmiMyPe()== 0) printf("PAPI_TLB_DM used\n");
      papiEvents[numPapiEvents++] = PAPI_TLB_DM;   // data TLB misses
    }
    if (PAPI_query_event(PAPI_MEM_RCY) == PAPI_OK) {
      if (CmiMyPe()== 0) printf("PAPI_MEM_RCY used\n");
      papiEvents[numPapiEvents++] = PAPI_MEM_RCY;  // idle cycle waiting for reads
    }
*/

    int status = PAPI_start_counters(papiEvents, numPapiEvents);
    if (status != PAPI_OK) {
      CmiPrintf("PAPI_start_counters error code: %d\n", status);
      switch (status) {
      case PAPI_ENOEVNT:  CmiPrintf("PAPI Error> Hardware Event does not exist\n"); break;
      case PAPI_ECNFLCT:  CmiPrintf("PAPI Error> Hardware Event exists, but cannot be counted due to counter resource limitations\n"); break;
      }
      CmiAbort("Unable to start PAPI counters!\n");
    }
}

int read_counters(long_long *papiValues, int n) 
{
  // PAPI_read_counters resets the counter, hence it behaves like the perfctr
  // code for Origin2000
  int status;
  status = PAPI_read_counters(papiValues, n);
  if (status != PAPI_OK) {
    CmiPrintf("PAPI_read_counters error: %d\n", status);
    CmiAbort("Failed to read PAPI counters!\n");
  }
  
  /*
  total_ins += papiValues[0];
  total_fps += papiValues[1];
  total_l1_dcm += papiValues[2];
  total_mem_rcy += papiValues[3];
  */
  
  return 0;
}

inline void CountPapiEvents(){
  for(int i=0 ;i<numPapiEvents; i++)
	total_papi_counters[i] += papiValues[i];
}

inline double Count2Time(long_long *papiValues, int n) { 
  return papiValues[1]*cva(bgMach).fpfactor; 
}
#endif

/*****************************************************************************
     Handler Table, one per thread
****************************************************************************/

extern "C" void defaultBgHandler(char *null, void *uPtr)
{
  CmiAbort("BG> Invalid Handler called!\n");
}

HandlerTable::HandlerTable()
{
    handlerTableCount = 1;
    handlerTable = new BgHandlerInfo [MAX_HANDLERS];
    for (int i=0; i<MAX_HANDLERS; i++) {
      handlerTable[i].fnPtr = defaultBgHandler;
      handlerTable[i].userPtr = NULL;
    }
}

inline int HandlerTable::registerHandler(BgHandler h)
{
    ASSERT(!cva(simState).inEmulatorInit);
    /* leave 0 as blank, so it can report error luckily */
    int cur = handlerTableCount++;
    if (cur >= MAX_HANDLERS)
      CmiAbort("BG> HandlerID exceed the maximum.\n");
    handlerTable[cur].fnPtr = (BgHandlerEx)h;
    handlerTable[cur].userPtr = NULL;
    return cur;
}

inline int HandlerTable::registerHandlerEx(BgHandlerEx h, void *uPtr)
{
    ASSERT(!cva(simState).inEmulatorInit);
    /* leave 0 as blank, so it can report error luckily */
    int cur = handlerTableCount++;
    if (cur >= MAX_HANDLERS)
      CmiAbort("BG> HandlerID exceed the maximum.\n");
    handlerTable[cur].fnPtr = h;
    handlerTable[cur].userPtr = uPtr;
    return cur;
}

inline void HandlerTable::numberHandler(int idx, BgHandler h)
{
    ASSERT(!cva(simState).inEmulatorInit);
    if (idx >= handlerTableCount || idx < 1)
      CmiAbort("BG> HandlerID exceed the maximum!\n");
    handlerTable[idx].fnPtr = (BgHandlerEx)h;
    handlerTable[idx].userPtr = NULL;
}

inline void HandlerTable::numberHandlerEx(int idx, BgHandlerEx h, void *uPtr)
{
    ASSERT(!cva(simState).inEmulatorInit);
    if (idx >= handlerTableCount || idx < 1)
      CmiAbort("BG> HandlerID exceed the maximum!\n");
    handlerTable[idx].fnPtr = h;
    handlerTable[idx].userPtr = uPtr;
}

inline BgHandlerInfo * HandlerTable::getHandle(int handler)
{
#if 0
    if (handler >= handlerTableCount) {
      CmiPrintf("[%d] handler: %d handlerTableCount:%d. \n", tMYNODEID, handler, handlerTableCount);
      CmiAbort("Invalid handler!");
    }
#endif
    if (handler >= handlerTableCount || handler<0) return NULL;
    return &handlerTable[handler];
}

/*****************************************************************************
      low level API
*****************************************************************************/

int BgRegisterHandler(BgHandler h)
{
  ASSERT(!cva(simState).inEmulatorInit);
#if CMK_BIGSIM_NODE
  return tMYNODE->handlerTable.registerHandler(h);
#else
  if (tTHREADTYPE == COMM_THREAD) {
    return tMYNODE->handlerTable.registerHandler(h);
  }
  else {
    return tHANDLETAB.registerHandler(h);
  }
#endif
}

int BgRegisterHandlerEx(BgHandlerEx h, void *uPtr)
{
  ASSERT(!cva(simState).inEmulatorInit);
#if CMK_BIGSIM_NODE
  return tMYNODE->handlerTable.registerHandlerEx(h, uPtr);
#else
  if (tTHREADTYPE == COMM_THREAD) {
    return tMYNODE->handlerTable.registerHandlerEx(h, uPtr);
  }
  else {
    return tHANDLETAB.registerHandlerEx(h, uPtr);
  }
#endif
}


void BgNumberHandler(int idx, BgHandler h)
{
  ASSERT(!cva(simState).inEmulatorInit);
#if CMK_BIGSIM_NODE
  tMYNODE->handlerTable.numberHandler(idx,h);
#else
  if (tTHREADTYPE == COMM_THREAD) {
    tMYNODE->handlerTable.numberHandler(idx, h);
  }
  else {
    tHANDLETAB.numberHandler(idx, h);
  }
#endif
}

void BgNumberHandlerEx(int idx, BgHandlerEx h, void *uPtr)
{
  ASSERT(!cva(simState).inEmulatorInit);
#if CMK_BIGSIM_NODE
  tMYNODE->handlerTable.numberHandlerEx(idx,h,uPtr);
#else
  if (tTHREADTYPE == COMM_THREAD) {
    tMYNODE->handlerTable.numberHandlerEx(idx,h,uPtr);
  }
  else {
    tHANDLETAB.numberHandlerEx(idx,h,uPtr);
  }
#endif
}

/*****************************************************************************
      BG Timing Functions
*****************************************************************************/

void resetVTime()
{
  /* reset start time */
  int timingMethod = cva(bgMach).timingMethod;
  if (timingMethod == BG_WALLTIME) {
    double ct = BG_TIMER();
    if (tTIMERON) CmiAssert(ct >= tSTARTTIME);
    tSTARTTIME = ct;
  }
  else if (timingMethod == BG_ELAPSE)
    tSTARTTIME = tCURRTIME;
#ifdef CMK_ORIGIN2000
  else if (timingMethod == BG_COUNTER) {
    if (start_counters(0, 21) <0) {
      perror("start_counters");;
    }
  }
#elif CMK_HAS_COUNTER_PAPI
  else if (timingMethod == BG_COUNTER) {
    // do a fake read to reset the counters. It would be more efficient
    // to use the low level API, but that would be a lot more code to
    // write for now.
    if (read_counters(papiValues, numPapiEvents) < 0) perror("read_counters");
  }
#endif
}

void startVTimer()
{
  CmiAssert(tTIMERON == 0);
  resetVTime();
  tTIMERON = 1;
}

// should be used only when BG_WALLTIME
static inline void advanceTime(double inc)
{
  if (BG_ABS(inc) < 1e-10) inc = 0.0;    // ignore floating point errors
  if (inc < 0.0) return;
  CmiAssert(inc>=0.0);
  inc *= cva(bgMach).cpufactor;
  tCURRTIME += inc;
  CmiAssert(tTIMERON==1);
}

void stopVTimer()
{
  int k;
#if 0
  if (tTIMERON != 1) {
    CmiAbort("stopVTimer called without startVTimer!\n");
  }
  CmiAssert(tTIMERON == 1);
#else
  if (tTIMERON == 0) return;         // already stopped
#endif
  const int timingMethod = cva(bgMach).timingMethod;
  if (timingMethod == BG_WALLTIME) {
    const double tp = BG_TIMER();
    double inc = tp-tSTARTTIME;
    advanceTime(inc-cva(bgMach).timercost);
//    tSTARTTIME = BG_TIMER();	// skip the above time
  }
  else if (timingMethod == BG_ELAPSE) {
    // if no bgelapse called, assume it takes 1us
    if (tCURRTIME-tSTARTTIME < 1E-9) {
//      tCURRTIME += 1e-6;
    }
  }
  else if (timingMethod == BG_COUNTER)  {
#if CMK_ORIGIN2000
    long long c0, c1;
    if (read_counters(0, &c0, 21, &c1) < 0) perror("read_counters");
    tCURRTIME += Count2Time(c1);
#elif CMK_HAS_COUNTER_PAPI
    if (read_counters(papiValues, numPapiEvents) < 0) perror("read_counters");
    CountPapiEvents();
    tCURRTIME += Count2Time(papiValues, numPapiEvents);
#endif
  }
  tTIMERON = 0;
}

double BgGetTime()
{
#if 1
  const int timingMethod = cva(bgMach).timingMethod;
  if (timingMethod == BG_WALLTIME) {
    /* accumulate time since last starttime, and reset starttime */
    if (tTIMERON) {
      const double tp2= BG_TIMER();
      double &startTime = tSTARTTIME;
      double inc = tp2 - startTime;
      advanceTime(inc-cva(bgMach).timercost);
      startTime = BG_TIMER();
    }
    return tCURRTIME;
  }
  else if (timingMethod == BG_ELAPSE) {
    return tCURRTIME;
  }
  else if (timingMethod == BG_COUNTER) {
    if (tTIMERON) {
#if CMK_ORIGIN2000
      long long c0, c1;
      if (read_counters(0, &c0, 21, &c1) <0) perror("read_counters");;
      tCURRTIME += Count2Time(c1);
      if (start_counters(0, 21)<0) perror("start_counters");;
#elif CMK_HAS_COUNTER_PAPI
    if (read_counters(papiValues, numPapiEvents) < 0) perror("read_counters");
    tCURRTIME += Count2Time(papiValues, numPapiEvents);
#endif
    }
    return tCURRTIME;
  }
  else 
    CmiAbort("Unknown Timing Method.");
  return -1;
#else
  /* sometime I am interested in real wall time */
  tCURRTIME = CmiWallTimer();
  return tCURRTIME;
#endif
}

// moved to blue_logs.C
double BgGetCurTime()
{
  ASSERT(tTHREADTYPE == WORK_THREAD);
  return tCURRTIME;
}

extern "C" 
void BgElapse(double t)
{
//  ASSERT(tTHREADTYPE == WORK_THREAD);
  if (cva(bgMach).timingMethod == BG_ELAPSE)
    tCURRTIME += t;
}

// advance virtual timer no matter what scheme is used
extern "C" 
void BgAdvance(double t)
{
//  ASSERT(tTHREADTYPE == WORK_THREAD);
  tCURRTIME += t;
}

/* BG API Func
 * called by a communication thread to test if poll data 
 * in the node's INBUFFER for its own queue 
 */
char * getFullBuffer()
{
  /* I must be a communication thread */
  if (tTHREADTYPE != COMM_THREAD) 
    CmiAbort("GetFullBuffer called by a non-communication thread!\n");

  return tMYNODE->getFullBuffer();
}

/**  BG API Func
 * called by a Converse handler or sendPacket()
 * add message msgPtr to a bluegene node's inbuffer queue 
 */
extern "C"
void addBgNodeInbuffer(char *msgPtr, int lnodeID)
{
#if CMK_ERROR_CHECKING
  if (lnodeID >= cva(numNodes)) CmiAbort("NodeID is out of range!");
#endif
  nodeInfo &nInfo = cva(nodeinfo)[lnodeID];

  //printf("Adding a msg %p to local node %d and its thread %d\n", msgPtr, lnodeID, CmiBgMsgThreadID(msgPtr));	
	
  nInfo.addBgNodeInbuffer(msgPtr);
}

/** BG API Func 
 *  called by a comm thread
 *  add a message to a thread's affinity queue in same node 
 */
void addBgThreadMessage(char *msgPtr, int threadID)
{
#if CMK_ERROR_CHECKING
  if (!cva(bgMach).isWorkThread(threadID)) CmiAbort("ThreadID is out of range!");
#endif
  workThreadInfo *tInfo = (workThreadInfo *)tMYNODE->threadinfo[threadID];
  tInfo->addAffMessage(msgPtr);
}

/** BG API Func 
 *  called by a comm thread, add a message to a node's non-affinity queue 
 */
void addBgNodeMessage(char *msgPtr)
{
  tMYNODE->addBgNodeMessage(msgPtr);
}

void BgEnqueue(char *msg)
{
#if 0
  ASSERT(tTHREADTYPE == WORK_THREAD);
  workThreadInfo *tinfo = (workThreadInfo *)cta(threadinfo);
  tinfo->addAffMessage(msg);
#else
  nodeInfo *myNode = cta(threadinfo)->myNode;
  addBgNodeInbuffer(msg, myNode->id);
#endif
}

/** BG API Func 
 *  check if inBuffer on this node has msg available
 */
int checkReady()
{
  if (tTHREADTYPE != COMM_THREAD)
    CmiAbort("checkReady called by a non-communication thread!\n");
  return !tINBUFFER.isEmpty();
}

/* handler to process the msg */
void msgHandlerFunc(char *msg)
{
  /* bgmsg is CmiMsgHeaderSizeBytes offset of original message pointer */
  int gnodeID = CmiBgMsgNodeID(msg);
  if (gnodeID >= 0) {
#if CMK_ERROR_CHECKING
    if (nodeInfo::Global2PE(gnodeID) != CmiMyPe())
      CmiAbort("msgHandlerFunc received wrong message!");
#endif
    int lnodeID = nodeInfo::Global2Local(gnodeID);
    if (cva(bgMach).inReplayMode()) {
      int x, y, z;
      int node;
      if (cva(bgMach).replay != -1) {
        node = cva(bgMach).replay;
        node = node / cva(bgMach).numWth;
      }
      if (cva(bgMach).replaynode != -1) {
        node = cva(bgMach).replaynode;
      }
      BgGetXYZ(node, &x, &y, &z);
      if (nodeInfo::XYZ2Local(x,y,z) != lnodeID) return;
      else lnodeID = 0;
    }
    addBgNodeInbuffer(msg, lnodeID);
  }
  else {
    CmiAbort("Invalid message!");
  }
}

/* Converse handler for node level broadcast message */
void nodeBCastMsgHandlerFunc(char *msg)
{
  /* bgmsg is CmiMsgHeaderSizeBytes offset of original message pointer */
  int gnodeID = CmiBgMsgNodeID(msg);
  CmiInt2 threadID = CmiBgMsgThreadID(msg);
  int lnodeID;

  if (gnodeID < -1) {
    gnodeID = - (gnodeID+100);
    if (cva(bgMach).replaynode != -1) {
      if (gnodeID == cva(bgMach).replaynode)
          lnodeID = 0;
      else
          lnodeID = -1;
    }
    else if (nodeInfo::Global2PE(gnodeID) == CmiMyPe())
      lnodeID = nodeInfo::Global2Local(gnodeID);
    else
      lnodeID = -1;
  }
  else {
    ASSERT(gnodeID == -1);
    lnodeID = gnodeID;
  }
  // broadcast except lnodeID:threadId
  int len = CmiBgMsgLength(msg);
  int count = 0;
  for (int i=0; i<cva(numNodes); i++)
  {
    if (i==lnodeID) continue;
    char *dupmsg;
    if (count == 0) dupmsg = msg;
    else dupmsg = CmiCopyMsg(msg, len);
    DEBUGF(("addBgNodeInbuffer to %d\n", i));
    CmiBgMsgNodeID(dupmsg) = nodeInfo::Local2Global(i);		// updated
    addBgNodeInbuffer(dupmsg, i);
    count ++;
  }
  if (count == 0) CmiFree(msg);
}

// clone a msg, only has a valid header, plus a pointer to the real msg
char *BgCloneMsg(char *msg)
{
  int size = CmiBlueGeneMsgHeaderSizeBytes + sizeof(char *);
  char *dupmsg = (char *)CmiAlloc(size);
  memcpy(dupmsg, msg, CmiBlueGeneMsgHeaderSizeBytes);
  *(char **)(dupmsg + CmiBlueGeneMsgHeaderSizeBytes) = msg;
  CmiBgMsgRefCount(msg) ++;
  CmiBgMsgFlag(dupmsg) = BG_CLONE;
  return dupmsg;
}

// expand the cloned msg to the full size msg
char *BgExpandMsg(char *msg)
{
  char *origmsg = *(char **)(msg + CmiBlueGeneMsgHeaderSizeBytes);
  int size = CmiBgMsgLength(origmsg);
  char *dupmsg = (char *)CmiAlloc(size);
  memcpy(dupmsg, msg, CmiBlueGeneMsgHeaderSizeBytes);
  memcpy(dupmsg+CmiBlueGeneMsgHeaderSizeBytes, origmsg+CmiBlueGeneMsgHeaderSizeBytes, size-CmiBlueGeneMsgHeaderSizeBytes);
  CmiFree(msg);
  CmiBgMsgRefCount(origmsg) --;
  if (CmiBgMsgRefCount(origmsg) == 0) CmiFree(origmsg);
  return dupmsg;
}

/* Converse handler for thread level broadcast message */
void threadBCastMsgHandlerFunc(char *msg)
{
  /* bgmsg is CmiMsgHeaderSizeBytes offset of original message pointer */
  int gnodeID = CmiBgMsgNodeID(msg);
  CmiInt2 threadID = CmiBgMsgThreadID(msg);
  if (cva(bgMach).replay != -1) {
    if (gnodeID < -1) {
      gnodeID = - (gnodeID+100);
      if (gnodeID == cva(bgMach).replay/cva(bgMach).numWth && threadID == cva(bgMach).replay%cva(bgMach).numWth)
        return;
    }
    CmiBgMsgThreadID(msg) = 0;
    DEBUGF(("[%d] addBgNodeInbuffer to %d tid:%d\n", CmiMyPe(), i, j));
    addBgNodeInbuffer(msg, 0);
    return;
  }
  int lnodeID;
  if (gnodeID < -1) {
      gnodeID = - (gnodeID+100);
      if (cva(bgMach).replaynode != -1) {
        if (gnodeID == cva(bgMach).replaynode)
          lnodeID = 0;
        else
          lnodeID = -1;
      }
      else if (nodeInfo::Global2PE(gnodeID) == CmiMyPe())
	lnodeID = nodeInfo::Global2Local(gnodeID);
      else
	lnodeID = -1;
      CmiAssert(threadID != ANYTHREAD);
  }
  else {
    ASSERT(gnodeID == -1);
    lnodeID = gnodeID;
  }
  // broadcast except nodeID:threadId
  int len = CmiBgMsgLength(msg);
  // optimization needed if the message size is big
  // making duplications can easily run out of memory
  int bigOpt = (len > 4096);
  for (int i=0; i<cva(numNodes); i++)
  {
      for (int j=0; j<cva(bgMach).numWth; j++) {
        if (i==lnodeID && j==threadID) continue;
        // for big message, clone a message token instead of a real msg
        char *dupmsg = bigOpt? BgCloneMsg(msg) : CmiCopyMsg(msg, len);
        CmiBgMsgNodeID(dupmsg) = nodeInfo::Local2Global(i);
        CmiBgMsgThreadID(dupmsg) = j;
        DEBUGF(("[%d] addBgNodeInbuffer to %d tid:%d\n", CmiMyPe(), i, j));
        addBgNodeInbuffer(dupmsg, i);
      }
  }
  // for big message, will free after all tokens are done
  if (!bigOpt) CmiFree(msg);
}

/**
 *		BG Messaging Functions
 */

static inline double MSGTIME(int ox, int oy, int oz, int nx, int ny, int nz, int bytes)
{
  return cva(bgMach).network->latency(ox, oy, oz, nx, ny, nz, bytes);
}

/**
 *   a simple message streaming on demand and special purpose
 *   user call  BgStartStreaming() and BgEndStreaming()
 *   each worker thread call one send, and all sends are sent
 *   via multiplesend at the end
 */

static int bg_streaming = 0;

class BgStreaming {
public:
  char **streamingMsgs;
  int  *streamingMsgSizes;
  int count;
  int totalWorker;
  int pe;
public:
  BgStreaming() {
    streamingMsgs = NULL;
    streamingMsgSizes = NULL;
    count = 0;
    totalWorker = 0;
    pe = -1;
  }
  ~BgStreaming() {
    if (streamingMsgs) {
      delete [] streamingMsgs;
      delete [] streamingMsgSizes;
    }
  }
  void init(int nNodes) {
    totalWorker = nNodes * BgGetNumWorkThread();
    streamingMsgs = new char *[totalWorker];
    streamingMsgSizes = new int [totalWorker];
  }
  void depositMsg(int p, int size, char *m) {
    streamingMsgs[count] = m;
    streamingMsgSizes[count] = size;
    count ++;
    if (pe == -1) pe = p;
    else CmiAssert(pe == p);
    if (count == totalWorker) {
      // CkPrintf("streaming send\n");
      CmiMultipleSend(pe, count, streamingMsgSizes, streamingMsgs);
      for (int i=0; i<count; i++) CmiFree(streamingMsgs[i]);
      pe = -1;
      count = 0;
    }
  }
};

BgStreaming bgstreaming;

void BgStartStreaming()
{
  bg_streaming = 1;
}

void BgEndStreaming()
{
  bg_streaming = 0;
}

void CmiSendPacketWrapper(int pe, int msgSize,char *msg, int streaming)
{
  if (streaming && pe != CmiMyPe())
    bgstreaming.depositMsg(pe, msgSize, msg);
  else
    CmiSyncSendAndFree(pe, msgSize, msg);
}


void CmiSendPacket(int x, int y, int z, int msgSize,char *msg)
{
//  CmiSyncSendAndFree(nodeInfo::XYZ2RealPE(x,y,z),msgSize,(char *)msg);
#if !DELAY_SEND
  const int pe = nodeInfo::XYZ2RealPE(x,y,z);
  CmiSendPacketWrapper(pe, msgSize, msg, bg_streaming);
#else
  if (!correctTimeLog) {
    const int pe = nodeInfo::XYZ2RealPE(x,y,z);
    CmiSendPacketWrapper(pe, msgSize, msg, bg_streaming);
  }
  // else messages are kept in the log (MsgEntry), and only will be sent
  // after timing correction has done on that log.
  // TODO: streaming has no effect if time correction is on.
#endif
}

/* send will copy data to msg buffer */
/* user data is not free'd in this routine, user can reuse the data ! */
void sendPacket_(nodeInfo *myNode, int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char* sendmsg, int local)
{
  //CmiPrintStackTrace(0);
  double sendT = BgGetTime();

  double latency;
  CmiSetHandler(sendmsg, cva(simState).msgHandler);
  CmiBgMsgNodeID(sendmsg) = nodeInfo::XYZ2Global(x,y,z);
  CmiBgMsgThreadID(sendmsg) = threadID;
  CmiBgMsgHandle(sendmsg) = handlerID;
  CmiBgMsgType(sendmsg) = type;
  CmiBgMsgLength(sendmsg) = numbytes;
  CmiBgMsgFlag(sendmsg) = 0;
  CmiBgMsgRefCount(sendmsg) = 0;
  if (local) {
    if (correctTimeLog) BgAdvance(CHARM_OVERHEAD);
    latency = 0.0;
  }
  else {
    if (correctTimeLog) BgAdvance(cva(bgMach).network->alphacost());
    latency = MSGTIME(myNode->x, myNode->y, myNode->z, x,y,z, numbytes);
    CmiAssert(latency >= 0);
  }
  CmiBgMsgRecvTime(sendmsg) = latency + sendT;
  
  // timing
  BG_ADDMSG(sendmsg, CmiBgMsgNodeID(sendmsg), threadID, sendT, local, 1);

  //static int addCnt=1; //for debugging only
  //DEBUGM(4, ("N[%d] add a msg (handler=%d | cnt=%d | len=%d | type=%d | node id:%d\n", BgMyNode(), handlerID, addCnt, numbytes, type, CmiBgMsgNodeID(sendmsg)));  
  //addCnt++;

  if (local){
      /* Here local refers to the fact that msg is sent to the processor itself
       * therefore, we just add this msg to the thread itself
       */
      //addBgThreadMessage(sendmsg,threadID);
      addBgNodeInbuffer(sendmsg, myNode->id);
  }    
  else
    CmiSendPacket(x, y, z, numbytes, sendmsg);

  // bypassing send time
  resetVTime();
}

/* broadcast will copy data to msg buffer */
static inline void nodeBroadcastPacketExcept_(int node, CmiInt2 threadID, int handlerID, WorkType type, int numbytes, char* sendmsg)
{
  double sendT = BgGetTime();

  nodeInfo *myNode = cta(threadinfo)->myNode;
  CmiSetHandler(sendmsg, cva(simState).nBcastMsgHandler);
  if (node >= 0)
    CmiBgMsgNodeID(sendmsg) = -node-100;
  else
    CmiBgMsgNodeID(sendmsg) = node;
  CmiBgMsgThreadID(sendmsg) = threadID;	
  CmiBgMsgHandle(sendmsg) = handlerID;	
  CmiBgMsgType(sendmsg) = type;	
  CmiBgMsgLength(sendmsg) = numbytes;
  CmiBgMsgFlag(sendmsg) = 0;
  CmiBgMsgRefCount(sendmsg) = 0;
  /* FIXME */
  CmiBgMsgRecvTime(sendmsg) = MSGTIME(myNode->x, myNode->y, myNode->z, 0,0,0, numbytes) + sendT;

  // timing
  // FIXME
  BG_ADDMSG(sendmsg, CmiBgMsgNodeID(sendmsg), threadID, sendT, 0, 1);

  DEBUGF(("[%d]CmiSyncBroadcastAllAndFree node: %d\n", BgMyNode(), node));
#if DELAY_SEND
  if (!correctTimeLog)
#endif
  CmiSyncBroadcastAllAndFree(numbytes,sendmsg);

  resetVTime();
}

/* broadcast will copy data to msg buffer */
static inline void threadBroadcastPacketExcept_(int node, CmiInt2 threadID, int handlerID, WorkType type, int numbytes, char* sendmsg)
{
  CmiSetHandler(sendmsg, cva(simState).tBcastMsgHandler);	
  if (node >= 0)
    CmiBgMsgNodeID(sendmsg) = -node-100;
  else
    CmiBgMsgNodeID(sendmsg) = node;
  CmiBgMsgThreadID(sendmsg) = threadID;	
  CmiBgMsgHandle(sendmsg) = handlerID;	
  CmiBgMsgType(sendmsg) = type;	
  CmiBgMsgLength(sendmsg) = numbytes;
  CmiBgMsgFlag(sendmsg) = 0;
  CmiBgMsgRefCount(sendmsg) = 0;
  /* FIXME */
  if (correctTimeLog) BgAdvance(cva(bgMach).network->alphacost());
  double sendT = BgGetTime();
  CmiBgMsgRecvTime(sendmsg) = sendT;	

  // timing
#if 0
  if (node == BG_BROADCASTALL) {
    for (int i=0; i<_bgSize; i++) {
      for (int j=0; j<cva(numWth); j++) {
        BG_ADDMSG(sendmsg, node);
      }
    }
  }
  else {
    CmiAssert(node >= 0);
    BG_ADDMSG(sendmsg, (node+100));
  }
#else
  // FIXME
  BG_ADDMSG(sendmsg, CmiBgMsgNodeID(sendmsg), threadID, sendT, 0, 1);
#endif

  DEBUGF(("[%d]CmiSyncBroadcastAllAndFree node: %d tid:%d recvT:%f\n", BgMyNode(), node, threadID, CmiBgMsgRecvTime(sendmsg)));
#if DELAY_SEND
  if (!correctTimeLog)
#endif
  CmiSyncBroadcastAllAndFree(numbytes,sendmsg);

  resetVTime();
}


/* sendPacket to route */
/* this function can be called by any thread */
void BgSendNonLocalPacket(nodeInfo *myNode, int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  if (cva(bgMach).inReplayMode()) return;     // replay mode, no outgoing msg

#if CMK_ERROR_CHECKING
  if (x<0 || y<0 || z<0 || x>=cva(bgMach).x || y>=cva(bgMach).y || z>=cva(bgMach).z) {
    CmiPrintf("Trying to send packet to a nonexisting node: (%d %d %d)!\n", x,y,z);
    CmiAbort("Abort!\n");
  }
#endif

  sendPacket_(myNode, x, y, z, threadID, handlerID, type, numbytes, data, 0);
}

static void _BgSendLocalPacket(nodeInfo *myNode, int threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  if (cva(bgMach).replay!=-1) { // replay mode
    int t = cva(bgMach).replay%BgGetNumWorkThread();
    if (t == threadID) threadID = 0;
    else return;
  }

  sendPacket_(myNode, myNode->x, myNode->y, myNode->z, threadID, handlerID, type, numbytes, data, 1);
}

void BgSendLocalPacket(int threadID, int handlerID, WorkType type,
                       int numbytes, char* data)
{
  nodeInfo *myNode = cta(threadinfo)->myNode;

  if (cva(bgMach).replay!=-1) {     // replay mode
    threadID = 0;
    CmiAssert(threadID != -1);
  }

  _BgSendLocalPacket(myNode, threadID, handlerID, type, numbytes, data);
}

/* wrapper of the previous two functions */
void BgSendPacket(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  nodeInfo *myNode = cta(threadinfo)->myNode;
  if (myNode->x == x && myNode->y == y && myNode->z == z)
    _BgSendLocalPacket(myNode,threadID, handlerID, type, numbytes, data);
  else
    BgSendNonLocalPacket(myNode,x,y,z,threadID,handlerID, type, numbytes, data);
}

void BgBroadcastPacketExcept(int node, CmiInt2 threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  nodeBroadcastPacketExcept_(node, threadID, handlerID, type, numbytes, data);
}

void BgBroadcastAllPacket(int handlerID, WorkType type, int numbytes, char * data)
{
  nodeBroadcastPacketExcept_(BG_BROADCASTALL, ANYTHREAD, handlerID, type, numbytes, data);
}

void BgThreadBroadcastPacketExcept(int node, CmiInt2 threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  if (cva(bgMach).replay!=-1) return;    // replay mode
  else if (cva(bgMach).replaynode!=-1) {    // replay mode
    //if (node!=-1 && node == cva(bgMach).replaynode/cva(bgMach).numWth)
    //  return;
  }
  threadBroadcastPacketExcept_(node, threadID, handlerID, type, numbytes, data);
}

void BgThreadBroadcastAllPacket(int handlerID, WorkType type, int numbytes, char * data)
{
  if (cva(bgMach).replay!=-1) {      // replay mode, send only to itself
    int t = cva(bgMach).replay%BgGetNumWorkThread();
    BgSendLocalPacket(t, handlerID, type, numbytes, data);
    return;
  }
  threadBroadcastPacketExcept_(BG_BROADCASTALL, ANYTHREAD, handlerID, type, numbytes, data);
}

/**
 send a msg to a list of processors (processors represented in global seq #
*/
void BgSyncListSend(int npes, int *pes, int handlerID, WorkType type, int numbytes, char *msg)
{
  nodeInfo *myNode = cta(threadinfo)->myNode;

  CmiSetHandler(msg, cva(simState).msgHandler);
  CmiBgMsgHandle(msg) = handlerID;
  CmiBgMsgType(msg) = type;
  CmiBgMsgLength(msg) = numbytes;
  CmiBgMsgFlag(msg) = 0;
  CmiBgMsgRefCount(msg) = 0;

  if (correctTimeLog) BgAdvance(cva(bgMach).network->alphacost());

  double now = BgGetTime();

  // send one by one
  for (int i=0; i<npes; i++)
  {
    int local = 0;
    int x,y,z,t;
    int pe = pes[i];
    int node;
#if CMK_BIGSIM_NODE
    CmiAbort("Not implemented yet!");
#else
    t = pe%BgGetNumWorkThread();
    node = pe/BgGetNumWorkThread();
    BgGetXYZ(node, &x, &y, &z);
#endif

    char *sendmsg = CmiCopyMsg(msg, numbytes);
    CmiBgMsgNodeID(sendmsg) = nodeInfo::XYZ2Global(x,y,z);
    CmiBgMsgThreadID(sendmsg) = t;
    double latency = MSGTIME(myNode->x, myNode->y, myNode->z, x,y,z, numbytes);
    CmiAssert(latency >= 0);
    CmiBgMsgRecvTime(sendmsg) = latency + now;

    if (myNode->x == x && myNode->y == y && myNode->z == z) local = 1;

    // timing and make sure all msgID are the same
    if (i!=0) CpvAccess(msgCounter) --;
    BG_ADDMSG(sendmsg, CmiBgMsgNodeID(sendmsg), t, now, local, i==0?npes:-1);

#if 0
    BgSendPacket(x, y, z, t, handlerID, type, numbytes, sendmsg);
#else
    if (myNode->x == x && myNode->y == y && myNode->z == z)
      addBgNodeInbuffer(sendmsg, myNode->id);
    else {
      if (cva(bgMach).inReplayMode()) continue;  // replay mode, no outgoing msg
      CmiSendPacket(x, y, z, numbytes, sendmsg);
    }
#endif
  }

  CmiFree(msg);

  resetVTime();
}

/*****************************************************************************
      BG node level API - utilities
*****************************************************************************/

/* must be called in a communication or worker thread */
void BgGetMyXYZ(int *x, int *y, int *z)
{
  ASSERT(!cva(simState).inEmulatorInit);
  *x = tMYX; *y = tMYY; *z = tMYZ;
}

void BgGetXYZ(int seq, int *x, int *y, int *z)
{
  nodeInfo::Global2XYZ(seq, x, y, z);
}

void BgGetSize(int *sx, int *sy, int *sz)
{
  cva(bgMach).getSize(sx, sy, sz);
}

int BgTraceProjectionOn(int pe)
{
  return cva(bgMach).traceProjections(pe);
}

/* return the total number of Blue gene nodes */
int BgNumNodes()
{
  return _bgSize;
}

void BgSetNumNodes(int x)
{
  _bgSize = x;
}

/* can only called in emulatorinit */
void BgSetSize(int sx, int sy, int sz)
{
  ASSERT(cva(simState).inEmulatorInit);
  cva(bgMach).setSize(sx, sy, sz);
}

/* return number of bg nodes on this emulator node */
int BgNodeSize()
{
  ASSERT(!cva(simState).inEmulatorInit);
  return cva(numNodes);
}

/* return the bg node ID (local array index) */
int BgMyRank()
{
#if CMK_ERROR_CHECKING
  if (tMYNODE == NULL) CmiAbort("Calling BgMyRank in the main thread!");
#endif
  ASSERT(!cva(simState).inEmulatorInit);
  return tMYNODEID;
}

/* return my serialed blue gene node number */
int BgMyNode()
{
#if CMK_ERROR_CHECKING
  if (tMYNODE == NULL) CmiAbort("Calling BgMyNode in the main thread!");
#endif
  return nodeInfo::XYZ2Global(tMYX, tMYY, tMYZ);
}

/* return a real processor number from a bg node */
int BgNodeToRealPE(int node)
{
  return nodeInfo::Global2PE(node);
}

// thread ID on a BG node
int BgGetThreadID()
{
  ASSERT(tTHREADTYPE == WORK_THREAD || tTHREADTYPE == COMM_THREAD);
//  if (cva(bgMach).numWth == 1) return 0;   // accessing ctv is expensive
  return tMYID;
}

void BgSetThreadID(int x)
{
  tMYID = x;
}

int BgGetGlobalThreadID()
{
  ASSERT(tTHREADTYPE == WORK_THREAD || tTHREADTYPE == COMM_THREAD);
  return nodeInfo::Local2Global(tMYNODEID)*(cva(bgMach).numTh())+tMYID;
  //return tMYGLOBALID;
}

int BgGetGlobalWorkerThreadID()
{
  ASSERT(tTHREADTYPE == WORK_THREAD);
//  return nodeInfo::Local2Global(tMYNODEID)*cva(bgMach).numWth+tMYID;
  return tMYGLOBALID;
}

void BgSetGlobalWorkerThreadID(int pe)
{
  ASSERT(tTHREADTYPE == WORK_THREAD);
//  return nodeInfo::Local2Global(tMYNODEID)*cva(bgMach).numWth+tMYID;
  tMYGLOBALID = pe;
}

char *BgGetNodeData()
{
  return tUSERDATA;
}

void BgSetNodeData(char *data)
{
  ASSERT(!cva(simState).inEmulatorInit);
  tUSERDATA = data;
}

int BgGetNumWorkThread()
{
  return cva(bgMach).numWth;
}

void BgSetNumWorkThread(int num)
{
  if (!cva(bgMach).inReplayMode()) ASSERT(cva(simState).inEmulatorInit);
  cva(bgMach).numWth = num;
}

int BgGetNumCommThread()
{
  return cva(bgMach).numCth;
}

void BgSetNumCommThread(int num)
{
  ASSERT(cva(simState).inEmulatorInit);
  cva(bgMach).numCth = num;
}

/*****************************************************************************
      Communication and Worker threads
*****************************************************************************/

BgStartHandler  workStartFunc = NULL;

void BgSetWorkerThreadStart(BgStartHandler f)
{
  workStartFunc = f;
}

extern "C" void CthResumeNormalThread(CthThreadToken* token);

// kernel function for processing a bluegene message
void BgProcessMessageDefault(threadInfo *tinfo, char *msg)
{
  DEBUGM(5, ("=====Begin of BgProcessing a msg on node[%d]=====\n", BgMyNode()));
  int handler = CmiBgMsgHandle(msg);
  //CmiPrintf("[%d] call handler %d\n", BgMyNode(), handler);
  CmiAssert(handler < 1000);

  BgHandlerInfo *handInfo;
#if  CMK_BIGSIM_NODE
  HandlerTable hdlTbl = tMYNODE->handlerTable;
  handInfo = hdlTbl.getHandle(handler);
#else
  HandlerTable hdlTbl = tHANDLETAB;
  handInfo = hdlTbl.getHandle(handler);
  if (handInfo == NULL) handInfo = tMYNODE->handlerTable.getHandle(handler);
#endif

  if (handInfo == NULL) {
    CmiPrintf("[%d] invalid handler: %d. \n", tMYNODEID, handler);
    CmiAbort("BgProcessMessage Failed!");
  }
  BgHandlerEx entryFunc = handInfo->fnPtr;

  if (programExit == 2) return;    // program exit already

  CmiSetHandler(msg, CmiBgMsgHandle(msg));

  // optimization for broadcast messages:
  // if the msg is a broadcast token msg, expand it to a real msg
  if (CmiBgMsgFlag(msg) == BG_CLONE) {
    msg = BgExpandMsg(msg);
  }

  if (tinfo->watcher) tinfo->watcher->record(msg);

  // don't count thread overhead and timing overhead
  startVTimer();

  DEBUGM(5, ("Executing function %p\n", entryFunc));    

  entryFunc(msg, handInfo->userPtr);

  stopVTimer();

  DEBUGM(5, ("=====End of BgProcessing a msg on node[%d]=====\n\n", BgMyNode()));
}

void  (*BgProcessMessage)(threadInfo *t, char *msg) = BgProcessMessageDefault;

void scheduleWorkerThread(char *msg)
{
  CthThread tid = (CthThread)msg;
//CmiPrintf("scheduleWorkerThread %p\n", tid);
  CthAwaken(tid);
}

// thread entry
// for both comm and work thread, virtual function
void run_thread(threadInfo *tinfo)
{
  /* set the thread-private threadinfo */
  cta(threadinfo) = tinfo;
  tinfo->run();
}

/* should be done only once per bg node */
void BgNodeInitialize(nodeInfo *ninfo)
{
  CthThread t;
  int i;

  /* this is here because I will put a message to node inbuffer */
  tCURRTIME = 0.0;
  tSTARTTIME = CmiWallTimer();

  /* creat work threads */
  for (i=0; i< cva(bgMach).numWth; i++)
  {
    threadInfo *tinfo = ninfo->threadinfo[i];
    t = CthCreate((CthVoidFn)run_thread, tinfo, cva(bgMach).stacksize);
    if (t == NULL) CmiAbort("BG> Failed to create worker thread. \n");
    tinfo->setThread(t);
    /* put to thread table */
    tTHREADTABLE[tinfo->id] = t;
#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
    //initial scheduling points for workthreads
    if(bgUseOutOfCore) schedWorkThds->push((workThreadInfo *)tinfo);
#endif
    CthAwaken(t);
  }

  /* creat communication thread */
  for (i=0; i< cva(bgMach).numCth; i++)
  {
    threadInfo *tinfo = ninfo->threadinfo[i+cva(bgMach).numWth];
    t = CthCreate((CthVoidFn)run_thread, tinfo, cva(bgMach).stacksize);
    if (t == NULL) CmiAbort("BG> Failed to create communication thread. \n");
    tinfo->setThread(t);
    /* put to thread table */
    tTHREADTABLE[tinfo->id] = t;
    CthAwaken(t);
  }

}

static void beginExitHandlerFunc(void *msg);
static void writeToDisk();
static void sendCorrectionStats();

void callAllUserTracingFunction()
{
  if (userTracingFn == NULL) return;
  int origPe = -2;
  // close all tracing modules
  for (int j=0; j<cva(numNodes); j++)
    for (int i=0; i<cva(bgMach).numWth; i++) {
      int pe = nodeInfo::Local2Global(j)*cva(bgMach).numWth+i;
      int oldPe = CmiSwitchToPE(pe);
      if (cva(bgMach).replay != -1)
        if ( pe != cva(bgMach).replay ) continue;
      if (origPe == -2) origPe = oldPe;
      traceCharmClose();
      delete cva(nodeinfo)[j].threadinfo[i]->watcher;   // force dump watcher
      cva(nodeinfo)[j].threadinfo[i]->watcher = NULL;
      if (userTracingFn) userTracingFn();
    }
    if (origPe!=-2) CmiSwitchToPE(origPe);
}

static CmiHandler exitHandlerFunc(char *msg)
{
  // TODO: free memory before exit
  int i,j;

  programExit = 2;
#if BIGSIM_TIMING
  // timing
  if (0)	// detail
  if (genTimeLog) {
    for (j=0; j<cva(numNodes); j++)
    for (i=0; i<cva(bgMach).numWth; i++) {
      BgTimeLine &log = cva(nodeinfo)[j].timelines[i].timeline;	
//      BgPrintThreadTimeLine(nodeInfo::Local2Global(j), i, log);
      int x,y,z;
      nodeInfo::Local2XYZ(j, &x, &y, &z);
      BgWriteThreadTimeLine(arg_argv[0], x, y, z, i, log);
    }

  }
#endif
  if (genTimeLog) sendCorrectionStats();

  if (genTimeLog) writeToDisk();

//  if (tTHREADTYPE == WORK_THREAD)
  {
  int origPe = -2;
  // close all tracing modules
  for (j=0; j<cva(numNodes); j++)
    for (i=0; i<cva(bgMach).numWth; i++) {
      int pe = nodeInfo::Local2Global(j)*cva(bgMach).numWth+i;
      int oldPe = CmiSwitchToPE(pe);
      if (cva(bgMach).replay != -1)
        if ( pe != cva(bgMach).replay ) continue;
      if (origPe == -2) origPe = oldPe;
      traceCharmClose();
//      CmiSwitchToPE(oldPe);
      delete cva(nodeinfo)[j].threadinfo[i]->watcher;   // force dump watcher
      cva(nodeinfo)[j].threadinfo[i]->watcher = NULL;
      if (userTracingFn) userTracingFn();
    }
    if (origPe!=-2) CmiSwitchToPE(origPe);
  }

#if 0
  delete [] cva(nodeinfo);
  delete [] cva(inBuffer);
  for (i=0; i<cva(numNodes); i++) CmmFree(cva(msgBuffer)[i]);
  delete [] cva(msgBuffer);
#endif

#if CMK_HAS_COUNTER_PAPI
  if (cva(bgMach).timingMethod == BG_COUNTER) {
/*	  
  CmiPrintf("BG[PE %d]> cycles: %lld\n", CmiMyPe(), total_ins);
  CmiPrintf("BG[PE %d]> floating point instructions: %lld\n", CmiMyPe(), total_fps);
  CmiPrintf("BG[PE %d]> L1 cache misses: %lld\n", CmiMyPe(), total_l1_dcm);
  //CmiPrintf("BG[PE %d]> cycles stalled waiting for memory access: %lld\n", CmiMyPe(), total_mem_rcy);
 */
	for(int i=0; i<numPapiEvents; i++){
	  CmiPrintf("BG[PE %d]> %s: %lld\n", CmiMyPe(), papi_counters_desc[i], total_papi_counters[i]);
	   delete papi_counters_desc[i];
	}
	delete papiEvents;
	delete papiValues;
	delete papi_counters_desc;
	delete total_papi_counters;
  }
#endif

  //ConverseExit();
  if (genTimeLog)
    { if (CmiMyPe() != 0) CsdExitScheduler(); }
  else
    CsdExitScheduler();

  //if (CmiMyPe() == 0) CmiPrintf("\nBG> BlueGene emulator shutdown gracefully!\n");

  return 0;
}

static void sanityCheck()
{
  if (cva(bgMach).x==0 || cva(bgMach).y==0 || cva(bgMach).z==0)  {
    if (CmiMyPe() == 0)
      CmiPrintf("\nMissing parameters for BlueGene machine size!\n<tip> use command line options: +x, +y, or +z.\n");
    BgShutdown(); 
  } 
  else if (cva(bgMach).numCth==0 || cva(bgMach).numWth==0) { 
#if 1
    if (cva(bgMach).numCth==0) cva(bgMach).numCth=1;
    if (cva(bgMach).numWth==0) cva(bgMach).numWth=1;
#else
    if (CmiMyPe() == 0)
      CmiPrintf("\nMissing parameters for number of communication/worker threads!\n<tip> use command line options: +cth or +wth.\n");
    BgShutdown(); 
#endif
  }
  if (cva(bgMach).getNodeSize()<CmiNumPes()) {
    CmiAbort("\nToo few BigSim nodes!\n");
  }
}

#undef CmiSwitchToPE
extern "C" int CmiSwitchToPEFn(int pe);

// main
CmiStartFn bgMain(int argc, char **argv)
{
  int i;
  char *configFile = NULL;

  BgProcessMessage = BgProcessMessageDefault;
#if CMK_CONDS_USE_SPECIAL_CODE
  // overwrite possible implementation in machine.c
  CmiSwitchToPE = CmiSwitchToPEFn;
#endif

  /* initialize all processor level data */
  CpvInitialize(BGMach,bgMach);
  cva(bgMach).nullify();

  CmiArgGroup("Charm++","BlueGene Simulator");
  if (CmiGetArgStringDesc(argv, "+bgconfig", &configFile, "BlueGene machine config file")) {
   cva(bgMach). read(configFile);
  }
  CmiGetArgIntDesc(argv, "+x", &cva(bgMach).x, 
		"The x size of the grid of nodes");
  CmiGetArgIntDesc(argv, "+y", &cva(bgMach).y, 
		"The y size of the grid of nodes");
  CmiGetArgIntDesc(argv, "+z", &cva(bgMach).z, 
		"The z size of the grid of nodes");
  CmiGetArgIntDesc(argv, "+cth", &cva(bgMach).numCth, 
		"The number of simulated communication threads per node");
  CmiGetArgIntDesc(argv, "+wth", &cva(bgMach).numWth, 
		"The number of simulated worker threads per node");

  CmiGetArgIntDesc(argv, "+bgstacksize", &cva(bgMach).stacksize, 
		"Blue Gene thread stack size");

  if (CmiGetArgFlagDesc(argv, "+bglog", "Write events to log file"))
     genTimeLog = 1;
  if (CmiGetArgFlagDesc(argv, "+bgcorrect", "Apply timestamp correction to logs"))
    correctTimeLog = 1;
  schedule_flag = 0;
  if (correctTimeLog) {
    genTimeLog = 1;
    schedule_flag = 1;
  }

  if (CmiGetArgFlagDesc(argv, "+bgverbose", "Print debug info verbosely"))
    bgverbose = 1;

  // for timing method, default using elapse calls.
  if(CmiGetArgFlagDesc(argv, "+bgelapse", 
                       "Use user provided BgElapse for time prediction")) 
      cva(bgMach).timingMethod = BG_ELAPSE;
  if(CmiGetArgFlagDesc(argv, "+bgwalltime", 
                       "Use walltime method for time prediction")) 
      cva(bgMach).timingMethod = BG_WALLTIME;
#ifdef CMK_ORIGIN2000
  if(CmiGetArgFlagDesc(argv, "+bgcounter", "Use performance counter")) 
      cva(bgMach).timingMethod = BG_COUNTER;
#elif CMK_HAS_COUNTER_PAPI
  if (CmiGetArgFlagDesc(argv, "+bgpapi", "Use PAPI Performance counters")) {
    cva(bgMach).timingMethod = BG_COUNTER;
  }
  if (cva(bgMach).timingMethod == BG_COUNTER) {
    init_counters();
  }
#endif
  CmiGetArgDoubleDesc(argv,"+bgfpfactor", &cva(bgMach).fpfactor, 
		      "floating point to time factor");
  CmiGetArgDoubleDesc(argv,"+bgcpufactor", &cva(bgMach).cpufactor, 
		      "scale factor for wallclock time measured");
  CmiGetArgDoubleDesc(argv,"+bgtimercost", &cva(bgMach).timercost, 
		      "timer cost");
  #if 0
  if(cva(bgMach).timingMethod == BG_WALLTIME)
  {
      int count = 1e6;
      double start, stop, diff, cost, dummy;

      dummy = BG_TIMER(); // In case there's an initialization delay somewhere

      start = BG_TIMER();
      for (int i = 0; i < count; ++i)
	  dummy = BG_TIMER();
      stop = BG_TIMER();

      diff = stop - start;
      cost = diff / count;

      CmiPrintf("Measured timer cost: %g Actual: %g\n", 
                cost, cva(bgMach).timercost);
      cva(bgMach).timercost = cost;
  }
  #endif
  
  char *networkModel;
  if (CmiGetArgStringDesc(argv, "+bgnetwork", &networkModel, "Network model")) {
   cva(bgMach).setNetworkModel(networkModel);
  }

  bgcorroff = 0;
  if(CmiGetArgFlagDesc(argv, "+bgcorroff", "Start with correction off")) 
    bgcorroff = 1;

  bgstats_flag=0;
  if(CmiGetArgFlagDesc(argv, "+bgstats", "Print correction statistics")) 
    bgstats_flag = 1;

  if (CmiGetArgStringDesc(argv, "+bgtraceroot", &cva(bgMach).traceroot, "Directory to write bgTrace files to"))
  {
    char *root = (char*)malloc(strlen(cva(bgMach).traceroot) + 10);
    sprintf(root, "%s/", cva(bgMach).traceroot);
    cva(bgMach).traceroot = root;
  }

  // record/replay
  if (CmiGetArgFlagDesc(argv,"+bgrecord","Record message processing order for BigSim")) {
    cva(bgMach).record = 1;
    if (CmiMyPe() == 0)
      CmiPrintf("BG info> full record mode. \n");
  }
  if (CmiGetArgFlagDesc(argv,"+bgrecordnode","Record message processing order for BigSim")) {
    cva(bgMach).recordnode = 1;
    if (CmiMyPe() == 0)
      CmiPrintf("BG info> full record mode on node. \n");
  }
  int replaype;
  if (CmiGetArgIntDesc(argv,"+bgreplay", &replaype, "Re-play message processing order for BigSim")) {
    cva(bgMach).replay = replaype;
  }
  else {
    if (CmiGetArgFlagDesc(argv,"+bgreplay","Record message processing order for BigSim"))
    cva(bgMach).replay = 0;    // default to 0
  }
  if (cva(bgMach).replay >= 0) {
    if (CmiNumPes()>1)
      CmiAbort("BG> bgreplay mode must run on one physical processor.");
    if (cva(bgMach).x!=1 || cva(bgMach).y!=1 || cva(bgMach).z!=1 ||
         cva(bgMach).numWth!=1 || cva(bgMach).numCth!=1)
      CmiAbort("BG> bgreplay mode must run on one target processor.");
    CmiPrintf("BG info> replay mode for target processor %d.\n", cva(bgMach).replay);
  }
  char *procs = NULL;
  if (CmiGetArgStringDesc(argv, "+bgrecordprocessors", &procs, "A list of processors to record, e.g. 0,10,20-30")) {
    cva(bgMach).recordprocs.set(procs);
  }

    // record/replay at node level
  char *nodes = NULL;
  if (CmiGetArgStringDesc(argv, "+bgrecordnodes", &nodes, "A list of nodes to record, e.g. 0,10,20-30")) {
    cva(bgMach).recordnodes.set(nodes);
  }
  int replaynode;
  if (CmiGetArgIntDesc(argv,"+bgreplaynode", &replaynode, "Re-play message processing order for BigSim")) {      
    cva(bgMach).replaynode = replaynode; 
  }
  else {
    if (CmiGetArgFlagDesc(argv,"+bgreplaynode","Record message processing order for BigSim"))
    cva(bgMach).replaynode = 0;    // default to 0
  }
  if (cva(bgMach).replaynode >= 0) {
    int startpe, endpe;
    BgRead_nodeinfo(replaynode, startpe, endpe);
    if (cva(bgMach).numWth != endpe-startpe+1) {
      cva(bgMach).numWth = endpe-startpe+1;     // update wth
      CmiPrintf("BG info> numWth is changed to %d.\n", cva(bgMach).numWth);
    }
    if (CmiNumPes()>1)
      CmiAbort("BG> bgreplay mode must run on one physical processor.");
    if (cva(bgMach).x!=1 || cva(bgMach).y!=1 || cva(bgMach).z!=1)
      CmiAbort("BG> bgreplay mode must run on one target processor.");
    CmiPrintf("BG info> replay mode for target node %d.\n", cva(bgMach).replaynode);   
  }     
  CmiAssert(!(cva(bgMach).replaynode != -1 && cva(bgMach).replay != -1));


  /* parameters related with out-of-core execution */
  int tmpcap=0;
  if (CmiGetArgIntDesc(argv, "+bgooccap", &tmpcap, "Simulate with out-of-core support and the number of target processors allowed in memory")){
     TBLCAPACITY = tmpcap;
    }
  if (CmiGetArgDoubleDesc(argv, "+bgooc", &bgOOCMaxMemSize, "Simulate with out-of-core support and the threshhold of memory size")){
      bgUseOutOfCore = 1;
      _BgInOutOfCoreMode = 1; //the global (the whole converse layer) out-of-core flag

      double curFreeMem = bgGetSysFreeMemSize();
      if(fabs(bgOOCMaxMemSize - 0.0)<=1e-6){
	//using the memory available of the system right now
	//assuming no other programs will run after this program runs
	bgOOCMaxMemSize = curFreeMem;
	CmiPrintf("Using the system's current memory available: %.3fMB\n", bgOOCMaxMemSize);
      }
      if(bgOOCMaxMemSize > curFreeMem){
	CmiPrintf("Warning: not enough memory for the specified memory size, now use the current available memory %3.fMB.\n", curFreeMem);
	bgOOCMaxMemSize = curFreeMem;
      }
      DEBUGF(("out-of-core turned on!\n"));
  }      

#if BIGSIM_DEBUG_LOG
  {
    char ln[200];
    sprintf(ln,"bgdebugLog.%d",CmiMyPe());
    bgDebugLog=fopen(ln,"w");
  }
#endif

  arg_argv = argv;
  arg_argc = CmiGetArgc(argv);

  /* msg handler */
  CpvInitialize(SimState, simState);
  cva(simState).msgHandler = CmiRegisterHandler((CmiHandler) msgHandlerFunc);
  cva(simState).nBcastMsgHandler = CmiRegisterHandler((CmiHandler)nodeBCastMsgHandlerFunc);
  cva(simState).tBcastMsgHandler = CmiRegisterHandler((CmiHandler)threadBCastMsgHandlerFunc);
  cva(simState).exitHandler = CmiRegisterHandler((CmiHandler) exitHandlerFunc);

  cva(simState).beginExitHandler = CmiRegisterHandler((CmiHandler) beginExitHandlerFunc);
  cva(simState).inEmulatorInit = 1;
  /* call user defined BgEmulatorInit */
  BgEmulatorInit(arg_argc, arg_argv);
  cva(simState).inEmulatorInit = 0;

  /* check if all bluegene node size and thread information are set */
  sanityCheck();

  _bgSize = cva(bgMach).getNodeSize(); 

  timerFunc = BgGetTime;

  BgInitTiming();		// timing module

  if (CmiMyPe() == 0) {
    CmiPrintf("BG info> Simulating %dx%dx%d nodes with %d comm + %d work threads each.\n", cva(bgMach).x, cva(bgMach).y, cva(bgMach).z, cva(bgMach).numCth, cva(bgMach).numWth);
    CmiPrintf("BG info> Network type: %s.\n", cva(bgMach).network->name());
    cva(bgMach).network->print();
    CmiPrintf("BG info> cpufactor is %f.\n", cva(bgMach).cpufactor);
    CmiPrintf("BG info> floating point factor is %f.\n", cva(bgMach).fpfactor);
    if (cva(bgMach).stacksize)
      CmiPrintf("BG info> BG stack size: %d bytes. \n", cva(bgMach).stacksize);
    if (cva(bgMach).timingMethod == BG_ELAPSE) 
      CmiPrintf("BG info> Using BgElapse calls for timing method. \n");
    else if (cva(bgMach).timingMethod == BG_WALLTIME)
      CmiPrintf("BG info> Using WallTimer for timing method. \n");
    else if (cva(bgMach).timingMethod == BG_COUNTER)
      CmiPrintf("BG info> Using performance counter for timing method. \n");
    if (genTimeLog)
      CmiPrintf("BG info> Generating timing log. \n");
    if (correctTimeLog)
      CmiPrintf("BG info> Perform timestamp correction. \n");
    if (cva(bgMach).traceroot)
      CmiPrintf("BG info> bgTrace root is '%s'. \n", cva(bgMach).traceroot);
  }

  CtvInitialize(threadInfo *, threadinfo);

  /* number of bg nodes on this PE */
  CpvInitialize(int, numNodes);
  cva(numNodes) = nodeInfo::numLocalNodes();

  if (CmiMyRank() == 0)
    initLock = CmiCreateLock();     // used for BnvInitialize
  CmiNodeAllBarrier(); //barrier to make sure initLock is created

  bgstreaming.init(cva(numNodes));

  //Must initialize out-of-core related data structures before creating any BG nodes and procs
if(bgUseOutOfCore){
      initTblThreadInMem();
	  #if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
      //init prefetch status      
      thdsOOCPreStatus = new oocPrefetchStatus[cva(numNodes)*cva(bgMach).numWth];
      oocPrefetchSpace = new oocPrefetchBufSpace();
      schedWorkThds = new oocWorkThreadQueue();
	  #endif
  }

#if BIGSIM_OUT_OF_CORE
  //initialize variables related to get precise
  //physical memory usage info for a process
  bgMemPageSize = getpagesize();
  memset(bgMemStsFile, 0, 25); 
  sprintf(bgMemStsFile, "/proc/%d/statm", getpid());
#endif


  /* create BG nodes */
  CpvInitialize(nodeInfo *, nodeinfo);
  cva(nodeinfo) = new nodeInfo[cva(numNodes)];
  _MEMCHECK(cva(nodeinfo));

  cta(threadinfo) = new threadInfo(-1, UNKNOWN_THREAD, NULL);
  _MEMCHECK(cta(threadinfo));

  /* create BG processors for each node */
  for (i=0; i<cva(numNodes); i++)
  {
    nodeInfo *ninfo = cva(nodeinfo) + i;
    // create threads
    ninfo->initThreads(i);

    /* pretend that I am a thread */
    cta(threadinfo)->myNode = ninfo;

    /* initialize a BG node and fire all threads */
    BgNodeInitialize(ninfo);
  }

  // clear main thread.
  cta(threadinfo)->myNode = NULL;

  CpvInitialize(CthThread, mainThread);
  cva(mainThread) = CthSelf();

  CpvInitialize(int, CthResumeBigSimThreadIdx);

  cva(simState).simStartTime = CmiWallTimer();    
  return 0;
}

// for conv-conds:
// if -2 untouch
// if -1 main thread
#if CMK_BIGSIM_THREAD
extern "C" int CmiSwitchToPEFn(int pe)
{
  if (pe == -2) return -2;
  int oldpe;
//  ASSERT(tTHREADTYPE != COMM_THREAD);
  if (tMYNODE == NULL) oldpe = -1;
  else if (tTHREADTYPE == COMM_THREAD) oldpe = -BgGetThreadID();
  else if (tTHREADTYPE == WORK_THREAD) oldpe = BgGetGlobalWorkerThreadID();
  else oldpe = -1;
//CmiPrintf("CmiSwitchToPE from %d to %d\n", oldpe, pe);
  if (pe == -1) {
    CthSwitchThread(cva(mainThread));
  }
  else if (pe < 0) {
  }
  else {
//    if (cva(bgMach).inReplayMode()) pe = 0;         /* replay mode */
    int t = pe%cva(bgMach).numWth;
    int newpe = nodeInfo::Global2Local(pe/cva(bgMach).numWth);
    if (cva(bgMach).replay!=-1) newpe = 0;
    nodeInfo *ninfo = cva(nodeinfo) + newpe;;
    threadInfo *tinfo = ninfo->threadinfo[t];
    CthSwitchThread(tinfo->me);
  }
  return oldpe;
}
#else
extern "C" int CmiSwitchToPEFn(int pe)
{
  if (pe == -2) return -2;
  int oldpe;
  if (tMYNODE == NULL) oldpe = -1;
  else oldpe = BgMyNode();
  if (pe == -1) {
    cta(threadinfo)->myNode = NULL;
  }
  else {
    int newpe = nodeInfo::Global2Local(pe);
    cta(threadinfo)->myNode = cva(nodeinfo) + newpe;;
  }
  return oldpe;
}
#endif


/*****************************************************************************
			TimeLog correction
*****************************************************************************/

extern void processCorrectionMsg(int nodeidx);

// return the msg pointer, and the index of the message in the affinity queue.
static inline char* searchInAffinityQueue(int nodeidx, BgMsgID &msgId, CmiInt2 tID, int &index)
{
  CmiAssert(tID != ANYTHREAD);
  ckMsgQueue &affinityQ = cva(nodeinfo)[nodeidx].affinityQ[tID];
  for (int i=0; i<affinityQ.length(); i++)  {
      char *msg = affinityQ[i];
      BgMsgID md = BgMsgID(CmiBgMsgSrcPe(msg), CmiBgMsgID(msg));
      if (msgId == md) {
        index = i;
        return msg;
      }
  }
  return NULL;
}

// return the msg pointer, thread id and the index of the message in the affinity queue.
static char* searchInAffinityQueueInNode(int nodeidx, BgMsgID &msgId, CmiInt2 &tID, int &index)
{
  for (tID=0; tID<cva(bgMach).numWth; tID++) {
    char *msg = searchInAffinityQueue(nodeidx, msgId, tID, index);
    if (msg) return msg;
  }
  return NULL;
}

StateCounters stateCounters;

int updateRealMsgs(bgCorrectionMsg *cm, int nodeidx)
{
  char *msg;
  CmiInt2 tID = cm->tID;
  int index;
  if (tID == ANYTHREAD) {
    msg = searchInAffinityQueueInNode(nodeidx, cm->msgId, tID, index);
  }
  else {
    msg = searchInAffinityQueue(nodeidx, cm->msgId, tID, index);
  }
  if (msg == NULL) return 0;

  ckMsgQueue &affinityQ = cva(nodeinfo)[nodeidx].affinityQ[tID];
  CmiBgMsgRecvTime(msg) = cm->tAdjust;
  affinityQ.update(index);
  CthThread tid = cva(nodeinfo)[nodeidx].threadTable[tID];
  unsigned int prio = (unsigned int)(cm->tAdjust*PRIO_FACTOR)+1;
  CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
  stateCounters.corrMsgCRCnt++;
  return 1;       /* invalidate this msg */
}

extern void processBufferCorrectionMsgs(void *ignored);

// Coverse handler for begin exit
// flush and process all correction messages
static void beginExitHandlerFunc(void *msg)
{
  CmiFree(msg);
  delayCheckFlag = 0;
//CmiPrintf("\n\n\nbeginExitHandlerFunc called on %d\n", CmiMyPe());
  programExit = 1;
#if LIMITED_SEND
  CQdCreate(CpvAccess(cQdState), BgNodeSize());
#endif

#if 1
  for (int i=0; i<BgNodeSize(); i++) 
    processCorrectionMsg(i); 
#if USE_MULTISEND
  BgSendBufferedCorrMsgs();
#endif
#else
  // the msg queue should be empty now.
  // don't do insert adjustment, but start to do all timing correction from here
  int nodeidx, tID;
  for (nodeidx=0; nodeidx<BgNodeSize(); nodeidx++) {
    BgTimeLineRec *tlines = cva(nodeinfo)[nodeidx].timelines;
    for (tID=0; tID<cva(numWth); tID++) {
        BgTimeLineRec &tlinerec = tlines[tID];
        BgAdjustTimeLine(tlinerec, nodeidx, tID);
    }
  }

#endif

#if !THROTTLE_WORK
#if DELAY_CHECK
  CcdCallFnAfter(processBufferCorrectionMsgs,NULL,CHECK_INTERVAL);
#endif
#endif
}

#define HISTOGRAM_SIZE  100
// compute total CPU utilization for each timeline and 
// return the number of real msgs
void computeUtilForAll(int* array, int *nReal)
{
  double scale = 1.0e3;		// scale to ms

  //We measure from 1ms to 5001 ms in steps of 100 ms
  int min = 0, step = 1;
  int max = min + HISTOGRAM_SIZE*step;
  // [min, max: step]  HISTOGRAM_SIZE slots

  if (CmiMyPe()==0)
    CmiPrintf("computeUtilForAll: min:%d max:%d step:%d scale:%f.\n", min, max, step, scale);
  int size = (max-min)/step;
  CmiAssert(size == HISTOGRAM_SIZE);
  int allmin = -1, allmax = -1;
  for(int i=0;i<size;i++) array[i] = 0;

  for (int nodeidx=0; nodeidx<cva(numNodes); nodeidx++) {
    BgTimeLineRec *tlinerec = cva(nodeinfo)[nodeidx].timelines;
    for (int tID=0; tID<cva(bgMach).numWth; tID++) {
      int util = (int)(scale*(tlinerec[tID].computeUtil(nReal)));

      if (util >= max) { if (util>allmax||allmax==-1) allmax=util; util=max-1;}
      if (util < min) { if (util<allmin||allmin==-1) allmin=util; util=min; }
      array[(util-min)/step]++;
    }
  }
  if (allmin!=-1 || allmax!=-1)
    CmiPrintf("[%d] Warning: computeUtilForAll out of range %f - %f.\n", CmiMyPe(), (allmin==-1)?-1:allmin/scale, (allmax==-1)?-1:allmax/scale);
}

class StatsMessage {
  char core[CmiBlueGeneMsgHeaderSizeBytes];
public:
  int processCount;
  int corrMsgCount;
  int realMsgCount;
  int maxTimelineLen, minTimelineLen;
};

extern int processCount, corrMsgCount;

static void sendCorrectionStats()
{
  int msgSize = sizeof(StatsMessage)+sizeof(int)*HISTOGRAM_SIZE;
  StatsMessage *statsMsg = (StatsMessage *)CmiAlloc(msgSize);
  statsMsg->processCount = processCount;
  statsMsg->corrMsgCount = corrMsgCount;
  int numMsgs=0;
  int maxTimelineLen=-1, minTimelineLen=CMK_MAXINT;
  int totalMem = 0;
  if (bgstats_flag) {
  for (int nodeidx=0; nodeidx<cva(numNodes); nodeidx++) {
    BgTimeLineRec *tlines = cva(nodeinfo)[nodeidx].timelines;
    for (int tID=0; tID<cva(bgMach).numWth; tID++) {
        BgTimeLineRec &tlinerec = tlines[tID];
	int tlen = tlinerec.length();
	if (tlen>maxTimelineLen) maxTimelineLen=tlen;
	if (tlen<minTimelineLen) minTimelineLen=tlen;
        totalMem = tlen*sizeof(BgTimeLog);
//CmiPrintf("[%d node:%d] BgTimeLog: %dK len:%d size of bglog: %d bytes\n", CmiMyPe(), nodeidx, totalMem/1000, tlen, sizeof(BgTimeLog));
#if 0
        for (int i=0; i< tlinerec.length(); i++) {
          numMsgs += tlinerec[i]->msgs.length();
        }
#endif
    }
  }
  computeUtilForAll((int*)(statsMsg+1), &numMsgs);
  statsMsg->realMsgCount = numMsgs;
  statsMsg->maxTimelineLen = maxTimelineLen;
  statsMsg->minTimelineLen = minTimelineLen;
  }  // end if

  CmiSetHandler(statsMsg, cva(simState).bgStatCollectHandler);
  CmiSyncSendAndFree(0, msgSize, statsMsg);
}

// Converse handler for collecting stats
void statsCollectionHandlerFunc(void *msg)
{
  static int count=0;
  static int pc=0, cc=0, realMsgCount=0;
  static int maxTimelineLen=0, minTimelineLen=CMK_MAXINT;
  static int *histArray = NULL;
  int i;

  count++;
  if (histArray == NULL) {
    histArray = new int[HISTOGRAM_SIZE];
    for (i=0; i<HISTOGRAM_SIZE; i++) histArray[i]=0;
  }
  StatsMessage *m = (StatsMessage *)msg;
  pc += m->processCount;
  cc += m->corrMsgCount;
  realMsgCount += m->realMsgCount;
  if (minTimelineLen> m->minTimelineLen) minTimelineLen=m->minTimelineLen;
  if (maxTimelineLen< m->maxTimelineLen) maxTimelineLen=m->maxTimelineLen;
  int *array = (int *)(m+1);
  for (i=0; i<HISTOGRAM_SIZE; i++) histArray[i] += array[i];
  if (count == CmiNumPes()) {
    if (bgstats_flag) {
      CmiPrintf("Total procCount:%d corrMsgCount:%d realMsg:%d timeline:%d-%d\n", pc, cc, realMsgCount, minTimelineLen, maxTimelineLen);
      for (i=0; i<HISTOGRAM_SIZE; i++) {
        CmiPrintf("%d ", histArray[i]);
	if (i%20 == 19) CmiPrintf("\n");
      }
      CmiPrintf("\n");
    }
    CsdExitScheduler();
  }
  CmiFree(msg);
}

// update arrival time from buffer messages
// before start an entry, check message time against buffered timing
// correction message to update to the correct time.
void correctMsgTime(char *msg)
{
   if (!correctTimeLog) return;
//   if (CkMsgDoCorrect(msg) == 0) return;

   BgMsgID msgId(CmiBgMsgSrcPe(msg), CmiBgMsgID(msg));
   CmiInt2 tid = CmiBgMsgThreadID(msg);

   bgCorrectionQ &cmsg = cva(nodeinfo)[tMYNODEID].cmsg;
   int len = cmsg.length();
   for (int i=0; i<len; i++) {
     bgCorrectionMsg* m = cmsg[i];
     if (msgId == m->msgId && tid == m->tID) {
        if (m->tAdjust < 0.0) return;
        //CmiPrintf("correctMsgTime from %e to %e\n", CmiBgMsgRecvTime(msg), m->tAdjust);
	CmiBgMsgRecvTime(msg) = m->tAdjust;
	m->tAdjust = -1.0;       /* invalidate this msg */
//	cmsg.update(i);
        stateCounters.corrMsgRCCnt++;
	break;
     }
   }
}

extern void traceWriteSTS(FILE *stsfp,int nUserEvents);
//TODO: write disk bgTraceFiles
static void writeToDisk()
{

  char* d = new char[1025];
  //Num of simulated procs on this real pe
  int numLocalProcs = cva(numNodes)*cva(bgMach).numWth;

  const PUP::machineInfo &machInfo = PUP::machineInfo::current();

  // write summary file on PE0
  if(CmiMyPe()==0){
    
    sprintf(d, "%sbgTrace", cva(bgMach).traceroot?cva(bgMach).traceroot:""); 
    FILE *f2 = fopen(d,"wb");
    //Total emulating processors and total target BG processors
    int numEmulatingPes=CmiNumPes();
    int totalWorkerProcs = BgNumNodes()*cva(bgMach).numWth;

    if(f2==NULL) {
      CmiPrintf("[%d] Creating trace file %s  failed\n", CmiMyPe(), d);
      CmiAbort("BG> Abort");
    }
    PUP::toDisk p(f2);
    p((char *)&machInfo, sizeof(machInfo));
    p|totalWorkerProcs;
    p|cva(bgMach);
    p|numEmulatingPes;
    p|bglog_version;
    p|CpvAccess(CthResumeBigSimThreadIdx);
    
    CmiPrintf("[0] Number is numX:%d numY:%d numZ:%d numCth:%d numWth:%d numEmulatingPes:%d totalWorkerProcs:%d bglog_ver:%d\n",cva(bgMach).x,cva(bgMach).y,cva(bgMach).z,cva(bgMach).numCth,cva(bgMach).numWth,numEmulatingPes,totalWorkerProcs,bglog_version);
    
    fclose(f2);

    FILE* stsfp = fopen("tproj.sts", "w");
    if (stsfp == 0) {
         CmiAbort("Cannot open summary sts file for writing.\n");
    }
    traceWriteSTS(stsfp,0);
  }
  
  sprintf(d, "%sbgTrace%d", cva(bgMach).traceroot?cva(bgMach).traceroot:"", CmiMyPe()); 
  FILE *f = fopen(d,"wb");
 
  if(f==NULL)
    CmiPrintf("Creating bgTrace%d failed\n",CmiMyPe());
  PUP::toDisk p(f);
  
  p((char *)&machInfo, sizeof(machInfo));	// machine info
  p|numLocalProcs;

  // CmiPrintf("Timelines are: \n");
  int procTablePos = ftell(f);

  int *procOffsets = new int[numLocalProcs];
  int procTableSize = (numLocalProcs)*sizeof(int);
  fseek(f,procTableSize,SEEK_CUR); 

  for (int j=0; j<cva(numNodes); j++){
    for(int i=0;i<cva(bgMach).numWth;i++){
    #if BIGSIM_OUT_OF_CORE
	//When isomalloc is used, some events inside BgTimeLineRec are allocated
	//through isomalloc. Therefore, the memory containing those events needs
	//to be brought back into memory from disk. --Chao Mei		
	if(bgUseOutOfCore && CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC))
	    bgOutOfCoreSchedule(cva(nodeinfo)[j].threadinfo[i]);
     #endif	
      BgTimeLineRec &t = cva(nodeinfo)[j].timelines[i];
      procOffsets[j*cva(bgMach).numWth + i] = ftell(f);
      if(procOffsets[j*cva(bgMach).numWth + i] == -1) {
        CmiPrintf("ftell operation failure while writing bigsim logs to %s with"
                  " error %s (%d)\n", d, strerror(errno), errno);
        CmiAbort("Error while writing logs\n");
      }
      t.pup(p);
    }
  }
  
  fseek(f,procTablePos,SEEK_SET);
  p(procOffsets,numLocalProcs);
  fclose(f);

  if(CmiMyPe() == 0) 
    CmiPrintf("[%d] Wrote to disk for %d BG nodes. \n", CmiMyPe(), cva(numNodes));
}


/*****************************************************************************
             application Converse thread hook
*****************************************************************************/

CpvExtern(int      , CthResumeBigSimThreadIdx);

void CthEnqueueBigSimThread(CthThreadToken* token, int s,
                                   int pb,unsigned int *prio)
{
/*
  CmiSetHandler(token, CpvAccess(CthResumeNormalThreadIdx));
  CsdEnqueueGeneral(token, s, pb, prio);
*/
#if CMK_BIGSIM_THREAD
  int x, y, z;
  BgGetMyXYZ(&x, &y, &z);
  int t = BgGetThreadID();
#else
  #error "ERROR HERE"
#endif
    // local message into queue
  DEBUGM(4, ("In EnqueueBigSimThread method!\n"));

  DEBUGM(4, ("token [%p] is added to queue pointing to thread[%p]\n", token, token->thread));
  BgSendPacket(x,y,z, t, CpvAccess(CthResumeBigSimThreadIdx), LARGE_WORK, sizeof(CthThreadToken), (char *)token);
} 

CthThread CthSuspendBigSimThread()
{ 
  return  cta(threadinfo)->me;
}

static void bigsimThreadListener_suspend(struct CthThreadListener *l)
{
   // stop timer
   stopVTimer();
}

static void bigsimThreadListener_resume(struct CthThreadListener *l)
{
   // start timer by reset it
   resetVTime();
}


void BgSetStrategyBigSimDefault(CthThread t)
{ 
  CthSetStrategy(t,
                 CthEnqueueBigSimThread,
                 CthSuspendBigSimThread);

  CthThreadListener *a = new CthThreadListener;
  a->suspend = bigsimThreadListener_suspend;
  a->resume = bigsimThreadListener_resume;
  a->free = NULL;
  CthAddListener(t, a);
}

int BgIsMainthread()
{
    return tMYNODE == NULL;
}

int BgIsRecord()
{
    return cva(bgMach).record == 1 || cva(bgMach).recordnode == 1;
}

int BgIsReplay()
{
    return cva(bgMach).replay != -1 || cva(bgMach).replaynode != -1;
}

extern "C" void CkReduce(void *msg, int size, CmiReduceMergeFn mergeFn) {
  ((workThreadInfo*)cta(threadinfo))->reduceMsg = msg;
  //CmiPrintf("Called CkReduce from %d %hd\n",CmiMyPe(),cta(threadinfo)->globalId);
  int numLocal = 0, count = 0;
  for (int j=0; j<cva(numNodes); j++){
    for(int i=0;i<cva(bgMach).numWth;i++){
      workThreadInfo *t = (workThreadInfo*)cva(nodeinfo)[j].threadinfo[i];
      if (t->reduceMsg == NULL) return; /* we are not yet ready to reduce */
      numLocal ++;
    }
  }
  
  /* Since the current message is passed is as "local" to the merge function,
   * and it will not be nullified in the upcoming loop, make it NULL explicitely. */
  ((workThreadInfo*)cta(threadinfo))->reduceMsg = NULL;
  
  void **msgLocal = (void**)malloc(sizeof(void*)*(numLocal-1));
  for (int j=0; j<cva(numNodes); j++){
    for(int i=0;i<cva(bgMach).numWth;i++){
      workThreadInfo *t = (workThreadInfo*)cva(nodeinfo)[j].threadinfo[i];
      if (t == cta(threadinfo)) break;
      msgLocal[count++] = t->reduceMsg;
      t->reduceMsg = NULL;
    }
  }
  CmiAssert(count==numLocal-1);
  msg = mergeFn(&size, msg, msgLocal, numLocal-1);
  CmiReduce(msg, size, mergeFn);
  //CmiPrintf("Called CmiReduce %d\n",CmiMyPe());
  for (int i=0; i<numLocal-1; ++i) CmiFree(msgLocal[i]);
  free(msgLocal);
}

// for record/replay, to fseek back
void BgRewindRecord()
{
  threadInfo *tinfo = cta(threadinfo);
  if (tinfo->watcher) tinfo->watcher->rewind();
}


void BgRegisterUserTracingFunction(BgTracingFn fn)
{
  userTracingFn = fn;
}


