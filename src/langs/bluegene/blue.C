/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/


/*
  File: Blue.C -- Converse BlueGene Emulator Code
  Emulator written by Gengbin Zheng, gzheng@uiuc.edu on 2/20/2001
*/ 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cklists.h"

#define  DEBUGF(x)      //CmiPrintf x;

#include "queueing.h"
#include "blue.h"
#include "blue_impl.h"    	// implementation header file
#include "blue_timing.h" 	// timing module

#ifdef CMK_ORIGIN2000
extern "C" int start_counters(int e0, int e1);
extern "C" int read_counters(int e0, long long *c0, int e1, long long *c1);
static int counterStarted = 0;
inline double Count2Time(long long c) { return c*5.e-7; }
#endif

/* node level variables */
CpvDeclare(nodeInfo*, nodeinfo);		/* represent a bluegene node */

/* thread level variables */
CtvDeclare(threadInfo *, threadinfo);	/* represent a bluegene thread */

CpvStaticDeclare(CthThread, mainThread);

/* BG machine parameter */
CpvDeclare(int, numX);    /* size of bluegene nodes in cube */
CpvDeclare(int, numY);
CpvDeclare(int, numZ);
CpvDeclare(int, numCth);  /* number of threads */
CpvDeclare(int, numWth);
CpvDeclare(int, numNodes);        /* number of bg nodes on this PE */

/* emulator node level variables */
CpvDeclare(int,msgHandler);
CpvDeclare(int,nBcastMsgHandler);
CpvDeclare(int,tBcastMsgHandler);
CpvDeclare(int,exitHandler);
CpvDeclare(int,beginExitHandler);
CpvDeclare(int,bgStatCollectHandler);
CpvDeclare(int, inEmulatorInit);

static void sendCorrectionStats();
extern "C" void defaultBgHandler(char *, void *);
static void writeToDisk();

static int arg_argc;
static char **arg_argv;

static int printTimeLog = 0;
int bgSize = 0;
static int timingMethod = BG_ELAPSE;
int delayCheckFlag = 1;          // when enabled, only check correction 
					// messages after some interval
int programExit = 0;

int bgstats = 0;
FILE *bgDebugLog;			// for debugging

extern int processCount, corrMsgCount;

#define BGARGSCHECK   	\
  if (cva(numX)==0 || cva(numY)==0 || cva(numZ)==0)  { if (CmiMyPe() == 0) { CmiPrintf("\nMissing parameters for BlueGene machine size!\n<tip> use command line options: +x, +y, or +z.\n");} BgShutdown(); } \
  else if (cva(numCth)==0 || cva(numWth)==0) { if (CmiMyPe() == 0) { CmiPrintf("\nMissing parameters for number of communication/worker threads!\n<tip> use command line options: +cth or +wth.\n");} BgShutdown(); }	\
  else if (cva(numX)*cva(numY)*cva(numZ)<CmiNumPes()) {	\
    CmiAbort("\nToo few BlueGene nodes!\n");	\
  }


StateCounters stateCounters;


class StatsMessage {
  char core[CmiBlueGeneMsgHeaderSizeBytes];
public:
  int processCount;
  int corrMsgCount;
  int realMsgCount;
  int maxTimelineLen, minTimelineLen;
};


/*****************************************************************************
     Handler Table, one per thread
*****************************************************************************/

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
    ASSERT(!cva(inEmulatorInit));
    /* leave 0 as blank, so it can report error luckily */
    int cur = handlerTableCount++;
    if (cur >= MAX_HANDLERS)
      CmiAbort("BG> HandlerID exceed the maximum.\n");
    handlerTable[cur].fnPtr = (BgHandlerEx)h;
    handlerTable[cur].userPtr = NULL;
    return cur;
}

inline void HandlerTable::numberHandler(int idx, BgHandler h)
{
    ASSERT(!cva(inEmulatorInit));
    if (idx >= handlerTableCount || idx < 1)
      CmiAbort("BG> HandlerID exceed the maximum!\n");
    handlerTable[idx].fnPtr = (BgHandlerEx)h;
    handlerTable[idx].userPtr = NULL;
}

inline void HandlerTable::numberHandlerEx(int idx, BgHandlerEx h, void *uPtr)
{
    ASSERT(!cva(inEmulatorInit));
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
    if (handler >= handlerTableCount) return NULL;
    return &handlerTable[handler];
}

/**
  nodeInfo construtor
*/
nodeInfo::nodeInfo(): lastW(0), udata(NULL), started(0) 
{
    int i;

    inBuffer.initialize(INBUFFER_SIZE);
    msgBuffer = CmmNew();

    commThQ = new threadQueue;
    commThQ->initialize(cva(numCth));

    threadTable = new CthThread[cva(numWth)+cva(numCth)];
    _MEMCHECK(threadTable);
    threadinfo = new threadInfo*[cva(numWth)+cva(numCth)];
    _MEMCHECK(threadinfo);

    affinityQ = new ckMsgQueue[cva(numWth)];
    _MEMCHECK(affinityQ);

    // create threadinfo
    for (i=0; i< cva(numWth); i++)
    {
      threadinfo[i] = new threadInfo(i, WORK_THREAD, this);
      _MEMCHECK(threadinfo[i]);
    }
    for (i=0; i< cva(numCth); i++)
    {
      threadinfo[i+cva(numWth)] = new threadInfo(i+cva(numWth), COMM_THREAD, this);
      _MEMCHECK(threadinfo[i+cva(numWth)]);
    }
#if BLUEGENE_TIMING
    timelines = new BgTimeLineRec[cva(numWth)]; // set default size 1024
    _MEMCHECK(timelines);
#endif
  }

nodeInfo::~nodeInfo() 
{
    if (commThQ) delete commThQ;
    delete [] affinityQ;
    delete [] threadTable;
    delete [] threadinfo;
}


/* add a message to this bluegene node's inbuffer queue */
void nodeInfo::addBgNodeInbuffer(char *msgPtr)
{
  int tags[1];

  /* if inbuffer is full, store in the msgbuffer */
  if (inBuffer.isFull()) {
    tags[0] = id;
    CmmPut(msgBuffer, 1, tags, msgPtr);
  }
  else {
    DEBUGF(("inbuffer is not full.\n"));
    inBuffer.enq(msgPtr);
  }
  /* awake a communication thread to schedule it */
  CthThread t = commThQ->deq();
  if (t) {
    DEBUGF(("activate communication thread on node %d: %p.\n", nodeID, t));
#if 0
    unsigned int prio = 0;
    CthAwakenPrio(t, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
#else
    CthAwaken(t);
#endif
  }
}

/* called by comm thread to poll inBuffer */
char *nodeInfo::getFullBuffer()
{
  int tags[1], ret_tags[1];
  char *data, *mb;

  /* see if we have msg in inBuffer */
  if (inBuffer.isEmpty()) return NULL;
  data = inBuffer.deq(); 

  /* since we have just delete one from inbuffer, fill one from msgbuffer */
  tags[0] = CmmWildCard;
  mb = (char *)CmmGet(msgBuffer, 1, tags, ret_tags);
  if (mb) inBuffer.enq(mb);

  return data;
}

/* add a message to this bluegene node's non-affinity queue */
void nodeInfo::addBgNodeMessage(char *msgPtr)
{
  int i;
  /* find a idle worker thread */
  /* FIXME:  flat search is bad if there is many work threads */
  int wID = lastW;
  for (i=0; i<cva(numWth); i++) 
  {
    wID ++;
    if (wID == cva(numWth)) wID = 0;
    if (affinityQ[wID].length() == 0)
    {
      /* this work thread is idle, schedule the msg here */
      DEBUGF(("activate a work thread %d - %p.\n", wID, threadTable[wID]));
      affinityQ[wID].enq(msgPtr);
#if SCHEDULE_WORK
      double nextT = CmiBgMsgRecvTime(msgPtr);
      CthThread tid = threadTable[wID];
      unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
      CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
#else
      CthAwaken(threadTable[wID]);
#endif
      lastW = wID;
      return;
    }
  }
  /* all worker threads are busy */   
  DEBUGF(("all work threads are busy.\n"));
#if SCHEDULE_WORK
#if 0
  DEBUGF(("[N%d] activate all work threads on N%d.\n", id));
  double nextT = CmiBgMsgRecvTime(msgPtr);
  unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
  nodeQ.enq(msgPtr);
  for (i=0; i<cva(numWth); i++) 
  {
      CthThread tid = threadTable[i];
      CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
  }
#else
    // only awake rank 0 thread
  threadinfo[0]->addAffMessage(msgPtr);
/*
  affinityQ[0].enq(msgPtr);
  CthThread tid = threadTable[0];
  CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
*/
#endif

#else
  nodeQ.enq(msgPtr);
#endif
}

/**
  threadInfo methods
*/
void threadInfo::addAffMessage(char *msgPtr)
{
  ckMsgQueue &que = myNode->affinityQ[id];
  que.enq(msgPtr);
#if SCHEDULE_WORK
  /* don't awake directly, put into a priority queue sorted by recv time */
  double nextT = CmiBgMsgRecvTime(msgPtr);
  CthThread tid = me;
  unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
  DEBUGF(("[%d] awaken worker thread with prio %d.\n", tMYNODEID, prio));
  CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
#else
  if (que.length() == 1) {
    CthAwaken(me);
  }
#endif
}

/*****************************************************************************
      low level API
*****************************************************************************/

extern "C" void defaultBgHandler(char *null, void *uPtr)
{
  CmiAbort("BG> Invalid Handler called!\n");
}

int BgRegisterHandler(BgHandler h)
{
  ASSERT(!cva(inEmulatorInit));
  int cur;
#if CMK_BLUEGENE_NODE
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

void BgNumberHandler(int idx, BgHandler h)
{
  ASSERT(!cva(inEmulatorInit));
#if CMK_BLUEGENE_NODE
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
  ASSERT(!cva(inEmulatorInit));
#if CMK_BLUEGENE_NODE
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
void addBgNodeInbuffer(char *msgPtr, int lnodeID)
{
#ifndef CMK_OPTIMIZE
  if (lnodeID >= cva(numNodes)) CmiAbort("NodeID is out of range!");
#endif
  nodeInfo &nInfo = cva(nodeinfo)[lnodeID];
  nInfo.addBgNodeInbuffer(msgPtr);
}

/** BG API Func 
 *  called by a comm thread
 *  add a message to a thread's affinity queue in same node 
 */
void addBgThreadMessage(char *msgPtr, int threadID)
{
#ifndef CMK_OPTIMIZE
  if (threadID >= cva(numWth)) CmiAbort("ThreadID is out of range!");
#endif
  threadInfo *tInfo = tMYNODE->threadinfo[threadID];
  tInfo->addAffMessage(msgPtr);
}

/** BG API Func 
 *  called by a comm thread, add a message to a node's non-affinity queue 
 */
void addBgNodeMessage(char *msgPtr)
{
  tMYNODE->addBgNodeMessage(msgPtr);
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
#ifndef CMK_OPTIMIZE
    if (nodeInfo::Global2PE(gnodeID) != CmiMyPe())
      CmiAbort("msgHandlerFunc received wrong message!");
#endif
    int lnodeID = nodeInfo::Global2Local(gnodeID);
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
    if (nodeInfo::Global2PE(gnodeID) == CmiMyPe())
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

/* Converse handler for thread level broadcast message */
void threadBCastMsgHandlerFunc(char *msg)
{
  /* bgmsg is CmiMsgHeaderSizeBytes offset of original message pointer */
  int gnodeID = CmiBgMsgNodeID(msg);
  CmiInt2 threadID = CmiBgMsgThreadID(msg);
  int lnodeID;
  if (gnodeID < -1) {
      gnodeID = - (gnodeID+100);
      if (nodeInfo::Global2PE(gnodeID) == CmiMyPe())
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
  for (int i=0; i<cva(numNodes); i++)
  {
    for (int j=0; j<cva(numWth); j++) {
      if (i==lnodeID && j==threadID) continue;
      char *dupmsg = CmiCopyMsg(msg, len);
      CmiBgMsgNodeID(dupmsg) = nodeInfo::Local2Global(i);
      CmiBgMsgThreadID(dupmsg) = j;
      DEBUGF(("[%d] addBgNodeInbuffer to %d tid:%d\n", CmiMyPe(), i, j));
      addBgNodeInbuffer(dupmsg, i);
    }
  }
  CmiFree(msg);
}

/**
 *		BG Messaging Functions
 */

static inline double MSGTIME(int ox, int oy, int oz, int nx, int ny, int nz)
{
  int xd=ABS(ox-nx), yd=ABS(oy-ny), zd=ABS(oz-nz);
  int ncorners = 2;
  ncorners -= (xd?0:1 + yd?0:1 + zd?0:1);
  ncorners = (ncorners<0)?0:ncorners;
  return (ncorners*CYCLES_PER_CORNER + (xd+yd+zd)*CYCLES_PER_HOP)*CYCLE_TIME_FACTOR*1E-6;
}

void CmiSendPacket(int x, int y, int z, int msgSize,char *msg)
{
//  CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z),msgSize,(char *)msg);
#if !DELAY_SEND
  CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z), msgSize, msg);
#else
  if (!correctTimeLog)
      CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z), msgSize, msg);
#endif
}

/* send will copy data to msg buffer */
/* user data is not freeed in this routine, user can reuse the data ! */
void sendPacket_(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char* sendmsg, int local)
{
  CmiSetHandler(sendmsg, cva(msgHandler));
  CmiBgMsgNodeID(sendmsg) = nodeInfo::XYZ2Global(x,y,z);
  CmiBgMsgThreadID(sendmsg) = threadID;
  CmiBgMsgHandle(sendmsg) = handlerID;
  CmiBgMsgType(sendmsg) = type;
  CmiBgMsgLength(sendmsg) = numbytes;
  BgElapse(ALPHACOST);
  CmiBgMsgRecvTime(sendmsg) = MSGTIME(tMYX, tMYY, tMYZ, x,y,z) + BgGetTime();

  // timing
  BG_ADDMSG(sendmsg, CmiBgMsgNodeID(sendmsg), threadID, local);

  if (local)
    addBgNodeInbuffer(sendmsg, tMYNODEID);
  else
    CmiSendPacket(x, y, z, numbytes, sendmsg);
}

/* broadcast will copy data to msg buffer */
static inline void nodeBroadcastPacketExcept_(int node, CmiInt2 threadID, int handlerID, WorkType type, int numbytes, char* sendmsg)
{
  CmiSetHandler(sendmsg, cva(nBcastMsgHandler));	
  if (node >= 0)
    CmiBgMsgNodeID(sendmsg) = -node-100;
  else
    CmiBgMsgNodeID(sendmsg) = node;
  CmiBgMsgThreadID(sendmsg) = threadID;	
  CmiBgMsgHandle(sendmsg) = handlerID;	
  CmiBgMsgType(sendmsg) = type;	
  CmiBgMsgLength(sendmsg) = numbytes;
  /* FIXME */
  CmiBgMsgRecvTime(sendmsg) = BgGetTime();	

  // timing
  // FIXME
  BG_ADDMSG(sendmsg, CmiBgMsgNodeID(sendmsg), threadID, 0);

  DEBUGF(("[%d]CmiSyncBroadcastAllAndFree node: %d\n", BgMyNode(), node));
#if DELAY_SEND
  if (!correctTimeLog)
#endif
  CmiSyncBroadcastAllAndFree(numbytes,sendmsg);
}

/* broadcast will copy data to msg buffer */
static inline void threadBroadcastPacketExcept_(int node, CmiInt2 threadID, int handlerID, WorkType type, int numbytes, char* sendmsg)
{
  CmiSetHandler(sendmsg, cva(tBcastMsgHandler));	
  if (node >= 0)
    CmiBgMsgNodeID(sendmsg) = -node-100;
  else
    CmiBgMsgNodeID(sendmsg) = node;
  CmiBgMsgThreadID(sendmsg) = threadID;	
  CmiBgMsgHandle(sendmsg) = handlerID;	
  CmiBgMsgType(sendmsg) = type;	
  CmiBgMsgLength(sendmsg) = numbytes;
  /* FIXME */
  BgElapse(ALPHACOST);
  CmiBgMsgRecvTime(sendmsg) = BgGetTime();	

  // timing
#if 0
  if (node == BG_BROADCASTALL) {
    for (int i=0; i<bgSize; i++) {
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
  BG_ADDMSG(sendmsg, CmiBgMsgNodeID(sendmsg), threadID, 0);
#endif

  DEBUGF(("[%d]CmiSyncBroadcastAllAndFree node: %d tid:%d\n", BgMyNode(), node, threadID));
#if DELAY_SEND
  if (!correctTimeLog)
#endif
  CmiSyncBroadcastAllAndFree(numbytes,sendmsg);
}

/* sendPacket to route */
/* this function can be called by any thread */
void BgSendNonLocalPacket(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  if (x<0 || y<0 || z<0 || x>=cva(numX) || y>=cva(numY) || z>=cva(numZ)) {
    CmiPrintf("Trying to send packet to a nonexisting node: (%d %d %d)!\n", x,y,z);
    CmiAbort("Abort!\n");
  }

  sendPacket_(x, y, z, threadID, handlerID, type, numbytes, data, 0);
}

void BgSendLocalPacket(int threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  sendPacket_(tMYX, tMYY, tMYZ, threadID, handlerID, type, numbytes, data, 1);
}

/* wrapper of the previous two functions */
void BgSendPacket(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  if (tMYX == x && tMYY==y && tMYZ==z)
    BgSendLocalPacket(threadID, handlerID, type, numbytes, data);
  else
    BgSendNonLocalPacket(x,y,z,threadID,handlerID, type, numbytes, data);
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
  threadBroadcastPacketExcept_(node, threadID, handlerID, type, numbytes, data);
}

void BgThreadBroadcastAllPacket(int handlerID, WorkType type, int numbytes, char * data)
{
  threadBroadcastPacketExcept_(BG_BROADCASTALL, ANYTHREAD, handlerID, type, numbytes, data);
}

/*****************************************************************************
      BG node level API - utilities
*****************************************************************************/

/* must be called in a communication or worker thread */
void BgGetMyXYZ(int *x, int *y, int *z)
{
  ASSERT(!cva(inEmulatorInit));
  *x = tMYX; *y = tMYY; *z = tMYZ;
}

void BgGetXYZ(int seq, int *x, int *y, int *z)
{
  nodeInfo::Global2XYZ(seq, x, y, z);
}

void BgGetSize(int *sx, int *sy, int *sz)
{
  *sx = cva(numX); *sy = cva(numY); *sz = cva(numZ);
}

/* return the total number of Blue gene nodes */
int BgNumNodes()
{
  return bgSize;
}

/* can only called in emulatorinit */
void BgSetSize(int sx, int sy, int sz)
{
  ASSERT(cva(inEmulatorInit));
  cva(numX) = sx; cva(numY) = sy; cva(numZ) = sz;
}

/* return number of bg nodes on this emulator node */
int BgNodeSize()
{
  ASSERT(!cva(inEmulatorInit));
  return cva(numNodes);
}

/* return the bg node ID (local array index) */
int BgMyRank()
{
#ifndef CMK_OPTIMIZE
  if (tMYNODE == NULL) CmiAbort("Calling BgMyRank in the main thread!");
#endif
  ASSERT(!cva(inEmulatorInit));
  return tMYNODEID;
}

/* return my serialed blue gene node number */
int BgMyNode()
{
#ifndef CMK_OPTIMIZE
  if (tMYNODE == NULL) CmiAbort("Calling BgMyNode in the main thread!");
#endif
  return nodeInfo::XYZ2Global(tMYX, tMYY, tMYZ);
}

/* return a real processor number from a bg node */
int BgNodeToPE(int node)
{
  return nodeInfo::Global2PE(node);
}

int BgGetThreadID()
{
  ASSERT(tTHREADTYPE == WORK_THREAD || tTHREADTYPE == COMM_THREAD);
  return tMYID;
}

int BgGetGlobalThreadID()
{
  ASSERT(tTHREADTYPE == WORK_THREAD || tTHREADTYPE == COMM_THREAD);
  return nodeInfo::Local2Global(tMYNODE->id)*(cva(numCth)+cva(numWth))+tMYID;
//  return tMYGLOBALID;
}

int BgGetGlobalWorkerThreadID()
{
  ASSERT(tTHREADTYPE == WORK_THREAD);
  return nodeInfo::Local2Global(tMYNODE->id)*cva(numWth)+tMYID;
//  return tMYGLOBALID;
}

char *BgGetNodeData()
{
  return tUSERDATA;
}

void BgSetNodeData(char *data)
{
  ASSERT(!cva(inEmulatorInit));
  tUSERDATA = data;
}

int BgGetNumWorkThread()
{
  return cva(numWth);
}

void BgSetNumWorkThread(int num)
{
  ASSERT(cva(inEmulatorInit));
  cva(numWth) = num;
}

int BgGetNumCommThread()
{
  return cva(numCth);
}

void BgSetNumCommThread(int num)
{
  ASSERT(cva(inEmulatorInit));
  cva(numCth) = num;
}

/*****************************************************************************
      BG Timing Functions
*****************************************************************************/

static inline void startVTimer()
{
  if (timingMethod == BG_WALLTIME)
    tSTARTTIME = CmiWallTimer();
  else if (timingMethod == BG_ELAPSE)
    tSTARTTIME = tCURRTIME;
#ifdef CMK_ORIGIN2000
  else if (timingMethod == BG_COUNTER) {
    if (start_counters(0, 21) <0) {
      perror("start_counters");;
    }
    counterStarted = 1;
  }
#endif
}

static inline void stopVTimer()
{
  if (timingMethod == BG_WALLTIME) {
    tCURRTIME += (CmiWallTimer()-tSTARTTIME);
    tSTARTTIME = CmiWallTimer();
  }
  else if (timingMethod == BG_ELAPSE) {
    // if no bgelapse called, assume it takes 1us
    if (tCURRTIME-tSTARTTIME < 1E-9) {
//      tCURRTIME += 1e-6;
    }
  }
#ifdef CMK_ORIGIN2000
  else if (timingMethod == BG_COUNTER)  {
    long long c0, c1;
    if (read_counters(0, &c0, 21, &c1) < 0) perror("read_counters");
    tCURRTIME += Count2Time(c1);
    counterStarted = 0;
  }
#endif
}

double BgGetTime()
{
#if 1
  if (timingMethod == BG_WALLTIME) {
    /* accumulate time since last starttime, and reset starttime */
    double tp2= CmiWallTimer();
    tCURRTIME += (tp2 - tSTARTTIME);
    tSTARTTIME = tp2;
    return tCURRTIME;
  }
  else if (timingMethod == BG_ELAPSE) {
    return tCURRTIME;
  }
#ifdef CMK_ORIGIN2000
  else if (timingMethod == BG_COUNTER) {
    if (counterStarted) {
      long long c0, c1;
      if (read_counters(0, &c0, 21, &c1) <0) perror("read_counters");;
      tCURRTIME += Count2Time(c1);
      if (start_counters(0, 21)<0) perror("start_counters");;
    }
    return tCURRTIME;
  }
#endif
  else 
    CmiAbort("Unknown Timing Method.");
#else
  /* sometime I am interested in real wall time */
  tCURRTIME = CmiWallTimer();
  return tCURRTIME;
#endif
}

// moved to blue_logs.C
double BgGetCurTime()
{
  return tCURRTIME;
}

extern "C" 
void BgElapse(double t)
{
//  ASSERT(tTHREADTYPE == WORK_THREAD);
  if (timingMethod == BG_ELAPSE)
    tCURRTIME += t;
}

/*****************************************************************************
      Communication and Worker threads
*****************************************************************************/

BgStartHandler  workStartFunc = NULL;

void BgSetWorkerThreadStart(BgStartHandler f)
{
  workStartFunc = f;
}

#if 0
static void InitHandlerTable()
{
  /* init handlerTable */
  BGInitialize(int, handlerTableCount);
  BGAccess(handlerTableCount) = 1;
  BGInitialize(BgHandler*, handlerTable);
  BGAccess(handlerTable) = (BgHandler *)malloc(MAX_HANDLERS * sizeof(BgHandler));
  for (int i=0; i<MAX_HANDLERS; i++) BGAccess(handlerTable)[i] = defaultBgHandler;
}
#endif

static inline void ProcessMessage(char *msg)
{
  int handler = CmiBgMsgHandle(msg);
  DEBUGF(("[%d] call handler %d\n", BgMyNode(), handler));

  BgHandlerInfo *handInfo;
  BgHandlerEx entryFunc;
#if  CMK_BLUEGENE_NODE
  handInfo = tMYNODE->handlerTable.getHandle(handler);
#else
  handInfo = tHANDLETAB.getHandle(handler);
  if (handInfo == NULL) handInfo = tMYNODE->handlerTable.getHandle(handler);
#endif

  if (handInfo == NULL) {
    CmiPrintf("[%d] invalid handler: %d. \n", tMYNODEID, handler);
    CmiAbort("");
  }
  entryFunc = handInfo->fnPtr;

  CmiSetHandler(msg, CmiBgMsgHandle(msg));

  // don't count thread overhead and timing overhead
  startVTimer();

  entryFunc(msg, handInfo->userPtr);

  stopVTimer();
}

void correctMsgTime(char *msg);

void threadInfo::run_comm_thread()
{
  tSTARTTIME = CmiWallTimer();

  if (!tSTARTED) {
    tSTARTED = 1;
//    InitHandlerTable();
    BgNodeStart(arg_argc, arg_argv);
    /* bnv should be initialized */
  }

  threadQueue *commQ = myNode->commThQ;

  for (;;) {
    char *msg = getFullBuffer();
    if (!msg) { 
//      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      commQ->enq(CthSelf());
      DEBUGF(("[%d] comm thread suspend.\n", BgMyNode()));
      CthSuspend(); 
      DEBUGF(("[%d] comm thread assume.\n", BgMyNode()));
//      tSTARTTIME = CmiWallTimer();
      continue;
    }
    DEBUGF(("[%d] comm thread has a msg.\n", BgMyNode()));
    /* schedule a worker thread, if small work do it itself */
    if (CmiBgMsgType(msg) == SMALL_WORK) {
      if (CmiBgMsgRecvTime(msg) > tCURRTIME)  tCURRTIME = CmiBgMsgRecvTime(msg);
//      tSTARTTIME = CmiWallTimer();
      /* call user registered handler function */
      ProcessMessage(msg);
    }
    else {
#if BLUEGENE_TIMING
      correctMsgTime(msg);
#endif
      if (CmiBgMsgThreadID(msg) == ANYTHREAD) {
        DEBUGF(("anythread, call addBgNodeMessage\n"));
        addBgNodeMessage(msg);			/* non-affinity message */
      }
      else {
        DEBUGF(("[N%d] affinity msg, call addBgThreadMessage to tID:%d\n", 
			BgMyNode(), CmiBgMsgThreadID(msg)));
        addBgThreadMessage(msg, CmiBgMsgThreadID(msg));
      }
    }
    /* let other communication thread do their jobs */
//    tCURRTIME += (CmiWallTimer()-tSTARTTIME);
#if !SCHEDULE_WORK
    CthYield();
#endif
    tSTARTTIME = CmiWallTimer();
  }
}


void scheduleWorkerThread(char *msg)
{
  CthThread tid = (CthThread)msg;
//CmiPrintf("scheduleWorkerThread %p\n", tid);
  CthAwaken(tid);
}

void threadInfo::run_work_thread()
{
  tSTARTTIME = CmiWallTimer();

//  InitHandlerTable();
  if (workStartFunc) {
    DEBUGF(("[N%d] work thread %d start.\n", BgMyNode(), id));
    // timing
    startVTimer();
    BG_ENTRYSTART((char*)NULL);
    char **Cmi_argvcopy = CmiCopyArgs(arg_argv);
    workStartFunc(arg_argc, Cmi_argvcopy);
    BG_ENTRYEND();
    stopVTimer();
  }

  ckMsgQueue &q1 = myNode->nodeQ;
  ckMsgQueue &q2 = myNode->affinityQ[id];

  for (;;) {
    char *msg=NULL;
    int e1 = q1.isEmpty();
    int e2 = q2.isEmpty();
    int fromQ2 = 0;		// delay the deq of msg from affinity queue

    // not deq from nodeQ assuming no interrupt in the handler
    if (e1 && !e2) { msg = q2[0]; fromQ2 = 1;}
//    else if (e2 && !e1) { msg = q1.deq(); }
    else if (e2 && !e1) { msg = q1[0]; }
    else if (!e1 && !e2) {
      if (CmiBgMsgRecvTime(q1[0]) < CmiBgMsgRecvTime(q2[0])) {
//        msg = q1.deq();
        msg = q1[0];
      }
      else {
        msg = q2[0];
        fromQ2 = 1;
      }
    }
    /* if no msg is ready, put it to sleep */
    if ( msg == NULL ) {
//      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      CthSuspend();
      DEBUGF(("[N%d] work thread T%d awakened.\n", BgMyNode(), id));
      continue;
    }
#if BLUEGENE_TIMING
    correctMsgTime(msg);
#if THROTTLE_WORK
    if (correctTimeLog) {
      if (CmiBgMsgRecvTime(msg) > gvt+ LEASH) {
	double nextT = CmiBgMsgRecvTime(msg);
	unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
// CmiPrintf("Thread %d YieldPrio: %g gvt: %g leash: %g\n", id, nextT, gvt, LEASH);
	CthYieldPrio(CQS_QUEUEING_IFIFO, sizeof(int), &prio);
	continue;
      }
    }
#endif
#endif   /* TIMING */
    DEBUGF(("[N%d] work thread T%d has a msg.\n", BgMyNode(), id));

//if (tMYNODEID==0)
//CmiPrintf("[%d] recvT: %e\n", tMYNODEID, CmiBgMsgRecvTime(msg));

    if (CmiBgMsgRecvTime(msg) > currTime) {
      tCURRTIME = CmiBgMsgRecvTime(msg);
    }

    BG_ENTRYSTART(msg);
    // ProcessMessage may trap into scheduler
    ProcessMessage(msg);
    BG_ENTRYEND();

    // counter of processed real mesgs
    stateCounters.realMsgProcCnt++;

    if (fromQ2 == 1) q2.deq();
    else q1.deq();

    DEBUGF(("[N%d] work thread T%d finish a msg.\n", BgMyNode(), id));

    /* let other work thread do their jobs */
#if SCHEDULE_WORK
    DEBUGF(("[N%d] work thread T%d suspend when done - %d to go.\n", BgMyNode(), tMYID, q2.length()));
    CthSuspend();
    DEBUGF(("[N%d] work thread T%d awakened here.\n", BgMyNode(), id));
#else
    CthYield();
#endif
  }
}

// comm thread entry
void comm_thread(threadInfo *tinfo)
{
  /* set the thread-private threadinfo */
  cta(threadinfo) = tinfo;
  tinfo->run_comm_thread();
}

// worker thread entry
void work_thread(threadInfo *tinfo)
{
  cta(threadinfo) = tinfo;
  tinfo->run_work_thread();
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
  for (i=0; i< cva(numWth); i++)
  {
    threadInfo *tinfo = ninfo->threadinfo[i];
    t = CthCreate((CthVoidFn)work_thread, tinfo, 0);
    if (t == NULL) CmiAbort("BG> Failed to create worker thread. \n");
    tinfo->setThread(t);
    /* put to thread table */
    tTHREADTABLE[tinfo->id] = t;
    CthAwaken(t);
  }

  /* creat communication thread */
  for (i=0; i< cva(numCth); i++)
  {
    threadInfo *tinfo = ninfo->threadinfo[i+cva(numWth)];
    t = CthCreate((CthVoidFn)comm_thread, tinfo, 0);
    if (t == NULL) CmiAbort("BG> Failed to create communication thread. \n");
    tinfo->setThread(t);
    /* put to thread table */
    tTHREADTABLE[tinfo->id] = t;
    CthAwaken(t);
  }

}

static void beginExitHandlerFunc(void *msg);

CmiHandler exitHandlerFunc(char *msg)
{
  // TODO: free memory before exit
  int i,j;

  programExit = 2;
#if BLUEGENE_TIMING
  // timing
  if (0)	// detail
  if (genTimeLog) {
    for (j=0; j<cva(numNodes); j++)
    for (i=0; i<cva(numWth); i++) {
      BgTimeLine &log = cva(nodeinfo)[j].timelines[i].timeline;	
//      BgPrintThreadTimeLine(nodeInfo::Local2Global(j), i, log);
      int x,y,z;
      nodeInfo::Local2XYZ(j, &x, &y, &z);
      BgWriteThreadTimeLine(arg_argv, x, y, z, i, log);
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
    for (i=0; i<cva(numWth); i++) {
      int oldPe = CmiSwitchToPE(nodeInfo::Local2Global(j)*cva(numWth)+i);
      if (origPe == -2) origPe = oldPe;
      traceCharmClose();
//      CmiSwitchToPE(oldPe);
    }
    if (origPe!=-2) CmiSwitchToPE(origPe);
  }

#if 0
  delete [] cva(nodeinfo);
  delete [] cva(inBuffer);
  for (i=0; i<cva(numNodes); i++) CmmFree(cva(msgBuffer)[i]);
  delete [] cva(msgBuffer);
#endif

  //ConverseExit();
  if (genTimeLog)
    { if (CmiMyPe() != 0) CsdExitScheduler(); }
  else
    CsdExitScheduler();

  //if (CmiMyPe() == 0) CmiPrintf("\nBG> BlueGene emulator shutdown gracefully!\n");

  return 0;
}

CmiStartFn bgMain(int argc, char **argv)
{
  int i;

  /* initialize all processor level data */
  CpvInitialize(int,numX);
  CpvInitialize(int,numY);
  CpvInitialize(int,numZ);
  CpvInitialize(int,numCth);
  CpvInitialize(int,numWth);
  cva(numX) = cva(numY) = cva(numZ) = 0;
  cva(numCth) = cva(numWth) = 0;

  CmiArgGroup("Charm++","BlueGene Simulator");
  CmiGetArgIntDesc(argv, "+x", &cva(numX), "The x size of the grid of nodes");
  CmiGetArgIntDesc(argv, "+y", &cva(numY), "The y size of the grid of nodes");
  CmiGetArgIntDesc(argv, "+z", &cva(numZ), "The z size of the grid of nodes");
  CmiGetArgIntDesc(argv, "+cth", &cva(numCth), "The number of simulated communication threads per node");
  CmiGetArgIntDesc(argv, "+wth", &cva(numWth), "The number of simulated worker threads per node");

  genTimeLog = CmiGetArgFlagDesc(argv, "+bglog", "Write events to log file");
  correctTimeLog = CmiGetArgFlagDesc(argv, "+bgcorrect", "Apply timestamp correction to logs");
  if (correctTimeLog) genTimeLog = 1;

  // for timing method, default using elapse calls.
  timingMethod = BG_ELAPSE;
  if(CmiGetArgFlagDesc(argv, "+bgwalltime", 
                       "Use walltime, not estimated time, for time estimate")) 
      timingMethod = BG_WALLTIME;
#ifdef CMK_ORIGIN2000
  if(CmiGetArgFlagDesc(argv, "+bgcounter", "Use performance counter")) 
      timingMethod = BG_COUNTER;
#endif
  
  bgcorroff = 0;
  if(CmiGetArgFlagDesc(argv, "+bgcorroff", "Start with correction off")) 
    bgcorroff = 1;

  bgstats=0;
  if(CmiGetArgFlagDesc(argv, "+bgstats", "Print correction statistics")) 
    bgstats = 1;

#if BLUEGENE_DEBUG_LOG
  {
    char ln[200];
    sprintf(ln,"bgdebugLog.%d",CmiMyPe());
    bgDebugLog=fopen(ln,"w");
  }
#endif

  arg_argv = argv;
  arg_argc = CmiGetArgc(argv);

  /* msg handler */
  CpvInitialize(int,msgHandler);
  cva(msgHandler) = CmiRegisterHandler((CmiHandler) msgHandlerFunc);
  CpvInitialize(int,nBcastMsgHandler);
  cva(nBcastMsgHandler) = CmiRegisterHandler((CmiHandler)nodeBCastMsgHandlerFunc);
  CpvInitialize(int,tBcastMsgHandler);
  cva(tBcastMsgHandler) = CmiRegisterHandler((CmiHandler)threadBCastMsgHandlerFunc);

  CpvInitialize(int,exitHandler);
  cva(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);

  CpvInitialize(int,beginExitHandler);
  cva(beginExitHandler) = CmiRegisterHandler((CmiHandler) beginExitHandlerFunc);

  CpvInitialize(int, inEmulatorInit);
  cva(inEmulatorInit) = 1;
  /* call user defined BgEmulatorInit */
  BgEmulatorInit(arg_argc, arg_argv);
  cva(inEmulatorInit) = 0;

  /* check if all bluegene node size and thread information are set */
  BGARGSCHECK;

  timerFunc = BgGetTime;

  BgInitTiming();		// timing module

  if (CmiMyPe() == 0) {
    CmiPrintf("BG info> Simulating %dx%dx%d nodes with %d comm + %d work threads each.\n", cva(numX), cva(numY), cva(numZ), cva(numCth), cva(numWth));
    if (timingMethod == BG_ELAPSE) 
      CmiPrintf("BG info> Using BgElapse calls for timing method. \n");
    else if (timingMethod == BG_WALLTIME)
      CmiPrintf("BG info> Using WallTimer for timing method. \n");
    else if (timingMethod == BG_COUNTER)
      CmiPrintf("BG info> Using performance counter for timing method. \n");
    if (genTimeLog)
      CmiPrintf("BG info> Generating timing log. \n");
    if (correctTimeLog)
      CmiPrintf("BG info> Perform timing log correction. \n");
  }

  bgSize = cva(numX)*cva(numY)*cva(numZ);

  CtvInitialize(threadInfo *, threadinfo);

  /* number of bg nodes on this PE */
  CpvInitialize(int, numNodes);
  cva(numNodes) = nodeInfo::numLocalNodes();

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
    ninfo->id = i;
    nodeInfo::Local2XYZ(i, &ninfo->x, &ninfo->y, &ninfo->z);

    /* pretend that I am a thread */
    cta(threadinfo)->myNode = ninfo;

    /* initialize a BG node and fire all threads */
    BgNodeInitialize(ninfo);
  }
  // clear main thread.
  cta(threadinfo)->myNode = NULL;
  CpvInitialize(CthThread, mainThread);
  cva(mainThread) = CthSelf();

  return 0;
}

// for conv-conds:
// if -2 untouch
// if -1 main thread
#undef CmiSwitchToPE
#if CMK_BLUEGENE_THREAD
extern "C" int CmiSwitchToPE(int pe)
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
    int t = pe%cva(numWth);
    int newpe = nodeInfo::Global2Local(pe/cva(numWth));
    nodeInfo *ninfo = cva(nodeinfo) + newpe;;
    threadInfo *tinfo = ninfo->threadinfo[t];
    CthSwitchThread(tinfo->me);
  }
  return oldpe;
}
#else
extern "C" int CmiSwitchToPE(int pe)
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
static inline char* searchInAffinityQueue(int nodeidx, int msgID, int srcnode, CmiInt2 tID, int &index)
{
  CmiAssert(tID != ANYTHREAD);
  ckMsgQueue &affinityQ = cva(nodeinfo)[nodeidx].affinityQ[tID];
  for (int i=0; i<affinityQ.length(); i++)  {
      char *msg = affinityQ[i];
      int m_msgID = CmiBgMsgID(msg);
      int m_srcnode = CmiBgMsgSrcPe(msg);
      if (msgID == m_msgID && srcnode == m_srcnode) {
        index = i;
        return msg;
      }
  }
  return NULL;
}

// return the msg pointer, thread id and the index of the message in the affinity queue.
static char* searchInAffinityQueueInNode(int nodeidx, int msgID, int srcnode, CmiInt2 &tID, int &index)
{
  for (tID=0; tID<cva(numWth); tID++) {
    char *msg = searchInAffinityQueue(nodeidx, msgID, srcnode, tID, index);
    if (msg) return msg;
  }
  return NULL;
}

int updateRealMsgs(bgCorrectionMsg *cm, int nodeidx)
{
  char *msg;
  CmiInt2 tID = cm->tID;
  int index;
  if (tID == ANYTHREAD) {
    msg = searchInAffinityQueueInNode(nodeidx, cm->msgID, cm->srcNode, tID, index);
  }
  else {
    msg = searchInAffinityQueue(nodeidx, cm->msgID, cm->srcNode, tID, index);
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

#if 0
  ckMsgQueue &affinityQ = cva(nodeinfo)[nodeidx].affinityQ[cm->tID];
  for (int i=0; i<affinityQ.length(); i++)  {
    char *msg = affinityQ[i];
//    if (CkMsgDoCorrect(msg) == 0) return 0;
    int msgID = CmiBgMsgID(msg);
    int srcnode = CmiBgMsgSrcPe(msg);
    if (msgID == cm->msgID && srcnode == cm->srcNode) {
	CmiBgMsgRecvTime(msg) = cm->tAdjust;
        affinityQ.update(i);
        CthThread tid = cva(nodeinfo)[nodeidx].threadTable[cm->tID];
  	unsigned int prio = (unsigned int)(cm->tAdjust*PRIO_FACTOR)+1;
        CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
        stateCounters.corrMsgCRCnt++;
	return 1;       /* invalidate this msg */
    }
  }
  return 0;
}
#endif

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
  double scale = 1000.0;	// scale to ms

  //We measure from 1ms to 5001 ms in steps of 100 ms
  int min = 0, max = HISTOGRAM_SIZE, step = 1;

  int size = (max-min)/step;
  CmiAssert(size == HISTOGRAM_SIZE);
  for(int i=0;i<size;i++) array[i] = 0;

  for (int nodeidx=0; nodeidx<cva(numNodes); nodeidx++) {
    BgTimeLineRec *tlinerec = cva(nodeinfo)[nodeidx].timelines;
    for (int tID=0; tID<cva(numWth); tID++) {
      int util = (int)(scale*(tlinerec[tID].computeUtil(nReal)));

      if (util >= max) util=max-1;
      array[(util-min)/step]++;
    }
  }
}

static void sendCorrectionStats()
{
  int msgSize = sizeof(StatsMessage)+sizeof(int)*HISTOGRAM_SIZE;
  StatsMessage *statsMsg = (StatsMessage *)CmiAlloc(msgSize);
  statsMsg->processCount = processCount;
  statsMsg->corrMsgCount = corrMsgCount;
  int numMsgs=0;
  int maxTimelineLen=-1, minTimelineLen=CMK_MAXINT;
  int totalMem = 0;
  if (bgstats) {
  for (int nodeidx=0; nodeidx<cva(numNodes); nodeidx++) {
    BgTimeLineRec *tlines = cva(nodeinfo)[nodeidx].timelines;
    for (int tID=0; tID<cva(numWth); tID++) {
        BgTimeLineRec &tlinerec = tlines[tID];
	int tlen = tlinerec.length();
	if (tlen>maxTimelineLen) maxTimelineLen=tlen;
	if (tlen<minTimelineLen) minTimelineLen=tlen;
        totalMem = tlen*sizeof(bgTimeLog);
//CmiPrintf("[%d node:%d] bgTimeLog: %dK len:%d size of bglog: %d bytes\n", CmiMyPe(), nodeidx, totalMem/1000, tlen, sizeof(bgTimeLog));
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

  CmiSetHandler(statsMsg, cva(bgStatCollectHandler));
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
    if (bgstats) {
      CmiPrintf("Total procCount:%d corrMsgCount:%d realMsg:%d timeline:%d-%d\n", pc, cc, realMsgCount, minTimelineLen, maxTimelineLen);
      for (i=0; i<HISTOGRAM_SIZE; i++) CmiPrintf("%d ", histArray[i]);
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

   int msgID = CmiBgMsgID(msg);
   int srcnode = CmiBgMsgSrcPe(msg);
   CmiInt2 tid = CmiBgMsgThreadID(msg);

   bgCorrectionQ &cmsg = cva(nodeinfo)[tMYNODEID].cmsg;
   int len = cmsg.length();
   for (int i=0; i<len; i++) {
     bgCorrectionMsg* m = cmsg[i];
     if (msgID == m->msgID && srcnode == m->srcNode && tid == m->tID) {
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


//TODO: Right now works only for BG/L
static void writeToDisk()
{

  char* d = new char[10];
  //Num of simulated procs on this real pe
  int numProcs = cva(numNodes)*cva(numWth);

  const PUP::machineInfo &machInfo = PUP::machineInfo::current();

  // write summary file on PE0
  if(CmiMyPe()==0){
    
    FILE *f2 = fopen("bgTrace","w");
    //Total real and toal BG processors
    int numPes=CmiNumPes();
    int totalProcs = BgNumNodes()*cva(numWth);

    if(f2==NULL)
      CmiPrintf("Creating bgTrace failed\n");
//    PUP::toDisk p(f2);
    PUP::toDisk p(f2);
    p((char *)&machInfo, sizeof(machInfo));
    p|totalProcs;
    p|cva(numX); p|cva(numY); p|cva(numZ);
    p|cva(numCth);p|cva(numWth);
    p|numPes;
    
    CmiPrintf("[0] Number is numX:%d numY:%d numZ:%d numCth:%d numWth:%d numPes:%d totalProcs:%d\n",cva(numX),cva(numY),cva(numZ),cva(numCth),cva(numWth),numPes,totalProcs);
    
    fclose(f2);
  }
  
  CmiPrintf("seq correct called on %d \n",CmiMyPe());
  sprintf(d,"bgTrace%d",CmiMyPe());
  FILE *f = fopen(d,"w");
 
  int *procOffsets = new int[numProcs];
  if(f==NULL)
    CmiPrintf("Creating bgTrace%d failed\n",CmiMyPe());
  PUP::toDisk p(f);
  
  p((char *)&machInfo, sizeof(machInfo));	// machine info
  p|numProcs;

  // CmiPrintf("Timelines are: \n");
  int procTablePos = ftell(f);
  int procTableSize = (numProcs)*sizeof(int);
  fseek(f,procTableSize,SEEK_CUR); 

  for (int j=0; j<cva(numNodes); j++){
    for(int i=0;i<cva(numWth);i++){
    BgTimeLineRec &t = cva(nodeinfo)[j].timelines[i];
    procOffsets[j*cva(numWth) + i] = ftell(f);
    //  CmiPrintf("Timeline %d is has offset: %d\n",j,procOffsets[j]);
    /*    CmiPrintf("\nTimeline j is\n");
    for(int i=0;i<t.length();i++)
      CmiPrintf("\tName: %s",t[i]->name);
    */
    // t[i]->print(-1,-1);
    
    t.pup(p);
    }
  }
  
  fseek(f,procTablePos,SEEK_SET);
  p(procOffsets,numProcs);
  fclose(f);

  CmiPrintf("[%d] Wrote to disk for BG node:%d work:%d \n", CmiMyPe(), cva(numNodes),cva(numWth));
}


