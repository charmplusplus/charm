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

#define  DEBUGF(x)     // CmiPrintf x;

#include "blue.h"

#include "blue_impl.h"    	// implementation header file
#include "blue_timing.h" 	// timing module

template<class T> class bgQueue;

typedef bgQueue<int>  	    threadIDQueue;
typedef bgQueue<CthThread>  threadQueue;
typedef bgQueue<char *>     msgQueue;
//typedef CkQ<char *> 	    ckMsgQueue;
// use a queue sorted by recv time
typedef minMsgHeap 	    ckMsgQueue;
typedef CkQ<bgCorrectionMsg *> 	    bgCorrectionQ;

class nodeInfo;
class threadInfo;

/* node level variables */
CpvDeclare(nodeInfo*, nodeinfo);		/* represent a bluegene node */

/* thread level variables */
CtvDeclare(threadInfo *, threadinfo);	/* represent a bluegene thread */

CpvStaticDeclare(CthThread, mainThread);

/* emulator node level variables */
CpvDeclare(int,msgHandler);
CpvDeclare(int,nBcastMsgHandler);
CpvDeclare(int,tBcastMsgHandler);
CpvDeclare(int,bgCorrectionHandler);
CpvDeclare(int,exitHandler);

CmiHandler msgHandlerFunc(char *msg);
CmiHandler nodeBCastMsgHandlerFunc(char *msg);
CmiHandler threadBCastMsgHandlerFunc(char *msg);

extern "C" void defaultBgHandler(char *);

CpvStaticDeclare(msgQueue *,inBuffer);	/* emulate the bluegene fix-size inbuffer */
CpvStaticDeclare(CmmTable *,msgBuffer);	/* if inBuffer is full, put to this buffer */

CpvDeclare(int, inEmulatorInit);
static int printTimeLog;

#define ASSERT(x)	if (!(x)) { CmiPrintf("Assert failure at %s:%d\n", __FILE__,__LINE__); CmiAbort("Abort!"); }

#define BGARGSCHECK   	\
  if (cva(numX)==0 || cva(numY)==0 || cva(numZ)==0)  { CmiPrintf("\nMissing parameters for BlueGene machine size!\n<tip> use command line options: +x, +y, or +z.\n"); BgShutdown(); } \
  if (cva(numCth)==0 || cva(numWth)==0) { CmiAbort("\nMissing parameters for number of communication/worker threads!\n<tip> use command line options: +cth or +wth.\n"); BgShutdown(); }	\
  if (cva(numX)*cva(numY)*cva(numZ)<CmiNumPes()) {	\
    CmiAbort("\nToo few BlueGene nodes!\n");	\
    BgShutdown(); 	\
  }


/*****************************************************************************
   used internally, define Queue for scheduler and fixed size msgqueue
*****************************************************************************/

/* scheduler queue */
template <class T>
class bgQueue {
private:
  T *data;
  int fp, count, size;
public:
  bgQueue(): data(NULL), fp(0), count(0) {};
  ~bgQueue() { delete[] data; }
  inline void initialize(int max) {  size = max; data = new T[max]; }
  T deq() {
      if (count > 0) {
        T ret = data[fp];
        fp = (fp+1)%size;
        count --;
        return ret;
      }
      else
        return 0;
  }
  void enq(T item) {
      ASSERT(count < size);
      data[(fp+count)%size] = item;
      count ++;
  }
  inline int isFull() { return count == size; }
  inline int isEmpty() { return count == 0; }
};

/**
  definition of Handler Table;
  there are two kinds of handle tables: 
  one is node level, the other is at thread level
*/
class HandlerTable {
public:
  int          handlerTableCount; 
  BgHandler *  handlerTable;     
public:
  HandlerTable()
  {
    handlerTableCount = 1;
    handlerTable = (BgHandler *)malloc(MAX_HANDLERS * sizeof(BgHandler));
    for (int i=0; i<MAX_HANDLERS; i++) handlerTable[i] = defaultBgHandler;
  }
  int registerHandler(BgHandler h)
  {
    ASSERT(!cva(inEmulatorInit));
    /* leave 0 as blank, so it can report error luckily */
    int cur = handlerTableCount++;
    if (cur >= MAX_HANDLERS)
      CmiAbort("BG> HandlerID exceed the maximum.\n");
    handlerTable[cur] = h;
    return cur;
  }
  void numberHandler(int idx, BgHandler h)
  {
    ASSERT(!cva(inEmulatorInit));
    if (idx >= handlerTableCount || idx < 1)
      CmiAbort("BG> HandlerID exceed the maximum!\n");
    handlerTable[idx] = h;
  }
  BgHandler getHandle(int handler)
  {
#if 0
    if (handler >= handlerTableCount) {
      CmiPrintf("[%d] handler: %d handlerTableCount:%d. \n", tMYNODEID, handler, handlerTableCount);
      CmiAbort("Invalid handler!");
    }
#endif
    if (handler >= handlerTableCount) return NULL;
    return handlerTable[handler];
  }
};

/*****************************************************************************
      NodeInfo:
        including a group of functions defining the mapping, terms used here:
        XYZ: (x,y,z)
        Global:  map (x,y,z) to a global serial number
        Local:   local index of this nodeinfo in the emulator's node 
*****************************************************************************/
class BlockMapInfo;
class CyclicMapInfo;

class nodeInfo: public CyclicMapInfo  {
public:
  int id;
  int x,y,z;
  threadQueue *commThQ;		/* suspended comm threads queue */
  CthThread   *threadTable;	/* thread table for both work and comm threads*/
  threadInfo  **threadinfo;
  ckMsgQueue   nodeQ;		/* non-affinity msg queue */
  ckMsgQueue  *affinityQ;	/* affinity msg queue for each work thread */
  char        *udata;		/* node specific data pointer */
  double       startTime;	/* start time for a thread */
  char         started;		/* flag indicate if this node is started */
 
  HandlerTable handlerTable; /* node level handler table */

  // for timing
  BgTimeLine *timelines;
  bgCorrectionQ cmsg;

public:
  nodeInfo();

  ~nodeInfo() {
    if (commThQ) delete commThQ;
    delete [] affinityQ;
    delete [] threadTable;
    delete [] threadinfo;
  }
  
};	// end of nodeInfo

/*****************************************************************************
      ThreadInfo:  each thread has a thread private threadInfo structure.
      It has a local id, a global serial id. 
      myNode: point to the nodeInfo it belongs to.
      currTime: is the elapse time for this thread;
      me:   point to the CthThread converse thread handler.
*****************************************************************************/

class threadInfo {
public:
  short id;
//  int globalId;
  ThreadType  type;		/* worker or communication thread */
  CthThread me;			/* Converse thread handler */
  nodeInfo *myNode;		/* the node belonged to */
  double  currTime;		/* thread timer */

#if  CMK_BLUEGENE_THREAD
  HandlerTable   handlerTable;      /* thread level handler table */
#endif

public:
  threadInfo(int _id, ThreadType _type, nodeInfo *_node): id(_id), type(_type), myNode(_node) {
    currTime=0.0;
//    if (id != -1) globalId = nodeInfo::Local2Global(_node->id)*(cva(numCth)+cva(numWth))+_id;
  }
  inline void setThread(CthThread t) { me = t; }
  inline CthThread getThread() { return me; }
}; 


/**
  nodeInfo construtor
*/
nodeInfo::nodeInfo(): udata(NULL), started(0) 
{
    int i;
    commThQ = new threadQueue;
    commThQ->initialize(cva(numCth));

    threadTable = new CthThread[cva(numWth)+cva(numCth)];
    threadinfo = new (threadInfo*)[cva(numWth)+cva(numCth)];

    affinityQ = new ckMsgQueue[cva(numWth)];

    // create threadinfo
    for (i=0; i< cva(numWth); i++)
    {
      threadinfo[i] = new threadInfo(i, WORK_THREAD, this);
    }
    for (i=0; i< cva(numCth); i++)
    {
      threadinfo[i+cva(numWth)] = new threadInfo(i+cva(numWth), COMM_THREAD, this);
    }

    // timing
    timelines = new BgTimeLine[cva(numWth)];
  }

/*****************************************************************************
      low level API
*****************************************************************************/

extern "C" void defaultBgHandler(char *null)
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
  return tMYNODE->handlerTable.numberHandler(idx,h);
#else
  if (tTHREADTYPE == COMM_THREAD) {
    return tMYNODE->handlerTable.numberHandler(idx, h);
  }
  else {
    return tHANDLETAB.numberHandler(idx, h);
  }
#endif
}

/* communication thread call getFullBuffer to test if there is data ready 
   in INBUFFER for its own queue */
char * getFullBuffer()
{
  int tags[1], ret_tags[1];
  char *data, *mb;

  /* I must be a communication thread */
  if (tTHREADTYPE != COMM_THREAD) 
    CmiAbort("GetFullBuffer called by a non-communication thread!\n");

  /* see if we have msg in inBuffer */
  if (tINBUFFER.isEmpty()) return NULL;
  data = tINBUFFER.deq(); 
  /* since we have just delete one from inbuffer, fill one from msgbuffer */
  tags[0] = CmmWildCard;
  mb = (char *)CmmGet(tMSGBUFFER, 1, tags, ret_tags);
  if (mb) tINBUFFER.enq(mb);

  return data;
}

/* add message msgPtr to a bluegene node's inbuffer queue */
void addBgNodeInbuffer(char *msgPtr, int nodeID)
{
  int tags[1];

#ifndef CMK_OPTIMIZE
  if (nodeID >= cva(numNodes)) CmiAbort("NodeID is out of range!");
#endif
  /* if inbuffer is full, store in the msgbuffer */
  if (cva(inBuffer)[nodeID].isFull()) {
    tags[0] = nodeID;
    CmmPut(cva(msgBuffer)[nodeID], 1, tags, msgPtr);
  }
  else {
    DEBUGF(("inbuffer is not full.\n"));
    cva(inBuffer)[nodeID].enq(msgPtr);
  }
  /* awake a communication thread to schedule it */
  CthThread t=cva(nodeinfo)[nodeID].commThQ->deq();
  DEBUGF(("activate communication thread: %p.\n", t));
  if (t) CthAwaken(t);
}

/* add a message to a thread's affinity queue */
void addBgThreadMessage(char *msgPtr, int threadID)
{
#ifndef CMK_OPTIMIZE
  if (threadID >= cva(numWth)) CmiAbort("ThreadID is out of range!");
#endif
  ckMsgQueue &que = tMYNODE->affinityQ[threadID];
  que.enq(msgPtr);
  if (que.length() == 1)
    CthAwaken(tTHREADTABLE[threadID]);
}

/* add a message to a node's non-affinity queue */
void addBgNodeMessage(char *msgPtr)
{
  /* find a idle worker thread */
  /* FIXME:  flat search is bad if there is many work threads */
  for (int i=0; i<cva(numWth); i++)
    if (tMYNODE->affinityQ[i].length() == 0)
    {
      /* this work thread is idle, schedule the msg here */
      DEBUGF(("activate a work thread %d - %p.\n", i, tTHREADTABLE[i]));
      tMYNODE->affinityQ[i].enq(msgPtr);
      CthAwaken(tTHREADTABLE[i]);
      return;
    }
  /* all worker threads are busy */   
  DEBUGF(("all work threads are busy.\n"));
  tNODEQ.enq(msgPtr);
}

int checkReady()
{
  if (tTHREADTYPE != COMM_THREAD)
    CmiAbort("checkReady called by a non-communication thread!\n");
  return !tINBUFFER.isEmpty();
}

void sendPacket(int x, int y, int z, int msgSize,char *msg)
{
  CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z),msgSize,(char *)msg);
}

/* handler to process the msg */
CmiHandler msgHandlerFunc(char *msg)
{
  /* bgmsg is CmiMsgHeaderSizeBytes offset of original message pointer */
  int nodeID = CmiBgMsgNodeID(msg);
  if (nodeID >= 0) {
#ifndef CMK_OPTIMIZE
    if (nodeInfo::Global2PE(nodeID) != CmiMyPe())
      CmiAbort("msgHandlerFunc received wrong message!");
#endif
    nodeID = nodeInfo::Global2Local(nodeID);
    addBgNodeInbuffer(msg, nodeID);
  }
  else {
    CmiAbort("Invalid message!");
  }
  return 0;
}

CmiHandler nodeBCastMsgHandlerFunc(char *msg)
{
  /* bgmsg is CmiMsgHeaderSizeBytes offset of original message pointer */
  int nodeID = CmiBgMsgNodeID(msg);
  if (nodeID < -1) {
      nodeID = - (nodeID+100);
      if (nodeInfo::Global2PE(nodeID) == CmiMyPe())
	nodeID = nodeInfo::Global2Local(nodeID);
      else
	nodeID = -1;
  }
  CmiUInt2 threadID = CmiBgMsgThreadID(msg);
  // broadcast except nodeID:threadId
  int len = CmiBgMsgLength(msg);
  int count = 0;
  for (int i=0; i<cva(numNodes); i++)
  {
    if (i==nodeID) continue;
    char *dupmsg;
    if (count == 0) dupmsg = msg;
    else {
      dupmsg = (char *)CmiAlloc(len);
      memcpy(dupmsg, msg, len);
    }
    DEBUGF(("[%d] addBgNodeInbuffer to %d\n", BgMyNode(), i));
    addBgNodeInbuffer(dupmsg, i);
    count ++;
  }
  return 0;
}

CmiHandler threadBCastMsgHandlerFunc(char *msg)
{
  /* bgmsg is CmiMsgHeaderSizeBytes offset of original message pointer */
  int nodeID = CmiBgMsgNodeID(msg);
  if (nodeID < -1) {
      nodeID = - (nodeID+100);
      if (nodeInfo::Global2PE(nodeID) == CmiMyPe())
	nodeID = nodeInfo::Global2Local(nodeID);
      else
	nodeID = -1;
  }
  CmiUInt2 threadID = CmiBgMsgThreadID(msg);
  // broadcast except nodeID:threadId
  int len = CmiBgMsgLength(msg);
  for (int i=0; i<cva(numNodes); i++)
  {
    char *dupmsg;
    for (int j=0; j<cva(numWth); j++) {
      if (i==nodeID && j==threadID) continue;
      dupmsg = (char *)CmiAlloc(len);
      memcpy(dupmsg, msg, len);
      CmiBgMsgThreadID(dupmsg) = j;
      DEBUGF(("[%d] addBgNodeInbuffer to %d tid:%d\n", CmiMyPe(), i, j));
      addBgNodeInbuffer(dupmsg, i);
    }
  }
  CmiFree(msg);
  return 0;
}

/****************************************************************************************
			TimeLog correction
****************************************************************************************/
static inline int handleCorrectionMsg(BgTimeLine *logs, bgCorrectionMsg *m)
{
	CmiUInt2 tID = m->tID;
	if (tID == ANYTHREAD) {
	  int found = 0;
	  for (tID=0; tID<cva(numWth); tID++) {
        BgTimeLine &tline = logs[tID];	
		for (int j=0; j<tline.length(); j++)
		  if (tline[j]->msgID == m->msgID) { found = 1; break; }
		if (found) break;    
	  }
	  if (!found) {
//	    CmiPrintf("Correction message arrived early. \n");
		return 0;
	  }
	}
	BgAdjustTimeLineForward(m->msgID, m->tAdjust, logs[tID]);
	CmiFree(m);
	return 1;
}

void bgCorrectionFunc(char *msg)
{
    int i;
	bgCorrectionMsg* m = (bgCorrectionMsg*)msg;
	int nodeidx = nodeInfo::Global2Local(m->destNode);	
    bgCorrectionQ &cmsg = cva(nodeinfo)[nodeidx].cmsg;
    BgTimeLine *logs = cva(nodeinfo)[nodeidx].timelines;

	cmsg.enq(m);
	int len = cmsg.length();
    for (i=0; i<len; i++) {
	  bgCorrectionMsg *cm = cmsg.deq();
	  if (handleCorrectionMsg(logs, cm) == 0)
	    cmsg.enq(cm);
	}
}

#define ABS(x) (((x)<0)? -(x) : (x))

static double MSGTIME(int ox, int oy, int oz, int nx, int ny, int nz)
{
  int xd=ABS(ox-nx), yd=ABS(oy-ny), zd=ABS(oz-nz);
  int ncorners = 2;
  ncorners -= (xd?0:1 + yd?0:1 + zd?0:1);
  ncorners = (ncorners<0)?0:ncorners;
  return (ncorners*CYCLES_PER_CORNER + (xd+yd+zd)*CYCLES_PER_HOP)*CYCLE_TIME_FACTOR*1E-6;
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
  CmiBgMsgRecvTime(sendmsg) = MSGTIME(tMYX, tMYY, tMYZ, x,y,z) + BgGetTime();

  // timing
  BG_ADDMSG(sendmsg);

  if (local)
    addBgNodeInbuffer(sendmsg, tMYNODEID);
  else
    CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z),numbytes,sendmsg);
}

/* broadcast will copy data to msg buffer */
void nodeBroadcastPacketExcept_(int node, CmiUInt2 threadID, int handlerID, WorkType type, int numbytes, char* sendmsg)
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
  BG_ADDMSG(sendmsg);

  DEBUGF(("[%d]CmiSyncBroadcastAllAndFree node: %d\n", BgMyNode(), node));
  CmiSyncBroadcastAllAndFree(numbytes,sendmsg);
}

/* broadcast will copy data to msg buffer */
void threadBroadcastPacketExcept_(int node, CmiUInt2 threadID, int handlerID, WorkType type, int numbytes, char* sendmsg)
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
  CmiBgMsgRecvTime(sendmsg) = BgGetTime();	

  // timing
  BG_ADDMSG(sendmsg);

  DEBUGF(("[%d]CmiSyncBroadcastAllAndFree node: %d tid:%d\n", BgMyNode(), node, threadID));
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

void BgBroadcastPacketExcept(int node, CmiUInt2 threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  nodeBroadcastPacketExcept_(node, threadID, handlerID, type, numbytes, data);
}

void BgBroadcastAllPacket(int handlerID, WorkType type, int numbytes, char * data)
{
  nodeBroadcastPacketExcept_(BG_BROADCASTALL, ANYTHREAD, handlerID, type, numbytes, data);
}

void BgThreadBroadcastPacketExcept(int node, CmiUInt2 threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  threadBroadcastPacketExcept_(node, threadID, handlerID, type, numbytes, data);
}

void BgThreadBroadcastAllPacket(int handlerID, WorkType type, int numbytes, char * data)
{
  threadBroadcastPacketExcept_(BG_BROADCASTALL, ANYTHREAD, handlerID, type, numbytes, data);
}

/*****************************************************************************
      BG node level API
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

int BgGetTotalSize()
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
int BgNumNodes()
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

int BgMyNode()
{
#ifndef CMK_OPTIMIZE
  if (tMYNODE == NULL) CmiAbort("Calling BgMyNode in the main thread!");
#endif
  return nodeInfo::XYZ2Global(tMYX, tMYY, tMYZ);
}

int BgNodeToPE(int node)         /* return a real processor number from a bg node */
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

double BgGetTime()
{
#if 1
  /* accumulate time since last starttime, and reset starttime */
  double tp2= CmiWallTimer();
  tCURRTIME += (tp2 - tSTARTTIME);
  tSTARTTIME = tp2;
  return tCURRTIME;
#else
  /* sometime I am interested in real wall time */
  tCURRTIME = CmiWallTimer();
  return tCURRTIME;
#endif
}

void BgShutdown()
{
  // timing
#if BLUEGENE_TIMING
  if (printTimeLog) {
    for (int j=0; j<cva(numNodes); j++)
    for (int i=0; i<cva(numWth); i++) {
      BgTimeLine &log = cva(nodeinfo)[j].timelines[i];	
      BgPrintThreadTimeLine(nodeInfo::Local2Global(j), i, log);
    }
  }
#endif

  int msgSize = CmiBlueGeneMsgHeaderSizeBytes;
  void *sendmsg = CmiAlloc(msgSize);
  CmiSetHandler(sendmsg, cva(exitHandler));
  
  /* broadcast to shutdown */
  CmiSyncBroadcastAllAndFree(msgSize, sendmsg);
  //CmiAbort("\nBG> BlueGene emulator shutdown gracefully!\n");
  // CmiPrintf("\nBG> BlueGene emulator shutdown gracefully!\n");
  /* don't return */
  // ConverseExit();
  CmiDeliverMsgs(-1);
  CmiPrintf("\nBG> BlueGene emulator shutdown gracefully!\n");
  ConverseExit();
  exit(0);
  
/*
  int i;
  // TODO: free memory 
  delete [] cva(nodeinfo);
  delete [] inBuffer;
  for (i=0; i<numNodes; i++) CmmFree(msgBuffer[i]);
  delete [] msgBuffer;

  CmiAbort("\nBG> BlueGene emulator shutdown gracefully!\n");
*/
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

static void ProcessMessage(char *msg)
{
  int handler = CmiBgMsgHandle(msg);
  DEBUGF(("[%d] call handler %d\n", BgMyNode(), handler));

  BgHandler entryFunc;
#if  CMK_BLUEGENE_NODE
  entryFunc = tMYNODE->handlerTable.getHandle(handler);
#else
  entryFunc = tHANDLETAB.getHandle(handler);
  if (entryFunc == NULL) entryFunc = tMYNODE->handlerTable.getHandle(handler);
#endif

  if (entryFunc == NULL) {
    CmiPrintf("[%d] invalid handler: %d. \n", tMYNODEID, handler);
    CmiAbort("");
  }

  CmiSetHandler(msg, CmiBgMsgHandle(msg));

  entryFunc(msg);

}

void comm_thread(threadInfo *tinfo)
{
  /* set the thread-private threadinfo */
  cta(threadinfo) = tinfo;

  tSTARTTIME = CmiWallTimer();

  if (!tSTARTED) {
    tSTARTED = 1;
//    InitHandlerTable();
    BgNodeStart(arg_argc, arg_argv);
    /* bnv should be initialized */
  }

  for (;;) {
    char *msg = getFullBuffer();
    if (!msg) { 
//      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      tCOMMTHQ->enq(CthSelf());
      CthSuspend(); 
      tSTARTTIME = CmiWallTimer();
      continue;
    }
    DEBUGF(("[%d] comm thread has a msg.\n", BgMyNode()));
    /* schedule a worker thread, if small work do it itself */
    if (CmiBgMsgType(msg) == SMALL_WORK) {
      if (CmiBgMsgRecvTime(msg) > tCURRTIME)  tCURRTIME = CmiBgMsgRecvTime(msg);
      tSTARTTIME = CmiWallTimer();
      /* call user registered handler function */
      ProcessMessage(msg);
    }
    else {
      if (CmiBgMsgThreadID(msg) == ANYTHREAD) {
        DEBUGF(("anythread, call addBgNodeMessage\n"));
        addBgNodeMessage(msg);			/* non-affinity message */
      }
      else {
        DEBUGF(("affinity msg, call addBgThreadMessage to %d\n", CmiBgMsgThreadID(msg)));
        addBgThreadMessage(msg, CmiBgMsgThreadID(msg));
      }
    }
    /* let other communication thread do their jobs */
//    tCURRTIME += (CmiWallTimer()-tSTARTTIME);
    CthYield();
    tSTARTTIME = CmiWallTimer();
  }
}

void work_thread(threadInfo *tinfo)
{
  cta(threadinfo) = tinfo;

  tSTARTTIME = CmiWallTimer();

//  InitHandlerTable();
  if (workStartFunc) {
    // timing
    BG_ENTRYSTART(-1, NULL);
    workStartFunc(arg_argc, arg_argv);
    BG_ENTRYEND();
  }

  for (;;) {
    char *msg=NULL;
    ckMsgQueue &q1 = tNODEQ;
    ckMsgQueue &q2 = tAFFINITYQ;
    int e1 = q1.isEmpty();
    int e2 = q2.isEmpty();
    int fromQ2 = 0;		// delay the deq of msg from affinity queue

    if (e1 && !e2) { msg = q2[0]; fromQ2 = 1;}
    else if (e2 && !e1) { msg = q1.deq(); }
    else if (!e1 && !e2) {
      if (CmiBgMsgRecvTime(q1[0]) < CmiBgMsgRecvTime(q2[0])) {
        msg = q1.deq();
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
      tSTARTTIME = CmiWallTimer();
      DEBUGF(("[%d] work thread %d awakened.\n", BgMyNode(), tMYID));
      continue;
    }
    DEBUGF(("[%d] work thread %d has a msg.\n", BgMyNode(), tMYID));

    if (CmiBgMsgRecvTime(msg) > tCURRTIME) tCURRTIME = CmiBgMsgRecvTime(msg);

    // timing
    BG_ENTRYSTART(CmiBgMsgHandle(msg), msg);

    // don't count thread overhead and timinmg overhead
    tSTARTTIME = CmiWallTimer();

    // ProcessMessage may trap into scheduler
    ProcessMessage(msg);

    /* let other work thread do their jobs */
    tCURRTIME += (CmiWallTimer()-tSTARTTIME);

    // timing
    BG_ENTRYEND();

    if (fromQ2 == 1) q2.deq();
    DEBUGF(("[%d] work thread %d finish a msg.\n", BgMyNode(), tMYID));

    // suspend work thread, awaken at line 347 - addBgThreadMessage().
    CthYield();
    tSTARTTIME = CmiWallTimer();
  }
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

CmiHandler exitHandlerFunc(char *msg)
{
  // TODO: free memory before exit
  int i;
  delete [] cva(nodeinfo);
  delete [] cva(inBuffer);
  for (i=0; i<cva(numNodes); i++) CmmFree(cva(msgBuffer)[i]);
  delete [] cva(msgBuffer);

  //ConverseExit();
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

  CmiGetArgInt(argv, "+x", &cva(numX));
  CmiGetArgInt(argv, "+y", &cva(numY));
  CmiGetArgInt(argv, "+z", &cva(numZ));
  CmiGetArgInt(argv, "+cth", &cva(numCth));
  CmiGetArgInt(argv, "+wth", &cva(numWth));
  printTimeLog = CmiGetArgFlag(argv, "+bglog");

  arg_argv = argv;
  arg_argc = CmiGetArgc(argv);

  /* msg handler */
  CpvInitialize(int,msgHandler);
  cva(msgHandler) = CmiRegisterHandler((CmiHandler) msgHandlerFunc);
  CpvInitialize(int,nBcastMsgHandler);
  cva(nBcastMsgHandler) = CmiRegisterHandler((CmiHandler)nodeBCastMsgHandlerFunc);
  CpvInitialize(int,tBcastMsgHandler);
  cva(tBcastMsgHandler) = CmiRegisterHandler((CmiHandler)threadBCastMsgHandlerFunc);

  CpvInitialize(int,bgCorrectionHandler);
  cva(bgCorrectionHandler) = CmiRegisterHandler((CmiHandler) bgCorrectionFunc);

  CpvInitialize(int,exitHandler);
  cva(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);

  CpvInitialize(int, inEmulatorInit);
  cva(inEmulatorInit) = 1;
  /* call user defined BgEmulatorInit */
  BgEmulatorInit(arg_argc, arg_argv);
  cva(inEmulatorInit) = 0;

  /* check if all bluegene node size and thread information are set */
  BGARGSCHECK;

  BgInitTiming();		// timing module

  bgSize = cva(numX)*cva(numY)*cva(numZ);

  CtvInitialize(threadInfo *, threadinfo);

  /* number of bg nodes on this PE */
  CpvInitialize(int, numNodes);
  cva(numNodes) = nodeInfo::numLocalNodes();

  CpvInitialize(msgQueue *, inBuffer);
  cva(inBuffer) = new msgQueue[cva(numNodes)];
  _MEMCHECK(cva(inBuffer));
  for (i=0; i<cva(numNodes); i++) cva(inBuffer)[i].initialize(INBUFFER_SIZE);
  CpvInitialize(CmmTable *, msgBuffer);
  cva(msgBuffer) = new CmmTable[cva(numNodes)];
  _MEMCHECK(cva(msgBuffer));
  for (i=0; i<cva(numNodes); i++) cva(msgBuffer)[i] = CmmNew();

  /* create BG nodes */
  CpvInitialize(nodeInfo *, nodeinfo);
  cva(nodeinfo) = new nodeInfo[cva(numNodes)];
  _MEMCHECK(cva(nodeinfo));
  cta(threadinfo) = new threadInfo(-1, UNKNOWN_THREAD, NULL);
  _MEMCHECK(cta(threadinfo));
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

int main(int argc,char *argv[])
{
  ConverseInit(argc,argv,(CmiStartFn)bgMain,0,0);
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
  ASSERT(tTHREADTYPE != COMM_THREAD);
  if (tMYNODE == NULL) oldpe = -1;
  else if (tTHREADTYPE == COMM_THREAD) oldpe = -BgGetThreadID();
  else oldpe = BgGetGlobalWorkerThreadID();
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


