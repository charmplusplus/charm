/*
  File: Blue.C -- Converse BlueGene Emulator Code
  Emulator written by Gengbin Zheng, gzheng@uiuc.edu on 2/20/2001
*/ 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cklists.h"

#include "blue.h"

/* define system parameters */
#define INBUFFER_SIZE	32

#define CYCLES_PER_HOP     5
#define CYCLES_PER_CORNER  75
#define CYCLE_TIME_FACTOR  0.001  /* one cycle = nanosecond = 10^(-3) us */
/* end of system parameters */

#define MAX_HANDLERS	32
#define BROADCAST	-1
#define BROADCASTALL	-2
template<class T> class bgQueue;

typedef char ThreadType;
const char UNKNOWN_THREAD=0, COMM_THREAD=1, WORK_THREAD=2;

/* converse message send to other bgnodes */
class bgMsg {
public:
  int node;		/* bluegene node serial number */
  int threadID;		/* the thread ID in the node */
  int handlerID;
  WorkType type;
  int len;
  double recvTime;
  char  first[1];	/* first byte of user data */
public:
  bgMsg() {};
};

typedef bgQueue<int>  	    threadIDQueue;
typedef bgQueue<CthThread>  threadQueue;
typedef bgQueue<bgMsg *>    msgQueue;
typedef CkQ<bgMsg *> 	    ckMsgQueue;

class nodeInfo;
class threadInfo;

/* node level variables */
CpvDeclare(nodeInfo*, nodeinfo);		/* represent a bluegene node */

/* thread level variables */
CtvDeclare(threadInfo *, threadinfo);	/* represent a bluegene thread */

/* emulator node level variables */
CpvDeclare(int,msgHandler);
CmiHandler msgHandlerFunc(char *msg);

CpvDeclare(int,exitHandler);

typedef void (*BgStartHandler) (int, char **);
CpvStaticDeclare(BgHandler *, handlerTable);
CpvStaticDeclare(int, handlerTableCount);

static int arg_argc;
static char **arg_argv;

CpvStaticDeclare(msgQueue *,inBuffer);	/* emulate the bluegene fix-size inbuffer */
CpvStaticDeclare(CmmTable *,msgBuffer);	/* if inBuffer is full, put to this buffer */

CpvStaticDeclare(int, numX);	/* size of bluegene nodes in cube */
CpvStaticDeclare(int, numY);
CpvStaticDeclare(int, numZ);
CpvStaticDeclare(int, numCth);	/* number of threads */
CpvStaticDeclare(int, numWth);
CpvStaticDeclare(int, numNodes);	/* number of bg nodes on this PE */

CpvStaticDeclare(int, inEmulatorInit);


#define tMYID		CtvAccess(threadinfo)->id
#define tMYGLOBALID	CtvAccess(threadinfo)->globalId
#define tTHREADTYPE	CtvAccess(threadinfo)->type
#define tMYNODE		CtvAccess(threadinfo)->myNode
#define tSTARTTIME	tMYNODE->startTime
#define tCURRTIME	CtvAccess(threadinfo)->currTime
#define tMYX		tMYNODE->x
#define tMYY		tMYNODE->y
#define tMYZ		tMYNODE->z
#define tMYNODEID	tMYNODE->id
#define tCOMMTHQ	tMYNODE->commThQ
#define tINBUFFER	CpvAccess(inBuffer)[tMYNODE->id]
#define tMSGBUFFER	CpvAccess(msgBuffer)[tMYNODE->id]
#define tUSERDATA	tMYNODE->udata
#define tTHREADTABLE    tMYNODE->threadTable
#define tAFFINITYQ      tMYNODE->affinityQ[tMYID]
#define tNODEQ          tMYNODE->nodeQ
#define tSTARTED        tMYNODE->started

#define ASSERT(x)	if (!(x)) { CmiPrintf("Assert failure at %s:%d\n", __FILE__,__LINE__); CmiAbort("Abort!"); }

#define BGARGSCHECK   	\
  if (CpvAccess(numX)==0 || CpvAccess(numY)==0 || CpvAccess(numZ)==0)  { CmiPrintf("\nMissing parameters for BlueGene machine size!\n<tip> use command line options: +x, +y, or +z.\n"); BgShutdown(); } \
  if (CpvAccess(numCth)==0 || CpvAccess(numWth)==0) { CmiAbort("\nMissing parameters for number of communication/worker threads!\n<tip> use command line options: +cth or +wth.\n"); BgShutdown(); }

#define HANDLERCHECK(handler)	\
  if (CpvAccess(handlerTable)[handler] == NULL) {	\
    CmiPrintf("Handler %d unregistered!\n", handler);	\
    CmiAbort("Abort!\n");	\
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

/*****************************************************************************
      NodeInfo:
        including a group of functions defining the mapping, terms used here:
        XYZ: (x,y,z)
        Global:  map (x,y,z) to a global serial number
        Local:   local index of this nodeinfo in the emulator's node 
*****************************************************************************/

class nodeInfo {
public:
  int id;
  int x,y,z;
  threadQueue *commThQ;		/* suspended comm threads queue */
  CthThread *threadTable;	/* thread table for both work and comm threads*/
  ckMsgQueue nodeQ;		/* non-affinity msg queue */
  ckMsgQueue *affinityQ;	/* affinity msg queue for each work thread */
  char *udata;			/* node specific data pointer */
  double startTime;		/* start time for a thread */
  char started;			/* flag indicate if this node is started */

public:
  nodeInfo(): udata(NULL), started(0) {
    commThQ = new threadQueue;
    commThQ->initialize(CpvAccess(numCth));

    threadTable = new CthThread[CpvAccess(numWth)+CpvAccess(numCth)];

    affinityQ = new ckMsgQueue[CpvAccess(numWth)];
  }

  ~nodeInfo() {
    if (commThQ) delete commThQ;
    delete [] affinityQ;
    delete [] threadTable;
  }
  
    /* return the number of bg nodes on this physical emulator PE */
  inline static int numLocalNodes()
  {
    int n, m;
    n = (CpvAccess(numX) * CpvAccess(numY) * CpvAccess(numZ)) / CmiNumPes();
    m = (CpvAccess(numX) * CpvAccess(numY) * CpvAccess(numZ)) % CmiNumPes();
    if (CmiMyPe() < m) n++;
    return n;
  }

    /* map global serial number to (x,y,z) ++++ */
  inline static void Global2XYZ(int seq, int *x, int *y, int *z) {
    *x = seq / (CpvAccess(numY) * CpvAccess(numZ));
    *y = (seq - *x * CpvAccess(numY) * CpvAccess(numZ)) / CpvAccess(numZ);
    *z = (seq - *x * CpvAccess(numY) * CpvAccess(numZ)) % CpvAccess(numZ);
  }

    /* calculate global serial number of (x,y,z) ++++ */
  inline static int XYZ2Global(int x, int y, int z) {
    return x*(CpvAccess(numY) * CpvAccess(numZ)) + y*CpvAccess(numZ) + z;
  }

    /* map (x,y,z) to emulator PE ++++ */
  inline static int XYZ2PE(int x, int y, int z) {
    return Global2PE(XYZ2Global(x,y,z));
  }

  inline static int XYZ2Local(int x, int y, int z) {
    return Global2Local(XYZ2Global(x,y,z));
  }

    /* local node index number to x y z ++++ */
  inline static void Local2XYZ(int num, int *x, int *y, int *z)  {
    Global2XYZ(Local2Global(num), x, y, z);
  }

    /* map global serial node number to PE ++++ */
  inline static int Global2PE(int num) { return num % CmiNumPes(); }

    /* map global serial node ID to local node array index  ++++ */
  inline static int Global2Local(int num) { return num/CmiNumPes(); }

    /* map local node index to global serial node id ++++ */
  inline static int Local2Global(int num) { return CmiMyPe()+num*CmiNumPes();}

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
  int id;
  int globalId;
  ThreadType  type;
  CthThread me;			/* Converse thread handler */
  nodeInfo *myNode;		/* the node belonged to */
  double  currTime;		/* thread timer */

public:
  threadInfo(int _id, ThreadType _type, nodeInfo *_node): id(_id), type(_type), myNode(_node) {
    currTime=0.0;
    if (id != -1) globalId = nodeInfo::Local2Global(_node->id)*(CpvAccess(numCth)+CpvAccess(numWth))+_id;
  }
  inline void setThread(CthThread t) { me = t; }
  inline CthThread getThread() { return me; }
}; 


/*****************************************************************************
      low level API
*****************************************************************************/

extern "C" void defaultBgHandler(char *null)
{
  CmiAbort("BG> Invalid Handler called!\n");
}

int BgRegisterHandler(BgHandler h)
{
  /* leave 0 as blank, so it can report error luckily */
  int cur = CpvAccess(handlerTableCount)++;

  ASSERT(CpvAccess(inEmulatorInit));
  if (cur >= MAX_HANDLERS)
    CmiAbort("BG> HandlerID exceed the maximum.\n");
  CpvAccess(handlerTable)[cur] = h;
  return cur;
}

/* communication thread call getFullBuffer to test if there is data ready 
   in INBUFFER for its own queue */
bgMsg * getFullBuffer()
{
  int tags[1], ret_tags[1];
  bgMsg *data, *mb;

  /* I must be a communication thread */
  if (tTHREADTYPE != COMM_THREAD) 
    CmiAbort("GetFullBuffer called by a non-communication thread!\n");

  /* see if we have msg in inBuffer */
  if (tINBUFFER.isEmpty()) return NULL;
  data = tINBUFFER.deq(); 
  /* since we have just delete one from inbuffer, fill one from msgbuffer */
  tags[0] = CmmWildCard;
  mb = (bgMsg *)CmmGet(tMSGBUFFER, 1, tags, ret_tags);
  if (mb) tINBUFFER.enq(mb);

  return data;
}

/* add message msgPtr to a bluegene node's inbuffer queue */
void addBgNodeInbuffer(bgMsg *msgPtr, int nodeID)
{
  int tags[1];

  /* if inbuffer is full, store in the msgbuffer */
  if (CpvAccess(inBuffer)[nodeID].isFull()) {
    tags[0] = nodeID;
    CmmPut(CpvAccess(msgBuffer)[nodeID], 1, tags, (char *)msgPtr);
  }
  else {
    CpvAccess(inBuffer)[nodeID].enq(msgPtr);
  }
  /* awake a communication thread to schedule it */
  CthThread t=CpvAccess(nodeinfo)[nodeID].commThQ->deq();
  if (t) CthAwaken(t);
}

/* add a message to a thread's affinity queue */
void addBgThreadMessage(bgMsg *msgPtr, int threadID)
{
  ckMsgQueue &que = tMYNODE->affinityQ[threadID];
  que.enq(msgPtr);
  if (que.length() == 1)
    CthAwaken(tTHREADTABLE[threadID]);
}

/* add a message to a node's non-affinity queue */
void addBgNodeMessage(bgMsg *msgPtr)
{
  /* find a idle worker thread */
  /* FIXME:  flat search is bad if there is many work threads */
  for (int i=0; i<CpvAccess(numWth); i++)
    if (tMYNODE->affinityQ[i].length() == 0)
    {
      /* this work thread is idle, schedule the msg here */
      tMYNODE->affinityQ[i].enq(msgPtr);
      CthAwaken(tTHREADTABLE[i]);
      return;
    }
  /* all worker threads are busy */   
  tNODEQ.enq(msgPtr);
}

int checkReady()
{
  if (tTHREADTYPE != COMM_THREAD)
    CmiAbort("checkReady called by a non-communication thread!\n");
  return !tINBUFFER.isEmpty();
}

void sendPacket(int x, int y, int z, int msgSize,bgMsg *msg)
{
  CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z),msgSize,(char *)msg);
}

/* handler to process the msg */
CmiHandler msgHandlerFunc(char *msg)
{
  /* bgmsg is CmiMsgHeaderSizeBytes offset of original message pointer */
  bgMsg *bgmsg = (bgMsg *)(msg+CmiMsgHeaderSizeBytes);
  int nodeID = bgmsg->node;
  if (nodeID >= 0) {
    nodeID = nodeInfo::Global2Local(nodeID);
    addBgNodeInbuffer(bgmsg, nodeID);
  }
  else {
    int len = bgmsg->len;
    addBgNodeInbuffer(bgmsg, 0);
    for (int i=1; i<CpvAccess(numNodes); i++)
    {
      char *dupmsg = (char *)malloc(len);
      memcpy(dupmsg, msg, len);
      addBgNodeInbuffer((bgMsg*)(dupmsg+CmiMsgHeaderSizeBytes), i);
    }
  }
  return 0;
}


#define ABS(x) (((x)<0)? -(x) : (x))

static double MSGTIME(int ox, int oy, int oz, int nx, int ny, int nz)
{
  int xd=ABS(ox-nx), yd=ABS(oy-ny), zd=ABS(oz-nz);
  int ncorners = 2;
  ncorners -= (xd?0:1 + yd?0:1 + zd?0:1);
  ncorners = (ncorners<0)?0:ncorners;
  return (ncorners*CYCLES_PER_CORNER + (xd+yd+zd)*CYCLES_PER_HOP)*CYCLE_TIME_FACTOR;
}

/* send will copy data to msg buffer */
/* user data is not freeed in this routine, user can reuse the data ! */
void sendPacket_(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char* data, int local)
{
  int msgSize = CmiMsgHeaderSizeBytes+sizeof(bgMsg)-1+numbytes;
  void *sendmsg = CmiAlloc(msgSize);
  CmiSetHandler(sendmsg, CpvAccess(msgHandler));
  bgMsg *bgmsg = (bgMsg *)((char *)sendmsg+CmiMsgHeaderSizeBytes);
  bgmsg->node = nodeInfo::XYZ2Global(x,y,z);
  bgmsg->threadID = threadID;
  bgmsg->handlerID = handlerID;
  bgmsg->type = type;
  bgmsg->len = msgSize;
  bgmsg->recvTime = MSGTIME(tMYX, tMYY, tMYZ, x,y,z) + BgGetTime();
  if (numbytes) memcpy(bgmsg->first, data, numbytes);

  if (local)
    addBgNodeInbuffer(bgmsg, tMYNODEID);
  else
    CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z),msgSize,sendmsg);
}

/* broadcast will copy data to msg buffer */
/* user data is not freeed in this routine, user can reuse the data ! */
void broadcastPacket_(int bcasttype, int threadID, int handlerID, WorkType type, int numbytes, char* data)
{
  int msgSize = CmiMsgHeaderSizeBytes+sizeof(bgMsg)-1+numbytes;	
  void *sendmsg = CmiAlloc(msgSize);	
  CmiSetHandler(sendmsg, CpvAccess(msgHandler));	
  bgMsg *bgmsg = (bgMsg *)((char *)sendmsg+CmiMsgHeaderSizeBytes);	
  bgmsg->node = bcasttype;
  bgmsg->threadID = threadID;	
  bgmsg->handlerID = handlerID;	
  bgmsg->type = type;	
  bgmsg->len = msgSize;
  /* FIXME */
  bgmsg->recvTime = BgGetTime();	
  if (numbytes) memcpy(bgmsg->first, data, numbytes);

  CmiSyncBroadcastAndFree(msgSize,sendmsg);
}

/* sendPacket to route */
/* this function can be called by any thread */
void BgSendNonLocalPacket(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  if (x<0 || y<0 || z<0 || x>=CpvAccess(numX) || y>=CpvAccess(numY) || z>=CpvAccess(numZ)) {
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


/*****************************************************************************
      BG node level API
*****************************************************************************/

/* must be called in a communication or worker thread */
void BgGetXYZ(int *x, int *y, int *z)
{
  ASSERT(!CpvAccess(inEmulatorInit));
  *x = tMYX; *y = tMYY; *z = tMYZ;
}

void BgGetSize(int *sx, int *sy, int *sz)
{
  *sx = CpvAccess(numX); *sy = CpvAccess(numY); *sz = CpvAccess(numZ);
}

/* can only called in emulatorinit */
void BgSetSize(int sx, int sy, int sz)
{
  ASSERT(CpvAccess(inEmulatorInit));
  CpvAccess(numX) = sx; CpvAccess(numY) = sy; CpvAccess(numZ) = sz;
}

int BgGetThreadID()
{
  ASSERT(tTHREADTYPE == WORK_THREAD || tTHREADTYPE == COMM_THREAD);
  return tMYID;
}

int BgGetGlobalThreadID()
{
  ASSERT(tTHREADTYPE == WORK_THREAD || tTHREADTYPE == COMM_THREAD);
  return tMYGLOBALID;
}

char *BgGetNodeData()
{
  return tUSERDATA;
}

void BgSetNodeData(char *data)
{
  ASSERT(!CpvAccess(inEmulatorInit));
  tUSERDATA = data;
}

int BgGetNumWorkThread()
{
  return CpvAccess(numWth);
}

void BgSetNumWorkThread(int num)
{
  ASSERT(CpvAccess(inEmulatorInit));
  CpvAccess(numWth) = num;
}

int BgGetNumCommThread()
{
  return CpvAccess(numCth);
}

void BgSetNumCommThread(int num)
{
  ASSERT(CpvAccess(inEmulatorInit));
  CpvAccess(numCth) = num;
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
  tCURRTIME = CmiWallTimer();
  return tCURRTIME;
#endif
}

void BgShutdown()
{
  int msgSize = CmiMsgHeaderSizeBytes;
  void *sendmsg = CmiAlloc(msgSize);
  CmiSetHandler(sendmsg, CpvAccess(exitHandler));
  
  /* broadcast to shutdown */
  CmiSyncBroadcastAllAndFree(msgSize, sendmsg);
  //CmiAbort("\nBG> BlueGene simulator shutdown gracefully!\n");
  // CmiPrintf("\nBG> BlueGene simulator shutdown gracefully!\n");
  /* don't return */
  // ConverseExit();
  CmiDeliverMsgs(-1);
  CmiPrintf("\nBG> BlueGene simulator shutdown gracefully!\n");
  ConverseExit();
  exit(0);
  
/*
  int i;
  // TODO: free memory 
  delete [] CpvAccess(nodeinfo);
  delete [] inBuffer;
  for (i=0; i<numNodes; i++) CmmFree(msgBuffer[i]);
  delete [] msgBuffer;

  CmiAbort("\nBG> BlueGene simulator shutdown gracefully!\n");
*/
}

/*****************************************************************************
      Communication and Worker threads
*****************************************************************************/

void comm_thread(threadInfo *tinfo)
{
  /* set the thread-private threadinfo */
  CtvAccess(threadinfo) = tinfo;

  tSTARTTIME = CmiWallTimer();

  if (!tSTARTED) {
    tSTARTED = 1;
    BgNodeStart(arg_argc, arg_argv);
  }

  for (;;) {
    bgMsg *msg = (bgMsg *)getFullBuffer();
    if (!msg) { 
      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      tCOMMTHQ->enq(CthSelf());
      CthSuspend(); 
      tSTARTTIME = CmiWallTimer();
      continue;
    }
    /* schedule a worker thread, if small work do it itself */
    if (msg->type == SMALL_WORK) {
      if (msg->recvTime > tCURRTIME)  tCURRTIME = msg->recvTime;
      /* call user registered handler function */
      int handler = msg->handlerID;
      CpvAccess(handlerTable)[handler](msg->first);
      /* free the message */
      CmiFree((char *)msg-CmiMsgHeaderSizeBytes); 
    }
    else {
      if (msg->threadID == ANYTHREAD) {
        addBgNodeMessage(msg);			/* non-affinity message */
      }
      else {
        addBgThreadMessage(msg, msg->threadID);
      }
    }
    /* let other communication thread do their jobs */
    tCURRTIME += (CmiWallTimer()-tSTARTTIME);
    CthYield();
    tSTARTTIME = CmiWallTimer();
  }
}

void work_thread(threadInfo *tinfo)
{
  int handler;

  CtvAccess(threadinfo) = tinfo;

  tSTARTTIME = CmiWallTimer();
  for (;;) {
    bgMsg *msg=NULL;
    ckMsgQueue &q1 = tNODEQ;
    ckMsgQueue &q2 = tAFFINITYQ;
    int e1 = q1.isEmpty();
    int e2 = q2.isEmpty();

    if (e1 && !e2) { msg = q2.deq(); }
    else if (e2 && !e1) { msg = q1.deq(); }
    else if (!e1 && !e2) {
      if (q1[0]->recvTime < q2[0]->recvTime) {
        msg = q1.deq();
      }
      else {
        msg = q2.deq();
      }
    }
    /* if no msg is ready, put it to sleep */
    if ( msg == NULL ) {
      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      CthSuspend();
      tSTARTTIME = CmiWallTimer();
      continue;
    }
    if (msg->recvTime > tCURRTIME)  tCURRTIME = msg->recvTime;
    handler = msg->handlerID;
    
    /* call user registered handler function */
    CpvAccess(handlerTable)[handler](msg->first);
      /* free the msg and clear the buffer */
    CmiFree((char *)msg-CmiMsgHeaderSizeBytes); 
    /* let other work thread do their jobs */
    tCURRTIME += (CmiWallTimer()-tSTARTTIME);
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
  for (i=0; i< CpvAccess(numWth); i++)
  {
    threadInfo *tinfo = new threadInfo(i, WORK_THREAD, ninfo);
    t = CthCreate((CthVoidFn)work_thread, tinfo, 0);
    tinfo->setThread(t);
    /* put to thread table */
    tTHREADTABLE[tinfo->id] = t;
    CthAwaken(t);
  }

  /* creat communication thread */
  for (i=0; i< CpvAccess(numCth); i++)
  {
    threadInfo *tinfo = new threadInfo(i+CpvAccess(numWth), COMM_THREAD, ninfo);
    t = CthCreate((CthVoidFn)comm_thread, tinfo, 0);
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
  delete [] CpvAccess(nodeinfo);
  delete [] CpvAccess(inBuffer);
  for (i=0; i<CpvAccess(numNodes); i++) CmmFree(CpvAccess(msgBuffer)[i]);
  delete [] CpvAccess(msgBuffer);

  //ConverseExit();
  CsdExitScheduler();

  //if (CmiMyPe() == 0) CmiPrintf("\nBG> BlueGene emulator shutdown gracefully!\n");

  return 0;
}

CmiStartFn mymain(int argc, char **argv)
{
  int i;

  
  /* initialize all processor level data */
  CpvInitialize(int,numX);
  CpvInitialize(int,numY);
  CpvInitialize(int,numZ);
  CpvInitialize(int,numCth);
  CpvInitialize(int,numWth);
  CpvAccess(numX) = CpvAccess(numY) = CpvAccess(numZ) = 0;
  CpvAccess(numCth) = CpvAccess(numWth) = 0;

  CmiGetArgInt(argv, "+x", &CpvAccess(numX));
  CmiGetArgInt(argv, "+y", &CpvAccess(numY));
  CmiGetArgInt(argv, "+z", &CpvAccess(numZ));
  CmiGetArgInt(argv, "+cth", &CpvAccess(numCth));
  CmiGetArgInt(argv, "+wth", &CpvAccess(numWth));

  arg_argv = argv;
  arg_argc = CmiGetArgc(argv);

  /* msg handler */
  CpvInitialize(int,msgHandler);
  CpvAccess(msgHandler) = CmiRegisterHandler((CmiHandler) msgHandlerFunc);

  CpvInitialize(int,exitHandler);
  CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);

  /* init handlerTable */
  CpvInitialize(int, handlerTableCount);
  CpvAccess(handlerTableCount) = 1;
  CpvInitialize(BgHandler*, handlerTable);
  CpvAccess(handlerTable) = (BgHandler *)malloc(MAX_HANDLERS * sizeof(BgHandler));
  for (i=0; i<MAX_HANDLERS; i++) CpvAccess(handlerTable)[i] = defaultBgHandler;

  CpvInitialize(int, inEmulatorInit);
  CpvAccess(inEmulatorInit) = 0;

  CpvAccess(inEmulatorInit) = 1;
  /* call user defined BgEmulatorInit */
  BgEmulatorInit(arg_argc, arg_argv);
  CpvAccess(inEmulatorInit) = 0;

  /* check if all bluegene node size and thread information are set */
  BGARGSCHECK;

  CtvInitialize(threadInfo *, threadinfo);

  /* number of bg nodes on this PE */
  CpvInitialize(int, numNodes);
  CpvAccess(numNodes) = nodeInfo::numLocalNodes();

  CpvInitialize(msgQueue *, inBuffer);
  CpvAccess(inBuffer) = new msgQueue[CpvAccess(numNodes)];
  for (i=0; i<CpvAccess(numNodes); i++) CpvAccess(inBuffer)[i].initialize(INBUFFER_SIZE);
  CpvInitialize(CmmTable *, msgBuffer);
  CpvAccess(msgBuffer) = new CmmTable[CpvAccess(numNodes)];
  for (i=0; i<CpvAccess(numNodes); i++) CpvAccess(msgBuffer)[i] = CmmNew();

  /* create BG nodes */
  CpvInitialize(nodeInfo *, nodeinfo);
  CpvAccess(nodeinfo) = new nodeInfo[CpvAccess(numNodes)];
  CtvAccess(threadinfo) = new threadInfo(-1, UNKNOWN_THREAD, NULL);
  for (i=0; i<CpvAccess(numNodes); i++)
  {
    nodeInfo *ninfo = CpvAccess(nodeinfo) + i;
    ninfo->id = i;
    nodeInfo::Local2XYZ(i, &ninfo->x, &ninfo->y, &ninfo->z);

    /* pretend that I am a thread */
    CtvAccess(threadinfo)->myNode = ninfo;

    /* initialize a BG node and fire all threads */
    BgNodeInitialize(ninfo);
  }

  return 0;
}

int main(int argc,char *argv[])
{
  ConverseInit(argc,argv,(CmiStartFn)mymain,0,0);
  return 0;
}

