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
#if 0
class bgMsg {
public:
  char core[CmiMsgHeaderSizeBytes];
  int node;		/* bluegene node serial number */
  int threadID;		/* the thread ID in the node */
  int handlerID;	/* handler function registered */
  WorkType type;	/* work type */
  int len;
  double recvTime;
  char  first[1];	/* first byte of user data */
public:
  bgMsg() {};
};
#endif

typedef bgQueue<int>  	    threadIDQueue;
typedef bgQueue<CthThread>  threadQueue;
typedef bgQueue<char *>    msgQueue;
typedef CkQ<char *> 	    ckMsgQueue;

class nodeInfo;
class threadInfo;

#define cva CpvAccess
#define cta CtvAccess

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

CpvDeclare(int, inEmulatorInit);
CpvDeclare(int, cpvinited);

#define tMYID		cta(threadinfo)->id
#define tMYGLOBALID	cta(threadinfo)->globalId
#define tTHREADTYPE	cta(threadinfo)->type
#define tMYNODE		cta(threadinfo)->myNode
#define tSTARTTIME	tMYNODE->startTime
#define tCURRTIME	cta(threadinfo)->currTime
#define tMYX		tMYNODE->x
#define tMYY		tMYNODE->y
#define tMYZ		tMYNODE->z
#define tMYNODEID	tMYNODE->id
#define tCOMMTHQ	tMYNODE->commThQ
#define tINBUFFER	cva(inBuffer)[tMYNODE->id]
#define tMSGBUFFER	cva(msgBuffer)[tMYNODE->id]
#define tUSERDATA	tMYNODE->udata
#define tTHREADTABLE    tMYNODE->threadTable
#define tAFFINITYQ      tMYNODE->affinityQ[tMYID]
#define tNODEQ          tMYNODE->nodeQ
#define tSTARTED        tMYNODE->started

#define ASSERT(x)	if (!(x)) { CmiPrintf("Assert failure at %s:%d\n", __FILE__,__LINE__); CmiAbort("Abort!"); }

#define BGARGSCHECK   	\
  if (cva(numX)==0 || cva(numY)==0 || cva(numZ)==0)  { CmiPrintf("\nMissing parameters for BlueGene machine size!\n<tip> use command line options: +x, +y, or +z.\n"); BgShutdown(); } \
  if (cva(numCth)==0 || cva(numWth)==0) { CmiAbort("\nMissing parameters for number of communication/worker threads!\n<tip> use command line options: +cth or +wth.\n"); BgShutdown(); }

#define HANDLERCHECK(handler)	\
  if (cva(handlerTable)[handler] == NULL) {	\
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
    commThQ->initialize(cva(numCth));

    threadTable = new CthThread[cva(numWth)+cva(numCth)];

    affinityQ = new ckMsgQueue[cva(numWth)];
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
    n = (cva(numX) * cva(numY) * cva(numZ)) / CmiNumPes();
    m = (cva(numX) * cva(numY) * cva(numZ)) % CmiNumPes();
    if (CmiMyPe() < m) n++;
    return n;
  }

    /* map global serial number to (x,y,z) ++++ */
  inline static void Global2XYZ(int seq, int *x, int *y, int *z) {
    *x = seq / (cva(numY) * cva(numZ));
    *y = (seq - *x * cva(numY) * cva(numZ)) / cva(numZ);
    *z = (seq - *x * cva(numY) * cva(numZ)) % cva(numZ);
  }

    /* calculate global serial number of (x,y,z) ++++ */
  inline static int XYZ2Global(int x, int y, int z) {
    return x*(cva(numY) * cva(numZ)) + y*cva(numZ) + z;
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
    if (id != -1) globalId = nodeInfo::Local2Global(_node->id)*(cva(numCth)+cva(numWth))+_id;
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
  int cur;

  for (cur=1; cur<cva(handlerTableCount); cur++)
    if (cva(handlerTable)[cur] == h) return cur;
  cva(handlerTableCount)++;
  if (cur >= MAX_HANDLERS)
    CmiAbort("BG> HandlerID exceed the maximum.\n");
  cva(handlerTable)[cur] = h;
  return cur;
}

void BgNumberHandler(int idx, BgHandler h)
{
  cva(handlerTable)[idx] = h;
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
    cva(inBuffer)[nodeID].enq(msgPtr);
  }
  /* awake a communication thread to schedule it */
  CthThread t=cva(nodeinfo)[nodeID].commThQ->deq();
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

void sendPacket(int x, int y, int z, int msgSize,char *msg)
{
  CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z),msgSize,(char *)msg);
}

/* handler to process the msg */
CmiHandler msgHandlerFunc(char *msg)
{
  /* bgmsg is CmiMsgHeaderSizeBytes offset of original message pointer */
  int nodeID = CmiBgGetNodeID(msg);
  if (nodeID >= 0) {
    nodeID = nodeInfo::Global2Local(nodeID);
    addBgNodeInbuffer(msg, nodeID);
  }
  else {
    int len = CmiBgGetLength(msg);
    addBgNodeInbuffer(msg, 0);
    for (int i=1; i<cva(numNodes); i++)
    {
      char *dupmsg = (char *)CmiAlloc(len);
      memcpy(dupmsg, msg, len);
      addBgNodeInbuffer(dupmsg, i);
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
void sendPacket_(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char* sendmsg, int local)
{
  CmiSetHandler(sendmsg, cva(msgHandler));
  CmiBgGetNodeID(sendmsg) = nodeInfo::XYZ2Global(x,y,z);
  CmiBgGetThreadID(sendmsg) = threadID;
  CmiBgGetHandle(sendmsg) = handlerID;
  CmiBgGetType(sendmsg) = type;
  CmiBgGetLength(sendmsg) = numbytes;
  CmiBgGetRecvTime(sendmsg) = MSGTIME(tMYX, tMYY, tMYZ, x,y,z) + BgGetTime();

  if (local)
    addBgNodeInbuffer(sendmsg, tMYNODEID);
  else
    CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z),numbytes,sendmsg);
}

/* broadcast will copy data to msg buffer */
/* user data is not freeed in this routine, user can reuse the data ! */
void broadcastPacket_(int bcasttype, int threadID, int handlerID, WorkType type, int numbytes, char* sendmsg)
{
  CmiSetHandler(sendmsg, cva(msgHandler));	
  CmiBgGetNodeID(sendmsg) = bcasttype;
  CmiBgGetThreadID(sendmsg) = threadID;	
  CmiBgGetHandle(sendmsg) = handlerID;	
  CmiBgGetType(sendmsg) = type;	
  CmiBgGetLength(sendmsg) = numbytes;
  /* FIXME */
  CmiBgGetRecvTime(sendmsg) = BgGetTime();	

  CmiSyncBroadcastAndFree(numbytes,sendmsg);
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


/*****************************************************************************
      BG node level API
*****************************************************************************/

/* must be called in a communication or worker thread */
void BgGetXYZ(int *x, int *y, int *z)
{
  ASSERT(!cva(inEmulatorInit));
  *x = tMYX; *y = tMYY; *z = tMYZ;
}

void BgGetSize(int *sx, int *sy, int *sz)
{
  *sx = cva(numX); *sy = cva(numY); *sz = cva(numZ);
}

int BgGetTotalSize()
{
  return cva(numX)*cva(numY)*cva(numZ);
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
  ASSERT(!cva(inEmulatorInit));
  return tMYNODEID;
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

void comm_thread(threadInfo *tinfo)
{
  /* set the thread-private threadinfo */
  cta(threadinfo) = tinfo;

  tSTARTTIME = CmiWallTimer();

  if (!tSTARTED) {
    tSTARTED = 1;
    BgNodeStart(arg_argc, arg_argv);
    /* bnv should be initialized */
    cva(cpvinited) = 1;
  }

  for (;;) {
    char *msg = getFullBuffer();
    if (!msg) { 
      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      tCOMMTHQ->enq(CthSelf());
      CthSuspend(); 
      tSTARTTIME = CmiWallTimer();
      continue;
    }
    /* schedule a worker thread, if small work do it itself */
    if (CmiBgGetType(msg) == SMALL_WORK) {
      if (CmiBgGetRecvTime(msg) > tCURRTIME)  tCURRTIME = CmiBgGetRecvTime(msg);
      /* call user registered handler function */
      int handler = CmiBgGetHandle(msg);
      cva(handlerTable)[handler](msg);
    }
    else {
      if (CmiBgGetThreadID(msg) == ANYTHREAD) {
        addBgNodeMessage(msg);			/* non-affinity message */
      }
      else {
        addBgThreadMessage(msg, CmiBgGetThreadID(msg));
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

  cta(threadinfo) = tinfo;

  tSTARTTIME = CmiWallTimer();
  for (;;) {
    char *msg=NULL;
    ckMsgQueue &q1 = tNODEQ;
    ckMsgQueue &q2 = tAFFINITYQ;
    int e1 = q1.isEmpty();
    int e2 = q2.isEmpty();

    if (e1 && !e2) { msg = q2.deq(); }
    else if (e2 && !e1) { msg = q1.deq(); }
    else if (!e1 && !e2) {
      if (CmiBgGetRecvTime(q1[0]) < CmiBgGetRecvTime(q2[0])) {
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
    if (CmiBgGetRecvTime(msg) > tCURRTIME)  tCURRTIME = CmiBgGetRecvTime(msg);
    handler = CmiBgGetHandle(msg);
    
    /* call user registered handler function */
    cva(handlerTable)[handler](msg);
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
  for (i=0; i< cva(numWth); i++)
  {
    threadInfo *tinfo = new threadInfo(i, WORK_THREAD, ninfo);
    _MEMCHECK(tinfo);
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
    threadInfo *tinfo = new threadInfo(i+cva(numWth), COMM_THREAD, ninfo);
    _MEMCHECK(tinfo);
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

  arg_argv = argv;
  arg_argc = CmiGetArgc(argv);

  /* msg handler */
  CpvInitialize(int,msgHandler);
  cva(msgHandler) = CmiRegisterHandler((CmiHandler) msgHandlerFunc);

  CpvInitialize(int,exitHandler);
  cva(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);

  /* init handlerTable */
  CpvInitialize(int, handlerTableCount);
  cva(handlerTableCount) = 1;
  CpvInitialize(BgHandler*, handlerTable);
  cva(handlerTable) = (BgHandler *)malloc(MAX_HANDLERS * sizeof(BgHandler));
  for (i=0; i<MAX_HANDLERS; i++) cva(handlerTable)[i] = defaultBgHandler;

  CpvInitialize(int, inEmulatorInit);
  cva(inEmulatorInit) = 0;

  CpvInitialize(int, cpvinited);
  cva(cpvinited) = 0;

  cva(inEmulatorInit) = 1;
  /* call user defined BgEmulatorInit */
  BgEmulatorInit(arg_argc, arg_argv);
  cva(inEmulatorInit) = 0;

  /* check if all bluegene node size and thread information are set */
  BGARGSCHECK;

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

  return 0;
}

int main(int argc,char *argv[])
{
  ConverseInit(argc,argv,(CmiStartFn)bgMain,0,0);
  return 0;
}

