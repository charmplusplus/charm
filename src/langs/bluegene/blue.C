// BlueGene simulator Code

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cklists.h"

#include "blue.h"

#define MAX_HANDLERS	16

/* define system parameters */
#define INBUFFER_SIZE	32

#define CYCLES_PER_HOP     5
#define CYCLES_PER_CORNER  75
#define CYCLE_TIME_FACTOR  0.001  /* one cycle = nanosecond = 10^(-3) us */
/* end of system parameters */

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
  double sendTime, recvTime;
  char  first[1];	/* first byte of user data */
public:
  bgMsg() {};
};

typedef bgQueue<int>  	    threadIDQueue;
typedef bgQueue<CthThread>  threadQueue;
typedef bgQueue<bgMsg *>    msgQueue;
typedef CkQ<bgMsg *> 	    ckMsgQueue;

class nodeInfo;

class threadInfo {
public:
  int id;
  ThreadType  type;
  CthThread me;
  nodeInfo *myNode;
  double  currTime;

public:
  threadInfo(int _id, ThreadType _type, nodeInfo *_node): id(_id), type(_type), myNode(_node) {
    currTime=0.0;
  }
  inline void setThread(CthThread t) { me = t; }
  inline CthThread getThread() { return me; }
}; 

#define tMYID		CtvAccess(threadinfo)->id
#define tTHREADTYPE	CtvAccess(threadinfo)->type
#define tMYNODE		CtvAccess(threadinfo)->myNode
#define tSTARTTIME	tMYNODE->startTime
#define tCURRTIME	CtvAccess(threadinfo)->currTime
#define tMYX		tMYNODE->x
#define tMYY		tMYNODE->y
#define tMYZ		tMYNODE->z
#define tMYNODEID	tMYNODE->id
#define tCOMMTHQ	tMYNODE->commThQ
#define tINBUFFER	inBuffer[tMYNODE->id]
#define tMSGBUFFER	msgBuffer[tMYNODE->id]
#define tUSERDATA	tMYNODE->udata
#define tTHREADTABLE    tMYNODE->threadTable
#define tAFFINITYQ      tMYNODE->affinityQ[tMYID]
#define tNODEQ          tMYNODE->nodeQ

CtvDeclare(threadInfo *, threadinfo);	/* represent a bluegene thread */

/* process level variables */
nodeInfo *nodeinfo;		/* represent a bluegene node */

CpvDeclare(int,msgHandler);
CmiHandler msgHandlerFunc(char *msg);

typedef void (*BgStartHandler) (int, char **);
static BgHandler *handlerTable;
static int bgNodeStartHandler;

static int arg_argc;
static char **arg_argv;

static msgQueue *inBuffer;	/* simulate the bluegene inbuffer */
static CmmTable *msgBuffer;	/* if inBuffer is full, put to this buffer */

static int numX, numY, numZ;	/* size of bluegene nodes in cube */
static int numCth, numWth;	/* number of threads */
static int numNodes;		/* number of bg nodes on this PE */

static int inglobalinit=0;

#define ASSERT(x)	if (!(x)) { CmiPrintf("Assert failure at %s:%d\n", __FILE__,__LINE__); CmiAbort("Abort!"); }

#define BGARGSCHECK   	\
  if (numX==0 || numY==0 || numZ==0)  CmiAbort("\nMissing parameters for BlueGene node size!\n<tip> use command line options: +x, +y, or +z.\n");  \
  if (numCth==0 || numWth==0) CmiAbort("\nMissing parameters for number of communication/worker threads!\n<tip> use command line options: +cth or +wth.\n");

#define HANDLERCHECK(handler)	\
  if (handlerTable[handler] == NULL) {	\
    CmiPrintf("Handler %d unregistered!\n", handler);	\
    CmiAbort("Abort!\n");	\
  }


/*****************************************************************************
   used internally, define Queue for scheduler and msgqueue
*****************************************************************************/

/* scheduler queue */
template <class T>
class bgQueue {
private:
  T *data;
  int fp, count, size;
public:
  bgQueue() { fp = count = 0; }
  ~bgQueue() { delete[] data; }
  inline void initialize(int max) {  size = max; data = new T[max]; }
  T deq() {
      T ret = 0;
      if (count > 0) {
        ret = data[fp];
        fp = (fp+1)%size;
        count --;
      }
      return ret;
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
        including a group of functions defining the mapping, terms using here:
        XYZ: (x,y,z)
        Global:  map (x,y,z) to a global serial number
        Local:   local index of this nodeinfo in the node array
*****************************************************************************/

class nodeInfo {
public:
  int id;
  int x,y,z;
  threadQueue *commThQ;
  CthThread *threadTable;
  ckMsgQueue nodeQ;
  ckMsgQueue *affinityQ;
  char *udata;
  double startTime;		/* start time for a thread */

public:
  nodeInfo(): udata(NULL) {
    commThQ = new threadQueue;
    commThQ->initialize(numCth);

    threadTable = new CthThread[numWth+numCth];

    affinityQ = new ckMsgQueue[numWth];
  }

  ~nodeInfo() {
    if (commThQ) delete commThQ;
    delete [] threadTable;
  }
  
  inline static int numLocalNodes()
  {
    int n, m;
    n = (numX * numY * numZ) / CmiNumPes();
    m = (numX * numY * numZ) % CmiNumPes();
    if (CmiMyPe() < m) n++;
    return n;
  }

    /* map (x,y,z) to virtual node ++++ */
  inline static void Global2XYZ(int seq, int *x, int *y, int *z) {
    *x = seq / (numY * numZ);
    *y = (seq - *x * numY * numZ) / numZ;
    *z = (seq - *x * numY * numZ) % numZ;
  }

    /* calculate virtual node number of (x,y,z) ++++ */
  inline static int XYZ2Global(int x, int y, int z) {
    return x*(numY * numZ) + y*numZ + z;
  }

    /* map (x,y,z) to PE ++++ */
  inline static int XYZ2PE(int x, int y, int z) {
    return Global2PE(XYZ2Global(x,y,z));
  }

  inline static int XYZ2Local(int x, int y, int z) {
    return Global2Local(XYZ2Global(x,y,z));
  }

    /* local node number to x y z ++++ */
  inline static void Local2XYZ(int num, int *x, int *y, int *z)  {
    Global2XYZ(Local2Global(num), x, y, z);
  }

    /* map virtual node number to PE ++++ */
  inline static int Global2PE(int num) { return num % CmiNumPes(); }

    /* map virtual node ID to local node array index  ++++ */
  inline static int Global2Local(int num) { return num/CmiNumPes(); }

    /* map local node index to virtual node id ++++ */
  inline static int Local2Global(int num) { return CmiMyPe()+num*CmiNumPes();}

};	// end of nodeInfo


/*****************************************************************************
      low level API
*****************************************************************************/

extern "C" void defaultBgHandler(char *null)
{
  CmiAbort("BG> Invalid Handler called!\n");
}

int BgRegisterHandler(BgHandler h)
{
  /* leave 0 as blank, so it can report error */
  static int count=1;
  int cur = count++;

  ASSERT(inglobalinit);
  if (cur >= MAX_HANDLERS)
    CmiAbort("BG> HandlerID exceed the maximum.\n");
  handlerTable[cur] = h;
  return cur;
}

/* communication thread call getFullBuffer to test if there is data ready 
   in INBUFFER for its own queue */
bgMsg * getFullBuffer()
{
  int tags[1], ret_tags[1];
  char *last_data;
  bgMsg *data, *mb;

  /* I must be a communication thread */
  if (tTHREADTYPE != COMM_THREAD) 
    CmiAbort("GetFullBuffer called by a non-communication thread!\n");

  /* see if we have msg in inBuffer */
  data = tINBUFFER.deq(); 
  if (data) {
    /* since we have at least one slot empty in inbuffer, add from msgbuffer */
    tags[0] = CmmWildCard;
    mb = (bgMsg *)CmmGet(tMSGBUFFER, 1, tags, ret_tags);
    if (mb) tINBUFFER.enq(mb);
  }

  return data;
}

/* add message msgPtr to a bluegene node's inbuffer queue */
void addBgNodeInbuffer(bgMsg *msgPtr, int nodeID)
{
  int tags[1];

  /* if inbuffer is full, store in the msgbuffer */
  if (inBuffer[nodeID].isFull()) {
    tags[0] = nodeID;
    CmmPut(msgBuffer[nodeID], 1, tags, (char *)msgPtr);
  }
  else {
    inBuffer[nodeID].enq(msgPtr);
  }
  CthThread t=nodeinfo[nodeID].commThQ->deq();
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
  ckMsgQueue &que = tNODEQ;
  que.enq(msgPtr);
  for (int i=0; i<numWth; i++)
    if (tMYNODE->affinityQ[i].length() == 0)
    {
      CthAwaken(tTHREADTABLE[i]);
      break;
    }
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
  int nodeID;
  bgMsg *bgmsg;

  bgmsg = (bgMsg *)(msg+CmiMsgHeaderSizeBytes);
  nodeID = nodeInfo::Global2Local(bgmsg->node);
/*
  tags[0] = nodeID;
  CmmPut(inBuffer[nodeID], 1, tags, (char *)bgmsg);
*/
  addBgNodeInbuffer(bgmsg, nodeID);
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
void sendPacket_(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char* data, int local)
{
  void *sendmsg;
  bgMsg *bgmsg;
  int msgSize;
  
  msgSize = CmiMsgHeaderSizeBytes+sizeof(bgMsg)-1+numbytes;
  sendmsg = CmiAlloc(msgSize);
  CmiSetHandler(sendmsg, CpvAccess(msgHandler));
  bgmsg = (bgMsg *)((char *)sendmsg+CmiMsgHeaderSizeBytes);
  bgmsg->node = nodeInfo::XYZ2Global(x,y,z);
  bgmsg->threadID = threadID;
  bgmsg->handlerID = handlerID;
  bgmsg->type = type;
  bgmsg->sendTime = BgGetTime();
  bgmsg->recvTime = MSGTIME(tMYX, tMYY, tMYZ, x,y,z) + bgmsg->sendTime;
  if (numbytes) memcpy(bgmsg->first, data, numbytes);

  if (local)
    addBgNodeInbuffer(bgmsg, tMYNODEID);
  else
    CmiSyncSendAndFree(nodeInfo::XYZ2PE(x,y,z),msgSize,sendmsg);
}

/* sendPacket to route */
/* this function can be called by any thread */
void BgSendNonLocalPacket(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  if (x<0 || y<0 || z<0 || x>=numX || y>=numY || z>=numZ) {
    CmiPrintf("Trying to send packet to a nonexisting node: (%d %d %d)!\n", x,y,z);
    CmiAbort("Abort!\n");
  }

  sendPacket_(x, y, z, threadID, handlerID, type, numbytes, data, 0);
}

void BgSendLocalPacket(int threadID, int handlerID, WorkType type, int numbytes, char * data)
{
  sendPacket_(tMYX, tMYY, tMYZ, threadID, handlerID, type, numbytes, data, 1);
}

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
  ASSERT(!inglobalinit);
  *x = tMYX; *y = tMYY; *z = tMYZ;
}

void BgGetSize(int *sx, int *sy, int *sz)
{
  *sx = numX; *sy = numY; *sz = numZ;
}

/* can only called in globalinit */
void BgSetSize(int sx, int sy, int sz)
{
  ASSERT(inglobalinit);
  numX = sx; numY = sy; numZ = sz;
}

int BgGetThreadID()
{
  ASSERT(tTHREADTYPE == WORK_THREAD || tTHREADTYPE == COMM_THREAD);
  return tMYID;
}

char *BgGetNodeData()
{
  return tUSERDATA;
}

void BgSetNodeData(char *data)
{
  ASSERT(!inglobalinit);
  tUSERDATA = data;
}

int BgGetNumWorkThread()
{
  return numWth;
}

void BgSetNumWorkThread(int num)
{
  ASSERT(inglobalinit);
  numWth = num;
}

int BgGetNumCommThread()
{
  return numCth;
}

void BgSetNumCommThread(int num)
{
  ASSERT(inglobalinit);
  numCth = num;
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

/* TODO: need broadcast */
void BgShutdown()
{
  int i;
  /* TODO: free memory */
  delete [] nodeinfo;
  delete [] inBuffer;
  for (i=0; i<numNodes; i++) CmmFree(msgBuffer[i]);
  delete [] msgBuffer;

  CmiAbort("\nBG> BlueGene simulator shutdown gracefully!\n");
}

/*****************************************************************************
      Communication and Worker threads
*****************************************************************************/

void comm_thread(threadInfo *tinfo)
{
  CthThread worker;
  int workerID;

  /* set the thread-private threadinfo */
  CtvAccess(threadinfo) = tinfo;

  tSTARTTIME = CmiWallTimer();
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
      if (handler == bgNodeStartHandler)
        ((BgStartHandler)handlerTable[handler])(arg_argc, arg_argv);
      else {
        handlerTable[handler](msg->first);
        CmiFree((char *)msg-CmiMsgHeaderSizeBytes); 
      }
    }
    else {
      if (msg->threadID == -1) {
        addBgNodeMessage(msg);
      }
      else {
        addBgThreadMessage(msg, msg->threadID);
      }
/*
      worker = NULL;
      while (!worker) {
        workerID = -1;
	if (tWORKTHQ->isEmpty()) {
          tCURRTIME += (CmiWallTimer()-tSTARTTIME);
          CthYield();
          tSTARTTIME = CmiWallTimer();
	}
	else {
	  workerID = tWORKTHQ->deq();
	  ASSERT(workerID < numCth+numWth && workerID >=0);
	  break;
	}
      }
      tWORKBUFFER(workerID) = msg;
      worker = tTHREADTABLE[workerID];
      CthAwaken(worker);
*/
    }
    /* let other communication thread do their jobs */
    tCURRTIME += (CmiWallTimer()-tSTARTTIME);
    CthYield();
    tSTARTTIME = CmiWallTimer();
  }
}

void work_thread(threadInfo *tinfo)
{
  int id;
  int handler;

  CtvAccess(threadinfo) = tinfo;
  id = tMYID;

  tSTARTTIME = CmiWallTimer();
  for (;;) {
    bgMsg *m1=NULL, *m2=NULL;
    bgMsg *msg=NULL;
    ckMsgQueue &q1 = tNODEQ;
    ckMsgQueue &q2 = tAFFINITYQ;
    if (!q1.isEmpty())  m1 = q1[0];
    if (!q2.isEmpty()) m2 = q2[0];

    if (!m1 && m2) { msg = m2; q2.deq(); }
    else if (!m2 && m1) { msg = m1; q1.deq(); }
    else if (m1 && m2) {
      if (m1->recvTime < m2->recvTime) {
        msg = m1; q1.deq();
      }
      else {
        msg = m2; q2.deq();
      }
    }
    /* if no msg is ready, put it to sleep in workQueue */
    if ( msg == NULL ) {
      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      CthSuspend();
      tSTARTTIME = CmiWallTimer();
      continue;
    }
    if (msg->recvTime > tCURRTIME)  tCURRTIME = msg->recvTime;
    handler = msg->handlerID;
    
    /* call user registered handler function */
    handlerTable[handler](msg->first);
      /* free the msg and clear the buffer */
      /* ??? delete ??? */
    CmiFree((char *)msg-CmiMsgHeaderSizeBytes); 
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

  /* creat computation threads */
  for (i=0; i< numWth; i++)
  {
    threadInfo *tinfo = new threadInfo(i, WORK_THREAD, ninfo);
    t = CthCreate((CthVoidFn)work_thread, tinfo, 0);
    tinfo->setThread(t);
    /* put to thread table */
    tTHREADTABLE[tinfo->id] = t;
    CthAwaken(t);
  }

  /* creat communication thread */
  for (i=0; i< numCth; i++)
  {
    threadInfo *tinfo = new threadInfo(i+numWth, COMM_THREAD, ninfo);
    t = CthCreate((CthVoidFn)comm_thread, tinfo, 0);
    tinfo->setThread(t);
    /* put to thread table */
    tTHREADTABLE[tinfo->id] = t;
    CthAwaken(t);
  }

  /* send a null message to itself to start bgnode */ 
  BgSendLocalPacket(-1, bgNodeStartHandler, SMALL_WORK, 0, NULL);
}

CmiStartFn mymain(int argc, char **argv)
{
  int i;

  numX = numY = numZ = numCth = numWth = 0;

  /* initialize all processor level data */
  CmiGetArgInt(argv, "+x", &numX);
  CmiGetArgInt(argv, "+y", &numY);
  CmiGetArgInt(argv, "+z", &numZ);
  CmiGetArgInt(argv, "+cth", &numCth);
  CmiGetArgInt(argv, "+wth", &numWth);

  arg_argv = argv;
  arg_argc = CmiGetArgc(argv);

  /* msg handler */
  CpvInitialize(int,msgHandler);
  CpvAccess(msgHandler) = CmiRegisterHandler((CmiHandler) msgHandlerFunc);

  /* init handlerTable */
  handlerTable = (BgHandler *)malloc(MAX_HANDLERS * sizeof(BgHandler));
  for (i=0; i<MAX_HANDLERS; i++) handlerTable[i] = defaultBgHandler;

  /* register user must defined BgNodeStart */
  inglobalinit = 1;
  bgNodeStartHandler = BgRegisterHandler((BgHandler)BgNodeStart);

  /* call user defined BgEmulatorInit */
  BgEmulatorInit(arg_argc, arg_argv);
  inglobalinit = 0;

  /* check if all bluegene node size and thread information are set */
  BGARGSCHECK;

  CtvInitialize(threadInfo *, threadinfo);

  /* number of bg nodes on this PE */
  numNodes = nodeInfo::numLocalNodes();

  inBuffer = new bgQueue<bgMsg*>[numNodes];
  for (i=0; i<numNodes; i++) inBuffer[i].initialize(INBUFFER_SIZE);
  msgBuffer = new CmmTable[numNodes];
  for (i=0; i<numNodes; i++) msgBuffer[i] = CmmNew();

  /* create BG nodes */
  nodeinfo = new nodeInfo[numNodes];
  CtvAccess(threadinfo) = new threadInfo(-1, UNKNOWN_THREAD, NULL);
  for (i=0; i<numNodes; i++)
  {
    nodeInfo *ninfo = nodeinfo + i;
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

