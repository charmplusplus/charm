#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "blue.h"

#define MAX_HANDLERS	16

/* define system parameters */
#define INBUFFER_SIZE	32

#define CYCLES_PER_HOP     5
#define CYCLES_PER_CORNER  75
#define CYCLE_TIME_FACTOR  0.001  /* one cycle = nanosecond = 10^(-3) us */


/* scheduler queue */
typedef struct bgQueueTag {
  void **data;
  int fp, count, size;
} bgQueue;

typedef enum ThreadType {COMM_THREAD=1, WORK_THREAD} ThreadType;

/* converse message send to other bgnodes */
typedef struct bgMsgTag {
  int node;		/* virtual node number */
  int handlerID;
  WorkType type;
  double sendTime, recvTime;
  char  first[1];	/* first byte of user data */
} bgMsg;

typedef struct nodeinfoTag {
  int id;
  int x,y,z;
  bgMsg **workBuffer;		/* each worker thread has a workBuffer entry */
  bgQueue *workThQ;
  char *udata;
} nodeInfo;

typedef struct threadinfoTag {
  int id;
  ThreadType  type;
  nodeInfo *myNode;
  double  currTime, startTime;
} threadInfo;

#define tMYID		CtvAccess(threadinfo)->id
#define tTHREADTYPE	CtvAccess(threadinfo)->type
#define tMYNODE		CtvAccess(threadinfo)->myNode
#define tSTARTTIME	CtvAccess(threadinfo)->startTime
#define tCURRTIME	CtvAccess(threadinfo)->currTime
#define tMYX		tMYNODE->x
#define tMYY		tMYNODE->y
#define tMYZ		tMYNODE->z
#define tMYNODEID	tMYNODE->id
#define tWORKTHQ	tMYNODE->workThQ
#define tWORKBUFFER(id)	tMYNODE->workBuffer[id]
#define tINBUFFER	inBuffer[tMYNODE->id]
#define tMSGBUFFER	msgBuffer[tMYNODE->id]
#define tUSERDATA	tMYNODE->udata

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

static bgQueue **inBuffer;	/* simulate the bluegene inbuffer */
static CmmTable *msgBuffer;	/* if inBuffer is full, put to this buffer */
static CmmTable threadTable;
static CmmTable matchTable;

static int numX, numY, numZ;	/* size of bluegene nodes in cube */
static int numCth, numWth;	/* number of threads */
static int numNodes;		/* number of bg nodes on this PE */

static int inglobalinit=0;

#define ASSERT(x)	if (!(x)) { CmiPrintf("Assert failure at %s:%d\n", __FILE__,__LINE__); CmiAbort("Abort!"); }

#define BGARGSCHECK   	\
  if (numX==0 || numY==0 || numZ==0)  CmiAbort("\nMissing parameters for BlueGene node size!\n<tip> use command line options: +x, +y, or +z.\n");  \
  if (numCth==0 || numWth==0) CmiAbort("\nMissing parameters for number of communication/work threads!\n<tip> use command line options: +cth or +wth.\n");

#define HANDLERCHECK(handler)	\
  if (handlerTable[handler] == NULL) {	\
    CmiPrintf("Handler %d unregistered!\n", handler);	\
    CmiAbort("Abort!\n");	\
  }

/*****************************************************************************
  these functions define the mapping of BG node to physical simulating 
  processors
         (x,y,z) <=> seq
*****************************************************************************/

/* calculate virtual node number of (x,y,z) ++++ */
int XYZ2Virtual(int x, int y, int z)
{
  return x*(numY * numZ) + y*numZ + z;
}

/* map (x,y,z) to virtual node ++++ */
void Virtual2XYZ(int seq, int *x, int *y, int *z)
{
  *x = seq / (numY * numZ);
  *y = (seq - *x * numY * numZ) / numZ;
  *z = (seq - *x * numY * numZ) % numZ;
}

/* map virtual node number to PE ++++ */
#define Virtual2PE(num) ((num) % CmiNumPes())

/* map virtual node ID to local node array index  ++++ */
#define Virtual2Local(num) ((num)/CmiNumPes())

/* map local node index to virtual node id ++++ */
#define local2Virtual(num)  (CmiMyPe() + (num)*CmiNumPes())

/* map (x,y,z) to PE ++++ */
int XYZ2PE(int x, int y, int z)
{
  return Virtual2PE(XYZ2Virtual(x,y,z));
}

/* local node number to x y z ++++ */
void local2XYZ(int num, int *x, int *y, int *z)  
{
  Virtual2XYZ(local2Virtual(num), x, y, z);
}

int XYZ2local(int x, int y, int z)
{
  return Virtual2Local(XYZ2Virtual(x,y,z));
}

int numLocalNodes()
{
  int n, m;
  n = (numX * numY * numZ) / CmiNumPes();
  m = (numX * numY * numZ) % CmiNumPes();
  if (CmiMyPe() < m) n++;
  return n;
}

/*****************************************************************************
   used internally, define Queue for scheduler and msgqueue
*****************************************************************************/

bgQueue* newbgQueue(int max)
{
  bgQueue *que = (bgQueue *)malloc(sizeof(bgQueue));
  que->size = max;
  que->data = (void **)malloc(max*sizeof(void *));
  que->fp = que->count = 0;
  return que;
}

void* bgDequeue(bgQueue *que)
{
  void *ret = NULL;
  if (que->count > 0) {
    ret = que->data[que->fp];
    que->fp = (que->fp+1)%que->size;
    que->count --;
  }
  return ret;
}

void bgEnqueue(bgQueue *que, void *item)
{
  ASSERT(que->count < que->size);
  que->data[(que->fp+que->count)%que->size] = item;
  que->count ++;
}

int bgQueueFull(bgQueue *que)
{
  return que->count==que->size;
}

int bgQueueEmpty(bgQueue *que)
{
  return que->count==0;
}

/*****************************************************************************
      internal thread register and lookup functions
*****************************************************************************/
void registerThread(CthThread t, int id)
{
    int tags[2];
    tags[0] = id; tags[1] = (size_t)t;
    CmmPut(threadTable, 2, tags, &t);
}

int getThreadID(CthThread th)
{
  int tags[2], ret_tags[2];
  tags[0] = CmmWildCard; tags[1] = (size_t)th;
  if (CmmProbe(threadTable, 2, tags, ret_tags)) return ret_tags[0];
  return -1;
}

/*****************************************************************************
      low level API
*****************************************************************************/

static void defaultHandler(char *null)
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

/* communication thread call GetFullBuffer to test if there is data ready 
   for its own queue */
char * GetFullBuffer()
{
  int tags[1], ret_tags[1];
  char *last_data;
  char *data, *mb;
  int id = tMYID;

  /* I must be a communication thread */
  if (tTHREADTYPE != COMM_THREAD) {
    CmiAbort("GetFullBuffer called by a non-communication thread!\n");
  }

  /* free the data returned last time */
/*
  tags[0] = id;
  last_data = (char *)CmmGet(matchTable, 1, tags, ret_tags);
  if (last_data) CmiFree(last_data-CmiMsgHeaderSizeBytes);
*/

  /* see if we have msg in inBuffer */
  data = (char *)bgDequeue(tINBUFFER);
  if (data) {
    /* since we have at least one slot empty in inbuffer, add from msgbuffer */
    tags[0] = CmmWildCard;
    mb = (char *)CmmGet(tMSGBUFFER, 1, tags, ret_tags);
    if (mb) bgEnqueue(tINBUFFER, (void *)mb);
  }

/*
  if (data) {
      tags[0] = id;
      CmmPut(matchTable, 1, tags, data);
  }
*/
  return data;
}

/* add message msgPtr to a bluegene node's inbuffer queue */
void addBgNodeMessage(bgMsg *msgPtr, int nodeID)
{
  int tags[1];

  tags[0] = nodeID;
  /* if inbuffer is full, store in the msgbuffer */
  if (bgQueueFull(inBuffer[nodeID])) {
    CmmPut(msgBuffer[nodeID], 1, tags, (char *)msgPtr);
  }
  else {
    bgEnqueue(inBuffer[nodeID], (void *)msgPtr);
  }
}

int checkReady()
{
  if (tTHREADTYPE != COMM_THREAD)
    CmiAbort("checkReady called by a non-communication thread!\n");
  return !bgQueueEmpty(tINBUFFER);
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
void sendPacket_(int x, int y, int z, int handlerID, WorkType type, int numbytes, char* data, int local)
{
  void *sendmsg;
  bgMsg *bgmsg;
  int msgSize;
  
  msgSize = CmiMsgHeaderSizeBytes+sizeof(bgMsg)-1+numbytes;
  sendmsg = CmiAlloc(msgSize);
  CmiSetHandler(sendmsg, CpvAccess(msgHandler));
  bgmsg = (bgMsg *)((char *)sendmsg+CmiMsgHeaderSizeBytes);
  bgmsg->node = XYZ2Virtual(x,y,z);
  bgmsg->handlerID = handlerID;
  bgmsg->type = type;
  bgmsg->sendTime = BgGetThreadTime();
  bgmsg->recvTime = MSGTIME(tMYX, tMYY, tMYZ, x,y,z) + bgmsg->sendTime;
  if (numbytes) memcpy(bgmsg->first, data, numbytes);

  if (local)
    addBgNodeMessage(bgmsg, tMYNODEID);
  else
    CmiSyncSendAndFree(XYZ2PE(x,y,z),msgSize,sendmsg);
}

/* sendPacket to route */
/* this function can be called by any thread */
void BgSendNonLocalPacket(int x, int y, int z, int handlerID, WorkType type, int numbytes, char * data)
{
  sendPacket_(x, y, z, handlerID, type, numbytes, data, 0);
}

void BgSendLocalPacket(int handlerID, WorkType type, int numbytes, char * data)
{
  sendPacket_(tMYX, tMYY, tMYZ, handlerID, type, numbytes, data, 1);
}

void BgSendPacket(int x, int y, int z, int handlerID, WorkType type, int numbytes, char * data)
{
  if (tMYX == x && tMYY==y && tMYZ==z)
    BgSendLocalPacket(handlerID, type, numbytes, data);
  else
    BgSendNonLocalPacket(x,y,z,handlerID, type, numbytes, data);
}


/*****************************************************************************
      BG node level API
*****************************************************************************/

/* must be called in a communication or worker thread */
void BgGetXYZ(int *x, int *y, int *z)
{
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

int BgGetWorkThreadID()
{
  if (tTHREADTYPE != WORK_THREAD) {
    CmiAbort("GetWorkThreadID called by a non-worker thread!\n");
  }
  return tMYID - numCth;
}

char *BgGetNodeData()
{
  return tUSERDATA;
}

void BgSetNodeData(char *data)
{
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

double BgGetThreadTime()
{
  /* accumulate time since last starttime, and reset starttime */
  double tp2= CmiWallTimer();
  tCURRTIME += (tp2 - tSTARTTIME);
  tSTARTTIME = tp2;
  return tCURRTIME;
}

/* TODO: need broadcast */
void BgShutdown()
{
  /* TODO: free memory */

  CmiAbort("\nBG> BlueGene simulator shutdown gracefully!");
}

/*****************************************************************************
      Communication and Worker threads
*****************************************************************************/

void comm_thread(threadInfo *tinfo)
{
  CthThread worker;

  /* set the thread-private threadinfo */
  CtvAccess(threadinfo) = tinfo;

  tCURRTIME = 0.0;
  tSTARTTIME = CmiWallTimer();
  for (;;) {
    bgMsg *msg = (bgMsg *)GetFullBuffer();
    if (!msg) { 
      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      CthYield(); 
      tSTARTTIME = CmiWallTimer();
      continue;
    }
    /* schedule a worker thread, if small work do it itself */
    if (msg->type == SMALL_WORK) {
      if (msg->recvTime > tCURRTIME)  tCURRTIME = msg->recvTime;
      /* call user registered handler function */
      handlerTable[msg->handlerID](msg->first);
      CmiFree((char *)msg-CmiMsgHeaderSizeBytes); 
    }
    else {
      worker = NULL;
      while (!worker) {
        worker = (CthThread)bgDequeue(tWORKTHQ);
        if (!worker) {
          tCURRTIME += (CmiWallTimer()-tSTARTTIME);
          CthYield();
          tSTARTTIME = CmiWallTimer();
        }
        else break;
      }
      tWORKBUFFER(getThreadID(worker)-numCth) = msg;
      CthAwaken(worker);
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
  id = tMYID - numCth;

  tCURRTIME = 0.0;
  tSTARTTIME = CmiWallTimer();
  for (; ;) {
    bgMsg *msg=tWORKBUFFER(id);
    /* if no msg is ready, put it to sleep in workQueue */
    if ( msg == NULL ) {
      bgEnqueue(tWORKTHQ, CthSelf());
      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      CthSuspend();
      tSTARTTIME = CmiWallTimer();
      continue;
    }
    if (msg->recvTime > tCURRTIME)  tCURRTIME = msg->recvTime;
    handler = msg->handlerID;
    
    /* call user registered handler function */
    if (handler == bgNodeStartHandler)
      ((BgStartHandler)handlerTable[handler])(arg_argc, arg_argv);
    else {
      handlerTable[handler](msg->first);
      /* free the msg and clear the buffer */
      /* ??? delete ??? */
      CmiFree((char *)msg-CmiMsgHeaderSizeBytes); 
    }   
    tWORKBUFFER(id) = NULL;
  }
}

/* handler to process the msg */
CmiHandler msgHandlerFunc(char *msg)
{
  int nodeID;
  bgMsg *bgmsg;

  bgmsg = (bgMsg *)(msg+CmiMsgHeaderSizeBytes);
  nodeID = Virtual2Local(bgmsg->node);
/*
  tags[0] = nodeID;
  CmmPut(inBuffer[nodeID], 1, tags, (char *)bgmsg);
*/
  addBgNodeMessage(bgmsg, nodeID);
  return 0;
}

/* should be done only once per bg node */
void BgNodeInitialize(nodeInfo *ninfo)
{
  CthThread t;
  int i;

  /* initialize all nodeinfo data */
  ninfo->workThQ = newbgQueue(numWth);

  ninfo->workBuffer = (bgMsg **)malloc(numWth*sizeof(bgMsg *));
  for (i=0; i<numWth; i++) ninfo->workBuffer[i] = NULL;

  /* creat communication thread */
  for (i=0; i< numCth; i++)
  {
    threadInfo *tinfo = (threadInfo *)malloc(sizeof(threadInfo));
    tinfo->type = COMM_THREAD;
    tinfo->id = i;
    tinfo->myNode = ninfo;
    t = CthCreate((CthVoidFn)comm_thread, tinfo, 0);
    /* put to thread table */
    registerThread(t, i);
    CthAwaken(t);
  }

  /* creat computation threads */
  for (i=0; i< numWth; i++)
  {
    threadInfo *tinfo = (threadInfo *)malloc(sizeof(threadInfo));
    tinfo->type = WORK_THREAD;
    tinfo->id = i+numCth;
    tinfo->myNode = ninfo;
    t = CthCreate((CthVoidFn)work_thread, tinfo, 0);
    /* put to thread table */
    registerThread(t, i+numCth);
    CthAwaken(t);
  }

  /* send a null message to itself to start bgnode */ 
  BgSendLocalPacket(bgNodeStartHandler, LARGE_WORK, 0, NULL);
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

  arg_argc = argc;
  arg_argv = argv;

  /* msg handler */
  CpvInitialize(int,msgHandler);
  CpvAccess(msgHandler) = CmiRegisterHandler((CmiHandler) msgHandlerFunc);

  /* init handlerTable */
  handlerTable = (BgHandler *)malloc(MAX_HANDLERS * sizeof(BgHandler));
  for (i=0; i<MAX_HANDLERS; i++) handlerTable[i] = defaultHandler;

  /* register user must defined BgNodeStart */
  inglobalinit = 1;
  bgNodeStartHandler = BgRegisterHandler((BgHandler)BgNodeStart);

  /* call user defined BgGlobalInit */
  BgGlobalInit(arg_argc, arg_argv);
  inglobalinit = 0;

  /* check if all bluegene node size and thread information are set */
  BGARGSCHECK;

  CtvInitialize(threadInfo *, threadinfo);

  matchTable = CmmNew();
  threadTable = CmmNew();

  /* number of bg nodes on this PE */
  numNodes = numLocalNodes();

  inBuffer = (bgQueue **)malloc(numNodes * sizeof(bgQueue *));
  for (i=0; i<numNodes; i++) inBuffer[i] = newbgQueue(INBUFFER_SIZE);
  msgBuffer = (CmmTable *)malloc(numNodes * sizeof(CmmTable));
  for (i=0; i<numNodes; i++) msgBuffer[i] = CmmNew();

  /* create BG nodes */
  nodeinfo = (nodeInfo *)malloc(numNodes * sizeof(nodeInfo));
  CtvAccess(threadinfo) = (threadInfo *)malloc(sizeof(threadInfo));
  for (i=0; i<numNodes; i++)
  {
    nodeInfo *ninfo = nodeinfo + i;
    ninfo->id = i;
    local2XYZ(i, &ninfo->x, &ninfo->y, &ninfo->z);

    /* pretend that I am a thread */
    CtvAccess(threadinfo)->myNode = ninfo;

    /* initialize a BG node and fire all threads */
    BgNodeInitialize(ninfo);
  }

  return 0;
}

main(int argc,char *argv[])
{
  ConverseInit(argc,argv,(CmiStartFn)mymain,0,0);
}

