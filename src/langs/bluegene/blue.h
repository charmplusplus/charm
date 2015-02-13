/*
  File: Blue.h -- header file defines the user API for Converse Bluegene 
        Emulator application. 
  Emulator written by Gengbin Zheng, gzheng@uiuc.edu on 2/20/2001
*/ 

#ifndef BIGSIM_H 
#define BIGSIM_H

#define __BIGSIM__

#include "converse.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
  conform to Converse message header
*/

/**
  macros to access Blue Gene message header fields
*/
  /* small or big work */
#define CmiBgMsgType(m)  (((CmiBlueGeneMsgHeader*)m)->t)
#define CmiBgMsgRecvTime(m)  (((CmiBlueGeneMsgHeader*)m)->rt)
#define CmiBgMsgLength(m)  (((CmiBlueGeneMsgHeader*)m)->n)
#define CmiBgMsgNodeID(m)  (((CmiBlueGeneMsgHeader*)m)->nd)
#define CmiBgMsgThreadID(m)  (((CmiBlueGeneMsgHeader*)m)->tID)
#define CmiBgMsgHandle(m)  (((CmiBlueGeneMsgHeader*)m)->hID)
  /* msg id is local to the source pe */
#define CmiBgMsgID(m)      (((CmiBlueGeneMsgHeader*)m)->msgID)
#define CmiBgMsgSrcPe(m)   (((CmiBlueGeneMsgHeader*)m)->srcPe)
  /* for future use */
#define CmiBgMsgFlag(m)  (((CmiBlueGeneMsgHeader*)m)->flag)
  /* reference counter */
#define CmiBgMsgRefCount(m)  (((CmiBlueGeneMsgHeader*)m)->ref)

  /* a msg is not a full message, with header and a pointer to the real msg */
#define BG_CLONE	0x1

/**
   indicate a message is for any thread;
   when send packets, this means it is a non-affinity message 
*/
#define ANYTHREAD   ((CmiInt2)-1)

/**
   indicate a message is a broacast to all message
*/
#define BG_BROADCASTALL	        -1

/************************* API data structures *************************/
/** 
   define size of a work which helps communication threads schedule 
*/
typedef enum WorkType {LARGE_WORK, SMALL_WORK} WorkType;

/**
   user handler function prototype 
   mimic the Converse Handler data structure
*/
typedef void (*BgHandler) (char *msg);
typedef void (*BgHandlerEx) (char *msg, void *userPtr);

typedef struct {
        BgHandlerEx fnPtr;
        void *userPtr;
} BgHandlerInfo;

/***********  user defined functions called by bluegene ***********/
/** 
   called exactly once per process, used to check argc/argv,
   setup bluegene emulation machine size, number of communication/worker
   threads and register user handler functions
*/
extern void BgEmulatorInit(int argc, char **argv);

/** 
   called on every bluegene node to trigger the computation 
*/
extern void BgNodeStart(int argc, char **argv);

typedef void (*BgStartHandler) (int, char **);

/** 
   register function 'f' to be called first thing in each worker thread
*/
extern void BgSetWorkerThreadStart(BgStartHandler f);

/*********************** API functions ***********************/
/** 
  get a bluegene node coordinate 
*/
void BgGetMyXYZ(int *x, int *y, int *z);
void BgGetXYZ(int pe, int *x, int *y, int *z);

/** 
  get and set blue gene cube size
  set functions can only be called in user's BgGlobalInit code
*/
void BgGetSize(int *sx, int *sy, int *sz);
int  BgNumNodes();	/**<  total number of Blue Gene nodes */
void BgSetSize(int sx, int sy, int sz);
int  BgNodeSize();      /* return the number of nodes on this emulator pe */
int  BgMyRank();	/* node ID, this is local ID */
int  BgMyNode();        /* global node serial number */

int BgNodeToRealPE(int node);         /* return a real processor number from a bg node */
int BgTraceProjectionOn(int pe);    /* true if pe is on for trace projections */

/**
   get and set number of worker and communication thread 
*/
int  BgGetNumWorkThread();
void BgSetNumWorkThread(int num);
int  BgGetNumCommThread();
void BgSetNumCommThread(int num);

/** return thread's local id on the Blue Gene node  */
int  BgGetThreadID();
/** return thread's global id(including both worker and comm threads) */
int  BgGetGlobalThreadID();
/** return thread's global id(including only worker threads) */
int  BgGetGlobalWorkerThreadID();

void BgSetThreadID(int pe);
void BgSetGlobalWorkerThreadID(int pe);
void BgSetNumNodes(int x);

/**
   register user defined function and get a handler, 
   should only be called in BgGlobalInit() 
*/
int  BgRegisterHandler(BgHandler h);
int  BgRegisterHandlerEx(BgHandlerEx h, void *userPtr);
void BgNumberHandler(int, BgHandler h);
void BgNumberHandlerEx(int, BgHandlerEx h, void *userPtr);

/************************ send packet functions ************************/
/**
  Send a packet to a thread in same Blue Gene node
*/
void BgSendLocalPacket(int threadID, int handlerID, WorkType type, 
                       int numbytes, char* data);
#if 0
/**
  Send a packet to a thread(threadID) to Blue Gene node (x,y,z)
*/
void BgSendNonLocalPacket(int x, int y, int z, int threadID, int handlerID, 
                          WorkType type, int numbytes, char* data);
#endif
/**
  Send a packet to a thread(threadID) on Blue Gene node (x,y,z)
  this is a wrapper of above two.
  message "data" will be freed
*/
void BgSendPacket(int x, int y, int z, int threadID, int handlerID, 
                  WorkType type, int numbytes, char* data);

void BgStartStreaming();
void BgEndStreaming();
void BgEnqueue(char *msg);         /* enqueue a msg to local queue */

/************************ collective functions ************************/

/**
  Broadcast a packet to all Blue Gene nodes;
  each BG node receive one message.
*/
void BgBroadcastAllPacket(int handlerID, WorkType type, int numbytes, 
                          char * data);
/**
  Broadcast a packet to all Blue Gene nodes except to "node" and "threadID";
  each BG node receive one message.
*/
void BgBroadcastPacketExcept(int node, CmiInt2 threadID, int handlerID, 
                             WorkType type, int numbytes, char * data);
/**
  Broadcast a packet to all Blue Gene threads 
  each BG thread receive one message.
*/
void BgThreadBroadcastAllPacket(int handlerID, WorkType type, int numbytes, 
                                char * data);
/**
  Broadcast a packet to all Blue Gene threads except to "node" and "threadID"
  each BG thread receive one message.
*/
void BgThreadBroadcastPacketExcept(int node, CmiInt2 threadID, int handlerID, 
                                WorkType type, int numbytes, char * data);

/*
  Multicast a message to a list of processors
*/
void BgSyncListSend(int npes, int *pes, int handlerID, WorkType type, 
				int numbytes, char *data);

void BgSetStrategyBigSimDefault(CthThread t);

/************************ utility functions ************************/

/** 
  shutdown the emulator 
*/
void BgShutdown();

/**
  get Blue Gene timer, one for each thread
*/
double BgGetTime();

/**
   get and set user-defined node private data 
*/
char *BgGetNodeData();
void BgSetNodeData(char *data);

int BgIsMainthread();

/************************ Timing utility functions ************************/

#define BG_ELAPSE      1
#define BG_WALLTIME    2
#define BG_COUNTER     3

/*#define BG_CPUTIMER    1*/

#if BG_CPUTIMER
#define BG_TIMER       CmiCpuTimer
#else
#define BG_TIMER       CmiWallTimer
#endif

typedef void (*bgEventCallBackFn)(void *data, double adjust, double recvT, void *usrPtr);

void BgElapse(double t);
void BgAdvance(double t);
void BgStartCorrection();

/* BG event types */
#define BG_EVENT_PROJ		1
#define BG_EVENT_PRINT		2
#define BG_EVENT_MARK           3

void *BgCreateEvent(int eidx);
void BgEntrySplit(const char *name);
void BgSetEntryName(const char *name, void **log);
void * BgSplitEntry(const char* name, void **parentlogs, int n);
void bgAddProjEvent(void *data, int idx, double t, 
		    bgEventCallBackFn fn, void *uptr, int eType);
void bgUpdateProj(int eType);
double BgGetRecvTime();
void BgResetRecvTime();

/************************ Scheduler  ************************/

void BgScheduler(int ret);
void BgExitScheduler();
void BgDeliverMsgs(int);

/************************ Supporting AMPI ************************/

void BgAttach(CthThread t);
void BgSetStartOutOfCore();
void BgUnsetStartOutOfCore();

/*********************** Record / Replay *************************/
int BgIsRecord();
int BgIsReplay();
void BgRewindRecord();

#if defined(__cplusplus)
}
#endif

/*****************************************************************************
      Node Private variables(Bnv) functions and macros
*****************************************************************************/

#if 0
#define BnvDeclare(T, v)    CpvDeclare(T*, v)=0; 
#define BnvStaticDeclare(T, v)    CpvStaticDeclare(T*, v)=0; 
#define BnvExtern(T, v)    CpvExtern(T*, v); CpvExtern(int, v##_flag_)
#define BnvInitialize(T, v)    \
  do { 	\
    if (CpvAccess(inEmulatorInit)) CmiAbort("BnvInitialize cannot be in BgEmulator\n");	\
    if (BgMyRank() == 0) { /* rank 0 must execute NodeStart() first */ 	\
      CpvInitialize(T*, v); 	 \
      CpvAccess(v) = (T*)malloc(BgNodeSize()*sizeof(T)); 	\
    }\
  } while(0)
#define BnvAccess(v)       CpvAccess(v)[BgMyRank()]
#define BnvAccessOther(v, r)       CpvAccess(v)[r]

#else

extern CmiNodeLock initLock;

#ifdef __cplusplus
template <class d>
class Cnv {
public:
  d **data;
public:
  Cnv(): data(NULL) {}
  inline void init() {
    CmiLock(initLock);
    if (data == NULL) {
      data = new d*[CmiMyNodeSize()];
      CmiAssert(data);
      for (int i=0; i<CmiMyNodeSize(); i++)
        data[i] = new d[BgNodeSize()];
    }
    CmiUnlock(initLock);
  }
  inline int inited() { return data != NULL; }
};
#define BnvDeclare(T,v)        Cnv<T> CMK_CONCAT(Bnv_Var, v); 
#define BnvStaticDeclare(T,v)  static Cnv<T> CMK_CONCAT(Bnv_Var, v); 
#define BnvExtern(T,v)         extern Cnv<T> CMK_CONCAT(Bnv_Var, v);
#define BnvInitialize(T,v)     CMK_CONCAT(Bnv_Var, v).init()
#define BnvInitialized(v)      CMK_CONCAT(Bnv_Var, v).inited()
#define BnvAccess(v)           CMK_CONCAT(Bnv_Var, v).data[CmiMyRank_()][BgMyRank()]
#define BnvAccessOther(v, r)   CMK_CONCAT(Bnv_Var, v).data[CmiMyRank_()][r]
#endif

#endif

#if 1
#ifdef __cplusplus
template <class d>
class Cpv {
public:
  d ***data;
public:
  Cpv(): data(NULL) {}
  inline void init() {
    if (data == NULL) {
      data = new d**[CmiMyNodeSize()];
      for (int i=0; i<CmiMyNodeSize(); i++) {
        data[i] = new d*[BgNodeSize()];
	for (int j=0; j<BgNodeSize(); j++)
	  data[i][j] = new d[BgGetNumWorkThread()];
      }
    }
  }
  inline int inited() { return data != NULL; }
/*
  inline d getThreadData() {
    ASSERT(tTHREADTYPE == WORK_THREAD || tTHREADTYPE == COMM_THREAD);
    ASSERT(!cva(simState).inEmulatorInit);
    threadInfo *tinfo = cta(threadinfo);
    return data[CmiMyRank()][thinfo->myNode->id][thinfo->id];
  }
*/
};
#define BpvDeclare(T,v)        Cpv<T> CMK_CONCAT(Bpv_Var, v); 
#define BpvStaticDeclare(T,v)  static Cpv<T> CMK_CONCAT(Bpv_Var, v); 
#define BpvExtern(T,v)         extern Cpv<T> CMK_CONCAT(Bpv_Var, v);
#define BpvInitialize(T,v)     CMK_CONCAT(Bpv_Var, v).init()
#define BpvInitialized(v)      CMK_CONCAT(Bpv_Var, v).inited()
#define BpvAccess(v)           CMK_CONCAT(Bpv_Var, v).data[CmiMyRank_()][BgMyRank()][BgGetThreadID()]
/*#define BpvAccess(v)           (CMK_CONCAT(Bpv_Var, v).getThreadData())*/
#define BpvAccessOther(v, r)   CMK_CONCAT(Bpv_Var, v).data[CmiMyRank_()][BgMyRank()][r]
#endif

#else
#define BpvDeclare(T, v)            CtvDeclare(T, v)
#define BpvStaticDeclare(T, v)      CtvStaticDeclare(T, v)
#define BpvExtern(T, v)             CtvExtern(T, v)
#define BpvInitialize(T, v)         CtvInitialize(T, v)
#define BpvAccess(v)                CtvAccess(v)
#define BpvAccessOther(v, r)        CtvAccess(v, r)
#endif

#include "blue-conv.h"

typedef void (*BgTracingFn) ();
void BgRegisterUserTracingFunction(BgTracingFn fn);

#endif



