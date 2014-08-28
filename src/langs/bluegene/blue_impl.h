#ifndef BLUE_IMPL_H
#define BLUE_IMPL_H

#include "conv-mach.h"
#include <stdlib.h>

#include "ckliststring.h"

#include "blue_types.h"
#include "bigsim_timing.h"
#include "bigsim_network.h"

/* alway use handler table per node */
#if ! defined(CMK_BIGSIM_NODE) && ! defined(CMK_BIGSIM_THREAD)
#define CMK_BIGSIM_THREAD   1
#endif

/* define system parameters */
#define INBUFFER_SIZE	32

/* end of system parameters */

#define MAX_HANDLERS	100

class BGMach {
public:
  int x, y, z;             /* size of bluegene nodes in cube */
  int numCth, numWth;      /* number of threads */
  int stacksize;	   /* bg thread stack size */
  int timingMethod;	   /* timing method */
  double cpufactor;	   /* cpu factor to multiply to the time for walltime */
  double fpfactor;         /* fp time factor */
  double timercost;        /* cost of timer */
  int    record, recordnode, replay, replaynode;   /* record/replay */
  CkListString recordprocs;
  CkListString recordnodes;
  char *traceroot;	   /* bgTraceFile prefix */
  BigSimNetwork *network;  /* network setup */
  CkListString procList;   /* a list of processor numbers with projections */
public:
  BGMach() {  nullify(); }
  ~BGMach() { if (network) delete network; }
  void nullify() { 
	x=y=z=0; 
	numCth=numWth=1; stacksize=0; 
        record=recordnode=replay=replaynode=-1;
	timingMethod = BG_WALLTIME; cpufactor=1.0; fpfactor=0.0;
	traceroot=NULL; 
	network=new BlueGeneNetwork;
        timercost = 0.0;
  }
  void setSize(int xx, int yy, int zz) 
	{ x=xx; y=yy; z=zz; }
  void getSize(int *xx, int *yy, int *zz) 
	{ *xx=x; *yy=y; *zz=z; }
  int numTh()
	{ return numCth + numWth; }
  int getNodeSize()  { return x*y*z; }
  int isWorkThread(int tid) { return tid>=0 && tid<numWth; }
  int read(char *file);
  void pup(PUP::er &p) { 
        p|x; p|y; p|z; p|numCth; p|numWth; 
	p|stacksize; p|timingMethod; 
       }
  int traceProjections(int pe);
  void setNetworkModel(char *model);
  int inReplayMode() { return replay != -1 || replaynode != -1; }
};

// simulation state
// one copy per host machine processor  (Cpv)
class SimState {
public:
  // converse handlers
  int msgHandler;
  int nBcastMsgHandler;
  int tBcastMsgHandler;
  int exitHandler;
  int beginExitHandler;
  int bgStatCollectHandler;
  // state variables
  int inEmulatorInit;
  // simulation start timer
  double simStartTime;
};

CpvExtern(BGMach, bgMach);	/* BG machine size */
CpvExtern(SimState, simState);	/* simulation state variables */
CpvExtern(int, numNodes);	/* number of bg nodes on this PE */

typedef char ThreadType;
const char UNKNOWN_THREAD=0, COMM_THREAD=1, WORK_THREAD=2;

typedef bgQueue<int>  	    threadIDQueue;
typedef bgQueue<CthThread>  threadQueue;
typedef bgQueue<char *>     msgQueue;
//typedef CkQ<char *> 	    ckMsgQueue;
// use a queue sorted by recv time
typedef minMsgHeap 	    ckMsgQueue;
typedef CkQ<bgCorrectionMsg *> 	    bgCorrectionQ;
//typedef minHeap<bgCorrectionMsg *> 	    bgCorrectionQ;

/**
  definition of Handler Table;
  there are two kinds of handle tables: 
  one is node level, the other is at thread level
*/
class HandlerTable {
public:
  int          handlerTableCount; 
  BgHandlerInfo *  handlerTable;     
public:
  HandlerTable();
  inline int registerHandler(BgHandler h);
  inline int registerHandlerEx(BgHandlerEx h, void *userPtr);
  inline void numberHandler(int idx, BgHandler h);
  inline void numberHandlerEx(int idx, BgHandlerEx h, void *userPtr);
  inline BgHandlerInfo* getHandle(int handler);
#if 0
  HandlerTable()
  {
    handlerTableCount = 1;
    handlerTable = (BgHandler *)malloc(MAX_HANDLERS * sizeof(BgHandler));
    for (int i=0; i<MAX_HANDLERS; i++) handlerTable[i] = defaultBgHandler;
  }
  inline int registerHandler(BgHandler h)
  {
    ASSERT(!cva(inEmulatorInit));
    /* leave 0 as blank, so it can report error luckily */
    int cur = handlerTableCount++;
    if (cur >= MAX_HANDLERS)
      CmiAbort("BG> HandlerID exceed the maximum.\n");
    handlerTable[cur] = h;
    return cur;
  }
  inline void numberHandler(int idx, BgHandler h)
  {
    ASSERT(!cva(inEmulatorInit));
    if (idx >= handlerTableCount || idx < 1)
      CmiAbort("BG> HandlerID exceed the maximum!\n");
    handlerTable[idx] = h;
  }
  inline BgHandlerInfo getHandle(int handler)
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
#endif
};


#define cva CpvAccess
#define cta CtvAccess

class threadInfo;
CtvExtern(threadInfo *, threadinfo); 
class nodeInfo;
CpvExtern(nodeInfo *, nodeinfo); 
extern double (*timerFunc) (void);

#define tMYID		cta(threadinfo)->id
#define tMYGLOBALID	cta(threadinfo)->globalId
#define tTHREADTYPE	cta(threadinfo)->type
#define tMYNODE		cta(threadinfo)->myNode
#define tSTARTTIME	tMYNODE->startTime
#define tTIMERON	tMYNODE->timeron_flag
#define tCURRTIME	cta(threadinfo)->currTime
#define tHANDLETAB	cta(threadinfo)->handlerTable
#define tHANDLETABFNPTR	cta(threadinfo)->handlerTable.fnPtr
#define tHANDLETABUSERPTR	cta(threadinfo)->handlerTable.userPtr
#define tMYX		tMYNODE->x
#define tMYY		tMYNODE->y
#define tMYZ		tMYNODE->z
#define tMYNODEID	tMYNODE->id
#define tTIMELINEREC	tMYNODE->timelines[tMYID]
#define tTIMELINE	tMYNODE->timelines[tMYID].timeline
#define tINBUFFER	tMYNODE->inBuffer
#define tUSERDATA	tMYNODE->udata
#define tTHREADTABLE    tMYNODE->threadTable
#define tSTARTED        tMYNODE->started

extern int _bgSize;

/*****************************************************************************
   used internally, define BG Node to real Processor mapping
*****************************************************************************/

class BlockMapInfo {
public:
  /* return the number of bg nodes on this physical emulator PE */
  inline static int numLocalNodes()
  {
    int n, m;
    n = _bgSize / CmiNumPes();
    m = _bgSize % CmiNumPes();
    if (CmiMyPe() < m) n++;
    return n;
  }

    /* map global serial number to (x,y,z) ++++ */
  inline static void Global2XYZ(int seq, int *x, int *y, int *z) {
    /** OLD SCHEME:
    *x = seq / (cva(bgMach).y * cva(bgMach).z);
    *y = (seq - *x * cva(bgMach).y * cva(bgMach).z) / cva(bgMach).z;
    *z = (seq - *x * cva(bgMach).y * cva(bgMach).z) % cva(bgMach).z; */

    /* assumed TXYZ */
    *x = seq % cva(bgMach).x;
    *y = (seq % (cva(bgMach).x * cva(bgMach).y)) / cva(bgMach).x;
    *z = seq / (cva(bgMach).x * cva(bgMach).y);
  }


    /* calculate global serial number of (x,y,z) ++++ */
  inline static int XYZ2Global(int x, int y, int z) {
    /** OLD SCHEME:
    return x*(cva(bgMach).y * cva(bgMach).z) + y*cva(bgMach).z + z; */

    /* assumed TXYZ */
    return x + y*cva(bgMach).x + z*(cva(bgMach).x * cva(bgMach).y);
  }

    /* map (x,y,z) to emulator PE ++++ */
  inline static int XYZ2RealPE(int x, int y, int z) {
    return Global2PE(XYZ2Global(x,y,z));
  }

  inline static int XYZ2Local(int x, int y, int z) {
    return Global2Local(XYZ2Global(x,y,z));
  }

    /* local node index number to x y z ++++ */
  inline static void Local2XYZ(int num, int *x, int *y, int *z)  {
    Global2XYZ(Local2Global(num), x, y, z);
  }

#define NOT_FAST_CODE 1
    /* map global serial node number to real PE ++++ */
  inline static int Global2PE(int num) { 
#if NOT_FAST_CODE
    int n = _bgSize/CmiNumPes();
    int bn = _bgSize%CmiNumPes();
    int start = 0; 
    int end = 0;
    for (int i=0; i< CmiNumPes(); i++) {
      end = start + n-1;
      if (i<bn) end++;
      if (num >= start && num <= end) return i;
      start = end+1;
    }
    CmiAbort("Global2PE: unknown pe!");
    return -1;
#else
   int avgNs = _bgSize/CmiNumPes();
   int remains = _bgSize%CmiNumPes();
   /*
    * the bg nodes are mapped like the following:
    * avgNs+1, avgNs+1, ..., avgNs+1 (of remains proccessors)
    * avgNs, ..., avgNs (of CmiNumPes()-remains processors)
    */

   int middleCnt = (avgNs+1)*remains;
   if(num < middleCnt){
      //in the first part of emulating processors
      int ret = num/(avgNs+1);
   #if CMK_ERROR_CHECKING
      if(ret<0){
          CmiAbort("Global2PE: unknown pe!");
          return -1;
      }
   #endif
      return ret;
   }else{
      //in the second part of emulating processors
      int ret = (num-middleCnt)/avgNs+remains;
   #if CMK_ERROR_CHECKING
      if(ret>=CmiNumPes()){
          CmiAbort("Global2PE: unknown pe!");
          return -1;
      }
   #endif
      return ret;
   }
#endif
  }

    /* map global serial node ID to local node array index  ++++ */
  inline static int Global2Local(int num) { 
    int n = _bgSize/CmiNumPes();
    int bn = _bgSize%CmiNumPes();
    int start = 0; 
    int end = 0;
    for (int i=0; i< CmiNumPes(); i++) {
      end = start + n-1;
      if (i<bn) end++;
      if (num >= start && num <= end) return num-start;
      start = end+1;
    }
    CmiAbort("Global2Local:unknown pe!");
    return -1;
  }

    /* map local node index to global serial node id ++++ */
  inline static int Local2Global(int num) { 
    int n = _bgSize/CmiNumPes();
    int bn = _bgSize%CmiNumPes();
    int start = 0; 
    int end = 0;
    for (int i=0; i< CmiMyPe(); i++) {
      end = start + n-1;
      if (i<bn) end++;
      start = end+1;
    }
    return start+num;
  }

  inline static int FileOffset(int pe, int numWth, int numEmulatingPes, int totalWorkerProcs, int &fileNum, int &offset) {
    fileNum = offset = -1;
    return -1;                   /* fix me */
  }
};

class CyclicMapInfo {
public:
  /* return the number of bg nodes on this physical emulator PE */
  inline static int numLocalNodes()
  {
    int n, m;
    n = _bgSize / CmiNumPes();
    m = _bgSize % CmiNumPes();
    if (CmiMyPe() < m) n++;
    return n;
  }

    /* map global serial node number to (x,y,z) ++++ */
  inline static void Global2XYZ(int seq, int *x, int *y, int *z) {
    /** OLD SCHEME:
    *x = seq / (cva(bgMach).y * cva(bgMach).z);
    *y = (seq - *x * cva(bgMach).y * cva(bgMach).z) / cva(bgMach).z;
    *z = (seq - *x * cva(bgMach).y * cva(bgMach).z) % cva(bgMach).z; */

    /* assumed TXYZ */
    *x = seq % cva(bgMach).x;
    *y = (seq % (cva(bgMach).x * cva(bgMach).y)) / cva(bgMach).x;
    *z = seq / (cva(bgMach).x * cva(bgMach).y);
  }


    /* calculate global serial number of (x,y,z) ++++ */
  inline static int XYZ2Global(int x, int y, int z) {
    /** OLD SCHEME:
    return x*(cva(bgMach).y * cva(bgMach).z) + y*cva(bgMach).z + z; */

    /* assumed TXYZ */
    return x + y*cva(bgMach).x + z*(cva(bgMach).x * cva(bgMach).y);
  }

    /* map (x,y,z) to emulator PE ++++ */
  inline static int XYZ2RealPE(int x, int y, int z) {
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

    /* timeline for each worker thread is dump to a bgTrace file.
       All nodes on a emulating processor is dumped to a single file
       this function identify the sequence number of a given PE. 
    */
  inline static int FileOffset(int pe, int numWth, int numEmulatingPes, int totalWorkerProcs, int &fileNum, int &offset) {
    int nodeNum = pe/numWth;
    int numNodes = totalWorkerProcs/numWth;
    fileNum = nodeNum%numEmulatingPes;
    offset=0;

    for(int i=0;i<fileNum;i++)
      offset += (numNodes/numEmulatingPes + ((i < numNodes%numEmulatingPes)?1:0))*numWth;

    offset += (nodeNum/numEmulatingPes)*numWth + pe%numWth;
    return 1;
  }
  virtual ~CyclicMapInfo(){}
};


/*****************************************************************************
      NodeInfo:
        including a group of functions defining the mapping, terms used here:
        XYZ: (x,y,z)
        Global:  map (x,y,z) to a global serial number
        Local:   local index of this nodeinfo in the emulator's node 
*****************************************************************************/
class nodeInfo: public CyclicMapInfo  {
public:
  int id;
  int x,y,z;
  msgQueue     inBuffer;	/* emulate the fix-size inbuffer */
  CmmTable     msgBuffer;	/* buffer when inBuffer is full */
  CthThread   *threadTable;	/* thread table for both work and comm threads*/
  threadInfo  **threadinfo;
  threadQueue *commThQ;		/* suspended comm threads queue */
  ckMsgQueue   nodeQ;		/* non-affinity msg queue */
  ckMsgQueue  *affinityQ;	/* affinity msg queue for each work thread */
  double       startTime;	/* start time for a thread */
  double       nodeTime;	/* node time to coordinate thread times */
  short        lastW;           /* last worker thread assigned msg */
  char         started;		/* flag indicate if this node is started */
  char        *udata;		/* node specific data pointer */
  char 	       timeron_flag;	/* true if timer started */
 
  HandlerTable handlerTable; /* node level handler table */
#if BIGSIM_TIMING
  // for timing
  BgTimeLineRec *timelines;
  bgCorrectionQ cmsg;
#endif
public:
  nodeInfo();
  void initThreads(int _id);  		/* init threads */
  ~nodeInfo();
  /**
   *  add a message to this bluegene node's inbuffer queue
   */
  void addBgNodeInbuffer(char *msgPtr);
  /**
   *  add a message to this bluegene node's non-affinity queue
   */
  void addBgNodeMessage(char *msgPtr);
  /**
   *  called by comm thread to poll inBuffer
   */
  char * getFullBuffer();
};	// end of nodeInfo

/*****************************************************************************
      ThreadInfo:  each thread has a thread private threadInfo structure.
      It has a local id, a global serial id. 
      myNode: point to the nodeInfo it belongs to.
      currTime: is the elapse time for this thread;
      me:   point to the CthThread converse thread handler.
*****************************************************************************/

class BgMessageWatcher;

class threadInfo {
public:
  short id;
  int globalId;
  ThreadType  type;		/* worker or communication thread */
  CthThread me;			/* Converse thread handler */
  nodeInfo *myNode;		/* the node belonged to */
  double  currTime;		/* thread timer */

  BgMessageWatcher *watcher;
  int     cth_serialNo;         /* for record/replay */

  /*
   * It is needed for out-of-core scheduling
   * If it is set to 0, then we know its core file is not on disk
   * and it is first time for this thread to process a msg
   * thus no need to bring it into memory.
   * It is initialized to be 0;
   * It is should be set to 1 when it is taken out of memory to disk
   * and set to 0 when it is brought into memory
   */
    int isCoreOnDisk;
    
    double memUsed;            /* thread's memory footprint (unit is MB) */ 
   
    //Used for AMPI programs
    int startOutOfCore;
    int startOOCChanged;

#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
    //the index into the global array (thdsOOCPreStatus) that records
    //the prefetch status of each worker thread
    int preStsIdx;
#endif

#if  CMK_BIGSIM_THREAD
  HandlerTable   handlerTable;      /* thread level handler table */
#endif

public:
  threadInfo(int _id, ThreadType _type, nodeInfo *_node): 
  	id(_id), globalId(-1), type(_type), myNode(_node), currTime(0.0), 
        watcher(NULL), cth_serialNo(2),
        isCoreOnDisk(0), memUsed(0.0),
	startOutOfCore(1), startOOCChanged(0){}
  inline void setThread(CthThread t) { me = t; }
  inline CthThread getThread() const { return me; }
  virtual void run() { CmiAbort("run not imlplemented"); }

  //=====Begin of stuff related with out-of-core scheduling===========
  void broughtIntoMem();
  void takenOutofMem();
  //=====End of stuff related with out-of-core scheduling=============


}; 

class workThreadInfo : public threadInfo {
private:
  int CsdStopFlag;
public:
  void* reduceMsg;
  
  workThreadInfo(int _id, nodeInfo *_node): 
        threadInfo(_id, WORK_THREAD, _node), reduceMsg(NULL) { 
    CsdStopFlag=0; 
    watcher = NULL;
    if (_id != -1) {
      globalId = nodeInfo::Local2Global(_node->id)*(cva(bgMach).numWth)+_id;
    }
#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
    preStsIdx = _node->id * cva(bgMach).numWth + _id;
#endif
  }
  void addAffMessage(char *msgPtr);        ///  add msg to affinity queue
  void run();
  void scheduler(int count);
  void stopScheduler() { CsdStopFlag++; }
};

class commThreadInfo : public threadInfo {
public:
  commThreadInfo(int _id, nodeInfo *_node): 
     threadInfo(_id, COMM_THREAD, _node) {}
  void run();
};

// functions

double BgGetCurTime();
char ** BgGetArgv();
int     BgGetArgc();
void    startVTimer();
void    stopVTimer();
void    resetVTime();

char * getFullBuffer();
void   addBgNodeMessage(char *msgPtr);
void   addBgThreadMessage(char *msgPtr, int threadID);
void   BgProcessMessageDefault(threadInfo *t, char *msg);
extern void (*BgProcessMessage)(threadInfo *t, char *msg);


/* blue gene debug */

#define BIGSIM_DEBUG 0

#if BIGSIM_DEBUG
/**Controls amount of debug messages: 1 (the lowest priority) is 
extremely verbose, 2 shows most procedure entrance/exits, 
3 shows most communication, and 5 only shows rare or unexpected items.
Displaying lower priority messages doesn't stop higher priority ones.
*/
#define BIGSIM_DEBUG_PRIO 2
#define BIGSIM_DEBUG_LOG 1 /**Controls whether output goes to log file*/

extern FILE *bgDebugLog;
# define BGSTATE_I(prio,args) if ((prio)>=BIGSIM_DEBUG_PRIO) {\
	fprintf args ; fflush(bgDebugLog); }
# define BGSTATE(prio,str) \
	BGSTATE_I(prio,(bgDebugLog,"[%.3f]> "str"\n",CmiWallTimer()))
# define BGSTATE1(prio,str,a) \
	BGSTATE_I(prio,(bgDebugLog,"[%.3f]> "str"\n",CmiWallTimer(),a))
# define BGSTATE2(prio,str,a,b) \
	BGSTATE_I(prio,(bgDebugLog,"[%.3f]> "str"\n",CmiWallTimer(),a,b))
# define BGSTATE3(prio,str,a,b,c) \
	BGSTATE_I(prio,(bgDebugLog,"[%.3f]> "str"\n",CmiWallTimer(),a,b,c))
# define BGSTATE4(prio,str,a,b,c,d) \
	BGSTATE_I(prio,(bgDebugLog,"[%.3f]> "str"\n",CmiWallTimer(),a,b,c,d))
#else
# define BIGSIM_DEBUG_LOG 0
# define BGSTATE(n,x) /*empty*/
# define BGSTATE1(n,x,a) /*empty*/
# define BGSTATE2(n,x,a,b) /*empty*/
# define BGSTATE3(n,x,a,b,c) /*empty*/
# define BGSTATE4(n,x,a,b,c,d) /*empty*/
#endif

#endif
