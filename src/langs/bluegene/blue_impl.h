#ifndef BLUE_IMPL_H
#define BLUE_IMPL_H

#include "conv-mach.h"
#include <stdlib.h>

#include "blue_types.h"
#include "blue_timing.h"

/* alway use handler table per node */
#if ! defined(CMK_BLUEGENE_NODE) && ! defined(CMK_BLUEGENE_THREAD)
#define CMK_BLUEGENE_THREAD   1
#endif

/* define system parameters */
#define INBUFFER_SIZE	32

#define CYCLES_PER_HOP     5
#define CYCLES_PER_CORNER  75
#define CYCLE_TIME_FACTOR  0.001  /* one cycle = nanosecond = 10^(-3) us */
/* end of system parameters */

#define MAX_HANDLERS	100

CpvStaticDeclare(int, numX);	/* size of bluegene nodes in cube */
CpvStaticDeclare(int, numY);
CpvStaticDeclare(int, numZ);
CpvStaticDeclare(int, numCth);	/* number of threads */
CpvStaticDeclare(int, numWth);
CpvStaticDeclare(int, numNodes);	/* number of bg nodes on this PE */

typedef char ThreadType;
const char UNKNOWN_THREAD=0, COMM_THREAD=1, WORK_THREAD=2;

typedef bgQueue<int>  	    threadIDQueue;
typedef bgQueue<CthThread>  threadQueue;
typedef bgQueue<char *>     msgQueue;
//typedef CkQ<char *> 	    ckMsgQueue;
// use a queue sorted by recv time
typedef minMsgHeap 	    ckMsgQueue;
typedef CkQ<bgCorrectionMsg *> 	    bgCorrectionQ;

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
  HandlerTable();
  inline int registerHandler(BgHandler h);
  inline void numberHandler(int idx, BgHandler h);
  inline BgHandler getHandle(int handler);
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
  inline BgHandler getHandle(int handler)
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

#define tMYID		cta(threadinfo)->id
#define tMYGLOBALID	cta(threadinfo)->globalId
#define tTHREADTYPE	cta(threadinfo)->type
#define tMYNODE		cta(threadinfo)->myNode
#define tSTARTTIME	tMYNODE->startTime
#define tCURRTIME	cta(threadinfo)->currTime
#define tHANDLETAB	cta(threadinfo)->handlerTable
#define tMYX		tMYNODE->x
#define tMYY		tMYNODE->y
#define tMYZ		tMYNODE->z
#define tMYNODEID	tMYNODE->id
#define tCOMMTHQ	tMYNODE->commThQ
#define tTIMELINE	tMYNODE->timelines[tMYID]
#define tINBUFFER	cva(inBuffer)[tMYNODE->id]
#define tMSGBUFFER	cva(msgBuffer)[tMYNODE->id]
#define tUSERDATA	tMYNODE->udata
#define tTHREADTABLE    tMYNODE->threadTable
#define tAFFINITYQ      tMYNODE->affinityQ[tMYID]
#define tNODEQ          tMYNODE->nodeQ
#define tSTARTED        tMYNODE->started

extern int bgSize;

/*****************************************************************************
   used internally, define BG Node to real Processor mapping
*****************************************************************************/

class BlockMapInfo {
public:
  /* return the number of bg nodes on this physical emulator PE */
  inline static int numLocalNodes()
  {
    int n, m;
    n = bgSize / CmiNumPes();
    m = bgSize % CmiNumPes();
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
  inline static int Global2PE(int num) { 
    int n = bgSize/CmiNumPes();
    int bn = bgSize%CmiNumPes();
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
  }

    /* map global serial node ID to local node array index  ++++ */
  inline static int Global2Local(int num) { 
    int n = bgSize/CmiNumPes();
    int bn = bgSize%CmiNumPes();
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
    int n = bgSize/CmiNumPes();
    int bn = bgSize%CmiNumPes();
    int start = 0; 
    int end = 0;
    for (int i=0; i< CmiMyPe(); i++) {
      end = start + n-1;
      if (i<bn) end++;
      start = end+1;
    }
    return start+num;
  }
};

class CyclicMapInfo {
public:
  /* return the number of bg nodes on this physical emulator PE */
  inline static int numLocalNodes()
  {
    int n, m;
    n = bgSize / CmiNumPes();
    m = bgSize % CmiNumPes();
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
  threadQueue *commThQ;		/* suspended comm threads queue */
  CthThread   *threadTable;	/* thread table for both work and comm threads*/
  threadInfo  **threadinfo;
  ckMsgQueue   nodeQ;		/* non-affinity msg queue */
  ckMsgQueue  *affinityQ;	/* affinity msg queue for each work thread */
  double       startTime;	/* start time for a thread */
  double       nodeTime;	/* node time to coordinate thread times */
  short        lastW;           /* last worker thread assigned msg */
  char         started;		/* flag indicate if this node is started */
  char        *udata;		/* node specific data pointer */
 
  HandlerTable handlerTable; /* node level handler table */
#if BLUEGENE_TIMING
  // for timing
  BgTimeLine *timelines;
  bgCorrectionQ cmsg;
#endif
public:
  nodeInfo();
  ~nodeInfo();
#if 0
  ~nodeInfo() {
    if (commThQ) delete commThQ;
    delete [] affinityQ;
    delete [] threadTable;
    delete [] threadinfo;
  }
#endif
  
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
  threadInfo(int _id, ThreadType _type, nodeInfo *_node): 
  	id(_id), type(_type), myNode(_node), currTime(0.0) 
  {
//    if (id != -1) globalId = nodeInfo::Local2Global(_node->id)*(cva(numCth)+cva(numWth))+_id;
  }
  inline void setThread(CthThread t) { me = t; }
  inline const CthThread getThread() const { return me; }
}; 

extern double BgGetCurTime();

#endif
