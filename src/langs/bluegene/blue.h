/*
  File: Blue.h -- header file defines the user API for Converse Bluegene 
        Emulator application. 
  Emulator written by Gengbin Zheng, gzheng@uiuc.edu on 2/20/2001
*/ 

#ifndef BLUEGENE_H
#define BLUEGENE_H

#define __BLUEGENE__

#include "converse.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
  conform to Converse message header
*/
typedef struct CMK_MSG_HEADER_BLUEGENE   CmiBlueGeneMsgHeader;
#define CmiBlueGeneMsgHeaderSizeBytes (sizeof(CmiBlueGeneMsgHeader))

/**
  macros to access Blue Gene message header fields
*/
#define CmiBgMsgType(m)  (((CmiBlueGeneMsgHeader*)m)->t)
#define CmiBgMsgRecvTime(m)  (((CmiBlueGeneMsgHeader*)m)->rt)
#define CmiBgMsgLength(m)  (((CmiBlueGeneMsgHeader*)m)->n)
#define CmiBgMsgNodeID(m)  (((CmiBlueGeneMsgHeader*)m)->nd)
#define CmiBgMsgThreadID(m)  (((CmiBlueGeneMsgHeader*)m)->tID)
#define CmiBgMsgHandle(m)  (((CmiBlueGeneMsgHeader*)m)->hID)
#define CmiBgMsgID(m)      (((CmiBlueGeneMsgHeader*)m)->msgID)
#define CmiBgMsgSrcPe(m)   (((CmiBlueGeneMsgHeader*)m)->srcPe)

/**
   indicate a message is for any thread;
   when send packets, this means it is a non-affinity message 
*/
#define ANYTHREAD   ((CmiUInt2)-1)

/**
   indicate a message is a broacast to all message
*/
#define BG_BROADCASTALL	-1

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
int  BgGetTotalSize();	/**<  total Blue Gene nodes */
void BgSetSize(int sx, int sy, int sz);
int  BgNumNodes();      /* return the number of nodes on this emulator pe */
int  BgMyRank();	/* node ID, this is local ID */
int  BgMyNode();

int BgNodeToPE(int node);         /* return a real processor number from a bg node */

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

/**
   register user defined function and get a handler, 
   should only be called in BgGlobalInit() 
*/
int  BgRegisterHandler(BgHandler h);
void BgNumberHandler(int, BgHandler h);
void BgNumberHandlerEx(int, BgHandlerEx h, void *userPtr);

/************************ send packet functions ************************/
/**
  Send a packet to a thread in same Blue Gene node
*/
void BgSendLocalPacket(int threadID, int handlerID, WorkType type, 
                       int numbytes, char* data);
/**
  Send a packet to a thread(threadID) to Blue Gene node (x,y,z)
*/
void BgSendNonLocalPacket(int x, int y, int z, int threadID, int handlerID, 
                          WorkType type, int numbytes, char* data);
/**
  Send a packet to a thread(threadID) to Blue Gene node (x,y,z)
  this is a wrapper of above two.
*/
void BgSendPacket(int x, int y, int z, int threadID, int handlerID, 
                  WorkType type, int numbytes, char* data);

/************************ broadcast functions ************************/

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
void BgBroadcastPacketExcept(int node, CmiUInt2 threadID, int handlerID, 
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
void BgThreadBroadcastPacketExcept(int node, CmiUInt2 threadID, int handlerID, 
                                   WorkType type, int numbytes, char * data);

/************************ utility functions ************************/

/** 
  shutdown the emulator 
*/
void BgShutdown();

/**
  get Blue Gene timer, one for each thread
*/
double BgGetTime();

CpvExtern(int, inEmulatorInit);

/**
   get and set user-defined node private data 
*/
char *BgGetNodeData();
void BgSetNodeData(char *data);

/************************ Timing utility functions ************************/

#define BG_ELAPSE      1
#define BG_WALLTIME    2

typedef void (*bgEventCallBackFn)(void *data, double adjust, double recvT, void *usrPtr);

void BgElapse(double t);

void *BgCreateEvent(int eidx);
void bgAddProjEvent(void *data, double t, bgEventCallBackFn fn);
void bgUpdateProj(void *ptr);

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
      CpvAccess(v) = (T*)malloc(BgNumNodes()*sizeof(T)); 	\
    }\
  } while(0)
#define BnvAccess(v)       CpvAccess(v)[BgMyRank()]
#define BnvAccessOther(v, r)       CpvAccess(v)[r]

#else

#ifdef __cplusplus
template <class d>
class Cpv {
public:
  d **data;
public:
  Cpv(): data(NULL) {}
  inline void init() {
    if (data == NULL) {
      data = new d*[CmiMyNodeSize()];
      for (int i=0; i<CmiMyNodeSize(); i++)
        data[i] = new d[BgNumNodes()];
    }
  }
};
#define BnvDeclare(T,v)        Cpv<T> CMK_CONCAT(Bnv_Var, v); 
#define BnvStaticDeclare(T,v)  static Cpv<T> CMK_CONCAT(Bnv_Var, v); 
#define BnvExtern(T,v)         extern Cpv<T> CMK_CONCAT(Bnv_Var, v);
#define BnvInitialize(T,v)     CMK_CONCAT(Bnv_Var, v).init()
#define BnvAccess(v)       CMK_CONCAT(Bnv_Var, v).data[CmiMyRank()][BgMyRank()]
#define BnvAccessOther(v, r)       CMK_CONCAT(Bnv_Var, v).data[CmiMyRank()][r]
#endif

#endif

#define BpvDeclare(T, v)            CtvDeclare(T, v)
#define BpvStaticDeclare(T, v)      CtvStaticDeclare(T, v)
#define BpvExtern(T, v)             CtvExtern(T, v)
#define BpvInitialize(T, v)         CtvInitialize(T, v)
#define BpvAccess(v)                CtvAccess(v)
#define BpvAccessOther(v, r)        CtvAccess(v, r)

#endif
