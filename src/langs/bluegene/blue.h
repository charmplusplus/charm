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

typedef struct CMK_MSG_HEADER_BLUEGENE   CmiBlueGeneMsgHeader;
#define CmiBlueGeneMsgHeaderSizeBytes (sizeof(CmiBlueGeneMsgHeader))

#define CmiBgMsgType(m)  (((CmiBlueGeneMsgHeader*)m)->t)
#define CmiBgMsgRecvTime(m)  (((CmiBlueGeneMsgHeader*)m)->rt)
#define CmiBgMsgLength(m)  (((CmiBlueGeneMsgHeader*)m)->n)
#define CmiBgMsgNodeID(m)  (((CmiBlueGeneMsgHeader*)m)->nd)
#define CmiBgMsgThreadID(m)  (((CmiBlueGeneMsgHeader*)m)->tID)
#define CmiBgMsgHandle(m)  (((CmiBlueGeneMsgHeader*)m)->hID)

	/* when send packets, this means it is a non-affinity message */
#define ANYTHREAD   ((CmiUInt2)-1)

#define BG_BROADCASTALL	-1

/* API data structures */
	/* define size of a work which helps communication threads schedule */
typedef enum WorkType {LARGE_WORK, SMALL_WORK} WorkType;

	/* user handler function prototype */
typedef void (*BgHandler) (char *);

/* API functions */
	/* get a bluegene node coordinate */
void BgGetMyXYZ(int *x, int *y, int *z);
void BgGetXYZ(int seq, int *x, int *y, int *z);
	/* get and set blue gene cube size */
	/* set functions can only be called in user's BgGlobalInit code */
void BgGetSize(int *sx, int *sy, int *sz);
int BgGetTotalSize();
void BgSetSize(int sx, int sy, int sz);
int  BgNumNodes();      /* return the number of nodes on this emulator pe */
int  BgMyRank();	/* node ID, this is local ID */
int  BgMyNode();

	/* get and set number of worker and communication thread */
int  BgGetNumWorkThread();
void BgSetNumWorkThread(int num);
int  BgGetNumCommThread();
void BgSetNumCommThread(int num);

int  BgGetThreadID();
int  BgGetGlobalThreadID();

	/* get and set user-defined node private data */
char *BgGetNodeData();
void BgSetNodeData(char *data);

	/* register user defined function and get a handler, 
	   should only be called in BgGlobalInit() */
int  BgRegisterHandler(BgHandler h);
void BgNumberHandler(int, BgHandler h);

double BgGetTime();

	/* send packet functions */
void BgSendLocalPacket(int threadID, int handlerID, WorkType type, int numbytes, char* data);
void BgSendNonLocalPacket(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char* data);
void BgSendPacket(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char* data);

void BgBroadcastAllPacket(int handlerID, WorkType type, int numbytes, char * data);
void BgBroadcastPacketExcept(int node, CmiUInt2 threadID, int handlerID, WorkType type, int numbytes, char * data);

	/** shutdown the emulator */
void BgShutdown();

/* user defined functions called by bluegene */
	/* called exactly once per process, used to check argc/argv,
	   setup bluegene emulation machine size, number of communication/worker
	   threads and register user handler functions */
extern void BgEmulatorInit(int argc, char **argv);
	/* called every bluegene node to trigger the computation */
extern void BgNodeStart(int argc, char **argv);

CpvExtern(int, inEmulatorInit);

#if defined(__cplusplus)
}
#endif

/*****************************************************************************
      Node Private variables macros (Bnv)
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
      data = new (d*)[CmiMyNodeSize()];
      for (int i=0; i<CmiMyNodeSize(); i++)
        data[i] = new d[BgNumNodes()];
    }
  }
};
#define BnvDeclare(T, v)    Cpv<T> CMK_CONCAT(Bnv_Var, v); 
#define BnvStaticDeclare(T, v)    static Cpv<T> CMK_CONCAT(Bnv_Var, v); 
#define BnvExtern(T, v)           extern Cpv<T> CMK_CONCAT(Bnv_Var, v);
#define BnvInitialize(T, v)       CMK_CONCAT(Bnv_Var, v).init()
#define BnvAccess(v)       CMK_CONCAT(Bnv_Var, v).data[CmiMyRank()][BgMyRank()]
#endif

#endif

#define BpvDeclare(T, v)            CtvDeclare(T, v)
#define BpvStaticDeclare(T, v)      CtvStaticDeclare(T, v)
#define BpvExtern(T, v)             CtvExtern(T, v)
#define BpvInitialize(T, v)         CtvInitialize(T, v)
#define BpvAccess(v)                CtvAccess(v)

#endif
