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

#define CmiBgGetType(m)  (((CmiBlueGeneMsgHeader*)m)->t)
#define CmiBgGetRecvTime(m)  (((CmiBlueGeneMsgHeader*)m)->rt)
#define CmiBgGetLength(m)  (((CmiBlueGeneMsgHeader*)m)->n)
#define CmiBgGetNodeID(m)  (((CmiBlueGeneMsgHeader*)m)->nd)
#define CmiBgGetThreadID(m)  (((CmiBlueGeneMsgHeader*)m)->tID)
#define CmiBgGetHandle(m)  (((CmiBlueGeneMsgHeader*)m)->hID)

	/* when send packets, this means it is a non-affinity message */
#define ANYTHREAD   ((CmiUInt2)-1)

/* API data structures */
	/* define size of a work which helps communication threads schedule */
typedef enum WorkType {LARGE_WORK, SMALL_WORK} WorkType;

	/* user handler function prototype */
typedef void (*BgHandler) (char *);

/* API functions */
	/* get a bluegene node coordinate */
void BgGetXYZ(int *x, int *y, int *z);
	/* get and set blue gene cube size */
	/* set functions can only be called in user's BgGlobalInit code */
void BgGetSize(int *sx, int *sy, int *sz);
int BgGetTotalSize();
void BgSetSize(int sx, int sy, int sz);
int  BgNumNodes();      /* return the number of nodes on this emulator pe */
int  BgMyRank();	/* node ID, this is local ID */

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

	/** shutdown the emulator */
void BgShutdown();

/* user defined functions called by bluegene */
	/* called exactly once per process, used to check argc/argv,
	   setup bluegene emulation machine size, number of communication/worker
	   threads and register user handler functions */
extern void BgEmulatorInit(int argc, char **argv);
	/* called every bluegene node to trigger the computation */
extern void BgNodeStart(int argc, char **argv);

/*****************************************************************************
      Node Private variables macros (Bnv)
*****************************************************************************/

CpvExtern(int, inEmulatorInit);
CpvExtern(int, cpvinited);

#define BnvDeclare(T, v)    CpvDeclare(T*, v)
#define BnvStaticDeclare(T, v)    CpvStaticDeclare(T*, v)
#define BnvExtern(T, v)    CpvExtern(T*, v)
#define BnvInitialize(T, v)    \
  do { 	\
    if (CpvAccess(inEmulatorInit)) CmiAbort("BnvInitialize cannot be in BgEmulator\n");	\
    if (CpvAccess(cpvinited)==0) {	\
      CpvInitialize(T*, v);	\
      CpvAccess(v) = (T*)malloc(BgNumNodes()*sizeof(T)); } \
  } while(0)
#define BnvAccess(v)       CpvAccess(v)[BgMyRank()]

#define BpvDeclare(T, v)            CtvDeclare(T, v)
#define BpvStaticDeclare(T, v)      CtvStaticDeclare(T, v)
#define BpvExtern(T, v)             CtvExtern(T, v)
#define BpvInitialize(T, v)         CtvInitialize(T, v)
#define BpvAccess(v)                CtvAccess(v)

#if defined(__cplusplus)
}
#endif

#endif
