/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef CHARM_H
#define CHARM_H

#include "converse.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 *
 * Converse Concepts, renamed to CK
 *
 *****************************************************************************/

#define CK_QUEUEING_FIFO   CQS_QUEUEING_FIFO
#define CK_QUEUEING_LIFO   CQS_QUEUEING_LIFO
#define CK_QUEUEING_IFIFO  CQS_QUEUEING_IFIFO
#define CK_QUEUEING_ILIFO  CQS_QUEUEING_ILIFO
#define CK_QUEUEING_BFIFO  CQS_QUEUEING_BFIFO
#define CK_QUEUEING_BLIFO  CQS_QUEUEING_BLIFO

#define CkTimer  	CmiTimer
#define CkWallTimer  	CmiWallTimer

#define CkMyPe		CmiMyPe
#define CkMyRank	CmiMyRank
#define CkMyNode	CmiMyNode
#define CkNumPes	CmiNumPes
#define CkNumNodes	CmiNumNodes
#define CkNodeFirst	CmiNodeFirst
#define CkNodeSize	CmiNodeSize
#define CkNodeOf	CmiNodeOf
#define CkRankOf	CmiRankOf

#define CkPrintf                CmiPrintf
#define CkScanf                 CmiScanf
#define CkError			CmiError
#define CkAbort                 CmiAbort

/******************************************************************************
 *
 * Miscellaneous Constants
 *
 *****************************************************************************/

#define CK_PE_ALL        CLD_BROADCAST_ALL
#define CK_PE_ALL_BUT_ME CLD_BROADCAST
#define CK_PE_ANY        CLD_ANYWHERE

/******************************************************************************
 *
 * Message Allocation Calls
 *
 *****************************************************************************/

extern void* CkAllocSysMsg(void);
extern void  CkFreeSysMsg(void *msg);
extern void* CkAllocMsg(int msgIdx, int msgBytes, int prioBits);
extern void* CkAllocBuffer(void *msg, int bufsize);
extern void  CkFreeMsg(void *msg);
extern void* CkCopyMsg(void **pMsg);
extern void  CkSetQueueing(void *msg, int strategy);
extern void* CkPriorityPtr(void *msg);

/******************************************************************************
 *
 * Data Structures.
 *
 *****************************************************************************/

typedef struct _ckargmsg {
  int argc;
  char **argv;
#ifdef __cplusplus
  void operator delete(void *ptr) { CkFreeMsg(ptr); }
#endif
} CkArgMsg;

typedef struct {
  int   onPE;
  int   magic;
  void* objPtr;
} CkChareID;

typedef int CkGroupID;
typedef int CkFutureID;

/******************************************************************************
 *
 * Function Pointer Types
 *
 *****************************************************************************/

typedef void* (*CkPackFnPtr)(void *msg);
typedef void* (*CkUnpackFnPtr)(void *buf);
typedef void* (*CkCoerceFnPtr)(void *buf);
typedef void  (*CkCallFnPtr) (void *msg, void *obj);

/******************************************************************************
 *
 * Registration Calls
 *
 *****************************************************************************/

extern int CkRegisterMsg(const char *name, CkPackFnPtr pack, 
                       CkUnpackFnPtr unpack, CkCoerceFnPtr coerce, size_t size);
extern int CkRegisterEp(const char *name, CkCallFnPtr call, int msgIdx, 
                        int chareIdx);
extern int CkRegisterChare(const char *name, int dataSz);
extern int CkRegisterMainChare(int chareIndex, int epIndex);
extern void CkRegisterDefaultCtor(int chareIndex, int ctorEpIndex);
extern void CkRegisterMigCtor(int chareIndex, int ctorEpIndex);
extern void CkRegisterReadonly(int size, void *ptr);
extern void CkRegisterReadonlyMsg(void** pMsg);
extern void CkRegisterMainModule(void);

/******************************************************************************
 *
 * Object Creation Calls
 *
 *****************************************************************************/

extern void CkCreateChare(int chareIdx, int constructorIdx, void *msg,
                          CkChareID *vid, int destPE);
extern CkGroupID CkCreateGroup(int chareIdx, int constructorIdx, void *msg,
                         int returnEpIdx, CkChareID *returnChare);
extern CkGroupID CkCreateGroupSync(int cIdx, int consIdx, void *msg);
extern CkGroupID CkCreateNodeGroup(int chareIdx, int constructorIdx, void *msg,
                         int returnEpIdx, CkChareID *returnChare);
extern CkGroupID CkCreateNodeGroupSync(int cIdx, int consIdx, void *msg);

/******************************************************************************
 *
 * Asynchronous Remote Method Invocation Calls
 *
 *****************************************************************************/

extern void CkSendMsg(int entryIndex, void *msg, CkChareID *chare);
extern void CkSendMsgBranch(int eIdx, void *msg, int destPE, CkGroupID gID);
extern void CkSendMsgBranchMulti(int eIdx, void *msg, int npes, int *pes, 
                                 CkGroupID gID);
extern void CkSendMsgNodeBranch(int eIdx, void *msg, int destNode, 
                                CkGroupID gID);
extern void CkBroadcastMsgBranch(int eIdx, void *msg, CkGroupID gID);
extern void CkBroadcastMsgNodeBranch(int eIdx, void *msg, CkGroupID gID);

extern void CkSetRefNum(void *msg, int ref);
extern int  CkGetRefNum(void *msg);
extern int  CkGetSrcPe(void *msg);
extern int  CkGetSrcNode(void *msg);

/******************************************************************************
 *
 * Blocking Method Invocation Calls
 *
 *****************************************************************************/

extern void* CkRemoteCall(int eIdx, void *msg, CkChareID *chare);
extern void* CkRemoteBranchCall(int eIdx, void *msg, CkGroupID gID, int pe);
extern void* CkRemoteNodeBranchCall(int eIdx, void *msg, CkGroupID gID, int node);
extern CkFutureID CkRemoteCallAsync(int eIdx, void *msg, CkChareID *chare);
extern CkFutureID CkRemoteBranchCallAsync(int eIdx, void *msg, CkGroupID gID, 
                                          int pe);
extern CkFutureID CkRemoteNodeBranchCallAsync(int eIdx, void *msg, 
                                              CkGroupID gID, int node);
extern void* CkWaitFuture(CkFutureID futNum);
extern void CkWaitVoidFuture(CkFutureID futNum);
extern void CkReleaseFuture(CkFutureID futNum);
extern int CkProbeFuture(CkFutureID futNum);
extern void  CkSendToFuture(CkFutureID futNum, void *msg, int pe);

/******************************************************************************
 *
 * Quiescence Calls
 *
 *****************************************************************************/

extern void CkStartQD(int eIdx, CkChareID *chare);
extern void CkWaitQD(void);

/******************************************************************************
 *
 * Miscellaneous Calls
 *
 *****************************************************************************/

extern void *CkLocalBranch(int gID);
extern void *CkLocalNodeBranch(int gID);
extern void  CkGetChareID(CkChareID *pcid);
extern CkGroupID   CkGetGroupID(void);
extern CkGroupID   CkGetNodeGroupID(void);
extern void  CkExit(void);

extern void CkSummary_MarkEvent(int);
extern void CkSummary_StartPhase(int);

#ifdef __cplusplus
}
#endif
#endif
