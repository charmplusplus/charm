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

typedef struct {
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
                         CkUnpackFnPtr unpack, CkCoerceFnPtr coerce, int size);
extern int CkRegisterEp(const char *name, CkCallFnPtr call, int msgIdx, 
                        int chareIdx);
extern int CkRegisterChare(const char *name, int dataSz);
extern int CkRegisterMainChare(int chareIndex, int epIndex);
extern void CkRegisterReadonly(int size, void *ptr);
extern void CkRegisterReadonlyMsg(void** pMsg);

/******************************************************************************
 *
 * Object Creation Calls
 *
 *****************************************************************************/

extern void CkCreateChare(int chareIdx, int constructorIdx, void *msg,
                          CkChareID *vid, int destPE);
extern int CkCreateGroup(int chareIdx, int constructorIdx, void *msg,
                         int returnEpIdx, CkChareID *returnChare);
extern int CkCreateNodeGroup(int chareIdx, int constructorIdx, void *msg,
                         int returnEpIdx, CkChareID *returnChare);

/******************************************************************************
 *
 * Asynchronous Remote Method Invocation Calls
 *
 *****************************************************************************/

extern void CkSendMsg(int entryIndex, void *msg, CkChareID *chare);
extern void CkSendMsgBranch(int entryIdx, void *msg, int destPE, int groupID);
extern void CkSendMsgNodeBranch(int entryIdx, void *msg, int destNode, int groupID);
extern void CkBroadcastMsgBranch(int entryIdx, void *msg, int groupID);
extern void CkBroadcastMsgNodeBranch(int entryIdx, void *msg, int groupID);

extern void CkSetRefNum(void *msg, int ref);
extern int  CkGetRefNum(void *msg);
extern int  CkGetSrcPe(void *msg);
extern int  CkGetSrcNode(void *msg);

/******************************************************************************
 *
 * Blocking Method Invocation Calls
 *
 *****************************************************************************/

extern void* CkRemoteCall(int entryIdx, void *msg, CkChareID *chare);
extern void* CkRemoteBranchCall(int entryIdx, void *msg, int groupID, int pe);
extern void* CkRemoteNodeBranchCall(int entryIdx, void *msg, int groupID, int node);
extern void  CkSendToFuture(int futNum, void *msg, int pe);

/******************************************************************************
 *
 * Quiescence Calls
 *
 *****************************************************************************/

extern void CkStartQD(int entryIdx, CkChareID *chare);
extern void CkWaitQD(void);

/******************************************************************************
 *
 * Miscellaneous Calls
 *
 *****************************************************************************/

extern void *CkLocalBranch(int groupID);
extern void *CkLocalNodeBranch(int groupID);
extern void  CkGetChareID(CkChareID *pcid);
extern int   CkGetGroupID(void);
extern int   CkGetNodeGroupID(void);
extern void  CkExit(void);

#ifdef __cplusplus
}
#endif
#endif
