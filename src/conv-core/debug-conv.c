/*
Converse-level debugger support

Collected from convcore.c, conv-ccs.c, register.c by
Orion Sky Lawlor, olawlor@acm.org, 4/10/2001
 */
#include <stdio.h> /*for sscanf*/
#include <string.h> /*for strcmp*/
#include "converse.h"
#include "conv-trace.h"
#include "queueing.h"
#include "conv-ccs.h"
#include <errno.h>

CpvStaticDeclare(int, freezeModeFlag);
CpvStaticDeclare(int, continueFlag);
CpvStaticDeclare(int, stepFlag);
CpvDeclare(void *, debugQueue);
int _debugHandlerIdx;
CpvDeclare(int, skipBreakpoint); /* This is a counter of how many breakpoints we should skip */

char ** memoryBackup;

/***************************************************
  The CCS interface to the debugger
*/

#include <string.h>

#include "pup_c.h"
void * (*CpdDebugGetAllocationTree)(int *);
void (*CpdDebug_pupAllocationPoint)(pup_er p, void *data);
void (*CpdDebug_deleteAllocationPoint)(void *ptr);
void * (*CpdDebug_MergeAllocationTree)(void *data, void **remoteData, int numRemote);
CpvDeclare(int, CpdDebugCallAllocationTree_Index);
CpvStaticDeclare(CcsDelayedReply, allocationTreeDelayedReply);

static void CpdDebugReturnAllocationTree(void *tree) {
  pup_er sizer = pup_new_sizer();
  char *buf;
  pup_er packer;
  int i;
  CpdDebug_pupAllocationPoint(sizer, tree);
  buf = (char *)malloc(pup_size(sizer));
  packer = pup_new_toMem(buf);
  CpdDebug_pupAllocationPoint(packer, tree);
  /*CmiPrintf("size=%d tree:",pup_size(sizer));
  for (i=0;i<100;++i) CmiPrintf(" %02x",((unsigned char*)buf)[i]);
  CmiPrintf("\n");*/
  CcsSendDelayedReply(CpvAccess(allocationTreeDelayedReply), pup_size(sizer),buf);
  pup_destroy(sizer);
  pup_destroy(packer);
  free(buf);
}

static void CpdDebugCallAllocationTree(char *msg)
{
  int numNodes;
  int forPE;
  void *tree;
  sscanf(msg+CmiMsgHeaderSizeBytes, "%d", &forPE);
  if (CmiMyPe() == forPE) CpvAccess(allocationTreeDelayedReply) = CcsDelayReply();
  if (forPE == -1 && CmiMyPe()==0) {
    CpvAccess(allocationTreeDelayedReply) = CcsDelayReply();
    CmiSetXHandler(msg, CpvAccess(CpdDebugCallAllocationTree_Index));
    CmiSetHandler(msg, _debugHandlerIdx);
    CmiSyncBroadcast(CmiMsgHeaderSizeBytes+strlen(msg+CmiMsgHeaderSizeBytes)+1, msg);
  }
  tree = CpdDebugGetAllocationTree(&numNodes);
  if (forPE == CmiMyPe()) CpdDebugReturnAllocationTree(tree);
  else if (forPE == -1) CmiReduceStruct(tree, CpdDebug_pupAllocationPoint, CpdDebug_MergeAllocationTree,
                                CpdDebugReturnAllocationTree, CpdDebug_deleteAllocationPoint);
  else CmiAbort("Received allocationTree request for another PE!");
  CmiFree(msg);
}

static void CpdDebugHandler(char *msg)
{
    char name[128];
    sscanf(msg+CmiMsgHeaderSizeBytes, "%s", name);

    if (strcmp(name, "freeze") == 0) {
      CpdFreeze();
    }
    else if (strcmp(name, "unfreeze") == 0) {
      CpdUnFreeze();
    }
    else if (strncmp(name, "step", strlen("step")) == 0){
      CmiPrintf("step received\n");
      CpvAccess(stepFlag) = 1;
      CpdUnFreeze();
    }
    else if (strncmp(name, "continue", strlen("continue")) == 0){
      CmiPrintf("continue received\n");
      CpvAccess(continueFlag) = 1;
      CpdUnFreeze();
    }
    else if (strncmp(name, "status", strlen("status")) == 0) {
      char reply = CpdIsFrozen() ? 0 : 1;
      CcsSendReply(1, &reply);
    }
#if 0
    else if (strncmp(name, "setBreakPoint", strlen("setBreakPoint")) == 0){
      CmiPrintf("setBreakPoint received\n");
      temp = strstr(name, "#");
      temp++;
      setBreakPoints(temp);
    }
#endif
    else{
      CmiPrintf("bad debugger command:%s received,len=%ld\n",name,strlen(name));
    }
}


/*
 Start the freeze-- call will not return until unfrozen
 via a CCS request.
 */
void CpdFreeze(void)
{
  CmiPrintf("CPD: Frozen processor %d\n",CmiMyPe());
  if (CpvAccess(freezeModeFlag)) return; /*Already frozen*/
  CpvAccess(freezeModeFlag) = 1;
  CpdFreezeModeScheduler();
}

void CpdUnFreeze(void)
{
  CpvAccess(freezeModeFlag) = 0;
}

int CpdIsFrozen(void) {
  return CpvAccess(freezeModeFlag);
}

/* Deliver a single message in the queue while not unfreezing the program */
void CpdNext(void) {
  
}

/* This converse handler is used by the debugger itself, to send messages
 * even when the scheduler is in freeze mode.
 */
void handleDebugMessage(void *msg) {
  CmiSetHandler(msg, CmiGetXHandler(msg));
  CmiHandleMessage(msg);
}

/* Special scheduler-type loop only executed while in
freeze mode-- only executes CCS requests.
*/
void CcsServerCheck(void);
extern int _isCcsHandlerIdx(int idx);
int (*CpdIsDebugMessage)(void *);

void CpdFreezeModeScheduler(void)
{
#if CMK_CCS_AVAILABLE
    void *msg;
    void *debugQ=CpvAccess(debugQueue);
    CsdSchedulerState_t state;
    CsdSchedulerState_new(&state);

    /* While frozen, queue up messages */
    while (CpvAccess(freezeModeFlag)) {
#if NODE_0_IS_CONVHOST
      CcsServerCheck(); /*Make sure we can get CCS messages*/
#endif
      msg = CsdNextMessage(&state);

      if (msg!=NULL) {
        /*int hIdx=CmiGetHandler(msg);*/
	  /*
	  if(_isCcsHandlerIdx(hIdx))
	  / *A CCS request-- handle it immediately* /
          {
	    CmiHandleMessage(msg);
          }
	  else if (hIdx == _debugHandlerIdx ||
	          (hIdx == CmiGetReductionHandler() && CmiGetReductionDestination() == CpdDebugReturnAllocationTree)) {
	    / * Debug messages should be handled immediately * /
	    CmiHandleMessage(msg);
	  } else */
	  if (CpdIsDebugMessage(msg)) {
	    CmiHandleMessage(msg);
	  }
	  else
	  /*An ordinary charm++ message-- queue it up*/
	    CdsFifo_Enqueue(debugQ, msg);
      } else CmiNotifyIdle();
    }
    /* Before leaving freeze mode, execute the messages 
       in the order they would have executed before.*/
    while (!CdsFifo_Empty(debugQ))
    {
	char *queuedMsg = (char *)CdsFifo_Dequeue(debugQ);
        CmiHandleMessage(queuedMsg);
    }
#endif
}


void CpdInit(void)
{
  CpvInitialize(int, freezeModeFlag);
  CpvAccess(freezeModeFlag) = 0;

  CpvInitialize(void *, debugQueue);
  CpvAccess(debugQueue) = CdsFifo_Create();

  CcsRegisterHandler("ccs_debug", (CmiHandler)CpdDebugHandler);
  CcsRegisterHandler("ccs_debug_allocationTree", (CmiHandler)CpdDebugCallAllocationTree);
  CpvInitialize(int, CpdDebugCallAllocationTree_Index);
  CpvAccess(CpdDebugCallAllocationTree_Index) = CmiRegisterHandler((CmiHandler)CpdDebugCallAllocationTree);
  
  _debugHandlerIdx = CmiRegisterHandler((CmiHandler)handleDebugMessage);
#if 0
  CpdInitializeObjectTable();
  CpdInitializeHandlerArray();
  CpdInitializeBreakPoints();

  /* To allow start in freeze state: */
  msgListCleanup();
  msgListCache();
#endif
  
}

















