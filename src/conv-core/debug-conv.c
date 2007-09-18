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

static void CpdDebugReturnAllocationTree(void *tree) {
  pup_er sizer = pup_new_sizer();
  CpdDebug_pupAllocationPoint(sizer, tree);
  char *buf = (char *)malloc(pup_size(sizer));
  pup_er packer = pup_new_toMem(buf);
  CpdDebug_pupAllocationPoint(packer, tree);
  /*CmiPrintf("size=%d tree:",pup_size(sizer));
  int i;
  for (i=0;i<100;++i) CmiPrintf(" %02x",((unsigned char*)buf)[i]);
  CmiPrintf("\n");*/
  CcsSendReply(pup_size(sizer),buf);
  pup_destroy(sizer);
  pup_destroy(packer);
  free(buf);
}

static void CpdDebugCallAllocationTree(char *msg)
{
  int numNodes;
  int forPE;
  sscanf(msg+CmiMsgHeaderSizeBytes, "%d", &forPE);
  if (forPE == -1 && CmiMyPe()==0) {
    CmiSetHandler(msg, CpvAccess(CpdDebugCallAllocationTree_Index));
    CmiSyncBroadcast(CmiMsgHeaderSizeBytes+sizeof(int), msg);
  }
  void *tree = CpdDebugGetAllocationTree(&numNodes);
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
  if (CpvAccess(freezeModeFlag)) return; /*Already frozen*/
  CpvAccess(freezeModeFlag) = 1;
  CpdFreezeModeScheduler();
}

void CpdUnFreeze(void)
{
  CpvAccess(freezeModeFlag) = 0;
}

/* Special scheduler-type loop only executed while in
freeze mode-- only executes CCS requests.
*/
void CcsServerCheck(void);
extern int _isCcsHandlerIdx(int idx);

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
	  int hIdx=CmiGetHandler(msg);
	  if(_isCcsHandlerIdx(hIdx))
	  /*A CCS request-- handle it immediately*/
          {
	    CmiHandleMessage(msg);
          }
	  else
	  /*An ordinary message-- queue it up*/
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
  
#if 0
  CpdInitializeObjectTable();
  CpdInitializeHandlerArray();
  CpdInitializeBreakPoints();

  /* To allow start in freeze state: */
  msgListCleanup();
  msgListCache();
#endif
  
}

















