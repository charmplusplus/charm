/**
FIXME: (OSL, 2/28/2003)
 Messages sent from the communications thread or SIGIO
 (i.e., messages sent from immediate messages) add
 mCreated and mProcessed to the communication thread's
 CpvAccess(cQdState), not to any real processor's
 CpvAccess(cQdState).  Thus processor rank 0 should
 add CpvAccessOther(cQdState,CmiNodeSize())'s counts to its own.
 
Until this is fixed, if you send or receive immediate 
messages (e.g., from reductions) you CANNOT use converse 
quiescence detection!
*/

#include "converse.h"
#include "quiescence.h"
#include <stdio.h>
#include <stdlib.h>
#ifndef  DEBUGF
#define  DEBUGF(x) /*printf x*/ 
#endif

CpvDeclare(CQdState, cQdState);
unsigned int _CQdHandlerIdx;
unsigned int _CQdAnnounceHandlerIdx;


int  CQdMsgGetPhase(CQdMsg msg) 
{ return msg->phase; }

void CQdMsgSetPhase(CQdMsg msg, int p) 
{ msg->phase = p; }

int  CQdMsgGetCreated(CQdMsg msg) 
{ CmiAssert(msg->phase==1); return msg->u.p1.created; }

void CQdMsgSetCreated(CQdMsg msg, int c) 
{ CmiAssert(msg->phase==1); msg->u.p1.created = c; }

int  CQdMsgGetProcessed(CQdMsg msg) 
{ CmiAssert(msg->phase==1); return msg->u.p1.processed; }

void CQdMsgSetProcessed(CQdMsg msg, int p) 
{ CmiAssert(msg->phase==1); msg->u.p1.processed = p; }

int  CQdMsgGetDirty(CQdMsg msg) 
{ CmiAssert(msg->phase==2); return msg->u.p2.dirty; }

void CQdMsgSetDirty(CQdMsg msg, int d) 
{ CmiAssert(msg->phase==2); msg->u.p2.dirty = d; }


int CQdGetCreated(CQdState state)
{ return state->mCreated; }

void CQdCreate(CQdState state, int n)
{ state->mCreated += n; }

int CQdGetProcessed(CQdState state)
{ return state->mProcessed; }

void CQdProcess(CQdState state, int n)
{ state->mProcessed += n; }

void CQdPropagate(CQdState state, CQdMsg msg) 
{   
  int i;
  CmiSetHandler(msg, _CQdHandlerIdx);
  for(i=0; i<state->nChildren; i++) {
    CQdCreate(state, -1);
    CmiSyncSend(state->children[i], sizeof(struct ConvQdMsg), (char *)msg);
  }
}

int  CQdGetParent(CQdState state) 
{ return state->parent; }
    
int  CQdGetCCreated(CQdState state) 
{ return state->cCreated; }

int  CQdGetCProcessed(CQdState state) 
{ return state->cProcessed; }

void CQdSubtreeCreate(CQdState state, int c) 
{ state->cCreated += c; }

void CQdSubtreeProcess(CQdState state, int p) 
{ state->cProcessed += p; }

int  CQdGetStage(CQdState state) 
{ return state->stage; }

void CQdSetStage(CQdState state, int p) 
{ state->stage = p; }

void CQdReported(CQdState state) 
{ state->nReported++; }

int  CQdAllReported(CQdState state) 
{ return state->nReported==(state->nChildren+1);}

void CQdReset(CQdState state) 
{ 
  state->nReported=0; state->cCreated=0; 
  state->cProcessed=0; state->cDirty=0; 
}

void CQdMarkProcessed(CQdState state) 
{ state->oProcessed = state->mProcessed; }

int  CQdIsDirty(CQdState state) 
{ return ((state->mProcessed > state->oProcessed) || state->cDirty); }

void CQdSubtreeSetDirty(CQdState state, int d) 
{ state->cDirty = state->cDirty || d; }

CQdState CQdStateCreate(void)
{
  CQdState state = (CQdState) malloc(sizeof(struct ConvQdState));
  _MEMCHECK(state);
  state->mCreated = 0;
  state->mProcessed = 0;
  state->stage = 0;
  state->nReported = 0;
  state->oProcessed = 0;
  state->cCreated = 0;
  state->cProcessed = 0;
  state->cDirty = 0;
  state->nChildren = CmiNumSpanTreeChildren(CmiMyPe());
  state->parent = CmiSpanTreeParent(CmiMyPe());
  /* fixed bug on SP3, when nChildren is 0, NULL will be returned by malloc */
  if (state->nChildren) {
    state->children = (int *) malloc(state->nChildren*sizeof(int));
    _MEMCHECK(state->children);
  }
  else 
    state->children = NULL;
  CmiSpanTreeChildren(CmiMyPe(), state->children);

  return state;
}


static void CQdBcastQD1(CQdState state, CQdMsg msg)
{  
  CQdMsgSetPhase(msg, 0); 
  CQdPropagate(state, msg); 
  CQdMsgSetPhase(msg, 1); 
  CQdMsgSetCreated(msg, CQdGetCreated(state)); 
  CQdMsgSetProcessed(msg, CQdGetProcessed(state)); 
  CQdCreate(state, -1);
  CmiSyncSendAndFree(CmiMyPe(), sizeof(struct ConvQdMsg), (char *) msg);
  CQdMarkProcessed(state); 
  CQdReset(state); 
  CQdSetStage(state, 1); 
}


static void CQdBcastQD2(CQdState state, CQdMsg msg)
{
  CQdMsgSetPhase(msg, 1); 
  CQdPropagate(state, msg); 
  CQdMsgSetPhase(msg, 2); 
  CQdMsgSetDirty(msg, CQdIsDirty(state)); 
  CQdCreate(state, -1);
  CmiSyncSendAndFree(CmiMyPe(), sizeof(struct ConvQdMsg), (char *) msg);
  CQdReset(state); 
  CQdSetStage(state, 2); 
}


static void CQdHandlePhase0(CQdState state, CQdMsg msg)
{
  CmiAssert(CmiMyPe()==0 || CQdGetStage(state)==0);
  if(CQdGetStage(state)==0)
    CQdBcastQD1(state, msg);
  else
    CmiFree(msg);
}


static void CQdHandlePhase1(CQdState state, CQdMsg msg)
{
  switch(CQdGetStage(state)) { 		
  case 0 :
    CmiAssert(CmiMyPe()!=0);
    CQdBcastQD2(state, msg);
    break;
  case 1 :
    CQdSubtreeCreate(state, CQdMsgGetCreated(msg)); 
    CQdSubtreeProcess(state, CQdMsgGetProcessed(msg)); 
    CQdReported(state); 
    
    if(CQdAllReported(state)) {
      if(CmiMyPe()==0) {
	if(CQdGetCCreated(state) == CQdGetCProcessed(state)) 
	  CQdBcastQD2(state, msg); 
	else 
	  CQdBcastQD1(state, msg);
      } 
      else {
	CQdMsgSetCreated(msg, CQdGetCCreated(state)); 
	CQdMsgSetProcessed(msg, CQdGetCProcessed(state)); 
	CQdCreate(state, -1);
	CmiSyncSendAndFree(CQdGetParent(state), 
			   sizeof(struct ConvQdMsg), (char *) msg);
	DEBUGF(("PE = %d, My parent = %d\n", CmiMyPe(), CQdGetParent(state)));
	CQdReset(state); 
	CQdSetStage(state, 0); 
      }
    } 
    else
      CmiFree(msg);
    break;
  default: 
    CmiAbort("Internal QD Error. Contact Developers.!\n");
  }
}


static void CQdHandlePhase2(CQdState state, CQdMsg msg)
{
  CmiAssert(CQdGetStage(state)==2);
  CQdSubtreeSetDirty(state, CQdMsgGetDirty(msg));    
  CQdReported(state);
  if(CQdAllReported(state)) { 
    if(CmiMyPe()==0) {
      if(CQdIsDirty(state)) 
	CQdBcastQD1(state, msg);
      else {
	CmiSetHandler(msg, _CQdAnnounceHandlerIdx);
	CQdCreate(state, 0-CmiNumPes());
	CmiSyncBroadcastAllAndFree(sizeof(struct ConvQdMsg), (char *) msg);
	CQdReset(state); 
	CQdSetStage(state, 0); 
      }
    } 
    else {
      CQdMsgSetDirty(msg, CQdIsDirty(state)); 
      CQdCreate(state, -1);
      CmiSyncSendAndFree(CQdGetParent(state), 
			 sizeof(struct ConvQdMsg), (char *) msg);
      CQdReset(state); 
      CQdSetStage(state, 0); 
    }
  } 
  else
    CmiFree(msg);
}


static void CQdCallWhenIdle(CQdMsg msg)
{
  CQdState state = CpvAccess(cQdState);
  
  switch(CQdMsgGetPhase(msg)) {
  case 0 : CQdHandlePhase0(state, msg); break;
  case 1 : CQdHandlePhase1(state, msg); break;
  case 2 : CQdHandlePhase2(state, msg); break;
  default: CmiAbort("Internal QD Error. Contact Developers.!\n");
  }
}


void CQdHandler(CQdMsg msg)
{
  CQdProcess(CpvAccess(cQdState), -1);
  CcdCallOnCondition(CcdPROCESSOR_STILL_IDLE, 
		     (CcdVoidFn)CQdCallWhenIdle, (void*) msg);  
}


void CQdRegisterCallback(CQdVoidFn fn, void *arg)
{
  CcdCallOnCondition(CcdQUIESCENCE, fn, arg);
}

void CQdAnnounceHandler(CQdMsg msg)
{
  CQdProcess(CpvAccess(cQdState), -1);
  CcdRaiseCondition(CcdQUIESCENCE);
}

void CQdCpvInit(void) {
  CpvInitialize(CQdState, cQdState);
  CpvAccess(cQdState) = CQdStateCreate();
}

void CQdInit(void)
{
  CQdCpvInit();
  _CQdHandlerIdx = CmiRegisterHandler((CmiHandler)CQdHandler);
  _CQdAnnounceHandlerIdx = 
    CmiRegisterHandler((CmiHandler)CQdAnnounceHandler);
}

void CmiStartQD(CQdVoidFn fn, void *arg)
{
  register CQdMsg msg = (CQdMsg) CmiAlloc(sizeof(struct ConvQdMsg)); 
  CQdRegisterCallback(fn, arg);
  CQdMsgSetPhase(msg, 0);  
  CmiSetHandler(msg, _CQdHandlerIdx);
  CQdCreate(CpvAccess(cQdState), -1);
  CmiSyncSendAndFree(0, sizeof(struct ConvQdMsg), (char *)msg);
}



