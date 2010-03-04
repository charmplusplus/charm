
#include <stdlib.h>
#include "queueing.h"
#include "cldb.h"
#include <math.h>

typedef char *BitVector;

CpvDeclare(int, CldHandlerIndex);
CpvDeclare(BitVector, CldPEBitVector);
CpvDeclare(int, CldBalanceHandlerIndex);

CpvDeclare(int, CldRelocatedMessages);
CpvDeclare(int, CldLoadBalanceMessages);
CpvDeclare(int, CldMessageChunks);
CpvDeclare(int, CldLoadNotify);

CpvDeclare(CmiNodeLock, cldLock);

extern void LoadNotifyFn(int);

char* _lbtopo = "torus2d";

/* Estimator stuff.  Of any use? */
/*
CpvStaticDeclare(CldEstimatorTable, _estfns);
*/
void CldRegisterEstimator(CldEstimator fn)
{
  /*CpvAccess(_estfns).fns[CpvAccess(_estfns).count++] = fn;*/
}

/* 
int CldEstimate(void)
{
  CldEstimatorTable *estab = &(CpvAccess(_estfns));
  int i, load=0;
  for(i=0; i<estab->count; i++)
    load += (*(estab->fns[i]))();
  return load;
}

static int CsdEstimator(void)
{
  return CsdLength();
}
*/

CpvDeclare(int, CldLoadOffset);


int CldRegisterInfoFn(CldInfoFn fn)
{
  return CmiRegisterHandler((CmiHandler)fn);
}

int CldRegisterPackFn(CldPackFn fn)
{
  return CmiRegisterHandler((CmiHandler)fn);
}

/* CldSwitchHandler takes a message and a new handler number.  It
 * changes the handler number to the new handler number and move the
 * old to the Xhandler part of the header.  When the message gets
 * handled, the handler should call CldRestoreHandler to put the old
 * handler back.
 *
 * CldPutToken puts a message in the scheduler queue in such a way
 * that it can be retreived from the queue.  Once the message gets
 * handled, it can no longer be retreived.  CldGetToken removes a
 * message that was placed in the scheduler queue in this way.
 * CldCountTokens tells you how many tokens are currently retreivable.  
*/

void CldSwitchHandler(char *cmsg, int handler)
{
#if CMK_MEM_CHECKPOINT
  int old_phase = CmiGetRestartPhase(cmsg);
#endif
  CmiSetXHandler(cmsg, CmiGetHandler(cmsg));
  CmiSetHandler(cmsg, handler);
#if CMK_MEM_CHECKPOINT
  CmiGetRestartPhase(cmsg) = old_phase;
#endif
}

void CldRestoreHandler(char *cmsg)
{
#if CMK_MEM_CHECKPOINT
  int old_phase = CmiGetRestartPhase(cmsg);
#endif
  CmiSetHandler(cmsg, CmiGetXHandler(cmsg));
#if CMK_MEM_CHECKPOINT
  CmiGetRestartPhase(cmsg) = old_phase;
#endif
}

void Cldhandler(char *);
 
typedef struct CldToken_s {
  char msg_header[CmiMsgHeaderSizeBytes];
  char *msg;  /* if null, message already removed */
  struct CldToken_s *pred;
  struct CldToken_s *succ;
} *CldToken;

typedef struct CldProcInfo_s {
  int tokenhandleridx;
  int load; /* number of items in doubly-linked circle besides sentinel */
  CldToken sentinel;
} *CldProcInfo;

CpvDeclare(CldProcInfo, CldProc);

static void CldTokenHandler(CldToken tok)
{
  CldProcInfo proc = CpvAccess(CldProc);
  if (tok->msg) {
    tok->pred->succ = tok->succ;
    tok->succ->pred = tok->pred;
    proc->load --;
    CmiHandleMessage(tok->msg);
  }
  else 
    CpvAccess(CldLoadOffset)--;
  if (CpvAccess(CldLoadNotify))
    LoadNotifyFn(CpvAccess(CldProc)->load);
  CmiFree(tok);
}

int CldCountTokens(void)
{
  return (CpvAccess(CldProc)->load);
}

int CldLoad(void)
{
  return (CsdLength() - CpvAccess(CldLoadOffset));
}

int CldLoadRank(int rank)
{
  int len, offset;
  /* CmiLock(CpvAccessOther(cldLock, rank));  */
  len = CqsLength(CpvAccessOther(CsdSchedQueue, rank));
     /* CldLoadOffset is the empty token counter */
  offset = CpvAccessOther(CldLoadOffset, rank);
  /* CmiUnlock(CpvAccessOther(cldLock, rank)); */
  return len - offset;
}

void CldPutToken(char *msg)
{
  CldProcInfo proc = CpvAccess(CldProc);
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  CldToken tok;
  int len, queueing, priobits; unsigned int *prioptr;
  CldPackFn pfn;

  CmiLock(CpvAccess(cldLock));
  tok = (CldToken)CmiAlloc(sizeof(struct CldToken_s));
  tok->msg = msg;

  /* add token to the doubly-linked circle */
  tok->pred = proc->sentinel->pred;
  tok->succ = proc->sentinel;
  tok->pred->succ = tok;
  tok->succ->pred = tok;
  proc->load ++;
  /* add token to the scheduler */
  CmiSetHandler(tok, proc->tokenhandleridx);
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  /* not sigio or thread safe */
  CsdEnqueueGeneral(tok, queueing, priobits, prioptr);
  CmiUnlock(CpvAccess(cldLock));
}


static void * _CldGetTokenMsg(CldProcInfo proc)
{
  CldToken tok;
  void *msg;
  
  tok = proc->sentinel->succ;
  if (tok == proc->sentinel) {
    return NULL;
  }
  tok->pred->succ = tok->succ;
  tok->succ->pred = tok->pred;
  proc->load --;
  msg = tok->msg;
  tok->msg = 0;
  return msg;
}

void CldGetToken(char **msg)
{
  CldProcInfo proc = CpvAccess(CldProc);
  CmiNodeLock cldlock = CpvAccess(cldLock);
  CmiLock(cldlock);
  *msg = _CldGetTokenMsg(proc);
  if (*msg) CpvAccess(CldLoadOffset)++;
  CmiUnlock(cldlock);
}

/* called at node level */
/* get token from processor of rank pe */
static void CldGetTokenFromRank(char **msg, int rank)
{
  CldProcInfo proc = CpvAccessOther(CldProc, rank);
  CmiNodeLock cldlock = CpvAccessOther(cldLock, rank);
  CmiLock(cldlock);
  *msg = _CldGetTokenMsg(proc);
  if (*msg) CpvAccessOther(CldLoadOffset, rank)++;
  CmiUnlock(cldlock);
}

/* Bit Vector Stuff */

int CldPresentPE(int pe)
{
  return CpvAccess(CldPEBitVector)[pe];
}

void CldMoveAllSeedsAway()
{
  char *msg;
  int len, queueing, priobits, pe;
  unsigned int *prioptr;
  CldInfoFn ifn;  CldPackFn pfn;

  CldGetToken(&msg);
  while (msg != 0) {
    ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CldSwitchHandler(msg, CpvAccess(CldBalanceHandlerIndex));
    pe = (((CrnRand()+CmiMyPe())&0x7FFFFFFF)%CmiNumPes());
    while (!CldPresentPE(pe))
      pe = (((CrnRand()+CmiMyPe())&0x7FFFFFFF)%CmiNumPes());
    CmiSyncSendAndFree(pe, len, msg);
    CldGetToken(&msg);
  }
}

void CldSetPEBitVector(const char *newBV)
{
  int i;
  
  for (i=0; i<CmiNumPes(); i++)
    CpvAccess(CldPEBitVector)[i] = newBV[i];
  if (!CldPresentPE(CmiMyPe()))
    CldMoveAllSeedsAway();
}

/* End Bit Vector Stuff */

void CldModuleGeneralInit(char **argv)
{
  CldToken sentinel = (CldToken)CmiAlloc(sizeof(struct CldToken_s));
  CldProcInfo proc;
  int i;

  CpvInitialize(CldProcInfo, CldProc);
  CpvInitialize(int, CldLoadOffset);
  CpvAccess(CldLoadOffset) = 0;
  CpvInitialize(int, CldLoadNotify);
  CpvInitialize(BitVector, CldPEBitVector);
  CpvAccess(CldPEBitVector) = (char *)malloc(CmiNumPes()*sizeof(char));
  for (i=0; i<CmiNumPes(); i++)
    CpvAccess(CldPEBitVector)[i] = 1;
  CpvAccess(CldProc) = (CldProcInfo)CmiAlloc(sizeof(struct CldProcInfo_s));
  proc = CpvAccess(CldProc);
  proc->load = 0;
  proc->tokenhandleridx = CmiRegisterHandler((CmiHandler)CldTokenHandler);
  proc->sentinel = sentinel;
  sentinel->succ = sentinel;
  sentinel->pred = sentinel;

  /* lock to protect token queue for immediate message and smp */
  CpvInitialize(CmiNodeLock, cldLock);
  CpvAccess(cldLock) = CmiCreateLock();
}

/* function can be called in an immediate handler at node level
   rank specify the rank of processor for the node to represent
   This function can also send as immeidate messages
*/
void CldMultipleSend(int pe, int numToSend, int rank, int immed)
{
  char **msgs;
  int len, queueing, priobits, *msgSizes, i, numSent, done=0, parcelSize;
  unsigned int *prioptr;
  CldInfoFn ifn;
  CldPackFn pfn;

  if (numToSend == 0)
    return;

  msgs = (char **)calloc(numToSend, sizeof(char *));
  msgSizes = (int *)calloc(numToSend, sizeof(int));

  while (!done) {
    numSent = 0;
    parcelSize = 0;
    for (i=0; i<numToSend; i++) {
      CldGetTokenFromRank(&msgs[i], rank);
      if (msgs[i] != 0) {
	done = 1;
	numSent++;
	ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msgs[i]));
	ifn(msgs[i], &pfn, &len, &queueing, &priobits, &prioptr);
	msgSizes[i] = len;
	parcelSize += len;
	CldSwitchHandler(msgs[i], CpvAccessOther(CldBalanceHandlerIndex, rank));
        if (immed) CmiBecomeImmediate(msgs[i]);
      }
      else {
	done = 1;
	break;
      }
      if (parcelSize > MAXMSGBFRSIZE) {
	if(i<numToSend-1)
	  done = 0;
	numToSend -= numSent;
	break;
      }
    }
    if (numSent > 1) {
      if (immed)
        CmiMultipleIsend(pe, numSent, msgSizes, msgs);
      else
        CmiMultipleSend(pe, numSent, msgSizes, msgs);
      for (i=0; i<numSent; i++)
	CmiFree(msgs[i]);
      CpvAccessOther(CldRelocatedMessages, rank) += numSent;
      CpvAccessOther(CldMessageChunks, rank)++;
    }
    else if (numSent == 1) {
      if (immed) CmiBecomeImmediate(msgs[0]);
      CmiSyncSendAndFree(pe, msgSizes[0], msgs[0]);
      CpvAccessOther(CldRelocatedMessages, rank)++;
      CpvAccessOther(CldMessageChunks, rank)++;
    }
  }
  free(msgs);
  free(msgSizes);
}

/* simple scheme - just send one by one. useful for multicore */
void CldSimpleMultipleSend(int pe, int numToSend)
{
  char *msg;
  int len, queueing, priobits, *msgSizes, i, numSent, done=0;
  unsigned int *prioptr;
  CldInfoFn ifn;
  CldPackFn pfn;

  if (numToSend == 0)
    return;

  numSent = 0;
  while (!done) {
    for (i=0; i<numToSend; i++) {
      CldGetToken(&msg);
      if (msg != 0) {
	done = 1;
	numToSend--;
	ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
	ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
	CldSwitchHandler(msg, CpvAccessOther(CldBalanceHandlerIndex, pe));
        CmiSyncSendAndFree(pe, len, msg);
        if (numToSend == 0) done = 1;
      }
      else {
	done = 1;
	break;
      }
    }
  }
}
