#include "cldb.h"

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
  CmiSetXHandler(cmsg, CmiGetHandler(cmsg));
  CmiSetHandler(cmsg, handler);
}

void CldRestoreHandler(char *cmsg)
{
  CmiSetHandler(cmsg, CmiGetXHandler(cmsg));
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
}

int CldCountTokens()
{
  return (CpvAccess(CldProc)->load);
}

int CldLoad()
{
  return (CsdLength() - CpvAccess(CldLoadOffset));
}

void CldPutToken(char *msg)
{
  CldProcInfo proc = CpvAccess(CldProc);
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  CldToken tok = (CldToken)CmiAlloc(sizeof(struct CldToken_s));
  int len, queueing, priobits; unsigned int *prioptr;
  CldPackFn pfn;

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

  queueing = CQS_QUEUEING_LIFO; 
  CsdEnqueueGeneral(tok, queueing, priobits, prioptr);
}

void CldGetToken(char **msg)
{
  CldProcInfo proc = CpvAccess(CldProc);
  CldToken tok;
  
  tok = proc->sentinel->succ;
  if (tok == proc->sentinel) {
    *msg = 0; return;
  }
  tok->pred->succ = tok->succ;
  tok->succ->pred = tok->pred;
  proc->load --;
  *msg = tok->msg;
  tok->msg = 0;
  CpvAccess(CldLoadOffset)++;
}

void CldModuleGeneralInit()
{
  CldToken sentinel = (CldToken)CmiAlloc(sizeof(struct CldToken_s));
  CldProcInfo proc;

  CpvInitialize(CldProcInfo, CldProc);
  CpvInitialize(int, CldLoadOffset);
  CpvAccess(CldLoadOffset) = 0;
  CpvAccess(CldProc) = (CldProcInfo)CmiAlloc(sizeof(struct CldProcInfo_s));
  proc = CpvAccess(CldProc);
  proc->load = 0;
  proc->tokenhandleridx = CmiRegisterHandler((CmiHandler)CldTokenHandler);
  proc->sentinel = sentinel;
  sentinel->succ = sentinel;
  sentinel->pred = sentinel;
}

void CldMultipleSend(int pe, int numToSend)
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
      CldGetToken(&msgs[i]);
      if (msgs[i] != 0) {
	done = 1;
	numSent++;
	ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msgs[i]));
	ifn(msgs[i], &pfn, &len, &queueing, &priobits, &prioptr);
	msgSizes[i] = len;
	parcelSize += len;
	CldSwitchHandler(msgs[i], CpvAccess(CldBalanceHandlerIndex));
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
      CmiMultipleSend(pe, numSent, msgSizes, msgs);
      for (i=0; i<numSent; i++)
	CmiFree(msgs[i]);
      CpvAccess(CldRelocatedMessages) += numSent;
      CpvAccess(CldMessageChunks)++;
    }
    else if (numSent == 1) {
      CmiSyncSend(pe, msgSizes[0], msgs[0]);
      CmiFree(msgs[0]);
      CpvAccess(CldRelocatedMessages)++;
      CpvAccess(CldMessageChunks)++;
    }
  }
  free(msgs);
  free(msgSizes);
}
