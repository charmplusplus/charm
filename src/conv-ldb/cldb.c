#include "converse.h"

/*
 * CldSwitchHandler takes a message and a new handler number.  It
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
 *
 * Caution: these functions are using the function "CmiReference"
 * which I just added to the Cmi memory allocator (it increases the
 * reference count field, making it possible to free the memory
 * twice.)  I'm not sure how well this is going to work.  I need this
 * because the message should not be freed until it's out of the
 * scheduler queue AND out of the user's hands.  It needs to stay
 * around while it's in the scheduler queue because it may contain
 * a priority.  I should probably rewrite these subroutines so that
 * they simply copy the priority, I would feel safer that way.
 *
 */

int CldRegisterInfoFn(CldInfoFn fn)
{
  return CmiRegisterHandler((CmiHandler)fn);
}

int CldRegisterPackFn(CldPackFn fn)
{
  return CmiRegisterHandler((CmiHandler)fn);
}

void CldSwitchHandler(char *cmsg, int handler)
{
  CmiSetXHandler(cmsg, CmiGetHandler(cmsg));
  CmiSetHandler(cmsg, handler);
}

void CldRestoreHandler(char *cmsg)
{
  CmiSetHandler(cmsg, CmiGetXHandler(cmsg));
}

void Cldhandler(void *);
 
typedef struct CldToken_s {
  char msg_header[CmiMsgHeaderSizeBytes];
  void *msg;  /* if null, message already removed */
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
  CldToken pred, succ;
  if (tok->pred) {
    tok->pred->succ = tok->succ;
    tok->succ->pred = tok->pred;
    proc->load --;
    CmiHandleMessage(tok->msg);
  } else {
    /* CmiFree(tok->msg); */
  }
}

int CldCountTokens()
{
  CldProcInfo proc = CpvAccess(CldProc);
  return proc->load;
}

void CldPutToken(void *msg)
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
  CsdEnqueueGeneral(tok, queueing, priobits, prioptr);
}

void CldGetToken(void **msg)
{
  CldProcInfo proc = CpvAccess(CldProc);
  CldToken tok;
  tok = proc->sentinel->succ;
  if (tok == proc->sentinel) {
    *msg = 0; return;
  }
  tok->pred->succ = tok->succ;
  tok->succ->pred = tok->pred;
  tok->succ = 0;
  tok->pred = 0;
  proc->load --;
  *msg = tok->msg;
  CmiReference(*msg);
}

void CldModuleGeneralInit()
{
  CldToken sentinel = (CldToken)CmiAlloc(sizeof(struct CldToken_s));
  CldProcInfo proc;

  CpvInitialize(CldProcInfo, CldProc);
  CpvAccess(CldProc) = (CldProcInfo)CmiAlloc(sizeof(struct CldProcInfo_s));
  proc = CpvAccess(CldProc);
  proc->load = 0;
  proc->tokenhandleridx = CmiRegisterHandler((CmiHandler)CldTokenHandler);
  proc->sentinel = sentinel;
  sentinel->succ = sentinel;
  sentinel->pred = sentinel;
}
