/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.0  1995-06-29 21:19:36  narain
 * *** empty log message ***
 *
 ***************************************************************************/
#include "conversedefns.h"
CsvExtern(int, CallProcessMsg_Index);
CsvExtern(int, HANDLE_INCOMING_MSG_Index);

CpvExtern(int, LdbBocNum);
CpvExtern(int, numPe);
#include "env_macros.h"
#include "msg_macros.h"
#include "communication.h"
#include <memory.h>

#define QsEnqNewChareMsg(ptr) CsdEnqueue((ENVELOPE *)ptr)
#define CkMemError(ptr) \
{\
  if(!(ptr))\
    { CkPrintf("Memory for ptr not allocated\n"); \
      CkExit();\
    }\
}


#define CkSendLdbMsg(msg, pe) \
GeneralSendMsgBranch(LdbNbrStatus_EP, msg, pe, IMMEDIATEcat, LdbMsg, CpvAccess(LdbBocNum))

extern int CmiNumNeighbours();
extern McGetNodeNeighbours();
/*extern void CsdEnqueue(); */
extern trace_creation();
extern int netSend();
extern McNeighboursIndex();
extern abs();
extern CallBocAfter();
extern CqsEmpty();
extern CqsLength();
extern CqsDequeue();
extern CqsEnqueue();
extern log();
extern CkMakeFreeCharesMessage();
extern CkQueueFreeCharesMessage();
extern QsMyLoad();
extern void QsPickFreeChare();
extern CmiSyncSend();

#define SetMsgParameters(msg, EP, cat, type, bocnum, fixed) \
{ \
  ENVELOPE *env = ENVELOPE_UPTR(msg); \
  SetEnv_category(env, cat); \
  SetEnv_msgType(env, type); \
  SetEnv_destPeFixed(env, fixed); \
  SetEnv_boc_num(env, bocnum);  \
  SetEnv_EP(env, EP);          \
}

#define SetLdbMsgParameters(msg) \
     SetMsgParameters( msg, LdbNbrStatus_EP, IMMEDIATEcat, LdbMsg, CpvAccess(LdbBocNum), TRUE);

#define DestPE_LDB(ldbptr) GetEnv_destPE((ENVELOPE_LDBPTR(ldbptr)))
#define DestPE_Msg(msg) GetEnv_destPE(((ENVELOPE_UPTR(msg))))
#define DestFixed_Msg(msg) GetEnv_destPeFixed((ENVELOPE_UPTR(msg)))

#define SEND_TO(msg, pe) { \
        ENVELOPE *env =(ENVELOPE *)ENVELOPE_UPTR(msg); \
        trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env); \
	SetEnv_destPE(env,pe); \
        CkSend(pe, env); }

#define SEND_FIXED_TO(msg, fixed, pe) { \
        ENVELOPE *env = (ENVELOPE *) ENVELOPE_UPTR(msg); \
        trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env); \
	SetEnv_destPE(env,pe); \
	SetEnv_destPeFixed(env, fixed); \
        CkSend(pe, env); }

#define COPY_AND_SEND_TO(msg, pe) { \
        ENVELOPE *new = (ENVELOPE *) CkCopyEnv(ENVELOPE_UPTR(msg)); \
        trace_creation(GetEnv_msgType(new), GetEnv_EP(new), new); \
	SetEnv_destPE(new,pe); \
        CkSend(pe, new); }

#define determine_msg_trans(msg) determine_translation(ENVELOPE_UPTR(msg))


/* --------------------- Queue related stuff ----------------------*/

#define CmiSetMsgHndlr(x, y) CmiSetHandler(ENVELOPE_UPTR(x), y)

#define EnqUsrMsg(msg) CsdEnqueue(ENVELOPE_UPTR(msg))

#define Qs_EnQMsg(queueptr, msg) CqsEnqueue(queueptr, ENVELOPE_UPTR(msg))

#define Qs_DeQMsg(queueptr, msgptr) { \
  ENVELOPE *Penv; \
  CqsDequeue(queueptr, &Penv); \
  *msgptr = USER_MSG_PTR(Penv); \
}

#define QsPickFreeChareMsg(msgPtr) { \
ENVELOPE *env; \
QsPickFreeChare(&env); \
*(msgPtr) = (env)? USER_MSG_PTR(env) : NULL; \
}

message {
  int number_chares;
  char ptr;
} PackChareMsg;

typedef char * charptr;

message [CharesMsg] {
  int number_chares;
  void **msgPtrs;

  pack packfn(in, out, length)
    CharesMsg *in;
    PackChareMsg  **out;
    int *length;
  {
    int total_size, i, *size_array;
    char *tempptr;
    ENVELOPE *env;
    size_array = (int *)CmiAlloc(sizeof(int) * in->number_chares);
    total_size = sizeof(int) * ( in->number_chares + 1);
    for(i = 0; i< in->number_chares; i++)
      {
	env = ENVELOPE_UPTR(in->msgPtrs[i]); /* Below this, the 
						msgPtrs array 
						contains ENV *s */
	PACK(env);
	in->msgPtrs[i] = (void *)env;
	total_size += (size_array[i] = CmiSize(env));
      }
    (*out) = (PackChareMsg *) CkAllocPackBuffer(in, total_size);
    (*out)->number_chares = in->number_chares;
    *length = total_size;
    tempptr = &((*out)->ptr);
    for(i=0; i< in->number_chares; i++)
      {
	tempptr = (char *)((int *)tempptr);
	*((int *)tempptr) = size_array[i];
	tempptr += sizeof(int);
	memcpy(tempptr, in->msgPtrs[i], size_array[i]);
	tempptr += size_array[i];
	CmiFree(in->msgPtrs[i]);
      }
    CmiFree(size_array);
    CkFreeMsg(in);
  }
  
  unpack unpackfn(in, out)
    PackChareMsg *in;
    CharesMsg **out;
  {
    int i, len;
    char *tempptr;
    void *ptr;
    ENVELOPE *env;
    
    (*out) = (CharesMsg *) CkAllocMsg(CharesMsg);
    (*out)->number_chares = in->number_chares;
    (*out)->msgPtrs = (void **) CmiAlloc(sizeof(void *) * in->number_chares);

    tempptr = &(in->ptr);
    for(i=0; i< in->number_chares; i++)
      {
	tempptr = (char *)((int *)tempptr);
	len = *((int *)tempptr);
	tempptr += sizeof(int);
	ptr = (void *)CmiAlloc(len);
	memcpy(ptr, tempptr, len);
	env = (ENVELOPE *)ptr;
	tempptr += len;
	UNPACK(env);
	(*out)->msgPtrs[i] = USER_MSG_PTR(env);
      }
  }
} CharesMsg;

#define TRACE(x) 
