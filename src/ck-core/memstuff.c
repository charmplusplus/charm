#include <stdio.h>

#define _CK_MEMORY_MANAGER
#include "charm.h"
#undef _CK_MEMORY_MANAGER

#define align(var) ((var+sizeof(void *)-1)&(~(sizeof(void *)-1)))

#include "trace.h" 
 
void *CkAllocPackBuffer(msg, bytespacked)
char *msg;
unsigned int bytespacked;
{
  int i;
  unsigned int priowords;
  unsigned int headersize;
  unsigned int totalsize;
  unsigned int *ptr1, *ptr2;
  ENVELOPE *envelope, *pack_envelope;
  
  bytespacked = align(bytespacked);
  priowords = GetEnv_priowords(ENVELOPE_UPTR(msg));
  totalsize = TOTAL_MSG_SIZE(bytespacked, priowords);
  pack_envelope = (ENVELOPE *)CmiAlloc(totalsize);
  
  /*** Now we need to copy the envelope, then adjust totalsize  ***/
  envelope = ENVELOPE_UPTR(msg); 
  headersize = (msg - (char *) envelope);
  memcpy( ((char *) pack_envelope), ((char *) envelope), headersize);
  SetEnv_TotalSize(pack_envelope,totalsize);
  SetEnv_packid(pack_envelope,GetEnv_packid(envelope));
  
  /*** Now we copy the priority field ***/
  ptr1 = GetEnv_prioend(envelope);
  ptr2 = GetEnv_prioend(pack_envelope);
  while (priowords) { *(--ptr2) = *(--ptr1); priowords--; }
  return((void *)USER_MSG_PTR(pack_envelope));
}


ENVELOPE *badmsg;

void *CkAllocMsg(msgbytes)
unsigned int msgbytes;
{
  unsigned int totalsize;
  ENVELOPE *envptr;
  
  msgbytes = align(msgbytes);
  totalsize = TOTAL_MSG_SIZE(msgbytes, 0);
  envptr = (ENVELOPE *)CmiAlloc(totalsize);
  if (envptr == badmsg) {
    CmiPrintf("Bad Message.\n");
  }
  CkMemError(envptr);
  SetEnv_isPACKED(envptr, NO_PACK);
  SetEnv_TotalSize(envptr, totalsize);
  SetEnv_packid(envptr, 0);
  SetEnv_queueing(envptr, CpvAccess(QueueingDefault));
  SetEnv_priosize(envptr, 0);
  return((void *)USER_MSG_PTR(envptr));
}




void CkFreeMsg(ptr)
char *ptr;
{
    ENVELOPE *envptr;

    envptr = ENVELOPE_UPTR(ptr);
    CmiFree(envptr);
}


/* Makes a copy of the envelope and the priority fields of the message passed
   to it and returns the envelope */

ENVELOPE *CkCopyEnv(env)
ENVELOPE *env ;
{
  int size = GetEnv_TotalSize(env) ;   /* size of env in bytes */
  ENVELOPE *newenv ;
  
  newenv = (ENVELOPE *) CmiAlloc(size) ;
  memcpy(newenv, env, size) ;
  
  return(newenv) ;
}

/* Makes a copy of the entire system message (Envelope, priorities, user 
   message et al.) and returns a pointer to the user message */

void *CkCopyMsg(sourceUptr)
char *sourceUptr ;
{
  int size ;
  ENVELOPE *env, *newenv;
  
  env = ENVELOPE_UPTR(sourceUptr) ;
  
  PACK(env);
  size = GetEnv_TotalSize(env) ;   /* size of env in bytes */
  newenv = (ENVELOPE *) CmiAlloc(size) ;
  memcpy(newenv, env, size) ;
  UNPACK(env);
  UNPACK(newenv);
  return((void *)USER_MSG_PTR(newenv)) ;
}



/*****************************************************************
The message allocation calls CkAllocMsg and CkAllocPrioMsg	
inside the user program are translated to the GenericCkAlloc
call in the following manner :

CkAllocMsg(Type) -->  GenericCkAlloc(TypeId, sizeof(Type), 0) ;
CkAllocPrioMsg(Type,prio) -->  GenericCkAlloc(TypeId, sizeof(Type), prio) ;

For varsize msgs :
CkAllocMsg(Type,sizearray) -->  
	(CsvAccess(MsgToStructTable)[TypeId].allocfn)(TypeId,sizeof(Type),sizearray,0) ;
CkAllocPrioMsg(Type,prio,sizearray) -->  
	(CsvAccess(MsgToStructTable)[TypeId].allocfn)(TypeId,sizeof(Type),sizearray,prio);
where the translator-generated allocfn template is :
allocfn(id, msgsize, sizearray, prio)
{
	... determine total message size (including varsize arrays) ...
	GenericCkAlloc(id, total_msg_size, prio) ;
	... set varsize array pointers ...
}

- Sanjeev

*****************************************************************/

void *GenericCkAlloc(int msgno, unsigned int msgbytes, unsigned int priobits)
{
    unsigned int msgwords;
    unsigned int priowords;
    unsigned int totalsize;
    ENVELOPE *env;

    msgbytes = align(msgbytes);
    priowords = (priobits+(sizeof(int)*8)-1)/(sizeof(int)*8);
    totalsize = TOTAL_MSG_SIZE(msgbytes, priowords);
    env = (ENVELOPE *)CmiAlloc(totalsize);
    CkMemError(env);

    SetEnv_priosize(env, priobits);
    SetEnv_queueing(env, CpvAccess(QueueingDefault));

    if (CsvAccess(MsgToStructTable)[msgno].packfn != NULL)
    {
        SetEnv_isPACKED(env, UNPACKED);
        SetEnv_TotalSize(env, totalsize);
        SetEnv_packid(env, msgno);
    }
    else
    {
        SetEnv_isPACKED(env, NO_PACK);
        SetEnv_TotalSize(env, totalsize);
        SetEnv_packid(env, 0);
    }
    return ((void *)USER_MSG_PTR(env));
}

