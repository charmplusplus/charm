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
 * Revision 2.4  1995-07-12 16:28:45  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/06/29  21:49:07  narain
 * Changed member of MSG_STRUCT to packfn
 *
 * Revision 2.2  1995/06/14  19:38:29  gursoy
 * CmiAllocPackBuffer -> CkAllocPackBuffer
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.5  1995/04/25  04:32:43  narain
 * moved CKMEM_UNIT from mem_common.c to both memory managers
 *
 * Revision 1.4  1995/04/02  00:44:51  sanjeev
 * changes for separating Converse
 *
 * Revision 1.3  1995/03/12  17:12:47  sanjeev
 * changes for new msg macros
 *
 * Revision 1.2  1994/11/11  06:06:17  brunner
 * Since this file is #include-ed, got rid of static char in RCS header
 *
 * Revision 1.1  1994/11/03  17:34:41  brunner
 * Initial revision
 *
 ***************************************************************************/
#include <stdio.h>

#define _CK_MEMORY_MANAGER
#include "chare.h"
#undef _CK_MEMORY_MANAGER

#define align(var) ((var+sizeof(int)-1)&(~(sizeof(int)-1)))
 
#include "trans_defs.h"
#include "trans_decls.h"

void *CkAllocPackBuffer(msg, bytespacked)
char *msg;
unsigned int bytespacked;
{
  int i;
  unsigned int priowords;
  unsigned int priobytes;
  unsigned int headersize;
  unsigned int totalsize;
  char *ptr1, *ptr2; int size1, size2, size;
  ENVELOPE *envelope, *pack_envelope;
  
  bytespacked = align(bytespacked);
  priowords = MSG_PRIOSIZE_WORDS(msg);
  priobytes = priowords * sizeof(int);
  totalsize = TOTAL_MSG_SIZE(bytespacked, priowords);
  pack_envelope = (ENVELOPE *)CmiAlloc(totalsize);
  
  /*** Now we need to copy the envelope, then adjust totalsize  ***/
  envelope = ENVELOPE_UPTR(msg); 
  headersize = (msg - (char *) envelope);
  memcpy( ((char *) pack_envelope), ((char *) envelope), headersize);
  SetEnv_TotalSize_packid(pack_envelope,totalsize,GetEnv_packid(envelope));
  
  /*** Now we copy the priority field ***/
  ptr1 = GetEnv_prioend(envelope);
  ptr2 = GetEnv_prioend(pack_envelope);
  memcpy(ptr2-priobytes, ptr1-priobytes, priobytes);
  return((void *)USER_MSG_PTR(pack_envelope));
}


void *CkAllocMsg(msgbytes)
unsigned int msgbytes;
{
  unsigned int totalsize;
  ENVELOPE *envptr;
  
  msgbytes = align(msgbytes);
  totalsize = TOTAL_MSG_SIZE(msgbytes, 0);
  envptr = (ENVELOPE *)CmiAlloc(totalsize);
  CkMemError(envptr);
  SetEnv_isPACKED(envptr, NO_PACK);
  SetEnv_TotalSize_packid(envptr, totalsize, 0);
  SetEnv_priosize(envptr, 0);
  SetEnv_queueing(envptr, CK_QUEUEING_FIFO);
  return((void *)USER_MSG_PTR(envptr));
}




CkFreeMsg(ptr)
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
	(MsgToStructTable[TypeId].allocfn)(TypeId,sizeof(Type),sizearray,0) ;
CkAllocPrioMsg(Type,prio,sizearray) -->  
	(MsgToStructTable[TypeId].allocfn)(TypeId,sizeof(Type),sizearray,prio);
where the translator-generated allocfn template is :
allocfn(id, msgsize, sizearray, prio)
{
	... determine total message size (including varsize arrays) ...
	GenericCkAlloc(id, total_msg_size, prio) ;
	... set varsize array pointers ...
}

- Sanjeev

/*****************************************************************/

void *GenericCkAlloc(msgno, msgbytes, priobits)
int msgno;
unsigned int msgbytes;
unsigned int priobits;
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
    SetEnv_queueing(env, CK_QUEUEING_BFIFO);

    if (CsvAccess(MsgToStructTable)[msgno].packfn != NULL)
    {
        SetEnv_isPACKED(env, UNPACKED);
        SetEnv_TotalSize_packid(env, totalsize, msgno);
    }
    else
    {
        SetEnv_isPACKED(env, NO_PACK);
        SetEnv_TotalSize_packid(env, totalsize, 0);
    }
    return ((void *)USER_MSG_PTR(env));
}

