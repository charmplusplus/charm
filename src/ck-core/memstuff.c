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
 * Revision 2.17  1998-02-03 21:27:45  milind
 * Added pack and unpack events to tracing modules.
 *
 * Revision 2.16  1997/10/29 23:52:48  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.15  1997/03/24 23:09:35  milind
 * Corrected alignment problems on 64-bit machines.
 *
 * Revision 2.14  1995/10/27 09:09:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.13  1995/09/29  09:51:12  jyelon
 * Many small corrections.
 *
 * Revision 2.12  1995/09/14  21:23:52  jyelon
 * Added "globals.h"
 *
 * Revision 2.11  1995/09/14  20:49:17  jyelon
 * Added +fifo +lifo +ififo +ilifo +bfifo +blifo command-line options.
 *
 * Revision 2.10  1995/09/07  21:21:38  jyelon
 * Added prefixes to Cpv and Csv macros, fixed bugs thereby revealed.
 *
 * Revision 2.9  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.8  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.7  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.6  1995/07/22  23:45:15  jyelon
 * *** empty log message ***
 *
 * Revision 2.5  1995/07/19  22:15:30  jyelon
 * *** empty log message ***
 *
 * Revision 2.4  1995/07/12  16:28:45  jyelon
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

#define align(var) ((var+sizeof(void *)-1)&(~(sizeof(void *)-1)))

#include "trace.h" 
#include "globals.h" 
#include "trans_defs.h"
#include "trans_decls.h"

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

