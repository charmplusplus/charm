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
 * Revision 2.2  1995-06-14 19:38:29  gursoy
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
 
#include "trans_defs.h"
#include "trans_decls.h"

void *CkAllocPackBuffer(msg, size)
char *msg;
int size;
{
	int i;
	int prio_size;
	int pack_size;
	int headersize;
	ENVELOPE *envelope, *pack_envelope;

    pack_size = ((size + 3) & ~3);
    prio_size = MSG_PRIORITY_SIZE(msg);
    pack_envelope = (ENVELOPE *) 
	CmiAlloc(TOTAL_MSG_SIZE(pack_size, prio_size));

	/*** Now we need to copy the envelope  ***/
	envelope = ENVELOPE_UPTR(msg); 
	headersize = (msg - (char *) envelope);
	memcpy( ((char *) pack_envelope), ((char *) envelope), headersize);

	/**************** Converted to memcpy (Amitabh) ************/
	/** for (i=0; i<headersize; i++) **/  
	/** ((char *) pack_envelope)[i] = ((char *) envelope)[i]; **/
	/**************** Converted to memcpy (Amitabh) ************/

	/*** Now we insert the priority field ***/
    INSERT_PRIO_OFFSET(pack_envelope, pack_size, prio_size);
	COPY_PRIORITY(envelope, pack_envelope);
    return( (void *) USER_MSG_PTR(pack_envelope));
}


void *CkAllocMsg(request)
int request;
{
    ENVELOPE *envptr;

    request = ((request + 3) & ~3);
    envptr = (ENVELOPE *) CmiAlloc(TOTAL_MSG_SIZE(request, 0));
    CkMemError(envptr);
    SetEnv_isPACKED(envptr, NO_PACK);
    SetEnv_packid(envptr, 0);
    INSERT_PRIO_OFFSET(envptr, request, 0);
    return( (void *) USER_MSG_PTR(envptr));
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
        int size = CmiSize(env) ;   /* size of env in bytes */
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
    size = CmiSize(env) ;   /* size of env in bytes */
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

void * GenericCkAlloc(msgno, msgsize, prio_size)
int msgno;
int msgsize;
int prio_size;
{
	ENVELOPE *env;

    /* msgsize is in bytes since theres a sizeof(MsgType) done by caller */
    msgsize = ((msgsize + 3) & ~3);

    /* priosize is in words */
    prio_size *= 4 ;
/*    prio_size = ((prio_size + 3) & ~3);  this was a bug : Sanjeev 3/11/95 */

    env = (ENVELOPE *) CmiAlloc(TOTAL_MSG_SIZE(msgsize, prio_size));
    CkMemError(env);

    INSERT_PRIO_OFFSET(env, msgsize, prio_size);

    SetEnv_isPACKED(env, NO_PACK);
    SetEnv_packid(env, 0);

    if (CsvAccess(MsgToStructTable)[msgno].pack != NULL)
#ifdef SHARED
	{	
	   env->needsPack = UNPACKED;
       env->packid = msgno;
	}
#else
    {
        SetEnv_isPACKED(env, UNPACKED);
        SetEnv_packid(env, msgno);
    }
#endif
    return( (void *) USER_MSG_PTR(env));
}

