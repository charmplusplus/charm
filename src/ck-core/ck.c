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
 * Revision 2.24  1998-01-13 17:03:20  milind
 * Made charm++ to compile and run with Solaris 2.6.
 * In particular, changed INTBITS to CINTBITS, and handled EALREADY.
 *
 * Revision 2.23  1997/10/03 19:51:31  milind
 * Made charmc to work again, after inserting trace calls in converse part,
 * i.e. threads and user events.
 *
 * Revision 2.22  1997/07/18 21:21:03  milind
 * all files of the form perf-*.c have been changed to trace-*.c, with
 * name expansions. For example, perf-proj.c has been changed to
 * trace-projections.c.
 * performance.h has been renamed as trace.h, and perfio.c has been
 * renamed as traceio.c.
 * Corresponding changes have been made in the Makefile too.
 * Earlier, there used to be three libck-core-*.a where * was projections,
 * summary or none. Now, there will be a single libck-core.a and
 * three libck-trace-*.a where *=projections, summary and none.
 * The execmode parameter to charmc script has been renamed as
 * tracemode.
 * Also, the perfModuleInit function has been renamed as traceModuleInit,
 * RecdPerfMsg => RecdTraceMsg
 * CollectPerfFromNodes => CollectTraceFromNodes
 *
 * Revision 2.21  1997/04/21 20:58:48  jyelon
 * Simplified the shutdown protocol a little.
 *
 * Revision 2.20  1997/03/24 23:14:01  milind
 * Made Charm-runtime 64-bit safe by removing conversions of pointers to
 * integers. Also, removed charm runtime's dependence of unused argv[]
 * elements being 0. Also, added sim-irix-64 version. It works.
 *
 * Revision 2.19  1995/11/13 04:04:33  gursoy
 * made changes related to initial msg synchronization
 *
 * Revision 2.18  1995/11/07  17:53:45  sanjeev
 * fixed bugs in statistics collection
 *
 * Revision 2.17  1995/11/05  18:26:26  sanjeev
 * removed trace_creation in CkLdbSend
 *
 * Revision 2.16  1995/10/27  23:56:49  jyelon
 * removed more ansi
 *
 * Revision 2.15  1995/10/27  21:31:25  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.14  1995/10/27  09:09:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.13  1995/10/11  17:54:40  sanjeev
 * fixed Charm++ chare creation
 *
 * Revision 2.12  1995/09/06  21:48:50  jyelon
 * Eliminated 'CkProcess_BocMsg', using 'CkProcess_ForChareMsg' instead.
 *
 * Revision 2.11  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.10  1995/08/24  15:48:26  gursoy
 * worng cpv-macro usage for EpInfoTable (it is a Csv type not Cpv)
 * fixed
 *
 * Revision 2.9  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.8  1995/07/25  00:29:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.7  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.6  1995/07/22  23:44:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.5  1995/07/19  22:15:22  jyelon
 * *** empty log message ***
 *
 * Revision 2.4  1995/07/12  16:28:45  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/07/06  22:42:11  narain
 * Changes for LDB interface revision
 *
 * Revision 2.2  1995/06/29  21:38:00  narain
 * Added #define CldNewChareFromLocal, and code for CkMakeFreeCharesMessage,
 * CkQueueFreeCharesMessage, and SetNewChareMsg
 *
 * Revision 2.1  1995/06/08  17:09:41  gursoy
 * Cpv macro changes done
 *
 * Revision 1.13  1995/05/04  22:02:40  jyelon
 * *** empty log message ***
 *
 * Revision 1.12  1995/04/23  20:52:58  sanjeev
 * Removed Core....
 *
 * Revision 1.11  1995/04/23  14:27:44  brunner
 * Now includes converse.h, to get declaration of sysDone
 *
 * Revision 1.10  1995/04/14  21:05:01  milind
 * Changed HostPeNum to NumPes
 *
 * Revision 1.9  1995/04/13  20:53:15  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.8  1995/04/02  00:47:39  sanjeev
 * changes for separating Converse
 *
 * Revision 1.7  1995/03/25  18:24:05  sanjeev
 * *** empty log message ***
 *
 * Revision 1.6  1995/03/24  16:41:38  sanjeev
 * *** empty log message ***
 *
 * Revision 1.5  1995/03/17  23:36:25  sanjeev
 * changes for better message format
 *
 * Revision 1.4  1994/12/01  23:55:30  sanjeev
 * interop stuff
 *
 * Revision 1.3  1994/11/18  20:33:53  narain
 * Changed CkExit() into CkEndCharm and CkExit() with CkExit only setting the
 * value of sysDone to 1. (CkExit messages now have encoded sequence numbers
 *   - Sanjeev and Narain
 *
 * Revision 1.2  1994/11/09  21:43:18  sanjeev
 * printf consts
 *
 * Revision 1.1  1994/11/03  17:38:49  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "chare.h"
#include "globals.h"
#include "trace.h"
#include "converse.h"

#include <varargs.h>

extern void *FIFO_Create();
void CkLdbSend();
extern CHARE_BLOCK *CreateChareBlock();


CpvStaticDeclare(int, num_exits);
CpvStaticDeclare(int, num_endcharms);



void ckModuleInit()
{
   CpvInitialize(int, num_exits);
   CpvInitialize(int, num_endcharms);

   CpvAccess(num_exits)=0;
   CpvAccess(num_endcharms)=0;
}


/*************************************************************************
  EXIT PROTOCOL FOR CHARM

  How do the CkExit and CkEndCharm protocols work?

  CkExits send a message with an iteration number whose value is num_exits
  CkEndCharms send -1 in the message ..

  To prevent the sending of two messages for a particular CkExit,
    *  the seq number in the message should equal the value of num_exits for
       a CkExit to be processed (thus no two CkExits with the same seq number
       are processed).
  Also, CkEndCharm has a synchronization : the broadcast for statistics
  collection is done only after all processors have reported a CkEndCharm.

  The scheme is necessitated by the possibility of use of more than one 
  "DoCharm()" (now Scheduler()) and CkExits in each of them.

  The control flow - Any processor calling CkExit or CkEndCharm sends a 
  BroadcastExitMessage to node 0 which broadcasts to ExitMessage on all 
  processors.

  ExitMessage handles CkExit and CkEndCharm requests differently.

**************************************************************************/


CkExit()
{
	int *msg;

	msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(msg);
	*msg = CpvAccess(num_exits);

	GeneralSendMsgBranch(CsvAccess(CkEp_Stat_BroadcastExitMessage),
			msg, 0, BocMsg, StatisticBocNum);
}


CkEndCharm()
{
	int *msg;

	msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(msg);
	*msg = -1;

	GeneralSendMsgBranch(CsvAccess(CkEp_Stat_BroadcastExitMessage),
			msg, 0, BocMsg, StatisticBocNum);
}

BroadcastExitMessage(usr, data)
void *usr, *data;
{
/* This function is executed only on node 0 - corresponds to 
                 CsvAccess(CkEp_Stat_BroadcastExitMessage) */

	int *msg;
	
	if (*((int *)usr) == -1) { /* For CkEndCharm */
		CpvAccess(num_endcharms)++ ;
        	if( CpvAccess(num_endcharms) < CmiNumPes() ) 
	        	return;
	}
	else {  /* For CkExit */
		if(*((int *)usr) < CpvAccess(num_exits))
                	 return;
		CpvAccess(num_exits)++;
	}
	
	msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(msg);
	*msg = *((int *)usr);
	GeneralBroadcastMsgBranch(CsvAccess(CkEp_Stat_ExitMessage), msg, 
			BroadcastBocMsg, StatisticBocNum);
	CpvAccess(disable_sys_msgs) = 1;
}

ExitMessage(usr, data)
void *usr, *data;
{
        if(*((int *)usr) == -1) /* If the user called CkEndCharm */
	{
		SendNodeStatistics();
		send_log();
		if ( CmiMyPe() != 0 && CpvAccess(CtrRecdTraceMsg) 
		 		    && CpvAccess(RecdStatMsg) ) 
			ExitNode();
	}
	else /* If the user called CkExit */
	{
	        CkEndCharm();
		if(CmiMyPe())
			CpvAccess(num_exits)++;
	}
}


SendNodeStatistics()
{
	(*(CsvAccess(EpInfoTable)[CsvAccess(CkEp_Stat_Data)].function)) 
								(NULL,NULL);
}


ExitNode()
{
	char *msg;
	ENVELOPE *env;

	/* close_log(); moved to convcore.c */
	if (CmiMyPe() == 0)
	{
		/* First print out statistics. */
		PrintOutStatistics();
	}

	/* Complete the loop */	
        CsdExitScheduler();
}






/**********************************************************************
 * These are utility routines for chares 
***********************************************************************/

ChareExit()
{
	SetID_chare_magic_number(CpvAccess(currentChareBlock)->selfID,-1);
	CmiFree(CpvAccess(currentChareBlock));
}


CHARE_BLOCK *CreateChareBlock(sizeData, kind, magic)
int sizeData, magic, kind;
{
  CHARE_BLOCK *p = (CHARE_BLOCK *)CmiAlloc(sizeof(CHARE_BLOCK) + sizeData);
  CkMemError(p);
  SetID_chare_magic_number(p->selfID, magic);
  SetID_onPE(p->selfID, CmiMyPe());
  SetID_chareBlockPtr(p->selfID, p);
  p->charekind = kind;

  /* the chare data area is just after the chare-block, by default */
  p->chareptr = (void *)(p+1); 
  return((void *) p);
}

IsChareLocal(chareid)
ChareIDType * chareid;
{
	if (GetID_onPE((*chareid)) == CmiMyPe()) return 1;
	return 0;
}

void *GetChareDataPtr(chareid)
ChareIDType * chareid;
{
	return ((CHARE_BLOCK *) GetID_chareBlockPtr((*chareid))) + 1;
}

MyChareID(pChareID)
ChareIDType * pChareID;
{
    *pChareID = CpvAccess(currentChareBlock)->selfID;
}


/* Deleted already commented out MyParentID function : SANJEEV May 24, 93 */

MainChareID(pChareID)
ChareIDType * pChareID;
{
	SetID_onPE((*pChareID), 0);
	if (CmiMyPe() == 0)
		SetID_chare_magic_number((*pChareID),
		    GetID_chare_magic_number(CpvAccess(mainChareBlock)->selfID));
	else
		SetID_chare_magic_number((*pChareID), CpvAccess(mainChare_magic_number));
	SetID_chareBlockPtr((*pChareID), CpvAccess(mainChareBlock));
}



/* this is the general CreateChare call: all the user level CreateChare
   calls are mapped to this call: they include 

 	CreateChare(Charename, Entry, Msg, [vid [,destPE]]) 

   If vid is NULL_VID it is a CreateChare call ("without ID"). 
   if DestPe is CK_PE_ANY  then it may go to any destination node
   if DestPe is not CK_PE_SPECIAL then the message is bound for a regular destination

*/

CreateChare(id, Entry, Msg, vid, destPE)
int id;
EntryNumType Entry;
void *Msg;
ChareIDType *vid;
int destPE;
{
  ENVELOPE *env;
  CHARE_BLOCK *vidblock;

  if (id!=CsvAccess(EpInfoTable)[Entry].chareindex) 
    CmiPrintf("** ERROR ** Illegal combination of CHAREINDEX/EP in CreateChare\n");

  TRACE(CmiPrintf("[%d] CreateChare: Entry=%d\n", CmiMyPe(), Entry));
  
  CpvAccess(nodecharesCreated)++;
  env = ENVELOPE_UPTR(Msg);
  
  SetEnv_EP(env, Entry);
  
  if (vid != NULL_VID) {
    vidblock = (CHARE_BLOCK *)CreateChareBlock(0, CHAREKIND_UVID, rand());
    vidblock->x.vid_queue = (void *)FIFO_Create();
    (*vid) = vidblock->selfID;
    SetEnv_vidPE(env, GetID_onPE(vidblock->selfID));
    SetEnv_vidBlockPtr(env, GetID_chareBlockPtr(vidblock->selfID));
  } else {
    SetEnv_vidPE(env, -1);
    SetEnv_vidBlockPtr(env, NULL);
  }
  
  trace_creation(NewChareMsg, Entry, env);
  QDCountThisCreation(Entry, USERcat, NewChareMsg, 1);

  if (CK_PE_SPECIAL(destPE)) {
    if (destPE != CK_PE_ANY) {
      CmiPrintf("** ERROR ** Illegal destPE in CreateChare\n");
    }
    SetEnv_msgType(env, NewChareMsg);
    /* This CmiSetHandler is here because load balancer will fail to call */
    /* CkCheck_and_Send on local messages.  Fix this.                     */
    CmiSetHandler(env, CpvAccess(HANDLE_INCOMING_MSG_Index));
    CldNewSeedFromLocal(env, LDB_ELEMENT_PTR(env),
			CkLdbSend,
			GetEnv_queueing(env),
			GetEnv_priosize(env),
			GetEnv_priobgn(env));
  } else {
    SetEnv_msgType(env, NewChareNoBalanceMsg);
    CkCheck_and_Send(destPE, env);
  }

}



SendMsg(Entry, Msg, pChareID)
int Entry;
void * Msg;
ChareIDType * pChareID;
{
  ENVELOPE * env;
  int destPE = GetID_onPE((*pChareID));

  CpvAccess(nodeforCharesCreated)++;
  env = ENVELOPE_UPTR(Msg);
  SetEnv_msgType(env, ForChareMsg);
  SetEnv_EP(env, Entry);
  SetEnv_chareBlockPtr(env, GetID_chareBlockPtr((*pChareID)));
  SetEnv_chare_magic_number(env, GetID_chare_magic_number((*pChareID)));
  QDCountThisCreation(Entry, USERcat, ForChareMsg, 1);
  trace_creation(GetEnv_msgType(env), Entry, env);
  CkCheck_and_Send(destPE, env);
}


/*****************************************************************/
/** Gets reference number.					**/
/*****************************************************************/
GetRefNumber(msg)
void *msg;
{
	ENVELOPE *env = (ENVELOPE *) ENVELOPE_UPTR(msg);

	return GetEnv_ref(env);
}


/*****************************************************************/
/** Sets reference number.					**/
/*****************************************************************/
SetRefNumber(msg, number)
void *msg;
int number;
{
	ENVELOPE *env = (ENVELOPE *) ENVELOPE_UPTR(msg);

	SetEnv_ref(env, number);
}


void CkSetQueueing(usrptr, kind)
void *usrptr;
int kind;
{
  SetEnv_queueing(ENVELOPE_UPTR(usrptr), kind);
}

/*****************************************************************************
 * CkLdbSend is a function that is passed to the Ldb strategy to send out a
 * message to another processor
 *****************************************************************************/

void CkLdbSend(msgst, destPE)
     void *msgst;
     int destPE;
{
  ENVELOPE *env = (ENVELOPE *)msgst;

/* trace_creation is NOT needed here because it has already been done in
   the CreateChare
  trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);  */

  CkCheck_and_Send(destPE, env);
}

CkEnqueue(env)
void *env;
{
  CsdEnqueueGeneral(env,
    GetEnv_queueing(env),
    GetEnv_priosize(env),
    GetEnv_priobgn(env));
}

/************************************************************************
 *
 * CkPrioConcat
 *
 * Copies all the priority bits from the bitvector in 'srcmsg' onto
 * the bitvector in 'dstmsg', then, if there is any space left in the
 * bitvector of 'dstmsg', that space is filled by bits taken from the
 * lsb of 'delta'.
 *
 * The code works as follows:
 *
 * step 1: Copy old bitvector onto new. Always copies a multiple of
 * 32 bits, therefore, may copy some "padding" bits.  The number of
 * padding bits copied can be found in 'padbits'.
 *
 * step 2: move bits in delta to msb-end.
 *
 * step 3: if any padding-bits were copied, overwrite them with a
 * piece of delta.
 *
 * step 4: if padding-bits were insufficient to hold all of delta,
 * store remainder of delta in next word.
 *
 ************************************************************************/

#define CINTBITS (sizeof(int)*8)

void CkPrioConcatFn(srcmsg, dstmsg, delta)
void *srcmsg;
void *dstmsg;
unsigned int delta;
{
  int padbits, deltabits;
  ENVELOPE *srcenv = ENVELOPE_UPTR(srcmsg);
  ENVELOPE *dstenv = ENVELOPE_UPTR(dstmsg);
  int srcbits = GetEnv_priosize(srcenv);
  int dstbits = GetEnv_priosize(dstenv);
  int srcwords = (srcbits+CINTBITS-1)/CINTBITS;
  int dstwords = (dstbits+CINTBITS-1)/CINTBITS;
  unsigned int *srcptr = GetEnv_prioend(srcenv) - srcwords;
  unsigned int *dstptr = GetEnv_prioend(dstenv) - dstwords;
  deltabits = dstbits - srcbits;
  if (deltabits < 0) {
    CmiPrintf("CkPrioConcat: prio-bits from source message don't fit in destination message.\n");
    exit(1);
  }
  if (deltabits > CINTBITS) {
    CmiPrintf("CkPrioConcat: prio-bits from source message plus bits of delta don't fill destination-message.\n");
    exit(1);
  }
  while (srcbits>0) { *dstptr++ = *srcptr++; srcbits -= CINTBITS; }
  padbits = -srcbits;
  delta <<= (CINTBITS-deltabits);
  if (padbits) {
    dstptr[-1] &= (((unsigned int)(-1))<<padbits);
    dstptr[-1] |= (delta>>(CINTBITS-padbits));
  }
  if (deltabits>padbits) dstptr[0] = (delta<<padbits);
}

int CkPrioSizeBitsFn(msg) void *msg;
{
    return GetEnv_priosize(ENVELOPE_UPTR(msg));
}

int CkPrioSizeBytesFn(msg) void *msg;
{
    return GetEnv_priobytes(ENVELOPE_UPTR(msg));
}

int CkPrioSizeWordsFn(msg) void *msg;
{
    return GetEnv_priowords(ENVELOPE_UPTR(msg));
}

unsigned int *CkPrioPtrFn(msg) void *msg;
{
    return GetEnv_priobgn(ENVELOPE_UPTR(msg));
}
