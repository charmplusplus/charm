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
 * Revision 2.8  1995-07-25 00:29:31  jyelon
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
 * Changed HostPeNum to NumPe
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
#include "performance.h"
#include "converse.h"

#include <varargs.h>

extern void *FIFO_Create();
void CkLdbSend();

CpvStaticDeclare(int, num_exits);
CpvStaticDeclare(int, done);

/* 
  How does this work?

  CkExits send on a message with an iteration number whose value is num_exits
  CkEndCharm send -1 in the message ..

  To prevent the sending of two messages for a particular CkExit or CkEndCharm,
    *  the static variable done makes sure that only one CkEndCharm is done
    *  the seq number in the message should equal the value of num_exits for
       a CkExit to be processed (thus no two CkExits with the same seq number
       are processed

  The scheme is necessitated by the possibility of use of more than one 
  "DoCharm()" (now Scheduler()) and CkExits in each of them.

  The control flow - Any processor calling CkExit or CkEndCharm sends a 
  BroadcastExitMessage to node 0 which broadcasts to ExitMessage on all 
  processors.

  ExitMessage handles CkExit and CkEndCharm requests differently.

*/



void ckModuleInit()
{
   CpvInitialize(int, num_exits);
   CpvInitialize(int, done);

   CpvAccess(num_exits)=0;
   CpvAccess(done)     =0;
}




CkExit()
{
	int *msg;

	msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(msg);
	*msg = CpvAccess(num_exits);

	GeneralSendMsgBranch(StatBroadcastExitMessage_EP,
			msg, 0, USERcat,  BocMsg, StatisticBocNum);
}


EndCharm()
{
       CkEndCharm();
       CsdScheduler(-1);
}


CkEndCharm()
{
	int *msg;

	msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(msg);
	*msg = -1;

	GeneralSendMsgBranch(StatBroadcastExitMessage_EP,
			msg, 0, USERcat,  BocMsg, StatisticBocNum);
}

BroadcastExitMessage(usr, data)
void *usr, *data;
{
/*
   This function is executed only on node 0 - corresponds to 
                 StatBroadcastExitMessage_EP
*/
	int *msg;
	
TRACE(CmiPrintf("[%d] BroadcastExitMessage: sending out message to everybody.\n",
CmiMyPe()))
;

	if (*((int *)usr) == -1) /* For CkEndCharm */
        {
              if(CpvAccess(done)) 
	            return;
	      else
	            CpvAccess(done) = 1;
	}
	else                     /* For CkExit */
        {
	      if(*((int *)usr) < CpvAccess(num_exits))
                     return;
	      CpvAccess(num_exits)++;
	}
	
	msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(msg);
	*msg = *((int *)usr);
	GeneralBroadcastMsgBranch(StatExitMessage_EP, msg, 
			USERcat, BroadcastBocMsg, StatisticBocNum);
	CpvAccess(disable_sys_msgs) = 1;
}

ExitMessage(usr, data)
void *usr, *data;
{
        if(*((int *)usr) == -1) /* If the user called CkEndCharm */
	{
	    /*
	      if (CmiMyPe() != 0) CmiFlushPrintfs();
	    */
	    SendNodeStatistics();
	    send_log();
	    if (CmiMyPe() != 0 &&  (CpvAccess(RecdPerfMsg) && CpvAccess(RecdStatMsg))) ExitNode();
	}
	else /* If the user called CkExit */
	{
	      CsdExitScheduler();
	      if(CmiMyPe())
		   CpvAccess(num_exits)++;
	}
}


/* This is what you do before you exit the main node */
ExitNode()
{
	char *msg;
	ENVELOPE *env;

TRACE(CmiPrintf("[%d] ExitNode: RecdPerfMsg=%d, RecdStatMsg=%d\n", CmiMyPe(),
CpvAccess(RecdPerfMsg), CpvAccess(RecdStatMsg)));
	close_log();
	if (CmiMyPe() == 0)
	{
		/* First print out statistics. */
		PrintOutStatistics();
	}

	/* Complete the loop */	
        CsdExitScheduler();
}


ChareExit()
{
	SetID_chare_magic_number(CpvAccess(currentChareBlock)->selfID,-1) ;
	CmiFree(CpvAccess(currentChareBlock));
}


SendNodeStatistics()
{
	/*NodeCollectStatistics(NULL, NULL);*/
	(*(CsvAccess(EpTable)[StatData_EP])) (NULL,NULL);
}


void * CreateChareBlock(sizeData)
int sizeData;
{
	char *p;

	p = (char *) CmiAlloc( sizeof(CHARE_BLOCK) + sizeData );
	CkMemError(p);
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
  ENVELOPE * env;
  VID_BLOCK * vidblock;
  int DataMag ;
  
  if ( IsCharmPlus(Entry) )
    DataMag = bytes_to_magnitude(id);
  else
    DataMag = bytes_to_magnitude(CsvAccess(ChareSizesTable)[id]);
  
  TRACE(CmiPrintf("[%d] CreateChare: Entry=%d\n", CmiMyPe(), Entry));
  
  CpvAccess(nodecharesCreated)++;
  env = ENVELOPE_UPTR(Msg);
  
  SetEnv_dataMag(env, DataMag);
  SetEnv_EP(env, Entry);
  
  if (vid != NULL_VID)
    {
      vidblock   = (VID_BLOCK *)  CmiAlloc(sizeof(VID_BLOCK));
      CkMemError(vidblock);
      vidblock->vidPenum = -1;
      vidblock->info_block.vid_queue = (void *) FIFO_Create();
      SetID_onPE((*vid), CmiMyPe());
      SetID_vidBlockPtr((*vid),  (struct vid_block *) vidblock);
      SetEnv_vidPE(env, CmiMyPe());
      SetEnv_vidBlockPtr(env, (int) vidblock);
    }
  else
    {
      SetEnv_vidPE(env, -1);
      SetEnv_vidBlockPtr(env, NULL);
    }
  
  TRACE(CmiPrintf("[%d] CreateChare: vid=0x%x\n",
		  CmiMyPe(), vid));
  
  TRACE(CmiPrintf("[%d] CreateChare: category=%d, msgType=%d, ep=%d\n",
		  CmiMyPe(), GetEnv_category(env), 
		  GetEnv_msgType(env), GetEnv_EP(env)));
  
  
  QDCountThisCreation(Entry, USERcat, NewChareMsg, 1);
  
  
  /********************  SANJEEV May 24, 93 **************************/
  /* The CldNewChareFromLocal, FIFO_EnQueue, and CkCheck_and_Send
     calls were moved inside this if-then-else  */
  
  trace_creation(GetEnv_msgType(env), Entry, env);
  if (CK_PE_SPECIAL(destPE)) {
    if (destPE != CK_PE_ANY) {
      CmiPrintf("** ERROR ** Illegal destPE in CreateChare\n");
    }
    SetEnv_msgType(env, NewChareMsg);
    CmiSetHandler(env,CsvAccess(CkProcess_NewChareMsg_Index)) ;
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
  
  if (GetID_isBOC((*pChareID)))
    GeneralSendMsgBranch(Entry, Msg, 
			 GetID_onPE((*pChareID)), USERcat, BocMsg, GetID_boc_num((*pChareID)));
  else
    {
      int destPE = GetID_onPE((*pChareID));
      CpvAccess(nodeforCharesCreated)++;
      env = ENVELOPE_UPTR(Msg);
      SetEnv_msgType(env, ForChareMsg);
      SetEnv_EP(env, Entry);
      if (!GetID_isVID((*pChareID)))
	{
	  SetEnv_chareBlockPtr(env, (int)
			       GetID_chareBlockPtr((*pChareID)));
	  SetEnv_chare_magic_number(env,
				    GetID_chare_magic_number((*pChareID)));
	  QDCountThisCreation(Entry, USERcat, ForChareMsg, 1);
	  
	}
      else 
	{
	  SetEnv_msgType(env, VidEnqueueMsg);
	  SetEnv_vidBlockPtr(env, (int) GetID_vidBlockPtr((*pChareID)));
	  QDCountThisCreation(VidQueueUpInVidBlock_EP,
			      IMMEDIATEcat, VidEnqueueMsg, 1);
	}
      trace_creation(GetEnv_msgType(env), Entry, env);
      CkCheck_and_Send(destPE, env);
    }
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

void *CkPriorityPtr(usrptr)
void *usrptr;
{
  return PRIORITY_UPTR(usrptr);
}

int   CkPriorityBits(usrptr)
void *usrptr;
{
  return GetEnv_priosize(ENVELOPE_UPTR(usrptr));
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
  CmiSetHandler(env, CsvAccess(HANDLE_INCOMING_MSG_Index));
  trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env); 
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

