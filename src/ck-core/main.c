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
 * Revision 2.11  1995-07-27 20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.10  1995/07/25  00:29:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.9  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.8  1995/07/22  23:44:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.7  1995/07/19  22:15:28  jyelon
 * *** empty log message ***
 *
 * Revision 2.6  1995/07/12  16:28:45  jyelon
 * *** empty log message ***
 *
 * Revision 2.5  1995/07/06  22:42:11  narain
 * Changes for LDB interface revision
 *
 * Revision 2.4  1995/07/05  21:04:11  narain
 * *** empty log message ***
 *
 * Revision 2.3  1995/07/05  19:38:31  narain
 * No CldFillBlock and CldStripMsg while InsideDataInit
 *
 * Revision 2.2  1995/06/29  21:44:50  narain
 * Added macros for CldStripMsg and CldNewChareFromNet
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.11  1995/05/03  20:57:07  sanjeev
 * bug fixes for finding uninitialized modules
 *
 * Revision 1.10  1995/04/24  20:17:13  sanjeev
 * fixed typo
 *
 * Revision 1.9  1995/04/23  20:53:26  sanjeev
 * Removed Core....
 *
 * Revision 1.8  1995/04/13  20:54:53  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.7  1995/03/25  18:23:59  sanjeev
 * *** empty log message ***
 *
 * Revision 1.6  1995/03/23  22:12:51  sanjeev
 * *** empty log message ***
 *
 * Revision 1.5  1995/03/17  23:35:04  sanjeev
 * changes for better message format
 *
 * Revision 1.4  1995/03/09  22:21:53  sanjeev
 * fixed bug in BlockingLoop
 *
 * Revision 1.3  1994/12/01  23:55:10  sanjeev
 * interop stuff
 *
 * Revision 1.2  1994/11/18  20:32:17  narain
 * Added a parameter (number of iterations) to Loop()
 * Added functions DoCharm and EndCharm (main seperation)
 *  - Sanjeev and Narain
 *
 * Revision 1.1  1994/11/03  17:38:31  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";


/*************************************************************************/
/** This file now contains only the Charm/Charm++ part of the run-time.
    The core (scheduler/Converse) part is in converse.c 		 */
/*************************************************************************/

#include "chare.h"
#include "globals.h"
#include "performance.h"

/* This is the "processMsg()" for Charm and Charm++ */

/* This is the "handleMsg()" for Charm and Charm++ */
int HANDLE_INCOMING_MSG() ;

extern void CkLdbSend();

void mainModuleInit()
{
}


void CkProcess_ForChareMsg(envelope)
ENVELOPE *envelope;
{
  int current_ep, current_magic, current_msgType; void *current_usr;
  CHARE_BLOCK *current_block;
  EP_STRUCT *current_epinfo;
  current_ep = GetEnv_EP(envelope);
  current_epinfo = CsvAccess(EpInfoTable) + current_ep;
  if (current_epinfo->language == CHARMPLUSPLUS)
    { CkProcess_Cplus_ForChareMsg(envelope); return; }
  current_magic = GetEnv_chare_magic_number(envelope);
  current_msgType = GetEnv_msgType(envelope);
  current_usr = USER_MSG_PTR(envelope);
  current_block = GetEnv_chareBlockPtr(envelope);
  CpvAccess(currentChareBlock) = (void *)current_block;
  CpvAccess(nodeforCharesProcessed)++;

  if (current_magic != GetID_chare_magic_number(current_block->selfID))
    CmiPrintf("[%d] *** ERROR *** Invalid or expired chareID used at entry point %s.\n", CmiMyPe(), current_epinfo->name);

  /* Run the entry-point */
  trace_begin_execute(envelope);
  (current_epinfo->function) (current_usr,current_block+1);
  trace_end_execute(current_magic, current_msgType, current_ep);
  QDCountThisProcessing(current_msgType);
}


void CkProcess_DynamicBocInitMsg(envelope)
ENVELOPE *envelope;
{
  /* ProcessBocInitMsg handles Charm++ bocs properly */
  /* This process of registering the new boc using the */
  /* spanning tree is exactly the same for Charm++ */
  int current_msgType = GetEnv_msgType(envelope);
  int executing_boc_num = ProcessBocInitMsg(envelope);
  RegisterDynamicBocInitMsg(&executing_boc_num, NULL);
  QDCountThisProcessing(current_msgType);
}

void CkProcess_NewChareMsg(envelope)
ENVELOPE *envelope;
{
  int current_ep, current_msgType, current_magic, current_chare;
  void *current_usr;
  CHARE_BLOCK *current_block;
  EP_STRUCT *current_epinfo;
  void * CreateChareBlock();

  current_ep = GetEnv_EP(envelope);
  current_epinfo = CsvAccess(EpInfoTable) + current_ep;
  if (current_epinfo->language==CHARMPLUSPLUS)
    { CkProcess_Cplus_NewChareMsg(envelope); return; }
  current_chare = current_epinfo->chareindex;
  current_usr = USER_MSG_PTR(envelope);
  current_msgType = GetEnv_msgType(envelope);
  current_block= (CHARE_BLOCK *)CreateChareBlock
    (CsvAccess(ChareSizesTable)[current_epinfo->chareindex]);
  current_magic = CpvAccess(nodecharesProcessed)++;
  CpvAccess(currentChareBlock) = current_block;
  SetID_chare_magic_number(current_block->selfID, current_magic);
  SetID_onPE(current_block->selfID, CmiMyPe());
  SetID_chareBlockPtr(current_block->selfID, current_block);

  /* If virtual block exists, get all messages for this chare	*/
  if (GetEnv_vidBlockPtr(envelope))
    VidSend(current_block,
	    GetEnv_vidPE(envelope),
	    GetEnv_vidBlockPtr(envelope));
  
  /* run the entry point */
  trace_begin_execute(envelope);
  (current_epinfo->function)(current_usr, current_block + 1);
  trace_end_execute(current_magic, current_msgType, current_ep);
  QDCountThisProcessing(current_msgType);
}


void CkProcess_BocMsg(env)
ENVELOPE *env;
{
  int current_ep, current_msgType, current_bocnum; void *current_usr;
  EP_STRUCT *current_epinfo;
  current_ep = GetEnv_EP(env);
  current_epinfo = CsvAccess(EpInfoTable) + current_ep;
  if (current_epinfo->language == CHARMPLUSPLUS)
    { CkProcess_Cplus_BocMsg(env); return; }
  current_usr = USER_MSG_PTR(env);
  current_msgType = GetEnv_msgType(env);
  current_bocnum = GetEnv_boc_num(env);
  trace_begin_execute(env);
  (current_epinfo->function)(current_usr, GetBocDataPtr(current_bocnum));
  trace_end_execute(current_bocnum, current_msgType, current_ep);
  CpvAccess(nodebocMsgsProcessed)++;
  QDCountThisProcessing(current_msgType);
}

void CkProcess_VidEnqueueMsg(env)
ENVELOPE *env;
{
  int current_msgType = GetEnv_msgType(env);
  void *current_usr = USER_MSG_PTR(env);
  trace_begin_execute(env);
  VidQueueUpInVidBlock(current_usr, NULL);
  trace_end_execute(0, current_msgType, 0);
  QDCountThisProcessing(current_msgType);
}

void CkProcess_VidSendOverMsg(env)
ENVELOPE *env;
{
  int current_msgType = GetEnv_msgType(env);
  void *current_usr = USER_MSG_PTR(env);
  trace_begin_execute(env);
  VidSendOverMessages(current_usr, NULL);
  trace_end_execute(0, current_msgType, 0);
  QDCountThisProcessing(current_msgType);
}

/* This is the handler function for Charm and Charm++, which is called
   immediately when a message is received (from self or network) */

HANDLE_INCOMING_MSG(env)
ENVELOPE *env;
{
  int ep, type = GetEnv_msgType(env);
  UNPACK(env);
  if (GetEnv_LdbFull(env))
    if(CpvAccess(InsideDataInit))
      CldStripLdb(LDB_ELEMENT_PTR(env));
  switch (type)
    {
    case NewChareMsg:
      CmiSetHandler(env,CsvAccess(CkProcIdx_NewChareMsg));
      CldNewSeedFromNet(env, LDB_ELEMENT_PTR(env),
			CkLdbSend,
			GetEnv_queueing(env),
			GetEnv_priosize(env),
			GetEnv_priobgn(env));
      break;
    case NewChareNoBalanceMsg:
      CmiSetHandler(env,CsvAccess(CkProcIdx_NewChareMsg));
      CkEnqueue(env);
      break;
    case ForChareMsg:
      CmiSetHandler(env,CsvAccess(CkProcIdx_ForChareMsg));
      CkEnqueue(env);
      break;
    case VidEnqueueMsg:
      CkProcess_VidEnqueueMsg(env);
      break;
    case VidSendOverMsg:
      CkProcess_VidSendOverMsg(env);
      break;
    case BocMsg:
    case BroadcastBocMsg:
      CmiSetHandler(env,CsvAccess(CkProcIdx_BocMsg));
      CkEnqueue(env);
      break;
    case LdbMsg:
    case QdBocMsg:
    case QdBroadcastBocMsg:
    case ImmBocMsg:
    case ImmBroadcastBocMsg:
      CkProcess_BocMsg(env);
      break;
    case DynamicBocInitMsg:
      CmiSetHandler(env,CsvAccess(CkProcIdx_DynamicBocInitMsg));
      CkEnqueue(env);
      break;
    default:
      CmiPrintf("** ERROR ** bad message type: %d\n",type);
    }
}


