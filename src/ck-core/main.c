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
 * Revision 2.29  1998-02-27 11:52:06  jyelon
 * Cleaned up header files, replaced load-balancer.
 *
 * Revision 2.28  1998/01/28 17:52:49  milind
 * Removed unnecessary function calls to tracing functions.
 * Added macros to turn tracing on and off at runtime.
 *
 * Revision 2.27  1997/12/12 05:03:43  jyelon
 * Fixed bug, wasn't doing CmiGrabBuffer.
 *
 * Revision 2.26  1997/07/18 21:21:09  milind
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
 * Revision 2.25  1997/07/18 19:14:52  milind
 * Fixed the perfModuleInit call to pass command-line params.
 * Also added trace_enqueue call to Charm message handler.
 *
 * Revision 2.24  1996/02/10 18:11:11  sanjeev
 * fixed bug: if(CpvAccess(InsideDataInit)) CldStripLdb(LDB_ELEMENT_PTR(env));
 *
 * Revision 2.23  1995/11/06 00:17:26  sanjeev
 * in CkProcess_ForChareMsg, magic number taken from chareblock, not env
 *
 * Revision 2.22  1995/10/13  18:15:53  jyelon
 * K&R changes.
 *
 * Revision 2.21  1995/10/11  17:54:40  sanjeev
 * fixed Charm++ chare creation
 *
 * Revision 2.20  1995/09/29  09:51:12  jyelon
 * Many small corrections.
 *
 * Revision 2.19  1995/09/20  16:36:26  jyelon
 * *** empty log message ***
 *
 * Revision 2.18  1995/09/20  15:41:38  gursoy
 * removed the if form handle incoming message
 *
 * Revision 2.17  1995/09/20  14:24:27  jyelon
 * *** empty log message ***
 *
 * Revision 2.16  1995/09/07  05:27:11  gursoy
 * modified HANDLE_INCOMING_MSG to buffer messages
 *
 * Revision 2.15  1995/09/06  21:48:50  jyelon
 * Eliminated 'CkProcess_BocMsg', using 'CkProcess_ForChareMsg' instead.
 *
 * Revision 2.14  1995/09/06  04:08:51  sanjeev
 * fixed bugs
 *
 * Revision 2.13  1995/09/05  22:01:29  sanjeev
 * modified CkProcess_ForChareMsg, CkProcess_NewChareMsg, CkProcess_BocMsg
 * to integrate Charm++
 *
 * Revision 2.12  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.11  1995/07/27  20:29:34  jyelon
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

#include "charm.h"

#include "trace.h"

CHARE_BLOCK *GetBocBlockPtr();

extern void CkLdbSend();
void HANDLE_INCOMING_MSG();

void mainModuleInit()
{
}

void CheckMagicNumber(chare, env)
    CHARE_BLOCK *chare; ENVELOPE *env;
{
  if (GetID_chare_magic_number(chare->selfID) !=
      GetEnv_chare_magic_number(env)) {
    CmiPrintf("[%d] *** ERROR *** dead chare or bad chareID used at entry point %s.\n", 
	      CmiMyPe(), CsvAccess(EpInfoTable)[GetEnv_EP(env)].name);
    exit(1);
  }
}

void CkProcess_ForChareMsg_to_UVID(env)
ENVELOPE *env;
{
  if(CpvAccess(traceOn))
    trace_begin_execute(env);
  VidEnqueueMsg(env);
  if(CpvAccess(traceOn))
    trace_end_execute(0, ForChareMsg, 0);
  QDCountThisProcessing(ForChareMsg);
}

void CkProcess_ForChareMsg_to_FVID(env)
ENVELOPE *env;
{
  if(CpvAccess(traceOn))
    trace_begin_execute(env);
  VidForwardMsg(env);
  if(CpvAccess(traceOn))
    trace_end_execute(0, ForChareMsg, 0);
  QDCountThisProcessing(ForChareMsg);
}

void CkProcess_ForChareMsg_to_Chare(env)
ENVELOPE *env;
{
  CHARE_BLOCK *chareblock;
  int          current_ep;
  EP_STRUCT   *current_epinfo;
  void        *current_usr;
  int          current_magic;
  int          current_msgType;

  chareblock      = GetEnv_chareBlockPtr(env);
  current_ep      = GetEnv_EP(env);
  current_epinfo  = CsvAccess(EpInfoTable) + current_ep;
  current_usr     = USER_MSG_PTR(env);
  current_magic   = chareblock->selfID.magic;
  current_msgType = GetEnv_msgType(env);

  CpvAccess(currentChareBlock) = (void *)chareblock;
  CpvAccess(nodeforCharesProcessed)++;
  if(CpvAccess(traceOn))
    trace_begin_execute(env);
  (current_epinfo->function)(current_usr,chareblock->chareptr);
  if(CpvAccess(traceOn))
    trace_end_execute(current_magic, current_msgType, current_ep);
  QDCountThisProcessing(current_msgType);
}

void CkProcess_ForChareMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *chare = GetEnv_chareBlockPtr(env);
  CheckMagicNumber(chare, env);
  switch (chare->charekind) {
  case CHAREKIND_UVID: CkProcess_ForChareMsg_to_UVID(env); break;
  case CHAREKIND_FVID: CkProcess_ForChareMsg_to_FVID(env); break;
  case CHAREKIND_CHARE: CkProcess_ForChareMsg_to_Chare(env); break;
  case CHAREKIND_BOCNODE: CkProcess_ForChareMsg_to_Chare(env); break;
  }
}

void CkProcess_BocMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *chare;
  chare = GetBocBlockPtr(GetEnv_boc_num(env));
  SetEnv_chareBlockPtr(env, chare);
  CkProcess_ForChareMsg_to_Chare(env);
}

void CkProcess_DynamicBocInitMsg(env)
ENVELOPE *env;
{
  /* ProcessBocInitMsg handles Charm++ bocs properly */
  /* This process of registering the new boc using the */
  /* spanning tree is exactly the same for Charm++ */
  int current_msgType, executing_boc_num;

  current_msgType = GetEnv_msgType(env);
  executing_boc_num = ProcessBocInitMsg(env);
  RegisterDynamicBocInitMsg(&executing_boc_num, NULL);
  QDCountThisProcessing(current_msgType);
}

void CkProcess_NewChareMsg(env)
ENVELOPE *env;
{
  int current_ep, current_msgType, current_magic, current_chare;
  void *current_usr;
  CHARE_BLOCK *current_block;
  EP_STRUCT *current_epinfo;
  CHARE_BLOCK *CreateChareBlock();

  current_ep = GetEnv_EP(env);
  current_epinfo = CsvAccess(EpInfoTable) + current_ep;
  current_chare = current_epinfo->chareindex;
  current_usr = USER_MSG_PTR(env);
  current_msgType = GetEnv_msgType(env);
  current_magic = CpvAccess(nodecharesProcessed)++;
  CpvAccess(currentChareBlock) = current_block = CreateChareBlock(
				    CsvAccess(ChareSizesTable)[current_chare],
				    CHAREKIND_CHARE, current_magic);

  /* If virtual block exists, get all messages for this chare	*/
  if (GetEnv_vidBlockPtr(env))
    VidRetrieveMessages(current_block,
	    GetEnv_vidPE(env),
	    GetEnv_vidBlockPtr(env));

  /* Now call the entry point : For Charm, this is the actual call to the 
     EP function. For Charm++, this is a call to the translator-generated
     stub function which does "new ChareType(msg)". 
     NOTE: CreateChareBlock sets current_block->chareptr to (current_block + 1)
     because ChareSizesTable[] holds the correct size of the chare for both 
     Charm and Charm++, so it allocates the correct amount of space.
     - Sanjeev 10/10/95 */

  if(CpvAccess(traceOn))
    trace_begin_execute(env);
  (current_epinfo->function)(current_usr, current_block->chareptr);
  if(CpvAccess(traceOn))
    trace_end_execute(current_magic, current_msgType, current_ep);
  QDCountThisProcessing(current_msgType);
}

void CkProcess_VidSendOverMsg(env)
ENVELOPE *env;
{
  int current_msgType; void *current_usr;
  
  current_msgType = GetEnv_msgType(env);
  current_usr = USER_MSG_PTR(env);
  if(CpvAccess(traceOn))
    trace_begin_execute(env);
  VidSendOverMessages(current_usr, NULL);
  if(CpvAccess(traceOn))
    trace_end_execute(0, current_msgType, 0);
  QDCountThisProcessing(current_msgType);
}




/* This is the message handler for non-init messages during the
   initialization phase. It simply buffers the messafe */
void BUFFER_INCOMING_MSG(env)
ENVELOPE *env;
{
  if (CpvAccess(CkInitPhase)) {
    CmiGrabBuffer(&env);
    FIFO_EnQueue(CpvAccess(CkBuffQueue),(void *)env);
  } else {
    HANDLE_INCOMING_MSG(env);
  }
}


/* This is the handler function for Charm and Charm++, which is called
   immediately when a message is received (from self or network) */

void HANDLE_INCOMING_MSG(env)
ENVELOPE *env;
{
  CmiGrabBuffer(&env);
  UNPACK(env);
  if(CpvAccess(traceOn))
    trace_enqueue(env);
  switch (GetEnv_msgType(env)) {
  case NewChareMsg:          CkProcess_NewChareMsg(env); break;
  case ForChareMsg:          CkProcess_ForChareMsg(env); break;
  case LdbMsg:               CkProcess_BocMsg(env); break;
  case QdBocMsg:             CkProcess_BocMsg(env); break;
  case QdBroadcastBocMsg:    CkProcess_BocMsg(env); break;
  case ImmBocMsg:            CkProcess_BocMsg(env); break;
  case ImmBroadcastBocMsg:   CkProcess_BocMsg(env); break;
  case BocMsg:               CkProcess_BocMsg(env); break;
  case BroadcastBocMsg:      CkProcess_BocMsg(env); break;
  case VidSendOverMsg:       CkProcess_VidSendOverMsg(env); break;
  case DynamicBocInitMsg:    CkProcess_DynamicBocInitMsg(env); break;
  case NewChareNoBalanceMsg:
    CmiAbort("** ERROR ** obsolete message type.\n");
  default:
    CmiAbort("** ERROR ** bad message type.\n");
  }
}


