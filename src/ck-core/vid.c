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
 * Revision 2.10  1997-07-18 21:21:13  milind
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
 * Revision 2.9  1997/03/24 23:14:05  milind
 * Made Charm-runtime 64-bit safe by removing conversions of pointers to
 * integers. Also, removed charm runtime's dependence of unused argv[]
 * elements being 0. Also, added sim-irix-64 version. It works.
 *
 * Revision 2.8  1995/10/13 18:15:53  jyelon
 * K&R changes.
 *
 * Revision 2.7  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.6  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.5  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.4  1995/07/22  23:45:15  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/07/05  22:51:39  sanjeev
 * added comments at top
 *
 * Revision 2.2  1995/07/05  22:11:59  sanjeev
 * put SetEnv_EP(env, 0) in VidSend() to fix CM5 bug
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.4  1995/04/13  20:55:50  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.3  1995/04/02  00:48:02  sanjeev
 * changes for separating Converse
 *
 * Revision 1.2  1994/12/01  23:57:00  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:38:55  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";


/************************************************************************
Comments : 7/5/95 (after debugging bug on CM5 with Megatest++).
Bug was intermittent crash in TTAB and ACC.
Caused due to EP field not being filled in VidSend, so causing
EpLanguageTable to be indexed by 65535 in CallProcessMsg.
************************************************************************/




#include "chare.h"
#include "globals.h"
#include "trace.h"
#include "vid.h"

VidEnqueueMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *vid = GetEnv_chareBlockPtr(env);
  void *vidqueue = vid->x.vid_queue;
  FIFO_EnQueue(vidqueue, env);
}

VidForwardMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *vid = GetEnv_chareBlockPtr(env);
  SetEnv_chareBlockPtr(env, GetID_chareBlockPtr(vid->x.realID));
  SetEnv_chare_magic_number(env, GetID_chare_magic_number(vid->x.realID));
  trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
  CkCheck_and_Send(GetID_onPE(vid->x.realID), env);
  QDCountThisCreation(GetEnv_EP(env), USERcat, ForChareMsg, 1);
}

/************************************************************************/
/*			VidSendOverMessage				*/
/*	Once the chare ha been created it needs to get the messages	*/
/*	that were sent to it while it hadn't been created. These	*/
/* 	messages are queued up in its virtual id block, whose address	*/
/* 	it has available. The messages are then dequeued and sent over  */
/*      to the processor on which the chare was finally created.        */
/************************************************************************/

VidSendOverMessages(msgPtr, data_area)
CHARE_ID_MSG *msgPtr;
void *data_area;
{
    ChareIDType         ID;
    ENVELOPE 		*env;
    CHARE_BLOCK		*vidblock;
    void 		*vidqueue;
    int 		chare_magic;
    int                 chare_pe;
    CHARE_BLOCK        *chare_block;

    ID = msgPtr->ID;
    env = ENVELOPE_UPTR(msgPtr);
    vidblock = GetEnv_vidBlockPtr(env);
    chare_magic = GetID_chare_magic_number(ID);
    chare_pe = GetID_onPE(ID);
    chare_block  = GetID_chareBlockPtr(ID);

    if (vidblock->charekind != CHAREKIND_UVID) {
      CmiPrintf("system error #12983781\n");
      exit(1);
    }
    vidqueue = vidblock->x.vid_queue;
    while (!FIFO_Empty(vidqueue))
    {
	FIFO_DeQueue(vidqueue, &env);
	SetEnv_chareBlockPtr(env, chare_block);
	SetEnv_chare_magic_number(env, chare_magic);
	trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
	CkCheck_and_Send(chare_pe, env);
	QDCountThisCreation(GetEnv_EP(env), USERcat, ForChareMsg, 1);
   }
   vidblock->charekind = CHAREKIND_FVID;
   vidblock->x.realID = ID;
}


/************************************************************************/
/*			VidRetrieveMessages                             */
/*	This is used by the newly-created chare to request the          */
/*      messages that were stored in its VID                            */
/************************************************************************/

VidRetrieveMessages(chareblockPtr,vidPE,vidBlockPtr)
CHARE_BLOCK * chareblockPtr;
PeNumType  vidPE;
CHARE_BLOCK *vidBlockPtr;
{
    CHARE_ID_MSG * msg;
    ENVELOPE * env;

    msg = (CHARE_ID_MSG *)CkAllocMsg(sizeof(CHARE_ID_MSG));
    CkMemError(msg);
    msg->ID = CpvAccess(currentChareBlock)->selfID;
    env = ENVELOPE_UPTR(msg);
    SetEnv_msgType(env, VidSendOverMsg);
    SetEnv_vidBlockPtr(env, vidBlockPtr);
    SetEnv_EP(env, 0);

    QDCountThisCreation(0, IMMEDIATEcat, VidSendOverMsg, 1);
    trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
    CkCheck_and_Send(vidPE, env);
}
