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
 * Revision 2.16  1998-07-02 02:52:03  jyelon
 * Changed CldEnqueue to three parameters ( no pack function )
 *
 * Revision 2.15  1998/06/15 22:16:44  milind
 * Reduced Charm++ overhead by reducing variable accesses.
 *
 * Revision 2.14  1998/03/18 15:38:14  milind
 * Fixed a memory leak caused by vid_queue not being destroyed.
 *
 * Revision 2.13  1998/02/27 11:52:24  jyelon
 * Cleaned up header files, replaced load-balancer.
 *
 * Revision 2.12  1998/01/28 17:52:51  milind
 * Removed unnecessary function calls to tracing functions.
 * Added macros to turn tracing on and off at runtime.
 *
 * Revision 2.11  1997/10/29 23:52:54  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.10  1997/07/18 21:21:13  milind
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




#include "charm.h"
#include "trace.h"

typedef struct data_brnch_vid {
	int dummy;
} DATA_BR_VID;

typedef struct chare_id_msg{
        ChareIDType ID;
} CHARE_ID_MSG;

void VidEnqueueMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *vid = GetEnv_chareBlockPtr(env);
  void *vidqueue = vid->x.vid_queue;
  FIFO_EnQueue(vidqueue, env);
}

void VidForwardMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *vid = GetEnv_chareBlockPtr(env);
  SetEnv_chareBlockPtr(env, GetID_chareBlockPtr(vid->x.realID));
  SetEnv_chare_magic_number(env, GetID_chare_magic_number(vid->x.realID));
  if(CpvAccess(traceOn))
    trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
  CldEnqueue(GetID_onPE(vid->x.realID), env, CpvAccess(CkInfo_Index));
  QDCountThisCreation(1);
}

/************************************************************************/
/*			VidSendOverMessage				*/
/*	Once the charm.ha been created it needs to get the messages	*/
/*	that were sent to it while it hadn't been created. These	*/
/* 	messages are queued up in its virtual id block, whose address	*/
/* 	it has available. The messages are then dequeued and sent over  */
/*      to the processor on which the chare was finally created.        */
/************************************************************************/

void VidSendOverMessages(msgPtr, data_area)
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
        if(CpvAccess(traceOn))
	  trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
	CldEnqueue(chare_pe, env, CpvAccess(CkInfo_Index));
	QDCountThisCreation(1);
   }
   FIFO_Destroy(vidqueue);
   vidblock->charekind = CHAREKIND_FVID;
   vidblock->x.realID = ID;
}


/************************************************************************/
/*			VidRetrieveMessages                             */
/*	This is used by the newly-created chare to request the          */
/*      messages that were stored in its VID                            */
/************************************************************************/

void VidRetrieveMessages(chareblockPtr,vidPE,vidBlockPtr)
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

    QDCountThisCreation(1);
    if(CpvAccess(traceOn))
      trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
    CmiSetHandler(env, CpvAccess(HANDLE_INCOMING_MSG_Index));
    CldEnqueue(vidPE, env, CpvAccess(CkInfo_Index));
}
