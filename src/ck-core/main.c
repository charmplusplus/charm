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
 * Revision 2.1  1995-06-08 17:07:12  gursoy
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
int CallProcessMsg() ;
CsvDeclare(int, CallProcessMsg_Index);
/* This is the "handleMsg()" for Charm and Charm++ */
int HANDLE_INCOMING_MSG() ;
CsvDeclare(int, HANDLE_INCOMING_MSG_Index);



void mainModuleInit()
{
}




#ifdef REPLAY_DEBUGGING

HANDLE_INCOMING_MSG(env)
ENVELOPE *env;
{
	/* Fill in the language field in the message */
	CmiSetHandler(env,CsvAccess(CallProcessMsg_Index)) ;
	
	CsdEnqueue(env);
}

#else

/* This is the handler function for Charm and Charm++, which is called
   immediately when a message is received from the network */

HANDLE_INCOMING_MSG(env)
ENVELOPE *env;
{
	/* send to ldb strategy to extract load information */
	LdbStripMsg(env);

	switch (GetEnv_category(env)) {
	case IMMEDIATEcat :
		CallProcessMsg(env, USER_MSG_PTR(env));
		break;

	case USERcat :
		/* Fill in the language field in the message */
		CmiSetHandler(env,CsvAccess(CallProcessMsg_Index)) ;

        	if (!GetEnv_destPeFixed(env)) { 
			/* if destPeFixed==0, msg is always USERcat */
                	Ldb_NewChare_FromNet(env);
		}
		else 
			CsdEnqueue(env);
		break;

	default :
		CmiPrintf("*** ERROR *** Illegal Message Cat. %d\n", 
		    GetEnv_category(env));
	}
	TRACE(CmiPrintf("[%d] Handled message.\n", CmiMyPe()));
}
#endif





CallProcessMsg(envelope)
ENVELOPE *envelope;
{
	EntryPointType current_ep = GetEnv_EP(envelope);

	UNPACK(envelope);

	switch ( CsvAccess(EpLanguageTable)[current_ep] ) {
	    case CHARM :
		ProcessMsg(envelope) ;
		break ;

	    case CHARMPLUSPLUS :
		CPlus_ProcessMsg(envelope) ;
		break ;

	    default :
		CmiPrintf("[%d] ERROR : Language type of entry-point %d undefined. Possibly uninitialized module.\n",CmiMyPe(),current_ep) ;
	}
}
	

ProcessMsg(envelope)
ENVELOPE *envelope;
{
/* only Charm messages come here */

	int id;
	void * CreateChareBlock();
	ChareNumType executing_boc_num;
	int current_msgType = GetEnv_msgType(envelope);
	EntryPointType current_ep = GetEnv_EP(envelope);
	void *usrMsg ;

	TRACE(CmiPrintf("[%d] ProcessMsg: msgType = %d, ep = %d\n",
	    CmiMyPe(), current_msgType, current_ep));

	usrMsg = USER_MSG_PTR(envelope);
	switch (current_msgType)
	{

	case NewChareMsg:

		TRACE(CmiPrintf("[%d] Loop: isVID=%d, sizeData=%d\n",
		    CmiMyPe(), GetEnv_isVID(envelope), GetEnv_sizeData(envelope)));

		/* allocate data area, and strart execution. */
		CpvAccess(currentChareBlock) = (struct chare_block *)
		    CreateChareBlock(GetEnv_sizeData(envelope));
		SetID_chare_magic_number(CpvAccess(currentChareBlock)->selfID,
		    CpvAccess(nodecharesProcessed));

		TRACE(CmiPrintf("[%d] Loop: currentChareBlock=0x%x, magic=%d\n",
		    CmiMyPe(), CpvAccess(currentChareBlock), 
		    GetID_chare_magic_number(CpvAccess(currentChareBlock)->selfID)));

		/* If virtual block exists, get all messages for this chare	*/
		if (GetEnv_isVID(envelope))
			VidSend(CpvAccess(currentChareBlock), GetEnv_onPE(envelope),
			    GetEnv_vidBlockPtr(envelope));
		trace_begin_execute(envelope);
		(*(CsvAccess(EpTable)[current_ep])) (usrMsg, CpvAccess(currentChareBlock) + 1);
		trace_end_execute(CpvAccess(nodecharesProcessed), current_msgType, current_ep);

		CpvAccess(nodecharesProcessed)++;
		break;


	case ForChareMsg:
		TRACE(CmiPrintf("[%d] Loop: Message type is ForChareMsg.\n",
		    CmiMyPe()));

		CpvAccess(currentChareBlock) = (void *) GetEnv_chareBlockPtr(envelope);
		CpvAccess(nodeforCharesProcessed)++;

		TRACE(CmiPrintf("[%d] Loop: currentChareBlock=0x%x\n",
		    CmiMyPe(), CpvAccess(currentChareBlock)));
		TRACE(CmiPrintf("[%d] Loop: envelope_magic=%d, id_magic=%d\n",
		    CmiMyPe(), GetEnv_chare_magic_number(envelope),
		    GetID_chare_magic_number(CpvAccess(currentChareBlock)->selfID)));

		if (GetEnv_chare_magic_number(envelope) ==
		    GetID_chare_magic_number(CpvAccess(currentChareBlock)->selfID))
		{
			id = GetEnv_chare_magic_number(envelope);
			trace_begin_execute(envelope);
			(*(CsvAccess(EpTable)[current_ep]))
			    (usrMsg,CpvAccess(currentChareBlock) + 1);
			trace_end_execute(id, current_msgType, current_ep);
		}
		else 
			CmiPrintf("[%d] *** ERROR *** Message to dead chare at entry point %d.\n", CmiMyPe(),  CsvAccess(EpChareTable)[current_ep]);

		break;


	case DynamicBocInitMsg:

		/* ProcessBocInitMsg handles Charm++ bocs properly */
		executing_boc_num = ProcessBocInitMsg(envelope);

		/* This process of registering the new boc using the
			   spanning tree is exactly the same for Charm++ */
		RegisterDynamicBocInitMsg(&executing_boc_num, NULL);
		break;


	case BocMsg:
	case LdbMsg:
	case QdBocMsg:
	case BroadcastBocMsg:
	case QdBroadcastBocMsg:
		executing_boc_num = GetEnv_boc_num(envelope);
		trace_begin_execute(envelope);

TRACE(CmiPrintf("[%d] ProcessMsg: Executing message for %d boc %d\n", 
CmiMyPe(), current_ep, executing_boc_num));

		(*(CsvAccess(EpTable)[current_ep]))(usrMsg, 
			    GetBocDataPtr(executing_boc_num));

		trace_end_execute(executing_boc_num, current_msgType, current_ep);
		CpvAccess(nodebocMsgsProcessed)++;
		break;


	case VidMsg:
		current_ep = GetEnv_vidEP(envelope);
		trace_begin_execute(envelope);
		(*(CsvAccess(EpTable)[current_ep])) (usrMsg, NULL);
		trace_end_execute(VidBocNum, current_msgType, current_ep);
		break;


	default :
		CmiPrintf("*** ERROR *** Illegal Msg %d in Loop for EP %d.\n", GetEnv_msgType(envelope), GetEnv_EP(envelope));
	}
	QDCountThisProcessing(current_msgType);
}
