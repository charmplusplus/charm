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
 * Revision 2.0  1995-06-02 17:27:40  brunner
 * Reorganized directory structure
 *
 * Revision 1.5  1995/04/23  20:55:17  sanjeev
 * Removed Core....
 *
 * Revision 1.4  1995/03/17  23:38:13  sanjeev
 * changes for better message format
 *
 * Revision 1.3  1994/12/02  00:02:05  sanjeev
 * interop stuff
 *
 * Revision 1.2  1994/11/11  05:31:21  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:43  brunner
 * Initial revision
 *
 ***************************************************************************/
/* #define IsCharmPlus(Entry) (Entry & 0xffff8000)	*/

#define IsCharmPlus(Entry) (EpLanguageTable[Entry]==CHARMPLUSPLUS)

#define IsCharmPlusPseudo(id) (PseudoTable[id].language==CHARMPLUSPLUS)


#define CkMemError(ptr) if (ptr == NULL) \
                CmiPrintf("*** ERROR *** Memory Allocation Failed --- consider +m command-line option.\n");

#define QDCountThisProcessing(msgType) \
         if ((msgType != QdBocMsg) && (msgType != QdBroadcastBocMsg) && \
			(msgType != LdbMsg)) msgs_processed++; 

#define QDCountThisCreation(ep, category, type, x) \
         if ((type != QdBocMsg) && (type != QdBroadcastBocMsg) && \
			(type != LdbMsg)) msgs_created += x;

#ifdef DEBUGGING_MODE
#define COPY_AND_SEND(env)  { \
	ENVELOPE *new = (ENVELOPE *) CkCopyEnv(env); \
	trace_creation(GetEnv_msgType(new), GetEnv_EP(new), new); \
        CmiSetHandler(env,HANDLE_INCOMING_MSG_Index); \
        CkAsyncSend(GetEnv_destPE(new), \
        CmiSize(new), new); }
#else
#define COPY_AND_SEND(env)  { \
	ENVELOPE *new = (ENVELOPE *) CkCopyEnv(env); \
        CmiSetHandler(env,HANDLE_INCOMING_MSG_Index); \
        CkAsyncSend(GetEnv_destPE(new), \
        CmiSize(new), new); }
#endif
