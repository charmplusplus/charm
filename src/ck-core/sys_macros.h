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
 * Revision 2.3  1995-07-27 20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.2  1995/07/22  23:45:15  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
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

#define IsCharmPlus(Entry)\
    (CsvAccess(EpInfoTable)[Entry].language==CHARMPLUSPLUS)

#define IsCharmPlusPseudo(id) (CsvAccess(PseudoTable)[id].language==CHARMPLUSPLUS)


#define CkMemError(ptr) if (ptr == NULL) \
                CmiPrintf("*** ERROR *** Memory Allocation Failed --- consider +m command-line option.\n");

#define QDCountThisProcessing(msgType) \
         if ((msgType != QdBocMsg) && (msgType != QdBroadcastBocMsg) && \
			(msgType != LdbMsg)) CpvAccess(msgs_processed)++; 

#define QDCountThisCreation(ep, category, type, x) \
         if ((type != QdBocMsg) && (type != QdBroadcastBocMsg) && \
			(type != LdbMsg)) CpvAccess(msgs_created) += x;

