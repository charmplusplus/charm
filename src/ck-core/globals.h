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
 * Revision 1.8  1995/04/23  17:47:02  sanjeev
 * removed declaration of LanguageHandlerTable
 *
 * Revision 1.7  1995/04/23  14:26:27  brunner
 * Removed sysDone, since it is in converse.h
 *
 * Revision 1.6  1995/04/02  00:48:37  sanjeev
 * changes for separating Converse
 *
 * Revision 1.5  1995/03/24  16:42:51  sanjeev
 * *** empty log message ***
 *
 * Revision 1.4  1995/03/17  23:38:03  sanjeev
 * changes for better message format
 *
 * Revision 1.3  1994/12/02  00:02:03  sanjeev
 * interop stuff
 *
 * Revision 1.2  1994/11/11  05:24:52  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:17  brunner
 * Initial revision
 *
 ***************************************************************************/
#include "trans_defs.h"
#include "trans_decls.h"

extern int numPe;
extern int SysMem;
extern int TotalEps;
extern int TotalMsgs;
extern int TotalPseudos;
extern int NumReadMsg;
extern int MsgCount; 		

extern int MainDataSize;  	/* size of dataarea for main chare 	*/
extern int currentBocNum;
extern int InsideDataInit;
extern int mainChare_magic_number;
extern struct chare_block * mainChareBlock;
extern struct chare_block * currentChareBlock;

extern int          * EpLanguageTable ;
/* extern void **_CK_9_ReadMsgTable;  was in trans_decls.h, no longer global */
extern FUNCTION_PTR *ROCopyFromBufferTable, *ROCopyToBufferTable ;
extern int * EpIsImplicitTable;
extern int * EpToMsgTable;
extern int * EpChareTypeTable;
extern FUNCTION_PTR * EpTable;
extern MSG_STRUCT * MsgToStructTable; 
extern int  * ChareSizesTable;
extern FUNCTION_PTR * ChareFnTable ;
extern PSEUDO_STRUCT * PseudoTable;
extern char	     ** EpNameTable;

extern char **ChareNamesTable;
extern int *EpChareTable;

extern int msgs_processed, msgs_created;

extern int disable_sys_msgs;
extern int nodecharesProcessed;
extern int nodebocMsgsProcessed;
extern int nodeforCharesProcessed;
extern int nodecharesCreated;
extern int nodeforCharesCreated;
extern int nodebocMsgsCreated;

extern void *LocalQueueHead;
extern void *SchedQueue;

extern int PrintQueStat; 
extern int PrintMemStat; 
extern int PrintChareStat;
extern int PrintSummaryStat;

extern int RecdStatMsg;
extern int RecdPerfMsg;

extern int numHeapEntries, numCondChkArryElts;
extern int MainChareLanguage ;

extern int CallProcessMsg_Index ;
extern int HANDLE_INCOMING_MSG_Index ;

