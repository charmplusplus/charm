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

CsvExtern(int, TotalEps);
CpvExtern(int, TotalMsgs);
CpvExtern(int, TotalPseudos);
CpvExtern(int, NumReadMsg);
CpvExtern(int, MsgCount); 		

CpvExtern(int, MainDataSize);  	/* size of dataarea for main chare 	*/
CpvExtern(int, currentBocNum);
CpvExtern(int, InsideDataInit);
CpvExtern(int, mainChare_magic_number);
typedef struct chare_block *CHARE_BLOCK_;
CpvExtern(CHARE_BLOCK_, mainChareBlock);
CpvExtern(CHARE_BLOCK_, currentChareBlock);

CsvExtern(int*, EpLanguageTable);
CsvExtern(FUNCTION_PTR*, ROCopyFromBufferTable);
CsvExtern(FUNCTION_PTR*, ROCopyToBufferTable);
CsvExtern(int*, EpIsImplicitTable);
CsvExtern(int*, EpToMsgTable);
CsvExtern(int*, EpChareTypeTable);
CsvExtern(FUNCTION_PTR*,  EpTable);
CsvExtern(MSG_STRUCT*, MsgToStructTable); 
CsvExtern(int*,  ChareSizesTable);
CsvExtern(FUNCTION_PTR*,  ChareFnTable);
CsvExtern(PSEUDO_STRUCT*, PseudoTable);
CsvExtern(char**, EpNameTable);

CsvExtern(char**, ChareNamesTable);
CsvExtern(int*, EpChareTable);

CpvExtern(int, msgs_processed);
CpvExtern(int, msgs_created);

CpvExtern(int, disable_sys_msgs);
CpvExtern(int, nodecharesProcessed);
CpvExtern(int, nodebocMsgsProcessed);
CpvExtern(int, nodeforCharesProcessed);
CpvExtern(int, nodecharesCreated);
CpvExtern(int, nodeforCharesCreated);
CpvExtern(int, nodebocMsgsCreated);


CpvExtern(int, PrintQueStat); 
CpvExtern(int, PrintMemStat); 
CpvExtern(int, PrintChareStat);
CpvExtern(int, PrintSummaryStat);

CpvExtern(int, RecdStatMsg);
CpvExtern(int, RecdPerfMsg);

CpvExtern(int, numHeapEntries);
CpvExtern(int, numCondChkArryElts);

CsvExtern(int, MainChareLanguage);

CsvExtern(int, CallProcessMsg_Index);
CsvExtern(int, HANDLE_INCOMING_MSG_Index);

