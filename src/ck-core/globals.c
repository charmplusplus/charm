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
 * Revision 1.7  1995/05/04  22:03:51  jyelon
 * *** empty log message ***
 *
 * Revision 1.6  1995/05/03  20:57:13  sanjeev
 * bug fixes for finding uninitialized modules
 *
 * Revision 1.5  1995/04/02  00:47:16  sanjeev
 * changes for separating Converse
 *
 * Revision 1.4  1995/03/12  17:08:13  sanjeev
 * changes for new msg macros
 *
 * Revision 1.3  1995/01/17  23:45:52  knauff
 * Added variables for the '++outputfile' option in the network version.
 *
 * Revision 1.2  1994/12/01  23:55:06  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:38:32  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "chare.h"
#include "trans_defs.h"
#include <stdio.h>

/**********************************************************************/
/* These fields are needed by message macros. Any changes must be
reflected there. */
/**********************************************************************/

CpvDeclare(int, PAD_SIZE);
CpvDeclare(int, HEADER_SIZE);
CpvDeclare(int, _CK_Env_To_Usr);
CpvDeclare(int, _CK_Ldb_To_Usr);
CpvDeclare(int, _CK_Usr_To_Env);
CpvDeclare(int, _CK_Usr_To_Ldb);




/**********************************************************************/
/* Other global variables. */
/**********************************************************************/
CsvDeclare(int, TotalEps);
CpvDeclare(int, NumReadMsg);
CpvDeclare(int, MsgCount); 	/* for the initial, pre-loop phase.
				to count up all the messages
		 		being sent out to nodes 		*/
CpvDeclare(int, InsideDataInit);
CpvDeclare(int, mainChare_magic_number);
typedef struct chare_block *CHARE_BLOCK_;
CpvDeclare(CHARE_BLOCK_,  mainChareBlock);
CpvDeclare(CHARE_BLOCK_, currentChareBlock);
CpvDeclare(int, currentBocNum);
CpvDeclare(int, MainDataSize);  /* size of dataarea for main chare 	*/


CsvDeclare(int*, EpLanguageTable);


CsvDeclare(FUNCTION_PTR*, ROCopyFromBufferTable);
CsvDeclare(FUNCTION_PTR*, ROCopyToBufferTable);
CsvDeclare(int*, EpIsImplicitTable);
CsvDeclare(FUNCTION_PTR*, EpTable); /* actual table to be allocated dynamically
			     	    depending on the number of entry-points */
CsvDeclare(int*, EpToMsgTable);  /* Table mapping EPs to associated messages.*/
CsvDeclare(int*, EpChareTypeTable);  /* Table mapping EPs to chare type 
				      (CHARE or BOC) */ 	
CsvDeclare(MSG_STRUCT*, MsgToStructTable); /* Table mapping message to 
                                           struct table*/
CsvDeclare(PSEUDO_STRUCT*,  PseudoTable);
CsvDeclare(int*, EpChareTable);
CsvDeclare(char**, EpNameTable);
CsvDeclare(int*, ChareSizesTable);
CsvDeclare(FUNCTION_PTR*, ChareFnTable);
CsvDeclare(char**, ChareNamesTable);

CpvDeclare(int, msgs_processed);
CpvDeclare(int, msgs_created);

CpvDeclare(int, nodecharesCreated);
CpvDeclare(int, nodeforCharesCreated);
CpvDeclare(int, nodebocMsgsCreated);
CpvDeclare(int, nodecharesProcessed);
CpvDeclare(int, nodebocMsgsProcessed);
CpvDeclare(int, nodeforCharesProcessed);


CpvDeclare(int, PrintChareStat); 
CpvDeclare(int, PrintSummaryStat);

CpvDeclare(int, RecdStatMsg);
CpvDeclare(int, RecdPerfMsg);

CpvDeclare(int, numHeapEntries);      /* heap of tme-dep calls   */
CpvDeclare(int, numCondChkArryElts);  /* arry hldng conditon check info */


CpvDeclare(int, _CK_13PackOffset);
CpvDeclare(int, _CK_13PackMsgCount);
CpvDeclare(int, _CK_13ChareEPCount);
CpvDeclare(int, _CK_13TotalMsgCount);

CsvDeclare(FUNCTION_PTR*,  _CK_9_GlobalFunctionTable);

CsvDeclare(int, MainChareLanguage);


void globalsModuleInit()
{

   CpvInitialize(int, PAD_SIZE);
   CpvInitialize(int, HEADER_SIZE);
   CpvInitialize(int, _CK_Env_To_Usr);
   CpvInitialize(int, _CK_Ldb_To_Usr);
   CpvInitialize(int, _CK_Usr_To_Env);
   CpvInitialize(int, _CK_Usr_To_Ldb);
   CpvInitialize(int, NumReadMsg);
   CpvInitialize(int, MsgCount); 
   CpvInitialize(int, InsideDataInit);
   CpvInitialize(int, mainChare_magic_number);
   CpvInitialize(CHARE_BLOCK_,  mainChareBlock);
   CpvInitialize(CHARE_BLOCK_, currentChareBlock);
   CpvInitialize(int, currentBocNum);
   CpvInitialize(int, MainDataSize); 
   CpvInitialize(int, msgs_processed);
   CpvInitialize(int, msgs_created);
   CpvInitialize(int, nodecharesCreated);
   CpvInitialize(int, nodeforCharesCreated);
   CpvInitialize(int, nodebocMsgsCreated);
   CpvInitialize(int, nodecharesProcessed);
   CpvInitialize(int, nodebocMsgsProcessed);
   CpvInitialize(int, nodeforCharesProcessed);
   CpvInitialize(int, PrintChareStat);
   CpvInitialize(int, PrintSummaryStat);
   CpvInitialize(int, RecdStatMsg);
   CpvInitialize(int, RecdPerfMsg);
   CpvInitialize(int, numHeapEntries);      
   CpvInitialize(int, numCondChkArryElts); 
   CpvInitialize(int, _CK_13PackOffset);
   CpvInitialize(int, _CK_13PackMsgCount);
   CpvInitialize(int, _CK_13ChareEPCount);
   CpvInitialize(int, _CK_13TotalMsgCount);

  
   CpvAccess(NumReadMsg)             = 0; 
   CpvAccess(InsideDataInit)         = 0;
   CpvAccess(currentBocNum)          = (NumSysBoc - 1); /* was set to  -1 */
   CpvAccess(nodecharesCreated)      = 0;
   CpvAccess(nodeforCharesCreated)   = 0;
   CpvAccess(nodebocMsgsCreated)     = 0;
   CpvAccess(nodecharesProcessed)    = 0;
   CpvAccess(nodebocMsgsProcessed)   = 0;
   CpvAccess(nodeforCharesProcessed) = 0;
   CpvAccess(PrintChareStat)         = 0;
   CpvAccess(PrintSummaryStat)       = 0;
   CpvAccess(numHeapEntries)         = 0;  
   CpvAccess(numCondChkArryElts)     = 0; 
   CsvAccess(MainChareLanguage)      = -1;
}
