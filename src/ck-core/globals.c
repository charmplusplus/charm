#include "charm.h"
#include <stdio.h>

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
CpvDeclare(CHARE_BLOCK_,  mainChareBlock);
CpvDeclare(CHARE_BLOCK_, currentChareBlock);
CpvDeclare(int, currentBocNum);
CpvDeclare(int, MainDataSize);  /* size of dataarea for main chare 	*/


CsvDeclare(FUNCTION_PTR*, ROCopyFromBufferTable);
CsvDeclare(FUNCTION_PTR*, ROCopyToBufferTable);
CsvDeclare(EP_STRUCT*, EpInfoTable);
CsvDeclare(MSG_STRUCT*, MsgToStructTable);
CsvDeclare(PSEUDO_STRUCT*, PseudoTable);
CsvDeclare(char**, EventTable);
CsvDeclare(int*, ChareSizesTable);
CsvDeclare(char**, ChareNamesTable);

CpvDeclare(int, msgs_processed);
CpvDeclare(int, msgs_created);

CpvDeclare(int, nodecharesCreated);
CpvDeclare(int, nodeforCharesCreated);
CpvDeclare(int, nodebocMsgsCreated);
CpvDeclare(int, nodecharesProcessed);
CpvDeclare(int, nodebocInitProcessed);
CpvDeclare(int, nodebocMsgsProcessed);
CpvDeclare(int, nodeforCharesProcessed);


CpvDeclare(int, PrintChareStat); 
CpvDeclare(int, PrintSummaryStat);
CpvDeclare(int, QueueingDefault);

CpvDeclare(int, RecdStatMsg);

CpvDeclare(int, numHeapEntries);      /* heap of tme-dep calls   */
CpvDeclare(int, numCondChkArryElts);  /* arry hldng conditon check info */


CpvDeclare(int, _CK_13PackOffset);
CpvDeclare(int, _CK_13PackMsgCount);
CpvDeclare(int, _CK_13ChareEPCount);
CpvDeclare(int, _CK_13TotalMsgCount);

CsvDeclare(FUNCTION_PTR*,  _CK_9_GlobalFunctionTable);

CsvDeclare(int, MainChareLanguage);

/* Handlers for various message-types */
CpvDeclare(int, HANDLE_INCOMING_MSG_Index);
CsvDeclare(int, BUFFER_INCOMING_MSG_Index);
CsvDeclare(int, MAIN_HANDLE_INCOMING_MSG_Index);
CsvDeclare(int, HANDLE_INIT_MSG_Index);

/* System-defined chare numbers */
CsvDeclare(int, CkChare_ACC);
CsvDeclare(int, CkChare_MONO);

/* Entry points for Quiescence detection BOC 	*/
CsvDeclare(int, CkEp_QD_Init);
CsvDeclare(int, CkEp_QD_InsertQuiescenceList);
CsvDeclare(int, CkEp_QD_PhaseIBroadcast);
CsvDeclare(int, CkEp_QD_PhaseIMsg);
CsvDeclare(int, CkEp_QD_PhaseIIBroadcast);
CsvDeclare(int, CkEp_QD_PhaseIIMsg);

/* Entry points for Write Once Variables 	*/
CsvDeclare(int, CkEp_WOV_AddWOV);
CsvDeclare(int, CkEp_WOV_RcvAck);
CsvDeclare(int, CkEp_WOV_HostAddWOV);
CsvDeclare(int, CkEp_WOV_HostRcvAck);

/* Entry points for dynamic tables BOC    	*/
CsvDeclare(int, CkEp_Tbl_Unpack);

/* Entry points for accumulator BOC		*/
CsvDeclare(int, CkEp_ACC_CollectFromNode);
CsvDeclare(int, CkEp_ACC_LeafNodeCollect);
CsvDeclare(int, CkEp_ACC_InteriorNodeCollect);
CsvDeclare(int, CkEp_ACC_BranchInit);

/* Entry points for monotonic BOC		*/
CsvDeclare(int, CkEp_MONO_BranchInit);
CsvDeclare(int, CkEp_MONO_BranchUpdate);
CsvDeclare(int, CkEp_MONO_ChildrenUpdate);

/* These are the entry points necessary for the dynamic BOC creation. */
CsvDeclare(int, CkEp_DBOC_RegisterDynamicBocInitMsg);
CsvDeclare(int, CkEp_DBOC_OtherCreateBoc);
CsvDeclare(int, CkEp_DBOC_InitiateDynamicBocBroadcast);

/* These are the entry points for the statistics BOC */
CsvDeclare(int, CkEp_Stat_CollectNodes);
CsvDeclare(int, CkEp_Stat_Data);
CsvDeclare(int, CkEp_Stat_TraceCollectNodes);
CsvDeclare(int, CkEp_Stat_BroadcastExitMessage);
CsvDeclare(int, CkEp_Stat_ExitMessage);

/* Entry points for LoadBalancing BOC 		*/
CsvDeclare(int, CkEp_Ldb_NbrStatus);

CsvDeclare(int, NumSysBocEps);


/* Initialization phase count variables for synchronization */
CpvDeclare(int,CkInitCount);
CpvDeclare(int,CkCountArrived);


/* Buffer for the non-init messages received during the initialization phase */
CpvDeclare(void*, CkBuffQueue);


/* Initialization phase flag : 1 if in the initialization phase */
CpvDeclare(int, CkInitPhase);










void globalsModuleInit()
{
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
   CpvInitialize(int, nodebocInitProcessed);
   CpvInitialize(int, nodebocMsgsProcessed);
   CpvInitialize(int, nodeforCharesProcessed);
   CpvInitialize(int, PrintChareStat);
   CpvInitialize(int, PrintSummaryStat);
   CpvInitialize(int, QueueingDefault);
   CpvInitialize(int, RecdStatMsg);
   CpvInitialize(int, numHeapEntries);      
   CpvInitialize(int, numCondChkArryElts); 
   CpvInitialize(int, _CK_13PackOffset);
   CpvInitialize(int, _CK_13PackMsgCount);
   CpvInitialize(int, _CK_13ChareEPCount);
   CpvInitialize(int, _CK_13TotalMsgCount);
   CpvInitialize(int, CkInitPhase);
   CpvInitialize(int, CkInitCount);
   CpvInitialize(int, CkCountArrived);
   CpvInitialize(void*, CkBuffQueue); 
   CpvInitialize(int, HANDLE_INCOMING_MSG_Index);

   CpvAccess(NumReadMsg)             = 0; 
   CpvAccess(InsideDataInit)         = 0;
   CpvAccess(currentBocNum)          = (NumSysBoc - 1); /* was set to  -1 */
   CpvAccess(nodecharesCreated)      = 0;
   CpvAccess(nodeforCharesCreated)   = 0;
   CpvAccess(nodebocMsgsCreated)     = 0;
   CpvAccess(nodecharesProcessed)    = 0;
   CpvAccess(nodebocInitProcessed)   = 0;
   CpvAccess(nodebocMsgsProcessed)   = 0;
   CpvAccess(nodeforCharesProcessed) = 0;
   CpvAccess(PrintChareStat)         = 0;
   CpvAccess(PrintSummaryStat)       = 0;
   CpvAccess(numHeapEntries)         = 0;  
   CpvAccess(numCondChkArryElts)     = 0; 
   CpvAccess(CkInitPhase)            = 1;
   CpvAccess(CkInitCount)            = 0;
   CpvAccess(CkCountArrived)         = 0; 
   CpvAccess(CkBuffQueue)            = NULL;   

   if (CmiMyRank() == 0) CsvAccess(MainChareLanguage)  = -1;
}
