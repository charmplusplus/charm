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
 * Revision 2.17  1997-07-18 21:21:06  milind
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
 * Revision 2.16  1997/03/14 20:23:49  milind
 * Made MAXLOGBUFSIZE in projections a commandline parameter.
 * One can now specify it as "+logsize 10000" on the program
 * command line.
 *
 * Revision 2.15  1995/11/13 04:04:33  gursoy
 * made changes related to initial msg synchronization
 *
 * Revision 2.14  1995/11/06  22:59:01  sanjeev
 * fixes for statistics collection
 *
 * Revision 2.13  1995/10/27  21:31:25  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.12  1995/10/11  17:52:51  sanjeev
 * fixed Charm++ chare creation
 *
 * Revision 2.11  1995/09/20  16:07:56  jyelon
 * fyab
 *
 * Revision 2.10  1995/09/20  15:40:58  gursoy
 * added new handler indexes
 *
 * Revision 2.9  1995/09/14  20:49:01  jyelon
 * Added +fifo +lifo +ififo +ilifo +bfifo +blifo command-line options.
 *
 * Revision 2.8  1995/09/07  05:26:14  gursoy
 * introduced new global variables used by HANDLE_INIT_MSG
 *
 * Revision 2.7  1995/09/06  21:48:50  jyelon
 * Eliminated 'CkProcess_BocMsg', using 'CkProcess_ForChareMsg' instead.
 *
 * Revision 2.6  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.5  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.4  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/07/06  22:42:11  narain
 * Changes for LDB interface revision
 *
 * Revision 2.2  1995/06/29  21:32:35  narain
 * Added Extern declarations for LdbBocNum, NumPes and LDB_ELEM_SIZE
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
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

CsvExtern(FUNCTION_PTR*, ROCopyFromBufferTable);
CsvExtern(FUNCTION_PTR*, ROCopyToBufferTable);
CsvExtern(EP_STRUCT*, EpInfoTable);
CsvExtern(MSG_STRUCT*, MsgToStructTable); 
CsvExtern(int*,  ChareSizesTable);
CsvExtern(PSEUDO_STRUCT*, PseudoTable);

CsvExtern(char**, ChareNamesTable);

CpvExtern(int, msgs_processed);
CpvExtern(int, msgs_created);

CpvExtern(int, disable_sys_msgs);
CpvExtern(int, nodecharesProcessed);
CpvExtern(int, nodebocInitProcessed);
CpvExtern(int, nodebocMsgsProcessed);
CpvExtern(int, nodeforCharesProcessed);
CpvExtern(int, nodecharesCreated);
CpvExtern(int, nodeforCharesCreated);
CpvExtern(int, nodebocMsgsCreated);


CpvExtern(int, PrintQueStat); 
CpvExtern(int, PrintMemStat); 
CpvExtern(int, PrintChareStat);
CpvExtern(int, PrintSummaryStat);
CpvExtern(int, QueueingDefault);
CpvExtern(int, LogBufSize);

CpvExtern(int, RecdStatMsg);
CpvExtern(int, RecdTraceMsg);

CpvExtern(int, numHeapEntries);
CpvExtern(int, numCondChkArryElts);

CsvExtern(int, MainChareLanguage);

CpvExtern(int, LDB_ELEM_SIZE);

/* Handlers for various message-types */
CpvExtern(int, HANDLE_INCOMING_MSG_Index);
CsvExtern(int, BUFFER_INCOMING_MSG_Index);
CsvExtern(int, MAIN_HANDLE_INCOMING_MSG_Index);
CsvExtern(int, HANDLE_INIT_MSG_Index);
CsvExtern(int, CkProcIdx_ForChareMsg);
CsvExtern(int, CkProcIdx_DynamicBocInitMsg);
CsvExtern(int, CkProcIdx_NewChareMsg);
CsvExtern(int, CkProcIdx_VidSendOverMsg);

/* System-defined chare numbers */
CsvExtern(int, CkChare_ACC);
CsvExtern(int, CkChare_MONO);

/* Entry points for Quiescence detection BOC 	*/
CsvExtern(int, CkEp_QD_Init);
CsvExtern(int, CkEp_QD_InsertQuiescenceList);
CsvExtern(int, CkEp_QD_PhaseIBroadcast);
CsvExtern(int, CkEp_QD_PhaseIMsg);
CsvExtern(int, CkEp_QD_PhaseIIBroadcast);
CsvExtern(int, CkEp_QD_PhaseIIMsg);

/* Entry points for Write Once Variables 	*/
CsvExtern(int, CkEp_WOV_AddWOV);
CsvExtern(int, CkEp_WOV_RcvAck);
CsvExtern(int, CkEp_WOV_HostAddWOV);
CsvExtern(int, CkEp_WOV_HostRcvAck);


/* Entry points for dynamic tables BOC    	*/
CsvExtern(int, CkEp_Tbl_Unpack);

/* Entry points for accumulator BOC		*/
CsvExtern(int, CkEp_ACC_CollectFromNode);
CsvExtern(int, CkEp_ACC_LeafNodeCollect);
CsvExtern(int, CkEp_ACC_InteriorNodeCollect);
CsvExtern(int, CkEp_ACC_BranchInit);

/* Entry points for monotonic BOC		*/
CsvExtern(int, CkEp_MONO_BranchInit);
CsvExtern(int, CkEp_MONO_BranchUpdate);
CsvExtern(int, CkEp_MONO_ChildrenUpdate);

/* These are the entry points necessary for the dynamic BOC creation. */
CsvExtern(int, CkEp_DBOC_RegisterDynamicBocInitMsg);
CsvExtern(int, CkEp_DBOC_OtherCreateBoc);
CsvExtern(int, CkEp_DBOC_InitiateDynamicBocBroadcast);

/* These are the entry points for the statistics BOC */
CsvExtern(int, CkEp_Stat_CollectNodes);
CsvExtern(int, CkEp_Stat_Data);
CsvExtern(int, CkEp_Stat_TraceCollectNodes);
CsvExtern(int, CkEp_Stat_BroadcastExitMessage);
CsvExtern(int, CkEp_Stat_ExitMessage);

/* Entry points for LoadBalancing BOC 		*/
CsvExtern(int, CkEp_Ldb_NbrStatus);

CsvExtern(int, NumSysBocEps);



/* Initialization phase count variables for synchronization */
CpvExtern(int,CkInitCount);
CpvExtern(int,CkCountArrived);


/* Buffer for the non-init messages received during the initialization phase */
CpvExtern(void*, CkBuffQueue);

/* Initialization phase flag : 1 if in the initialization phase */
CpvExtern(int, CkInitPhase);
