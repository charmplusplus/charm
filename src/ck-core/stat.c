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
 * Revision 2.14  1997-07-18 21:21:11  milind
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
 * Revision 2.13  1997/07/08 15:28:40  milind
 * fixed an old bug that had been reintroduced in statistics collection
 * module.
 *
 * Revision 2.12  1995/11/07 17:56:51  sanjeev
 * fixed bugs in statistics collection
 *
 * Revision 2.11  1995/11/06  22:59:01  sanjeev
 * fixes for statistics collection
 *
 * Revision 2.10  1995/10/27  21:31:25  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.9  1995/09/07  05:27:47  gursoy
 * fixed some missing CpvAccess usages.
 *
 * Revision 2.8  1995/09/06  21:48:50  jyelon
 * Eliminated 'CkProcess_BocMsg', using 'CkProcess_ForChareMsg' instead.
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
 * Revision 2.4  1995/07/06  22:42:11  narain
 * Changes for LDB interface revision
 *
 * Revision 2.3  1995/06/29  21:53:35  narain
 * Changed LdbBocNum to CpvAccess(LdbBocNum)
 *
 * Revision 2.2  1995/06/14  16:31:04  brunner
 * Initialized EpLanguageTable in StatAddSysBocEps().  This should probably
 * call RegisterBocEp() instead.
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.5  1995/05/04  22:05:47  jyelon
 * *** empty log message ***
 *
 * Revision 1.4  1995/05/04  22:04:17  jyelon
 * *** empty log message ***
 *
 * Revision 1.3  1995/04/13  20:55:41  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.2  1994/12/09  15:42:37  sanjeev
 * interoperability stuff
 *
 * Revision 1.1  1994/11/03  17:39:42  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#define NODESTAT

#include "chare.h"
#include "globals.h"
#include "trace.h"
#include "stat.h"

typedef int **ARRAY_;
CpvStaticDeclare(ARRAY_, HostStat);
CpvStaticDeclare(ARRAY_, HostMemStatistics);
CpvStaticDeclare(int, NumPes);

extern CollectTraceFromNodes();
extern CHARE_BLOCK *CreateChareBlock();


void statModuleInit()
{
    CpvInitialize(ARRAY_, HostStat);
    CpvInitialize(ARRAY_, HostMemStatistics);
    CpvInitialize(int, NumPes);
}





StatInit()
{
	CHARE_BLOCK *bocBlock;
	int i;

	CpvAccess(NumPes) = CmiNumPes();
	CpvAccess(RecdStatMsg) = 1;
	if (CmiMyPe() == 0)
		CpvAccess(RecdStatMsg) = 0;
}

StatisticBocInit()
{
	CHARE_BLOCK *bocBlock;

	bocBlock = CreateChareBlock(0, CHAREKIND_BOCNODE, 0);
        bocBlock->x.boc_num = StatisticBocNum;
	SetBocBlockPtr(StatisticBocNum, bocBlock);
}


NodeCollectStatistics(msgPtr, localdataPtr)
void *msgPtr;
void *localdataPtr;
{
	STAT_MSG *sPtr;
	int i;

TRACE(CmiPrintf("Node %d: Enter NodeCollectStatistics() \n", CmiMyPe()));

	sPtr = (STAT_MSG *) CkAllocMsg(sizeof(STAT_MSG)); 
	CkMemError(sPtr);

	sPtr->srcPE = CmiMyPe();
	sPtr->chareQueueLength = CpvAccess(CstatsMaxChareQueueLength);
	sPtr->forChareQueueLength = CpvAccess(CstatsMaxForChareQueueLength);
	sPtr->fixedChareQueueLength = CpvAccess(CstatsMaxFixedChareQueueLength);
	sPtr->charesCreated = CpvAccess(nodecharesCreated);
	sPtr->charesProcessed = CpvAccess(nodecharesProcessed)
					- CpvAccess(nodebocInitProcessed) - 1 ;
	/* The -1 above is because counting for user-created chares starts
	   from 1. The main-chare on proc 0 is hence not counted. */

	sPtr->forCharesCreated = CpvAccess(nodeforCharesCreated) 
					+ CpvAccess(nodebocMsgsCreated) ;
	/* This count includes both messages to chares and messages to BOCs,
	   because it is incremented in CkProcess_ForChareMsg() */
	sPtr->forCharesProcessed = CpvAccess(nodeforCharesProcessed);

	/* Messages to BOCs are not counted separately for now 
	sPtr->bocMsgsCreated = CpvAccess(nodebocMsgsCreated);
	sPtr->bocMsgsProcessed = CpvAccess(nodebocMsgsProcessed);
	*/

	for (i=0; i < MAXMEMSTAT; i++)
		sPtr->nodeMemStat[i] = CstatMemory(i);

	GeneralSendMsgBranch(CsvAccess(CkEp_Stat_CollectNodes), sPtr, 
		0, BocMsg, StatisticBocNum);
}




CollectStatistics()
{
	DUMMY_STAT_MSG *mPtr;

	TRACE(CmiPrintf("Host: Enter CollectStatistics(): and Call BroadcastMsgBranch()\n"));
	mPtr = (DUMMY_STAT_MSG *) CkAllocMsg(sizeof(DUMMY_STAT_MSG));
	CkMemError(mPtr);
	GeneralBroadcastMsgBranch(CsvAccess(CkEp_Stat_Data), mPtr,
				 ImmBroadcastBocMsg,
                                 StatisticBocNum);
}


CollectFromNodes(msgPtr, localdataptr)
void *msgPtr, *localdataptr;
{
	int i,j,k;
	STAT_MSG *mPtr = (STAT_MSG *) msgPtr;

TRACE(CmiPrintf("Host %d: Enter CollectFromNodes(): NumPes %d\n",
	 CmiMyPe(), CpvAccess(NumPes)));
	if (CpvAccess(NumPes) == CmiNumPes())
	{
	       CpvAccess(HostMemStatistics)=
                              (int **) CmiAlloc(sizeof(int *)*CpvAccess(NumPes));
	       CkMemError(CpvAccess(HostMemStatistics));
		for (i=0; i<CpvAccess(NumPes); i++)
		{
			CpvAccess(HostMemStatistics)[i] = 
                           (int *) CmiAlloc(sizeof(int)*MAXMEMSTAT);
			CkMemError(CpvAccess(HostMemStatistics)[i]);
		}
		CpvAccess(HostStat) = 
                           (int **) CmiAlloc(sizeof(int *)*CpvAccess(NumPes));
		CkMemError(CpvAccess(HostStat));
		for (i=0; i<CpvAccess(NumPes); i++)
		{
			CpvAccess(HostStat)[i]=(int *)CmiAlloc(sizeof(int)*10);
			CkMemError(CpvAccess(HostStat)[i]);
		}
		for (i=0; i<CpvAccess(NumPes); i++)
		{
			for (j=0; j<MAXMEMSTAT; j++)
				CpvAccess(HostMemStatistics)[i][j] = 0;
			for (j=0; j<10; j++)
				CpvAccess(HostStat)[i][j] = 0;
		}
	}
	CpvAccess(NumPes)--;

	CpvAccess(HostStat)[mPtr->srcPE][0] = mPtr->chareQueueLength;
	CpvAccess(HostStat)[mPtr->srcPE][1] = mPtr->forChareQueueLength;
	CpvAccess(HostStat)[mPtr->srcPE][2] = mPtr->fixedChareQueueLength;
	CpvAccess(HostStat)[mPtr->srcPE][3] = mPtr->charesCreated;
	CpvAccess(HostStat)[mPtr->srcPE][4] = mPtr->charesProcessed;
	CpvAccess(HostStat)[mPtr->srcPE][5] = mPtr->forCharesCreated;
	CpvAccess(HostStat)[mPtr->srcPE][6] = mPtr->forCharesProcessed;
	CpvAccess(HostStat)[mPtr->srcPE][7] = mPtr->bocMsgsCreated;
	CpvAccess(HostStat)[mPtr->srcPE][8] = mPtr->bocMsgsProcessed;

	for (k=0; k < MAXMEMSTAT; k++)
	  CpvAccess(HostMemStatistics)[mPtr->srcPE][k] = mPtr->nodeMemStat[k];
	
	/* Exit when statistics from all the nodes have been received */
	if (CpvAccess(NumPes) == 0)
	{
		CpvAccess(RecdStatMsg) = 1;
		if (CpvAccess(RecdTraceMsg)) ExitNode();
	}
}



PrintOutStatistics()
{
	int i,j,k;
	int col = 0;
	int totalHops = 0;
	int  totalChares = 0;
	int  totalCharesCrea = 0;
	int  totalCharesProc = 0;
	int totalChareQ = 0, totalForChareQ = 0, totalMemoryUsage = 0;
	int totalMemoryOverflow = 0;
	ENVELOPE * env;
	char *msg;

	if (CstatPrintQueueStats())
	{
		CmiPrintf("Queue Statistics: (NODE)[MaxChareQ, MaxForChareQ, MaxFixedChareQ]\n");
		for (k=0; k < CmiNumPes(); k++)
		{
			totalChareQ += CpvAccess(HostStat)[k][0];
			totalForChareQ += CpvAccess(HostStat)[k][1];
		}
		CmiPrintf("Average Queue Sizes: [AvgMaxChareQ %d, AvgMaxForChareQ %d]\n",
		    totalChareQ/CmiNumPes(), totalForChareQ/CmiNumPes());

		for (k=0; k < CmiNumPes(); k++)
			CmiPrintf("(%d)[%d, %d, %d], ", k, 
                                   CpvAccess(HostStat)[k][0], 
                                   CpvAccess(HostStat)[k][1], 
                                   CpvAccess(HostStat)[k][2]);
		CmiPrintf("\n\n");
	}


        if (CpvAccess(PrintChareStat) || CpvAccess(PrintSummaryStat))
		for (k=0; k < CmiNumPes(); k++)
		{
			totalCharesCrea += CpvAccess(HostStat)[k][3];
			totalCharesProc += CpvAccess(HostStat)[k][4];
		}


	if (CpvAccess(PrintSummaryStat))
	{
		CmiPrintf("\nPrinting Chare Summary Statistics:\n");
		CmiPrintf("Total Chares: [Created %d, Processed %d]\n",
		    totalCharesCrea, totalCharesProc);
	}


	if (CpvAccess(PrintChareStat))
	{
		CmiPrintf("----------------------------------------------\n");
		CmiPrintf("Printing Chare Statistics:\n");
		CmiPrintf("PE  Chares-Created Chares-Processed Messages-Created Messages-Processed\n");
		for (k=0; k < CmiNumPes(); k++) {
			CmiPrintf("%-3d",k) ; 

			CmiPrintf(" %14d %16d", CpvAccess(HostStat)[k][3], 
					    CpvAccess(HostStat)[k][4]);
			CmiPrintf(" %16d %18d", CpvAccess(HostStat)[k][5],
					    CpvAccess(HostStat)[k][6]);
			CmiPrintf("\n") ;
		}

		CmiPrintf("\nNumber of Branch-Office Chares : %d\n",
					CpvAccess(nodebocInitProcessed));


		/* Boc msgs not counted separately for now 
		CmiPrintf("For Boc Messages: ");
		for (k=0; k < CmiNumPes(); k++)
			CmiPrintf("(%d)[%d, %d], ", k, 
                          CpvAccess(HostStat)[k][7], CpvAccess(HostStat)[k][8]);
		*/
		
		CmiPrintf("\n\n");

	}
        
	if (CstatPrintMemStats())
	{
		CmiPrintf("Printing Memory Statistics:\n\n");
                CmiPrintf("Available Memory: %d (words)\n",
                         CpvAccess(HostMemStatistics)[0][0]);
                CmiPrintf(" Node     Unused         Allocated                   Freed\n");
                CmiPrintf(" Node     (words)     (no.req, words)            (no.req, words)\n");
                CmiPrintf("------   --------    ---------------------      ---------------------\n");
		for (k=0; k < CmiNumPes(); k++)
                        CmiPrintf("%4d    %8d     [%8d,%10d]      [%8d,%10d]\n",
                        k,CpvAccess(HostMemStatistics)[k][1],
                        CpvAccess(HostMemStatistics)[k][2],
                        CpvAccess(HostMemStatistics)[k][3]*2,
                        CpvAccess(HostMemStatistics)[k][4],
                        CpvAccess(HostMemStatistics)[k][5]*2);
                CmiPrintf("\n");
	}
}

StatAddSysBocEps()
{
  extern BroadcastExitMessage(), ExitMessage();

  CsvAccess(CkEp_Stat_CollectNodes)=
    registerBocEp("CkEp_Stat_CollectNodes",
		  CollectFromNodes,
		  CHARM, 0, 0);
  CsvAccess(CkEp_Stat_Data)=
    registerBocEp("CkEp_Stat_Data",
		  NodeCollectStatistics,
		  CHARM, 0, 0);
  CsvAccess(CkEp_Stat_TraceCollectNodes)=
    registerBocEp("CkEp_Stat_TraceCollectNodes",
		  CollectTraceFromNodes,
		  CHARM, 0, 0);
  CsvAccess(CkEp_Stat_BroadcastExitMessage)=
    registerBocEp("CkEp_Stat_BroadcastExitMessage",
		  BroadcastExitMessage,
		  CHARM, 0, 0);
  CsvAccess(CkEp_Stat_ExitMessage)=
    registerBocEp("CkEp_Stat_ExitMessage",
		  ExitMessage,
		  CHARM, 0, 0);
}
