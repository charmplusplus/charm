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
 * Revision 2.3  1995-06-29 21:53:35  narain
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
#include "performance.h"
#include "stat.h"

typedef int **ARRAY_;
CpvStaticDeclare(ARRAY_, HostStat);
CpvStaticDeclare(ARRAY_, HostMemStatistics);
CpvStaticDeclare(int, NumPe);

extern CollectPerfFromNodes();



void statModuleInit()
{
    CpvInitialize(ARRAY_, HostStat);
    CpvInitialize(ARRAY_, HostMemStatistics);
    CpvInitialize(int, NumPe);
}





StatInit()
{
	BOC_BLOCK *bocBlock;
	int i;

	CpvAccess(NumPe) = CmiNumPe();
	CpvAccess(RecdStatMsg) = 1;
	if (CmiMyPe() == 0)
		CpvAccess(RecdStatMsg) = 0;
}

StatisticBocInit()
{
	BOC_BLOCK *bocBlock;

	bocBlock = (BOC_BLOCK *) CreateBocBlock(0);
	bocBlock->boc_num = StatisticBocNum;
	SetBocDataPtr(StatisticBocNum, (void *) (bocBlock + 1));
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
	sPtr->chareQueueLength = CstatsMaxChareQueueLength;
	sPtr->forChareQueueLength = CstatsMaxForChareQueueLength;
	sPtr->fixedChareQueueLength = CstatsMaxFixedChareQueueLength;
	sPtr->charesCreated = CpvAccess(nodecharesCreated);
	sPtr->charesProcessed = CpvAccess(nodecharesProcessed);
	sPtr->forCharesCreated = CpvAccess(nodeforCharesCreated);
	sPtr->forCharesProcessed = CpvAccess(nodeforCharesProcessed);
	sPtr->bocMsgsCreated = CpvAccess(nodebocMsgsCreated);
	sPtr->bocMsgsProcessed = CpvAccess(nodebocMsgsProcessed);

	for (i=0; i < MAXMEMSTAT; i++)
		sPtr->nodeMemStat[i] = CstatMemory(i);

	GeneralSendMsgBranch(StatCollectNodes_EP, sPtr, 
		0, USERcat, BocMsg, CpvAccess(LdbBocNum));
}




CollectStatistics()
{
	DUMMY_STAT_MSG *mPtr;

	TRACE(CmiPrintf("Host: Enter CollectStatistics(): and Call BroadcastMsgBranch()\n"));
	mPtr = (DUMMY_STAT_MSG *) CkAllocMsg(sizeof(DUMMY_STAT_MSG));
	CkMemError(mPtr);
	GeneralBroadcastMsgBranch(StatData_EP, mPtr,
				 IMMEDIATEcat, BroadcastBocMsg, CpvAccess(LdbBocNum));
}


CollectFromNodes(msgPtr, localdataptr)
void *msgPtr, *localdataptr;
{
	int i,j,k;
	STAT_MSG *mPtr = (STAT_MSG *) msgPtr;

TRACE(CmiPrintf("Host %d: Enter CollectFromNodes(): NumPe %d\n",
	 CmiMyPe(), CpvAccess(NumPe)));
	if (CpvAccess(NumPe) == CmiNumPe())
	{
	       CpvAccess(HostMemStatistics)=
                              (int **) CmiAlloc(sizeof(int)*CpvAccess(NumPe));
	       CkMemError(CpvAccess(HostMemStatistics));
		for (i=0; i<CpvAccess(NumPe); i++)
		{
			CpvAccess(HostMemStatistics)[i] = 
                           (int *) CmiAlloc(sizeof(int)*MAXMEMSTAT);
			CkMemError(CpvAccess(HostMemStatistics)[i]);
		}
		CpvAccess(HostStat) = 
                           (int **) CmiAlloc(sizeof(int)*CpvAccess(NumPe));
		CkMemError(CpvAccess(HostStat));
		for (i=0; i<CpvAccess(NumPe); i++)
		{
			CpvAccess(HostStat)[i]=(int *)CmiAlloc(sizeof(int)*10);
			CkMemError(CpvAccess(HostStat)[i]);
		}
		for (i=0; i<CpvAccess(NumPe); i++)
		{
			for (j=0; j<MAXMEMSTAT; j++)
				CpvAccess(HostMemStatistics)[i][j] = 0;
			for (j=0; j<10; j++)
				CpvAccess(HostStat)[i][j] = 0;
		}
	}
	CpvAccess(NumPe)--;

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
	if (CpvAccess(NumPe) == 0)
	{
		CpvAccess(RecdStatMsg) = 1;
		if (CpvAccess(RecdPerfMsg)) ExitNode();
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
		for (k=0; k < CmiNumPe(); k++)
		{
			totalChareQ += CpvAccess(HostStat)[k][0];
			totalForChareQ += CpvAccess(HostStat)[k][1];
		}
		CmiPrintf("Average Queue Sizes: [AvgMaxChareQ %d, AvgMaxForChareQ %d]\n",
		    totalChareQ/CmiNumPe(), totalForChareQ/CmiNumPe());

		for (k=0; k < CmiNumPe(); k++)
			CmiPrintf("(%d)[%d, %d, %d], ", k, 
                                   CpvAccess(HostStat)[k][0], 
                                   CpvAccess(HostStat)[k][1], 
                                   CpvAccess(HostStat)[k][2]);
		CmiPrintf("\n\n");
	}


        if (CpvAccess(PrintChareStat) || CpvAccess(PrintSummaryStat))
		for (k=0; k < CmiNumPe(); k++)
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
		CmiPrintf("\nPrinting Chare Statistics:\n");
		CmiPrintf("Individual Chare Info: (NODE)[Created, Processed]\n");
		for (k=0; k < CmiNumPe(); k++)
			CmiPrintf("(%d)[%d, %d], ", k, 
                          CpvAccess(HostStat)[k][3], CpvAccess(HostStat)[k][4]);
		CmiPrintf("\nFor Chare Messages: ");
		for (k=0; k < CmiNumPe(); k++)
			CmiPrintf("(%d)[%d, %d], ", k, 
                          CpvAccess(HostStat)[k][5],CpvAccess(HostStat)[k][6]);
		CmiPrintf("\n");

		CmiPrintf("For Boc Messages: ");
		for (k=0; k < CmiNumPe(); k++)
			CmiPrintf("(%d)[%d, %d], ", k, 
                          CpvAccess(HostStat)[k][7], CpvAccess(HostStat)[k][8]);
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
		for (k=0; k < CmiNumPe(); k++)
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

	CsvAccess(EpTable)[StatCollectNodes_EP] = CollectFromNodes;
	CsvAccess(EpLanguageTable)[StatCollectNodes_EP] = CHARM;
	CsvAccess(EpTable)[StatData_EP] = NodeCollectStatistics;
	CsvAccess(EpLanguageTable)[StatData_EP] = CHARM;
	CsvAccess(EpTable)[StatPerfCollectNodes_EP] = CollectPerfFromNodes;
	CsvAccess(EpLanguageTable)[StatPerfCollectNodes_EP] = CHARM;
	CsvAccess(EpTable)[StatBroadcastExitMessage_EP] = BroadcastExitMessage;
	CsvAccess(EpLanguageTable)[StatBroadcastExitMessage_EP] = CHARM;
	CsvAccess(EpTable)[StatExitMessage_EP] = ExitMessage;
	CsvAccess(EpLanguageTable)[StatExitMessage_EP] = CHARM;
}
