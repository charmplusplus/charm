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

static int **HostStat;
static int **HostMemStatistics;
static int NumPe;

extern CollectPerfFromNodes();


StatInit()
{
	BOC_BLOCK *bocBlock;
	int i;

	NumPe = CmiNumPe();
	RecdStatMsg = 1;
	if (CmiMyPe() == 0)
		RecdStatMsg = 0;
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
	sPtr->charesCreated = nodecharesCreated;
	sPtr->charesProcessed = nodecharesProcessed;
	sPtr->forCharesCreated = nodeforCharesCreated;
	sPtr->forCharesProcessed = nodeforCharesProcessed;
	sPtr->bocMsgsCreated = nodebocMsgsCreated;
	sPtr->bocMsgsProcessed = nodebocMsgsProcessed;

	for (i=0; i < MAXMEMSTAT; i++)
		sPtr->nodeMemStat[i] = CstatMemory(i);

	GeneralSendMsgBranch(StatCollectNodes_EP, sPtr, 
		0, USERcat, BocMsg, LdbBocNum);
}




CollectStatistics()
{
	DUMMY_STAT_MSG *mPtr;

	TRACE(CmiPrintf("Host: Enter CollectStatistics(): and Call BroadcastMsgBranch()\n"));
	mPtr = (DUMMY_STAT_MSG *) CkAllocMsg(sizeof(DUMMY_STAT_MSG));
	CkMemError(mPtr);
	GeneralBroadcastMsgBranch(StatData_EP, mPtr,
				 IMMEDIATEcat, BroadcastBocMsg, LdbBocNum);
}


CollectFromNodes(msgPtr, localdataptr)
void *msgPtr, *localdataptr;
{
	int i,j,k;
	STAT_MSG *mPtr = (STAT_MSG *) msgPtr;

TRACE(CmiPrintf("Host %d: Enter CollectFromNodes(): NumPe %d\n",
	 CmiMyPe(), NumPe));
	if (NumPe == CmiNumPe())
	{
		HostMemStatistics = (int **) CmiAlloc(sizeof(int)*NumPe);
		CkMemError(HostMemStatistics);
		for (i=0; i<NumPe; i++)
		{
			HostMemStatistics[i] = 
                           (int *) CmiAlloc(sizeof(int)*MAXMEMSTAT);
			CkMemError(HostMemStatistics[i]);
		}
		HostStat = (int **) CmiAlloc(sizeof(int)*NumPe);
		CkMemError(HostStat);
		for (i=0; i<NumPe; i++)
		{
			HostStat[i] = (int *) CmiAlloc(sizeof(int)*10);
			CkMemError(HostStat[i]);
		}
		for (i=0; i<NumPe; i++)
		{
			for (j=0; j<MAXMEMSTAT; j++)
				HostMemStatistics[i][j] = 0;
			for (j=0; j<10; j++)
				HostStat[i][j] = 0;
		}
	}
	NumPe--;

	HostStat[mPtr->srcPE][0] = mPtr->chareQueueLength;
	HostStat[mPtr->srcPE][1] = mPtr->forChareQueueLength;
	HostStat[mPtr->srcPE][2] = mPtr->fixedChareQueueLength;
	HostStat[mPtr->srcPE][3] = mPtr->charesCreated;
	HostStat[mPtr->srcPE][4] = mPtr->charesProcessed;
	HostStat[mPtr->srcPE][5] = mPtr->forCharesCreated;
	HostStat[mPtr->srcPE][6] = mPtr->forCharesProcessed;
	HostStat[mPtr->srcPE][7] = mPtr->bocMsgsCreated;
	HostStat[mPtr->srcPE][8] = mPtr->bocMsgsProcessed;

	for (k=0; k < MAXMEMSTAT; k++)
		HostMemStatistics[mPtr->srcPE][k] = mPtr->nodeMemStat[k];
	
	/* Exit when statistics from all the nodes have been received */
	if (NumPe == 0)
	{
		RecdStatMsg = 1;
		if (RecdPerfMsg) ExitNode();
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
			totalChareQ += HostStat[k][0];
			totalForChareQ += HostStat[k][1];
		}
		CmiPrintf("Average Queue Sizes: [AvgMaxChareQ %d, AvgMaxForChareQ %d]\n",
		    totalChareQ/CmiNumPe(), totalForChareQ/CmiNumPe());

		for (k=0; k < CmiNumPe(); k++)
			CmiPrintf("(%d)[%d, %d, %d], ", k, HostStat[k][0], HostStat[k][1], HostStat[k][2]);
		CmiPrintf("\n\n");
	}


        if (PrintChareStat || PrintSummaryStat)
		for (k=0; k < CmiNumPe(); k++)
		{
			totalCharesCrea += HostStat[k][3];
			totalCharesProc += HostStat[k][4];
		}


	if (PrintSummaryStat)
	{
		CmiPrintf("\nPrinting Chare Summary Statistics:\n");
		CmiPrintf("Total Chares: [Created %d, Processed %d]\n",
		    totalCharesCrea, totalCharesProc);
	}


	if (PrintChareStat)
	{
		CmiPrintf("\nPrinting Chare Statistics:\n");
		CmiPrintf("Individual Chare Info: (NODE)[Created, Processed]\n");
		for (k=0; k < CmiNumPe(); k++)
			CmiPrintf("(%d)[%d, %d], ", k, HostStat[k][3], HostStat[k][4]);
		CmiPrintf("\nFor Chare Messages: ");
		for (k=0; k < CmiNumPe(); k++)
			CmiPrintf("(%d)[%d, %d], ", k, HostStat[k][5], HostStat[k][6]);
		CmiPrintf("\n");

		CmiPrintf("For Boc Messages: ");
		for (k=0; k < CmiNumPe(); k++)
			CmiPrintf("(%d)[%d, %d], ", k, HostStat[k][7], HostStat[k][8]);
		CmiPrintf("\n\n");

	}
        
	if (CstatPrintMemStats())
	{
		CmiPrintf("Printing Memory Statistics:\n\n");
                CmiPrintf("Available Memory: %d (words)\n",
                         HostMemStatistics[0][0]);
                CmiPrintf(" Node     Unused         Allocated                   Freed\n");
                CmiPrintf(" Node     (words)     (no.req, words)            (no.req, words)\n");
                CmiPrintf("------   --------    ---------------------      ---------------------\n");
		for (k=0; k < CmiNumPe(); k++)
                        CmiPrintf("%4d    %8d     [%8d,%10d]      [%8d,%10d]\n",
                        k,HostMemStatistics[k][1],
                        HostMemStatistics[k][2],HostMemStatistics[k][3]*2,
                        HostMemStatistics[k][4],HostMemStatistics[k][5]*2);
                CmiPrintf("\n");
	}
}



StatAddSysBocEps()
{
	extern BroadcastExitMessage(), ExitMessage();

	EpTable[StatCollectNodes_EP] = CollectFromNodes;
	EpTable[StatData_EP] = NodeCollectStatistics;
	EpTable[StatPerfCollectNodes_EP] = CollectPerfFromNodes;
	EpTable[StatBroadcastExitMessage_EP] = BroadcastExitMessage;
	EpTable[StatExitMessage_EP] = ExitMessage;
}
