#define NODESTAT

#include "charm.h"
#include "trace.h"

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





void StatInit(void)
{
	CHARE_BLOCK *bocBlock;
	int i;

	CpvAccess(NumPes) = CmiNumPes();
	CpvAccess(RecdStatMsg) = 1;
	if (CmiMyPe() == 0)
		CpvAccess(RecdStatMsg) = 0;
}

void StatisticBocInit(void)
{
	CHARE_BLOCK *bocBlock;

	bocBlock = CreateChareBlock(0, CHAREKIND_BOCNODE, 0);
        bocBlock->x.boc_num = StatisticBocNum;
	SetBocBlockPtr(StatisticBocNum, bocBlock);
}


void NodeCollectStatistics(msgPtr, localdataPtr)
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




void CollectStatistics(void)
{
	DUMMY_MSG *mPtr;

	TRACE(CmiPrintf("Host: Enter CollectStatistics(): and Call BroadcastMsgBranch()\n"));
	mPtr = (DUMMY_MSG *) CkAllocMsg(sizeof(DUMMY_MSG));
	CkMemError(mPtr);
	GeneralBroadcastMsgBranch(CsvAccess(CkEp_Stat_Data), mPtr,
				 ImmBroadcastBocMsg,
                                 StatisticBocNum);
}


void CollectFromNodes(msgPtr, localdataptr)
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
		if (CpvAccess(CtrRecdTraceMsg)) ExitNode();
	}
}



void PrintOutStatistics(void)
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

void StatAddSysBocEps(void)
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
