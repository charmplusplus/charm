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
 * Revision 1.16  1995/05/03  20:56:44  sanjeev
 * bug fixes for finding uninitialized modules
 *
 * Revision 1.15  1995/05/03  06:28:20  sanjeev
 * registered _CkNullFunc
 *
 * Revision 1.14  1995/05/02  20:37:46  milind
 * Added _CkNullFunc()
 *
 * Revision 1.13  1995/04/25  03:40:01  sanjeev
 * fixed dynamic boc creation bug in ProcessBocInitMsg
 *
 * Revision 1.12  1995/04/23  20:53:14  sanjeev
 * Removed Core....
 *
 * Revision 1.11  1995/04/13  20:54:18  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.10  1995/04/06  17:46:21  sanjeev
 * fixed bug in tracing Charm++ BranchInits
 *
 * Revision 1.9  1995/04/02  00:48:57  sanjeev
 * changes for separating Converse
 *
 * Revision 1.8  1995/03/25  18:25:40  sanjeev
 * *** empty log message ***
 *
 * Revision 1.7  1995/03/24  16:42:59  sanjeev
 * *** empty log message ***
 *
 * Revision 1.6  1995/03/21  20:55:23  sanjeev
 * Changes for new converse names
 *
 * Revision 1.5  1995/03/17  23:38:27  sanjeev
 * changes for better message format
 *
 * Revision 1.4  1995/03/12  17:10:39  sanjeev
 * changes for new msg macros
 *
 * Revision 1.3  1994/12/02  00:02:26  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:39:52  brunner
 * Initial revision
 *
 ***************************************************************************/
static char     ident[] = "@(#)$Header$";
/***************************************************************************/
/* This is the main process that runs on each pe                           */
/* */
/***************************************************************************/
#include "chare.h"
#include "globals.h"
#include "performance.h"


/* these two variables are set in registerMainChare() */
int             _CK_MainChareIndex;
int             _CK_MainEpIndex;

int             ReadBuffSize = 0; /* this is set in registerReadOnly() */
void          **_CK_9_ReadMsgTable;	/* was previously global */
char           *ReadBufIndex;
char           *ReadFromBuffer;

#define BLK_LEN 512
BOCINIT_QUEUE  *BocInitQueueHead;
void *BocInitQueueCreate() ;
ENVELOPE *DeQueueBocInitMsgs() ;

/* these store argc, argv for use by CharmInit */
extern int             userArgc;
extern char          **userArgv;


/* all these counts are incremented in register.c */
extern int      fnCount;
extern int      msgCount;
extern int      chareCount;
extern int      chareEpsCount;	/* count of chare AND boc eps */
extern int      pseudoCount;
extern int      readMsgCount;
extern int      readCount;	/* count of read-only functions. Should be
				 * same as total number of modules */
extern int 	handlerCount ;  /* count of "language" handlers */


extern void CPlus_ProcessBocInitMsg();	/* in cplus_node_init.c */
extern void CPlus_CallCharmInit();	/* in cplus_node_init.c */
extern void CPlus_SetMainChareID();	/* in cplus_node_init.c */
extern void *CreateChareBlock();


/* This is the "processMsg()" for Charm and Charm++ */
extern void CallProcessMsg() ;
extern int CallProcessMsg_Index ;

/* This is the "handleMsg()" for Charm and Charm++ */
extern void HANDLE_INCOMING_MSG() ;
extern int HANDLE_INCOMING_MSG_Index ;


/*Added By Milind 05/02/95 */

void _CkNullFunc()
{
	CmiPrintf("[%d] In Null Function: Module Uninitialized\n", CmiMyPe());
}

/**** This is now in converse.c *****************
startup(argc, argv)
int             argc;
char           *argv[];
{
	int             MainDataSize;
	int             eps, num_boc, i;
	void           *MainDataArea;
	userArgc = ParseCommandOptions(argc, argv);

	InitializeMessageMacros();

        CkMemInit();
        CmiInit(argc, argv);
	CsdInit();
	OtherQsInit();
	StatInit();
	InitializeDynamicBocMsgList();
	InitializeBocDataTable();
	InitializeBocIDMessageCountTable();
	InitializeEPTables();


	log_init();
#ifdef DEBUGGING_MODE
	trace_begin_computation();
#endif
	SysBocInit();	
	msgs_created = msgs_processed = 0;
	CondSendInit();
	userArgv = argv;
}
*************************************************/



static int CountArgs(argv)
char **argv;
{
    int argc=0;
    while (*argv) { argc++; argv++; }
    return argc;
}

char **userArgv;
int    userArgc;

StartCharm(argv)
char **argv;
{
	int             i;
	char           *ReadBufMsg;

        userArgc = ParseCommandOptions(CountArgs(argv), argv);
        userArgv = argv;

	InitializeMessageMacros();

	CmiSpanTreeInit();

        /* OtherQsInit(); this was combined with CsdInitialize */
        StatInit();
        InitializeDynamicBocMsgList();
        InitializeBocDataTable();
        InitializeBocIDMessageCountTable();
        InitializeEPTables();
 
        log_init();
#ifdef DEBUGGING_MODE
        trace_begin_computation();
#endif
        SysBocInit();
        msgs_created = msgs_processed = 0;
        CondSendInit();



	if (CmiMyPe() == 0)
	{
		MsgCount = 0;	/* count # of messages being sent to each
				 * node. assume an equal number gets sent to
				 * every one. if there is a difference, have
				 * to modify this somewhat */
#ifdef DEBUGGING_MODE
		trace_begin_charminit();
#endif
		if (MainChareLanguage == CHARMPLUSPLUS)
		{
			CPlus_CallCharmInit(userArgc, userArgv);
		}
		else
		{
			MainDataSize = ChareSizesTable[_CK_MainChareIndex];
			mainChareBlock = currentChareBlock =
				(CHARE_BLOCK *) CreateChareBlock(MainDataSize);

			SetID_chare_magic_number(mainChareBlock->selfID,
						 rand());
			/* Calling CharmInit entry point */
			NumReadMsg = 0;
			InsideDataInit = 1;

			(EpTable[_CK_MainEpIndex]) (NULL, currentChareBlock + 1,
						    userArgc, userArgv);
			InsideDataInit = 0;
		}
#ifdef DEBUGGING_MODE
		trace_end_charminit();
#endif
		/* create the buffer for the read only variables */
		ReadBufMsg = (char *) CkAllocMsg(ReadBuffSize);
		ReadBufIndex = ReadBufMsg;
		if (ReadBuffSize > 0)
			CkMemError(ReadBufMsg);

		/*
		 * in Charm++ the CopyToBuffer fns also send out the
		 * ReadonlyMsgs by calling ReadMsgInit()
		 */
		for (i = 0; i < readCount; i++)
			(ROCopyToBufferTable[i]) ();

		/*
		 * we are sending the id of the main chare along with the
		 * read only message. in future versions, we might eliminate
		 * this because the functionality can be expressed using
		 * readonly variables and MyChareID inside the main chare
		 */

		BroadcastReadBuffer(ReadBufMsg, ReadBuffSize, mainChareBlock);

		/*
		 * send a message with the count of initial messages sent so
		 * far, to all nodes; includes messages for read-buffer and
		 * bocs
		 */
		BroadcastCount();
	}
	else
	{
		CharmInitLoop();
	}
	SysPeriodicCheckInit();

	/* Loop(); 	- Narain 11/16 */
}


/*
 * Receive read only variable buffer, read only messages and  BocInit
 * messages. Start Boc's by allocating and filling the NodeBocTbl. Wait until
 * (a) the  "count" message is received and (b) "count" number of initial
 * messages are received.
 */
CharmInitLoop()
{
	int             i, id;
	void           *usrMsg;
	int             countInit = 0;
	extern void    *CmiGetMsg();
	int         countArrived = 0;
	ENVELOPE       *envelope, *readvarmsg;

	BocInitQueueHead = (BOCINIT_QUEUE *) BocInitQueueCreate();

	while ((!countArrived) || (countInit != 0))
	{
		envelope = NULL;
		while (envelope == NULL)
			envelope = (ENVELOPE *) CmiGetMsg();
		if ((GetEnv_msgType(envelope) == BocInitMsg) ||
		    (GetEnv_msgType(envelope) == ReadMsgMsg))
			UNPACK(envelope);
		usrMsg = USER_MSG_PTR(envelope);
		/* Have a valid message now. */
		switch (GetEnv_msgType(envelope))
		{

		case BocInitMsg:
			EnQueueBocInitMsgs(envelope);
			countInit++;
			break;

		case InitCountMsg:
			countArrived = 1;
			countInit -= GetEnv_count(envelope);
			CmiFree(envelope);
			break;

		case ReadMsgMsg:
			id = GetEnv_other_id(envelope);
			_CK_9_ReadMsgTable[id] = (void *) usrMsg;
			countInit++;
			break;

		case ReadVarMsg:
			ReadFromBuffer = usrMsg;
			countInit++;

			/* get the information about the main chare */
			mainChareBlock = (struct chare_block *)
				GetEnv_chareBlockPtr(envelope);
			mainChare_magic_number =
				GetEnv_chare_magic_number(envelope);
			if (MainChareLanguage == CHARMPLUSPLUS)
				CPlus_SetMainChareID();
			readvarmsg = envelope;
			break;

		default:
		        CmiSetHandler(envelope,CallProcessMsg_Index) ;
			CsdEnqueue(envelope);
			break;
		}
	}

	/*
	 * call all the CopyFromBuffer functions for ReadOnly variables.
	 * _CK_9_ReadMsgTable is passed as an arg because it is no longer
	 * global
	 */
	for (i = 0; i < readCount; i++)
		(ROCopyFromBufferTable[i]) (_CK_9_ReadMsgTable);
	CmiFree(readvarmsg);

	while ( (envelope=DeQueueBocInitMsgs()) != NULL ) 
		ProcessBocInitMsg(envelope);
}

ProcessBocInitMsg(envelope)
ENVELOPE       *envelope;
{
	BOC_BLOCK      *bocBlock;
	void           *usrMsg = USER_MSG_PTR(envelope);
	int             current_ep = GetEnv_EP(envelope);
	int             executing_boc_num = GetEnv_boc_num(envelope);
	int             current_msgType = GetEnv_msgType(envelope);
	if (IsCharmPlus(current_ep))
	{			/* Charm++ BOC */
		CPlus_ProcessBocInitMsg(envelope, usrMsg, executing_boc_num, 
					current_msgType, current_ep);
	}
	else
	{
		bocBlock = (BOC_BLOCK *) CreateBocBlock
			(GetEnv_sizeData(envelope));
		bocBlock->boc_num = executing_boc_num;
		SetBocDataPtr(executing_boc_num, (void *) (bocBlock + 1));
		trace_begin_execute(envelope);
		(*(EpTable[current_ep]))
			(usrMsg, GetBocDataPtr(executing_boc_num));
		trace_end_execute(executing_boc_num, current_msgType,
				  current_ep);
	}

	/* for dynamic BOC creation, used in node_main.c */
	return executing_boc_num ;
}


/* this call can only be made after the clock has been initialized */

SysPeriodicCheckInit()
{
	LdbPeriodicCheckInit();
}


int 
ParseCommandOptions(argc, argv)
int             argc;
char          **argv;
{
/* Removed Converse options into ConverseParseCommandOptions. - Sanjeev */
	/*
	 * configure the chare kernel according to command line parameters.
	 * by convention, chare kernel parameters begin with '+'.
	 */
	int             i, j, numSysOpts = 0, foundSysOpt = 0, end;
	int             mainflag = 0, memflag = 0;

	if (argc < 1)
	{
		CmiPrintf("Too few arguments. Usage> host_prog node_prog [...]\n");
		exit(1);
	}

	end = argc;
	if (CmiMyPe() == 0)
		mainflag = 1;

	for (i = 1; i < end; i++)
	{
		foundSysOpt = 0;
		if (strcmp(argv[i], "+cs") == 0)
		{
			PrintChareStat = 1;
			/*
			 * if (mainflag) CmiPrintf("Chare Statistics Turned
			 * On\n");
			 */
			foundSysOpt = 1;
		}
		else if (strcmp(argv[i], "+ss") == 0)
		{
			PrintSummaryStat = 1;
			/*
			 * if(mainflag)CmiPrintf("Summary Statistics Turned
			 * On\n");
			 */
			foundSysOpt = 1;
		}
                else if (strcmp(argv[i], "+p") == 0 && i + 1 < argc)
                {
                        sscanf(argv[i + 1], "%d", &numPe);
                        foundSysOpt = 2;
                }
                else if (sscanf(argv[i], "+p%d", &numPe) == 1)
                {
                        foundSysOpt = 1;
                }
		if (foundSysOpt)
		{
			/* if system option, remove it. */
			numSysOpts += foundSysOpt;
			end -= foundSysOpt;
			for (j = i; j < argc - foundSysOpt; j++)
			{
				argv[j] = argv[j + foundSysOpt];
			}
			/* reset i because we shuffled everything down one */
			i--;
		}

	}
	return (argc - numSysOpts);
}



#define TABLE_SIZE 256

InitializeEPTables()
{
	int             i;
	int             TotalFns;
	int             TotalMsgs;
	int             TotalChares;
	int             TotalModules;
	int             TotalReadMsgs;
	int             TotalPseudos;


	/*
	 * TotalEps 	=  _CK_5mainChareEPCount(); TotalFns	=
	 * _CK_5mainFunctionCount(); TotalMsgs	=  _CK_5mainMessageCount();
	 * TotalChares 	=  _CK_5mainChareCount(); TotalBocEps 	=
	 * NumSysBocEps + _CK_5mainBranchEPCount();
	 */
	TotalEps = TABLE_SIZE;
	TotalFns = TABLE_SIZE;
	TotalMsgs = TABLE_SIZE;
	TotalChares = TABLE_SIZE;
	TotalModules = TABLE_SIZE;
	TotalReadMsgs = TABLE_SIZE;
	TotalPseudos = TABLE_SIZE;

	/*
	 * this table is used to store all ReadOnly Messages on processors
	 * other than proc 0. After they are received, they are put in the
	 * actual variables in the user program in the ...CopyFromBuffer
	 * functions
	 */
	_CK_9_ReadMsgTable = (void **) CmiAlloc((TotalReadMsgs + 1) *
					       sizeof(void *));
	if (TotalReadMsgs > 0)
		CkMemError(_CK_9_ReadMsgTable);

	ROCopyFromBufferTable = (FUNCTION_PTR *) CmiAlloc((TotalModules + 1) *
						      sizeof(FUNCTION_PTR));
	ROCopyToBufferTable = (FUNCTION_PTR *) CmiAlloc((TotalModules + 1) *
						       sizeof(FUNCTION_PTR));
	if (TotalModules > 0)
	{
		CkMemError(ROCopyFromBufferTable);
		CkMemError(ROCopyToBufferTable);
	}

	EpTable = (FUNCTION_PTR *) CmiAlloc((TotalEps + 1) *
					   sizeof(FUNCTION_PTR));
	EpIsImplicitTable = (int *) CmiAlloc((TotalEps + 1) * sizeof(int));
	EpLanguageTable = (int *) CmiAlloc((TotalEps + 1) * sizeof(int));
	for (i = 0; i < TotalEps + 1; i++)
		EpIsImplicitTable[i] = 0;
	EpNameTable = (char **) CmiAlloc((TotalEps + 1) * sizeof(char *));
	EpChareTable = (int *) CmiAlloc((TotalEps + 1) * sizeof(int));
	EpToMsgTable = (int *) CmiAlloc((TotalEps + 1) * sizeof(int));
	EpChareTypeTable = (int *) CmiAlloc((TotalEps + 1) * sizeof(int));

	if (TotalEps > 0)
	{
		CkMemError(EpTable);
		CkMemError(EpIsImplicitTable);
		CkMemError(EpLanguageTable);
		CkMemError(EpNameTable);
		CkMemError(EpChareTable);
		CkMemError(EpToMsgTable);
		CkMemError(EpChareTypeTable);
	}

	/*
	 * set all the system BOC EPs to be CHARM bocs because they dont get
	 * registered in the normal way
	 */
	for (i = 0; i < chareEpsCount; i++)
		EpLanguageTable[i] = -1;


	_CK_9_GlobalFunctionTable = (FUNCTION_PTR *) CmiAlloc((TotalFns + 1) *
						      sizeof(FUNCTION_PTR));
	if (TotalFns > 0)
		CkMemError(_CK_9_GlobalFunctionTable);


	MsgToStructTable = (MSG_STRUCT *) CmiAlloc((TotalMsgs + 1) *
						  sizeof(MSG_STRUCT));
	if (TotalMsgs > 0)
		CkMemError(MsgToStructTable);


	ChareSizesTable = (int *) CmiAlloc((TotalChares + 1) * sizeof(int));
	ChareNamesTable = (char **) CmiAlloc(TotalChares * sizeof(char *));
	ChareFnTable = (FUNCTION_PTR *) CmiAlloc((TotalChares + 1) *
						sizeof(FUNCTION_PTR));
	if (TotalChares > 0)
	{
		CkMemError(ChareSizesTable);
		CkMemError(ChareNamesTable);
		CkMemError(ChareFnTable);
	}

	PseudoTable = (PSEUDO_STRUCT *) CmiAlloc((TotalPseudos + 1) *
						sizeof(PSEUDO_STRUCT));
	if (TotalPseudos > 0)
		CkMemError(PseudoTable);



	/** end of table allocation **/

	/* Register the NullFunction to detect uninitialized modules */
	registerMsg("NULLMSG",_CkNullFunc,_CkNullFunc,_CkNullFunc,0) ;
	registerEp("NULLEP",_CkNullFunc,0,0,0) ;
	registerChare("NULLCHARE",0,_CkNullFunc) ;
	registerFunction(_CkNullFunc) ;
	registerMonotonic("NULLMONO",_CkNullFunc,_CkNullFunc,CHARM) ;
	registerTable("NULLTABLE",_CkNullFunc,_CkNullFunc) ;
	registerAccumulator("NULLACC",_CkNullFunc,_CkNullFunc,_CkNullFunc,CHARM) ;

	chareEpsCount += AddSysBocEps(EpTable);

	/*
	 * This is the top level call to all modules for initialization. It
	 * is generated at link time by charmc, in module_init_fn.c
	 */
	_CK_module_init_fn();

	if ( MainChareLanguage == -1 ) {
		CmiPrintf("[%d] ERROR: registerMainChare() not called : uninitialized module exists\n",CmiMyPe()) ;
	}

	/* Register the Charm handlers with Converse */
	HANDLE_INCOMING_MSG_Index = CmiRegisterHandler(HANDLE_INCOMING_MSG) ;
	CallProcessMsg_Index = CmiRegisterHandler(CallProcessMsg) ;



	/* set all the "Total" variables so that the rest of the modules work */
	TotalEps = chareEpsCount;
	TotalFns = fnCount;
	TotalMsgs = msgCount;
	TotalChares = chareCount;
	TotalModules = readCount;
	TotalReadMsgs = readMsgCount;
	TotalPseudos = pseudoCount;
}

/* Adding entry points for system branch office chares. */
AddSysBocEps()
{
	LdbAddSysBocEps();
	QDAddSysBocEps();
	VidAddSysBocEps();
	WOVAddSysBocEps();
	TblAddSysBocEps();
	AccAddSysBocEps();
	MonoAddSysBocEps();
	DynamicAddSysBocEps();
	StatAddSysBocEps();

	return (NumSysBocEps);	/* number of system boc-eps */
}


/* Broadcast the count of messages that are received during initialization. */
BroadcastCount()
{
	ENVELOPE       *env;
	void           *dummy_msg;
	dummy_msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(dummy_msg);
	env = ENVELOPE_UPTR(dummy_msg);
	SetEnv_destPE(env, ALL_NODES_EXCEPT_ME);
	SetEnv_category(env, USERcat);
	SetEnv_msgType(env, InitCountMsg);
	SetEnv_destPeFixed(env, 1);

	SetEnv_count(env, currentBocNum - NumSysBoc + 2 + NumReadMsg);

	CkCheck_and_Broadcast(env, 0);
	/* CkFreeMsg(dummy_msg);  commented on Jun 23 */
}

int 
EmptyBocInitMsgs()
{
	return (BocInitQueueHead->length == 0);
}


void           *
BocInitQueueCreate()
{
	BOCINIT_QUEUE  *queue;
	queue = (BOCINIT_QUEUE *) CmiAlloc(sizeof(BOCINIT_QUEUE));
	queue->block = (void **) CmiAlloc(sizeof(void *) * BLK_LEN);
	queue->block_len = BLK_LEN;
	queue->first = queue->block_len;
	queue->length = 0;
	return (void *) queue;
}


EnQueueBocInitMsgs(envelope)
ENVELOPE       *envelope;
{
	int             num = GetEnv_boc_num(envelope);
	int i ;

	if (num > BocInitQueueHead->block_len)
	{
		void          **blk = BocInitQueueHead->block;
		int             last;
		BocInitQueueHead->block = (void **) CmiAlloc(sizeof(void *) * (num + BLK_LEN));
		last = BocInitQueueHead->first + BocInitQueueHead->length;
		for (i = BocInitQueueHead->first; i < last; i++)
			BocInitQueueHead->block[i] = blk[i];
		BocInitQueueHead->block[num] = envelope;
		BocInitQueueHead->length++;
		CmiFree(blk);
	}
	else
	{
		BocInitQueueHead->block[num] = envelope;
		BocInitQueueHead->length++;
		if (BocInitQueueHead->first > num)
			BocInitQueueHead->first = num;
	}
}


ENVELOPE *DeQueueBocInitMsgs()
{
	ENVELOPE      *envelope;
	if (BocInitQueueHead->length)
	{
		envelope = BocInitQueueHead->block[BocInitQueueHead->first++];
		BocInitQueueHead->length--;
	/*	if (!BocInitQueueHead->length)
			BocInitQueueHead->first = BocInitQueueHead->block_len;
	*/
		return envelope ;
	}
	else
		return NULL ;
}

SysBocInit()
{
	LdbBocInit();
	QDBocInit();
	VidBocInit();
	TblBocInit();
	WOVBocInit();
	DynamicBocInit();
	StatisticBocInit();
}



hostep_error(msg, mydata)
void           *msg, *mydata;
{
	CmiPrintf("****error*** main chare ep called on node %d.\n",
		 CmiMyPe());
}
