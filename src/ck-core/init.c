#include "charm.h"

#include "trace.h"


/* these two variables are set in registerMainChare() */
CsvDeclare(int, _CK_MainChareIndex);
CsvDeclare(int, _CK_MainEpIndex);

CsvDeclare(int, ReadBuffSize); /* this is set in registerReadOnly() */
CsvStaticDeclare(void **, _CK_9_ReadMsgTable);	/* was previously global */

CpvDeclare(char*, ReadBufIndex);
CpvDeclare(char*, ReadFromBuffer);

#define BLK_LEN 512
CpvStaticDeclare(BOCINIT_QUEUE*, BocInitQueueHead);
CpvStaticDeclare(ENVELOPE*, readvarmsg);

static int EmptyBocInitMsgs();
static void *BocInitQueueCreate();
static void BocInitQueueDestroy();
static void EnQueueBocInitMsgs();
static ENVELOPE *DeQueueBocInitMsgs() ;

/* these store argc, argv for use by CharmInit */
CpvStaticDeclare(int,   userArgc);
CpvStaticDeclare(char**,  userArgv);

CpvStaticDeclare(int, NumChildrenDoneWithStartCharm);
CpvStaticDeclare(FUNCTION_PTR, UserStartCharmDoneHandler);

/* all these counts are incremented in register.c */
CpvExtern(int,      fnCount);
CpvExtern(int,      msgCount);
CpvExtern(int,      chareCount);
CpvExtern(int,      chareEpsCount);	/* count of chare AND boc eps */
CpvExtern(int,      pseudoCount);
CpvExtern(int,      readMsgCount);
CpvExtern(int,      readCount);	/* count of read-only functions. Should be
				 * same as total number of modules */
CsvStaticDeclare(int, sharedReadCount); /* it holds the value of readCount, */
                                        /* and it is  shared within node    */

extern void CPlus_ProcessBocInitMsg();	/* in cplus_node_init.c */
extern void CPlus_CallCharmInit();	/* in cplus_node_init.c */
extern void CPlus_SetMainChareID();	/* in cplus_node_init.c */
extern CHARE_BLOCK *CreateChareBlock();


extern void BUFFER_INCOMING_MSG() ;
extern void HANDLE_INCOMING_MSG() ;
extern void HANDLE_INIT_MSG();

void SysPeriodicCheckInit(void);
void CharmRegisterHandlers();
void InitializeEPTables();
void AddSysBocEps(void);
void BroadcastCount(void);
void SysBocInit(void);

void initModuleInit()
{
    CpvInitialize(char*, ReadBufIndex);
    CpvInitialize(char*, ReadFromBuffer);
    CpvInitialize(BOCINIT_QUEUE*, BocInitQueueHead);
    CpvInitialize(int, userArgc);
    CpvInitialize(char**, userArgv);
    CpvInitialize(ENVELOPE*, readvarmsg);
    CpvInitialize(int, NumChildrenDoneWithStartCharm);
    CpvInitialize(FUNCTION_PTR, UserStartCharmDoneHandler);

    CpvAccess(NumChildrenDoneWithStartCharm) = 0;
    CpvAccess(BocInitQueueHead) = NULL;
    if (CmiMyRank() == 0) CsvAccess(ReadBuffSize) = 0;
}


/*Added By Milind 05/02/95 */

void _CkNullFunc()
{
	CmiPrintf("[%d] In Null Function: Module Uninitialized\n", CmiMyPe());
}


void InitializeCharm(argc, argv)
int argc;
char **argv;
{
/* these lines were in user_main */
  if (CmiMyRank() != 0) CmiNodeBarrier();

  bocModuleInit();
  ckModuleInit();
  condsendModuleInit();
  globalsModuleInit();
  initModuleInit();
  mainModuleInit();
  quiesModuleInit();
  registerModuleInit();
  statModuleInit();
  tblModuleInit(); 
  futuresModuleInit();

  if (CmiMyRank() == 0) CmiNodeBarrier();
}


static void EndInitPhase()
{
  int i;
  void  *buffMsg;
  ENVELOPE *bocMsg;

  CpvAccess(CkInitPhase) = 0;

  
  /* call all the CopyFromBuffer functions for ReadOnly variables.
   * _CK_9_ReadMsgTable is passed as an arg because it is no longer
   * global */
  
  if (CmiMyPe() != 0) {
    if (CmiMyRank() == 0) {
      for (i = 0; i < CsvAccess(sharedReadCount); i++)
	(CsvAccess(ROCopyFromBufferTable)[i]) (CsvAccess(_CK_9_ReadMsgTable));
    }
    CmiFree(CpvAccess(readvarmsg));
  
    while ( (bocMsg=DeQueueBocInitMsgs()) != NULL )
      ProcessBocInitMsg(bocMsg);
    BocInitQueueDestroy();
  }
  
  /* Charm is finally done Initializing */

  if (CpvAccess(UserStartCharmDoneHandler))
    CpvAccess(UserStartCharmDoneHandler)();
  
  /* process all the non-init messages arrived during the 
     initialization */
  while (!FIFO_Empty(CpvAccess(CkBuffQueue))  ) {
    FIFO_DeQueue(CpvAccess(CkBuffQueue), &buffMsg);
    CmiSetHandler(buffMsg, CpvAccess(HANDLE_INCOMING_MSG_Index));
    CmiSyncSendAndFree(CmiMyPe(), GetEnv_TotalSize(buffMsg), buffMsg);
  }
}


static void PropagateInitBarrier()
{
  if (CpvAccess(CkInitPhase) == 0) return;

  if (CpvAccess(CkCountArrived) && CpvAccess(CkInitCount)==0) {
    /* initialization phase is done, set the flag to 0 */
    if (CpvAccess(NumChildrenDoneWithStartCharm)==
        CmiNumSpanTreeChildren(CmiMyPe())) {
      int parent = CmiSpanTreeParent(CmiMyPe());
      void *msg = (void *)CkAllocMsg(0);
      ENVELOPE *henv = ENVELOPE_UPTR(msg);
      EndInitPhase();
      if (parent == -1) {
	SetEnv_msgType(henv, InitBarrierPhase2);
	CmiSetHandler(henv, CsvAccess(HANDLE_INIT_MSG_Index));
	CmiSyncBroadcastAllAndFree(GetEnv_TotalSize(henv), henv); 
      } else {
	SetEnv_msgType(henv, InitBarrierPhase1);
        CmiSetHandler(henv, CsvAccess(HANDLE_INIT_MSG_Index));
        CmiSyncSendAndFree(parent,GetEnv_TotalSize(henv), henv);
      }
    }
  }
}


void StartCharm(argc, argv, donehandler)
int argc;
char **argv;
FUNCTION_PTR donehandler;
{
	int             i;
	char           *ReadBufMsg;

        CpvAccess(UserStartCharmDoneHandler) = donehandler;
        CpvAccess(userArgc) = ParseCommandOptions(argc, argv);
        CpvAccess(userArgv) = argv;

	InitializeMessageMacros();

	/* CmiSpanTreeInit();  already done in CmiInitMc  -- Sanjeev 3/5/96 */

        /* OtherQsInit(); this was combined with CsdInitialize */
        StatInit();
        InitializeDynamicBocMsgList();
        InitializeBocDataTable();
        InitializeBocIDMessageCountTable();
        
        CharmRegisterHandlers();

       /* set the main message handler to buffering handler */
       /* after initialization phase, it will be assigned to regular handler */
       CpvAccess(HANDLE_INCOMING_MSG_Index)
             = CsvAccess(BUFFER_INCOMING_MSG_Index);

        if (CmiMyRank() == 0) InitializeEPTables();
        CmiNodeBarrier();          
  
 
        /* log_init(); Moved to convcore.c */
        if(CpvAccess(traceOn))
          trace_begin_computation();
        SysBocInit();
        CpvAccess(msgs_created) = CpvAccess(msgs_processed) = 0;

       /* create the queue for non-init messages arrived
          during initialization */
        CpvAccess(CkBuffQueue) = (void *) FIFO_Create();


	if (CmiMyPe() == 0)
	{
	        CpvAccess(CkCountArrived)=1;

		CpvAccess(MsgCount) = 0; 
                                 /* count # of messages being sent to each
				 * node. assume an equal number gets sent to
				 * every one. if there is a difference, have
				 * to modify this somewhat */
		CpvAccess(InsideDataInit) = 1;

		futuresCreateBOC();

		CpvAccess(InsideDataInit) = 0;

                if(CpvAccess(traceOn))
		  trace_begin_charminit();
		 
		CpvAccess(MainDataSize) = CsvAccess(ChareSizesTable)
					      [CsvAccess(_CK_MainChareIndex)];
		CpvAccess(mainChareBlock) =
                    CpvAccess(currentChareBlock) = 
			CreateChareBlock(CpvAccess(MainDataSize),
					 CHAREKIND_CHARE, 
					 CpvAccess(nodecharesProcessed)++);

		if (CsvAccess(MainChareLanguage) == CHARMPLUSPLUS) 
			CPlus_SetMainChareID() ;  /* set mainhandle */


		/* Calling CharmInit entry point */
		CpvAccess(NumReadMsg) = 0;
		CpvAccess(InsideDataInit) = 1;

		(CsvAccess(EpInfoTable)[CsvAccess(_CK_MainEpIndex)].function)
		  		(NULL, CpvAccess(currentChareBlock)->chareptr,
				   CpvAccess(userArgc), CpvAccess(userArgv));
		
		CpvAccess(InsideDataInit) = 0;
                if(CpvAccess(traceOn))
		  trace_end_charminit();

		/* create the buffer for the read only variables */
		ReadBufMsg = (char *) CkAllocMsg(CsvAccess(ReadBuffSize));
		CpvAccess(ReadBufIndex) = ReadBufMsg;
		if (CsvAccess(ReadBuffSize) > 0)
			CkMemError(ReadBufMsg);

		/*
		 * in Charm++ the CopyToBuffer fns also send out the
		 * ReadonlyMsgs by calling ReadMsgInit()
		 */
		for (i = 0; i < CsvAccess(sharedReadCount); i++)
			(CsvAccess(ROCopyToBufferTable)[i]) ();

		/*
		 * we are sending the id of the main chare along with the
		 * read only message. in future versions, we might eliminate
		 * this because the functionality can be expressed using
		 * readonly variables and MyChareID inside the main chare
		 */

		BroadcastReadBuffer(ReadBufMsg, CsvAccess(ReadBuffSize), CpvAccess(mainChareBlock));

		/*
		 * send a message with the count of initial messages sent so
		 * far, to all nodes; includes messages for read-buffer and
		 * bocs
		 */
		BroadcastCount();
		
		PropagateInitBarrier();
	}
	else
	{
       		/* This is so that all PEs have consistent magic numbers for 
	   	   BOCs. 0 is the magic # of main chare on proc 0, 
	   	   all other procs have magic numbers from 1 */
		CpvAccess(nodecharesProcessed) = 1 ;

                /* create the boc init message queue */
                CpvAccess(BocInitQueueHead) = (BOCINIT_QUEUE *) BocInitQueueCreate();
	}

	SysPeriodicCheckInit();
}



/*
 * This is the handler for initialization messages 
 * Start Boc's by allocating and filling the NodeBocTbl. 
 * When (a) the  "count" message is received and (b) "count" number of initial
 * messages are received, switch to the regular phase 
 */

void HANDLE_INIT_MSG(env)
ENVELOPE *env;
{
  int          i;
  int          id;
  int          type;
  void         *usrMsg;
  
  CmiGrabBuffer((void **)&env);
  if ((GetEnv_msgType(env) == BocInitMsg) ||
      (GetEnv_msgType(env) == ReadMsgMsg))
    UNPACK(env);
  usrMsg = USER_MSG_PTR(env);
  /* Have a valid message now. */
  type = GetEnv_msgType(env);
  
  switch (type)
    {
    case BocInitMsg:
      EnQueueBocInitMsgs(env);
      CpvAccess(CkInitCount)++;
      break;
      
    case InitCountMsg:
      CpvAccess(CkCountArrived) = 1;
      CpvAccess(CkInitCount) -= GetEnv_count(env);
      CmiFree(env);
      break;
      
    case ReadMsgMsg:
      id = GetEnv_other_id(env);
      CsvAccess(_CK_9_ReadMsgTable)[id] = (void *) usrMsg;
      CpvAccess(CkInitCount)++; 
      break;
      
    case ReadVarMsg:
      CpvAccess(ReadFromBuffer) = usrMsg;
      CpvAccess(CkInitCount)++; 
      
      /* get the information about the main chare */
      CpvAccess(mainChareBlock) = (struct chare_block *)
	GetEnv_chareBlockPtr(env);
      CpvAccess(mainChare_magic_number) =
	GetEnv_chare_magic_number(env);
      if (CsvAccess(MainChareLanguage) == CHARMPLUSPLUS)
	CPlus_SetMainChareID();
      CpvAccess(readvarmsg) = env;
      break;
      
    case InitBarrierPhase1:
      CpvAccess(NumChildrenDoneWithStartCharm)++;
      CmiFree(env);
      break;
      
    case InitBarrierPhase2:
      /* set the main handler to the unbuffering one */
      CpvAccess(HANDLE_INCOMING_MSG_Index) = 
          CsvAccess(MAIN_HANDLE_INCOMING_MSG_Index);
      CmiFree(env);
      break;
      
    default:
      CmiPrintf("** ERROR ** Unknown message type in initialization phase%d\n",type);
      
    }
  PropagateInitBarrier();
}





ProcessBocInitMsg(envelope)
ENVELOPE       *envelope;
{
  CHARE_BLOCK    *bocBlock;
  void           *usrMsg = USER_MSG_PTR(envelope);
  int             current_ep = GetEnv_EP(envelope);
  EP_STRUCT      *current_epinfo = CsvAccess(EpInfoTable) + current_ep;
  int             current_bocnum = GetEnv_boc_num(envelope);
  int             current_msgType = GetEnv_msgType(envelope);
  int             current_chare = current_epinfo->chareindex;
  int             current_magic = CpvAccess(nodecharesProcessed)++;
  CHARE_BLOCK    *prev_chare_block;

  CpvAccess(nodebocInitProcessed)++ ;

  prev_chare_block = CpvAccess(currentChareBlock);
  CpvAccess(currentChareBlock) = bocBlock = 
		CreateChareBlock(CsvAccess(ChareSizesTable)[current_chare], 
					CHAREKIND_BOCNODE, current_magic);
  bocBlock->x.boc_num = current_bocnum;

  SetBocBlockPtr(current_bocnum, bocBlock);
  if(CpvAccess(traceOn))
    trace_begin_execute(envelope);
  (current_epinfo->function)(usrMsg, bocBlock->chareptr);
  if(CpvAccess(traceOn))
    trace_end_execute(current_magic, current_msgType, current_ep);
  CpvAccess(currentChareBlock) = prev_chare_block;

  /* for dynamic BOC creation, used in node_main.c */
  return current_bocnum ;
}


/* this call can only be made after the clock has been initialized */

void SysPeriodicCheckInit(void)
{
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
  int             NumPes;
  if (argc < 1)
    {
      CmiPrintf("Too few arguments. Usage> host_prog node_prog [...]\n");
      exit(1);
    }

  end = argc;
  if (CmiMyPe() == 0)
    mainflag = 1;
  
  CpvAccess(QueueingDefault) = CK_QUEUEING_FIFO;
  for (i = 1; i < end; i++) {
    foundSysOpt = 0;
    if (strcmp(argv[i], "+cs") == 0) {
      CpvAccess(PrintChareStat) = 1;
      /*
       * if (mainflag) CmiPrintf("Chare Statistics Turned
       * On\n");
       */
      foundSysOpt = 1;
    } else if (strcmp(argv[i], "+ss") == 0) {
      CpvAccess(PrintSummaryStat) = 1;
      /*
       * if(mainflag)CmiPrintf("Summary Statistics Turned
       * On\n");
       */
      foundSysOpt = 1;
    } else if (strcmp(argv[i],"+fifo")==0) {
      CpvAccess(QueueingDefault) = CK_QUEUEING_FIFO;
      foundSysOpt = 1;
    } else if (strcmp(argv[i],"+lifo")==0) {
      CpvAccess(QueueingDefault) = CK_QUEUEING_LIFO;
      foundSysOpt = 1;
    } else if (strcmp(argv[i],"+ififo")==0) {
      CpvAccess(QueueingDefault) = CK_QUEUEING_IFIFO;
      foundSysOpt = 1;
    } else if (strcmp(argv[i],"+ilifo")==0) {
      CpvAccess(QueueingDefault) = CK_QUEUEING_ILIFO;
      foundSysOpt = 1;
    } else if (strcmp(argv[i],"+bfifo")==0) {
      CpvAccess(QueueingDefault) = CK_QUEUEING_BFIFO;
      foundSysOpt = 1;
    } else if (strcmp(argv[i],"+blifo")==0) {
      CpvAccess(QueueingDefault) = CK_QUEUEING_BLIFO;
      foundSysOpt = 1;
    } else if (strcmp(argv[i], "+p") == 0 && i + 1 < argc) {
      sscanf(argv[i + 1], "%d", &NumPes);
      foundSysOpt = 2;
    } else if (sscanf(argv[i], "+p%d", &NumPes) == 1) {
      foundSysOpt = 1;
    }
    if (foundSysOpt) {
      /* if system option, remove it. */
      numSysOpts += foundSysOpt;
      end -= foundSysOpt;
      for (j = i; j < argc - foundSysOpt; j++) {
	argv[j] = argv[j + foundSysOpt];
      }
      /* reset i because we shuffled everything down one */
      i--;
    }
  }
  return (argc - numSysOpts);
}



#define TABLE_SIZE 256

void CharmRegisterHandlers()
{
  /* Register the Charm handlers with Converse */
  CsvAccess(BUFFER_INCOMING_MSG_Index)
    = CmiRegisterHandler(BUFFER_INCOMING_MSG) ;
  CsvAccess(MAIN_HANDLE_INCOMING_MSG_Index)
    = CmiRegisterHandler(HANDLE_INCOMING_MSG) ;
  CsvAccess(HANDLE_INIT_MSG_Index)
    = CmiRegisterHandler(HANDLE_INIT_MSG);
}

void InitializeEPTables(void)
{
  int             i;
  int             TotalFns;
  int             TotalMsgs;
  int             TotalChares;
  int             TotalModules;
  int             TotalReadMsgs;
  int             TotalPseudos;
  int             TotalEvents;
  EP_STRUCT      *epinfo;
  
  /*
   * TotalEps 	=  _CK_5mainChareEPCount(); TotalFns	=
   * _CK_5mainFunctionCount(); TotalMsgs	=  _CK_5mainMessageCount();
   * TotalChares 	=  _CK_5mainChareCount(); TotalBocEps 	=
   * NumSysBocEps + _CK_5mainBranchEPCount();
   */
  CsvAccess(TotalEps) = TABLE_SIZE;
  TotalFns = TABLE_SIZE;
  TotalMsgs = TABLE_SIZE;
  TotalChares = TABLE_SIZE;
  TotalModules = TABLE_SIZE;
  TotalReadMsgs = TABLE_SIZE;
  TotalPseudos = TABLE_SIZE;
  TotalEvents = TABLE_SIZE;
  
  /*
   * this table is used to store all ReadOnly Messages on processors
   * other than proc 0. After they are received, they are put in the
   * actual variables in the user program in the ...CopyFromBuffer
   * functions
   */
  CsvAccess(_CK_9_ReadMsgTable) = (void **) 
    CmiSvAlloc((TotalReadMsgs + 1) * sizeof(void *));
  if (TotalReadMsgs > 0)
    CkMemError(CsvAccess(_CK_9_ReadMsgTable));
  
  CsvAccess(ROCopyFromBufferTable) = (FUNCTION_PTR *) 
    CmiSvAlloc((TotalModules + 1) * sizeof(FUNCTION_PTR));
  
  CsvAccess(ROCopyToBufferTable) = (FUNCTION_PTR *) 
    CmiSvAlloc((TotalModules + 1) * sizeof(FUNCTION_PTR));
  
  if (TotalModules > 0)
    {
      CkMemError(CsvAccess(ROCopyFromBufferTable));
      CkMemError(CsvAccess(ROCopyToBufferTable));
    }
  
  epinfo=(EP_STRUCT*)CmiSvAlloc((CsvAccess(TotalEps)+1)*sizeof(EP_STRUCT));
  CsvAccess(EpInfoTable)=epinfo;
  if (CsvAccess(TotalEps) > 0) {
    CkMemError(epinfo);
    memset((char *)epinfo, 0, (CsvAccess(TotalEps)+1)*sizeof(EP_STRUCT));
    for (i = 0; i < CpvAccess(chareEpsCount); i++)
      epinfo[i].language = -1;
  }
  
  CsvAccess(_CK_9_GlobalFunctionTable) = (FUNCTION_PTR *) 
    CmiSvAlloc((TotalFns + 1) * sizeof(FUNCTION_PTR));
  
  if (TotalFns > 0)
    CkMemError(CsvAccess(_CK_9_GlobalFunctionTable));
  
  
  CsvAccess(MsgToStructTable) = (MSG_STRUCT *) 
    CmiSvAlloc((TotalMsgs + 1) * sizeof(MSG_STRUCT));
  
  if (TotalMsgs > 0)
    CkMemError(CsvAccess(MsgToStructTable));
  
  
  CsvAccess(ChareSizesTable) = (int *) 
    CmiSvAlloc((TotalChares + 1) * sizeof(int));
  
  CsvAccess(ChareNamesTable) = (char **) CmiSvAlloc(TotalChares * sizeof(char *));
  
  if (TotalChares > 0)
    {
      CkMemError(CsvAccess(ChareSizesTable));
      CkMemError(CsvAccess(ChareNamesTable));
    }
  
  CsvAccess(EventTable) = (char **) CmiSvAlloc(TotalEvents * sizeof(char *));

  CsvAccess(PseudoTable) = (PSEUDO_STRUCT *) 
    CmiSvAlloc((TotalPseudos + 1) * sizeof(PSEUDO_STRUCT));
  
  if (TotalPseudos > 0)
    CkMemError(CsvAccess(PseudoTable));
  
  
  
  /** end of table allocation **/
  
  /* Register the NullFunction to detect uninitialized modules */
  registerMsg("NULLMSG",_CkNullFunc,_CkNullFunc,_CkNullFunc,0) ;
  registerEp("NULLEP",_CkNullFunc,0,0,0) ;
  registerChare("NULLCHARE",0,_CkNullFunc) ;
  registerFunction(_CkNullFunc) ;
  registerMonotonic("NULLMONO",_CkNullFunc,_CkNullFunc,CHARM) ;
  registerTable("NULLTABLE",_CkNullFunc,_CkNullFunc) ;
  registerAccumulator("NULLACC",_CkNullFunc,_CkNullFunc,_CkNullFunc,CHARM) ;
  
  /* Register all the built-in BOC's */
  AddSysBocEps();
  CsvAccess(NumSysBocEps) = CpvAccess(chareEpsCount);

  /*
   * This is the top level call to all modules for initialization. It
   * is generated at link time by charmc, in module_init_fn.c
   */
  _CK_module_init_fn();
  
  if ( CsvAccess(MainChareLanguage) == -1 ) {
    CmiPrintf("[%d] ERROR: registerMainChare() not called : uninitialized module exists\n",CmiMyPe()) ;
  }
  


  
  /* set all the "Total" variables so that the rest of the modules work */
  CsvAccess(TotalEps) = CpvAccess(chareEpsCount);
  TotalFns = CpvAccess(fnCount);
  TotalMsgs = CpvAccess(msgCount);
  TotalChares = CpvAccess(chareCount);
  TotalModules = CpvAccess(readCount);
  TotalReadMsgs = CpvAccess(readMsgCount);
  TotalPseudos = CpvAccess(pseudoCount);
  
  CsvAccess(sharedReadCount) = CpvAccess(readCount);
}

/* Adding entry points for system branch office chares. */
void AddSysBocEps(void)
{
	QDAddSysBocEps();
	WOVAddSysBocEps();
	TblAddSysBocEps();
	AccAddSysBocEps();
	MonoAddSysBocEps();
	DynamicAddSysBocEps();
	StatAddSysBocEps();
}


/* Broadcast the count of messages that are received during initialization. */
void BroadcastCount(void)
{
	ENVELOPE       *env;
	void           *dummy_msg;
	dummy_msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(dummy_msg);
	env = ENVELOPE_UPTR(dummy_msg);
	SetEnv_msgType(env, InitCountMsg);

	SetEnv_count(env, CpvAccess(currentBocNum) - NumSysBoc + 2 + CpvAccess(NumReadMsg));
	CmiSetHandler(env,CsvAccess(HANDLE_INIT_MSG_Index));
	CmiSyncBroadcastAndFree(GetEnv_TotalSize(env),env);
}

static int 
EmptyBocInitMsgs()
{
	return (CpvAccess(BocInitQueueHead)->length == 0);
}


static void           *
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


static void BocInitQueueDestroy()
{
    if (CpvAccess(BocInitQueueHead))
    { 
       if (CpvAccess(BocInitQueueHead)->block)
          CmiFree(CpvAccess(BocInitQueueHead)->block);
       CmiFree(CpvAccess(BocInitQueueHead));
    } 
}


static void EnQueueBocInitMsgs(envelope)
ENVELOPE       *envelope;
{
	int             num = GetEnv_boc_num(envelope);
	int i ;

	if (num > CpvAccess(BocInitQueueHead)->block_len)
	{
		void          **blk = CpvAccess(BocInitQueueHead)->block;
		int             last;
		CpvAccess(BocInitQueueHead)->block = (void **) CmiAlloc(sizeof(void *) * (num + BLK_LEN));
		last = CpvAccess(BocInitQueueHead)->first + CpvAccess(BocInitQueueHead)->length;
		for (i = CpvAccess(BocInitQueueHead)->first; i < last; i++)
			CpvAccess(BocInitQueueHead)->block[i] = blk[i];
		CpvAccess(BocInitQueueHead)->block[num] = envelope;
		CpvAccess(BocInitQueueHead)->length++;
		CmiFree(blk);
	}
	else
	{
		CpvAccess(BocInitQueueHead)->block[num] = envelope;
		CpvAccess(BocInitQueueHead)->length++;
		if (CpvAccess(BocInitQueueHead)->first > num)
			CpvAccess(BocInitQueueHead)->first = num;
	}
}


static ENVELOPE *DeQueueBocInitMsgs()
{
	ENVELOPE      *envelope;
	if (CpvAccess(BocInitQueueHead)->length)
	{
		envelope = CpvAccess(BocInitQueueHead)->block[CpvAccess(BocInitQueueHead)->first++];
		CpvAccess(BocInitQueueHead)->length--;
	/*	if (!CpvAccess(BocInitQueueHead)->length)
			CpvAccess(BocInitQueueHead)->first = CpvAccess(BocInitQueueHead)->block_len;
	*/
		return envelope ;
	}
	else
		return NULL ;
}

void SysBocInit(void)
{
	QDBocInit();
	TblBocInit();
	WOVBocInit();
	DynamicBocInit();
	StatisticBocInit();
}
