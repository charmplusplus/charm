#include "charm.h"

#include "trace.h"

#define MAXBOC 15

typedef struct msg_element {
        int ref;
        void *msg;
        ChareNumType ep;
        struct msg_element *next;
} MSG_ELEMENT;

typedef struct bocdata_queue_element {
	ChareNumType bocNum;
	CHARE_BLOCK *dataptr;
	struct bocdata_queue_element *next;
} BOCDATA_QUEUE_ELEMENT;

typedef struct bocid_message_count {
	int count;
	ChareNumType bocnum;
	ChareIDType ReturnID;
	EntryPointType ReturnEP;
	struct bocid_message_count *next;
} BOCID_MESSAGE_COUNT;



typedef MSG_ELEMENT *MSG_ELEMENT_;
typedef BOCDATA_QUEUE_ELEMENT *BOCDATA_QUEUE_ELEMENT_[MAXBOC];
typedef BOCID_MESSAGE_COUNT *BOCID_MESSAGE_COUNT_[MAXBOC];

CpvStaticDeclare(int, number_dynamic_boc);
CpvStaticDeclare(MSG_ELEMENT_, DynamicBocMsgList); 
CpvStaticDeclare(BOCDATA_QUEUE_ELEMENT_, BocDataTable);
CpvStaticDeclare(BOCID_MESSAGE_COUNT_, BocIDMessageCountTable);

CHARE_BLOCK *CreateChareBlock();

void GeneralSendMsgBranch();


void bocModuleInit(void)
{
   CpvInitialize(int, number_dynamic_boc);
   CpvInitialize(MSG_ELEMENT_, DynamicBocMsgList);
   CpvInitialize(BOCDATA_QUEUE_ELEMENT_, BocDataTable);
   CpvInitialize(BOCID_MESSAGE_COUNT_, BocIDMessageCountTable);

   CpvAccess(number_dynamic_boc)=1;
}




void InitializeDynamicBocMsgList(void)
{
	CpvAccess(DynamicBocMsgList) = (MSG_ELEMENT *) NULL;
}

void InitializeBocDataTable(void)
{
	int i;

	for (i=0; i<MAXBOC; i++)
	    CpvAccess(BocDataTable)[i] = (BOCDATA_QUEUE_ELEMENT *) NULL;
}

void InitializeBocIDMessageCountTable(void)
{
	int i;

	for (i=0; i<MAXBOC; i++)
	    CpvAccess(BocIDMessageCountTable)[i] = (BOCID_MESSAGE_COUNT *) NULL;
}

void GetDynamicBocMsg(ref, msg, ep)
int ref;
void **msg;
ChareNumType *ep;
{
	MSG_ELEMENT * previous = NULL;
	MSG_ELEMENT * temp; 

        temp = CpvAccess(DynamicBocMsgList);
	
	while (temp != NULL)
	{
		if (temp->ref == ref)
		{
			*msg = temp->msg;
			*ep = temp->ep;

TRACE(CmiPrintf("[%d] GetDynamicBocMsg: ref=%d, ep=%d\n",
		CmiMyPe(), ref, temp->ref)); 
	
			if (previous == NULL)
				CpvAccess(DynamicBocMsgList) = temp->next; 
			else
				previous->next = temp->next;
			CmiFree(temp);
			return;
		}
		else
		{
			previous = temp;
			temp = temp->next;
		}
	}
	CmiPrintf("[%d] *** ERROR *** Could not locate return address for dynamic creation %d.\n", CmiMyPe(), ref);
}


CHARE_BLOCK *GetBocBlockPtr(bocNum)
ChareNumType bocNum;
{
	int index;
	BOCDATA_QUEUE_ELEMENT *element;


	index = bocNum % MAXBOC;
	element = CpvAccess(BocDataTable)[index];

TRACE(CmiPrintf("[%d] GetBocBlockPtr: bocNum=%d, index=%d, element=0x%x\n",
		 CmiMyPe(), bocNum, index, element));

	while (element != NULL)
	{
		if (element->bocNum == bocNum)
			return(element->dataptr);
		else
			element = element->next;
	}
	CmiPrintf("[%d] *** ERROR *** Unable to locate BOC %d data ptr.\n",
		CmiMyPe(),  bocNum);
  return (CHARE_BLOCK *) 0;
}


void * GetBocDataPtr(ChareNumType bocNum)
{
    return GetBocBlockPtr(bocNum)->chareptr;
}


BOCID_MESSAGE_COUNT * GetBocIDMessageCount(bocnum)
ChareNumType bocnum;
{
	int index;
	BOCID_MESSAGE_COUNT *element;

	index = bocnum % MAXBOC;
	element = CpvAccess(BocIDMessageCountTable)[index];
	while (element != NULL)
	{
		if (element->bocnum == bocnum)
			return(element);
		else 	
			element = element->next;
	}
	TRACE(CmiPrintf("[%d] *** ERROR *** Incorrect boc number %d in GetBocIDMessageCount\n", 
			CmiMyPe(), bocnum));
	return(NULL);
}

SetDynamicBocMsg(msg,ep)
void *msg;
ChareNumType ep;
{
	MSG_ELEMENT * new; 
	
	new = (MSG_ELEMENT *) CmiAlloc(sizeof(MSG_ELEMENT));
	new->ref = CpvAccess(number_dynamic_boc)++;
	new->msg = msg;
	new->ep = ep;
	new->next = CpvAccess(DynamicBocMsgList); 
	CpvAccess(DynamicBocMsgList) = new; 

TRACE(CmiPrintf("[%d] SetDynamicBocMsg: ref=%d, ep=%d\n",
	CmiMyPe(), new->ref, new->ep));

	return (new->ref);
}

void SetBocBlockPtr(bocNum, ptr)
ChareNumType bocNum;
CHARE_BLOCK *ptr;
{
	int index;
	BOCDATA_QUEUE_ELEMENT *new;
	BOCDATA_QUEUE_ELEMENT *element;


	index = bocNum % MAXBOC;
	new = (BOCDATA_QUEUE_ELEMENT *) CmiAlloc(sizeof(BOCDATA_QUEUE_ELEMENT));
	CkMemError(new);
	new->bocNum = bocNum;
	new->dataptr = ptr;
	element = CpvAccess(BocDataTable)[index];
	new->next = element;	
	CpvAccess(BocDataTable)[index] = new;

TRACE(CmiPrintf("[%d] SetBocBlockPtr: bocNum=%d, index=%d, new=0x%x\n",
		 CmiMyPe(), bocNum, index, new));
}


BOCID_MESSAGE_COUNT * SetBocIDMessageCount(bocnum, count, ReturnEP, ReturnID)
ChareNumType bocnum;
int count;
EntryPointType ReturnEP;
ChareIDType *ReturnID;
{
	int index;
	BOCID_MESSAGE_COUNT *new, *element;

	index = bocnum % MAXBOC;
	new = (BOCID_MESSAGE_COUNT *) CmiAlloc(sizeof(BOCID_MESSAGE_COUNT));
	CkMemError(new);
	new->bocnum = bocnum;
	new->count = count;
	new->ReturnEP = ReturnEP;
	if (ReturnID != NULL) 
		new->ReturnID = *ReturnID;
	element = CpvAccess(BocIDMessageCountTable)[index];
	new->next = element;
	CpvAccess(BocIDMessageCountTable)[index] = new;
	return(new);
}


ChareNumType CreateBoc(id, Entry, Msg, ReturnEP, ReturnID)
int id;
EntryNumType Entry;
void *Msg;
EntryNumType ReturnEP;
ChareIDType *ReturnID;
{
  ENVELOPE *env ;

  if (id!=CsvAccess(EpInfoTable)[Entry].chareindex)
    CmiPrintf("** ERROR ** Illegal combination of CHARENUM/EP in CreateBOC\n");

  env = (ENVELOPE *) ENVELOPE_UPTR(Msg);

  if (CpvAccess(InsideDataInit))
    /* static boc creation */
    {
      int executing_boc_num; 
      
      SetEnv_boc_num(env, ++CpvAccess(currentBocNum));
      SetEnv_EP(env, Entry);
      SetEnv_msgType(env, BocInitMsg);
      if(CpvAccess(traceOn))
        trace_creation(GetEnv_msgType(env), Entry, env);
      PACK(env);
      CmiSetHandler(env,CsvAccess(HANDLE_INIT_MSG_Index));
      CmiSyncBroadcast(GetEnv_TotalSize(env),env);
      UNPACK(env);
      
      /* env becomes the usrMsg, hence should not be freed by us */
      executing_boc_num = ProcessBocInitMsg(env);
      if (ReturnEP >= 0)
	{
	  ChareNumType *msg;
	  
	  msg = (ChareNumType *)
	    CkAllocMsg(sizeof(ChareNumType));
	  *msg = CpvAccess(currentBocNum);
	  SendMsg(ReturnEP, msg, ReturnID); 
	}
      return(executing_boc_num);
    }
  else
    /* dynamic boc creation */
    {
      DYNAMIC_BOC_REQUEST_MSG *msg;
      
      msg = (DYNAMIC_BOC_REQUEST_MSG *) 
	CkAllocMsg(sizeof(DYNAMIC_BOC_REQUEST_MSG));
      msg->source = CmiMyPe();
      msg->ep = ReturnEP;
      msg->id = *ReturnID;
      msg->ref = SetDynamicBocMsg(Msg, Entry);
      
      GeneralSendMsgBranch(CsvAccess(CkEp_DBOC_OtherCreateBoc), msg,
			   0, ImmBocMsg, DynamicBocNum);
    }
    return 0;
}

void OtherCreateBoc(msg, mydata)
DYNAMIC_BOC_REQUEST_MSG *msg;
char *mydata;
{
  DYNAMIC_BOC_NUM_MSG *tmsg;
  BOCID_MESSAGE_COUNT *element;
  
  element = SetBocIDMessageCount(++CpvAccess(currentBocNum),
				 CmiNumSpanTreeChildren(CmiMyPe()),
				 msg->ep, &(msg->id));
  tmsg = (DYNAMIC_BOC_NUM_MSG *) CkAllocMsg(sizeof(DYNAMIC_BOC_NUM_MSG)); 
  tmsg->boc = CpvAccess(currentBocNum);
  tmsg->ref = msg->ref;
  
  TRACE(CmiPrintf("[%d] OtherCreateBoc: boc=%d, ref=%d\n",
		  CmiMyPe(), tmsg->boc, tmsg->ref));
  
  GeneralSendMsgBranch(CsvAccess(CkEp_DBOC_InitiateDynamicBocBroadcast), tmsg,
		       msg->source, ImmBocMsg, DynamicBocNum);
}

MyBocNum(mydata)
void *mydata;
{
        CHARE_BLOCK *chare = ((CHARE_BLOCK *)mydata)-1;
        return chare->x.boc_num;
}

void MyBranchID(pChareID, mydata)
ChareIDType *pChareID;
void *mydata;
{
        CHARE_BLOCK *chare = ((CHARE_BLOCK *)mydata)-1;
        *pChareID = chare->selfID;
}

void GeneralSendMsgBranch(ep, msg, destPE, type, bocnum)
EntryPointType ep;
void *msg;
PeNumType destPE;
MsgTypes type;
ChareNumType bocnum;
{
  ENVELOPE *env;

  /* Charm++ translator puts type as -1 to avoid using the 
     BocMsg macro. */
  if ( type == -1 ) 
    type = BocMsg ;
  
  env  = ENVELOPE_UPTR(msg);
  
  SetEnv_msgType(env, type);
  SetEnv_boc_num(env, bocnum);
  SetEnv_EP(env, ep);
  
  TRACE(CmiPrintf("[%d] GeneralSend: type=%d, msgType=%d\n",
		  CmiMyPe(), type, GetEnv_msgType(env)));
  
  /* if (bocnum >= NumSysBoc) */
  CpvAccess(nodebocMsgsCreated)++;
  
  if(CpvAccess(traceOn))
    trace_creation(GetEnv_msgType(env), ep, env);
  
  CmiSetHandler(env, CpvAccess(HANDLE_INCOMING_MSG_Index));
  CldEnqueue(destPE, env, CpvAccess(CkInfo_Index));
  if((type!=QdBocMsg)&&(type!=QdBroadcastBocMsg)&&(type!=LdbMsg))
    QDCountThisCreation(1);
}


void GeneralBroadcastMsgBranch(ep, msg, type, bocnum)
EntryPointType ep;
void *msg;
MsgTypes type;
ChareNumType bocnum;
{
  ENVELOPE *env;
  
  /* Charm++ translator puts type as -1 to avoid using the 
     BroadcastBocMsg macro. */
  if ( type == -1 ) 
    type = BroadcastBocMsg ;
  
  env = ENVELOPE_UPTR(msg);
  
  SetEnv_msgType(env, type);
  SetEnv_boc_num(env, bocnum);
  SetEnv_EP(env, ep);
  
  TRACE(CmiPrintf("[%d] GeneralBroadcast: type=%d, msgType=%d\n",
		  CmiMyPe(), type, GetEnv_msgType(env)));
  
  /* if (bocnum >= NumSysBoc) */
  CpvAccess(nodebocMsgsCreated)+=CmiNumPes();
  
  if(CpvAccess(traceOn))
    trace_creation(GetEnv_msgType(env), ep, env);
  CmiSetHandler(env,CpvAccess(HANDLE_INCOMING_MSG_Index));
  CldEnqueue(CLD_BROADCAST_ALL, env, CpvAccess(CkInfo_Index));
  if((type!=QdBocMsg)&&(type!=QdBroadcastBocMsg)&&(type!=LdbMsg))
    QDCountThisCreation(CmiNumPes());
}


void RegisterDynamicBocInitMsg(bocnumptr, mydata)
ChareNumType *bocnumptr;
void *mydata;
{
	ChareNumType *msg;
	int mype = CmiMyPe();
	BOCID_MESSAGE_COUNT  * bocdata = GetBocIDMessageCount(*bocnumptr);

TRACE(CmiPrintf("[%d] RegisterDynamicBoc: bocnum=%d, bocdata=0x%x\n",
		 CmiMyPe(), *bocnumptr, bocdata));
	if (bocdata == NULL)
		bocdata = SetBocIDMessageCount(*bocnumptr,
				CmiNumSpanTreeChildren(mype), -1, NULL);
	bocdata->count--;

	if (bocdata->count < 0)
	{
		msg = (ChareNumType *) CkAllocMsg(sizeof(ChareNumType));
		*msg = *bocnumptr;

		if (mype == 0)
		{
			if (bocdata->ReturnEP >= 0)
				SendMsg(bocdata->ReturnEP, msg,
					&bocdata->ReturnID);
		}
		else
			GeneralSendMsgBranch(CsvAccess(CkEp_DBOC_RegisterDynamicBocInitMsg), (void *) msg,
                           CmiSpanTreeParent(mype), ImmBocMsg, DynamicBocNum);
	}
}


void InitiateDynamicBocBroadcast(msg, mydata)
DYNAMIC_BOC_NUM_MSG *msg;
char *mydata;
{
  int dataSize;
  void *tmsg;
  ENVELOPE * env;
  ChareNumType ep;
  
  GetDynamicBocMsg(msg->ref, &tmsg, &ep); 
  
  TRACE(CmiPrintf("[%d] InitiateDynamicBocBroadcast: ref=%d, boc=%d, ep=%d\n",
		  CmiMyPe(), msg->ref, msg->boc, ep));
  
  env = (ENVELOPE *) ENVELOPE_UPTR(tmsg);
  SetEnv_boc_num(env, msg->boc);
  SetEnv_EP(env, ep);
  SetEnv_msgType(env, DynamicBocInitMsg);
  
  if(CpvAccess(traceOn))
    trace_creation(GetEnv_msgType(env), ep, env);
  CmiSetHandler(env,CpvAccess(HANDLE_INCOMING_MSG_Index));
  CldEnqueue(CLD_BROADCAST_ALL, env, CpvAccess(CkInfo_Index));
  
  QDCountThisCreation(CmiNumPes());
}

void DynamicBocInit(void)
{
    	CHARE_BLOCK *bocBlock;

	/* Create a dummy block */
    	bocBlock = CreateChareBlock(sizeof(int), CHAREKIND_BOCNODE, 0);
        bocBlock->x.boc_num = DynamicBocNum;
    	SetBocBlockPtr(DynamicBocNum, bocBlock);
}


void DynamicAddSysBocEps(void)
{
  CsvAccess(CkEp_DBOC_RegisterDynamicBocInitMsg) =
    registerBocEp("CkEp_DBOC_RegisterDynamicBocInitMsg",
		  RegisterDynamicBocInitMsg,
		  CHARM, 0, 0);
  CsvAccess(CkEp_DBOC_OtherCreateBoc) =
    registerBocEp("CkEp_DBOC_OtherCreateBoc",
		  OtherCreateBoc,
		  CHARM, 0, 0);
  CsvAccess(CkEp_DBOC_InitiateDynamicBocBroadcast) =
    registerBocEp("CkEp_DBOC_InitiateDynamicBocBroadcast",
		  InitiateDynamicBocBroadcast,
		  CHARM, 0, 0);
}


