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
 * Revision 2.15  1997-12-03 21:36:08  rbrunner
 * I fixed a bug with nested BOC creation.  The BOC number returned would
 * be incorrect for BOC X if the constructor for BOC X creates a second
 * BOC.
 *
 * Revision 2.14  1997/10/29 23:52:43  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.13  1997/07/18 21:21:02  milind
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
 * Revision 2.12  1995/11/07 17:53:45  sanjeev
 * fixed bugs in statistics collection
 *
 * Revision 2.11  1995/11/02  20:10:44  sanjeev
 * GeneralSendMsgBranch sets type to BocMsg if it is -1, used by Charm++
 *
 * Revision 2.10  1995/10/27  21:31:25  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.9  1995/09/07  05:24:58  gursoy
 * necessary changes done related to CharmInitLoop--> handler fuction
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
 * Revision 2.4  1995/07/22  23:44:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/07/12  16:28:45  jyelon
 * *** empty log message ***
 *
 * Revision 2.2  1995/07/05  19:38:31  narain
 * No LdbFillBlock and StripMsg while InsideDataInit
 *
 * Revision 2.1  1995/06/08  17:09:41  gursoy
 * Cpv macro changes done
 *
 * Revision 1.8  1995/05/09  20:04:52  milind
 * Corrected the SP1 fboc bug.
 *
 * Revision 1.7  1995/04/23  20:52:27  sanjeev
 * Removed Core....
 *
 * Revision 1.6  1995/04/13  20:52:44  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.5  1995/03/25  18:24:18  sanjeev
 * ,
 *
 * Revision 1.4  1995/03/24  16:41:50  sanjeev
 * *** empty log message ***
 *
 * Revision 1.3  1995/03/17  23:36:38  sanjeev
 * changes for better message format
 *
 * Revision 1.2  1994/12/01  23:55:42  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:38:52  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "chare.h"
#include "globals.h"
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


void * GetBocDataPtr(bocNum)
ChareNumType bocNum;
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
      trace_creation(GetEnv_msgType(env), Entry, env);
      CkCheck_and_BcastInitNFNL(env);
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

	trace_creation(GetEnv_msgType(env), ep, env);
	CkCheck_and_Send(destPE, env);
	QDCountThisCreation(ep, category, type, 1);
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

	trace_creation(GetEnv_msgType(env), ep, env);
	CkCheck_and_BroadcastAll(env); /* Asynchronous broadcast */
	QDCountThisCreation(ep, category, type, CmiNumPes());
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

	trace_creation(GetEnv_msgType(env), ep, env);
        CkCheck_and_BroadcastAll(env);

        QDCountThisCreation(ep, USERcat, DynamicBocInitMsg,CmiNumPes());

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


