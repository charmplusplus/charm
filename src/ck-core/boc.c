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
 * Revision 2.5  1995-07-24 01:54:40  jyelon
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
#include "performance.h"

#define MAXBOC 15

typedef struct msg_element {
        int ref;
        int size;
        void *msg;
        ChareNumType ep;
        struct msg_element *next;
} MSG_ELEMENT;

typedef struct bocdata_queue_element {
	ChareNumType bocNum;
	void *dataptr;
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


void bocModuleInit()
{
   CpvInitialize(int, number_dynamic_boc);
   CpvInitialize(MSG_ELEMENT_, DynamicBocMsgList);
   CpvInitialize(BOCDATA_QUEUE_ELEMENT_, BocDataTable);
   CpvInitialize(BOCID_MESSAGE_COUNT_, BocIDMessageCountTable);

   CpvAccess(number_dynamic_boc)=1;
}




InitializeDynamicBocMsgList()
{
	CpvAccess(DynamicBocMsgList) = (MSG_ELEMENT *) NULL;
}

InitializeBocDataTable()
{
	int i;

	for (i=0; i<MAXBOC; i++)
	    CpvAccess(BocDataTable)[i] = (BOCDATA_QUEUE_ELEMENT *) NULL;
}

InitializeBocIDMessageCountTable()
{
	int i;

	for (i=0; i<MAXBOC; i++)
	    CpvAccess(BocIDMessageCountTable)[i] = (BOCID_MESSAGE_COUNT *) NULL;
}

GetDynamicBocMsg(ref, msg, ep, size)
int ref;
void **msg;
ChareNumType *ep;
int *size;
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
			*size = temp->size;

TRACE(CmiPrintf("[%d] GetDynamicBocMsg: ref=%d, ep=%d, size=%d\n",
		CmiMyPe(), ref, temp->ref, temp->size)); 
	
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


void * GetBocDataPtr(bocNum)
ChareNumType bocNum;
{
	int index;
	BOCDATA_QUEUE_ELEMENT *element;


	index = bocNum % MAXBOC;
	element = CpvAccess(BocDataTable)[index];

TRACE(CmiPrintf("[%d] GetBocDataPtr: bocNum=%d, index=%d, element=0x%x\n",
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

SetDynamicBocMsg(msg,ep,size)
void *msg;
ChareNumType ep;
int size;
{
	MSG_ELEMENT * new; 
	
	new = (MSG_ELEMENT *) CmiAlloc(sizeof(MSG_ELEMENT));
	new->ref = CpvAccess(number_dynamic_boc)++;
	new->msg = msg;
	new->ep = ep;
	new->size = size;
	new->next = CpvAccess(DynamicBocMsgList); 
	CpvAccess(DynamicBocMsgList) = new; 

TRACE(CmiPrintf("[%d] SetDynamicBocMsg: ref=%d, ep=%d, size=%d\n",
	CmiMyPe(), new->ref, new->ep, new->size));

	return (new->ref);
}

SetBocDataPtr(bocNum, ptr)
ChareNumType bocNum;
void *ptr;
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

TRACE(CmiPrintf("[%d] SetBocDataPtr: bocNum=%d, index=%d, new=0x%x\n",
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




BOC_BLOCK *CreateBocBlock(sizeData)
int sizeData;
{
	BOC_BLOCK *p;

	p =  (BOC_BLOCK *)CmiAlloc( sizeof(BOC_BLOCK) + sizeData);
	CkMemError(p);
	return(p);
}



ChareNumType GeneralCreateBoc(SizeData, Entry, Msg, ReturnEP, ReturnID)
int SizeData;
EntryNumType Entry;
void *Msg;
EntryNumType ReturnEP;
ChareIDType *ReturnID;
{
	ENVELOPE *env ;
        int dataMag = bytes_to_magnitude(SizeData);

TRACE(CmiPrintf("[%d] GeneralCreateBoc: Entry=%d, ReturnEP=%d\n",
		CmiMyPe(), Entry, ReturnEP));

	env = (ENVELOPE *) ENVELOPE_UPTR(Msg);

	if ((CmiMyPe() == 0)  || CpvAccess(InsideDataInit))
	{
		SetEnv_dataMag(env, dataMag);
		SetEnv_boc_num(env, ++CpvAccess(currentBocNum));
		SetEnv_EP(env, Entry);
	}
	if (CpvAccess(InsideDataInit))
	/* static boc creation */
	{
		int executing_boc_num; 

		SetEnv_msgType(env, BocInitMsg);
		trace_creation(GetEnv_msgType(env), Entry, env);
		CkCheck_and_BroadcastNoFreeNoLdb(env);
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
		return(CpvAccess(currentBocNum));
	}
	else
	/* dynamic boc creation */
	{
		if (CmiMyPe() == 0)
		{
			BOCID_MESSAGE_COUNT *element;

			element = SetBocIDMessageCount(CpvAccess(currentBocNum), 
					CmiNumSpanTreeChildren(CmiMyPe()),
					ReturnEP, ReturnID);
			SetEnv_msgType(env, DynamicBocInitMsg);

			trace_creation(GetEnv_msgType(env), Entry, env);
			CkCheck_and_BroadcastNoFree(env);

		        CmiSetHandler(env,
                            CsvAccess(CkProcess_DynamicBocInitMsg_Index));
			CkEnqueue(env);
			QDCountThisCreation(Entry, USERcat, DynamicBocInitMsg, CmiNumPe());

TRACE(CmiPrintf("[%d] GeneralCreateBoc: bocdata=0x%x\n", CmiMyPe(), element));
		}
		else
		{
			DYNAMIC_BOC_REQUEST_MSG *msg;
					
			msg = (DYNAMIC_BOC_REQUEST_MSG *) 
				CkAllocMsg(sizeof(DYNAMIC_BOC_REQUEST_MSG));
			msg->source = CmiMyPe();
			msg->ep = ReturnEP;
			msg->id = *ReturnID;
			msg->ref = SetDynamicBocMsg(Msg, Entry, SizeData);

			GeneralSendMsgBranch(OtherCreateBoc_EP, msg,
			 	0, IMMEDIATEcat, ImmBocMsg,
				DynamicBocNum);
		}
	}
}

MyBocNum(mydata)
void *mydata;
{
	BOC_BLOCK * boc_block = (BOC_BLOCK * ) ((char *) mydata - sizeof(BOC_BLOCK));

	return(boc_block->boc_num);
}

MyBranchID(pChareID, mydata)
ChareIDType *pChareID;
void *mydata;
{
	SetID_onPE((*pChareID), CmiMyPe());
	SetID_boc_num((*pChareID), MyBocNum(mydata));
}

GeneralSendMsgBranch(ep, msg, destPE, category, type, bocnum)
EntryPointType ep;
void *msg;
PeNumType destPE;
MsgCategories category;
MsgTypes type;
ChareNumType bocnum;
{
	ENVELOPE *env;

	env  = ENVELOPE_UPTR(msg);

	SetEnv_msgType(env, type);
	SetEnv_boc_num(env, bocnum);
	SetEnv_EP(env, ep);

TRACE(CmiPrintf("[%d] GeneralSend: type=%d, msgType=%d\n",
		CmiMyPe(), type, GetEnv_msgType(env)));

	if (bocnum >= NumSysBoc)
        	CpvAccess(nodebocMsgsCreated)++;

	trace_creation(GetEnv_msgType(env), ep, env);
	CkCheck_and_Send(destPE, env);
	QDCountThisCreation(ep, category, type, 1);
}



GeneralBroadcastMsgBranch(ep, msg, category, type, bocnum)
EntryPointType ep;
void *msg;
MsgCategories category;
MsgTypes type;
ChareNumType bocnum;
{
	ENVELOPE *env;

	env = ENVELOPE_UPTR(msg);

	SetEnv_msgType(env, type);
	SetEnv_boc_num(env, bocnum);
	SetEnv_EP(env, ep);

TRACE(CmiPrintf("[%d] GeneralBroadcast: type=%d, msgType=%d\n",
		CmiMyPe(), type, GetEnv_msgType(env)));

	if (bocnum >= NumSysBoc)
        	CpvAccess(nodebocMsgsCreated)+=CmiNumPe();

	trace_creation(GetEnv_msgType(env), ep, env);
	CkCheck_and_BroadcastAll(env); /* Asynchronous broadcast */
	QDCountThisCreation(ep, category, type, CmiNumPe());
}


RegisterDynamicBocInitMsg(bocnumptr, mydata)
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
			GeneralSendMsgBranch(RegisterDynamicBocInitMsg_EP, msg,
                           CmiSpanTreeParent(mype), IMMEDIATEcat, ImmBocMsg,
			   DynamicBocNum);
	}
}


OtherCreateBoc(msg, mydata)
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

	GeneralSendMsgBranch(InitiateDynamicBocBroadcast_EP, tmsg,
                                msg->source, IMMEDIATEcat, ImmBocMsg,
                                DynamicBocNum);
}

InitiateDynamicBocBroadcast(msg, mydata)
DYNAMIC_BOC_NUM_MSG *msg;
char *mydata;
{
	int dataSize, dataMag;
	void *tmsg;
        ENVELOPE * env;
	ChareNumType ep;

	GetDynamicBocMsg(msg->ref, &tmsg, &ep, &dataSize); 
        dataMag = bytes_to_magnitude(dataSize);

TRACE(CmiPrintf("[%d] InitiateDynamicBocBroadcast: ref=%d, boc=%d, ep=%d\n",
		CmiMyPe(), msg->ref, msg->boc, ep));

        env = (ENVELOPE *) ENVELOPE_UPTR(tmsg);
        SetEnv_dataMag(env, dataMag);
        SetEnv_boc_num(env, msg->boc);
        SetEnv_EP(env, ep);
        SetEnv_msgType(env, DynamicBocInitMsg);

	trace_creation(GetEnv_msgType(env), ep, env);
        CkCheck_and_BroadcastAll(env);

        QDCountThisCreation(ep, USERcat, DynamicBocInitMsg,CmiNumPe());

}

DynamicBocInit()
{
    	BOC_BLOCK *bocBlock;

	/* Create a dummy block */
    	bocBlock = (BOC_BLOCK *) CreateBocBlock(sizeof(int));
	bocBlock->boc_num = DynamicBocNum;
    	SetBocDataPtr(DynamicBocNum, (void *) (bocBlock + 1));
}


DynamicAddSysBocEps()
{
   	CsvAccess(EpTable)[RegisterDynamicBocInitMsg_EP] = RegisterDynamicBocInitMsg;
   	CsvAccess(EpTable)[OtherCreateBoc_EP] = OtherCreateBoc;
	CsvAccess(EpTable)[InitiateDynamicBocBroadcast_EP] = 
				InitiateDynamicBocBroadcast;
}






ChareNumType CreateBoc(id, Entry, Msg, ReturnEP, ReturnID)
int id;
EntryNumType Entry;
void *Msg;
EntryNumType ReturnEP;
ChareIDType *ReturnID;
{
	if ( IsCharmPlus(Entry) )
        	return GeneralCreateBoc(id, Entry, Msg, ReturnEP, ReturnID);
	else
        	return GeneralCreateBoc(CsvAccess(ChareSizesTable)[id], Entry, Msg,
                                         		ReturnEP, ReturnID);
}

