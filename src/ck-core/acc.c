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
 * Revision 2.6  1997-10-29 23:52:42  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.5  1995/10/13 18:15:53  jyelon
 * K&R changes.
 *
 * Revision 2.4  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.3  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.2  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:09:41  gursoy
 * Cpv macro changes done
 *
 * Revision 1.3  1995/04/13  20:55:22  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.2  1994/12/01  23:57:49  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:39:00  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "chare.h"
#include "globals.h"
#include "acc.h"

void *GetAccMsgPtr() ;
extern void * CPlus_CallAccInit() ;
extern void * CPlus_GetAccMsgPtr() ;
extern void CPlus_CallCombineFn() ;
extern void CPlus_SetAccId() ;



/* internal functions */
static void ACC_CollectFromNode_Fn();
static void ACC_LeafNodeCollect_Fn();
static void ACC_InteriorNodeCollect_Fn();
static void ACC_BranchInit_Fn();


void AccAddSysBocEps(void)
{
  CsvAccess(CkChare_ACC) =
    registerChare("CkChare_ACC", sizeof(ACC_DATA), NULL);

  CsvAccess(CkEp_ACC_CollectFromNode) =
    registerBocEp("CkEp_ACC_CollectFromNode",
		  ACC_CollectFromNode_Fn,
		  CHARM, 0, CsvAccess(CkChare_ACC));
  CsvAccess(CkEp_ACC_LeafNodeCollect) =
    registerBocEp("CkEp_ACC_LeafNodeCollect",
		  ACC_LeafNodeCollect_Fn,
		  CHARM, 0, CsvAccess(CkChare_ACC));
  CsvAccess(CkEp_ACC_InteriorNodeCollect) =
    registerBocEp("CkEp_ACC_InteriorNodeCollect",
		  ACC_InteriorNodeCollect_Fn,
		  CHARM, 0, CsvAccess(CkChare_ACC));
  CsvAccess(CkEp_ACC_BranchInit) =
    registerBocEp("CkEp_ACC_BranchInit",
		  ACC_BranchInit_Fn,
		  CHARM, 0, CsvAccess(CkChare_ACC));
}





void CollectValue(bocnum, EP, CID)
int bocnum;
int EP;
ChareIDType *CID;
{
	ACC_COLLECT_MSG *msg;

	msg = (ACC_COLLECT_MSG *) CkAllocMsg(sizeof(ACC_COLLECT_MSG));
	msg->EP = EP; 
	msg->cid = *CID;
	if (CmiMyPe() == 0)
		ACC_CollectFromNode_Fn(msg, GetBocDataPtr(bocnum)); 
	else
		GeneralSendMsgBranch(CsvAccess(CkEp_ACC_CollectFromNode), msg,
			0, ImmBocMsg, bocnum);
}


static void ACC_CollectFromNode_Fn(msg, mydata)
ACC_COLLECT_MSG *msg;
ACC_DATA *mydata;
{
	int i;
	DummyMsg *tmsg;

	if (mydata->AlreadyDone)
		CmiPrintf("***ERROR*** Accumulation already done\n");	
	else
	{
		mydata->EP = msg->EP;
		mydata->CID = msg->cid;
		tmsg = (DummyMsg *) CkAllocMsg(sizeof(DummyMsg));
		mydata->AlreadyDone = 1;
		GeneralBroadcastMsgBranch(CsvAccess(CkEp_ACC_LeafNodeCollect), tmsg,
			ImmBroadcastBocMsg,
                        MyBocNum(mydata));
	}
}




static void ACC_LeafNodeCollect_Fn(msg, mydata)
DummyMsg *msg;
ACC_DATA *mydata;
{

	if (CmiNumSpanTreeChildren(mydata->Penum) == 0)
	{
		if (mydata->Penum == CmiSpanTreeRoot())
			SendMsg(mydata->EP,  GetAccMsgPtr(mydata),
				&(mydata->CID)); 
		else
		{
TRACE(CmiPrintf("[%d] ACC_NodeCollect : Sent message to parent\n",
	CmiMyPe()));
			mydata->Penum = CmiSpanTreeParent(mydata->Penum);
			GeneralSendMsgBranch(CsvAccess(CkEp_ACC_InteriorNodeCollect), 
					GetAccMsgPtr(mydata), mydata->Penum,
					ImmBocMsg,
                                        MyBocNum(mydata));
		}
	}
}



static void ACC_InteriorNodeCollect_Fn(msg, mydata)
char *msg;
ACC_DATA *mydata;
{

	if ( IsCharmPlusPseudo(mydata->id) ) 
		CPlus_CallCombineFn(mydata->dataptr,msg) ;
	else 
		(*(CsvAccess(PseudoTable)[mydata->id].pseudo_type.acc.combinefn))(mydata->dataptr, msg);
	mydata->NumChildren--;
	if (mydata->NumChildren <= 0)
	{
		if (mydata->Penum == CmiSpanTreeRoot())
			SendMsg(mydata->EP, GetAccMsgPtr(mydata),
				&(mydata->CID)); 
		else
		{
TRACE(CmiPrintf("[%d] ACC_NodeCollect : Sent message to parent\n",
		CmiMyPe()));
			mydata->Penum = CmiSpanTreeParent(mydata->Penum);
			GeneralSendMsgBranch(CsvAccess(CkEp_ACC_InteriorNodeCollect), 
					GetAccMsgPtr(mydata), mydata->Penum,
					ImmBocMsg,
                                        MyBocNum(mydata));
		}
	}
}


AccIDType CreateAcc(id, initmsg, ReturnEP, ReturnID)
int id;
void *initmsg;
EntryPointType ReturnEP;
ChareIDType *ReturnID;
{
	ChareNumType boc;
	ENVELOPE *envelope = (ENVELOPE *) ENVELOPE_UPTR(initmsg);

	SetEnv_other_id(envelope, id);
TRACE(CmiPrintf("[%d] CreateAcc: id=%d\n", CmiMyPe(), id));
	boc = CreateBoc(CsvAccess(CkChare_ACC), CsvAccess(CkEp_ACC_BranchInit),
			 initmsg, ReturnEP, ReturnID);
TRACE(CmiPrintf("[%d] CreateAcc: boc = %d\n", CmiMyPe(), (AccIDType ) boc));
	return((AccIDType) boc);
}


static void ACC_BranchInit_Fn(msg, mydata)
void *msg;
ACC_DATA *mydata; 
{
	ENVELOPE *env = (ENVELOPE *) ENVELOPE_UPTR(msg);
	int id = GetEnv_other_id(env);

TRACE(CmiPrintf("[%d] ACC_BranchInit : id = %d\n", CmiMyPe(), id));

	mydata->id = id; 
	mydata->AlreadyDone = 0;
	mydata->Penum = CmiMyPe();
	mydata->NumChildren = CmiNumSpanTreeChildren(mydata->Penum) ;	

	if ( IsCharmPlusPseudo(id) ) {
	    mydata->dataptr = CPlus_CallAccInit(id, msg) ;
	    CPlus_SetAccId(mydata->dataptr,MyBocNum(mydata)) ;
	}
	else 
	    mydata->dataptr = (void *) (*(CsvAccess(PseudoTable)[id].initfn))(NULL, msg);

TRACE(CmiPrintf("[%d] ACC_BranchInit : NumChildren = %d\n",
	CmiMyPe(),  mydata->NumChildren));
}


void * _CK_9GetAccDataPtr(accdata)
ACC_DATA * accdata; 
{
	return(accdata->dataptr);
}


FUNCTION_PTR _CK_9GetAccumulateFn(accdata)
ACC_DATA * accdata; 
{
	return(CsvAccess(PseudoTable)[accdata->id].pseudo_type.acc.addfn);
}


void * GetAccMsgPtr(mydata)
ACC_DATA *mydata ;
{
/* In Charm++, the dataptr points to the accumulator object, and the
   actual data (in the form of a message as for Charm) is a field inside 
   the object, which is accessed through the _CK_GetMsgPtr() generated
   by the translator */
	if ( IsCharmPlusPseudo(mydata->id) ) 
		return(CPlus_GetAccMsgPtr(mydata->dataptr)) ;
	else
		return(mydata->dataptr) ;	
}

