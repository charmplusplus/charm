/**********************************************************************
qd.c

This file deals with quiescence detection. It is implemented as
a branch- office-chare. These are the routines that deal with
the node part of the boc.
***********************************************************************/

#include "charm.h"

#include "converse.h"
#include <stdio.h>
#include <string.h>

/*** MACROS ***/
#define isLeaf(peNum) (CmiNumSpanTreeChildren(peNum) == 0)

typedef struct quiescence_msg {
	int msgs_processed;
} QUIESCENCE_MSG;


typedef struct {
    int msgs_processed;
    int msgs_created;
} PhaseIMSG;

typedef struct {
    int wasDirty;       /* indicates whether anything happened since */
                    /*  last phase */
} PhaseIIMSG;

typedef struct quiescence_entry {
    EntryPointType ep;
    ChareIDType chareid;
    struct quiescence_entry *next;
} QUIESCENCE_ENTRY;

typedef struct start_quiescence_msg {
    EntryPointType ep;
    ChareIDType chareid;
} START_QUIESCENCE_MSG;




CHARE_BLOCK *CreateChareBlock();




typedef int LIST_[1000];

CpvStaticDeclare(LIST_, creation_list);
CpvStaticDeclare(LIST_, process_list);


/* globals used for messages since we don't need to     */
/* re-alloc each time                                   */


CpvStaticDeclare(EntryPointType, userEntry);
CpvStaticDeclare(ChareIDType, userChareID);
CpvStaticDeclare(int, quiescenceStarted);
CpvStaticDeclare(int, myPe);
CpvStaticDeclare(int, mainPe);
CpvStaticDeclare(int, rootPe);
CpvStaticDeclare(int, numKids);
CpvStaticDeclare(int, parentPe);
CpvStaticDeclare(QUIESCENCE_ENTRY*, quiescence_list);


/* we use this variable to indicate when we are checking in the queue */
/* for messages. We need it because we might come through again and   */
/* we don't want to make another call to CallBocOnCondition if one is */
/* already pending. */

CpvStaticDeclare(int, subtree_msgs_processed);
CpvStaticDeclare(int, subtree_msgs_created);
CpvStaticDeclare(int, msgs_previously_processed);
CpvStaticDeclare(int, root_msgs_created);
CpvStaticDeclare(int, root_msgs_processed);
CpvStaticDeclare(int, HostQDdirty);
CpvStaticDeclare(int, QDdirty);   
                   /* QDdirty = 1 :some msg was enqueued since the last
                   time this bit was reset. OR one of child sent a
                   message in phaseII, with the wasDirty bit set 0. */

CpvStaticDeclare(int, countResponses); /* # of rspnses received from children in this phase. */

void InsertQuiescenceList();


void quiesModuleInit(void)
{
   CpvInitialize(LIST_, creation_list);
   CpvInitialize(LIST_, process_list);
   CpvInitialize(EntryPointType, userEntry);
   CpvInitialize(ChareIDType, userChareID);
   CpvInitialize(int, quiescenceStarted);
   CpvInitialize(int, myPe);
   CpvInitialize(int, mainPe);
   CpvInitialize(int, rootPe);
   CpvInitialize(int, numKids);
   CpvInitialize(int, parentPe);
   CpvInitialize(QUIESCENCE_ENTRY*, quiescence_list);
   CpvInitialize(int, subtree_msgs_processed);
   CpvInitialize(int, subtree_msgs_created);
   CpvInitialize(int, msgs_previously_processed);
   CpvInitialize(int, root_msgs_created);
   CpvInitialize(int, root_msgs_processed);
   CpvInitialize(int, HostQDdirty);
   CpvInitialize(int, QDdirty);   
   CpvInitialize(int, countResponses);

   CpvAccess(quiescence_list) = NULL;
}




/********************** FUNCTION PROTOTYPES *****************/
void QDBocInit();
/************************ phase I routines  *****************/
void StartPhaseI();
int  EndPhaseI();
void PhaseIBroadcast();
void HandlePhaseIMsg();
int  sendUpI();
/************************ phase II routines *****************/
void StartPhaseII();
int  EndPhaseII();
void PhaseIIBroadcast();
void HandlePhaseIIMsg();
int  sendUpII();
/************************ utility routine   *****************/
void QDAddSysBocEps();





/***************************************************************************
 this routine initiaizes the node part of the QD boc's. It is called by the 
 SysBocInit routine from the nodeBocinitLoop which is called from main.
****************************************************************************/
void QDBocInit()
{
    CHARE_BLOCK *bocblock;

    /* Create a dummy BOC node */
    bocblock = CreateChareBlock(0, CHAREKIND_BOCNODE, 0);
    bocblock->x.boc_num = QDBocNum;
    SetBocBlockPtr(QDBocNum, bocblock);

    CpvAccess(countResponses)         = 0;
    CpvAccess(msgs_previously_processed) = 0;
    CpvAccess(root_msgs_created) = CpvAccess(root_msgs_processed) = 0;
    
    CpvAccess(myPe)        = CmiMyPe();
    CpvAccess(mainPe)      = 0;
    CpvAccess(rootPe)      = 0;
    CpvAccess(numKids)     = CmiNumSpanTreeChildren(CpvAccess(myPe));
    CpvAccess(parentPe)    = (CpvAccess(myPe) == CpvAccess(rootPe)) ? CpvAccess(mainPe) : CmiSpanTreeParent(CpvAccess(myPe));
    CpvAccess(quiescenceStarted) = 0;
}

/***************************************************************************
The next two procedures start quiescence and insert every request into
quiescence list, so that when quiescence is detected a message can be 
sent back to the user at every requested entry point.
***************************************************************************/
void StartQuiescence(ep, chareid)
EntryPointType ep;
ChareIDType *chareid;
{
	START_QUIESCENCE_MSG *msg ;	
	msg = (START_QUIESCENCE_MSG *) CkAllocMsg(sizeof(START_QUIESCENCE_MSG));
	msg->ep = ep;
	msg->chareid = *chareid;
	if (CmiMyPe() == 0)
		InsertQuiescenceList(msg, NULL);
	else
                GeneralSendMsgBranch(CsvAccess(CkEp_QD_InsertQuiescenceList),
                                msg, 0,
                                QdBocMsg, QDBocNum);

}


void InsertQuiescenceList(msgptr, localdataptr)
START_QUIESCENCE_MSG *msgptr;
void *localdataptr;
{
	QUIESCENCE_ENTRY *new = (QUIESCENCE_ENTRY *)
					 CmiAlloc(sizeof(QUIESCENCE_ENTRY));
	new->ep = msgptr->ep;
	new->chareid = msgptr->chareid;
	CkFreeMsg(msgptr);	
	new->next = CpvAccess(quiescence_list);
	CpvAccess(quiescence_list) = new;
	if (!CpvAccess(quiescenceStarted)) {
            StartPhaseI();
			CpvAccess(quiescenceStarted) = 1;
	}
}

/***************************************************************************
***************************************************************************/
void PhaseIBroadcast(msgptr_,localdataptr_)
void *msgptr_, *localdataptr_;
{
	PhaseIMSG * msg = (PhaseIMSG *) msgptr_;

TRACE(CmiPrintf("Inside PhaseIBroadcast\n"));

	if (CpvAccess(countResponses) == 0)
		CpvAccess(subtree_msgs_created) = CpvAccess(subtree_msgs_processed) = 0;

	if (isLeaf(CpvAccess(myPe))) {
		TRACE(CmiPrintf("PhaseIBroadcast: lf[%d],cllng CallBocOnCnd\n",CpvAccess(myPe)));

		CallBocOnCondition(sendUpI, QDBocNum);
	}
	else {
		TRACE(CmiPrintf("PhaseIBroadcast: interior  node[%d], \n",CpvAccess(myPe)));
	}
	CkFreeMsg(msg);
}


/*************************************************************************** 
wait for responses from all children, and then set a
call-on-condition to call sendUpI when the queue is empty 
****************************************************************************/
void HandlePhaseIMsg(msgptr_,localdataptr_)
void *msgptr_, *localdataptr_;
{
	PhaseIMSG *msg = (PhaseIMSG *) msgptr_;

	if (CpvAccess(countResponses) == 0)
		CpvAccess(subtree_msgs_created) = CpvAccess(subtree_msgs_processed) = 0;
	CpvAccess(countResponses)++;
	CpvAccess(subtree_msgs_created) += msg->msgs_created;
	CpvAccess(subtree_msgs_processed) += msg->msgs_processed;

	TRACE(CmiPrintf("HandlePhaseIMsg nd[%d]  children[%d]  responses so far[%d]\n", 
	    CpvAccess(myPe), CpvAccess(numKids), CpvAccess(countResponses)));
	if (CpvAccess(countResponses) == CpvAccess(numKids)){
		TRACE(CmiPrintf("HandlePhaseIMsg nd[%d] have all resps, cllng CallBocOn..\n",
		    CpvAccess(myPe)));
		CallBocOnCondition(sendUpI, QDBocNum);
	}
	CkFreeMsg(msg);
}


/**************************************************************************
This is a boc access function, hence the unused reference to the bocNum. 
The logic is as follows:

if queue is empty
    if I am root
        start phase II after deltaT
    else
        send Msg to spanning tree parent
    return 1
else
    return 0
**************************************************************************/
int sendUpI(bocNum)
int bocNum;
{
	PhaseIMSG     *msg1;

TRACE(CmiPrintf("Inside sendUpI\n"));

/* SANJEEV, ATTILA Jun 8 : made AllAsyncMsgsSent an Mc function */

	if (CsdEmpty() && NoDelayedMsgs() ) {
		msg1 = (PhaseIMSG *)  CkAllocMsg(sizeof(PhaseIMSG));
		CkMemError(msg1);
		msg1->msgs_created = CpvAccess(subtree_msgs_created) + CpvAccess(msgs_created);
		msg1->msgs_processed = CpvAccess(subtree_msgs_processed) + CpvAccess(msgs_processed);
		CpvAccess(msgs_previously_processed) = CpvAccess(msgs_processed);

		TRACE(CmiPrintf("sendUpI : created = %d, processed = %d\n", 
		    msg1->msgs_created, msg1->msgs_processed));
		CpvAccess(countResponses) = 0;   /* reset here for next phase (timing reasons) */
		CpvAccess(QDdirty) = 0;      /* flag, 0 if we've seen no messages */
		if (CpvAccess(myPe) == CpvAccess(rootPe)) {
			CpvAccess(root_msgs_created) = msg1->msgs_created;
			CpvAccess(root_msgs_processed) = msg1->msgs_processed;
			CallBocOnCondition(EndPhaseI,QDBocNum);
			CkFreeMsg(msg1);
		}
		else {
			TRACE(CmiPrintf("Nd[%d] sendUpI() sndng endPhaseI msg up to nd[%d]\n",
			    CpvAccess(myPe),CpvAccess(parentPe)));
			GeneralSendMsgBranch(CsvAccess(CkEp_QD_PhaseIMsg),
				msg1, CpvAccess(parentPe),
				QdBocMsg, QDBocNum);
		}
		return(1); /* inform the conditional-wait manager that the cond.
				                     is satisfied, (and so should be removed from the Q) */
	}
	return(0);    /* not done yet, tell conditional wait mnger to try again */
}



/****************************************************************************
if you are a leaf, send up a phase II message to your parent.
otherwise, reset counts to 0 
****************************************************************************/
void PhaseIIBroadcast(msgptr_,localdataptr_)
void *msgptr_, *localdataptr_;
{
	PhaseIIMSG * msg = (PhaseIIMSG *) msgptr_;

	if (isLeaf(CpvAccess(myPe))) {
		TRACE(CmiPrintf("PhaseIIBroadcast lf nd[%d],dirty[%d] CallBoc..SendUpII\n",
		    CpvAccess(myPe),msg->wasDirty));
		CallBocOnCondition(sendUpII,QDBocNum);
	}
	else {
		TRACE(CmiPrintf("PhaseIIBroadcast interior nd[%d],dirty[%d]\n",
		    CpvAccess(myPe),msg->wasDirty));
		/* countResponses was reset in sendUpI, so don't do it here */
		/* just wait to receive messages from the kids */
	}
	CkFreeMsg(msg);
}


/**************************************************************************
wait for responses from all children, and then set a
call-on-condition to call sendUpII when the queue is empty 
**************************************************************************/
void HandlePhaseIIMsg(msgptr_,localdataptr_)
void *msgptr_, *localdataptr_;
{
	PhaseIIMSG *msg = (PhaseIIMSG *)  msgptr_;

	TRACE(CmiPrintf("HandlePhaseIIMsg() nd[%d], dirty[%d] kds[%d] resp[%d]\n",
	    CpvAccess(myPe),msg->wasDirty,CpvAccess(numKids),CpvAccess(countResponses)+1));
	CpvAccess(countResponses)++;
	CpvAccess(QDdirty) = (CpvAccess(QDdirty) || msg->wasDirty);
	if (CpvAccess(countResponses) == CpvAccess(numKids))
		CallBocOnCondition(sendUpII,QDBocNum);
	CkFreeMsg(msg);
}


/**************************************************************************
Returns 1 when our queue is empty
**************************************************************************/
int sendUpII(bocNum)
int bocNum;
{
	PhaseIIMSG     *msg2;

	if (CsdEmpty() && NoDelayedMsgs() ) {
		msg2 = (PhaseIIMSG *) CkAllocMsg(sizeof(PhaseIIMSG));
		CkMemError(msg2);
		/* flag to indicate activity until this pass */
		msg2->wasDirty = CpvAccess(QDdirty) || 
		    (CpvAccess(msgs_previously_processed) < CpvAccess(msgs_processed)) ;
		CpvAccess(QDdirty)        = 0;
		CpvAccess(countResponses )= 0;          /* reset for next iteration of program */
		TRACE(CmiPrintf("sendUPII :: QDdirty = %d\n", msg2->wasDirty));
		if (CpvAccess(myPe) == CpvAccess(rootPe)) {        /* root, pass to host */
			CpvAccess(HostQDdirty) = CpvAccess(HostQDdirty) || msg2->wasDirty;
			CallBocOnCondition(EndPhaseII,QDBocNum);
			CkFreeMsg(msg2);
		}
		else {                       /* not the root, pass it up */
			TRACE(CmiPrintf("sendUpII node[%d]sending endPhaseII msg up to nd[%d]\n",
			    CpvAccess(myPe),CmiSpanTreeParent(CpvAccess(myPe))));
			GeneralSendMsgBranch(CsvAccess(CkEp_QD_PhaseIIMsg),
			    	msg2, CpvAccess(parentPe),
				QdBocMsg, QDBocNum);
		}
		return(1); /* inform the cond.-wait manager that the cond. stsfd */
	}
	return(0);    /* not done yet, tell cond. wait mngr to try again later */
}



/***************************************************************************
*****************************************************************************/
void StartPhaseI()
{
	int ep;
	PhaseIMSG     *msg1;


	TRACE(CmiPrintf("Host: Starting PhaseI at [%d]\n",CkTimer()));

	msg1 = (PhaseIMSG *)  CkAllocMsg(sizeof(PhaseIMSG));
	CkMemError(msg1);
	ep   = CsvAccess(CkEp_QD_PhaseIBroadcast);

	GeneralBroadcastMsgBranch(ep, msg1,
			QdBroadcastBocMsg, QDBocNum);
}



/*****************************************************************************
*****************************************************************************/
int EndPhaseI(bocNum)
int bocNum;
{
	if(CsdEmpty() && NoDelayedMsgs() ) {

TRACE(CmiPrintf("EndPhaseI: root_created=%d, root_processed=%d\n",
CpvAccess(root_msgs_created), CpvAccess(root_msgs_processed)));

		if (CpvAccess(root_msgs_created) == CpvAccess(root_msgs_processed))
			StartPhaseII();
		else
			StartPhaseI();
		return(1);
	}
	else
		return(0);
}



/*****************************************************************************
*****************************************************************************/
void StartPhaseII(bocNum)
int bocNum;
{
	int ep;
	PhaseIIMSG     *msg2;

	TRACE(CmiPrintf("Host: starting PhaseII at T=[%d]\n",CkTimer()));
	CpvAccess(HostQDdirty) = 0;

	ep   = CsvAccess(CkEp_QD_PhaseIIBroadcast);
	msg2 = (PhaseIIMSG *) CkAllocMsg(sizeof(PhaseIIMSG));
	CkMemError(msg2);

	GeneralBroadcastMsgBranch(ep, msg2, 
		QdBroadcastBocMsg, QDBocNum);
}


/*****************************************************************************
*****************************************************************************/
int EndPhaseII(bocNum)
{
	PhaseIIMSG  *msg ;

/* 	PumpMsgs();	removed 3/17/95 : Sanjeev */

	if( CsdEmpty() && NoDelayedMsgs()) {
		TRACE(CmiPrintf("Host: EndPhaseII(), nodesaredone, q empty, no msgs\n"));
		if(!CpvAccess(HostQDdirty)) {
			QUIESCENCE_ENTRY *new;
			new = CpvAccess(quiescence_list);
			while (CpvAccess(quiescence_list) != NULL) {
				msg  = (PhaseIIMSG *)
					 CkAllocMsg(sizeof(PhaseIIMSG));
				CkMemError(msg);
				SendMsg(CpvAccess(quiescence_list)->ep, msg, 
						&CpvAccess(quiescence_list)->chareid);
				CpvAccess(quiescence_list) = CpvAccess(quiescence_list)->next;
				CmiFree(new);	
				new = CpvAccess(quiescence_list);
			}
			CpvAccess(quiescenceStarted) = 0;
		}
		else 
			StartPhaseI();
		return (1);
	}
	return(0);
}


/***************************************************************************
here we set up the entry points so we can call our boc ep functions
****************************************************************************/
void QDAddSysBocEps()
{
    CsvAccess(CkEp_QD_Init) =
      registerBocEp("CkEp_QD_Init", 
		 QDBocInit,
		 CHARM, 0, 0);
    CsvAccess(CkEp_QD_InsertQuiescenceList) =
      registerBocEp("CkEp_QD_InsertQuiescenceList",
		 InsertQuiescenceList,
		 CHARM, 0, 0);
    CsvAccess(CkEp_QD_PhaseIBroadcast) =
      registerBocEp("CkEp_QD_PhaseIBroadcast",
		 PhaseIBroadcast, 
		 CHARM, 0, 0);
    CsvAccess(CkEp_QD_PhaseIMsg) =
      registerBocEp("CkEp_QD_PhaseIMsg",
		 HandlePhaseIMsg, 
		 CHARM, 0, 0);
    CsvAccess(CkEp_QD_PhaseIIBroadcast) =
      registerBocEp("CkEp_QD_PhaseIIBroadcast",
		 PhaseIIBroadcast, 
		 CHARM, 0, 0);
    CsvAccess(CkEp_QD_PhaseIIMsg) =
      registerBocEp("CkEp_QD_PhaseIIMsg",
		 HandlePhaseIIMsg, 
		 CHARM, 0, 0);
}
