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
 * Revision 2.3  1997-12-22 21:57:20  jyelon
 * Changed LDB initialization scheme.
 *
 * Revision 2.2  1995/10/27 22:09:16  jyelon
 * Changed Cmi to Ck in all charm files.
 *
 * Revision 2.1  1995/10/27  21:35:54  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.0  1995/06/29  21:19:36  narain
 * *** empty log message ***
 *
 ***************************************************************************/
module ldb {
#include "ldb.h"


message {
    int dummy;
} DUMMYMSG;

typedef struct ldb_status {
	int	peLoad;
	int myLoadSent;
} LDB_STATUS;

typedef struct {
	int idlePE;
} LDB_ELEMENT;

export_to_C CldModuleInit()
{
  CldCommonInit();
}

export_to_C setLdbSize()
{
       CpvAccess(LDB_ELEM_SIZE) = sizeof(LDB_ELEMENT);
}


export_to_C LdbCreateBoc()
{
  DUMMYMSG *msg;
  msg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG);	
  CreateBoc(LDB, LDB@BranchInit, msg);
}

export_to_C LdbFillLDB(ldb)
  void *ldb;
{
}

export_to_C LdbStripLDB(ldb)
     void *ldb;
{
}


export_to_C Ldb_NewMsg_FromNet(msg) 
void *msg;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@NewMsg_FromNet(msg) );
}

export_to_C Ldb_NewMsg_FromLocal( msg)
void *msg;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@NewMsg_FromLocal(msg) );
}

export_to_C LdbProcessMsg(msgPtr, localdataPtr)
void *msgPtr, *localdataPtr;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@ProcessMsg(msgPtr, localdataPtr));
}

export_to_C LdbProcessorIdle()
{
	BranchCall(CpvAccess(LdbBocNum), LDB@ProcessorIdle());
}


export_to_C LdbPeriodicCheckInit()
{
}



BranchOffice LDB {
 

DUMMYMSG *statusMsg;

int LdbBoc;
int	numNeighbours;
int	myPE;
int * neighboursList;
LDB_STATUS * statusList;

int neighbourhoodLoad;
int deltaLoad;
int deltaStatus;
BOOLEAN	saturated; /* is the system (estimated to be) saturated? 1:0 */

int SendingIdleMsg;	/* This is a number of requests for chares that have 
			   been sent out */

private RecvUpdateStatus(ldb)
LDB_ELEMENT * ldb;
{
}


/******************************************************************
 * This is an update of the latest status information of oneself  *
 * that one has sent to a neighbour, so as to prevent unnecessary *
 * status messages from being sent                                *
 ******************************************************************/

private SentUpdateStatus(peNo)
int peNo;
{
}

private PrintNodeNeighbours()
{
	int i;

	TRACE(CkPrintf("Node %d: Neighbours: ",myPE));
	for (i=0; i < numNeighbours; i++)
		TRACE(CkPrintf("%d, ", neighboursList[i]));
	TRACE(CkPrintf("\n"));
}

/*************************************
 * LDB Branch Office Chare Functions *
 *************************************/

entry BranchInit : (message DUMMYMSG * dmsg)
{
	int i;

	CpvAccess(LdbBocNum) = LdbBoc = MyBocNum();
	CpvAccess(LDB_ELEM_SIZE) = sizeof(LDB_ELEMENT);
	numPe = CkNumPes();
	myPE = CkMyPe();
	numNeighbours = CkNumNeighbours(myPE);
	SendingIdleMsg = 0;	
}

entry RecvStatus : (message DUMMYMSG * dmsg)
{
	CkFreeMsg(dmsg);
}

/************************************************************
 * If another processor sends this processor a single chare *
 * in response to a  request                                *
 ************************************************************/

public NewMsg_FromNet(msg) 
void *msg;
{
  QsEnqUsrMsg(msg);
}

/***********************************************
 * Locally enqueue all locally produced chares *
 ***********************************************/

public NewMsg_FromLocal(msg)
void * msg;
{
  QsEnqUsrMsg(msg);
}
/****************************************************************
 * Some Ldb messages are sent across - this defines the actions *
 * to be taken on the arrival of such messages                  *
 ****************************************************************/

entry WorkRequest : (message DUMMYMSG *msgPtr)
{
  LDB_ELEMENT *ldb;
  int idlePE;

  ldb = LDB_UPTR(msgPtr);       
  idlePE = ldb->srcPE;	/* give my work to idlePE if I have enough. */
  PrivateCall(Strategy(idlePE));
  CkFreeMsg(msgPtr);		/* After processing the message, free it.*/
}

private NoWorkToGive(pe)
int pe;
{
  DUMMYMSG *msg;
  LDB_ELEMENT *ldb;	      

  msg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG); 
  CkMemError(msg);
  ImmSendMsgBranch(LDB@NegativeReply, msg, pe);
}

private Strategy(idlePE)
int idlePE;
{
  int load = QsMyLoad();
  int NofChares = load/2;        /* Give half of work I have */
  int NofSent;
  CharesMsg *cmsg;
  
  if(SendingIdleMsg)		 /* If I am myself asking for work */
    PrivateCall(NoWorkToGive(idlePE));
  else
    {				 /* Try to send chares */
      TRACE(CkPrintf("Give chares %3d =>%3d,NofChare=%3d\n",
		     CkMyPe(),idlePE,NofChares));
      cmsg = (CharesMsg *)CkAllocMsg(CharesMsg);
      CkMemError(cmsg);
      if(NofSent = CkMakeFreeCharesMessage(NofChares, cmsg))
	ImmSendMsgBranch(LDB@PositiveReply, cmsg, idlePE);
      else	/* If there were no chares to send, */
	{
	  PrivateCall(NoWorkToGive(idlePE)); 
	  /* tell that there are no work to give  */
	  CkFreeMsg(cmsg);
	}
    }
}


entry PositiveReply : (message CharesMsg* msgPtr)
{
  CkQueueFreeCharesMessage(msgPtr);	
  SendingIdleMsg = 0;
  CkFreeMsg(msgPtr);		/* After processing the message, free it.*/
}


entry NegativeReply : (message DUMMYMSG *msgPtr)
{
  LdbProcessorIdle();
  CkFreeMsg(msgPtr);		/* After processing the message, free it.*/
}

public ProcessMsg(msgPtr, localdataPtr)
void *msgPtr, *localdataPtr;
{
    CkPrintf("*** ERROR *** Ldb strategy received unknown ldb type \n");
}

private ImIdle(pe)
int pe;
{
  DUMMYMSG *msg;
  LDB_ELEMENT *ldb;
  msg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG); 
  CkMemError(msg);
  ldb = LDB_UPTR(msg);
  ldb->idlePE = myPE;
  ImmSendMsgBranch(LDB@WorkRequest, msg, pe);
}

public ProcessorIdle()
{
  int pe  = rand() % maxpe;
  
  while (pe == numPe)
    pe = rand() % numPe;

  if (SendingIdleMsg) 
    TRACE(CkPrintf("Idle count is%4d on PE#%3d(#ofPEs=%3d)\n",
		   SendingIdleMsg,CkMyPe(),numPe));
  
  SendingIdleMsg++;
  
  if (SendingIdleMsg <= numPe) /* Control number of idle message */
    PrivateCall(ImIdle(pe));   /* DOES'NT COVER EVERYBODY - FIX THIS
				  TRY ROUND ROBIN ?? */
}

}

}








