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
 * Revision 2.3  1997-12-22 21:57:24  jyelon
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
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * The RECV Load Balancing Strategy* * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#define ceil2(x) ((x != (int) (x)) ? (int) (x+1) : (int)(x) )

/************ Thresholds and Fan factor **********/

#define LowThresh   2
#define SendThresh  3
#define FF(numNeigh) ceil2(numNeigh/2.0)  /* Function to determine FanFactor */

#define MAXINT  0xffff
#define DELTALOAD 1
#define DELTASTATUS 3
#define STATUS_UPDATE_INTERVAL  50
#define WORKCHECK_INTERVAL  50
 
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
	int piggybackLoad;
	int srcPE;
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
  LDB_ELEMENT *ldb;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@FillLDB(ldb));
}

export_to_C LdbStripLDB(ldb)
     LDB_ELEMENT *ldb;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@StripLDB(ldb));
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
	BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicCheckInit());
}

export_to_C LdbPeriodicStatus()
{
	BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicStatus());
}

export_to_C LdbPeriodicLowCheck()
{
  BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicLowCheck_norepeat());
  CallBocAfter(LdbPeriodicLowCheck, CpvAccess(LdbBocNum), WORKCHECK_INTERVAL);
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


int FanFactor;



private RecvUpdateStatus(ldb)
LDB_ELEMENT * ldb;
{
  int i;
  
  i = McNeighboursIndex(myPE, ldb->srcPE);
  if (i > -1)
      statusList[i].peLoad = ldb->piggybackLoad;
}


/******************************************************************
 * This is an update of the latest status information of oneself  *
 * that one has sent to a neighbour, so as to prevent unnecessary *
 * status messages from being sent                                *
 ******************************************************************/

private SentUpdateStatus(peNo)
int peNo;
{
  int i;
  
  if (peNo >=0)
    {
      i = McNeighboursIndex(myPE, peNo);
      if (i > -1)
	statusList[i].myLoadSent = QsMyLoad();
    }
}
/**************************************************************
 * Following the sending of chares to a requesting processor, *
 * its status information is locally updated                  *
 **************************************************************/

private UpdateStatus(peNo, sent)
     int peNo, sent;
{
  int i;
  
  if (peNo >=0)
    {
      i = McNeighboursIndex(myPE, peNo);
      if (i > -1)
	statusList[i].peLoad += sent;
    }
}

private PrintNodeNeighbours()
{
	int i;

	TRACE(CkPrintf("Node %d: Neighbours: ",myPE));
	for (i=0; i < numNeighbours; i++)
		TRACE(CkPrintf("%d, ", neighboursList[i]));
	TRACE(CkPrintf("\n"));
}



private NegReply(pe)
int pe;
{
  DUMMYMSG *msg;

  msg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG); 
  CkMemError(msg);
  ImmSendMsgBranch(NegativeReply, msg, pe); 
}

private SendRequest(pe)
int pe;
{
  DUMMYMSG *msg;
  msg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG); 
  CkMemError(msg);
  ImmSendMsgBranch(WorkRequest, msg, pe);
}

/***********************************************************
 * Respond to request for chares, send chares if possible, *
 * else send a negative reply                              *
 ***********************************************************/

private SendChares(idlePE)
     int idlePE;
{
  CharesMsg *msg;
  int can_send, actually_sent, myLoad, did_send;
  
  myLoad = QsMyLoad();

  /* How many chares can I send ? */
  if (myLoad >= SendThresh)
    can_send =  ceil2((float)(myLoad - LowThresh)/FanFactor);
  else
    can_send = 0;
  
  if(can_send)
    {
      msg = (CharesMsg *) CkAllocMsg(CharesMsg);
      CkMemError(msg);
      if(did_send = CkMakeFreeCharesMessage(can_send, msg))
	{ 
	  ImmSendMsgBranch(PositiveReply, msg, idlePE);
	  PrivateCall(UpdateStatus(idlePE, did_send));
	}
      else
	CkFreeMsg(msg);
    }
  if(can_send == 0 || did_send == 0)
    PrivateCall(NegReply(idlePE));
}



/*************************************
 * LDB Branch Office Chare Functions *
 *************************************/

entry BranchInit : (message DUMMYMSG * dmsg)
{
	int i;

	LDB_ELEMENT *ldb;

	TRACE(CkPrintf("Enter Node LdbInit()\n"));
	
	CpvAccess(LdbBocNum) = LdbBoc = MyBocNum();
	
	CpvAccess(LDB_ELEM_SIZE) = sizeof(LDB_ELEMENT);

	numPe = CkNumPes();
	myPE = CkMyPe();
	numNeighbours = CkNumNeighbours(myPE);

	FanFactor = FF(numNeighbours);    /* Value for Fan Factor */

	if (numPe > 1)
	{
	    neighboursList = (int *) CkAlloc( numNeighbours * sizeof(int) );
            CkMemError(neighboursList);
	    McGetNodeNeighbours(myPE, neighboursList );
	    statusList = (LDB_STATUS *) CkAlloc(numNeighbours * sizeof(LDB_STATUS)); 
            CkMemError(statusList);
	    PrivateCall(PrintNodeNeighbours());

	    deltaLoad = DELTALOAD;
	    deltaStatus = DELTASTATUS;
	    saturated = FALSE;
	    
	    for (i=0; i < numNeighbours; i++)
	    {
	        statusList[i].peLoad = 0;
	        statusList[i].myLoadSent = 0;
	    
	        statusMsg = (DUMMYMSG *) CkAllocMsg(DUMMYMSG);
                CkMemError(statusMsg);

	    
	    /* send Initial Status update msgs to all neighbours */
		
	    	/* fill the LdbBlock with load status */
    	    	ldb = LDB_UPTR(statusMsg);
    	        ldb->srcPE = myPE;
    	        ldb->piggybackLoad = 0;
		ImmSendMsgBranch(RecvStatus, statusMsg, neighboursList[i]); 
	      }
	  }
	else /* for a single processor */
	  {
	    neighboursList = NULL;
	    statusList = NULL;
	  }

	TRACE(CkPrintf("Node LdbInit() Done: numPe %d, numNeighbours %d\n",
		numPe, numNeighbours));
}

entry RecvStatus : (message DUMMYMSG * dmsg)
{
	CkFreeMsg(dmsg);
}
/**************************************************************
 * Retrieve ldb information from the ldb element in a message *
 **************************************************************/

public StripLDB(ldb)
LDB_ELEMENT * ldb;
{
    if ((numPe > 1) && (ldb->srcPE != myPE) 
        &&  (ldb->srcPE != McHostPeNum()))
    		PrivateCall(RecvUpdateStatus(ldb));
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


/*********************************************************
 * Fill up the ldb element with the required information *
 *********************************************************/

public FillLDB(ldb)
LDB_ELEMENT *ldb;
{
  ldb->srcPE = myPE;
  ldb->piggybackLoad = QsMyLoad();
  PrivateCall(SentUpdateStatus(DestPE_LDB(ldb)));
}

/**********************************************************
 * Periodic broadcast of status information to neighbours *
 **********************************************************/

public void PeriodicStatus(bocNum)
ChareNumType bocNum;
{
    int i;
    int MyPeLoad;
    LDB_ELEMENT * ldb;

    MyPeLoad = QsMyLoad();

    for (i=0; i < numNeighbours; i++)
        if( abs(MyPeLoad - statusList[i].myLoadSent) > deltaStatus )
	{
	    statusList[i].myLoadSent = MyPeLoad;

	    /* fill the LdbBlock with load status */
	    statusMsg = (DUMMYMSG *) CkAllocMsg(DUMMYMSG);
	    CkMemError(statusMsg);
	    ImmSendMsgBranch(RecvStatus, statusMsg, neighboursList[i]); 
	}
    CallBocAfter(LdbPeriodicStatus, LdbBoc, STATUS_UPDATE_INTERVAL);
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
    PrivateCall(SendChares(idlePE));

  CkFreeMsg(msgPtr);		/* After processing the message, free it.*/
}

entry PositiveReply : (message CharesMsg* msgPtr)
{
  CkQueueFreeCharesMessage(msgPtr);		
  CkFreeMsg(msgPtr);		/* After processing the message, free it.*/
}


entry NegativeReply : (message DUMMYMSG *msgPtr)
{
  CkFreeMsg(msgPtr);		/* After processing the message, free it.*/
}

public ProcessMsg(msgPtr, localdataPtr)
void *msgPtr, *localdataPtr;
{
    CkPrintf("*** ERROR *** Ldb strategy received unknown ldb type \n");
}


/* ###If a processor is idle, do the same as a lowcheck with no repeat */

public ProcessorIdle()
{
}


public void PeriodicLowCheck_norepeat()
{
  int load = QsMyLoad();           /* Is this right ? */
  int j, i;
  int mynum = myPE;
  
  if(load < LowThresh)
    for( i = 0; i < numNeighbours; i++)
      if(statusList[i].peLoad >= SendThresh)
	{
	  j = neighboursList[i];
	  PrivateCall(SendRequest(j));
	}
}


/************************************************************
 * These are the functions that are called periodically for *
 * Load balancing purposes                                  *
 ************************************************************/
public PeriodicCheckInit()
{
   if (numPe > 1)
    {  
	CallBocAfter(LdbPeriodicStatus, LdbBoc, STATUS_UPDATE_INTERVAL); 
	CallBocAfter(LdbPeriodicLowCheck,LdbBoc, WORKCHECK_INTERVAL);
    }
}

}

}








