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
 * Revision 2.3  1997-12-22 21:57:18  jyelon
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
/* * * * * * * * The COMB Load Balancing Strategy* * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#define ceil2(x) ((x != (int) (x)) ? (int) (x+1) : (int)(x) )

/************ Thresholds and Fan factor **********/

#define LowThresh   2
#define SendThresh  3
#define FF(numNeigh) ceil2(numNeigh/2.0)  /* Function to determine FanFactor */

/* Allocation categories */
#define NEIGHBOURHOOD 0
#define LOCAL         1
#define RAND          2

/* Redistribution categories */
#define RECV 1

 

module ldb {
#include "ldb.h"

message {
    int dummy;
} DUMMYMSG;

typedef struct {
	PeNumType srcPE;
	int	piggybackLoad;
	int	msgHops;
} LDB_ELEMENT;

typedef struct {
	int	peLoad;
	int	myLoadSent;
	int timeLoadSent;
} LDB_STATUS;


#define MAXINT  0xffff
#define MINHOPS	1
#define MAXHOPS 3
#define LOWMARK 2
#define HIGHMARK 8
#define DELTALOAD 1
#define DELTAREDIST 5
#define DELTASTATUS 3
#define REDIST_UPDATE_INTERVAL	1000
#define STATUS_UPDATE_INTERVAL  100
#define WORKCHECK_INTERVAL  50

/* define neighbourhood load conditions */
#define LIGHT 1
#define MODERATE 2
#define HEAVY 3

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

export_to_C Ldb_NewMsg_FromLocal(msg)
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

export_to_C void LdbPeriodicRedist()
{
	BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicRedist());
}

export_to_C void LdbPeriodicStatus()
{
	BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicStatus());
}

export_to_C void LdbPeriodicCheckInit()
{
  BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicCheckInit());
}

export_to_C LdbPeriodicLowCheck()
{
  BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicLowCheck_norepeat());
  CallBocAfter(LdbPeriodicLowCheck, CpvAccess(LdbBocNum), WORKCHECK_INTERVAL);
}

export_to_C LdbSetAllocRedist(alloc, redist)
     int alloc, redist;
{
  BranchCall(CpvAccess(LdbBocNum), LDB@SetAllocRedist(alloc, redist));
}

export_to_C LdbSetRedist(redist)
     int redist;
{
  BranchCall(CpvAccess(LdbBocNum), LDB@SetRedist(redist));
}

export_to_C LdbSetAlloc(alloc)
     int alloc;
{
  BranchCall(CpvAccess(LdbBocNum), LDB@SetAlloc(alloc));
}

BranchOffice LDB {

/* ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... */

int	numNeighbours;
int * neighboursList;
int lastPeZeroLoadIndex;
int LdbBoc;
int myPE;
LDB_STATUS * statusList;
DUMMYMSG *statusMsg;
int leastLoadedPe;
int leastLoad;
int neighbourhoodLoad;
int minHops, maxHops;
int lowMark, highMark;
int deltaLoad;
int deltaRedist;
int deltaStatus;
BOOLEAN	saturated; /* is the system (estimated to be) saturated? 1:0 */
int redist_start;
int FanFactor;

int ALLOCATION, REDISTRIBUTION;

/* ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... */


/******************************************************************
 * This is an update of the latest status information of oneself  *
 * that one has sent to a neighbour, so as to prevent unnecessary *
 * status messages from being sent                                *
 ******************************************************************/

public SetAlloc(alloc)
  int alloc;
{
  ALLOCATION = alloc;
}
    
public SetRedist(redist)
  int redist;
{
  REDISTRIBUTION = redist;
}
    
public SetAllocRedist(alloc, redist)
  int alloc, redist;
{
  ALLOCATION = alloc;
  REDISTRIBUTION = redist;
}
    
private SentUpdateStatus(peNum)
int peNum;
{
    int i;

    if (peNum >=0)
    {
    	i = McNeighboursIndex(myPE, peNum);
    	if (i > -1)
    	{
	 	statusList[i].myLoadSent = QsMyLoad();
	 	/*statusList[i].timeLoadSent = McTimer();*/
    	}
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

private LeastLoadPe()
{
    int i;

    /* get leastloaded PE and the least Load */
    for (i=0; i < numNeighbours; i++)
        if ( statusList[i].peLoad < leastLoad)
	{
	    leastLoad = statusList[i].peLoad;
	    leastLoadedPe = neighboursList[i]; 
	}

    /* Random selection of a Neighbour if leastLoad == 0 */
    /*
    if (leastLoad == 0)
    {
    	i = rand() % numNeighbours;
	leastLoadedPe = neighboursList[i]; 
    }
    */

    /* RoundRobin selection of a Neighbour if leastLoad == 0 */
    if (leastLoad == 0)
    {
	lastPeZeroLoadIndex = (lastPeZeroLoadIndex + 1) % numNeighbours;
	leastLoadedPe = neighboursList[lastPeZeroLoadIndex]; 
    }

}

private UpdateMinMaxHops()
{
    	if (leastLoad < lowMark)
    	{
		minHops = MINHOPS; maxHops = MAXHOPS;
		neighbourhoodLoad = LIGHT;
    	}
    	else if (leastLoad > highMark)
    	{
		minHops = 0; maxHops = 0;
		neighbourhoodLoad = HEAVY;
    	}
    	else /* (lowMark <= leastLoad && leastLoad <= highMark)*/
    	{
		minHops = 0; maxHops = MAXHOPS;
		neighbourhoodLoad = MODERATE;
    	}
}




private RecvUpdateStatus(ldb)
LDB_ELEMENT * ldb;
{
  int i;
  
  i = McNeighboursIndex(myPE, ldb->srcPE);
  if (i > -1)
    {
      statusList[i].peLoad = ldb->piggybackLoad;
      
      if ( statusList[i].peLoad < leastLoad)
	{
	  leastLoad = statusList[i].peLoad;
	  leastLoadedPe = neighboursList[i]; 
	}
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

public ProcessMsg(msgPtr, localdataPtr)
void *msgPtr, *localdataPtr;
{
	CkFreeMsg(msgPtr);
}

/* LDB Branch Office Chare Functions */

entry BranchInit : (message DUMMYMSG * dmsg)
{
  int i;
  LDB_ELEMENT *ldb;
  
  TRACE(CkPrintf("Enter Node LdbInit()\n"));
  CpvAccess(LDB_ELEM_SIZE) = sizeof(LDB_ELEMENT);
  CpvAccess(LdbBocNum) = LdbBoc = MyBocNum();
  numPe = CkNumPes();
  myPE = CkMyPe();
  numNeighbours = CkNumNeighbours(myPE);
  
  lastPeZeroLoadIndex = 0;
  ALLOCATION = LOCAL;
  REDISTRIBUTION = RECV;
  
  FanFactor = FF(numNeighbours);    /* Value for Fan Factor */
  
  if (numPe > 1)
    {
      neighboursList = (int *) CkAlloc( numNeighbours * sizeof(int) );
      CkMemError(neighboursList);
      McGetNodeNeighbours(myPE, neighboursList );
      statusList = (LDB_STATUS *) CkAlloc(numNeighbours * sizeof(LDB_STATUS)); 
      CkMemError(statusList);
      PrivateCall(PrintNodeNeighbours());
      
      leastLoadedPe = neighboursList[0];
      leastLoad = MAXINT;
      neighbourhoodLoad = LIGHT;
      minHops = MINHOPS;
      maxHops = MAXHOPS;
      lowMark = LOWMARK;
      highMark = HIGHMARK;
      deltaLoad = DELTALOAD;
      deltaRedist = DELTAREDIST;
      deltaStatus = DELTASTATUS;
      saturated = FALSE;
      redist_start = 0;
      
      /* send Initial Status update msgs to all neighbours */
      for (i=0; i < numNeighbours; i++)
	{
	  statusList[i].peLoad = 0;
	  statusList[i].myLoadSent = 0;
	  statusList[i].timeLoadSent = 0;
	  PrivateCall(SentUpdateStatus(neighboursList[i]));
	  
	  statusMsg = (DUMMYMSG *) CkAllocMsg(DUMMYMSG);
	  CkMemError(statusMsg);
	  ldb = LDB_UPTR(statusMsg);
	  ldb->srcPE = myPE;
	  ldb->msgHops = 100;
	  ldb->piggybackLoad =0;
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

entry RecvStatus : (message DUMMYMSG *dmsg)
{
  CkFreeMsg(dmsg);
}

/* Load Balance messages received at the Node from the Network */
public StripLDB(ldb)
     LDB_ELEMENT *ldb;
{
  if ((numPe > 1) && (ldb->srcPE != myPE) 
      &&  (ldb->srcPE != McHostPeNum()))
    PrivateCall(RecvUpdateStatus(ldb));
}

private rand_strategy(msg) /* From rand */
void *msg;
{
    int pe = rand() % CkNumPes();
    if (pe == myPE)
      QsEnqUsrMsg(msg);
    else
      SEND_TO(msg, pe);	
}

private acwn_strategy(msg) /* from neighbourhood */
void *msg;
{
    LDB_ELEMENT * ldb;
    int MyPeLoad;

    ldb = LDB_UPTR(msg);
    ldb->srcPE = myPE;
    ldb->msgHops++;

    leastLoad = MAXINT;
    PrivateCall(LeastLoadPe());
    PrivateCall(UpdateMinMaxHops());
    /* Msg MUST travel minimum hops to reduce Horizon effect */
    if (ldb->msgHops < minHops)
    {
	PrivateCall(SentUpdateStatus(leastLoadedPe));
	SEND_TO(msg, leastLoadedPe);
    }
    /* Msg has travelled maxHops, time to enqueue and process it on this node */
    else if (ldb->msgHops >= maxHops)
    {

	TRACE(CkPrintf("LdbNodeStrategy:Node %d: Hops>=maxHops(%d), Enqueue Msg 0x%x\n", myPE, maxHops, msg));
	QsEnqUsrMsg(msg);
    }
    /* Msg has travelled between minHops and maxHops */
    else /* ( (minHops <= ldb->msgHops) && (ldb->msgHops < maxHops) ) */
      {
	switch (neighbourhoodLoad)
	  {
	    /* Lightly loaded neighbourhood: Send it to least loaded PE */
	  case LIGHT:
	    PrivateCall(SentUpdateStatus(leastLoadedPe));
	    SEND_TO(msg, leastLoadedPe);
	    break;

	  case MODERATE:
	    MyPeLoad = QsMyLoad();
	    if (MyPeLoad - leastLoad > deltaLoad)
	      {
		/* Update the load status for the least loaded PE */
		PrivateCall(SentUpdateStatus(leastLoadedPe));
		SEND_TO(msg, leastLoadedPe);
	      }
	    else
	      QsEnqUsrMsg(msg);
	    break;

    	    /*  Heavily Loaded Neighbourhood:
       	        maxHops = 0, such that NewChares are Enqueued at local PE.  */
	  case HEAVY:
	    QsEnqUsrMsg(msg);
	    break;
	}
    }
}


public NewMsg_FromNet(msg)
     void *msg;
{
  switch(ALLOCATION)
    {
    case NEIGHBOURHOOD :
      PrivateCall(acwn_strategy(msg));
      break;
    case LOCAL :
    case RAND :
      QsEnqUsrMsg(msg);
      break;
    default : 
      CkPrintf("Unknown allocation choice\n");
    };
}



public NewMsg_FromLocal(msg)
     void *msg;
{
  LDB_ELEMENT * ldb;
  
  switch(ALLOCATION)
    {
    case NEIGHBOURHOOD :
      ldb = LDB_UPTR(msg);
      ldb->msgHops = 0;  /* This stmt and the previous two moved here
			    on 5/22/93, by Sanjay   */
      PrivateCall(acwn_strategy(msg));
      break;
    case LOCAL :
      QsEnqUsrMsg(msg);
      break;
    case RAND :
      PrivateCall(rand_strategy(msg));
      break;
    default : 
      CkPrintf("Unknown allocation choice\n");
    };
}

public FillLDB(ldb)
     LDB_ELEMENT *ldb;
{
  int destpe;
  
  ldb->srcPE = myPE;
  ldb->piggybackLoad = QsMyLoad();
  /* ldb->msgHops = 0; shouldn't be here. Moved to NewChare From Local
     on 5/22/93 by Sanjay */
  destpe = DestPE_LDB(ldb);
  if (destpe != McHostPeNum())
    PrivateCall(SentUpdateStatus(destpe));
}


public void PeriodicRedist(bocNum)
     ChareNumType bocNum;
{
  int i, j;
  int MyPeLoad, total_load, total_low_load, excess_load, send_here, did_send;
  int below_avg;
  float avg_load;
  void *msg;
  CharesMsg *cmsg;
  
  if(REDISTRIBUTION == NEIGHBOURHOOD)
    {
      if (maxHops > 0) /* if Neighbourhood NOT in a HEAVY state */
	{
	  MyPeLoad = QsMyLoad();
	  for(i = 0, total_load = 0; i < numNeighbours;i++)
	    total_load += statusList[i].peLoad;
	  total_load += MyPeLoad;
	  avg_load = (float)total_load / (numNeighbours + 1);
	  if((excess_load = MyPeLoad - avg_load) > DELTALOAD)
	    {
	      for(i = 0, total_low_load = 0; i < numNeighbours;i++)
		if(statusList[i].peLoad < avg_load)
		  total_low_load += statusList[i].peLoad;
	      for(i = 0, j = redist_start; i<numNeighbours;
		  i++, j = (j+1) % numNeighbours)
		if((below_avg = statusList[j].peLoad - avg_load) < 0)
		  {
		    if(total_low_load)
		      send_here = below_avg / total_low_load * excess_load;
		    else
		      send_here = excess_load;
		    if(send_here)
		      {
			cmsg = (CharesMsg *)CkAllocMsg(CharesMsg);
			CkMemError(cmsg);
			if(did_send = CkMakeFreeCharesMessage(send_here, cmsg))
			  {
			    ImmSendMsgBranch(ReceiveChares, cmsg,
					  neighboursList[j]);
			    statusList[j].myLoadSent = MyPeLoad; 
			    /* I could call SentUpdateStatus with the PE num, 
			       but that would mean an unnecessary back 
			       conversion */
			  }
			else
			  CkFreeMsg(cmsg);
		      }
		  }
	      redist_start = (redist_start + 1) % numNeighbours;
	    }
	}
    }
  /* call LdbPeriodicRedist() AGAIN after REDIST_UPDATE_INTERVAL time */
  CallBocAfter(LdbPeriodicRedist, LdbBoc, REDIST_UPDATE_INTERVAL);
}

entry ReceiveChares : (message CharesMsg* msgPtr)
{
  CkQueueFreeCharesMessage(msgPtr);		
  CkFreeMsg(msgPtr);		/* After processing the message, free it.*/
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
        if ( abs(MyPeLoad - statusList[i].myLoadSent) > deltaStatus )
	{
	  statusList[i].myLoadSent = MyPeLoad;
	  
	  /* fill the LdbBlock with load status */
	  statusMsg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG);
	  CkMemError(statusMsg);
	  ldb = LDB_UPTR(statusMsg);
	  ldb->piggybackLoad = MyPeLoad;
	  PrivateCall(SentUpdateStatus(neighboursList[i]));
	  ImmSendMsgBranch(RecvStatus, statusMsg, neighboursList[i]) ;  
	}

    /* call LdbPeriodicStatus() AGAIN after STATUS_UPDATE_INTERVAL time */
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



public ProcessorIdle() 
{
  if(REDISTRIBUTION == RECV)
    BranchCall(LdbBoc, LDB@PeriodicLowCheck_norepeat());
}

public void PeriodicLowCheck_norepeat()
{
  int load = QsMyLoad();        
  int j, i;
  int mynum = myPE;
  
  if(REDISTRIBUTION == RECV)
    if(load < LowThresh)
      for( i = 0; i < numNeighbours; i++)
	if(statusList[i].peLoad >= SendThresh)
	  {
	    j = neighboursList[i];
	    PrivateCall(SendRequest(j));
	  }
}


public void PeriodicCheckInit()
{
  if (numPe > 1)
    {  
      CallBocAfter(LdbPeriodicRedist, CpvAccess(LdbBocNum), REDIST_UPDATE_INTERVAL);
      CallBocAfter(LdbPeriodicStatus, CpvAccess(LdbBocNum), STATUS_UPDATE_INTERVAL); 
      CallBocAfter(LdbPeriodicLowCheck,LdbBoc, WORKCHECK_INTERVAL);
    }
  
}
}

}










