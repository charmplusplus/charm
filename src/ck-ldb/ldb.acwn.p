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
 * Revision 2.13  1997-12-22 21:57:17  jyelon
 * Changed LDB initialization scheme.
 *
 * Revision 2.12  1997/07/30 17:31:02  jyelon
 * *** empty log message ***
 *
 * Revision 2.11  1996/02/08 23:52:06  sanjeev
 * added notes
 *
 * Revision 2.10  1995/11/07 00:41:00  sanjeev
 * fixed bug
 *
 * Revision 2.9  1995/11/07  00:03:48  sanjeev
 * ReadInit moved
 *
 * Revision 2.8  1995/11/06  22:07:38  milind
 * Moved ReadInit LdbBocNum from after CreateBoc to in BranchInit
 *
 * Revision 2.7  1995/11/06  21:44:19  milind
 * Fixed BranchInit to call CldPeriodicCheckInit().
 *
 * Revision 2.6  1995/11/06  17:55:09  milind
 * Changed to conform to the definition of functions NewSeedFrom*
 *
 * Revision 2.5  1995/10/27  22:09:16  jyelon
 * Changed Cmi to Ck in all charm files.
 *
 * Revision 2.4  1995/10/27  21:35:54  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.3  1995/08/14  18:39:44  brunner
 * Oops, forgot to take out debugging printfs
 *
 * Revision 2.2  1995/08/14  16:08:02  brunner
 * I corrected the function names to agree with those in use in ldb.rand.p.
 * I also tracked down a bug in FillLDB.  It used to call
 * SentUpdateStatus(destPe) i destPe != CmiNumPe().  Now it
 * calls it only if 0 <= destPe < CmiNumPe.  It doesn't seg fault now,
 * but this may not be a valid solution.
 *
 * Revision 2.1  1995/07/09  17:54:13  narain
 * Cleaned up working version.. interfaces with functions in ldbcfns.c
 *
 * Revision 2.0  1995/06/29  21:19:36  narain
 * *** empty log message ***
 *
 ***************************************************************************/
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * The ACWN Load Balancing Strategy* * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


/******* Following explanatory notes made by Sanjeev on 2/8/96. ***********

LDB_ELEMENT has srcPE, piggybackLoad, and msgHops.
 
FillLDB fills srcPE, piggybackLoad, then calls SentUpdateStatus,
which stores the load sent to the receiving pe.
 
StripLDB calls RecvUpdateStatus which stores the load of sender and
sets leastLoadedPe and leastLoad.
 
NewMsg_FromLocal and NewMsg_FromNet both call Strategy().
 
UpdateMinMaxHops sets the number of hops (less hops if heavy load,
more hops if light loads in nbrhood).
 
Strategy calls LeastLoadPe(), UpdateMinMaxHops(). 
If msgHops < minHops, the msg is sent to leastLoadedPe.
If msgHops > maxHops, msg is processed locally
else if lightly loaded neighbourhood: Send msg to least loaded PE
else if moderate load, send to least-loaded pe if difference is > deltaLoad
else if heavy load, process locally.
 
PeriodicRedist : if neighborhood not heavily loaded,
sends load to least loaded pe if difference > deltaLoad.
 
PeriodicStatus : if my current load is different from last load sent to 
nbr by deltaStatus, send my load again to nbr.
 

***************************************************************************/




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

message {
    int dummy;
} DUMMY_MSG;

typedef struct {
	int	peLoad;
	int	myLoadSent;
	int timeLoadSent;
} LDB_STATUS;

extern int CldAddToken();
extern int CldPickSeedAndSend();

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif


#define MAXINT  0xffff
#define MINHOPS	1
#define MAXHOPS 3
#define LOWMARK 2
#define HIGHMARK 8
#define DELTALOAD 1
#define DELTAREDIST 5
#define DELTASTATUS 3
#define REDIST_UPDATE_INTERVAL	200
#define STATUS_UPDATE_INTERVAL  100

/* define neighbourhood load conditions */
#define LIGHT 1
#define MODERATE 2
#define HEAVY 3

export_to_C CldModuleInit()
{
  CldCommonInit();
}

export_to_C CldGetLdbSize()
{
       return sizeof(LDB_ELEMENT);
}


export_to_C CldCreateBoc()
{
  DUMMYMSG *msg;
  msg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG);	
  LdbBocNum = CreateBoc(LDB, LDB@BranchInit, msg);
  ReadInit(LdbBocNum);
}

export_to_C CldFillLdb(destPe, ldb)
int destPe;
void *ldb;
{
	BranchCall(ReadValue(LdbBocNum), LDB@FillLDB(destPe, (LDB_ELEMENT *)ldb));
}

export_to_C CldStripLdb(ldb)
void *ldb;
{
	BranchCall(ReadValue(LdbBocNum), LDB@StripLDB((LDB_ELEMENT *)ldb));
}


export_to_C CldNewSeedFromNet(msgst, ldb, sendfn,queuing,priolen,prioptr) 
void *msgst, *ldb;
void (*sendfn)();
unsigned int queuing, priolen, *prioptr;
{
	BranchCall(ReadValue(LdbBocNum), LDB@NewMsg_FromNet(msgst, ldb, sendfn,queuing,priolen,prioptr));
}

export_to_C CldNewSeedFromLocal(msgst, ldb, sendfn,queuing,priolen,prioptr)
void *msgst, *ldb;
void (*sendfn)();
unsigned int queuing, priolen, *prioptr;
{
	BranchCall(ReadValue(LdbBocNum), LDB@NewMsg_FromLocal(msgst, ldb, sendfn,queuing,priolen,prioptr));
}

export_to_C CldProcessMsg(msgPtr, localdataPtr)
void *msgPtr, *localdataPtr;
{
	BranchCall(ReadValue(LdbBocNum), LDB@ProcessMsg(msgPtr, localdataPtr));
}

export_to_C CldProcessorIdle()
{
	BranchCall(ReadValue(LdbBocNum), LDB@ProcessorIdle());
}

export_to_C void CldPeriodicRedist()
{
	BranchCall(ReadValue(LdbBocNum), LDB@PeriodicRedist());
}

export_to_C void CldPeriodicStatus()
{
	BranchCall(ReadValue(LdbBocNum), LDB@PeriodicStatus());
}

/**
export_to_C void CldPeriodicCheckInit()
{
	BranchCall(ReadValue(LdbBocNum), LDB@PeriodicCheckInit());
}
**/

BranchOffice LDB {

/* ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... */

int numNeighbours;
int * neighboursList;
int lastPeZeroLoadIndex;
int numPe;	
int LdbBoc;
int myPE;
LDB_STATUS * statusList;
DUMMY_MSG *statusMsg;
int leastLoadedPe;
int leastLoad;
int neighbourhoodLoad;
int minHops, maxHops;
int lowMark, highMark;
int deltaLoad;
int deltaRedist;
int deltaStatus;
int	saturated; /* is the system (estimated to be) saturated? 1:0 */

/* ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... ..... */

private SentUpdateStatus(peNum)
int peNum;
{
    int i;

    if (peNum >=0)
    {
    	i = CmiNeighboursIndex(myPE, peNum);
    	if (i > -1)
    	{
	 	statusList[i].myLoadSent = CldMyLoad();
	 	/*statusList[i].timeLoadSent = CmiTimer();*/
    	}
    }
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


private Strategy(msg, ldbptr, sendfn,queuing,priolen,prioptr)
void *msg, *ldbptr;
void (*sendfn)();
unsigned int queuing,priolen,*prioptr;
{
    LDB_ELEMENT * ldb;
    int MyPeLoad;

    ldb = (LDB_ELEMENT *)ldbptr;
    ldb->srcPE = myPE;
    ldb->msgHops++;

    leastLoad = MAXINT;
    PrivateCall(LeastLoadPe());
    PrivateCall(UpdateMinMaxHops());
    /* Msg MUST travel minimum hops to reduce Horizon effect */
    if (ldb->msgHops < minHops)
    {
	PrivateCall(SentUpdateStatus(leastLoadedPe));
	(*sendfn)(msg, leastLoadedPe);
    }
    /* Msg has travelled maxHops, time to enqueue and process it on this node */
    else if (ldb->msgHops >= maxHops)
    {

	TRACE(CkPrintf("LdbNodeStrategy:Node %d: Hops>=maxHops(%d), Enqueue Msg 0x%x\n", myPE, maxHops, msg));
	CldAddToken(msg, sendfn,queuing,priolen,prioptr);
    }
    /* Msg has travelled between minHops and maxHops */
    else /* ( (minHops <= ldb->msgHops) && (ldb->msgHops < maxHops) ) */
      {
	switch (neighbourhoodLoad)
	  {
	    /* Lightly loaded neighbourhood: Send it to least loaded PE */
	  case LIGHT:
	    PrivateCall(SentUpdateStatus(leastLoadedPe));
	    (*sendfn)(msg, leastLoadedPe);
	    break;

	  case MODERATE:
	    MyPeLoad = CldMyLoad();
	    if (MyPeLoad - leastLoad > deltaLoad)
	      {
		/* Update the load status for the least loaded PE */
		PrivateCall(SentUpdateStatus(leastLoadedPe));
		(*sendfn)(msg, leastLoadedPe);
	      }
	    else
	      CldAddToken(msg, sendfn,queuing,priolen,prioptr);
	    break;

    	    /*  Heavily Loaded Neighbourhood:
       	        maxHops = 0, such that NewChares are Enqueued at local PE.  */
	  case HEAVY:
	    CldAddToken(msg, sendfn,queuing,priolen,prioptr);
	    break;
	}
    }
}


private RecvUpdateStatus(ldb)
LDB_ELEMENT * ldb;
{
    int i;

    i = CmiNeighboursIndex(myPE, ldb->srcPE);
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
	LdbBocNum = LdbBoc = MyBocNum();
        BranchCall(LdbBoc, LDB@PeriodicCheckInit());
	/* CldPeriodicCheckInit(); */
	numPe = CkNumPes();
	myPE = CkMyPe();
	numNeighbours = CmiNumNeighbours(myPE);
	lastPeZeroLoadIndex = 0;

	if (numPe > 1)
	{
	    neighboursList = (int *) CkAlloc( numNeighbours * sizeof(int) );
/*            CkMemError(neighboursList); */
	    CmiGetNodeNeighbours(myPE, neighboursList );
	    statusList = (LDB_STATUS *) CkAlloc(numNeighbours * sizeof(LDB_STATUS)); 
	    
/*            CkMemError(statusList); */
	    PrivateCall(PrintNodeNeighbours());

	    for (i=0; i < numNeighbours; i++)
	    {
	        statusList[i].peLoad = 0;
	        statusList[i].myLoadSent = 0;
	        statusList[i].timeLoadSent = 0;
	    }


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


	    /* send Initial Status update msgs to all neighbours */
	    for (i=0; i < numNeighbours; i++)
	    {
	      statusList[i].myLoadSent = 0;
	      PrivateCall(SentUpdateStatus(neighboursList[i]));
	    
	      statusMsg = (DUMMY_MSG *) CkAllocMsg(DUMMY_MSG);
/*	      CkMemError(statusMsg);  */

/*	      ldb = LDB_UPTR(statusMsg); 
	      ldb->srcPE = myPE;
	      ldb->msgHops = 100;
	      ldb->piggybackLoad =0;
*/
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

entry RecvStatus : (message DUMMY_MSG *dmsg)
{
	CkFreeMsg(dmsg);
}

/* Load Balance messages received at the Node from the Network */
public StripLDB(ldb)
LDB_ELEMENT *ldb;
{
    if ((numPe > 1) && (ldb->srcPE != myPE) 
        &&  (ldb->srcPE != CkNumPes()))
    		PrivateCall(RecvUpdateStatus(ldb));
}


public NewMsg_FromNet(msgst, ldb, sendfn,queuing,priolen,prioptr)
void *msgst, *ldb;
void (*sendfn)();
unsigned int queuing, priolen, *prioptr;
{
	PrivateCall(Strategy(msgst, ldb, sendfn,queuing,priolen,prioptr));
}


public NewMsg_FromLocal(msgst, ldbptr, sendfn,queuing,priolen,prioptr)
void *msgst, *ldbptr;
void (*sendfn)();
unsigned int queuing, priolen, *prioptr;
{
    LDB_ELEMENT * ldb;

    ldb = (LDB_ELEMENT *)ldbptr;
    ldb->msgHops = 0;  /* This stmt and the previous two moved here
 			  on 5/22/93, by Sanjay   */
    PrivateCall(Strategy(msgst, ldbptr, sendfn,queuing,priolen,prioptr));
}

/* 
   This performs the ACWN strategy for Load Balancing.
   Load is balanced only if the number of PEs are > 1.
*/
public FillLDB(destPe, ldb)
int destPe;
LDB_ELEMENT *ldb;
{

    ldb->srcPE = myPE;
    ldb->piggybackLoad = CldMyLoad();
    /* ldb->msgHops = 0; shouldn't be here. Moved to NewChare From Local
	on 5/22/93 by Sanjay */
    if ((destPe >= 0) && (destPe < CkNumPes()))
      PrivateCall(SentUpdateStatus(destPe));
}


public void PeriodicRedist(bocNum)
ChareNumType bocNum;
{
    int i;
    int MyPeLoad;
    void *msg;

 if (maxHops > 0) /* if Neighbourhood NOT in a HEAVY state */
 {
    MyPeLoad = CldMyLoad();
    leastLoad = MAXINT;
    PrivateCall(LeastLoadPe());

    if ( (MyPeLoad - leastLoad) > deltaRedist )
	if (CldPickSeedAndSend(leastLoadedPe))
	    PrivateCall(SentUpdateStatus(leastLoadedPe));
 }

    /* call LdbPeriodicRedist() AGAIN after REDIST_UPDATE_INTERVAL time */
    CallBocAfter(CldPeriodicRedist, LdbBoc, REDIST_UPDATE_INTERVAL);
}


public void PeriodicStatus(bocNum)
ChareNumType bocNum;
{
    int i;
    int MyPeLoad;
    LDB_ELEMENT * ldb;

    MyPeLoad = CldMyLoad();

    for (i=0; i < numNeighbours; i++)
        if ( abs(MyPeLoad - statusList[i].myLoadSent) > deltaStatus )
	{
	  statusList[i].myLoadSent = MyPeLoad;
	  
	  /* fill the LdbBlock with load status */
	  statusMsg = (DUMMY_MSG *)CkAllocMsg(DUMMY_MSG);
/* 	  CkMemError(statusMsg); */

/*	  ldb = LDB_UPTR(statusMsg);  Could'nt this be done by the fillblk? 
	  ldb->piggybackLoad = MyPeLoad;
*/
	  PrivateCall(SentUpdateStatus(neighboursList[i]));
	  ImmSendMsgBranch(RecvStatus, statusMsg, neighboursList[i]) ;  
	}

    /* call LdbPeriodicStatus() AGAIN after STATUS_UPDATE_INTERVAL time */
    CallBocAfter(CldPeriodicStatus, LdbBoc, STATUS_UPDATE_INTERVAL);
}





public ProcessorIdle()
{
}

public void PeriodicCheckInit()
{
 if (numPe > 1)
 {  
   CallBocAfter(CldPeriodicRedist, ReadValue(LdbBocNum), REDIST_UPDATE_INTERVAL);
   CallBocAfter(CldPeriodicStatus, ReadValue(LdbBocNum), STATUS_UPDATE_INTERVAL); 
 }
}


}

}
