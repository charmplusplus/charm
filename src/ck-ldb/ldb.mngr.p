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
 * Revision 2.7  1997-07-30 17:31:03  jyelon
 * *** empty log message ***
 *
 * Revision 2.6  1996/02/08 23:33:36  sanjeev
 * added documentation on how it works
 *
 * Revision 2.5  1995/11/06 17:55:09  milind
 * Changed to conform to the definition of functions NewSeedFrom*
 *
 * Revision 2.4  1995/10/27  22:09:16  jyelon
 * Changed Cmi to Ck in all charm files.
 *
 * Revision 2.3  1995/10/27  21:35:54  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.2  1995/08/21  15:10:34  brunner
 * Made changes to accomodate naming scheme and ldbcfnc.c.
 * The code compiles and seems to run ok, but I have not verified
 * whether any load balancing is actually being done.
 *
 * Revision 2.1  1995/07/10  07:04:16  narain
 * Temp version.. compiles and links, but gets stuck in a loop during program.
 *
 * Revision 2.0  1995/06/29  21:19:36  narain
 * *** empty log message ***
 *
 ***************************************************************************/
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * The MNGR Load Balancing Strategy* * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*  The following notes on how the multiple-managers strategy works
    were made by Sanjeev on 2/8/96.

MyLoad calls CldMyLoad which is the length of seed (token) queue.
Note: controllers usually have most of the tokens
 
MyController(myPE) : 0-15 is 15, 16-31 is 31, and so on.
CONTROLLER(pe) is true if pe is a controller.
controller variable is true if I am a controller.
mycontroller variable stores my controller
 
LDB_ELEMENT has srcPE and piggybackLoad fields
 
FillLDB sets piggybackLoad to MyLoad.
StripLDB calls RecvUpdateStatus on controllers, which sets
load_boss[] or load_cluster[].
 
EnqMsg enqueues a token in a token-list as well as in scheduler's queue
LeastLoadKids (called only from managers) returns the least loaded
kid and its load, or -1 if minload > KID_SATURATION.
LeastLoadBosses (called only from managers) returns the least loaded
manager out of the managers in the "exchange" group.
"exchanges" is log2(numBosses) : each manager exchanges load
with that many other managers. I dont understand what the use of "start_pe"
in LeastLoadBosses is.
 
NewMsg_FromLocal on managees sends the chare to controller.
on controllers it does EnqMsg().
NewMsg_FromNet on managees enqueues the msg locally.
On controllers it calls Strategy(), which does EnqMsg(seed).
Then it finds the least loaded controller/kid. (a kid is chosen if
its load is less than MINIMUM_KID_LOAD). If minload is < saturation,
Strategy calls SendFreeChare(least_load_pe), which calls CldPickSeedAndSend()
and then increments load of that pe.
CldPickSeedAndSend (in ldbcfns.c) picks any seed from token list and sends
it to the pe.
 
PeriodicKidStatus() on kids sends a message to RecvStatus in its
controller periodically, and boss doesnt seem to do anything.
(Im not sure it works because CldPeriodicKidStatus doesnt do anything).
PeriodicBossStatus() on controllers periodically sends msgs to
each of its neighbor controllers (round-robin). When bosses receive
the msg at RecvStatus they call do_redistribution() with sender.
 
do_redistribution() sends half the load difference to the other boss.
 
PeriodicKidsRedist() is called periodically on bosses. It sends all
tokens enqueued on the boss to kids until they are saturated.
PeriodicBossesRedist() doesnt seem to work because CldPeriodicBossesRedist
is not called from anywhere.


****************************************************************************/


















#define MAXINT  0xffff
#define HUGE_INT 9999999

#define MAX_BOSSES 32
#define CLUSTER_SIZE 16

#define  CONTROLLER(pe)  (((pe + 1)%CLUSTER_SIZE == 0) || ((pe + 1 ) == numPe))

#define  MAX_STEP  5
#define  MAX_EXCHANGES  6

#define KID_SATURATION			3
#define MINIMUM_KID_LOAD 		2
#define KID_STATUS_UPDATE_INTERVAL  	50

#define BOSS_SATURATION			3
#define MINIMUM_BOSS_LOAD 		1
#define BOSS_STATUS_UPDATE_INTERVAL  	25
#define BOSS_REDIST_UPDATE_INTERVAL	100

#ifndef TRUE 
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif



module ldb {
#include "ldb.h"

typedef struct ldb_element {
  unsigned int srcPE;
  int piggybackLoad;
} LDB_ELEMENT;

message {
    int srcPe;
} DUMMYMSG;

message {
    int dummy;
} DUMMY_MSG;

typedef struct ldb_status {
  int	peLoad;
  int	myLoadSent;
  int timeLoadSent;
  int statusMsgID;
} LDB_STATUS;

extern int CldAddToken();
extern int CldPickSeedAndSend();

extern void *CqsCreate();

export_to_C CldGetLdbSize()
{
       TRACE(CkPrintf("%d:CldGetLdbSize()\n",CkMyPe()));
       return sizeof(LDB_ELEMENT);
}


export_to_C CldCreateBoc()
{
  DUMMYMSG *msg;
  
  TRACE(CkPrintf("%d:CldCreateBoc()\n",CkMyPe()));
  msg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG);	
  CreateBoc(LDB, LDB@BranchInit, msg);
}


export_to_C CldFillLdb(destPe, ldb)
  int destPe;
  void *ldb;
{
  TRACE(CkPrintf("%d:CldFillLdb(%d,%x)\n",CkMyPe(),destPe,ldb));
  BranchCall(ReadValue(LdbBocNum), LDB@FillLDB(destPe, (LDB_ELEMENT *)ldb));
}

export_to_C CldStripLdb(ldb)
     void *ldb;
{
  TRACE(CkPrintf("%d:CldStripLdb(%x)\n",CkMyPe(),ldb));
  BranchCall(ReadValue(LdbBocNum), LDB@StripLDB((LDB_ELEMENT *)ldb));
}



export_to_C CldNewSeedFromNet(msgst, ldb, sendfn,queuing,priolen,prioptr) 
     void *msgst, *ldb;
     void (*sendfn)();
     unsigned int queuing, priolen, *prioptr;
{
  TRACE(CkPrintf("%d:CldNewSeedFromNet(%x,%x,%x)\n",CkMyPe(),msgst,ldb,sendfn));
  BranchCall(ReadValue(LdbBocNum), LDB@NewMsg_FromNet(msgst, ldb, sendfn,queuing,priolen,prioptr) );
}

export_to_C CldNewSeedFromLocal( msgst, ldb, sendfn,queuing,priolen,prioptr)
     void *msgst, *ldb;
     void (*sendfn)();
     unsigned int queuing, priolen, *prioptr;
{
  TRACE(CkPrintf("%d:CldNewSeedFromLocal(%x,%x,%x)\n",CkMyPe(),msgst,ldb,sendfn));
  BranchCall(ReadValue(LdbBocNum), LDB@NewMsg_FromLocal(msgst, ldb, sendfn,queuing,priolen,prioptr) );
}

export_to_C CldProcessMsg(msgPtr, localdataPtr)
     void *msgPtr, *localdataPtr;
{
  TRACE(CkPrintf("%d:CldProcessMsg(%x,%x)\n",CkMyPe(),msgPtr,localdataPtr));
  BranchCall(ReadValue(LdbBocNum), LDB@ProcessMsg(msgPtr, localdataPtr));
}

export_to_C CldProcessorIdle()
{
  TRACE(CkPrintf("%d:CldProcessorIdle()\n",CkMyPe()));
  BranchCall(ReadValue(LdbBocNum), LDB@ProcessorIdle());
}


export_to_C CldPeriodicCheckInit()
{
  TRACE(CkPrintf("%d:CldPeriodicCheckInit()\n",CkMyPe()));
  BranchCall(ReadValue(LdbBocNum), LDB@PeriodicCheckInit());
}

export_to_C void CldPeriodicBossesRedist()
{
  TRACE(CkPrintf("%d:CldPeriodicBossesRedist()\n",CkMyPe()));
  BranchCall(ReadValue(LdbBocNum), LDB@PeriodicBossesRedist());
}

export_to_C void CldPeriodicStatus()
{
}
export_to_C void CldPeriodicKidStatus()
{
}

export_to_C void CldPeriodicBossStatus()
{
  BranchCall(ReadValue(LdbBocNum), LDB@PeriodicBossStatus());
}

export_to_C void CldPeriodicRedist()
{
  /*
    LdbPeriodicBossesRedist(bocNum);
    */
  TRACE(CkPrintf("%d:CldPeriodicRedist()\n",CkMyPe()));
  BranchCall(ReadValue(LdbBocNum), LDB@PeriodicKidsRedist());
  CallBocAfter(CldPeriodicRedist, ReadValue(LdbBocNum), 
	       BOSS_REDIST_UPDATE_INTERVAL);
}

/* Use this structure to hold internal queues at the controllers */
typedef struct msginfo {
   void *msg;
   void *ldb;
   void (*sendfn)();
} MSGINFO;

MSGINFO *new_MSGINFO(msg, ldb, sendfn,queuing,priolen,prioptr)
void *msg, *ldb, (*sendfn)();
     unsigned int queuing, priolen, *prioptr;
{
   MSGINFO *ret;
   ret = (struct msginfo *)CkAlloc(sizeof(MSGINFO));

   ret->msg = msg;
   ret->ldb = ldb;
   ret->sendfn = sendfn;
  
   return ret;
}

BranchOffice LDB {
  
  
  int exchanges;
  int numPe;
  int controller;
  int mycontroller;
  DUMMYMSG *statusMsg;
  int load_cluster[CLUSTER_SIZE];
  
  int numBoss;
  int nbr_boss[MAX_BOSSES];
  int  load_boss[MAX_BOSSES];
  DUMMYMSG *boss_statusMsg;
  int myPE;
  int LdbBoc;
  
/*   void *LdbFreeChareQueue; */
  
  
  private unsigned int flipi(n, i)
    unsigned int n, i;
  {
    n /= CLUSTER_SIZE;
    return( ((n & (1<<i)) ^ (1<<i)) | (n & ~(1<<i)));
  }
  
  private unsigned int get_index(m)
    unsigned int m;
  {
    unsigned int i, n, result;
    
    if (CONTROLLER(m))
      {
	n = myPE;
	
	for (i=0; i<exchanges; i++)
	  {
	    result = PrivateCall(flipi(n, i));
	    if (m/CLUSTER_SIZE == result)
	      return i;
	  }
      }
    else
      return (m % CLUSTER_SIZE);
    return -1;
  }
  
  private increment_load(least_loaded_pe)
    int least_loaded_pe;
  {
    int index;
    
    index = PrivateCall(get_index(least_loaded_pe));
    if (CONTROLLER(least_loaded_pe))
      load_boss[index] ++;
    else
      load_cluster[index] += 1;
  }
  
  private LeastLoadKids(load)
    int *load;
  {
    int pe;
    int index;
    int min_pe = 0;
    int min = HUGE_INT;
    int start_pe, end_pe;
    
    end_pe = myPE;
    start_pe = (end_pe/CLUSTER_SIZE)*CLUSTER_SIZE;
    for (pe=start_pe; pe<end_pe; pe++)
      {
	index = PrivateCall(get_index(pe));
	if (load_cluster[index] < min)
	  {
	    min = load_cluster[index];
	    min_pe = pe;
	  }
      }
    *load = min;
    if (min < KID_SATURATION)
      return(min_pe);
    return -1;
  }
  

  private LeastLoadBosses()
    {
      int i, j;
      int min_pe;
      int min = HUGE_INT;
      static int start_pe = 0;
      
      if (numBoss <= 1) return -1;
      for (j=0; j<exchanges; j++)
	{
	  i = (start_pe + j) % exchanges;
	  if (load_boss[i] < min)
	    {
	      min = load_boss[i];
	      min_pe = nbr_boss[i];
	    }
	}
      if (min < BOSS_SATURATION)
	{
	  start_pe = min_pe; 
	  return min_pe;
	}
      else
	start_pe = 0;
      return -1;
    }
  
  
  private int MyLoad()
    {
/*      TRACE(CkPrintf("MyLoad=%d\n",CqsLength(LdbFreeChareQueue))); */
/*      return CqsLength(LdbFreeChareQueue); */
	return CldMyLoad();
    }

  private SendFreeChare(pe)
    int pe;
  {	
    MSGINFO *tempstr;

    if (CldPickSeedAndSend(pe))
	return 1;
    else return 0;
/*
    if(!CqsEmpty(LdbFreeChareQueue))
	{
           CqsDequeue(LdbFreeChareQueue, (void **)(&tempstr));
	   (*(tempstr->sendfn))(tempstr->msg, pe);
	   return 1;
        }		
    return 0;
*/
  }
  
  private do_redistribution(other, other_load)
    int other, other_load;
  {
    int i;
    int number;
    int SKIP = 1;
    int index = 0;
    int my_load = PrivateCall(MyLoad());
    
    number = 0;
    if (my_load > other_load)
      number = (my_load - other_load)/2;
#ifdef PRIORITY
    number = MAX_EXCHANGES/2;
#endif
    
    TRACE(CkPrintf("[%d] do_redist: my_load=%d, other_load=%d, number = %d\n",
		   myPE, my_load, other_load, number));	
    
    if (number > MAX_EXCHANGES)
      number = MAX_EXCHANGES;
    for (i=0; i<number; i++)
      PrivateCall(SendFreeChare(other));
    TRACE(CkPrintf("[%d] do_redist: Finish\n", myPE));	
  }
  
  private MyController(x)
    int x;
  {
    x = x / CLUSTER_SIZE;
    x = (x+1)*CLUSTER_SIZE;
    if (x>numPe-1)
      return numPe-1;
    return x-1;
  }
  
  
  private SentUpdateStatus(pe)
    int pe;
  {
  }
  
  private RecvUpdateStatus(ldb)
    LDB_ELEMENT * ldb;
  {
    int index;
    
    index = PrivateCall(get_index(ldb->srcPE));
    if (CONTROLLER(ldb->srcPE))
      load_boss[index] = ldb->piggybackLoad;
    else
      load_cluster[index] = ldb->piggybackLoad;
  }
  
  
  private PrintNodeNeighbours()
    {
    }
  
  public ProcessorIdle()
    {
    }
  
  
  private LeastLoadPe()
    {
      int pe;
      int boss;
      int kid_load;
      int kid = PrivateCall(LeastLoadKids(&kid_load));
      
      if (kid==-1)
	pe = PrivateCall(LeastLoadBosses());
      else
	if (kid_load < MINIMUM_KID_LOAD)
	  pe = kid;
	else
	  {
	    pe = PrivateCall(LeastLoadBosses());
	    /*
	      if (pe == -1)
	      pe = kid;
	      */
	  }
      
      return pe;
    }
  
  
  
  private EmptyQueue()
    {
      TRACE(CkPrintf("Don't know what to empty\n"));
/*      return CqsEmpty(LdbFreeChareQueue); */
    }

  
  
  
  private EnqMsg(msg, ldb, sendfn,queuing,priolen,prioptr)
    void *msg, *ldb, (*sendfn)();
     unsigned int queuing, priolen, *prioptr;
  {
    CldAddToken(msg, sendfn,queuing,priolen,prioptr);
/*    CqsEnqueue(LdbFreeChareQueue, new_MSGINFO(msg, ldb, sendfn,queuing,priolen,prioptr)); */
  }
  
  public PeriodicCheckInit()
    {
      if (numPe > 1)
	{
	  if (CONTROLLER(myPE))
	    {
	      if (numPe > CLUSTER_SIZE)
		CallBocAfter(CldPeriodicBossStatus, LdbBoc, 
			     BOSS_STATUS_UPDATE_INTERVAL);
	      CallBocAfter(CldPeriodicRedist, LdbBoc,
			   BOSS_REDIST_UPDATE_INTERVAL);
	    }
	  else
	    CallBocAfter(CldPeriodicKidStatus, LdbBoc, 
			 KID_STATUS_UPDATE_INTERVAL);
	}
    }
  
  
  public ProcessMsg(msgPtr, localdataPtr)
    void *msgPtr, *localdataPtr;
  {
  }
  
  
  /* LDB Branch Office Chare Functions */
  
  entry RecvStatus : (message DUMMYMSG *dmsg)
    {
      int index;
      
      if (CONTROLLER(dmsg->srcPe))
	{
	  index = PrivateCall(get_index(dmsg->srcPe));
	  PrivateCall(do_redistribution(dmsg->srcPe, load_boss[index]));
	}
      CkFreeMsg(dmsg);
    }
  
  entry BranchInit : (message DUMMYMSG * dmsg)
    {
      int i;
      void *Qs_Create();
      LDB_ELEMENT *ldb;
  
      TRACE(CkPrintf("Enter Node LdbInit()\n"));
      
      LdbBoc = MyBocNum();
      LdbBocNum = LdbBoc;
      ReadInit(LdbBocNum);
      
      numPe = CkNumPes();
      myPE = CkMyPe();
      controller = CONTROLLER(myPE);
      mycontroller = PrivateCall(MyController(myPE));
      exchanges = numBoss = 0;
      if (controller)
	{
/*	  LdbFreeChareQueue = CqsCreate(); */
	  for (i=0; i<CLUSTER_SIZE; i++)
	    load_cluster[i] = 0;
	  if (numPe%CLUSTER_SIZE == 0)
	    numBoss = numPe/CLUSTER_SIZE;
	  else
	    numBoss = numPe/CLUSTER_SIZE + 1;
	  if (numBoss > 1)
	    {
	      exchanges = (int) (log((double) numBoss) /
				 log((double) 2.0));
	      
	      TRACE(CkPrintf("[%d] Ldbinit: log(numBoss)=%f, log(2.0)=%f, exchanged=%f\n",
			     myPE, log((double) numBoss), log((double) 2.0),
			     log((double) numBoss)/log((double) 2.0)));
	      
	      TRACE(CkPrintf("[%d] LdbInit: exchanges=%d, numBoss=%d\n",
			     myPE, exchanges, numBoss));
	      
	      for (i=0; i<exchanges; i++)
		{
		  int boss_index;
		  
		  boss_index = PrivateCall(flipi(myPE, i));
		  load_boss[i] = 0;
		  if((nbr_boss[i] = (CLUSTER_SIZE-1) + boss_index*CLUSTER_SIZE)
		     >= numPe)
		    nbr_boss[i] = numPe - 1;
		}
	    }
	}
    }
  
  
  /* Load Balance messages received at the Node from the Network */
  public StripLDB(ldb)
    LDB_ELEMENT *ldb;
  {
    /*possible Neighbour update status from piggyback info from Node only*/
    if ((numPe > 1) && (ldb->srcPE != myPE) 
	&&  (ldb->srcPE != CkNumPes()) && controller)
      PrivateCall(RecvUpdateStatus(ldb));
  }
  
  private Strategy(msg, ldbptr, sendfn,queuing,priolen,prioptr)
    void *msg, *ldbptr, (*sendfn)();
     unsigned int queuing, priolen, *prioptr;
  {
    int load;
    LDB_ELEMENT *ldb;
    int least_loaded_pe, fixed;
    
    ldb = (LDB_ELEMENT *)ldbptr;
    if (CONTROLLER(ldb->srcPE))
      least_loaded_pe = PrivateCall(LeastLoadKids(&load));
    else
      least_loaded_pe = PrivateCall(LeastLoadPe());
    
    TRACE(CkPrintf("[%d] LdbStrategy: srcPE=%d, destPE=%d\n",
		   myPE, ldb->srcPE, least_loaded_pe));
    
    if (least_loaded_pe >= 0)
      {
#ifdef PRIORITY
	if (PrivateCall(EmptyQueue())) 
	  {
	    fixed = (!CONTROLLER(least_loaded_pe))?  TRUE: FALSE;
	    (*sendfn)(msg, least_loaded_pe);
	    PrivateCall(increment_load(least_loaded_pe));
	  }
	else
#endif
	  {
	    PrivateCall(EnqMsg(msg, ldbptr, sendfn,queuing,priolen,prioptr));
	    if (PrivateCall(SendFreeChare(least_loaded_pe)))
	      PrivateCall(increment_load(least_loaded_pe));
	  }
      }
    else
      PrivateCall(EnqMsg(msg, ldbptr, sendfn,queuing,priolen,prioptr));
  }

  public NewMsg_FromNet(msgst, ldb, sendfn,queuing,priolen,prioptr)
    void *msgst, *ldb;
    void (*sendfn)();
     unsigned int queuing, priolen, *prioptr;
  {
    if (controller)
      PrivateCall(Strategy(msgst, ldb, sendfn,queuing,priolen,prioptr));
    else
      { 
	TRACE(CkPrintf("[%d] Ldb_NewChare_Net:: Message from outside. \n",
		       myPE));
	CsdEnqueue(msgst);
      }
  }
  

  public NewMsg_FromLocal(msgst, ldb, sendfn,queuing,priolen,prioptr)
    void *msgst, *ldb;
    void (*sendfn)();
     unsigned int queuing, priolen, *prioptr;
  {
    if (controller)
      {
	TRACE(CkPrintf("[%d] Ldb_NewChare_Local:: Queuing up.\n",
		       myPE));
      PrivateCall(EnqMsg(msgst, ldb, sendfn,queuing,priolen,prioptr));
      }
    else
      {
	TRACE(CkPrintf("[%d] Ldb_NewChare_Local:: Send to control.\n",
		       myPE));
	(*sendfn)(msgst, mycontroller);
      }
  }

  
  
  public FillLDB(destPe, ldb)
    int destPe;
    LDB_ELEMENT *ldb;
  {
    ldb->srcPE = myPE;
    if (CONTROLLER(ldb->srcPE))
      ldb->piggybackLoad = CldMyLoad();
    else	
      ldb->piggybackLoad = PrivateCall(MyLoad());
  }
  
  
  
  public void PeriodicKidStatus()
    {
      statusMsg = (DUMMYMSG *) CkAllocMsg(DUMMYMSG);
/*      CkMemError(statusMsg); */
      statusMsg->srcPe = CkMyPe();
      ImmSendMsgBranch(RecvStatus, statusMsg, mycontroller);
      CallBocAfter(CldPeriodicKidStatus, LdbBoc,
		   KID_STATUS_UPDATE_INTERVAL);
    }

  
  
  public void PeriodicBossStatus()
    {
      static int index = 0;

      boss_statusMsg = (DUMMYMSG *) CkAllocMsg(DUMMYMSG);
/*      CkMemError(boss_statusMsg); */
      boss_statusMsg->srcPe = CkMyPe();
      ImmSendMsgBranch(RecvStatus, boss_statusMsg, nbr_boss[index]);
      index = (index+1) % exchanges;
      CallBocAfter(CldPeriodicBossStatus, LdbBoc, 
		   BOSS_STATUS_UPDATE_INTERVAL);
    }

  
  
  
  public void PeriodicKidsRedist()
    {
      int i, j;
      int picked, index;
      int done;
      int start_pe, end_pe;
      
      end_pe = myPE;
      start_pe = (end_pe/CLUSTER_SIZE) * CLUSTER_SIZE;
      done = FALSE;
      
      while (!done)
	{
	  picked=0;
	  for (i=start_pe; i<end_pe; i++)
	    {	
	      index = PrivateCall(get_index(i));
	      if (load_cluster[index] < KID_SATURATION)
		{
		  if (PrivateCall(SendFreeChare(i)))
		    {
		      picked++;
		      load_cluster[index]++;
		    }
		  else
		    {
		      done = TRUE;
		      break;
		    }
		}
	    }
	  if (!picked)
	    done = TRUE;
	}
    }
  
  public void PeriodicBossesRedist()
    {
      int i;
      
      for (i=0; i<exchanges; i++)
	if (load_boss[i] < BOSS_SATURATION)
	  if (PrivateCall(SendFreeChare(nbr_boss[i])))
	    load_boss[i]++;
	  else
	    break;
    }

  
}
  
  
}


