/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * The TOK Load Balancing Strategy * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#define MAXINT  0xffff
#define HUGE_INT 9999999

#define MAX_BOSSES 32
#define CLUSTER_SIZE 8

#define  MAX_STEP  5
#define  MAX_EXCHANGES  6

#define KID_SATURATION			4
#define MINIMUM_KID_LOAD 		3
#define KID_STATUS_UPDATE_INTERVAL  	100

#define BOSS_SATURATION			3
#define MINIMUM_BOSS_LOAD 		1
#define BOSS_STATUS_UPDATE_INTERVAL  	100
#define BOSS_REDIST_UPDATE_INTERVAL	500

/** This is closely linked to the bitvec_int_macros.h **/
#define HASH_TABLE_SIZE 2003
#define MULTIPLICAND	367

#ifdef BITVECTOR
#define MAX_HEAP_SIZE 10000
#else
#define MAX_HEAP_SIZE 90000
#endif
#define BITMAX 6

#ifdef BITVECTOR
#define TokenGreaterThan(p, q) GreaterThan(p, q)
#define TokenEqual(p, q) (!NotEqual(p, q))
#endif

#define  CONTROLLER(pe) (((pe + 1)%CLUSTER_SIZE == 0) || ((pe + 1 ) == numPe))
#define  MyLoad()	heap_index
module ldb {
#include "ldb.h"

extern GreaterThan();
extern NotEqual();

typedef struct ldb_element {
	PeNumType srcPE;
	int	piggybackLoad;
} LDB_ELEMENT;

message  {
    int dummy;
} DUMMYMSG;

typedef struct {
    int pe;
    TRACE(int index;)
#ifdef BITVECTOR
    unsigned int priority[BITMAX];
#else
    unsigned int priority;
#endif
} TOKEN_TYPE;


message {
    TOKEN_TYPE token; 
} OperationMsg;

message {
    int exchanges;
    TOKEN_TYPE tokens[MAX_EXCHANGES];
} RedistributionMsg;


typedef struct ldb_status {
	int	peLoad;
	int	myLoadSent;
	int timeLoadSent;
	int statusMsgID;
} LDB_STATUS;


typedef struct Entry {
#ifdef BITVECTOR
	unsigned int priority[BITMAX];
#else
	unsigned int priority;
#endif
	void *work;
	struct Entry *next;
} ENTRY;


#ifdef BITVECTOR
#define AssignPriority(x,y) { \
	int i, vector_size, vector_length; \
 	for (i=0; i<BITMAX; i++) \
		x[i] = 0; \
	vector_length = (y[0] >> 24) & 0x0ff; \
	if ( vector_length <= 24 ) \
                vector_size = 1; \
        else \
                vector_size = ((vector_length - 25) >> 5) + 2; \
	if (vector_size > BITMAX) \
		CkPrintf("[%d] *** ERROR *** Illegal vector length %d.\n", \
			myPE, vector_size);  \
	for (i=0; i<vector_size; i++) \
		x[i] = y[i]; \
}
#define CopyPriority(x,y) {int icp; for (icp=0; icp<BITMAX; icp++) x[icp] = y[icp];}
#else
#define AssignPriority(p,q) p=q
#define CopyPriority(p,q) p=q
#endif

export_to_C setLdbSize()
{
       CpvAccess(LDB_ELEM_SIZE) = sizeof(LDB_ELEMENT);
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


export_to_C LdbPeriodicCheckInit(ChareNumType bocNum)
{
	BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicCheckInit());
}

export_to_C LdbPeriodicRedist(ChareNumType bocNum)
{
	BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicKidsRedist());
	CallBocAfter(LdbPeriodicRedist, CpvAccess(LdbBocNum), 
	    BOSS_REDIST_UPDATE_INTERVAL);
}

export_to_C LdbPeriodicKidStatus(ChareNumType bocNum)
{
  BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicKidStatus());
  CallBocAfter(LdbPeriodicKidStatus, CpvAccess(LdbBocNum), 	
	       KID_STATUS_UPDATE_INTERVAL);
}

export_to_C LdbPeriodicBossStatus(ChareNumType bocNum)
{
  BranchCall(CpvAccess(LdbBocNum), LDB@PeriodicBossStatus());
  CallBocAfter(LdbPeriodicBossStatus, CpvAccess(LdbBocNum), 	
	       BOSS_STATUS_UPDATE_INTERVAL);
}


BranchOffice LDB {
  int LdbBoc;
  int numPe;
  int	myPE;
  long status;
  int exchanges;
  int mycontroller;
  BOOLEAN controller;
  RedistributionMsg *redist_msg;
  int load_cluster[CLUSTER_SIZE];
  OperationMsg *status_msg, *insert_msg, *delete_msg;
  int numBoss;
  int nbr_boss[MAX_BOSSES];
  int  load_boss[MAX_BOSSES];
  DUMMYMSG *boss_statusMsg;
  ENTRY *hash_table[HASH_TABLE_SIZE];
  int heap_index;
  TOKEN_TYPE *heap[MAX_HEAP_SIZE];
  void *LdbFreeChareQueue;
  
  private SentUpdateStatus(pe)
    int pe;
  {
  }
  /*****************************************************************/
  /** Get status report for this message.				**/
  /*****************************************************************/
  private get_status(index, ldb)
    int index;
  LDB_ELEMENT *ldb;
  {
    TRACE(CkPrintf("[%d] get_status: srcPE=%d, load=%d\n",
		   myPE, ldb->srcPE, ldb->piggybackLoad));
    if (CONTROLLER(ldb->srcPE))
      {
	load_boss[index] = ldb->piggybackLoad;
	PrivateCall(do_redistribution(ldb->srcPE, load_boss[index]));
      }
    else
      load_cluster[index] = ldb->piggybackLoad;
  }
  
  
  /*****************************************************************/
  /** Now follow a bunch of useless functions. We plan to do 	**/
  /** something with them in the future.				**/
  /*****************************************************************/
  private PrintNodeNeighbours()
    {
    }
  
  /*****************************************************************/
  /** This function determines the least loaded processor.	**/
  /*****************************************************************/
  private LeastLoadPe()
    {
      int kid_load;
      int pe = PrivateCall(LeastLoadKids(&kid_load));
      
      if (pe==-1) pe = PrivateCall(LeastLoadBosses());
      return pe;
    }
  
  
  /*****************************************************************/
  /** This function is used to determine the least loaded kid of	**/
  /** a manager.							**/
  /*****************************************************************/
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
  
  
  /*****************************************************************/
  /** This function is used to determine the least loaded managers**/
  /** in the dimensional neighbors graph.				**/
  /*****************************************************************/
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
      return -1;
    }
  
  
  private TokenStrategy(msg)
    OperationMsg *msg;
  {
    int load;
    int need;
    LDB_ELEMENT *ldb;
    int least_loaded_pe;
    
    ldb = LDB_UPTR(msg);
    if (CONTROLLER(ldb->srcPE))
      least_loaded_pe = PrivateCall(LeastLoadKids(&load));
    else
      least_loaded_pe = PrivateCall(LeastLoadPe());
    
    TRACE(CkPrintf("[%d] TokenStrategy: srcPE=%d, destPE=%d, heap_index=%d\n",
		   myPE, ldb->srcPE, least_loaded_pe, heap_index));
    
    
    need = FALSE;
    if (least_loaded_pe >= 0)
      {
	TOKEN_TYPE *token;
	
	token = &(msg->token);
	if (heap_index>0)
	  {
	    PrivateCall(insert_heap(&(msg->token)));
	    PrivateCall(delete_heap(&token));
	    need = TRUE;
	  }
	
	if (CONTROLLER(least_loaded_pe))
	  PrivateCall(SendManager(token, least_loaded_pe));
	else
	  PrivateCall(SendManagee(token, least_loaded_pe));
	if (need) CmiFree(token);
	PrivateCall(increment_load(least_loaded_pe));
      }
    else
      PrivateCall(insert_heap(&(msg->token)));
  }
  
  
  /*****************************************************************/
  /** Send token to manager.										**/
  /*****************************************************************/
  private SendManager(token, least_loaded_pe)
    TOKEN_TYPE *token;
  int least_loaded_pe;
  {
    insert_msg = (OperationMsg *) CkAllocMsg(OperationMsg);
    CkMemError(insert_msg);
    
    insert_msg->token.pe = token->pe;
    TRACE(insert_msg->token.index = token->index);
    CopyPriority(insert_msg->token.priority, token->priority);
    ImmSendMsgBranch(LDB@INSERT, insert_msg, least_loaded_pe);
  }
  
  /*****************************************************************/
  /** Send token to managee.										**/
  /*****************************************************************/
  private SendManagee(token, least_loaded_pe)
    TOKEN_TYPE *token;
  int least_loaded_pe;
  {
    delete_msg = (OperationMsg *) CkAllocMsg(OperationMsg);
    CkMemError(delete_msg);
    
    delete_msg->token.pe = least_loaded_pe;
    TRACE(delete_msg->token.index = token->index);
    CopyPriority(delete_msg->token.priority, token->priority);
    
    ImmSendMsgBranch(LDB@DELETE, delete_msg, token->pe);
  }
  
  
  
  /*****************************************************************/
  /** Send "number" tokens to processor "pe".			**/
  /*****************************************************************/
  private SendTokens(pe, number)
    int pe;
  int number;
  {
    int i;
    TOKEN_TYPE *token;
    redist_msg = (RedistributionMsg *)CkAllocMsg(RedistributionMsg);
    CkMemError(redist_msg);
    for (i=0; i<number; i++)
      {
	if (PrivateCall(delete_heap(&token)))
	  {
	    redist_msg->tokens[i].pe = token->pe;
	    TRACE(redist_msg->tokens[i].index = token->index;)
	      CopyPriority(redist_msg->tokens[i].priority, 
			   token->priority);
	    CmiFree(token);
	  }
	else
	  break;
      }
    redist_msg->exchanges = i;
    if (i>0)
      ImmSendMsgBranch(LDB@REDISTRIBUTION, redist_msg, pe);
    
    TRACE(CkPrintf("[%d] SendTokens: no=%d, last_pe=%d, last_prio=%d\n",
		   myPE, redist_msg->exchanges, 
		   redist_msg->tokens[i-1].pe, 
		   redist_msg->tokens[i-1].priority));
    if(i == 0)
      CkFreeMsg(redist_msg);
  }
  
  
  /*****************************************************************/
  /** Get the index of the ith dimensional neighbor for a manager.**/
  /*****************************************************************/
  private flipi(n, i)
    unsigned int n, i;
  {
    n /= CLUSTER_SIZE;
    return( ((n & (1<<i)) ^ (1<<i)) | (n & ~(1<<i)));
  }
  
  
  /*****************************************************************/
  /** Increments the status for the corresponding manager/kid.	**/
  /*****************************************************************/
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
  

  /*****************************************************************/
  /** Get the index of the manager (0..(dim-1)) or the kid	**/
  /** 0..(CLUSTER_SIZE-1)						**/
  /*****************************************************************/
  private get_index(m)
    unsigned int m;
  {
    int i;
    
    if (CONTROLLER(m))
      {
	int n = myPE;
	
	for (i=0; i<exchanges; i++)
	  if (m == PrivateCall(flipi(n, i)))
	    return i;
      }
    else
      return (m % CLUSTER_SIZE);
    return -1;
  }
  

  /*****************************************************************/
  /** This function redistributes tokens amongst the managers.	**/
  /*****************************************************************/
  private do_redistribution(other, other_load)
    int other, other_load;
  {
    int i;
    int number;
    int SKIP = 1;
    int index = 0;
    int my_load = MyLoad();
    
    number = 0;
    if (my_load > other_load)
      number = (my_load - other_load)/2;
    number = MAX_EXCHANGES/2;
    
    TRACE(CkPrintf("[%d] do_redist: my_load=%d, other_load=%d, number = %d\n",
		   myPE, my_load, other_load, number));
    
    if (number > MAX_EXCHANGES)
      number = MAX_EXCHANGES;
    PrivateCall(SendTokens(other, number));
    TRACE(CkPrintf("[%d] do_redist: Finish\n", myPE));
  }
  
  
  /*****************************************************************/
  /** This function returns the id of my controller.		**/
  /*****************************************************************/
  private MyController(x)
    int x;
  {
    x = x / CLUSTER_SIZE;
    x = (x+1)*CLUSTER_SIZE;
    if (x>numPe-1)
      return numPe-1;
    return x-1;
  }
  
  /*****************************************************************/
  /** Initialize hash table.					**/
  /*****************************************************************/
  private init_table()
    {
      int i;
      
      for (i=0; i<HASH_TABLE_SIZE; i++)
	hash_table[i] = NULL;
    }
  

  /*****************************************************************/
  /** Insert an entry into the hash table.			**/
  /*****************************************************************/
  private insert_table(priority, msg)
#ifdef BITVECTOR
    unsigned int priority[BITMAX];
#else
  unsigned int priority;
#endif
  void * msg;
  {
    ENTRY *new;
    int index = HashFunction(priority);
    
    TRACE(PrivateCall(_print_bit_vector("insert_table", index, priority));)
      
      if (index>=HASH_TABLE_SIZE || index <0)
	{
	  TRACE(PrivateCall(_print_bit_vector("insert_table", index, priority));)
	    CkPrintf("[%d] *** ERROR *** Illegal hash %d computed.\n",
		     myPE, index);
	}
    
    new = (ENTRY *) CmiAlloc(sizeof(ENTRY));
    CkMemError(new);
    CopyPriority(new->priority, priority);
    new->work = msg;
    new->next = hash_table[index];
    hash_table[index] = new;
  }


/*****************************************************************/
/** Delete an entry from the hash table.			**/
/*****************************************************************/
private delete_table(priority, msgPtr)
#ifdef BITVECTOR
unsigned int priority[BITMAX];
#else
unsigned int priority;
#endif
void **msgPtr;
{
	ENTRY *current, *previous;
	int index = HashFunction(priority);

	TRACE(PrivateCall(_print_bit_vector("delete_table", index, priority)));

	previous =  current = hash_table[index];
	if (current != NULL)
	{
		if (TokenEqual(current->priority, priority))
		{
			*msgPtr = current->work;
			hash_table[index] = current->next;
			return;
		}

		current = current->next;
		while (current != NULL)
		{
			if (TokenEqual(current->priority, priority))
			{
				*msgPtr = current->work;
				previous->next  = current->next;
				return;
			}
			current = current->next;
			previous = previous->next;
		}
	}
	*msgPtr = NULL;
	CkPrintf("[%d] *** ERROR *** Message not found in hash table %d.\n",
	    myPE, index);
	TRACE(PrivateCall(_print_bit_vector("delete_table", index, priority)));
}


/*****************************************************************/
/** Initialize the heap.					**/
/*****************************************************************/
private init_heap()
{
	int i;
	heap_index = 0;
	for (i=0; i<MAX_HEAP_SIZE; i++)
		heap[i] = (TOKEN_TYPE *) NULL;
}


/*****************************************************************/
/** Insert a new token into the heap.				**/
/*****************************************************************/
private insert_heap(insert_item)
TOKEN_TYPE *insert_item;
{
	int i, j;
	TOKEN_TYPE *item;


	if (heap_index >= MAX_HEAP_SIZE)
		CkPrintf("*** ERROR *** Heap Size Exceeded.\n");
	item = (TOKEN_TYPE *) CmiAlloc(sizeof(TOKEN_TYPE));
	CkMemError(item);
	item->pe = insert_item->pe;
	CopyPriority(item->priority, insert_item->priority);

	TRACE(item->index = insert_item->index;)
	TRACE(PrivateCall(check_token("insert heap", item));)

	heap[heap_index++] = item;
	j = heap_index - 1;
	i = (heap_index - 2)/2;

	while ((i>=0) && (TokenGreaterThan(heap[i]->priority, item->priority)))
	{
		heap[j] = heap[i];
		j = i;
		if (i==0)
			break;
		i = (i-1)/2;
	}
	heap[j] = item;
}




/*****************************************************************/
/** Delete a token from the heap.				**/
/*****************************************************************/
private delete_heap(item)
TOKEN_TYPE **item;
{
	int i, j;
	TOKEN_TYPE *temp;

	if (heap_index <= 0)
	{
		return 0;
	}
	*item = heap[0];

	TRACE(PrivateCall(_print_bit_vector("delete_heap", HashFunction(heap[0]->priority),
	    heap[0]->priority)));
	TRACE(PrivateCall(check_token("delete heap", *item)));

	j=1;
	i=0;
	temp = heap[0] = heap[heap_index-1];
	heap[heap_index-1] = NULL;
	heap_index--;
	while (j <= heap_index-1)
	{
		if ((j < heap_index-1) &&
		    TokenGreaterThan(heap[j]->priority, 
		    heap[j+1]->priority))
			j=j+1;
		if  (TokenGreaterThan(heap[j]->priority, temp->priority)  ||
		    TokenEqual(temp->priority, heap[j]->priority))
			break;
		else
			heap[(j-1)/2] = heap[j];
		j = 2*j + 1;
	}
	heap[(j-1)/2] = temp;

	return 1;
}

/*****************************************************************/
/** These are functions we need for debugging this strategy 	**/
/*****************************************************************/

private _print_bit_vector(s, index, vector_ptr)
char s[1000];
int index;
#ifdef BITVECTOR
unsigned        vector_ptr[BITMAX];
#else
unsigned        vector_ptr;
#endif
{
#ifdef BITVECTOR
        unsigned        *ptr, vector_length;
        int             vector_size;

        vector_length = (*vector_ptr >> 24) & 0x0ff;

        if ( vector_length <= 24 )
                vector_size = 1;
        else
                vector_size = ((vector_length - 25) >> 5) + 2;

        CkPrintf("[%d] %s: index=%d,  priority=", myPE, s, index);
        for (ptr=vector_ptr; ptr < vector_ptr+vector_size; ptr++)
                CkPrintf("%u\t", *ptr);
        CkPrintf("\n");
#else
        CkPrintf("[%d] %s: index=%d,  priority=%d\n", myPE, s, 
			index, vector_ptr);
#endif
}

TRACE(private check_token(s, token)
char s[1000];
TOKEN_TYPE *token;
{
	if (HashFunction(token->priority) != token->index)
	{
		CkPrintf("[%d] *** ERROR *** Index=%d. \n", myPE,
			HashFunction(token->priority));
		PrivateCall(_print_bit_vector(s, token->index, token->priority));
	}
}

)

public ProcessorIdle()
{
}


/*****************************************************************/
/** Initialize Periodic Checks.					**/
/*****************************************************************/
public PeriodicCheckInit()
{
	if (numPe > 1)
	{
		if (CONTROLLER(myPE))
		{
			if (numPe > CLUSTER_SIZE)
				CallBocAfter(LdbPeriodicBossStatus, LdbBoc, 
				    BOSS_STATUS_UPDATE_INTERVAL);
			CallBocAfter(LdbPeriodicRedist, LdbBoc,
			    BOSS_REDIST_UPDATE_INTERVAL);
		}
		else
			CallBocAfter(LdbPeriodicKidStatus, LdbBoc, 
			    KID_STATUS_UPDATE_INTERVAL);
	}
}

/*****************************************************************/
/** Initialize the LDB variables here.				**/
/*****************************************************************/


entry BranchInit : (message DUMMYMSG * dmsg)
{
	int i;
	LDB_ELEMENT *ldb;
	CpvAccess(LdbBocNum) = LdbBoc = MyBocNum();
	CpvAccess(LDB_ELEM_SIZE) = sizeof(LDB_ELEMENT);

	myPE = CmiMyPe();
/* Transferred from LdbBocInit 06/09 */
	PrivateCall(init_heap());
	PrivateCall(init_table());

	TRACE(CkPrintf("Enter Node LdbInit()\n"));

	numPe = CmiNumPe();
	controller = CONTROLLER(myPE);
	mycontroller = PrivateCall(MyController(myPE));

TRACE(CkPrintf("[%d] LdbInit: numPe=%d, controller=%d, mycontroller=%d\n",
myPE, numPe, controller, mycontroller));

	if (numPe%CLUSTER_SIZE == 0)
		numBoss = numPe/CLUSTER_SIZE;
	else
		numBoss = numPe/CLUSTER_SIZE + 1;


	if (controller)
	{


		for (i=0; i<CLUSTER_SIZE; i++)
			load_cluster[i] = 0;

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
				nbr_boss[i] = (CLUSTER_SIZE-1) + boss_index*CLUSTER_SIZE;
			}
		}
	}
	CkFreeMsg(dmsg);
}


/*****************************************************************/
/** This entry point is called to record Nbr. status. Currently	**/
/** I don't think it's ever called.				**/
/*****************************************************************/

entry RecvStatus : (message DUMMYMSG *dmsg)
{
	CkFreeMsg(dmsg);
}

entry INSERT : (message OperationMsg *msgPtr)
{
	LDB_ELEMENT *ldb = LDB_UPTR(msgPtr);
        int index = PrivateCall(get_index(ldb->srcPE));

	PrivateCall(get_status(index, ldb));
	TRACE(CkPrintf("[%d] INSERT: srcPE=%d, priority=%d\n",
	    myPE, ldb->srcPE, msgPtr->token.priority));
	TRACE(PrivateCall(check_token("INSERT", &(msgPtr->token))));
	PrivateCall(TokenStrategy(msgPtr));
	CkFreeMsg(msgPtr);
}

entry DELETE  : (message OperationMsg *msg)
{
	LDB_ELEMENT *ldb;
	void *work;

	TRACE(PrivateCall(check_token("DELETE", &(msg->token))));
	PrivateCall(delete_table(msg->token.priority, &work));

	ldb = LDB_UPTR(work);
	TRACE(CkPrintf("[%d] DELETE: priority=%d, destPE=%d\n",
	    myPE,msg->token.priority, msg->token.pe));
	if (msg->token.pe == myPE)
	  QsEnqUsrMsg(work);
	else
	  SEND_FIXED_TO(work, TRUE, msg->token.pe);
	
	CkFreeMsg(msg);
}

entry  REDISTRIBUTION : (message RedistributionMsg *rmsg)
{
	int i;
	TRACE(CkPrintf("[%d] REDISTRIBUTION: no=%d\n", myPE, rmsg->exchanges));
	for (i=0; i<rmsg->exchanges; i++)
	  PrivateCall(insert_heap(&(rmsg->tokens[i])));
	CkFreeMsg(rmsg);
}

entry STATUS : (message OperationMsg *msg)
{
	LDB_ELEMENT *ldb = LDB_UPTR(msg);
	int index;
	if ((numPe > 1) && (ldb->srcPE != myPE)
	    && (ldb->srcPE != McHostPeNum()))
	  {
	    index = PrivateCall(get_index(ldb->srcPE));
	    TRACE(CkPrintf("[%d] STATUS: srcPE=%d\n", myPE, ldb->srcPE));
	    PrivateCall(get_status(index, ldb));
	  }
	CkFreeMsg(msg);
}

public int ProcessMsg(msgPtr, localdataPtr)
void *msgPtr;
void *localdataPtr;
{	
	CkPrintf("In ProcessMsg - Error -- should'nt be here \n");
	CkFreeMsg(msgPtr);
}


/*****************************************************************/
/** Once a message is recd. this function is called to strip off**/
/** any ldb related information. For this strategy this message **/
/** does more than just provide status information. It doubles  **/
/** up as a message to insert a token, to delete a token and to **/
/** receive redistributed tokens.				**/
/*****************************************************************/
public StripLDB(ldb)
LDB_ELEMENT *ldb;
{
/*   Put the whole lot in LdbProcessMsg */
}


/*****************************************************************/
/** Function is called when a new chare is recd. from the net.	**/
/*****************************************************************/
public NewMsg_FromNet(x)
void *x;
{
	if (controller)
		CkPrintf("*** ERROR *** New chare sent to Manager.\n");
	else
	{
		TRACE(CkPrintf("[%d] Ldb_NewChare_Net:: Message from outside. \n",
		    myPE));
		QsEnqUsrMsg(x);
	}
}


/*****************************************************************/
/** Function is called when a new chare is recd. from local.	**/
/*****************************************************************/
public NewMsg_FromLocal(msg)
void *msg;
{
  unsigned int *priority;

	if (controller)
		CkPrintf("*** ERROR *** New chare created on Manager\n");
	else
	{
		insert_msg = (OperationMsg *)CkAllocMsg(OperationMsg);
		CkMemError(insert_msg);

		priority = (unsigned int *) CkPriorityPtr(msg);
#ifdef BITVECTOR
		AssignPriority(insert_msg->token.priority, priority);
		PrivateCall(insert_table(insert_msg->token.priority, msg));
		TRACE(insert_msg->token.index = HashFunction(insert_msg->token.priority);)

#else

		TRACE(CkPrintf("LdbLocal: priority=%d\n", *priority));
		PrivateCall(insert_table(*priority, msg));
		AssignPriority(insert_msg->token.priority, *priority);
		TRACE(insert_msg->token.index = HashFunction(*priority);)
#endif
		insert_msg->token.pe = myPE;
		LdbFillLDB(LDB_UPTR(insert_msg));

		TRACE(PrivateCall(check_token("from local", &(insert_msg->token)));)

		ImmSendMsgBranch(LDB@INSERT, insert_msg, mycontroller);
	}
}


/*****************************************************************/
/** This function fills in the ldb information before sending	**/
/** out a message.						**/
/*****************************************************************/
public FillLDB(ldb)
LDB_ELEMENT *ldb;
{
	ldb->srcPE = myPE;
	if (CONTROLLER(ldb->srcPE))
		ldb->piggybackLoad = MyLoad();
	else	
		ldb->piggybackLoad = QsMyLoad();
}




/*****************************************************************/
/** Functions to periodically send status. The kids send their 	**/
/** status to their managers, while the managers send their	**/
/** status to their neighbors in the dimensional exchange.	**/
/*****************************************************************/

public void PeriodicKidStatus()
{
	status_msg = (OperationMsg *) CkAllocMsg(OperationMsg);
	CkMemError(status_msg);
/*	TRACE(CkPrintf("[%d] LdbPeriodicKidStatus: load=%d\n", myPE, load)); */
	ImmSendMsgBranch(LDB@STATUS, status_msg, mycontroller);
}



public void PeriodicBossStatus()
{
	long status;
	static int index = 0;
	int load = MyLoad();

	boss_statusMsg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG);
	CkMemError(boss_statusMsg);

	ImmSendMsgBranch(LDB@RecvStatus, boss_statusMsg, nbr_boss[index]);
	index = (index+1) % exchanges;
}


/*****************************************************************/
/** These functions are called to periodically redistribute	**/
/** load among various managers.				**/
/*****************************************************************/


public void PeriodicKidsRedist()
{
	int i, j;
	int index;
	int picked;
	BOOLEAN done;
	TOKEN_TYPE *token;
	int start_pe, end_pe;

	end_pe = myPE;
	start_pe = (end_pe/CLUSTER_SIZE) * CLUSTER_SIZE;
	done = FALSE;


	TRACE(CkPrintf("[%d] LdbPeriodicKidRedist.\n", myPE));

	while (!done)
	{
		picked=0;
		for (i=start_pe; i<end_pe; i++)
		{
			index = PrivateCall(get_index(i));
			if (load_cluster[index] < KID_SATURATION)
			{
				if (PrivateCall(delete_heap(&token)))
				{
					delete_msg = (OperationMsg *) CkAllocMsg(OperationMsg);
					CkMemError(delete_msg);
					delete_msg->token.pe = i;
					TRACE(delete_msg->token.index = token->index);
					CopyPriority(delete_msg->token.priority
						     , token->priority);

					ImmSendMsgBranch(LDB@DELETE, delete_msg,token->pe);
					picked++;
					load_cluster[index]++;
					CmiFree(token);
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



}


/************************************************************************/
/* Hash function for bitvector and integer priorities.  		*/
/************************************************************************/

#ifdef BITVECTOR

HashFunction(bit_vec)
unsigned int *bit_vec;
{
    	int vector_size;
	unsigned int *b_vec;
 	int index_1 = (*bit_vec & 0x0ffffff); 

    	if ( (*bit_vec >> 24) > 24)
       		vector_size =  ((((*bit_vec >> 24) - 25) >> 5) +2) << 2;
    	else
       		vector_size = 4;

    	if ( vector_size > 1 ) 
      		for (b_vec=bit_vec, b_vec++; b_vec<bit_vec+vector_size; b_vec++)
        		index_1 = ( index_1 ^ (*b_vec) ); 
    	if (index_1 < 0)  
        	index_1 = (-index_1);
    	index_1 = index_1 % HASH_TABLE_SIZE;
	return index_1;
}


#else

int TokenGreaterThan(p,q)
unsigned int p, q;
{
	return (p > q);
}

int TokenEqual(p,q)
unsigned int p, q;
{
	return (p == q);
}


HashFunction(bit_vec)
int bit_vec;
{
	return(bit_vec*MULTIPLICAND % HASH_TABLE_SIZE);
}

#endif


}


