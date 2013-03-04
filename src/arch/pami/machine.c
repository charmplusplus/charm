#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "machine.h"
#include "converse.h"
#include "pcqueue.h"
#include "assert.h"
#include "malloc.h"

#include <hwi/include/bqc/A2_inlines.h>
#include "spi/include/kernel/process.h"
#include "spi/include/kernel/memory.h"
#include "pami.h"
#include "pami_sys.h"

#if MACHINE_DEBUG_LOG
FILE *debugLog = NULL;
#endif
//#if CMK_SMP
//#define CMK_USE_L2ATOMICS   1
//#endif

#if !CMK_SMP
#if CMK_ENABLE_ASYNC_PROGRESS
#error "async progress non supported with non-smp"
#endif
#endif


#if CMK_SMP && CMK_USE_L2ATOMICS
#include "L2AtomicQueue.h"
#include "L2AtomicMutex.h"
#include "memalloc.c"
#endif

#define CMI_LIKELY(x)    (__builtin_expect(x,1))
#define CMI_UNLIKELY(x)  (__builtin_expect(x,0))

char *ALIGN_32(char *p) {
  return((char *)((((unsigned long)p)+0x1f) & (~0x1FUL)));
}

/*To reduce the buffer used in broadcast and distribute the load from
  broadcasting node, define CMK_BROADCAST_SPANNING_TREE enforce the use of
  spanning tree broadcast algorithm.
  This will use the fourth short in message as an indicator of spanning tree
  root.
*/
#if CMK_SMP
#define CMK_BROADCAST_SPANNING_TREE    1
#else
#define CMK_BROADCAST_SPANNING_TREE    1
#endif /* CMK_SMP */

#define BROADCAST_SPANNING_FACTOR     4

//The root of the message infers the type of the message
// 1. root is 0, then it is a normal point-to-point message
// 2. root is larger than 0 (>=1), then it is a broadcast message across all processors (cores)
// 3. root is less than 0 (<=-1), then it is a broadcast message across all nodes
#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_IS_BCAST_ON_CORES(msg) (CMI_BROADCAST_ROOT(msg) > 0)
#define CMI_IS_BCAST_ON_NODES(msg) (CMI_BROADCAST_ROOT(msg) < 0)
#define CMI_GET_CYCLE(msg)               ((CmiMsgHeaderBasic *)msg)->root

#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank
#define CMI_MAGIC(msg)                   ((CmiMsgHeaderBasic *)msg)->magic

/* FIXME: need a random number that everyone agrees ! */
#define CHARM_MAGIC_NUMBER               126


#define CMI_PAMI_SHORT_DISPATCH           7
#define CMI_PAMI_RZV_DISPATCH             8
#define CMI_PAMI_ACK_DISPATCH             9
#define CMI_PAMI_DISPATCH                10

#define SHORT_CUTOFF   128
#define EAGER_CUTOFF   4096

#if !CMK_OPTIMIZE
static int checksum_flag = 0;
extern unsigned char computeCheckSum(unsigned char *data, int len);

#define CMI_SET_CHECKSUM(msg, len)      \
        if (checksum_flag)  {   \
          ((CmiMsgHeaderBasic *)msg)->cksum = 0;        \
          ((CmiMsgHeaderBasic *)msg)->cksum = computeCheckSum((unsigned char*)msg, len);        \
        }

#define CMI_CHECK_CHECKSUM(msg, len)    \
  int count; \
        if (checksum_flag)      \
          if (computeCheckSum((unsigned char*)msg, len) != 0)  { \
            printf("\n\n------------------------------\n\nReceiver %d size %d:", CmiMyPe(), len); \
            for(count = 0; count < len; count++) { \
                printf("%2x", msg[count]);                 \
            }                                             \
            printf("------------------------------\n\n"); \
            CmiAbort("Fatal error: checksum doesn't agree!\n"); \
          }
#else
#define CMI_SET_CHECKSUM(msg, len)
#define CMI_CHECK_CHECKSUM(msg, len)
#endif

#define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);

#  define CMI_SET_CYCLE(msg, cycle)

int               _Cmi_numpes;
int               _Cmi_mynode;    /* Which address space am I */
int               _Cmi_mynodesize;/* Number of processors in my address space */
int               _Cmi_numnodes;  /* Total number of address spaces */
int                Cmi_nodestart; /* First processor in this address space */
CpvDeclare(void*, CmiLocalQueue);


#if CMK_NODE_QUEUE_AVAILABLE
#define SMP_NODEMESSAGE   (0xFB) // rank of the node message when node queue
                                 // is available
#define NODE_BROADCAST_OTHERS (-1)
#define NODE_BROADCAST_ALL    (-2)
#endif


typedef struct ProcState {
    /* PCQueue      sendMsgBuf; */  /* per processor message sending queue */
#if CMK_SMP && CMK_USE_L2ATOMICS
    L2AtomicQueue   atomic_queue;
#endif
  /* CmiNodeLock  recvLock;  */            /* for cs->recv */
} ProcState;

static ProcState  *procState;

#if CMK_SMP && CMK_USE_L2ATOMICS
static L2AtomicMutex *node_recv_mutex;
static L2AtomicQueue node_recv_atomic_q;
#endif

#if CMK_SMP && !CMK_MULTICORE
//static volatile int commThdExit = 0;
//static CmiNodeLock commThdExitLock = 0;

//The random seed to pick destination context
__thread uint32_t r_seed = 0xdeadbeef;
__thread int32_t _cmi_bgq_incommthread = 0;
#endif

//int CmiInCommThread () {
//  //if (_cmi_bgq_incommthread)
//  //printf ("CmiInCommThread: %d\n", _cmi_bgq_incommthread);
//  return _cmi_bgq_incommthread;
//}

void ConverseRunPE(int everReturn);
static void CommunicationServer(int sleepTime);
static void CommunicationServerThread(int sleepTime);

static void CmiNetworkBarrier(int async);
static void CmiSendPeer (int rank, int size, char *msg);

//So far we dont define any comm threads
int Cmi_commthread = 0;

#include "machine-smp.c"
CsvDeclare(CmiNodeState, NodeState);
#include "immediate.c"

#if CMK_ENABLE_ASYNC_PROGRESS  
//Immediate messages not supported yet
#define AdvanceCommunications() 
#else
void AdvanceCommunications();
#endif

#if !CMK_SMP
/************ non SMP **************/
static struct CmiStateStruct Cmi_state;
int _Cmi_mype;
int _Cmi_myrank;

void CmiMemLock(void) {}
void CmiMemUnlock(void) {}

#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

//void CmiYield(void) { sleep(0); }

static void CmiStartThreads(char **argv) {
    CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
    _Cmi_mype = Cmi_nodestart;
    _Cmi_myrank = 0;
}
#endif  /* !CMK_SMP */

//int received_immediate;
//int received_broadcast;

void _alias_rank (int rank);

/*Add a message to this processor's receive queue, pe is a rank */
void CmiPushPE(int pe,void *msg) {
    CmiState cs = CmiGetStateN(pe);    
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
      //_alias_rank(CMI_DEST_RANK(msg));
      //CmiLock(CsvAccess(NodeState).immRecvLock);
      CmiHandleImmediateMessage(msg);
      //CmiUnlock(CsvAccess(NodeState).immRecvLock);
      //_alias_rank(0);
      return;
    }
#endif
    
#if CMK_SMP && CMK_USE_L2ATOMICS
    L2AtomicEnqueue(&procState[pe].atomic_queue, msg);
#else
    PCQueuePush(cs->recv,(char *)msg);
#endif
    
    //CmiIdleLock_addMessage(&cs->idle);
}

#if CMK_NODE_QUEUE_AVAILABLE
/*Add a message to this processor's receive queue */
static void CmiPushNode(void *msg) {
    MACHSTATE(3,"Pushing message into NodeRecv queue");
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
      //CmiLock(CsvAccess(NodeState).immRecvLock);
      CmiHandleImmediateMessage(msg);
      //CmiUnlock(CsvAccess(NodeState).immRecvLock);	
      return;
    }
#endif
#if CMK_SMP && CMK_USE_L2ATOMICS
    L2AtomicEnqueue(&node_recv_atomic_q, msg);    
#else
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv,msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
#endif
    //CmiState cs=CmiGetStateN(0);
    //CmiIdleLock_addMessage(&cs->idle);
}
#endif /* CMK_NODE_QUEUE_AVAILABLE */

#define MAX_NUM_CONTEXTS  64

#if CMK_SMP 
#define CMK_PAMI_MULTI_CONTEXT  1
#else
#define CMK_PAMI_MULTI_CONTEXT  0
#endif

#if CMK_PAMI_MULTI_CONTEXT
volatile int msgQueueLen [MAX_NUM_CONTEXTS];
volatile int outstanding_recvs [MAX_NUM_CONTEXTS];

//#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
//#define THREADS_PER_CONTEXT 2
//#define LTPS                1 //Log Threads Per Context (TPS)
//#else
#define THREADS_PER_CONTEXT 4
#define LTPS                2 //Log Threads Per Context (TPS)
//#endif

#define  MY_CONTEXT_ID() (CmiMyRank() >> LTPS)
#define  MY_CONTEXT()    (cmi_pami_contexts[CmiMyRank() >> LTPS])

#define  INCR_MSGQLEN()  //(msgQueueLen[CmiMyRank() >> LTPS] ++)
#define  DECR_MSGQLEN()  //(msgQueueLen[CmiMyRank() >> LTPS] --)
#define  MSGQLEN()       0 //(msgQueueLen[CmiMyRank() >> LTPS])
#define  INCR_ORECVS()   //(outstanding_recvs[CmiMyRank() >> LTPS] ++)
#define  DECR_ORECVS()   //(outstanding_recvs[CmiMyRank() >> LTPS] --)
#define  ORECVS()        0 //(outstanding_recvs[CmiMyRank() >> LTPS])
#else
#define LTPS    1 
volatile int msgQueueLen;
volatile int outstanding_recvs;
#define  MY_CONTEXT_ID() (0)
#define  MY_CONTEXT()    (cmi_pami_contexts[0])

#define  INCR_MSGQLEN()  (msgQueueLen ++)
#define  DECR_MSGQLEN()  (msgQueueLen --)
#define  MSGQLEN()       (msgQueueLen)
#define  INCR_ORECVS()   (outstanding_recvs ++)
#define  DECR_ORECVS()   (outstanding_recvs --)
#define  ORECVS()        (outstanding_recvs)
#endif

#if CMK_SMP  && !CMK_ENABLE_ASYNC_PROGRESS
#define PAMIX_CONTEXT_LOCK_INIT(x)
#define PAMIX_CONTEXT_LOCK(x)        if(LTPS) PAMI_Context_lock(x)
#define PAMIX_CONTEXT_UNLOCK(x)      if(LTPS) {ppc_msync(); PAMI_Context_unlock(x);}
#define PAMIX_CONTEXT_TRYLOCK(x)     ((LTPS)?(PAMI_Context_trylock(x) == PAMI_SUCCESS):(1))
#else
#define PAMIX_CONTEXT_LOCK_INIT(x)
#define PAMIX_CONTEXT_LOCK(x)
#define PAMIX_CONTEXT_UNLOCK(x)
#define PAMIX_CONTEXT_TRYLOCK(x)      1
#endif


static char     **Cmi_argv;
static char     **Cmi_argvcopy;
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */

extern void ConverseCommonInit(char **argv);
extern void ConverseCommonExit(void);
extern void CthInit(char **argv);

static void SendMsgsUntil(int);

#define A_PRIME 13
#define B_PRIME 19

static inline unsigned myrand (unsigned *seed) {
  *seed = A_PRIME * (*seed) + B_PRIME;
  return *seed;
}

void SendSpanningChildren(int size, char *msg, int from_rdone);
#if CMK_NODE_QUEUE_AVAILABLE
void SendSpanningChildrenNode(int size, char *msg, int from_rdone);
#endif

typedef struct {
    int sleepMs; /*Milliseconds to sleep while idle*/
    int nIdles; /*Number of times we've been idle in a row*/
    CmiState cs; /*Machine state*/
} CmiIdleState;

static CmiIdleState *CmiNotifyGetState(void) {
    CmiIdleState *s=(CmiIdleState *)CmiAlloc(sizeof(CmiIdleState));
    s->sleepMs=0;
    s->nIdles=0;
    s->cs=CmiGetState();
    return s;
}

static void send_done(pami_context_t ctxt, void *data, pami_result_t result) 
{
  CmiFree(data);
  DECR_MSGQLEN();
}


static void recv_done(pami_context_t ctxt, void *clientdata, pami_result_t result) 
/* recv done callback: push the recved msg to recv queue */
{
    char *msg = (char *) clientdata;
    int sndlen = ((CmiMsgHeaderBasic *) msg)->size;
    //int rank = *(int *) (msg + sndlen); //get rank from bottom of the message
    //CMI_DEST_RANK(msg) = rank;

    //fprintf (stderr, "%d Recv message done \n", CmiMyPe());
    /* then we do what PumpMsgs used to do:
     * push msg to recv queue */
    CMI_CHECK_CHECKSUM(msg, sndlen);
    if (CMI_MAGIC(msg) != CHARM_MAGIC_NUMBER) { /* received a non-charm msg */
        CmiAbort("Charm++ Warning: Non Charm++ Message Received. If your application has a large number of messages, this may be because of overflow in the low-level FIFOs. Please set the environment variable MUSPI_INJFIFOSIZE if the application has large number of small messages (<=4K bytes), and/or PAMI_RGETINJFIFOSIZE if the application has a large number of large messages. The default value of these variable is 65536 which is sufficient for 1000 messages in flight; please try a larger value. Please note that the memory used for these FIFOs eats up the memory = 10*FIFO_SIZE per core. Please contact Charm++ developers for further information. \n");     
        return;
    }

#if CMK_BROADCAST_SPANNING_TREE 
    if (CMI_IS_BCAST_ON_CORES(msg) ) 
        //Forward along spanning tree
        SendSpanningChildren(sndlen, msg, 1);
#endif

#if CMK_NODE_QUEUE_AVAILABLE
#if CMK_BROADCAST_SPANNING_TREE
    if (CMI_IS_BCAST_ON_NODES(msg)) 
      SendSpanningChildrenNode(sndlen, msg, 1);
#endif
    if (CMI_DEST_RANK(msg) == SMP_NODEMESSAGE)
      CmiPushNode(msg);
    else
#endif
      CmiPushPE(CMI_DEST_RANK(msg), (void *)msg);

    DECR_ORECVS();
}

typedef struct _cmi_pami_rzv {
  void           * buffer;
  size_t           offset;
  int              bytes;
  int              dst_context;
  pami_memregion_t mregion;
}CmiPAMIRzv_t;  

typedef struct _cmi_pami_rzv_recv {
  void           * msg;
  void           * src_buffer;
  int              src_ep;
  int              size;
  pami_memregion_t rmregion;
} CmiPAMIRzvRecv_t;

static void pkt_dispatch (pami_context_t       context,      
			  void               * clientdata,   
			  const void         * header_addr,  
			  size_t               header_size,  
			  const void         * pipe_addr,    
			  size_t               pipe_size,    
			  pami_endpoint_t      origin,
			  pami_recv_t         * recv)        
{
    //fprintf (stderr, "Received Message of size %d %p\n", pipe_size, recv);
    INCR_ORECVS();    
    int alloc_size = pipe_size;
    char * buffer  = (char *)CmiAlloc(alloc_size);
    //char * buffer  = (char *)CmiAlloc(alloc_size + sizeof(int));
    //*(int *)(buffer+alloc_size) = *(int *)header_addr;

    if (recv) {
      recv->local_fn = recv_done;
      recv->cookie   = buffer;
      recv->type     = PAMI_TYPE_BYTE;
      recv->addr     = buffer;
      recv->offset   = 0;
      recv->data_fn  = PAMI_DATA_COPY;
    }
    else {
      memcpy (buffer, pipe_addr, pipe_size);
      recv_done (NULL, buffer, PAMI_SUCCESS);
    }
}

static void short_pkt_dispatch (pami_context_t       context,      
				void               * clientdata,   
				const void         * header_addr,  
				size_t               header_size,  
				const void         * pipe_addr,    
				size_t               pipe_size,    
				pami_endpoint_t      origin,
				pami_recv_t         * recv)        
{
  int alloc_size = pipe_size;
  char * buffer  = (char *)CmiAlloc(alloc_size);
  //char * buffer  = (char *)CmiAlloc(alloc_size + sizeof(int));
  //*(int *)(buffer+alloc_size) = *(int *)header_addr;
  
  memcpy (buffer, pipe_addr, pipe_size);
  char *smsg = (char *)pipe_addr;
  char *msg  = (char *)buffer;

  CMI_CHECK_CHECKSUM(smsg, pipe_size);  
  if (CMI_MAGIC(smsg) != CHARM_MAGIC_NUMBER) {
    /* received a non-charm msg */
    CmiAbort("Charm++ Warning: Non Charm++ Message Received. If your application has a large number of messages, this may be because of overflow in the low-level FIFOs. Please set the environment variable MUSPI_INJFIFOSIZE if the application has large number of small messages (<=4K bytes), and/or PAMI_RGETINJFIFOSIZE if the application has a large number of large messages. The default value of these variable is 65536 which is sufficient for 1000 messages in flight; please try a larger value. Please note that the memory used for these FIFOs eats up the memory = 10*FIFO_SIZE per core. Please contact Charm++ developers for further information. \n");     
  }
 
  CmiPushPE(CMI_DEST_RANK(smsg), (void *)msg);
}


void rzv_pkt_dispatch (pami_context_t       context,   
		       void               * clientdata,
		       const void         * header_addr,
		       size_t               header_size,
		       const void         * pipe_addr,  
		       size_t               pipe_size,  
		       pami_endpoint_t      origin,
		       pami_recv_t         * recv);

void ack_pkt_dispatch (pami_context_t       context,   
		       void               * clientdata,
		       const void         * header_addr,
		       size_t               header_size,
		       const void         * pipe_addr,  
		       size_t               pipe_size,  
		       pami_endpoint_t      origin,
		       pami_recv_t         * recv);

void rzv_recv_done   (pami_context_t     ctxt, 
		      void             * clientdata, 
		      pami_result_t      result); 

//approx sleep command
size_t mysleep_iter = 0;
void mysleep (unsigned long cycles) {
    unsigned long start = GetTimeBase();
    unsigned long end = start + cycles;

    while (start < end) {
      mysleep_iter ++;
      start = GetTimeBase();
    }

    return;
}

static void * test_buf;
volatile int pami_barrier_flag = 0;
typedef pami_result_t (*pamix_proc_memalign_fn) (void**, size_t, size_t, const char*);

void pami_barrier_done (void *ctxt, void * clientdata, pami_result_t err)
{
  int * active = (int *) clientdata;
  (*active)--;
}

pami_client_t      cmi_pami_client;
pami_context_t   * cmi_pami_contexts;
size_t             cmi_pami_numcontexts;
pami_geometry_t    world_geometry;
pami_xfer_t        pami_barrier;
char clientname[] = "Converse";

#if 1
typedef struct _cmi_pami_mregion_t {
  pami_memregion_t   mregion;
  void             * baseVA;
} CmiPAMIMemRegion_t;

//one for each of the 64 possible contexts
CmiPAMIMemRegion_t  cmi_pami_memregion[64];
#endif

#include "malloc.h"
void *l2atomicbuf;

void _alias_rank (int rank) {
#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS

  CmiState cs = CmiGetState();
  CmiState cs_r = CmiGetStateN(rank);

  cs->rank = cs_r->rank;
  cs->pe   = cs_r->pe;
#endif
}

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS

pami_result_t init_comm_thread (pami_context_t   context,
				void           * cookie)
{
  CmiState cs  = CmiGetState();
  CmiState cs0 = CmiGetStateN(0);
  *cs = *cs0; //Alias comm thread to rank 0
  //printf("Initialized comm thread, my rank %d, my pe %d\n", 
  // CmiMyRank(), 
  // CmiMyPe());

  //Notify main thread comm thread has been initialized
  *(int*)cookie = 0;

#if 1
  //set the seed to choose destination context
  uint64_t rseedl = r_seed;
  rseedl |= (uint64_t)context;
  r_seed = ((uint32_t)rseedl)^((uint32_t)(rseedl >> 32));
#endif

  _cmi_bgq_incommthread = 1;

  return PAMI_SUCCESS;
}

typedef void (*pamix_progress_function) (pami_context_t context, void *cookie);
typedef pami_result_t (*pamix_progress_register_fn) 
  (pami_context_t            context,
   pamix_progress_function   progress_fn,
   pamix_progress_function   suspend_fn,
   pamix_progress_function   resume_fn,
   void                     * cookie);
typedef pami_result_t (*pamix_progress_enable_fn)(pami_context_t   context,
						  int              event_type);
typedef pami_result_t (*pamix_progress_disable_fn)(pami_context_t  context,
						   int             event_type);
#define PAMI_EXTENSION_OPEN(client, name, ext)  \
({                                              \
  pami_result_t rc;                             \
  rc = PAMI_Extension_open(client, name, ext);  \
  CmiAssert (rc == PAMI_SUCCESS);      \
})
#define PAMI_EXTENSION_FUNCTION(type, name, ext)        \
({                                                      \
  void* fn;                                             \
  fn = PAMI_Extension_symbol(ext, name);                \
  CmiAssert (fn != NULL);				\
  (type)fn;                                             \
})

pami_extension_t            cmi_ext_progress;
pamix_progress_register_fn  cmi_progress_register;
pamix_progress_enable_fn    cmi_progress_enable;
pamix_progress_disable_fn   cmi_progress_disable;

int CMI_Progress_init(int start, int ncontexts) {
  if (CmiMyPe() == 0)
    printf("Enabling communication threads\n");
  
  PAMI_EXTENSION_OPEN(cmi_pami_client,"EXT_async_progress",&cmi_ext_progress);
  cmi_progress_register = PAMI_EXTENSION_FUNCTION(pamix_progress_register_fn, "register", cmi_ext_progress);
  cmi_progress_enable   = PAMI_EXTENSION_FUNCTION(pamix_progress_enable_fn,   "enable",   cmi_ext_progress);
  cmi_progress_disable  = PAMI_EXTENSION_FUNCTION(pamix_progress_disable_fn,  "disable",  cmi_ext_progress);
  
  int i = 0;
  for (i = start; i < start+ncontexts; ++i) {
    //fprintf(stderr, "Enabling progress on context %d\n", i);
    cmi_progress_register (cmi_pami_contexts[i], 
			   NULL, 
			   NULL, 
			   NULL, NULL);
    cmi_progress_enable   (cmi_pami_contexts[i], 0 /*progress all*/);  
  }

  pami_work_t  work;
  volatile int x;
  for (i = start; i < start+ncontexts; ++i) {
    x = 1;
    PAMI_Context_post(cmi_pami_contexts[i], &work, 
		      init_comm_thread, (void*)&x);
    while(x);
  }
  
  return 0;
}

int CMI_Progress_finalize(int start, int ncontexts) {
  int i = 0;
  for (i = start; i < start+ncontexts; ++i) 
    cmi_progress_disable  (cmi_pami_contexts[i], 0 /*progress all*/);    
  PAMI_Extension_close (cmi_ext_progress);
}
#endif

#include "manytomany.c"

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret) {
    int n, i, count;

    /* processor per node */
    _Cmi_mynodesize = 1;
    CmiGetArgInt(argv,"+ppn", &_Cmi_mynodesize);
#if ! CMK_SMP
    if (_Cmi_mynodesize > 1 && _Cmi_mynode == 0)
      CmiAbort("+ppn cannot be used in non SMP version!\n");
#endif
    
    PAMI_Client_create (clientname, &cmi_pami_client, NULL, 0);
    size_t _n = 1;
#if CMK_PAMI_MULTI_CONTEXT
    if ((_Cmi_mynodesize % THREADS_PER_CONTEXT) == 0)
      _n = _Cmi_mynodesize / THREADS_PER_CONTEXT;  //have a context for each four threads
    else
      _n = 1 + (_Cmi_mynodesize / THREADS_PER_CONTEXT);  //have a context for each four threads
#endif

    cmi_pami_contexts = (pami_context_t *) malloc (sizeof(pami_context_t) * _n);
    pami_result_t rc = PAMI_Context_createv (cmi_pami_client, NULL, 0, cmi_pami_contexts, _n);
    if (rc != PAMI_SUCCESS) {
      fprintf(stderr, "PAMI_Context_createv failed for %d contexts\n", _n);
      assert(0);
    }
    cmi_pami_numcontexts = _n;

    //fprintf(stderr,"Creating %d pami contexts\n", _n);

    pami_configuration_t configuration;
    pami_result_t result;
    
    configuration.name = PAMI_CLIENT_TASK_ID;
    result = PAMI_Client_query(cmi_pami_client, &configuration, 1);
    _Cmi_mynode = configuration.value.intval;

    configuration.name = PAMI_CLIENT_NUM_TASKS;
    result = PAMI_Client_query(cmi_pami_client, &configuration, 1);
    _Cmi_numnodes = configuration.value.intval;
#if MACHINE_DEBUG_LOG
    char ln[200];
    sprintf(ln,"debugLog.%d", _Cmi_mynode);
    debugLog=fopen(ln,"w");
    if (debugLog == NULL)
    {
        CmiAbort("debug file not open\n");
    }
#endif


    pami_dispatch_hint_t options = (pami_dispatch_hint_t) {0};
    pami_dispatch_callback_function pfn;
    for (i = 0; i < _n; ++i) {
      pfn.p2p = pkt_dispatch;
      PAMI_Dispatch_set (cmi_pami_contexts[i],
			 CMI_PAMI_DISPATCH,
			 pfn,
			 NULL,
			 options);
      
      pfn.p2p = ack_pkt_dispatch;
      PAMI_Dispatch_set (cmi_pami_contexts[i],
			 CMI_PAMI_ACK_DISPATCH,
			 pfn,
			 NULL,
			 options);
      
      pfn.p2p = rzv_pkt_dispatch;
      PAMI_Dispatch_set (cmi_pami_contexts[i],
			 CMI_PAMI_RZV_DISPATCH,
			 pfn,
			 NULL,
			 options);      

      pfn.p2p = short_pkt_dispatch;
      PAMI_Dispatch_set (cmi_pami_contexts[i],
			 CMI_PAMI_SHORT_DISPATCH,
			 pfn,
			 NULL,
			 options);      
    }

#if 1
    size_t bytes_out;
    void * buf = malloc(sizeof(long));    
    uint32_t retval;
    Kernel_MemoryRegion_t k_mregion;
    retval = Kernel_CreateMemoryRegion (&k_mregion, buf, sizeof(long));
    assert(retval==0);  
    for (i = 0; i < _n; ++i) {
      cmi_pami_memregion[i].baseVA = k_mregion.BaseVa;
      retval = PAMI_Memregion_create (cmi_pami_contexts[i],
				      k_mregion.BaseVa,
				      k_mregion.Bytes,
				      &bytes_out,
				      &cmi_pami_memregion[i].mregion);
      assert(retval == PAMI_SUCCESS);
      assert (bytes_out == k_mregion.Bytes);
    }
    free(buf);
#endif

    //fprintf(stderr, "%d Initializing Converse PAMI machine Layer on %d tasks\n", _Cmi_mynode, _Cmi_numnodes);

    ///////////---------------------------------/////////////////////
    //////////----------- Initialize Barrier -------////////////////
    size_t               num_algorithm[2];
    pami_algorithm_t    *always_works_algo = NULL;
    pami_metadata_t     *always_works_md = NULL;
    pami_algorithm_t    *must_query_algo = NULL;
    pami_metadata_t     *must_query_md = NULL;
    pami_xfer_type_t     xfer_type = PAMI_XFER_BARRIER;

    /* Docs01:  Get the World Geometry */
    result = PAMI_Geometry_world (cmi_pami_client,&world_geometry);
    if (result != PAMI_SUCCESS)
      {
	fprintf (stderr, "Error. Unable to get world geometry: result = %d\n", result);
	return;
      }

    result = PAMI_Geometry_algorithms_num(world_geometry,
					  xfer_type,
					  (size_t*)num_algorithm);

    if (result != PAMI_SUCCESS || num_algorithm[0]==0)
      {
	fprintf (stderr,
		 "Error. Unable to query algorithm, or no algorithms available result = %d\n",
		 result);
	return;
      }

    always_works_algo = (pami_algorithm_t*)malloc(sizeof(pami_algorithm_t)*num_algorithm[0]);
    always_works_md  = (pami_metadata_t*)malloc(sizeof(pami_metadata_t)*num_algorithm[0]);
    must_query_algo   = (pami_algorithm_t*)malloc(sizeof(pami_algorithm_t)*num_algorithm[1]);
    must_query_md    = (pami_metadata_t*)malloc(sizeof(pami_metadata_t)*num_algorithm[1]);

    /* Docs05:  Query the algorithm lists */
    result = PAMI_Geometry_algorithms_query(world_geometry,
					    xfer_type,
					    always_works_algo,
					    always_works_md,
					    num_algorithm[0],
					    must_query_algo,
					    must_query_md,
					    num_algorithm[1]);
    
    int opt_alg = 0, nalg = 0;
    for (nalg = 0; nalg < num_algorithm[0]; ++nalg)
      if (strstr(always_works_md[nalg].name, "GI") != NULL) {
	opt_alg = nalg;
	break;
      }

    if (_Cmi_mynode == 0)
      printf ("Choosing optimized barrier algorithm name %s\n",
	      always_works_md[opt_alg]);
	      
    pami_barrier.cb_done   = pami_barrier_done;
    pami_barrier.cookie    = (void*) & pami_barrier_flag;
    pami_barrier.algorithm = always_works_algo[opt_alg];

    /* Docs06:  Query the algorithm lists */
    if (result != PAMI_SUCCESS)
      {
	fprintf (stderr, "Error. Unable to get query algorithm. result = %d\n", result);
	return;
      }

    CmiNetworkBarrier(0);
    CmiNetworkBarrier(0);
    CmiNetworkBarrier(0);

    _Cmi_numpes = _Cmi_numnodes * _Cmi_mynodesize;
    Cmi_nodestart = _Cmi_mynode * _Cmi_mynodesize;
    Cmi_argvcopy = CmiCopyArgs(argv);
    Cmi_argv = argv;
    Cmi_startfn = fn;
    Cmi_usrsched = usched;

    /* checksum flag */
    if (CmiGetArgFlag(argv,"+checksum")) {
#if !CMK_OPTIMIZE
        checksum_flag = 1;
        if (_Cmi_mynode == 0) CmiPrintf("Charm++: CheckSum checking enabled! \n");
#else
        if (_Cmi_mynode == 0) CmiPrintf("Charm++: +checksum ignored in optimized version! \n");
#endif
    }

    CsvInitialize(CmiNodeState, NodeState);
    CmiNodeStateInit(&CsvAccess(NodeState));

#if CMK_SMP && CMK_USE_L2ATOMICS
    //max available hardware threads
    int actualNodeSize = 64/Kernel_ProcessCount(); 
    //printf("Ranks per node %d, actualNodeSize %d CmiMyNodeSize() %d\n",
    //	   Kernel_ProcessCount(), actualNodeSize, _Cmi_mynodesize);
    
    //pami_result_t rc;
    pami_extension_t l2;
    pamix_proc_memalign_fn PAMIX_L2_proc_memalign;
    size_t size = (_Cmi_mynodesize + 2*actualNodeSize + 1) 
      * sizeof(L2AtomicState) + sizeof(L2AtomicMutex);

    rc = PAMI_Extension_open(NULL, "EXT_bgq_l2atomic", &l2);
    CmiAssert (rc == 0);
    PAMIX_L2_proc_memalign = (pamix_proc_memalign_fn)PAMI_Extension_symbol(l2, "proc_memalign");
    rc = PAMIX_L2_proc_memalign(&l2atomicbuf, 64, size, NULL);
    CmiAssert (rc == 0);    
#endif

    char *l2_start = (char *) l2atomicbuf;
    procState = (ProcState *)malloc((_Cmi_mynodesize) * sizeof(ProcState));
    for (i=0; i<_Cmi_mynodesize; i++) {
#if CMK_SMP && CMK_USE_L2ATOMICS
	L2AtomicQueueInit (l2_start + sizeof(L2AtomicState)*i,
			   sizeof(L2AtomicState),
			   &procState[i].atomic_queue,
			   1, /*use overflow*/
			   DEFAULT_SIZE /*1024 entries*/);
#endif
    }

#if CMK_SMP && CMK_USE_L2ATOMICS    
    l2_start += _Cmi_mynodesize * sizeof(L2AtomicState);
    CmiMemAllocInit_bgq (l2_start, 2*actualNodeSize*sizeof(L2AtomicState)); 
    l2_start += 2*actualNodeSize*sizeof(L2AtomicState); 

    L2AtomicQueueInit (l2_start,
		       sizeof(L2AtomicState),
		       &node_recv_atomic_q,
		       1, /*use overflow*/
		       DEFAULT_SIZE /*1024 entries*/);	 
    l2_start += sizeof(L2AtomicState);  
    node_recv_mutex = L2AtomicMutexInit(l2_start, sizeof(L2AtomicMutex));   
#endif
    
    //Initialize the manytomany api
    _cmidirect_m2m_initialize (cmi_pami_contexts, _n);

    //printf ("Starting Threads\n");
    CmiStartThreads(argv);

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
    CMI_Progress_init(0, _n);
#endif

    ConverseRunPE(initret);
}


int PerrorExit (char *err) {
  fprintf (stderr, "err\n\n");
    exit (-1);
    return -1;
}


void ConverseRunPE(int everReturn) {
    //    printf ("ConverseRunPE on rank %d\n", CmiMyPe());    

    CmiIdleState *s=CmiNotifyGetState();
    CmiState cs;
    char** CmiMyArgv;
    CmiNodeAllBarrier();

    cs = CmiGetState();
    CpvInitialize(void *,CmiLocalQueue);
    CpvAccess(CmiLocalQueue) = cs->localqueue;

    if (CmiMyRank())
        CmiMyArgv=CmiCopyArgs(Cmi_argvcopy);
    else
        CmiMyArgv=Cmi_argv;

    CthInit(CmiMyArgv);

    //printf ("Before Converse Common Init\n");
    ConverseCommonInit(CmiMyArgv);

    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);

    //printf ("before calling CmiBarrier() \n");
    CmiBarrier();

    /* Converse initialization finishes, immediate messages can be processed.
       node barrier previously should take care of the node synchronization */
    _immediateReady = 1;

    //printf("calling the startfn\n");

    if (!everReturn) {
      Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
      if (Cmi_usrsched==0) CsdScheduler(-1);
      ConverseExit();
    }    
}

#define MAX_BARRIER_THREADS 64
volatile unsigned char spin_bar_flag[2][MAX_BARRIER_THREADS];
unsigned char spin_iter[MAX_BARRIER_THREADS];
volatile unsigned char done_flag[2];

//slow barrier for ckexit that calls advance while spinning
void spin_wait_barrier () {
  int i = 0;
  int iter = spin_iter[CmiMyRank()];
  int iter_1 = (iter + 1) % 2;

  //start barrier 
  //reset next barrier iteration before this one starts
  done_flag[iter_1] = 0;
  spin_iter[CmiMyRank()] = iter_1;
  spin_bar_flag[iter_1][CmiMyRank()] = 0;
  
  //Notify arrival
  spin_bar_flag[iter][CmiMyRank()] = 1;

  if (CmiMyRank() == 0) {
    while (!done_flag[iter]) {
      for (i = 0; i < CmiMyNodeSize(); ++i)
	if (spin_bar_flag[iter][i] == 0)
	  break;
      if (i >= CmiMyNodeSize()) 
	done_flag[iter] = 1;
      AdvanceCommunications();
    }        
  }
  else {
    while (!done_flag[iter])
      AdvanceCommunications();
  }

  //barrier complete
}

void ConverseExit(void) {

  while (MSGQLEN() > 0 || ORECVS() > 0) {
    AdvanceCommunications();
  }

#if CMK_SMP
  spin_wait_barrier(); //barrier with advance
  CmiNodeBarrier();    //barrier w/o advance to wait for all advance 
  //calls to complete 
#else    
  CmiNodeBarrier();
#endif

  ConverseCommonExit();

  if (CmiMyPe() == 0) {
    printf("End of program\n");
  }

  int rank0 = 0;
  if (CmiMyRank() == 0) {
      rank0 = 1;
#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
      CMI_Progress_finalize(0, cmi_pami_numcontexts);
#endif
      PAMI_Context_destroyv(cmi_pami_contexts, cmi_pami_numcontexts);
      PAMI_Client_destroy(&cmi_pami_client);
    }

#if CMK_SMP
  CmiNodeBarrier();
  if (rank0) {
    Delat(100000);
    exit(0); 
  }
  else
    pthread_exit(0);
#else
  exit(0);
#endif
}

/* exit() called on any node would abort the whole program */
void CmiAbort(const char * message) {
    CmiError("------------- Processor %d Exiting: Called CmiAbort ------------\n"
             "{snd:%d,rcv:%d} Reason: %s\n",CmiMyPe(),
             MSGQLEN(), ORECVS(), message);

    //CmiPrintStackTrace(0);
    //while (msgQueueLen > 0 || outstanding_recvs > 0) {
    //  AdvanceCommunications();
    //}    
    //CmiBarrier();
    assert (0);
}

#if CMK_NODE_QUEUE_AVAILABLE
char *CmiGetNonLocalNodeQ(void) {
    //CmiState cs = CmiGetState();
    char *result = 0;
    //CmiIdleLock_checkMessage(&cs->idle);

#if CMK_SMP && CMK_USE_L2ATOMICS
    if (!L2AtomicQueueEmpty(&node_recv_atomic_q)) {
      if (L2AtomicMutexTryAcquire(node_recv_mutex) == 0) {
	result = (char*)L2AtomicDequeue(&node_recv_atomic_q);
	L2AtomicMutexRelease(node_recv_mutex);
      }
    }
#else
    if (!PCQueueEmpty(CsvAccess(NodeState).NodeRecv)) {
      MACHSTATE1(3,"CmiGetNonLocalNodeQ begin %d {", CmiMyPe());
      
      if (CmiTryLock(CsvAccess(NodeState).CmiNodeRecvLock) == 0) {
	//CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
	result = (char *) PCQueuePop(CsvAccess(NodeState).NodeRecv);
	CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
      }

      MACHSTATE1(3,"} CmiGetNonLocalNodeQ end %d ", CmiMyPe());
    }
#endif
    
    return result;
}
#endif


void *CmiGetNonLocal() {

    void *msg = NULL;
    CmiState cs = CmiGetState();
    //CmiIdleLock_checkMessage(&cs->idle);

#if CMK_SMP && CMK_USE_L2ATOMICS
    msg = L2AtomicDequeue(&procState[CmiMyRank()].atomic_queue);
#if !(CMK_ENABLE_ASYNC_PROGRESS)
    if (msg == NULL) {
      AdvanceCommunications();     
      msg = L2AtomicDequeue(&procState[CmiMyRank()].atomic_queue);
    }
#endif
#else
    if (PCQueueLength(cs->recv)==0)
      AdvanceCommunications();
    if (PCQueueLength(cs->recv)==0) return NULL;
    msg =  PCQueuePop(cs->recv);
#endif

    //if (msg != NULL)
    //fprintf(stderr, "%d: Returning a message\n", CmiMyPe());

    return msg;
}

static void CmiSendSelf(char *msg) {
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
      //CmiLock(CsvAccess(NodeState).immRecvLock);
      CmiHandleImmediateMessage(msg);
      //CmiUnlock(CsvAccess(NodeState).immRecvLock);
      return;
    }
#endif
    
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
}

#if CMK_SMP
static void CmiSendPeer (int rank, int size, char *msg) {
    //fprintf(stderr, "%d Send messages to peer\n", CmiMyPe());
    CmiPushPE (rank, msg);
}
#endif


void CmiGeneralFreeSendN (int node, int rank, int size, char * msg, int to_lock);


/* The general free send function
 * Send is synchronous, and free msg after posted
 */
void  CmiGeneralFreeSend(int destPE, int size, char* msg) {

    if (destPE < 0 || destPE > CmiNumPes ())
      printf ("Sending to %d\n", destPE);
    
    CmiAssert (destPE >= 0 && destPE < CmiNumPes());    
    CmiState cs = CmiGetState();

    if (destPE==cs->pe) {
        CmiSendSelf(msg);
        return;
    }

    CmiGeneralFreeSendN (CmiNodeOf (destPE), CmiRankOf (destPE), size, msg, 1);
}


pami_result_t machine_send_handoff (pami_context_t context, void *msg);
void  machine_send       (pami_context_t      context, 
			  int                 node, 
			  int                 rank, 
			  int                 size, 
			  char              * msg, 
			  int                 to_lock)__attribute__((always_inline));

void CmiGeneralFreeSendN(int node, int rank, int size, char * msg, int to_lock)
{
#if CMK_SMP
    CMI_DEST_RANK(msg) = rank;
    if (node == CmiMyNode()) {
        CmiSendPeer (rank, size, msg);
        return;
    }
#endif

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
    int c = node % cmi_pami_numcontexts;
    //int c = myrand(&r_seed) % cmi_pami_numcontexts;
    pami_context_t my_context = cmi_pami_contexts[c];    
    CmiMsgHeaderBasic *hdr = (CmiMsgHeaderBasic *)msg;
    hdr->dstnode = node;
    hdr->rank    = rank;

    PAMI_Context_post(my_context, (pami_work_t *)hdr->work, 
		      machine_send_handoff, msg);
#else
    pami_context_t my_context = MY_CONTEXT();    
    machine_send (my_context, node, rank, size, msg, to_lock);
#endif
}

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
pami_result_t machine_send_handoff (pami_context_t context, void *msg) {
  CmiMsgHeaderBasic *hdr = (CmiMsgHeaderBasic *)msg;
  int node = hdr->dstnode;
  int rank = hdr->rank;
  int size = hdr->size;
  
  //As this is executed on the comm thread no locking is necessary
  machine_send(context, node, rank, size, msg, 0);
  return PAMI_SUCCESS;
}
#endif

void  machine_send       (pami_context_t      context, 
			  int                 node, 
			  int                 rank, 
			  int                 size, 
			  char              * msg, 
			  int                 to_lock) 
{
    CMI_DEST_RANK(msg) = rank;

    pami_endpoint_t target;
#if CMK_PAMI_MULTI_CONTEXT
    //size_t dst_context = (rank != SMP_NODEMESSAGE) ? (rank>>LTPS) : (rand_r(&r_seed) % cmi_pami_numcontexts);
    //Choose a context at random
    size_t dst_context = myrand(&r_seed) % cmi_pami_numcontexts;
#else
    size_t dst_context = 0;
#endif
    PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)node, dst_context, &target);
    
    //fprintf (stderr, "Calling PAMI Send to %d magic %d size %d\n", node, CMI_MAGIC(msg), size);
    if (CMI_LIKELY(size < SHORT_CUTOFF)) {
      pami_send_immediate_t parameters;
      
      parameters.dispatch        = CMI_PAMI_DISPATCH;
      if ( CMI_LIKELY(CMI_BROADCAST_ROOT(msg) == 0))
#if CMK_NODE_QUEUE_AVAILABLE
	if ( CMI_LIKELY(rank != SMP_NODEMESSAGE) )
#endif
	  //use short callback if not a bcast and not an SMP node message
	  parameters.dispatch        = CMI_PAMI_SHORT_DISPATCH;

      parameters.header.iov_base = NULL; //&rank;
      parameters.header.iov_len  = 0;    //sizeof(int);
      parameters.data.iov_base   = msg;
      parameters.data.iov_len    = size;
      parameters.dest = target;
      
      if(to_lock)
	PAMIX_CONTEXT_LOCK(context);
      
      PAMI_Send_immediate (context, &parameters);
      
      if(to_lock)
	PAMIX_CONTEXT_UNLOCK(context);
      CmiFree(msg);
    }
    else if (size < EAGER_CUTOFF) {
      pami_send_t parameters;
      parameters.send.dispatch        = CMI_PAMI_DISPATCH;
      parameters.send.header.iov_base = NULL; //&rank;
      parameters.send.header.iov_len  = 0;    //sizeof(int);
      parameters.send.data.iov_base   = msg;
      parameters.send.data.iov_len    = size;
      parameters.events.cookie        = msg;
      parameters.events.local_fn      = send_done;
      parameters.events.remote_fn     = NULL;
      memset(&parameters.send.hints, 0, sizeof(parameters.send.hints));
      parameters.send.dest = target;
      
      if (to_lock)
	PAMIX_CONTEXT_LOCK(context);
      INCR_MSGQLEN();
      PAMI_Send (context, &parameters);
      if (to_lock)
	PAMIX_CONTEXT_UNLOCK(context);
    }
    else {
      CmiPAMIRzv_t   rzv;
      rzv.bytes       = size;
      rzv.buffer      = msg;
      rzv.offset      = (size_t)msg - (size_t)cmi_pami_memregion[0].baseVA;
      rzv.dst_context = dst_context;
      memcpy(&rzv.mregion, &cmi_pami_memregion[0].mregion, sizeof(pami_memregion_t));

      pami_send_immediate_t parameters;
      parameters.dispatch        = CMI_PAMI_RZV_DISPATCH;
      parameters.header.iov_base = &rzv;
      parameters.header.iov_len  = sizeof(rzv);
      parameters.data.iov_base   = NULL;
      parameters.data.iov_len    = 0;
      parameters.dest = target;
      
      if(to_lock)
	PAMIX_CONTEXT_LOCK(context);
      
      PAMI_Send_immediate (context, &parameters);
      
      if(to_lock)
	PAMIX_CONTEXT_UNLOCK(context);
    }
}

void CmiSyncSendFn(int destPE, int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiAssert(copymsg != NULL);
    CmiMemcpy(copymsg,msg,size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncSendFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeSendFn(destPE,size,copymsg);
}

void CmiFreeSendFn(int destPE, int size, char *msg) {    
    CQdCreate(CpvAccess(cQdState), 1);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeSendFn on comm thd on node %d\n", CmiMyNode());

    CMI_SET_BROADCAST_ROOT(msg,0);
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)msg)->size = size;
    CMI_SET_CHECKSUM(msg, size);

    CmiGeneralFreeSend(destPE,size,msg);
}

/* same as CmiSyncSendFn, but don't set broadcast root in msg header */
void CmiSyncSendFn1(int destPE, int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiMemcpy(copymsg, msg, size);

    CMI_MAGIC(copymsg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)copymsg)->size = size;
    CMI_SET_CHECKSUM(copymsg, size);    
    CmiGeneralFreeSend(destPE,size,copymsg);
}

/* send msg to its spanning children in broadcast. G. Zheng */
void SendSpanningChildren(int size, char *msg, int from_rdone) {
    int startnode = CMI_BROADCAST_ROOT(msg)-1;
    int myrank = CMI_DEST_RANK(msg);
    int i;

    //printf ("%d [%d]: In Send Spanning Tree\n",  CmiMyPe(), CmiMyNode());

    CmiAssert(startnode>=0 && startnode<_Cmi_numnodes);
    int dist = CmiMyNode() - startnode;
    if (dist < 0) dist+=_Cmi_numnodes;
    for (i=1; i <= BROADCAST_SPANNING_FACTOR; i++) {
        int p = BROADCAST_SPANNING_FACTOR*dist + i;
        if (p > _Cmi_numnodes - 1) break;
        p += startnode;
        p = p%_Cmi_numnodes;
        CmiAssert(p>=0 && p<_Cmi_numnodes && p!= CmiMyNode());

	char *copymsg = (char *)CmiAlloc(size);
	CmiMemcpy(copymsg, msg, size);

	CMI_MAGIC(copymsg) = CHARM_MAGIC_NUMBER;
	((CmiMsgHeaderBasic *)copymsg)->size = size;
	CMI_SET_CHECKSUM(copymsg, size);
	
	CmiGeneralFreeSendN(p,0,size,copymsg, !from_rdone);	
    }    

#if CMK_SMP    
    //Send data within the nodes
    for (i =0; i < _Cmi_mynodesize; ++i) {
      if (i != myrank) {
	char *copymsg = (char *)CmiAlloc(size);
	CmiMemcpy(copymsg, msg, size);			
	CmiPushPE (i, copymsg);
      }
    }
#endif
}

void CmiSyncBroadcastFn(int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiMemcpy(copymsg,msg,size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncBroadcastFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeBroadcastFn(size,copymsg);
}

void CmiFreeBroadcastFn(int size, char *msg) {

    //  printf("%d: Calling Broadcast %d\n", CmiMyPe(), size);

    CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE    
    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeBroadcastFn on comm thd on node %d\n", CmiMyNode());

    //printf ("%d: Starting Spanning Tree Broadcast of size %d bytes\n", CmiMyPe(), size);

    CMI_SET_BROADCAST_ROOT(msg, CmiMyNode()+1);
    CMI_DEST_RANK(msg) = CmiMyRank();
    SendSpanningChildren(size, msg, 0);
    CmiFree(msg);
#else
    int i;

    for ( i=cs->pe+1; i<_Cmi_numpes; i++ )
        CmiSyncSendFn(i,size,msg);

    for ( i=0; i<cs->pe; i++ )
        CmiSyncSendFn(i,size,msg);

    CmiFree(msg);
#endif
}

void CmiSyncBroadcastAllFn(int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiMemcpy(copymsg,msg,size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncBroadcastAllFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeBroadcastAllFn(size,copymsg);
}

void CmiFreeBroadcastAllFn(int size, char *msg) {

    //printf("%d: Calling All Broadcast %d\n", CmiMyPe(), size);

    CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE
    //printf ("%d: Starting Spanning Tree Broadcast of size %d bytes\n", CmiMyPe(), size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeBroadcastAllFn on comm thd on node %d\n", CmiMyNode());

    CmiSyncSendFn(cs->pe,size,msg);
    
    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);

    CMI_SET_BROADCAST_ROOT(msg, CmiMyNode()+1);
    CMI_DEST_RANK(msg) = CmiMyRank();
    SendSpanningChildren(size, msg, 0);
    CmiFree(msg);
#else
    int i ;

    for ( i=0; i<_Cmi_numpes; i++ ) {
        CmiSyncSendFn(i,size,msg);      
    }
    //SendMsgsUntil (0);

    CmiFree(msg);
#endif
}

#if !CMK_ENABLE_ASYNC_PROGRESS  
//threads have to progress contexts themselves   
void AdvanceCommunications() {
    pami_context_t my_context = MY_CONTEXT();

#if CMK_SMP
    //CmiAssert (my_context != NULL);
    if (PAMIX_CONTEXT_TRYLOCK(my_context))
    {
      PAMI_Context_advance(my_context, 1);
      PAMIX_CONTEXT_UNLOCK(my_context);
    }
#else
    PAMI_Context_advance(my_context, 1);
#endif
}
#endif


void CmiNotifyIdle() {
  AdvanceCommunications();
#if CMK_SMP && CMK_PAMI_MULTI_CONTEXT
#if !CMK_ENABLE_ASYNC_PROGRESS && CMK_USE_L2ATOMICS
  //Wait on the atomic queue to get a message with very low core
  //overheads. One thread calls advance more frequently
  if ((CmiMyRank()% THREADS_PER_CONTEXT) == 0)
    //spin wait for 2-4us when idle
    //process node queue messages every 10us
    //Idle cores will only use one LMQ slot and an int sum
    L2AtomicQueue2QSpinWait(&procState[CmiMyRank()].atomic_queue, 
			    &node_recv_atomic_q,
			    10);
  else
#endif
#if CMK_USE_L2ATOMICS
    //spin wait for 50-100us when idle waiting for a message
    L2AtomicQueue2QSpinWait(&procState[CmiMyRank()].atomic_queue, 
			    &node_recv_atomic_q,
			    1000);
#endif
#endif
}


/*==========================================================*/
/*==========================================================*/
/*==========================================================*/

/************ Recommended routines ***********************/
/************ You dont have to implement these but they are supported
 in the converse syntax and some rare programs may crash. But most
 programs dont need them. *************/

CmiCommHandle CmiAsyncSendFn(int dest, int size, char *msg) {
    CmiAbort("CmiAsyncSendFn not implemented.");
    return (CmiCommHandle) 0;
}

CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg) {
    CmiAbort("CmiAsyncBroadcastFn not implemented.");
    return (CmiCommHandle) 0;
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg) {
    CmiAbort("CmiAsyncBroadcastAllFn not implemented.");
    return (CmiCommHandle) 0;
}

int           CmiAsyncMsgSent(CmiCommHandle handle) {
    CmiAbort("CmiAsyncMsgSent not implemented.");
    return 0;
}
void          CmiReleaseCommHandle(CmiCommHandle handle) {
    CmiAbort("CmiReleaseCommHandle not implemented.");
}


/*==========================================================*/
/*==========================================================*/
/*==========================================================*/

/* Optional routines which could use common code which is shared with
   other machine layer implementations. */

/* MULTICAST/VECTOR SENDING FUNCTIONS

 * In relations to some flags, some other delivery functions may be needed.
 */

#if ! CMK_MULTICAST_LIST_USE_COMMON_CODE

void CmiSyncListSendFn(int npes, int *pes, int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiMemcpy(copymsg,msg,size);
    CmiFreeListSendFn(npes, pes, size, msg);
}

typedef struct ListMulticastVec_t {
  int   *pes;
  int    npes;
  char  *msg;
  int    size;
} ListMulticastVec;

void machineFreeListSendFn(pami_context_t    context, 
			   int               npes, 
			   int             * pes, 
			   int               size, 
			   char            * msg);

pami_result_t machineFreeList_handoff(pami_context_t context, void *cookie)
{
  ListMulticastVec *lvec = (ListMulticastVec *) cookie;
  machineFreeListSendFn(context, lvec->npes, lvec->pes, lvec->size, lvec->msg);
  CmiFree(cookie);
}

void CmiFreeListSendFn(int npes, int *pes, int size, char *msg) {
    //printf("%d: In Free List Send Fn imm %d\n", CmiMyPe(), CmiIsImmediate(msg));

    CMI_SET_BROADCAST_ROOT(msg,0);
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    CmiMsgHeaderBasic *hdr = (CmiMsgHeaderBasic *)msg;
    hdr->size = size;
    CMI_SET_CHECKSUM(msg, size);

    //Fast path
    if (npes == 1) {
      CmiGeneralFreeSendN(CmiNodeOf(pes[0]), CmiRankOf(pes[0]), size, msg, 1);
      return;
    }

    pami_context_t my_context = MY_CONTEXT();
#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
    ListMulticastVec *lvec = (ListMulticastVec *) 
      CmiAlloc(sizeof(ListMulticastVec) + sizeof(int)*npes);
    lvec->pes = (int*)((char*)lvec + sizeof(ListMulticastVec));
    int i = 0;
    for (i=0; i<npes; i++) 
      lvec->pes[i] = pes[i];
    lvec->npes = npes;
    lvec->msg  = msg;
    lvec->size = size;
    PAMI_Context_post(my_context, (pami_work_t*)hdr->work, 
		      machineFreeList_handoff, lvec);
#else
    machineFreeListSendFn(my_context, npes, pes, size, msg);
#endif
}

void machineFreeListSendFn(pami_context_t my_context, int npes, int *pes, int size, char *msg) {
    int i;
    char *copymsg;
#if CMK_SMP
    for (i=0; i<npes; i++) {
      if (CmiNodeOf(pes[i]) == CmiMyNode()) {
	//CmiSyncSend(pes[i], size, msg);
	copymsg = (char *)CmiAlloc(size);
	CmiAssert(copymsg != NULL);
	CmiMemcpy(copymsg,msg,size);	  
	CmiSendPeer(CmiRankOf(pes[i]), size, copymsg);
      }
    }
#else
    for (i=0; i<npes; i++) {
      if (CmiNodeOf(pes[i]) == CmiMyNode()) {
	CmiSyncSend(pes[i], size, msg);
      }
    }
#endif

    PAMIX_CONTEXT_LOCK(my_context);
    
    for (i=0;i<npes;i++) {
        if (CmiNodeOf(pes[i]) == CmiMyNode());
        else if (i < npes - 1) {
#if !CMK_SMP
	    CmiReference(msg);
	    copymsg = msg;
#else
	    copymsg = (char *)CmiAlloc(size);
	    CmiAssert(copymsg != NULL);
	    CmiMemcpy(copymsg,msg,size);
#endif
	    CmiGeneralFreeSendN(CmiNodeOf(pes[i]), CmiRankOf(pes[i]), size, copymsg, 0);
	    //machine_send(my_context, CmiNodeOf(pes[i]), CmiRankOf(pes[i]), size, copymsg, 0);
        }
    }

    if (npes  && CmiNodeOf(pes[npes-1]) != CmiMyNode())
      CmiGeneralFreeSendN(CmiNodeOf(pes[npes-1]), CmiRankOf(pes[npes-1]), size, msg, 0);
      //machine_send(my_context, CmiNodeOf(pes[npes-1]), CmiRankOf(pes[npes-1]), size, msg, 0);      
    else
      CmiFree(msg);    

    PAMIX_CONTEXT_UNLOCK(my_context);
}

CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int size, char *msg) {
    CmiAbort("CmiAsyncListSendFn not implemented.");
    return (CmiCommHandle) 0;
}

#if CMK_NODE_QUEUE_AVAILABLE

typedef struct ListNodeMulticastVec_t {
  int   *nodes;
  int    n_nodes;
  char  *msg;
  int    size;
} ListNodeMulticastVec;

void machineFreeNodeListSendFn(pami_context_t    context, 
			       int               n_nodes, 
			       int             * nodes, 
			       int               size, 
			       char            * msg);

pami_result_t machineFreeNodeList_handoff(pami_context_t context, void *cookie)
{
  ListNodeMulticastVec *lvec = (ListNodeMulticastVec *) cookie;
  machineFreeNodeListSendFn(context, lvec->n_nodes, 
			    lvec->nodes, lvec->size, lvec->msg);
  CmiFree(cookie);
}

void CmiFreeNodeListSendFn(int n_nodes, int *nodes, int size, char *msg) {

    //printf("%d In cmifreenodelistsendfn %d %d\n", CmiMyPe(), n_nodes, size);
    CMI_SET_BROADCAST_ROOT(msg,0);
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    CmiMsgHeaderBasic *hdr = (CmiMsgHeaderBasic *)msg;
    hdr->size = size;
    CMI_SET_CHECKSUM(msg, size);

    //Fast path
    if (n_nodes == 1) {
      CmiGeneralFreeSendN(nodes[0], SMP_NODEMESSAGE, size, msg, 1);
      return;
    }

    pami_context_t my_context = MY_CONTEXT();
#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
    ListNodeMulticastVec *lvec = (ListNodeMulticastVec *) 
      CmiAlloc(sizeof(ListNodeMulticastVec));
      
    lvec->nodes   = nodes;
    lvec->n_nodes = n_nodes;
    lvec->msg     = msg;
    lvec->size    = size;
    PAMI_Context_post(my_context, (pami_work_t*)hdr->work, 
		      machineFreeNodeList_handoff, lvec);
#else
    machineFreeNodeListSendFn(my_context, n_nodes, nodes, size, msg);
#endif
}

void CmiSendNodeSelf(char *msg);
void machineFreeNodeListSendFn(pami_context_t       my_context, 
			       int                  n_nodes, 
			       int                * nodes, 
			       int                  size, 
			       char               * msg) 
{
    int i;
    char *copymsg;

    for (i=0; i<n_nodes; i++) {
      if (nodes[i] == CmiMyNode()) {
	copymsg = (char *)CmiAlloc(size);
	CmiAssert(copymsg != NULL);
	CmiMemcpy(copymsg,msg,size);	  
	CmiSendNodeSelf(copymsg);
      }
    }

    PAMIX_CONTEXT_LOCK(my_context);
    
    for (i=0;i<n_nodes;i++) {
        if (nodes[i] == CmiMyNode());
        else if (i < n_nodes - 1) {
#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
	  CmiReference(msg);
	  copymsg = msg;
	  machine_send(my_context, nodes[i], SMP_NODEMESSAGE, size, copymsg, 0);
#else
	  copymsg = (char *)CmiAlloc(size);
	  CmiAssert(copymsg != NULL);
	  CmiMemcpy(copymsg,msg,size);
	  CmiGeneralFreeSendN(nodes[i], SMP_NODEMESSAGE, size, copymsg, 0);
#endif
        }
    }

    if (n_nodes  && nodes[n_nodes-1] != CmiMyNode())
      machine_send(my_context, nodes[n_nodes-1], SMP_NODEMESSAGE, size, msg, 0);
    else
      CmiFree(msg);    
    
    PAMIX_CONTEXT_UNLOCK(my_context);
}
#endif

#endif

/** NODE SENDING FUNCTIONS

 * If there is a node queue, and we consider also nodes as entity (tipically in
 * SMP versions), these functions are needed.
 */

#if CMK_NODE_QUEUE_AVAILABLE

void          CmiSyncNodeSendFn(int, int, char *);
CmiCommHandle CmiAsyncNodeSendFn(int, int, char *);
void          CmiFreeNodeSendFn(int, int, char *);

void          CmiSyncNodeBroadcastFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastFn(int, char *);
void          CmiFreeNodeBroadcastFn(int, char *);

void          CmiSyncNodeBroadcastAllFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastAllFn(int, char *);
void          CmiFreeNodeBroadcastAllFn(int, char *);

#endif


#if CMK_SHARED_VARS_POSIX_THREADS_SMP

int CmiMyPe();
int CmiMyRank();
int CmiNodeFirst(int node);
int CmiNodeSize(int node);
int CmiNodeOf(int pe);
int CmiRankOf(int pe);

int CmiMyPe(void) {
    return CmiGetState()->pe;
}

int CmiMyRank(void) {
    return CmiGetState()->rank;
}

int CmiNodeFirst(int node) {
    return node*_Cmi_mynodesize;
}
int CmiNodeSize(int node)  {
    return _Cmi_mynodesize;
}

int CmiNodeOf(int pe)      {
    return (pe/_Cmi_mynodesize);
}
int CmiRankOf(int pe)      {
    return pe%_Cmi_mynodesize;
}


/* optional, these functions are implemented in "machine-smp.c", so including
   this file avoid the necessity to reimplement them.
 */
void CmiNodeBarrier(void);
void CmiNodeAllBarrier(void);
CmiNodeLock CmiCreateLock();
void CmiDestroyLock(CmiNodeLock lock);

#endif

/** IMMEDIATE MESSAGES

 * If immediate messages are supported, the following function is needed. There
 * is an exeption if the machine progress is also defined (see later for this).

 * Moreover, the file "immediate.c" should be included, otherwise all its
 * functions and variables have to be redefined.
*/

#if CMK_CCS_AVAILABLE

#include "immediate.c"

#if ! CMK_MACHINE_PROGRESS_DEFINED /* Hack for some machines */
void CmiProbeImmediateMsg();
#endif

#endif


/* Dummy implementation */
extern int CmiBarrier() {
  CmiNodeBarrier();
  if (CmiMyRank() == 0)
    CmiNetworkBarrier(1);
  CmiNodeBarrier();
  return 0;
}


static pami_result_t machine_network_barrier(pami_context_t   my_context, 
					     int              to_lock) 
{
    pami_result_t result = PAMI_SUCCESS;    
    if (to_lock)
      PAMIX_CONTEXT_LOCK(my_context);    
    result = PAMI_Collective(my_context, &pami_barrier);       
    if (to_lock)
      PAMIX_CONTEXT_UNLOCK(my_context);

    if (result != PAMI_SUCCESS)
      fprintf (stderr, "Error. Unable to issue  collective. result = %d\n", result);

    return result;
}

pami_result_t network_barrier_handoff(pami_context_t context, void *msg)
{
  return machine_network_barrier(context, 0);
}

static void CmiNetworkBarrier(int async) {
    pami_context_t my_context = cmi_pami_contexts[0];
    pami_barrier_flag = 1;
#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
    if (async) {
      pami_work_t work;
      PAMI_Context_post(my_context, &work, network_barrier_handoff, NULL);
      while (pami_barrier_flag);
      //fprintf (stderr, "After Network Barrier\n");
    }
    else 
#endif
    {
      machine_network_barrier(my_context, 1);    
      PAMIX_CONTEXT_LOCK(my_context);
      while (pami_barrier_flag)
	PAMI_Context_advance (my_context, 100);
      PAMIX_CONTEXT_UNLOCK(my_context);
    }
}

#if CMK_NODE_QUEUE_AVAILABLE
void CmiSendNodeSelf(char *msg) {
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
      //CmiLock(CsvAccess(NodeState).immRecvLock);
      CmiHandleImmediateMessage(msg);
      //CmiUnlock(CsvAccess(NodeState).immRecvLock);
      return;
    }
#endif    
#if CMK_SMP && CMK_USE_L2ATOMICS
    L2AtomicEnqueue(&node_recv_atomic_q, msg);    
#else
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv, msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
#endif
}

CmiCommHandle CmiAsyncNodeSendFn(int dstNode, int size, char *msg) {
    CmiAbort ("Async Node Send not supported\n");
}

void CmiFreeNodeSendFn(int node, int size, char *msg) {

    CMI_SET_BROADCAST_ROOT(msg,0);
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)msg)->size = size;
    CMI_SET_CHECKSUM(msg, size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeNodeSendFn on comm thd on node %d\n", CmiMyNode());
    
    CQdCreate(CpvAccessOther(cQdState, CmiMyRank()), 1);

    if (node == _Cmi_mynode) {
        CmiSendNodeSelf(msg);
    } else {
      CmiGeneralFreeSendN(node, SMP_NODEMESSAGE, size, msg, 1);
    }
}

void CmiSyncNodeSendFn(int p, int s, char *m) {
    char *dupmsg;
    dupmsg = (char *)CmiAlloc(s);
    CmiMemcpy(dupmsg,m,s);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncNodeSendFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeNodeSendFn(p, s, dupmsg);
}

CmiCommHandle CmiAsyncNodeBroadcastFn(int s, char *m) {
    return NULL;
}

void SendSpanningChildrenNode(int size, char *msg, int from_rdone) {
    int startnode = -CMI_BROADCAST_ROOT(msg)-1;
    //printf("on node %d rank %d, send node spanning children with root %d\n", CmiMyNode(), CmiMyRank(), startnode);
    assert(startnode>=0 && startnode<CmiNumNodes());

    int dist = CmiMyNode()-startnode;
    if (dist<0) dist += CmiNumNodes();
    int i;
    for (i=1; i <= BROADCAST_SPANNING_FACTOR; i++) {
        int nid = BROADCAST_SPANNING_FACTOR*dist + i;
        if (nid > CmiNumNodes() - 1) break;
        nid += startnode;
        nid = nid%CmiNumNodes();
        assert(nid>=0 && nid<CmiNumNodes() && nid!=CmiMyNode());
        char *dupmsg = (char *)CmiAlloc(size);
        CmiMemcpy(dupmsg,msg,size);
        //printf("In SendSpanningChildrenNode, sending bcast msg (root %d) from node %d to node %d\n", startnode, CmiMyNode(), nid);
        CmiGeneralFreeSendN(nid, SMP_NODEMESSAGE, size, dupmsg,!from_rdone);
    }
}

/* need */
void CmiFreeNodeBroadcastFn(int s, char *m) {
  //printf("%d: In FreeNodeBroadcastAllFn\n", CmiMyPe());

#if CMK_BROADCAST_SPANNING_TREE
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeNodeBcastFn on comm thd on node %d\n", CmiMyNode());
    
    CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);

    int mynode = CmiMyNode();
    CMI_SET_BROADCAST_ROOT(m, -mynode-1);
    CMI_MAGIC(m) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)m)->size = s;
    CMI_SET_CHECKSUM(m, s);
    //printf("In CmiFreeNodeBroadcastFn, sending bcast msg from root node %d\n", CMI_BROADCAST_ROOT(m));

    SendSpanningChildrenNode(s, m, 0);
#else
    int i;
    for (i=0; i<CmiNumNodes(); i++) {
        if (i==CmiMyNode()) continue;
        char *dupmsg = (char *)CmiAlloc(s);
        CmiMemcpy(dupmsg,m,s);
        CmiFreeNodeSendFn(i, s, dupmsg);
    }
#endif
    CmiFree(m);    
}

void CmiSyncNodeBroadcastFn(int s, char *m) {
    char *dupmsg;
    dupmsg = (char *)CmiAlloc(s);
    CmiMemcpy(dupmsg,m,s);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncNodeBcastFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeNodeBroadcastFn(s, dupmsg);
}

/* need */
void CmiFreeNodeBroadcastAllFn(int s, char *m) {
  
    char *dupmsg = (char *)CmiAlloc(s);
    CmiMemcpy(dupmsg,m,s);
    CMI_MAGIC(dupmsg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)dupmsg)->size = s;
    CMI_SET_CHECKSUM(dupmsg, s);

    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeNodeBcastAllFn on comm thd on node %d\n", CmiMyNode());
    
    CQdCreate(CpvAccess(cQdState), 1);
    CmiSendNodeSelf(dupmsg);

    CmiFreeNodeBroadcastFn(s, m);
}

void CmiSyncNodeBroadcastAllFn(int s, char *m) {
    char *dupmsg;
    dupmsg = (char *)CmiAlloc(s);
    CmiMemcpy(dupmsg,m,s);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncNodeBcastAllFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeNodeBroadcastAllFn(s, dupmsg);
}


CmiCommHandle CmiAsyncNodeBroadcastAllFn(int s, char *m) {
    return NULL;
}
#endif //end of CMK_NODE_QUEUE_AVAILABLE


static void  sendAck (pami_context_t      context,
		      CmiPAMIRzvRecv_t      *recv) 
{
  pami_send_immediate_t parameters;
  parameters.dispatch        = CMI_PAMI_ACK_DISPATCH;
  parameters.header.iov_base = &recv->src_buffer; 
  parameters.header.iov_len  = sizeof(void *);    
  parameters.data.iov_base   = NULL;
  parameters.data.iov_len    = 0;
  parameters.dest            = recv->src_ep;
  
  //Called from advance and hence we dont need a mutex
  PAMI_Send_immediate (context, &parameters);
}


void rzv_recv_done   (pami_context_t     ctxt, 
		      void             * clientdata, 
		      pami_result_t      result) 
{
  CmiPAMIRzvRecv_t recv = *(CmiPAMIRzvRecv_t *)clientdata;
  recv_done(ctxt, recv.msg, PAMI_SUCCESS);
  sendAck(ctxt, &recv);
}

void rzv_pkt_dispatch (pami_context_t       context,   
		       void               * clientdata,
		       const void         * header_addr,
		       size_t               header_size,
		       const void         * pipe_addr,  
		       size_t               pipe_size,  
		       pami_endpoint_t      origin,
		       pami_recv_t         * recv) 
{
  INCR_ORECVS();    
  
  CmiPAMIRzv_t  *rzv_hdr = (CmiPAMIRzv_t *) header_addr;
  CmiAssert (header_size == sizeof(CmiPAMIRzv_t));  
  int alloc_size = rzv_hdr->bytes;
  char * buffer  = (char *)CmiAlloc(alloc_size + sizeof(CmiPAMIRzvRecv_t));
  //char *buffer=(char*)CmiAlloc(alloc_size+sizeof(CmiPAMIRzvRecv_t)+sizeof(int))
  //*(int *)(buffer+alloc_size) = *(int *)header_addr;  
  CmiAssert (recv == NULL);

  CmiPAMIRzvRecv_t *rzv_recv = (CmiPAMIRzvRecv_t *)(buffer+alloc_size);
  rzv_recv->msg        = buffer;
  rzv_recv->src_ep     = origin;
  rzv_recv->src_buffer = rzv_hdr->buffer;
  rzv_recv->size       = rzv_hdr->bytes;
  
  //CmiAssert (pipe_addr != NULL);
  //CmiAssert (pipe_size == sizeof(pami_memregion_t));
  //pami_memregion_t *mregion = (pami_memregion_t *) pipe_addr;
  memcpy(&rzv_recv->rmregion, &rzv_hdr->mregion, sizeof(pami_memregion_t));

  //Rzv inj fifos are on the 17th core shared by all contexts
  pami_rget_simple_t  rget;
  rget.rma.dest    = origin;
  rget.rma.bytes   = rzv_hdr->bytes;
  rget.rma.cookie  = rzv_recv;
  rget.rma.done_fn = rzv_recv_done;
  rget.rma.hints.buffer_registered = PAMI_HINT_ENABLE;
  rget.rma.hints.use_rdma = PAMI_HINT_ENABLE;
  //rget.rma.hints.use_shmem = PAMI_HINT_DISABLE;
  rget.rdma.local.mr      = &cmi_pami_memregion[rzv_hdr->dst_context].mregion;  
  rget.rdma.local.offset  = (size_t)buffer - 
    (size_t)cmi_pami_memregion[rzv_hdr->dst_context].baseVA;
  rget.rdma.remote.mr     = &rzv_recv->rmregion;
  rget.rdma.remote.offset = rzv_hdr->offset;

  //printf ("starting rget\n");
  pami_result_t rc;
  rc = PAMI_Rget (context, &rget);  
  //CmiAssert(rc == PAMI_SUCCESS);
}

void ack_pkt_dispatch (pami_context_t       context,   
		       void               * clientdata,
		       const void         * header_addr,
		       size_t               header_size,
		       const void         * pipe_addr,  
		       size_t               pipe_size,  
		       pami_endpoint_t      origin,
		       pami_recv_t         * recv) 
{
  char **buf = (char **)header_addr;
  CmiFree (*buf);
}

#include "cmimemcpy_qpx.h"
