
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
#include "pami.h"
#include "pami_sys.h"

char *ALIGN_32(char *p) {
  return((char *)((((unsigned long)p)+0x1f) & (~0x1FUL)));
}

CpvDeclare(PCQueue, broadcast_q);                 //queue to send broadcast messages
#if CMK_NODE_QUEUE_AVAILABLE
CsvDeclare(PCQueue, node_bcastq);
CsvDeclare(CmiNodeLock, node_bcastLock);
#endif

//#define ENABLE_BROADCAST_THROTTLE 1

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

#define BROADCAST_SPANNING_FACTOR     2

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

#if !CMK_OPTIMIZE
static int checksum_flag = 0;
extern unsigned char computeCheckSum(unsigned char *data, int len);

#define CMI_SET_CHECKSUM(msg, len)      \
        if (checksum_flag)  {   \
          ((CmiMsgHeaderBasic *)msg)->cksum = 0;        \
          ((CmiMsgHeaderBasic *)msg)->cksum = computeCheckSum((unsigned char*)msg, len);        \
        }

#define CMI_CHECK_CHECKSUM(msg, len)    \
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
    /* PCQueue      sendMsgBuf; */      /* per processor message sending queue */
    CmiNodeLock  recvLock;              /* for cs->recv */
    CmiNodeLock bcastLock;
} ProcState;

static ProcState  *procState;

#if CMK_SMP && !CMK_MULTICORE
//static volatile int commThdExit = 0;
//static CmiNodeLock commThdExitLock = 0;
#endif

void ConverseRunPE(int everReturn);
static void CommunicationServer(int sleepTime);
static void CommunicationServerThread(int sleepTime);

static void CmiNetworkBarrier();

//So far we dont define any comm threads
int Cmi_commthread = 0;

#include "machine-smp.c"
CsvDeclare(CmiNodeState, NodeState);
#include "immediate.c"

void AdvanceCommunications();


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

/*Add a message to this processor's receive queue, pe is a rank */
void CmiPushPE(int pe,void *msg) {
    CmiState cs = CmiGetStateN(pe);
    MACHSTATE2(3,"Pushing message into rank %d's queue %p{",pe, cs->recv);
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        /**(CmiUInt2 *)msg = pe;*/
        //received_immediate = 1;
        //printf("PushPE: N[%d]P[%d]R[%d] received an imm msg with hdl: %p\n", CmiMyNode(), CmiMyPe(), CmiMyRank(), CmiGetHandler(msg));
        //CMI_DEST_RANK(msg) = pe;
        CmiPushImmediateMsg(msg);
        return;
    }
#endif
#if CMK_SMP
    //CmiLock(procState[pe].recvLock);
#endif

    PCQueuePush(cs->recv,(char *)msg);
    //printf("%d: PCQueue length = %d, msg = %x\n", CmiMyPe(), PCQueueLength(cs->recv), msg);

#if CMK_SMP
    //CmiUnlock(procState[pe].recvLock);
#endif
    CmiIdleLock_addMessage(&cs->idle);
    MACHSTATE1(3,"} Pushing message into rank %d's queue done",pe);
}

#if CMK_NODE_QUEUE_AVAILABLE
/*Add a message to this processor's receive queue */
static void CmiPushNode(void *msg) {
    MACHSTATE(3,"Pushing message into NodeRecv queue");
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        //printf("PushNode: N[%d]P[%d]R[%d] received an imm msg with hdl: %p\n", CmiMyNode(), CmiMyPe(), CmiMyRank(), CmiGetHandler(msg));
        //CMI_DEST_RANK(msg) = 0;
        CmiPushImmediateMsg(msg);
        return;
    }
#endif
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv,msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    {
        CmiState cs=CmiGetStateN(0);
        CmiIdleLock_addMessage(&cs->idle);
    }
}
#endif /* CMK_NODE_QUEUE_AVAILABLE */

#define MAX_NUM_CONTEXTS  16

#if CMK_SMP 
#define CMK_PAMI_MULTI_CONTEXT  0
#else
#define CMK_PAMI_MULTI_CONTEXT  0
#endif

#if CMK_PAMI_MULTI_CONTEXT
volatile int msgQueueLen [MAX_NUM_CONTEXTS];
volatile int outstanding_recvs [MAX_NUM_CONTEXTS];
#define  MY_CONTEXT_ID() (CmiMyRank() >> 2)
#define  MY_CONTEXT()    (cmi_pami_contexts[CmiMyRank() >> 2])

#define  INCR_MSGQLEN()  (msgQueueLen[CmiMyRank() >> 2] ++)
#define  DECR_MSGQLEN()  (msgQueueLen[CmiMyRank() >> 2] --)
#define  MSGQLEN()       (msgQueueLen[CmiMyRank() >> 2])
#define  INCR_ORECVS()   (outstanding_recvs[CmiMyRank() >> 2] ++)
#define  DECR_ORECVS()   (outstanding_recvs[CmiMyRank() >> 2] --)
#define  ORECVS()        (outstanding_recvs[CmiMyRank() >> 2])
#else
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

static char     **Cmi_argv;
static char     **Cmi_argvcopy;
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */

extern void ConverseCommonInit(char **argv);
extern void ConverseCommonExit(void);
extern void CthInit(char **argv);

static void SendMsgsUntil(int);


void SendSpanningChildren(int size, char *msg);
#if CMK_NODE_QUEUE_AVAILABLE
void SendSpanningChildrenNode(int size, char *msg);
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

    //fprintf (stderr, "%d Recv message done \n", CmiMyPe());
    /* then we do what PumpMsgs used to do:
     * push msg to recv queue */
    int count=0;
    CMI_CHECK_CHECKSUM(msg, sndlen);
    if (CMI_MAGIC(msg) != CHARM_MAGIC_NUMBER) { /* received a non-charm msg */
        CmiAbort("Charm++ Warning: Non Charm++ Message Received. \n");
        return;
    }

#if CMK_BROADCAST_SPANNING_TREE 
    if (CMI_IS_BCAST_ON_CORES(msg) ) {
      int pe = CmiMyRank(); //CMI_DEST_RANK(msg);
        //printf ("%d: Receiving bcast message from %d with %d bytes for %d\n", CmiMyPe(), CMI_BROADCAST_ROOT(msg), sndlen, pe);
        char *copymsg;
        copymsg = (char *)CmiAlloc(sndlen);
        CmiMemcpy(copymsg,msg,sndlen);

        //received_broadcast = 1;
#if CMK_SMP
        CmiLock(procState[pe].bcastLock);
        PCQueuePush(CpvAccessOther(broadcast_q, pe), copymsg);
        CmiUnlock(procState[pe].bcastLock);
#else
        PCQueuePush(CpvAccess(broadcast_q), copymsg);
#endif
    }
#endif

#if CMK_NODE_QUEUE_AVAILABLE
#if CMK_BROADCAST_SPANNING_TREE
    if (CMI_IS_BCAST_ON_NODES(msg)) {
        //printf ("%d: Receiving node bcast message from %d with %d bytes for %d\n", CmiMyPe(), CMI_BROADCAST_ROOT(msg), sndlen, CMI_DEST_RANK(msg));
        char *copymsg = (char *)CmiAlloc(sndlen);
        CmiMemcpy(copymsg,msg,sndlen);
        //CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
        CmiLock(CsvAccess(node_bcastLock));
        PCQueuePush(CsvAccess(node_bcastq), copymsg);
        CmiUnlock(CsvAccess(node_bcastLock));
        //CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    }
#endif
    if (CMI_DEST_RANK(msg) == SMP_NODEMESSAGE)
      CmiPushNode(msg);
    else
#endif
      CmiPushPE(CMI_DEST_RANK(msg), (void *)msg);

    DECR_ORECVS();
}

static void pkt_dispatch (pami_context_t       context,      /**< IN: PAMI context */
			  void               * clientdata,   /**< IN: dispatch cookie */
			  const void         * header_addr,  /**< IN: header address */
			  size_t               header_size,  /**< IN: header size */
			  const void         * pipe_addr,    /**< IN: address of PAMI pipe buffer */
			  size_t               pipe_size,    /**< IN: size of PAMI pipe buffer */
			  pami_endpoint_t      origin,
			  pami_recv_t         * recv)        /**< OUT: receive message structure */
{
    //fprintf (stderr, "Received Message of size %d %p\n", pipe_size, recv);
    INCR_ORECVS();    
    int alloc_size = pipe_size;
    char * buffer  = (char *)CmiAlloc(alloc_size);

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


#if CMK_NODE_QUEUE_AVAILABLE
void sendBroadcastMessagesNode() {
    if (PCQueueLength(CsvAccess(node_bcastq))==0) return;
    //node broadcast message could be always handled by any cores (including
    //comm thd) on this node
    //CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    CmiLock(CsvAccess(node_bcastLock));
    char *msg = PCQueuePop(CsvAccess(node_bcastq));
    CmiUnlock(CsvAccess(node_bcastLock));
    //CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    while (msg) {
        //printf("sendBroadcastMessagesNode: node %d rank %d with msg root %d\n", CmiMyNode(), CmiMyRank(), CMI_BROADCAST_ROOT(msg));
        SendSpanningChildrenNode(((CmiMsgHeaderBasic *) msg)->size, msg);
        CmiFree(msg);
        //CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
        CmiLock(CsvAccess(node_bcastLock));
        msg = PCQueuePop(CsvAccess(node_bcastq));
        CmiUnlock(CsvAccess(node_bcastLock));
        //CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    }
}
#endif

void sendBroadcastMessages() {
  PCQueue toPullQ;
  toPullQ = CpvAccess(broadcast_q);

  if (PCQueueLength(toPullQ)==0) return;
#if CMK_SMP
  CmiLock(procState[CmiMyRank()].bcastLock);
#endif

    char *msg = (char *) PCQueuePop(toPullQ);

#if CMK_SMP
    CmiUnlock(procState[CmiMyRank()].bcastLock);
#endif

    while (msg) {

#if CMK_BROADCAST_SPANNING_TREE
        SendSpanningChildren(((CmiMsgHeaderBasic *) msg)->size, msg);
#endif

        CmiFree (msg);

#if CMK_SMP
        CmiLock(procState[CmiMyRank()].bcastLock);
#endif

        msg = (char *) PCQueuePop(toPullQ);

#if CMK_SMP
        CmiUnlock(procState[CmiMyRank()].bcastLock);
#endif
    }
}


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

#define CMI_PAMI_DISPATCH   10

#include "malloc.h"

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
    if ((_Cmi_mynodesize % 4) == 0)
      _n = _Cmi_mynodesize / 4;  //have a context for each four threads
    else
      _n = 1 + (_Cmi_mynodesize / 4);  //have a context for each four threads
#endif

    cmi_pami_contexts = (pami_context_t *) malloc (sizeof(pami_context_t) * _n);
    PAMI_Context_createv (cmi_pami_client, NULL, 0, cmi_pami_contexts, _n);
    cmi_pami_numcontexts = _n;

    pami_configuration_t configuration;
    pami_result_t result;
    
    configuration.name = PAMI_CLIENT_TASK_ID;
    result = PAMI_Client_query(cmi_pami_client, &configuration, 1);
    _Cmi_mynode = configuration.value.intval;

    configuration.name = PAMI_CLIENT_NUM_TASKS;
    result = PAMI_Client_query(cmi_pami_client, &configuration, 1);
    _Cmi_numnodes = configuration.value.intval;

    pami_dispatch_hint_t options = (pami_dispatch_hint_t) {0};
    pami_dispatch_callback_function pfn;
    pfn.p2p = pkt_dispatch;
    for (i = 0; i < _n; ++i)
      PAMI_Dispatch_set (cmi_pami_contexts[i],
			 CMI_PAMI_DISPATCH,
			 pfn,
			 NULL,
			 options);
    
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
    pami_barrier.cb_done   = pami_barrier_done;
    pami_barrier.cookie    = (void*) & pami_barrier_flag;
    pami_barrier.algorithm = always_works_algo[0];

    /* Docs06:  Query the algorithm lists */
    if (result != PAMI_SUCCESS)
      {
	fprintf (stderr, "Error. Unable to get query algorithm. result = %d\n", result);
	return;
      }

    CmiNetworkBarrier();
    CmiNetworkBarrier();
    CmiNetworkBarrier();

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

#if CMK_NODE_QUEUE_AVAILABLE
    CsvInitialize(PCQueue, node_bcastq);
    CsvAccess(node_bcastq) = PCQueueCreate();
    CsvInitialize(CmiNodeLock, node_bcastLock);
    CsvAccess(node_bcastLock) = CmiCreateLock();
#endif

    int actualNodeSize = _Cmi_mynodesize;
#if !CMK_MULTICORE
    actualNodeSize++; //considering the extra comm thread
#endif

    procState = (ProcState *)CmiAlloc((actualNodeSize) * sizeof(ProcState));
    for (i=0; i<actualNodeSize; i++) {
        /*    procState[i].sendMsgBuf = PCQueueCreate();   */
        procState[i].recvLock = CmiCreateLock();
        procState[i].bcastLock = CmiCreateLock();
    }

#if CMK_SMP && !CMK_MULTICORE
    //commThdExitLock = CmiCreateLock();
#endif

    //printf ("Starting Threads\n");
    CmiStartThreads(argv);
    ConverseRunPE(initret);
}


int PerrorExit (char *err) {
  fprintf (stderr, "err\n\n");
    exit (-1);
    return -1;
}


void ConverseRunPE(int everReturn) {
    //printf ("ConverseRunPE on rank %d\n", CmiMyPe());

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

    CpvInitialize(PCQueue, broadcast_q);
    CpvAccess(broadcast_q) = PCQueueCreate();

    //printf ("Before Converse Common Init\n");
    ConverseCommonInit(CmiMyArgv);

    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);

    CmiBarrier();

    /* Converse initialization finishes, immediate messages can be processed.
       node barrier previously should take care of the node synchronization */
    _immediateReady = 1;

    if (!everReturn) {
      Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
      if (Cmi_usrsched==0) CsdScheduler(-1);
      ConverseExit();
    }
}

#if CMK_SMP
static int inexit = 0;

/* test if all processors recv queues are empty */
static int RecvQueueEmpty() {
    int i;
    for (i=0; i<_Cmi_mynodesize; i++) {
        CmiState cs=CmiGetStateN(i);
        if (!PCQueueEmpty(cs->recv)) return 0;
    }
    return 1;
}

#endif


void ConverseExit(void) {

    while (MSGQLEN() > 0 || ORECVS() > 0) {
      AdvanceCommunications();
    }
    
    CmiNodeBarrier();
    ConverseCommonExit();

    if (CmiMyPe() == 0) {
        printf("End of program\n");
    }

    CmiNodeBarrier();
//  CmiNodeAllBarrier ();

    int rank0 = 0;
    if (CmiMyRank() == 0) {
        rank0 = 1;
        //CmiFree(procState);
	PAMI_Context_destroyv(cmi_pami_contexts, cmi_pami_numcontexts);
	PAMI_Client_destroy(&cmi_pami_client);
    }

    CmiNodeBarrier();
    //  CmiNodeAllBarrier ();
    //fprintf(stderr, "Before Exit\n");
#if CMK_SMP
    if (rank0)
      exit(1);
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
    CmiState cs = CmiGetState();
    char *result = 0;
    CmiIdleLock_checkMessage(&cs->idle);
    if (!PCQueueEmpty(CsvAccess(NodeState).NodeRecv)) {
        MACHSTATE1(3,"CmiGetNonLocalNodeQ begin %d {", CmiMyPe());

        if (CmiTryLock(CsvAccess(NodeState).CmiNodeRecvLock) == 0) {
            //CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
            result = (char *) PCQueuePop(CsvAccess(NodeState).NodeRecv);
            CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
        }

        MACHSTATE1(3,"} CmiGetNonLocalNodeQ end %d ", CmiMyPe());
    }
    return result;
}
#endif


void *CmiGetNonLocal() {

    CmiState cs = CmiGetState();

    void *msg = NULL;
    CmiIdleLock_checkMessage(&cs->idle);
    /* although it seems that lock is not needed, I found it crashes very often
       on mpi-smp without lock */

    /*if(CmiMyRank()==0) printf("Got stuck here on proc[%d] node[%d]\n", CmiMyPe(), CmiMyNode());*/

    if (PCQueueLength(cs->recv)==0)
      AdvanceCommunications();

    if (PCQueueLength(cs->recv)==0) return NULL;

#if CMK_SMP
    //CmiLock(procState[cs->rank].recvLock);
#endif

    msg =  PCQueuePop(cs->recv);

#if CMK_SMP
    //CmiUnlock(procState[cs->rank].recvLock);
#endif

    return msg;
}

static void CmiSendSelf(char *msg) {
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        /* CmiBecomeNonImmediate(msg); */
        //printf("In SendSelf, N[%d]P[%d]R[%d] received an imm msg with hdl: %p\n", CmiMyNode(), CmiMyPe(), CmiMyRank(), CmiGetHandler(msg));
        CmiPushImmediateMsg(msg);
#if CMK_MULTICORE
        CmiHandleImmediate();
#endif
        return;
    }
#endif
    
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
}

#if CMK_SMP
static void CmiSendPeer (int rank, int size, char *msg) {
#if CMK_BROADCAST_SPANNING_TREE
    if (CMI_BROADCAST_ROOT(msg) != 0) {
        char *copymsg;
        copymsg = (char *)CmiAlloc(size);
        CmiMemcpy(copymsg,msg,size);

        CmiLock(procState[rank].bcastLock);
        PCQueuePush(CpvAccessOther(broadcast_q, rank), copymsg);
        CmiUnlock(procState[rank].bcastLock);
    }
#endif
    
    CmiPushPE (rank, msg);
}
#endif


void CmiGeneralFreeSendN (int node, int rank, int size, char * msg);


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

    CmiGeneralFreeSendN (CmiNodeOf (destPE), CmiRankOf (destPE), size, msg);
}

void CmiGeneralFreeSendN (int node, int rank, int size, char * msg) {

    //printf ("%d, %d: Sending Message to node %d rank %d \n", CmiMyPe(),
    //  CmiMyNode(), node, rank);

#if CMK_SMP
    CMI_DEST_RANK(msg) = rank;
    //CMI_SET_CHECKSUM(msg, size);

    if (node == CmiMyNode()) {
        CmiSendPeer (rank, size, msg);
        return;
    }
#endif

    pami_endpoint_t target;
#if CMK_PAMI_MULTI_CONTEXT
    size_t dst_context = (rank != SMP_NODEMESSAGE) ? (rank>>2) : 0;
#else
    size_t dst_context = 0;
#endif
    PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)node, dst_context, &target);

    //fprintf (stderr, "Calling PAMI Send to %d magic %d size %d\n", node, CMI_MAGIC(msg), size);
    if (size < 128) {
      pami_send_immediate_t parameters;
      parameters.dispatch        = CMI_PAMI_DISPATCH;
      parameters.header.iov_base = NULL;
      parameters.header.iov_len  = 0;
      parameters.data.iov_base   = msg;
      parameters.data.iov_len    = size;
      parameters.dest = target;
      
      pami_context_t my_context = MY_CONTEXT();
      CmiAssert (my_context != NULL);

#if CMK_SMP
      PAMI_Context_lock(my_context);
#endif
      PAMI_Send_immediate (my_context, &parameters);
#if CMK_SMP
      PAMI_Context_unlock(my_context);
#endif
      CmiFree(msg);
    }
    else {
      pami_send_t parameters;
      parameters.send.dispatch        = CMI_PAMI_DISPATCH;
      parameters.send.header.iov_base = NULL;
      parameters.send.header.iov_len  = 0;
      parameters.send.data.iov_base   = msg;
      parameters.send.data.iov_len    = size;
      parameters.events.cookie        = msg;
      parameters.events.local_fn      = send_done;
      parameters.events.remote_fn     = NULL;
      memset(&parameters.send.hints, 0, sizeof(parameters.send.hints));
      parameters.send.dest = target;

      pami_context_t my_context = MY_CONTEXT();
      CmiAssert (my_context != NULL);
      
#if CMK_SMP
      PAMI_Context_lock(my_context);
#endif
      INCR_MSGQLEN();
      PAMI_Send (my_context, &parameters);
#if CMK_SMP
      PAMI_Context_unlock(my_context);
#endif
    }
}

void CmiSyncSendFn(int destPE, int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
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

    //  asm volatile("sync" ::: "memory");

    CMI_MAGIC(copymsg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)copymsg)->size = size;
    CMI_SET_CHECKSUM(copymsg, size);

    CmiGeneralFreeSend(destPE,size,copymsg);
}

/* send msg to its spanning children in broadcast. G. Zheng */
void SendSpanningChildren(int size, char *msg) {
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
	
	CmiGeneralFreeSendN(p,0,size,copymsg);	
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
    SendSpanningChildren(size, msg);
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
    SendSpanningChildren(size, msg);
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

void AdvanceCommunications() {

    pami_context_t my_context = MY_CONTEXT();
   
#if CMK_SMP
    CmiAssert (my_context != NULL);
    PAMI_Context_trylock_advancev(&my_context, 1, 1);
#else
    PAMI_Context_advance(my_context, 1);
#endif
    
    sendBroadcastMessages();
#if CMK_NODE_QUEUE_AVAILABLE
    sendBroadcastMessagesNode();
#endif
    
    
#if CMK_IMMEDIATE_MSG && CMK_MULTICORE
    CmiHandleImmediate();
#endif
}

#if 0
static void SendMsgsUntil(int targetm) {

    pami_context_t my_context = MY_CONTEXT();

    while (MSGQLEN() > targetm) {
#if CMK_SMP
      PAMI_Context_trylock_advancev(&my_context, 1, 1);
#else
      PAMI_Context_advance(my_context, 1);
#endif    
    }
}
#endif

void CmiNotifyIdle() {
  AdvanceCommunications();
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

//#define OPTIMIZED_MULTICAST  0

void CmiFreeListSendFn(int npes, int *pes, int size, char *msg) {

    CMI_SET_BROADCAST_ROOT(msg,0);
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)msg)->size = size;
    CMI_SET_CHECKSUM(msg, size);

    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeListSendFn on comm thd on node %d\n", CmiMyNode());

    //printf("%d: In Free List Send Fn\n", CmiMyPe());
    int new_npes = 0;

    int i, count = 0, my_loc = -1;
    for (i=0; i<npes; i++) {
        if (CmiNodeOf(pes[i]) == CmiMyNode()) 
            CmiSyncSend(pes[i], size, msg);
    }

    for (i=0;i<npes;i++) {
        if (CmiNodeOf(pes[i]) == CmiMyNode());
        else if (i < npes - 1) {
#if !CMK_SMP 
            CmiReference(msg);
            CmiGeneralFreeSend(pes[i], size, msg);
#else
            CmiSyncSend(pes[i], size, msg);
#endif
        }
    }

    if (npes  && CmiNodeOf(pes[npes-1]) != CmiMyNode())
      CmiSyncSendAndFree(pes[npes-1], size, msg); //Sameto CmiFreeSendFn
    else
      CmiFree(msg);    
}

CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int size, char *msg) {
    CmiAbort("CmiAsyncListSendFn not implemented.");
    return (CmiCommHandle) 0;
}
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
    CmiNetworkBarrier();
  CmiNodeBarrier();
  return 0;
}

static void CmiNetworkBarrier() {
    //mysleep(1000000000UL);

    pami_result_t result;
    pami_barrier_flag = 1;
    pami_context_t my_context = cmi_pami_contexts[0];
#if CMK_SMP
    PAMI_Context_lock(my_context);
#endif
    result = PAMI_Collective(my_context, &pami_barrier);
    
#if CMK_SMP
    PAMI_Context_unlock(my_context);
#endif    
    
    if (result != PAMI_SUCCESS)
    {
      fprintf (stderr, "Error. Unable to issue  collective. result = %d\n", result);
      return;
    }
    
#if CMK_SMP
    PAMI_Context_lock(my_context);
#endif
    while (pami_barrier_flag)
      result = PAMI_Context_advance (my_context, 100);
#if CMK_SMP
    PAMI_Context_unlock(my_context);
#endif
}

#if CMK_NODE_QUEUE_AVAILABLE
static void CmiSendNodeSelf(char *msg) {
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        //printf("SendNodeSelf: N[%d]P[%d]R[%d] received an imm msg with hdl: %p\n", CmiMyNode(), CmiMyPe(), CmiMyRank(), CmiGetHandler(msg));
        CmiPushImmediateMsg(msg);
#if CMK_MULTICORE
        CmiHandleImmediate();
#endif
        return;
    }
#endif    
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv, msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
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
    
    CQdCreate(CpvAccess(cQdState), 1);

    if (node == _Cmi_mynode) {
        CmiSendNodeSelf(msg);
    } else {
        CmiGeneralFreeSendN(node, SMP_NODEMESSAGE, size, msg);
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

void SendSpanningChildrenNode(int size, char *msg) {
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
        CmiGeneralFreeSendN(nid, SMP_NODEMESSAGE, size, dupmsg);
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

    SendSpanningChildrenNode(s, m);
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


//void bzero (void *__s, size_t __n) {
//  memset(__s, 0, __n);
//}

