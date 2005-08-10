/**
 ** Ammasso implementation of Converse NET version Contains Ammasso specific
 ** code for: CmiMachineInit() CmiNotifyIdle() DeliverViaNetwork()
 ** CommunicationServer() CmiMachineExit()
 **
 ** Written By:    Isaac Dooley      idooley2@uiuc.edu
 ** 03/12/05       Esteban Pauli     etpauli2@uiuc.edu
 **                Filippo Gioachin  gioachin@uiuc.edu
 **                David Kunzman     kunzman2@uiuc.edu
 **
 ** Change Log:
 **   03/12/05 : DMK : Initial Version
 **   04/30/05 : Filippo : Revised Version
 **
 ** Todo List:
 **
 **/


////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines, Types, etc. ////////////////////////////////////////////////////////////////////////////

#define AMMASSO_PORT        2583

/* The size of the initial allocated buffers is AMMASSO_BUFSIZE*AMMASSO_INITIAL_BUFFERS
   with 8500, the buffers become aligned to 64 bytes (16 bytes per buffer are support) */
#define AMMASSO_BUFSIZE            8488
#define AMMASSO_INITIAL_BUFFERS    1024
#define AMMASSO_BUFFERS_INFLY       128

/*
#define AMMASSO_CTRLMSG_LEN        7
#define CtrlHeader_Construct(buf, ctrlType)  { *((int*)buf) = Cmi_charmrun_pid;                               \
                                               *((short*)((char*)buf + sizeof(int))) = contextBlock->myNode;  \
                                               *((char*)buf + sizeof(int) + sizeof(short)) = ctrlType;        \
                                             }

#define CtrlHeader_GetCharmrunPID(buf)   (*((int*)buf))
#define CtrlHeader_GetNode(buf)          (*((short*)((char*)buf + sizeof(int))))
#define CtrlHeader_GetCtrlType(buf)      (*((char*)buf + sizeof(int) + sizeof(short)))
*/

#define ACK_MASK               (1<<31)

/* ACK_WRAPPING is used when the ACK overflows, rare condition */
#define ACK_WRAPPING           0x01
/* AMMASSO_ALLOCATE is used by the receiving side to allocate more buffers on
   the sender. The sender will reply with a AMMASSO_ALLOCATED message,
   specifying so where they have been allocated. AMMASSO_MOREBUFFERS is instead
   a message sent from the sender to the receiver asking for more buffers. The
   receiver can decide to allocate more buffers or just ignore the message. */
#define AMMASSO_ALLOCATE       0x02
#define AMMASSO_ALLOCATED      0x04
#define AMMASSO_MOREBUFFERS    0x08
/* AMMASSO_RELEASE is used by the receiver to ask the sender to release a
   certain amount of buffers which are allocated to it. The sender can answer
   with an AMMASSO_RELEASED message specifying how many buffers have been
   released. The sender can also decide to ignore an AMMASSO_RELEASE message, or
   to release some buffers on its own idea. */
#define AMMASSO_RELEASE        0x10
#define AMMASSO_RELEASED       0x20
/* AMMASSO_READY is used at the beginning of the program to synchronize all
   processors */
#define AMMASSO_READY          0x40

typedef struct __cmi_idle_state {
  char none;
} CmiIdleState;

typedef struct __ammasso_private_data {
  int node;
  cc_stag_t stag;
  cc_uint64_t to;
  cc_uint64_t ack_to;
} AmmassoPrivateData;

typedef struct __ammasso_token_description {
  cc_stag_t stag;
  cc_uint64_t to;
} AmmassoTokenDescription;

typedef CmiUInt4 ammasso_ack_t;

/* Structures to deal with the buffering */
typedef struct __ammasso_tailer {
  ammasso_ack_t ack;
  char flags;
  char pad;
  CmiUInt2 length;
} PACKED Tailer;

/* AMMASSO_BUFSIZE should be updated if this structure size is changed */
typedef struct __ammasso_buffer {
  char buf[AMMASSO_BUFSIZE];
  Tailer tail;
  union {
    struct {
      cc_stag_t stag;
      struct __ammasso_buffer *next;
    };
    char pad[12];
  };
} PACKED AmmassoBuffer;

typedef struct __ammasso_token {
  AmmassoBuffer *localBuf;
  AmmassoBuffer *remoteBuf;
  struct __ammasso_token *next;
  cc_sq_wr_t wr;
} AmmassoToken;

// if "name" is null, then "num_##name"==0 and "last_##name" is not defined
#define LIST_DEFINE(type, name) \
    type *name; \
    type *last_ ## name; \
    int    num_ ## name

// NOTE: in order to use LIST_*, no space has to be present in the parenthesis
#define LIST_ENQUEUE(prefix, suffix, newtoken) \
    prefix num_ ## suffix ++; \
    newtoken ->next = NULL; \
    if (prefix suffix != NULL) { \
      prefix last_ ## suffix ->next = newtoken; \
    } else { \
      prefix suffix = newtoken; \
    } \
    prefix last_ ## suffix = newtoken

#define LIST_DEQUEUE(prefix, suffix, thetoken) \
    CmiAssert(prefix num_ ## suffix > 0); \
    prefix num_ ## suffix --; \
    thetoken = prefix suffix; \
    prefix suffix = thetoken ->next

// DMK : This is copied from "qp_rping.c" in Ammasso's Example Code (modify as needed for our machine layer).
// User Defined Context that will be sent to function handlers for asynchronous events.
typedef struct __context_block {

  cc_rnic_handle_t       rnic;     // The RNIC Handle
  cc_eh_ce_handler_id_t  eh_id;    // Event Handler ID
  cc_pdid_t              pd_id;    // Protection Domain Handle

  // DMK : TODO : When there is a global buffer pool, the send and recv (or just single) completion queues' handles can go here
  //              instead of in the OtherNode structure.  All of the memory buffers will move here also.

  // Extra Useful Info
  int                    numNodes;                    // The number of nodes total
  int                    myNode;                      // "My" node index for this node
  int                    outstandingConnectionCount;  // Synch. variable used to block until all connections to other nodes are made
  int                    nodeReadyCount;              // Synch. variable used to block until all other nodes are ready

  // Free Receive Buffer Pool, typically empty when allocated to processors
  LIST_DEFINE(AmmassoBuffer,freeRecvBuffers);
  //CmiNodeLock bufferPoolLock;
  LIST_DEFINE(AmmassoToken,freeTokens);

  int pinnedMemory;
  int conditionRegistered;
} mycb_t;

// Global instance of the mycb_t structure to be used throughout this machine layer
mycb_t *contextBlock = NULL;

// NOTE: I couldn't find functions similar to these in the CCIL API, if found, use the API ones instead.  They
//   are basically just utility functions.
char* cc_status_to_string(cc_status_t errorCode);
char* cc_conn_error_to_string(cc_connect_status_t errorCode);
void displayQueueQuery(cc_qp_handle_t qp, cc_qp_query_attrs_t *attrs);
char* cc_qp_state_to_string(cc_qp_state_t qpState);
char* cc_event_id_to_string(cc_event_id_t id);
char* cc_connect_status_to_string(cc_connect_status_t status);


#define AMMASSO_STATS   0
#if AMMASSO_STATS

#define AMMASSO_STATS_VARS(event)   double event ## _start;    \
                                    double event ## _end;      \
                                    double event ## _total;    \
                                    long event ## _count;

typedef struct __ammasso_stats {

  AMMASSO_STATS_VARS(MachineInit)

  AMMASSO_STATS_VARS(AmmassoDoIdle)

  AMMASSO_STATS_VARS(DeliverViaNetwork)
  AMMASSO_STATS_VARS(DeliverViaNetwork_pre_lock)
  AMMASSO_STATS_VARS(DeliverViaNetwork_lock)
  AMMASSO_STATS_VARS(DeliverViaNetwork_post_lock)
  AMMASSO_STATS_VARS(DeliverViaNetwork_send)

  AMMASSO_STATS_VARS(getQPSendBuffer)
  AMMASSO_STATS_VARS(getQPSendBuffer_lock)
  AMMASSO_STATS_VARS(getQPSendBuffer_CEH)
  AMMASSO_STATS_VARS(getQPSendBuffer_loop)

  AMMASSO_STATS_VARS(sendDataOnQP)
  AMMASSO_STATS_VARS(sendDataOnQP_pre_send)
  AMMASSO_STATS_VARS(sendDataOnQP_send)
  AMMASSO_STATS_VARS(sendDataOnQP_post_send)

  AMMASSO_STATS_VARS(sendDataOnQP_1024)
  AMMASSO_STATS_VARS(sendDataOnQP_2048)
  AMMASSO_STATS_VARS(sendDataOnQP_4096)
  AMMASSO_STATS_VARS(sendDataOnQP_16384)
  AMMASSO_STATS_VARS(sendDataOnQP_over)

  AMMASSO_STATS_VARS(AsynchronousEventHandler)
  AMMASSO_STATS_VARS(CompletionEventHandler)
  AMMASSO_STATS_VARS(ProcessMessage)
  AMMASSO_STATS_VARS(processAmmassoControlMessage)

  AMMASSO_STATS_VARS(sendAck)

  AMMASSO_STATS_VARS(CommunicationServer)

} AmmassoStats;

AmmassoStats __stats;

#define AMMASSO_STATS_INIT_AUX(event)  { __stats.event ## _total = 0.0; __stats.event ## _count = 0; }
#define AMMASSO_STATS_INIT   {                                                  \
                               AMMASSO_STATS_INIT_AUX(MachineInit)                  \
                               AMMASSO_STATS_INIT_AUX(AmmassoDoIdle)                \
                               AMMASSO_STATS_INIT_AUX(DeliverViaNetwork)            \
                               AMMASSO_STATS_INIT_AUX(DeliverViaNetwork_pre_lock)   \
                               AMMASSO_STATS_INIT_AUX(DeliverViaNetwork_lock)       \
                               AMMASSO_STATS_INIT_AUX(DeliverViaNetwork_post_lock)  \
                               AMMASSO_STATS_INIT_AUX(DeliverViaNetwork_send)       \
                               AMMASSO_STATS_INIT_AUX(getQPSendBuffer)              \
                               AMMASSO_STATS_INIT_AUX(getQPSendBuffer_lock)         \
                               AMMASSO_STATS_INIT_AUX(getQPSendBuffer_CEH)          \
                               AMMASSO_STATS_INIT_AUX(getQPSendBuffer_loop)         \
                               AMMASSO_STATS_INIT_AUX(sendDataOnQP)                 \
                               AMMASSO_STATS_INIT_AUX(sendDataOnQP_pre_send)        \
                               AMMASSO_STATS_INIT_AUX(sendDataOnQP_send)            \
                               AMMASSO_STATS_INIT_AUX(sendDataOnQP_post_send)       \
                               AMMASSO_STATS_INIT_AUX(sendDataOnQP_1024)            \
                               AMMASSO_STATS_INIT_AUX(sendDataOnQP_2048)            \
                               AMMASSO_STATS_INIT_AUX(sendDataOnQP_4096)            \
                               AMMASSO_STATS_INIT_AUX(sendDataOnQP_16384)           \
                               AMMASSO_STATS_INIT_AUX(sendDataOnQP_over)            \
                               AMMASSO_STATS_INIT_AUX(AsynchronousEventHandler)     \
                               AMMASSO_STATS_INIT_AUX(CompletionEventHandler)       \
                               AMMASSO_STATS_INIT_AUX(ProcessMessage)               \
                               AMMASSO_STATS_INIT_AUX(processAmmassoControlMessage) \
                               AMMASSO_STATS_INIT_AUX(sendAck)                      \
                               AMMASSO_STATS_INIT_AUX(CommunicationServer)          \
			     }

#define TO_NS  ((double)1000000000.0)

#define AMMASSO_STATS_START(event) { __stats.event ## _start = CmiWallTimer();     \
                                   }

#define AMMASSO_STATS_END(event)   { __stats.event ## _end = CmiWallTimer();         \
                                     __stats.event ## _count++;                      \
                                     __stats.event ## _total += (__stats.event ## _end - __stats.event ## _start);   \
                                   }

#define AMMASSO_STATS_DISPLAY_VERBOSE(event) { char buf[128];                        \
                                               CmiPrintf("[%d] Ammasso Stats: event -> " #event "_count = %d\n", CmiMyPe(), __stats.event ## _count);   \
                                               CmiPrintf("[%d]                event -> " #event "_total = %.3fns\n", CmiMyPe(), __stats.event ## _total * TO_NS);   \
                                               CmiPrintf("[%d]                                                    " #event " average: %.3fns\n", CmiMyPe(), (((double)__stats.event ## _total)/(__stats.event ## _count)) * TO_NS); \
                                             }

#define AMMASSO_STATS_DISPLAY(event) {                                               \
                                       CmiPrintf("[%d] " #event " average: %.3fns\n", CmiMyPe(), (((double)__stats.event ## _total)/(__stats.event ## _count)) * TO_NS); \
				     }

#else

#define AMMASSO_STATS_INIT_AUX(event)
#define AMMASSO_STATS_INIT
#define AMMASSO_STATS_START(event)
#define AMMASSO_STATS_END(event)
#define AMMASSO_STATS_DISPLAY(event)

#endif

