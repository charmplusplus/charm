/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
*/

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "converse.h"

#define VMI_DEVICENAME "converse"

#include "vmi.h"




#define CMI_VMI_COLLECT_STATISTICS 0


/* This is the number of seconds to wait for connection setup. */
#define CMI_VMI_CONNECTION_TIMEOUT 300

/*
  Asynchronous messages of fewer bytes than the following boundary are
  sent by calling VMI_Stream_Send_Inline() which does an internal memcpy()
  and dispatches the message synchronously.  For sufficiently-short messages,
  this is faster than paying the cost of getting some pinned memory from
  cache.
*/
#define CMI_VMI_VERYSHORT_MESSAGE_BOUNDARY 512

/*
  Messages of fewer bytes than the following boundary are sent by the
  "short message protocol".  Right now, this means that they will be sent
  via a VMI Stream Send of some type.

  Messages larger than the boundary are sent by the "large message protocol".
  This is an RDMA rendezvous protocol where a rendezvous message is first
  sent via a stream to initiate an RDMA transfer of the message data.
*/
#define CMI_VMI_SHORT_MESSAGE_BOUNDARY 4096

/*
  Receive contexts are stored in an array of CMI_VMI_MAX_RECEIVE_HANDLES
  elements.
*/
#define CMI_VMI_MAX_RECEIVE_HANDLES 1000

/*
  RDMA puts for large messages are done in a pipeline.  The following two
  defines are for the maximum length of the pipeline (alternatively, the
  maximum number of RDMA send buffers that may be outstanding at once) and
  the maximum size of an individual chunk of message data that can be
  transferred in a single RDMA put.  This is to prevent VMI from trying to
  pin down unusually-large amounts of memory.
*/
#define CMI_VMI_RDMA_MAX_OUTSTANDING 3
#define CMI_VMI_RDMA_MAX_CHUNK 262144



#if CONVERSE_VERSION_VMI
#define CMI_VMI_BUCKET1_SIZE 1024
#define CMI_VMI_BUCKET2_SIZE 2048
#define CMI_VMI_BUCKET3_SIZE 4096
#define CMI_VMI_BUCKET4_SIZE 8192
#define CMI_VMI_BUCKET5_SIZE 16384

#define CMI_VMI_BUCKET1_PREALLOCATE 100
#define CMI_VMI_BUCKET1_GROW         50

#define CMI_VMI_BUCKET2_PREALLOCATE 100
#define CMI_VMI_BUCKET2_GROW         50

#define CMI_VMI_BUCKET3_PREALLOCATE 100
#define CMI_VMI_BUCKET3_GROW         50

#define CMI_VMI_BUCKET4_PREALLOCATE 100
#define CMI_VMI_BUCKET4_GROW         50

#define CMI_VMI_BUCKET5_PREALLOCATE 100
#define CMI_VMI_BUCKET5_GROW         50
#endif




/*
  If CMI_VMI_COLLECT_STATISTICS is defined, the following definitions set
  the boundaries for the sizes of messages that are counted for each type
  of send/broadcast.  Each send/broadcast routine has three separate
  counters for keeping track of various size messages.  Messages of less
  than BUCKET1 bytes are counted by the first counter; messages of less
  than BUCKET2 bytes are counted by the second counter; all other messages
  are counted by the third counter.  At program termination, the values of
  the various counters are printed for analysis.
*/
#if CMI_VMI_COLLECT_STATISTICS
#define CMI_VMI_SYNCSEND_BUCKET1_BOUNDARY 512
#define CMI_VMI_SYNCSEND_BUCKET2_BOUNDARY 4096

#define CMI_VMI_ASYNCSEND_BUCKET1_BOUNDARY 512
#define CMI_VMI_ASYNCSEND_BUCKET2_BOUNDARY 4096

#define CMI_VMI_FREESEND_BUCKET1_BOUNDARY 512
#define CMI_VMI_FREESEND_BUCKET2_BOUNDARY 4096

#define CMI_VMI_SYNCBROADCAST_BUCKET1_BOUNDARY 512
#define CMI_VMI_SYNCBROADCAST_BUCKET2_BOUNDARY 4096

#define CMI_VMI_ASYNCBROADCAST_BUCKET1_BOUNDARY 512
#define CMI_VMI_ASYNCBROADCAST_BUCKET2_BOUNDARY 4096

#define CMI_VMI_FREEBROADCAST_BUCKET1_BOUNDARY 512
#define CMI_VMI_FREEBROADCAST_BUCKET2_BOUNDARY 4096

#define CMI_VMI_SYNCBROADCASTALL_BUCKET1_BOUNDARY 512
#define CMI_VMI_SYNCBROADCASTALL_BUCKET2_BOUNDARY 4096

#define CMI_VMI_ASYNCBROADCASTALL_BUCKET1_BOUNDARY 512
#define CMI_VMI_ASYNCBROADCASTALL_BUCKET2_BOUNDARY 4096

#define CMI_VMI_FREEBROADCASTALL_BUCKET1_BOUNDARY 512
#define CMI_VMI_FREEBROADCASTALL_BUCKET2_BOUNDARY 4096
#endif



/*
  The following definitions are used to set various buffer pool sizes for
  the number of entries to preallocate when the pool is created and the
  number of entries to increase the pool when the pool is exhausted.
*/
#define CMI_VMI_MESSAGE_BUFFER_POOL_PREALLOCATE       64
#define CMI_VMI_MESSAGE_BUFFER_POOL_GROW              32

#define CMI_VMI_CMICOMMHANDLE_POOL_PREALLOCATE        64
#define CMI_VMI_CMICOMMHANDLE_POOL_GROW               32

#define CMI_VMI_HANDLE_POOL_PREALLOCATE               64
#define CMI_VMI_HANDLE_POOL_GROW                      32

#define CMI_VMI_RDMA_BYTES_SENT_POOL_PREALLOCATE      64
#define CMI_VMI_RDMA_BYTES_SENT_POOL_GROW             32

#define CMI_VMI_RDMA_PUT_CONTEXT_POOL_PREALLOCATE     64
#define CMI_VMI_RDMA_PUT_CONTEXT_POOL_GROW            32

#define CMI_VMI_RDMA_RECEIVE_CONTEXT_POOL_PREALLOCATE 64
#define CMI_VMI_RDMA_RECEIVE_CONTEXT_POOL_GROW        32

#define CMI_VMI_RDMA_CACHE_ENTRY_POOL_PREALLOCATE     64
#define CMI_VMI_RDMA_CACHE_ENTRY_POOL_GROW            32



/*
  If CMK_OPTIMIZE is defined, various optimizations are made to Charm++.
  Inside this machine.c, these optimizations include not checking the
  return codes of calls into VMI.  The following macro definition is how
  this optimization is carried out.
*/
#ifdef CMK_OPTIMIZE
#define CMI_VMI_CHECK_SUCCESS(status,message)
#else
#define CMI_VMI_CHECK_SUCCESS(status,message)                             \
          if (!VMI_SUCCESS (status)) {                                    \
            VMI_perror (message, status);                                 \
          }
#endif



/*
  If CMK_BROADCAST_SPANNING_TREE is defined, broadcasts are done via a
  spanning tree.  (Otherwise, broadcasts are done by iterating through
  each process in the process list and sending a separate message.)
*/
#if CMK_BROADCAST_SPANNING_TREE
#ifndef CMI_VMI_BROADCAST_SPANNING_FACTOR
#define CMI_VMI_BROADCAST_SPANNING_FACTOR 4
#endif

#define CMI_BROADCAST_ROOT(msg)   ((CmiMsgHeaderBasic *)msg)->root
#define CMI_DEST_RANK(msg)        ((CmiMsgHeaderBasic *)msg)->rank

#define CMI_SET_BROADCAST_ROOT(msg,root)   CMI_BROADCAST_ROOT(msg) = (root);
#endif



/* The following data structures hold the per-process state information. */
typedef enum
{
  CMI_VMI_CONNECTION_CONNECTING,
  CMI_VMI_CONNECTION_CONNECTED,
  CMI_VMI_CONNECTION_DISCONNECTING,
  CMI_VMI_CONNECTION_DISCONNECTED,
  CMI_VMI_CONNECTION_ERROR
} CMI_VMI_Connection_State_T;

typedef struct
{
  int                        rank;
  int                        nodeIP;
  CMI_VMI_Connection_State_T state;
  PVMI_CONNECT               connection;
} CMI_VMI_Process_Info_T;


/*
  The following data structures describe the various types of messages
  that can be sent in this machine layer.

  CMI_VMI_Connect_Message_T are connect messages that are only sent during
  the connection setup phase.

  CMI_VMI_Message_T are the standard messages used throughout the system.
  Each message consists of a CMI_VMI_Message_Header_T which identifies the
  type of message (currently, either a "short" message or a "rendezvous"
  message) and the size of the message data, and a CMI_VMI_Message_Body_T
  which is either the message data (for "short" messages) or the address
  of an RDMA context on the sending side which the receiver will reference
  when it publishes a buffer for the sender to put data into.
*/
typedef struct
{
  int rank;
} CMI_VMI_Connect_Message_T;

typedef enum
{
  CMI_VMI_MESSAGE_TYPE_SHORT,
  CMI_VMI_MESSAGE_TYPE_RENDEZVOUS
} CMI_VMI_Message_Type_T;

typedef struct
{
  CMI_VMI_Message_Type_T type;
  int                    msgsz;
} CMI_VMI_Message_Header_T;

typedef struct
{
  char msg;
} CMI_VMI_Message_Body_Short_T;

typedef struct
{
  VMI_virt_addr_t addr;
} CMI_VMI_Message_Body_Rendezvous_T;

typedef union
{
  CMI_VMI_Message_Body_Short_T      shortmsg;
  CMI_VMI_Message_Body_Rendezvous_T rendezvous;
} CMI_VMI_Message_Body_T;

typedef struct
{
  CMI_VMI_Message_Header_T hdr;
  CMI_VMI_Message_Body_T   body;
} CMI_VMI_Message_T;



/*
  The following data structures define the various types of send handles
  that are used throughout the machine layer.  These handles are used to
  hold the context for send operations that are currently underway.

  CMI_VMI_CmiCommHandle_T is the internal representation for CmiCommHandle
  which are returned to the higher-level Converse/Charm++ caller on an
  asynchronous send/broadcast.  The caller can use this handle to
  determine when the asynchronous operation has completed (by calling
  CmiAsyncMsgSent() with the CmiCommHandle) and then discards the handle
  when it is no longer needed (by calling CmiReleaseCommHandle()).

  CMI_VMI_Handle_T is only used internally to this machine layer to hold
  context for synchronous and asynchronous send/broadcast operations that
  are done via stream or RDMA.  If the "commhandle" field is set to point
  to a CMI_VMI_CmiCommHandle_T, the corresponding CmiCommHandle will be
  updated to reflect the state of the send.  The "refcount" field is
  examined by both the stream send completion function and the RDMA
  completion function and if the refcount is less than one, the handle is
  deallocated (along with the various other fields contained within).  This
  is normally the case in an asynchronous send operation.  Otherwise the
  completion handler assumes that either the send is still going on or that
  the caller will handle the deallocation of the handle and the other
  compoments of the send context.  (This is normally the case in a
  synchronous send operation.)  THEREFORE, IN THE CASE OF A SYNCHRONOUS
  SEND/BROADCAST, THE CALLER MUST SET THE VALUE OF THE REFCOUNT TO *ONE
  GREATER* THAN THE NUMBER OF SENDS DISPATCHED IN ORDER TO WAIT FOR THE
  OPERATION TO COMPLETE, OTHERWISE THE COMPLETION HANDLER WILL DEALLOCATE
  THE HANDLE!
*/
typedef struct
{
  int count;
} CMI_VMI_CmiCommHandle_T;

typedef enum
{
  CMI_VMI_HANDLE_TYPE_SYNC_SEND_STREAM,
  CMI_VMI_HANDLE_TYPE_ASYNC_SEND_STREAM,

  CMI_VMI_HANDLE_TYPE_SYNC_BROADCAST_STREAM,
  CMI_VMI_HANDLE_TYPE_ASYNC_BROADCAST_STREAM,

  CMI_VMI_HANDLE_TYPE_SYNC_SEND_RDMA,
  CMI_VMI_HANDLE_TYPE_ASYNC_SEND_RDMA,

  CMI_VMI_HANDLE_TYPE_SYNC_BROADCAST_RDMA,
  CMI_VMI_HANDLE_TYPE_ASYNC_BROADCAST_RDMA
} CMI_VMI_Handle_Type_T;

typedef struct
{
  PVMI_CACHE_ENTRY   cacheentry;
  CMI_VMI_Message_T *vmimsg;
} CMI_VMI_Handle_Stream_T;

typedef struct
{
  int bytes_sent;
} CMI_VMI_Handle_RDMA_T;

typedef struct
{
  int *bytes_sent;
} CMI_VMI_Handle_RDMABROAD_T;

typedef struct
{
  int                      refcount;
  char                    *msg;
  int                      msgsize;
  CMI_VMI_CmiCommHandle_T *commhandle;
  CMI_VMI_Handle_Type_T    type;

  union
  {
    CMI_VMI_Handle_Stream_T    stream;
    CMI_VMI_Handle_RDMA_T      rdma;
    CMI_VMI_Handle_RDMABROAD_T rdmabroad;
  } data;
} CMI_VMI_Handle_T;



/*
  The following data structures hold details of RDMA "long message protocol"
  contexts.

  CMI_VMI_RDMA_Context_T

  CMI_VMI_RDMA_Put_Context_T holds details of an RDMA put context.  This
  data structure is necessary due to RDMA broadcasts since each Put requires
  a separate cache entry (for each Put in an individual pipeline, and for
  each pipeline to each receiver).  It is tempting to believe we could
  allocate an array of cache entries within the RDMA send handle, except
  this approach will not work because there is no way for the RDMA
  completion handler to determine the rank of the process for which the Put
  completed and hence no way to index this array.
*/
typedef struct
{
  BOOLEAN           allocated;
  char             *msg;
  int               msgsize;
  int               bytes_pub;
  int               rdmacnt;
  int               sindx;
  int               rindx;
  int               bytes_rec;
  PVMI_CACHE_ENTRY *cacheentry;
  VMI_virt_addr_t   rhandleaddr;
} CMI_VMI_RDMA_Receive_Context_T;

typedef struct
{
  PVMI_CACHE_ENTRY  cacheentry;
  CMI_VMI_Handle_T *handle;
} CMI_VMI_RDMA_Put_Context_T;


/*********************/
/* BEGIN CRUFTY CODE */
/*********************/

#define MAX_STR_LENGTH 256
#define CRM_DEFAULT_PORT 7777
#define SOCKET_ERROR -1

/* Message Codes to CRM  */
#define CRM_MSG_SUCCESS		0 /* Response from CRM */
#define CRM_MSG_REGISTER	1
#define CRM_MSG_REMOVE		2
#define CRM_MSG_FAILED		3

/* Register Context. */
typedef struct regMsg{
  int np; /* Number of processors. */
  int shutdownPort; /* Port shutdown server is runnig on */
  int keyLength;
  char key[MAX_STR_LENGTH];
} regMsg, *PRegMsg;

/* Remove CRM Context Request Message */
typedef struct delMsg{
  int keyLength;
  char key[MAX_STR_LENGTH];
} delMsg, *PDelMsg;

/* Error Codes for failed response */
typedef struct errMsg{
#define CRM_ERR_CTXTCONFLICT	1
#define CRM_ERR_INVALIDCTXT	2
#define CRM_ERR_TIMEOUT         3
#define CRM_ERR_OVERFLOW        4
  int errCode;
} errMsg, *PErrMsg;

/* Response for valid registration */
typedef struct nodeCtx{
  int nodeIP;
  int shutdownPort;
  int nodePE;
} nodeCtx, *PNodeCtx;

typedef struct ctxMsg{
  int np; /* # of PE */
  struct nodeCtx *node;
} ctxMsg, *PCtxMsg;

typedef struct CRM_Msg{
  int msgCode;
  union{
    regMsg CRMReg;
    delMsg CRMDel;
    errMsg CRMErr;
    ctxMsg CRMCtx;
  } msg;
} CRM_Msg, *PCRM_Msg;

/*******************/
/* END CRUFTY CODE */
/*******************/


/* Function prototypes appear below. */
VMI_CONNECT_RESPONSE CMI_VMI_Connection_Accept_Handler (PVMI_CONNECT inconn,
     PVMI_SLAB slab, ULONG size);
void CMI_VMI_Connection_Response_Handler (PVOID context, PVOID response,
     USHORT size, PVOID handle, VMI_CONNECT_RESPONSE status);
void CMI_VMI_Connection_Disconnect_Handler (IN PVMI_CONNECT conn);
void CMI_VMI_Disconnection_Response_Handler (PVMI_CONNECT conn,
     PVOID context, VMI_STATUS status);
VMI_RECV_STATUS CMI_VMI_Stream_Receive_Handler (PVMI_CONNECT conn,
     PVMI_STREAM_RECV stream, VMI_STREAM_COMMAND command, PVOID context,
     PVMI_SLAB slab);
void CMI_VMI_Stream_Completion_Handler (PVOID ctxt, VMI_STATUS sstatus);
void CMI_VMI_RDMA_Fragment_Handler (PVMI_RDMA_OP op, PVOID ctxt,
     VMI_STATUS rstatus);
void CMI_VMI_RDMA_Completion_Handler (PVMI_RDMA_OP op, PVOID ctxt,
     VMI_STATUS rstatus);
void CMI_VMI_RDMA_Publish_Handler (PVMI_CONNECT conn, PVMI_REMOTE_BUFFER rbuf);
void CMI_VMI_RDMA_Notification_Handler (PVMI_CONNECT conn, UINT32 rdmasz,
     UINT32 context, VMI_STATUS rstatus);

int CMI_VMI_CRM_Register (PUCHAR key, int numProcesses, BOOLEAN reg);
BOOLEAN CMI_VMI_Open_Connections (PUCHAR synckey);
int CMI_VMI_Get_RDMA_Receive_Context ();
void *CMI_VMI_CmiAlloc (int size);
void CMI_VMI_CmiFree (void *ptr);
#if CMK_BROADCAST_SPANNING_TREE
int CMI_VMI_Spanning_Children_Count (char *msg);
void CMI_VMI_Send_Spanning_Children (int msgsize, char *msg);
#endif

void ConverseInit (int argc, char **argv, CmiStartFn startFn,
		   int userCallsScheduler, int initReturns);
void ConverseExit ();
void CmiAbort (const char *message);
void *CmiGetNonLocal (void);
void CmiMemLock ();
void CmiMemUnlock ();
void CmiNotifyIdle ();
void CmiSyncSendFn (int destrank, int msgsize, char *msg);
CmiCommHandle CmiAsyncSendFn (int destrank, int msgsize, char *msg);
void CmiFreeSendFn (int destrank, int msgsize, char *msg);
void CmiSyncBroadcastFn (int msgsize, char *msg);
CmiCommHandle CmiAsyncBroadcastFn (int msgsize, char *msg);
void CmiFreeBroadcastFn (int msgsize, char *msg);
void CmiSyncBroadcastAllFn (int msgsize, char *msg);
CmiCommHandle CmiAsyncBroadcastAllFn (int msgsize, char *msg);
void CmiFreeBroadcastAllFn (int msgsize, char *msg);
int CmiAllAsyncMsgsSent ();
int CmiAsyncMsgSent (CmiCommHandle cmicommhandle);
void CmiReleaseCommHandle (CmiCommHandle cmicommhandle);

BOOLEAN CRMInit ();
SOCKET createSocket(char *hostName, int port, int *localAddr);
BOOLEAN CRMRegister (char *key, ULONG numPE, int shutdownPort,
		     SOCKET *clientSock, int *clientAddr, PCRM_Msg *msg2);
BOOLEAN CRMParseMsg (PCRM_Msg msg, int rank, int *nodeIP,
		     int *shutdownPort, int *nodePE);
int CRMRecv (SOCKET s, char *msg, int n);
int CRMSend (SOCKET s, char *msg, int n);
