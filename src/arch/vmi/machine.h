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


#define CMI_VMI_USE_MEMORY_POOL   0
#define CMI_VMI_OPTIMIZE          0



#define CMI_VMI_MAX_HANDLES 1000


/* This is the number of seconds to wait for connection setup. */
#define CMI_VMI_CONNECTION_TIMEOUT 300

/* These are the message send strategy boundaries. */
#define CMI_VMI_SMALL_MESSAGE_BOUNDARY 512
#define CMI_VMI_MEDIUM_MESSAGE_BOUNDARY 4096

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
  The following definitions are used to set various buffer pool sizes for
  the number of entries to preallocate when the pool is created and the
  number of entries to increase the pool when the pool is exhausted.
*/
#define CMI_VMI_CMICOMMHANDLE_POOL_PREALLOCATE        64
#define CMI_VMI_CMICOMMHANDLE_POOL_GROW               32

#define CMI_VMI_RDMA_BYTES_SENT_POOL_PREALLOCATE      64
#define CMI_VMI_RDMA_BYTES_SENT_POOL_GROW             32

#define CMI_VMI_RDMA_PUT_CONTEXT_POOL_PREALLOCATE     64
#define CMI_VMI_RDMA_PUT_CONTEXT_POOL_GROW            32

#define CMI_VMI_RDMA_CACHE_ENTRY_POOL_PREALLOCATE     64
#define CMI_VMI_RDMA_CACHE_ENTRY_POOL_GROW            32



/*
  If CMI_VMI_OPTIMIZE is defined, various optimizations are made to Charm++.
  Inside this machine.c, these optimizations include not checking the
  return codes of calls into VMI.  The following macro definition is how
  this optimization is carried out.
*/
#if CMI_VMI_OPTIMIZE
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
  CMI_VMI_Connect_Message_T are connect messages that are only sent during
  the connection setup phase.
*/
typedef struct
{
  int rank;
} CMI_VMI_Connect_Message_T;





typedef struct
{
  char header[CmiMsgHeaderSizeBytes];
  int rank;
  int msgsize;
  VMI_virt_addr_t context;
} CMI_VMI_Rendezvous_Message_T;


typedef struct
{
  char header[CmiMsgHeaderSizeBytes];
  int rank;
  int maxsize;
  VMI_virt_addr_t context;
} CMI_VMI_Persistent_Request_Message_T;


typedef struct
{
  char header[CmiMsgHeaderSizeBytes];
  VMI_virt_addr_t context;
  int rdmarecvindx;
} CMI_VMI_Persistent_Grant_Message_T;


typedef struct
{
  char header[CmiMsgHeaderSizeBytes];
  int rdmarecvindx;
} CMI_VMI_Persistent_Destroy_Message_T;





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
  CMI_VMI_HANDLE_TYPE_SEND,
  CMI_VMI_HANDLE_TYPE_RECEIVE
} CMI_VMI_Handle_Type_T;


typedef enum
{
  CMI_VMI_SEND_HANDLE_TYPE_STREAM,
  CMI_VMI_SEND_HANDLE_TYPE_RDMA,
  CMI_VMI_SEND_HANDLE_TYPE_RDMABROAD,
  CMI_VMI_SEND_HANDLE_TYPE_PERSISTENT
} CMI_VMI_Send_Handle_Type_T;


typedef struct
{
  PVMI_CACHE_ENTRY cacheentry;
} CMI_VMI_Send_Handle_Stream_T;


typedef struct
{
  int bytes_sent;
} CMI_VMI_Send_Handle_RDMA_T;


typedef struct
{
  int *bytes_sent;
} CMI_VMI_Send_Handle_RDMABROAD_T;


typedef struct
{
  int                ready;
  PVMI_CONNECT       connection;
  int                destrank;
  int                maxsize;
  PVMI_REMOTE_BUFFER rbuf;
  int                bytes_sent;
  int                rdmarecvindx;
  // void *next;     needed for the "delete all persistent" function
} CMI_VMI_Send_Handle_Persistent_T;


typedef struct
{
  CMI_VMI_CmiCommHandle_T    *commhandle;
  CMI_VMI_Send_Handle_Type_T  send_handle_type;

  union
  {
    CMI_VMI_Send_Handle_Stream_T     stream;
    CMI_VMI_Send_Handle_RDMA_T       rdma;
    CMI_VMI_Send_Handle_RDMABROAD_T  rdmabroad;
    CMI_VMI_Send_Handle_Persistent_T persistent;
  } data;
} CMI_VMI_Send_Handle_T;


typedef enum
{
  CMI_VMI_RECEIVE_HANDLE_TYPE_RDMA,
  CMI_VMI_RECEIVE_HANDLE_TYPE_PERSISTENT
} CMI_VMI_Receive_Handle_Type_T;


typedef struct
{
  int               bytes_pub;
  int               rdmacnt;
  int               sindx;
  int               rindx;
  int               bytes_rec;
  PVMI_CACHE_ENTRY *cacheentry;
  VMI_virt_addr_t   rhandleaddr;
} CMI_VMI_Receive_Handle_RDMA_T;


typedef struct
{
  int               bytes_pub;
  int               rdmacnt;
  int               sindx;
  int               rindx;
  int               bytes_rec;
  PVMI_CACHE_ENTRY *cacheentry;
  VMI_virt_addr_t   rhandleaddr;
} CMI_VMI_Receive_Handle_Persistent_T;


typedef struct
{
  CMI_VMI_Receive_Handle_Type_T  receive_handle_type;

  union
  {
    CMI_VMI_Receive_Handle_RDMA_T       rdma;
    CMI_VMI_Receive_Handle_Persistent_T persistent;
  } data;
} CMI_VMI_Receive_Handle_T;


typedef struct
{
  int                    index;
  BOOLEAN                allocated;
  int                    refcount;
  char                  *msg;
  int                    msgsize;
  CMI_VMI_Handle_Type_T  handle_type;

  union
  {
    CMI_VMI_Send_Handle_T    send;
    CMI_VMI_Receive_Handle_T receive;
  } data;
} CMI_VMI_Handle_T;


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

int CMI_VMI_CRM_Register (char *key, int numProcesses, BOOLEAN reg);
BOOLEAN CMI_VMI_Open_Connections (char *synckey);
CMI_VMI_Handle_T *CMI_VMI_Allocate_Handle ();
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
