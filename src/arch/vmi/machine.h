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


#define CMI_VMI_OPTIMIZE                0
#define CMI_VMI_USE_MEMORY_POOL         1
#define CMI_VMI_CONNECTION_TIMEOUT      300
#define CMI_VMI_MAXIMUM_HANDLES         1000
#define CMI_VMI_SMALL_MESSAGE_BOUNDARY  512
#define CMI_VMI_MEDIUM_MESSAGE_BOUNDARY 4096
#define CMI_VMI_RDMA_CHUNK_COUNT        3
#define CMI_VMI_RDMA_CHUNK_SIZE         262144


#if CMI_VMI_OPTIMIZE
#define CMI_VMI_CHECK_SUCCESS(status,message)
#else
#define CMI_VMI_CHECK_SUCCESS(status,message)                             \
          if (!VMI_SUCCESS (status)) {                                    \
            VMI_perror (message, status);                                 \
          }
#endif


#define CMI_VMI_MESSAGE_TYPE(msg) ((CmiMsgHeaderBasic *)msg)->vmitype
#define CMI_VMI_MESSAGE_TYPE_STANDARD   1
#define CMI_VMI_MESSAGE_TYPE_RENDEZVOUS 2


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



#if CMI_VMI_USE_MEMORY_POOL
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






#define CMI_VMI_STARTUP_TYPE_CRM 0







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
  int                        node_IP;
  PVMI_CONNECT               connection;
  CMI_VMI_Connection_State_T state;
} CMI_VMI_Process_T;







/*
  CMI_VMI_Connect_Message_T are connect messages that are only sent during
  the connection setup phase.
*/
typedef struct
{
  int rank;
} CMI_VMI_Connect_Message_T;






/* These are send and receive handle structures. */
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
  int chunk_size;
  int bytes_sent;
  PVMI_CACHE_ENTRY cacheentry;
} CMI_VMI_Send_Handle_RDMA_T;


typedef struct
{
  int chunk_size;
  int *bytes_sent;
  PVMI_CACHE_ENTRY *cacheentry;
} CMI_VMI_Send_Handle_RDMABROAD_T;


typedef struct
{
  int                ready;
  PVMI_CONNECT       connection;
  int                destrank;
  int                maxsize;
  PVMI_REMOTE_BUFFER remote_buffer;
  int                rdma_receive_index;
  PVMI_CACHE_ENTRY   cacheentry;
} CMI_VMI_Send_Handle_Persistent_T;


typedef struct
{
  CMI_VMI_Send_Handle_Type_T  send_handle_type;
  BOOLEAN                     free_message;

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
  int              bytes_published;
  int              bytes_received;
  int              chunk_size;
  int              chunk_count;
  int              chunks_outstanding;
  int              send_index;
  int              receive_index;
  PVMI_CACHE_ENTRY cacheentry[CMI_VMI_RDMA_CHUNK_COUNT];
  VMI_virt_addr_t  remote_handle_address;
} CMI_VMI_Receive_Handle_RDMA_T;


typedef struct
{
  PVMI_CACHE_ENTRY cacheentry;
  VMI_virt_addr_t  remote_handle_address;
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
  int rdma_receive_index;
} CMI_VMI_Persistent_Grant_Message_T;


typedef struct
{
  char header[CmiMsgHeaderSizeBytes];
  int rdma_receive_index;
} CMI_VMI_Persistent_Destroy_Message_T;







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







/* Function prototypes appear here. */
int CMI_VMI_CRM_Register (char *key, int numProcesses, BOOLEAN reg);
int CMI_VMI_Startup_CRM (char *key);
VMI_CONNECT_RESPONSE
CMI_VMI_Connection_Accept_Handler (PVMI_CONNECT connection, PVMI_SLAB slab,
				   ULONG  data_size);
void CMI_VMI_Connection_Response_Handler (PVOID context, PVOID response,
					  USHORT size, PVOID handle,
					  VMI_CONNECT_RESPONSE status);
void CMI_VMI_Connection_Response_Handler (PVOID context, PVOID response,
					  USHORT size, PVOID handle,
					  VMI_CONNECT_RESPONSE status);
void CMI_VMI_Connection_Disconnect_Handler (PVMI_CONNECT connection);
int CMI_VMI_Open_Connections (char *key);

#if CMK_BROADCAST_SPANNING_TREE
int CMI_VMI_Spanning_Children_Count (char *msg);
void CMI_VMI_Send_Spanning_Children (int msgsize, char *msg);
#endif

VMI_RECV_STATUS CMI_VMI_Stream_Receive_Handler (PVMI_CONNECT connection,
						PVMI_STREAM_RECV stream,
						VMI_STREAM_COMMAND command,
						PVOID context,
						PVMI_SLAB slab);
void CMI_VMI_Stream_Completion_Handler (PVOID context, VMI_STATUS sstatus);
void CMI_VMI_RDMA_Fragment_Handler (PVMI_RDMA_OP op, PVOID context,
				    VMI_STATUS rstatus);
void CMI_VMI_RDMA_Completion_Handler (PVMI_RDMA_OP op, PVOID context,
				      VMI_STATUS rstatus);
void CMI_VMI_RDMA_Publish_Handler (PVMI_CONNECT connection,
				   PVMI_REMOTE_BUFFER remote_buffer);
void CMI_VMI_RDMA_Notification_Handler (PVMI_CONNECT connection,
					UINT32 rdma_size,
					UINT32 context,
					VMI_STATUS remote_status);

void ConverseInit (int argc, char **argv, CmiStartFn start_function,
		   int user_calls_scheduler, int init_returns);
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
int CmiAsyncMsgSent (CmiCommHandle commhandle);
void CmiReleaseCommHandle (CmiCommHandle commhandle);

#if CMK_PERSISTENT_COMM
void CmiPersistentInit ();
PersistentHandle CmiCreatePersistent (int destrank, int maxsize);
void CmiUsePersistentHandle (PersistentHandle *handle_array, int array_size);
void CmiDestroyPersistent (PersistentHandle phandle);
void CmiDestroyAllPersistent ();
void CMI_VMI_Persistent_Request_Handler (char *msg);
void CMI_VMI_Persistent_Grant_Handler (char *msg);
void CMI_VMI_Persistent_Destroy_Handler (char *msg);
#endif

#if CMK_MULTICAST_LIST_USE_SPECIAL_CODE
void CmiSyncListSendFn (int npes, int *pes, int len, char *msg);
CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int len, char *msg);
void CmiFreeListSendFn (int npes, int *pes, int msgsize, char *msg);
#endif

CMI_VMI_Handle_T *CMI_VMI_Allocate_Handle ();
void *CMI_VMI_CmiAlloc (int size);
void CMI_VMI_CmiFree (void *ptr);

BOOLEAN CRMInit ();
SOCKET createSocket(char *hostName, int port, int *localAddr);
BOOLEAN CRMRegister (char *key, ULONG numPE, int shutdownPort,
		     SOCKET *clientSock, int *clientAddr, PCRM_Msg *msg2);
BOOLEAN CRMParseMsg (PCRM_Msg msg, int rank, int *nodeIP,
		     int *shutdownPort, int *nodePE);
int CRMSend (SOCKET s, char *msg, int n);
int CRMRecv (SOCKET s, char *msg, int n);
