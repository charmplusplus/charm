/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
**
** PERSISTENT HANDLES
**    * charm/tmp/persistent.h
**    * charm/tmp/persistent.c
**    * Elan machine.c
**
** EFFICIENT DATA STRUCTURES
**    * pre-allocate arrays of send/receive/persistent handles at startup
**    * use same structure (with unions) for send/receive/persistent handles
**    * pass index as context instead of pointers
**
**    * fix persistent handles so they send data in chunks
**      (probably need a "number of chunks" field in rdma/persistent
**       send and receive handles that we either set to 1 or to
**       MAX_RDMA_CHUNKS)
**      (also, allocate MAX_RDMA_CHUNKS in each handle for cacheentries
**       instead of doing all that dynamic allocation mess we have now)
**
** IMPROVE STARTUP/SHUTDOWN
**    * startup via charmrun
**    * startup via improved CRM
**    * processes organize into a ring for improved normal/fault shutdown
**    * fix the race condition for shutdown
**    * open connections as-needed, not all at startup (optional)
**
** HETEROGENEOUS ARCHITECTURE SUPPORT
**    * use network byte order for components of msg hdr
**    * must use UINT64 instead of VMI_ADDR_CAST
**
** SHARED MEMORY SUPPORT
**
** CONVERSE VECTOR SEND ROUTINES
**    * not actually invoked by Charm++, so impact neglegible
*/

#include "machine.h"


/* The following are external variables used by the VMI core. */
extern USHORT VMI_DEVICE_RUNTIME;
extern PVMI_NETADDRESS localAddress;
extern VMIStreamRecv recvFn;


/*
  The following two globals hold this process's rank in the computation
  and the count of the total number of processes in the computation.
*/
int _Cmi_mype;
int _Cmi_numpes;

int _Cmi_myrank = 0;

/* This is the global variable for CMI_VMI_SMALL_MESSAGE_BOUNDARY. */
int CMI_VMI_Small_Message_Boundary;

/* This is the global variable for CMI_VMI_MEDIUM_MESSAGE_BOUNDARY. */
int CMI_VMI_Medium_Message_Boundary;

/* This is the global variable for CMI_VMI_RDMA_MAX_OUTSTANDING. */
int CMI_VMI_RDMA_Max_Outstanding;

/* This is the global variable for CMI_VMI_RDMA_MAX_CHUNK. */
int CMI_VMI_RDMA_Max_Chunk;


/* This is the global count of outstanding asynchronous messages. */
volatile int CMI_VMI_AsyncMsgCount;





int CMI_VMI_RDMA_Rendezvous_Handler_ID;
#if CMK_PERSISTENT_COMM
int CMI_VMI_Persistent_Request_Handler_ID;
int CMI_VMI_Persistent_Grant_Handler_ID;
int CMI_VMI_Persistent_Destroy_Handler_ID;

CMI_VMI_Handle_T *CMI_VMI_Persistent_Handles;
int CMI_VMI_Persistent_Handles_Size;
#endif   /* CMK_PERSISTENT_COMM */




/*
  The following global variables are used during connection setup.

  We keep track of the number of outgoing connections issued and the
  number of these that were accepted, rejected, or entered an error state.

  We keep track of the number of incoming connections expected and the
  number of these that were accepted, rejected, or entered an error state.
*/
volatile int CMI_VMI_OIssue;
volatile int CMI_VMI_OAccept;
volatile int CMI_VMI_OError;
volatile int CMI_VMI_OReject;

volatile int CMI_VMI_IExpect;
volatile int CMI_VMI_IAccept;
volatile int CMI_VMI_IError;
volatile int CMI_VMI_IReject;



#if CONVERSE_VERSION_VMI
PVMI_BUFFER_POOL CMI_VMI_Bucket1_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket2_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket3_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket4_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket5_Pool;
#endif



/*
  The following global variables are pointers to buffer pools used within
  the computation.
*/
PVMI_BUFFER_POOL CMI_VMI_MessageBuffer_Pool;
PVMI_BUFFER_POOL CMI_VMI_CmiCommHandle_Pool;
PVMI_BUFFER_POOL CMI_VMI_Handle_Pool;
PVMI_BUFFER_POOL CMI_VMI_RDMABytesSent_Pool;
PVMI_BUFFER_POOL CMI_VMI_RDMAPutContext_Pool;
PVMI_BUFFER_POOL CMI_VMI_RDMACacheEntry_Pool;


/* This is the global list of all processes in the computation. */
CMI_VMI_Process_Info_T *CMI_VMI_Procs;


/*
  The following global variable is used for RDMA receive contexts.
*/
CMI_VMI_RDMA_Receive_Context_T
     CMI_VMI_RDMA_Receive_Context[CMI_VMI_MAX_RECEIVE_HANDLES];


/*
  The following global variables are used to hold the queues of messages
  from the local process and from all remote processes.

  CmiLocalQueue is a name referenced within the Converse core -- it may
  not be changed.
*/
CpvDeclare (void *, CmiLocalQueue);
CpvDeclare (void *, CMI_VMI_RemoteQueue);


/* The following are for the NCSA CRM code which we currently use. */
char *CRMHost;
int CRMPort;


/*************************************************************************/
/*************************************************************************/
/******* A S Y N C H R O N O U S   H A N D L E R   R O U T I N E S *******/
/*************************************************************************/
/*************************************************************************/



/**************************************************************************
** This function is invoked asynchronously to handle an incoming connection
** request.
*/
VMI_CONNECT_RESPONSE CMI_VMI_Connection_Accept_Handler (PVMI_CONNECT incoming,
						 PVMI_SLAB slab, ULONG insize)
{
  VMI_STATUS status;

  CMI_VMI_Connect_Message_T *data;
  ULONG rank;
  ULONG size;
  PVMI_SLAB_STATE state;


  DEBUG_PRINT ("CMI_VMI_Connection_Accept_Handler() called.\n");

  /* Initialize the number of bytes expected from the connect data. */
  size = sizeof (CMI_VMI_Connect_Message_T);

  /* Make sure we received the expected number of bytes. */
  if (insize != size) {
    CMI_VMI_IError++;
    return VMI_CONNECT_RESPONSE_ERROR;
  }

  /* Allocate connection data structure. */
  data = (CMI_VMI_Connect_Message_T *) malloc (size);
  if (!data) {
    CMI_VMI_IError++;
    return VMI_CONNECT_RESPONSE_ERROR;
  }

  /* Save the slab state prior to reading. */
  status = VMI_Slab_Save_State (slab, &state);
  if (!VMI_SUCCESS (status)) {
    free (data);
    CMI_VMI_IError++;
    return VMI_CONNECT_RESPONSE_ERROR;
  }

  /* Copy connect data. */
  status = VMI_Slab_Copy_Bytes (slab, size, data);
  if (!VMI_SUCCESS (status)) {
    VMI_Slab_Restore_State (slab, state);
    free (data);
    CMI_VMI_IError++;
    return VMI_CONNECT_RESPONSE_ERROR;
  }

  /* Get rank of connecting process. */
  rank = ntohl (data->rank);

  DEBUG_PRINT ("Accepting a connection request from rank %u.\n", rank);

  /* Update the connection state. */
  (&CMI_VMI_Procs[rank])->connection = incoming;
  (&CMI_VMI_Procs[rank])->state = CMI_VMI_CONNECTION_CONNECTED;

  VMI_CONNECT_SET_RECEIVE_CONTEXT (incoming, (&CMI_VMI_Procs[rank]));

  status = VMI_RDMA_Set_Publish_Callback (incoming,
					  CMI_VMI_RDMA_Publish_Handler);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Publish_Callback()");

  status = VMI_RDMA_Set_Notification_Callback (incoming,
           CMI_VMI_RDMA_Notification_Handler);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Notification_Callback()");

  /* Free the connect data buffer. */
  free (data);

  /* Accepted the connection. */
  CMI_VMI_IAccept++;
  return VMI_CONNECT_RESPONSE_ACCEPT;
}



/**************************************************************************
** This function is invoked asynchronously to handle a process's response
** to our connection request.
*/
void CMI_VMI_Connection_Response_Handler (PVOID context, PVOID response,
                 USHORT size, PVOID handle, VMI_CONNECT_RESPONSE status)
{
  CMI_VMI_Process_Info_T *proc;


  DEBUG_PRINT ("CMI_VMI_Connection_Response_Handler() called.\n");

  /* Cast the context to a CMI_VMI_Process_Info pointer. */
  proc = (CMI_VMI_Process_Info_T *) context;

  switch (status)
  {
    case VMI_CONNECT_RESPONSE_ACCEPT:
      DEBUG_PRINT ("Process %d accepted connection.\n", proc->rank);

      /* Update the connection state. */
      proc->state = CMI_VMI_CONNECTION_CONNECTED;

      VMI_CONNECT_SET_RECEIVE_CONTEXT (proc->connection, proc);

      status = VMI_RDMA_Set_Publish_Callback (proc->connection,
					      CMI_VMI_RDMA_Publish_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Publish_Callback()");

      status = VMI_RDMA_Set_Notification_Callback (proc->connection,
	       CMI_VMI_RDMA_Notification_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Notification_Callback()");

      /* Increment the count of outgoing connection accepts. */
      CMI_VMI_OAccept++;

      break;

    case VMI_CONNECT_RESPONSE_REJECT:
      DEBUG_PRINT ("Process %d rejected connection.\n", proc->rank);

      /* Update the connection state. */
      proc->state = CMI_VMI_CONNECTION_DISCONNECTED;

      /* Increment the count of outgoing connection rejects. */
      CMI_VMI_OReject++;

      break;

    case VMI_CONNECT_RESPONSE_ERROR:
      DEBUG_PRINT ("Error connecting to process %d [%d.%d.%d.%d].\n",
		   proc->rank,
		   (proc->nodeIP >>  0) & 0xFF, (proc->nodeIP >>  8) & 0xFF,
		   (proc->nodeIP >> 16) & 0xFF, (proc->nodeIP >> 24) & 0xFF);

      /* Update the connection state. */
      proc->state = CMI_VMI_CONNECTION_ERROR;

      /* Increment the count of outgoing connection errors. */
      CMI_VMI_OError++;

      break;

    default:
      DEBUG_PRINT ("Error connecting to process %d\n", proc->rank);
      DEBUG_PRINT ("Error code 0x%08x\n", status);

      /* Update the connection state. */
      proc->state = CMI_VMI_CONNECTION_ERROR;

      /* Increment the count of outgoing connection errors. */
      CMI_VMI_OError++;

      break;
  }

  /* Deallocate the connection receive context. */
  VMI_Buffer_Deallocate ((PVMI_BUFFER) context);
}



/**************************************************************************
** This function is invoked asynchronously to handle an incoming disconnect
** request.
*/
void CMI_VMI_Connection_Disconnect_Handler (IN PVMI_CONNECT connection)
{
  CMI_VMI_Process_Info_T *proc;

  proc = (CMI_VMI_Process_Info_T *)
         VMI_CONNECT_GET_RECEIVE_CONTEXT (connection);
  proc->state = CMI_VMI_CONNECTION_DISCONNECTED;
}



/**************************************************************************
** This function is invoked asynchronously to handle a process's response
** to our disconnection request.
*/
void CMI_VMI_Disconnection_Response_Handler (PVMI_CONNECT connection,
				    PVOID context, VMI_STATUS status)
{
  CMI_VMI_Process_Info_T *proc;

  proc = (CMI_VMI_Process_Info_T *) context;
  proc->state = CMI_VMI_CONNECTION_DISCONNECTED;
}



/**************************************************************************
** This function is invoked asynchronously to handle an incoming message
** receive on a stream.
**
** This function is on the receive side.
*/
VMI_RECV_STATUS CMI_VMI_Stream_Receive_Handler (PVMI_CONNECT connection,
     PVMI_STREAM_RECV stream, VMI_STREAM_COMMAND command, PVOID context,
     PVMI_SLAB slab)
{
  VMI_STATUS status;

  PVMI_CACHE_ENTRY cacheentry;
  char *msg;
  ULONG size;
  PVMI_SLAB_STATE state;


  DEBUG_PRINT ("CMI_VMI_Stream_Receive_Handler() called.\n");

  /* Save the slab state. */
  status = VMI_Slab_Save_State (slab, &state);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Slab_Save_State()");

  size = VMI_SLAB_BYTES_REMAINING (slab);

  msg = CmiAlloc (size);

  /* Copy the message body into the message buffer. */
  status = VMI_Slab_Copy_Bytes (slab, size, msg);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Slab_Copy_Bytes()");

  /* Restore the slab state. */
  status = VMI_Slab_Restore_State (slab, state);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Slab_Restore_State()");

#if CMK_BROADCAST_SPANNING_TREE
  /* Send the message to our spanning children (if any). */
  if (CMI_BROADCAST_ROOT (msg)) {
    CMI_VMI_Send_Spanning_Children (size, msg);
  }
#endif

  /* Enqueue the message into the remote queue. */
  CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), msg);

  /* Tell VMI that the slab can be discarded. */
  return VMI_SLAB_DONE;
}



/**************************************************************************
** This function is invoked asynchronously to handle the completion of a
** send on a stream.
**
** This function is on the send side.
*/
void CMI_VMI_Stream_Completion_Handler (PVOID ctxt, VMI_STATUS sstatus)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;


  DEBUG_PRINT ("CMI_VMI_Stream_Completion_Handler() called.\n");

  /* Cast the context to a send handle. */
  handle = (CMI_VMI_Handle_T *) ctxt;

  /*
    If the handle is for some type of asynchronous operation, decrement the
    global count of outstanding asynchronous operations.
  */
  if ((handle->type == CMI_VMI_HANDLE_TYPE_ASYNC_SEND_STREAM) ||
      (handle->type == CMI_VMI_HANDLE_TYPE_ASYNC_BROADCAST_STREAM)) {
    CMI_VMI_AsyncMsgCount--;
  }

  /*
    If there is a CmiCommHandle associated with the handle (which would
    have been handed back to the caller), decrement its count.
  */
  if (handle->commhandle) {
    handle->commhandle->count--;
  }

  /*
    Decrement the handle's reference count.  If the reference count
    drops below one, deallocate the handle and its associated entries.
  */
  handle->refcount--;
  if (handle->refcount < 1) {
    /*
    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_MessageBuffer_Pool,
					 handle->data.stream.vmimsg);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
    */

    status = VMI_Cache_Deregister (handle->data.stream.cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_Handle_Pool, handle);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
  }
}



/**************************************************************************
** This function is invoked asynchronously to handle the completion of an
** RDMA Put for a fragment of a message (i.e., there is at least one more
** fragment left in the message after this one).
**
** This function is on the send side.
*/
void CMI_VMI_RDMA_Fragment_Handler (PVMI_RDMA_OP op, PVOID ctxt,
     VMI_STATUS rstatus)
{
  VMI_STATUS status;

  PVMI_CACHE_ENTRY cacheentry;


  DEBUG_PRINT ("CMI_VMI_RDMA_Fragment_Handler() called.\n");

  /* Cast the context to a cache entry. */
  cacheentry = (PVMI_CACHE_ENTRY) ctxt;

  /* Deallocate the RDMA op's buffer, the RDMA op, and the cache entry. */
  status = VMI_RDMA_Dealloc_Buffer (op->rbuffer);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Buffer()");

  status = VMI_RDMA_Dealloc_Op (op);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Op()");

  status = VMI_Cache_Deregister (cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
}



/**************************************************************************
** This function is invoked asynchronously to handle the completion of an
** RDMA Put for the last fragment of a message (i.e., there are no more
** fragments left in the message after this one).
**
** This function is on the send side.
*/
void CMI_VMI_RDMA_Completion_Handler (PVMI_RDMA_OP op, PVOID ctxt,
     VMI_STATUS rstatus)
{
  VMI_STATUS status;

  PVMI_CACHE_ENTRY cacheentry;
  CMI_VMI_Handle_T *handle;
  CMI_VMI_RDMA_Put_Context_T *rdmaputctxt;


  DEBUG_PRINT ("CMI_VMI_RDMA_Completion_Handler() called.\n");

  /* Cast the context to an RDMA Put context. */
  rdmaputctxt = (CMI_VMI_RDMA_Put_Context_T *) ctxt;

  /* Get the cache entry and handle from the context. */
  cacheentry = rdmaputctxt->cacheentry;
  handle = rdmaputctxt->handle;

  /* Deallocate the RDMA Put context. */
  status = VMI_Pool_Deallocate_Buffer (CMI_VMI_RDMAPutContext_Pool,
				       rdmaputctxt);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");

  /* Deallocate the RDMA op's buffer, the RDMA op, and the cache entry. */
  status = VMI_RDMA_Dealloc_Buffer (op->rbuffer);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Buffer()");

  status = VMI_RDMA_Dealloc_Op (op);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Op()");

  status = VMI_Cache_Deregister (cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

  /*
    If the handle is for some type of asynchronous operation, decrement the
    global count of outstanding asynchronous operations.
  */
  // TODO: We don't correctly deal with persistent asynchronous.
  if ((handle->type == CMI_VMI_HANDLE_TYPE_ASYNC_SEND_RDMA) ||
      (handle->type == CMI_VMI_HANDLE_TYPE_ASYNC_BROADCAST_RDMA)) {
    CMI_VMI_AsyncMsgCount--;
  }

  /*
    If there is a CmiCommHandle associated with the handle (which would
    have been handed back to the caller), decrement its count.
  */
  if (handle->commhandle) {
    handle->commhandle->count--;
  }

  /*
    Decrement the handle's reference count.  If the reference count
    drops below one, deallocate the handle.
  */
  handle->refcount--;
  if (handle->refcount < 1) {
    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_Handle_Pool, handle);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
  }
}



/**************************************************************************
** This function is invoked asynchronously to handle an RDMA publish of a
** buffer from a remote process.
**
** This function is on the send side.
*/
void CMI_VMI_RDMA_Publish_Handler (PVMI_CONNECT conn, PVMI_REMOTE_BUFFER rbuf)
{
  VMI_STATUS status;

  PVMI_CACHE_ENTRY cacheentry;
  BOOLEAN complete_flag;
  CMI_VMI_Handle_T *handle;
  CMI_VMI_Process_Info_T *proc;
  char *putaddr;
  VMIRDMAWriteComplete putfn;
  int putlen;
  int rank;
  PVMI_RDMA_OP rdmaop;
  CMI_VMI_RDMA_Put_Context_T *rdmaputctxt;


  DEBUG_PRINT ("CMI_VMI_RDMA_Publish_Handler() called.\n");

  /* Cast the remote buffer's local context to a send handle. */
  handle = (CMI_VMI_Handle_T *) (VMI_ADDR_CAST) rbuf->lctxt;

  /* Check to see if the handle is for an RDMA send or an RDMA broadcast. */
  if ((handle->type == CMI_VMI_HANDLE_TYPE_SYNC_SEND_RDMA) ||
      (handle->type == CMI_VMI_HANDLE_TYPE_ASYNC_SEND_RDMA)) {

    putaddr = handle->msg + handle->data.rdma.bytes_sent;
    putlen = handle->msgsize - handle->data.rdma.bytes_sent;

    if (putlen > CMI_VMI_RDMA_Max_Chunk) {
      putlen = CMI_VMI_RDMA_Max_Chunk;
      complete_flag = FALSE;
    } else {
      complete_flag = TRUE;
    }

    status = VMI_Cache_Register (putaddr, putlen, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    status = VMI_RDMA_Alloc_Op (&rdmaop);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

    rdmaop->numBufs = 1;
    rdmaop->buffers[0] = cacheentry->bufferHandle;
    rdmaop->addr[0] = putaddr;
    rdmaop->sz[0] = putlen;
    rdmaop->rbuffer = rbuf;
    rdmaop->roffset = 0;

    handle->data.rdma.bytes_sent += putlen;

    if (complete_flag) {
      status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMAPutContext_Pool,
					 (PVOID) &rdmaputctxt, NULL);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

      rdmaputctxt->cacheentry = cacheentry;
      rdmaputctxt->handle = handle;

      status = VMI_RDMA_Put (conn, rdmaop, (PVOID) rdmaputctxt,
	   (VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    } else {
      status = VMI_RDMA_Put (conn, rdmaop, (PVOID) cacheentry, 
	   (VMIRDMAWriteComplete) CMI_VMI_RDMA_Fragment_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    }
#if CMK_PERSISTENT_COMM
  } else if ((handle->type == CMI_VMI_HANDLE_TYPE_SYNC_BROADCAST_RDMA) ||
             (handle->type == CMI_VMI_HANDLE_TYPE_ASYNC_BROADCAST_RDMA)) {
#else   /* CMK_PERSISTENT_COMM */
  } else {
#endif
    proc = (CMI_VMI_Process_Info_T *) VMI_CONNECT_GET_RECEIVE_CONTEXT (conn);
    rank = proc->rank;

    putaddr = handle->msg + handle->data.rdmabroad.bytes_sent[rank];
    putlen = handle->msgsize - handle->data.rdmabroad.bytes_sent[rank];

    if (putlen > CMI_VMI_RDMA_Max_Chunk) {
      putlen = CMI_VMI_RDMA_Max_Chunk;
      complete_flag = FALSE;
    } else {
      complete_flag = TRUE;
    }

    status = VMI_Cache_Register (putaddr, putlen, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    status = VMI_RDMA_Alloc_Op (&rdmaop);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

    rdmaop->numBufs = 1;
    rdmaop->buffers[0] = cacheentry->bufferHandle;
    rdmaop->addr[0] = putaddr;
    rdmaop->sz[0] = putlen;
    rdmaop->rbuffer = rbuf;
    rdmaop->roffset = 0;

    handle->data.rdmabroad.bytes_sent[rank] += putlen;

    if (complete_flag) {
      status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMAPutContext_Pool,
					 (PVOID) &rdmaputctxt, NULL);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

      rdmaputctxt->cacheentry = cacheentry;
      rdmaputctxt->handle = handle;

      status = VMI_RDMA_Put (conn, rdmaop, (PVOID) rdmaputctxt,
	   (VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    } else {
      status = VMI_RDMA_Put (conn, rdmaop, (PVOID) cacheentry, 
	   (VMIRDMAWriteComplete) CMI_VMI_RDMA_Fragment_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    }
  }
#if CMK_PERSISTENT_COMM
  else {
    handle->data.persistent.rbuf = rbuf;
    handle->data.persistent.ready++;
  }
#endif   /* CMK_PERSISTENT_COMM */
}


/**************************************************************************
** This function is invoked asynchronously to handle the completion of an
** RDMA Put from a remote process.
**
** This function is on the receive side.
*/
void CMI_VMI_RDMA_Notification_Handler (PVMI_CONNECT conn, UINT32 rdmasz,
     UINT32 context, VMI_STATUS rstatus)
{
  VMI_STATUS status;

  CMI_VMI_RDMA_Receive_Context_T *rdmarecvctxt;

  char *pubaddr;
  int pubsz;

  PVMI_CACHE_ENTRY cacheentry;

  char *msg2;
  int pubsize;


  DEBUG_PRINT ("CMI_VMI_RDMA_Notification_Handler() called.\n");

  rdmarecvctxt = &(CMI_VMI_RDMA_Receive_Context[context]);

  rdmarecvctxt->bytes_rec += rdmasz;
  rdmarecvctxt->rdmacnt--;

  cacheentry = rdmarecvctxt->cacheentry[rdmarecvctxt->rindx];
  status = VMI_Cache_Deregister (cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

  rdmarecvctxt->rindx++;
  if (rdmarecvctxt->rindx >= CMI_VMI_RDMA_Max_Outstanding) {
    rdmarecvctxt->rindx = 0;
  }

  if (rdmarecvctxt->bytes_pub < rdmarecvctxt->msgsize) {
    pubaddr = rdmarecvctxt->msg + rdmarecvctxt->bytes_pub;
    pubsz = rdmarecvctxt->msgsize - rdmarecvctxt->bytes_pub;
    if (pubsz > CMI_VMI_RDMA_Max_Chunk) {
      pubsz = CMI_VMI_RDMA_Max_Chunk;
    }

    status = VMI_Cache_Register (pubaddr, pubsz, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    rdmarecvctxt->cacheentry[rdmarecvctxt->sindx] = cacheentry;

    rdmarecvctxt->sindx++;
    if (rdmarecvctxt->sindx >= CMI_VMI_RDMA_Max_Outstanding) {
      rdmarecvctxt->sindx = 0;
    }

    rdmarecvctxt->rdmacnt++;

    status = VMI_RDMA_Publish_Buffer (conn, cacheentry->bufferHandle,
         (VMI_virt_addr_t) (VMI_ADDR_CAST) pubaddr, pubsz,
         (VMI_virt_addr_t) rdmarecvctxt->rhandleaddr, context);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");

    rdmarecvctxt->bytes_pub += pubsz;
  }

  if (rdmarecvctxt->bytes_rec >= rdmarecvctxt->msgsize) {
#if CMK_BROADCAST_SPANNING_TREE
    if (CMI_BROADCAST_ROOT (rdmarecvctxt->msg)) {
      CMI_VMI_Send_Spanning_Children (rdmarecvctxt->msgsize,
				      rdmarecvctxt->msg);
    }
#endif

    CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), rdmarecvctxt->msg);

    if (rdmarecvctxt->persistent_flag) {
      msg2 = (char *) CmiAlloc (rdmarecvctxt->msgsize);

      rdmarecvctxt->msg = msg2;
      rdmarecvctxt->bytes_rec = 0;
      rdmarecvctxt->rdmacnt = 1;
      rdmarecvctxt->sindx = 0;
      rdmarecvctxt->rindx = 0;

      pubaddr = rdmarecvctxt->msg;
      pubsize = rdmarecvctxt->msgsize;

      status = VMI_Cache_Register (pubaddr, pubsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

      rdmarecvctxt->cacheentry[rdmarecvctxt->sindx] = cacheentry;

      status = VMI_RDMA_Publish_Buffer (conn,
	   cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) pubaddr,
	   pubsize, rdmarecvctxt->rhandleaddr,
	   (UINT32) rdmarecvctxt->index);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    } else {
      status = VMI_Pool_Deallocate_Buffer (CMI_VMI_RDMACacheEntry_Pool,
					   rdmarecvctxt->cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");

      (&(CMI_VMI_RDMA_Receive_Context[context]))->allocated = FALSE;
    }
  }
}




/**************************************************************************
** This is a Converse handler.
*/
void CMI_VMI_RDMA_Rendezvous_Handler (char *msg)
{
  VMI_STATUS status;

  int rank;
  int msgsize;
  VMI_virt_addr_t rhandleaddr;
  int rdmarecvindx;
  CMI_VMI_RDMA_Receive_Context_T *rdmarecvctxt;
  char *pubaddr;
  int pubsize;
  PVMI_CACHE_ENTRY cacheentry;
  char *msg2;


  rank = ((CMI_VMI_Rendezvous_Message_T *) msg)->rank;
  msgsize = ((CMI_VMI_Rendezvous_Message_T *) msg)->msgsize;
  rhandleaddr = ((CMI_VMI_Rendezvous_Message_T *) msg)->context;

  CmiFree (msg);

  rdmarecvindx = CMI_VMI_Get_RDMA_Receive_Context();
  rdmarecvctxt = &(CMI_VMI_RDMA_Receive_Context[rdmarecvindx]);

  msg2 = (char *) CmiAlloc (msgsize);

  rdmarecvctxt->rhandleaddr = rhandleaddr;
  rdmarecvctxt->msg = msg2;
  rdmarecvctxt->msgsize = msgsize;
  rdmarecvctxt->bytes_pub = 0;
  rdmarecvctxt->bytes_rec = 0;
  rdmarecvctxt->rdmacnt = 0;
  rdmarecvctxt->sindx = 0;
  rdmarecvctxt->rindx = 0;
  rdmarecvctxt->persistent_flag = FALSE;

  status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMACacheEntry_Pool,
       (PVOID) &(rdmarecvctxt->cacheentry), NULL);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

  while ((rdmarecvctxt->bytes_pub < rdmarecvctxt->msgsize) &&
	 (rdmarecvctxt->rdmacnt < CMI_VMI_RDMA_Max_Outstanding)) {

    pubaddr = rdmarecvctxt->msg + rdmarecvctxt->bytes_pub;
    pubsize = rdmarecvctxt->msgsize - rdmarecvctxt->bytes_pub;
    if (pubsize > CMI_VMI_RDMA_Max_Chunk) {
      pubsize = CMI_VMI_RDMA_Max_Chunk;
    }

    status = VMI_Cache_Register (pubaddr, pubsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    rdmarecvctxt->cacheentry[rdmarecvctxt->sindx] = cacheentry;

    rdmarecvctxt->sindx++;
    if (rdmarecvctxt->sindx >= CMI_VMI_RDMA_Max_Outstanding) {
      rdmarecvctxt->sindx = 0;
    }

    rdmarecvctxt->rdmacnt++;

    status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Procs[rank])->connection,
         cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) pubaddr,
         pubsize, rhandleaddr, (UINT32) rdmarecvindx);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");

    rdmarecvctxt->bytes_pub += pubsize;
  }
}


#if CMK_PERSISTENT_COMM
/**************************************************************************
** This is a Converse handler.
*/
void CMI_VMI_Persistent_Request_Handler (char *msg)
{
  VMI_STATUS status;

  char *msg2;

  int rank;
  int maxsize;
  VMI_virt_addr_t rhandleaddr;
  int rdmarecvindx;
  CMI_VMI_RDMA_Receive_Context_T *rdmarecvctxt;
  char *pubaddr;
  int pubsize;
  PVMI_CACHE_ENTRY cacheentry;

  CMI_VMI_Persistent_Grant_Message_T grant_msg;

  PVOID addrs[1];
  ULONG sz[1];


  rank = ((CMI_VMI_Persistent_Request_Message_T *) msg)->rank;
  maxsize = ((CMI_VMI_Persistent_Request_Message_T *) msg)->maxsize;
  rhandleaddr = ((CMI_VMI_Persistent_Request_Message_T *) msg)->context;

  CmiFree (msg);

  rdmarecvindx = CMI_VMI_Get_RDMA_Receive_Context();
  rdmarecvctxt = &(CMI_VMI_RDMA_Receive_Context[rdmarecvindx]);

  msg2 = (char *) CmiAlloc (maxsize);

  rdmarecvctxt->rhandleaddr = rhandleaddr;
  rdmarecvctxt->msg = msg2;
  rdmarecvctxt->msgsize = maxsize;
  rdmarecvctxt->bytes_pub = maxsize;
  rdmarecvctxt->bytes_rec = 0;
  rdmarecvctxt->rdmacnt = 1;
  rdmarecvctxt->sindx = 0;
  rdmarecvctxt->rindx = 0;
  rdmarecvctxt->persistent_flag = TRUE;

  status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMACacheEntry_Pool,
       (PVOID) &(rdmarecvctxt->cacheentry), NULL);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

  pubaddr = rdmarecvctxt->msg;
  pubsize = rdmarecvctxt->msgsize;

  status = VMI_Cache_Register (pubaddr, pubsize, &cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

  rdmarecvctxt->cacheentry[rdmarecvctxt->sindx] = cacheentry;

  status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Procs[rank])->connection,
       cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) pubaddr,
       pubsize, rhandleaddr, (UINT32) rdmarecvindx);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");

  CmiSetHandler (&grant_msg, CMI_VMI_Persistent_Grant_Handler_ID);
  grant_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) rhandleaddr;
  grant_msg.rdmarecvindx = rdmarecvindx;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (&grant_msg, 0);
#endif

  addrs[0] = (PVOID) &grant_msg;
  sz[0] = (ULONG) (sizeof (CMI_VMI_Persistent_Grant_Message_T));

  status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[rank])->connection,
       addrs, sz, 1, sizeof (CMI_VMI_Persistent_Grant_Message_T));
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
}


/**************************************************************************
** This is a Converse handler.
*/
void CMI_VMI_Persistent_Grant_Handler (char *msg)
{
  CMI_VMI_Handle_T *handle;


  handle = (CMI_VMI_Handle_T *) (VMI_ADDR_CAST)
            ((CMI_VMI_Persistent_Grant_Message_T *) msg)->context;
  handle->data.persistent.rdmarecvindx =
            ((CMI_VMI_Persistent_Grant_Message_T *) msg)->rdmarecvindx;

  CmiFree (msg);

  handle->data.persistent.ready++;
}


/**************************************************************************
** This is a Converse handler.
*/
void CMI_VMI_Persistent_Destroy_Handler (char *msg)
{
  VMI_STATUS status;
  int rdmarecvindx;
  CMI_VMI_RDMA_Receive_Context_T *rdmarecvctxt;


  rdmarecvindx = ((CMI_VMI_Persistent_Destroy_Message_T *) msg)->rdmarecvindx;

  CmiFree (msg);

  rdmarecvctxt = &(CMI_VMI_RDMA_Receive_Context[rdmarecvindx]);

  CmiFree (rdmarecvctxt->msg);

  status = VMI_Pool_Deallocate_Buffer (CMI_VMI_RDMACacheEntry_Pool,
				       rdmarecvctxt->cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");

  (&(CMI_VMI_RDMA_Receive_Context[rdmarecvindx]))->allocated = FALSE;
}
#endif   /* CMK_PERSISTENT_COMM */



/*************************************************************************/
/*************************************************************************/
/***** I N T E R N A L   M A C H I N E   L A Y E R   R O U T I N E S *****/
/*************************************************************************/
/*************************************************************************/


/**************************************************************************
** Use the CRM to synchronize with all other processes in the computation.
** This function may be used to register with the CRM to obtain rank and
** machine address information for all processes in the computation (if the
** caller supplies TRUE for the "reg" argument) or to simply use the CRM to
** barrier for some set of processes (if the caller supplies FALSE for the
** "reg" argument).  If registration is requested, the global variable
** CMI_VMI_ProcessList is updated with information about all processes that
** registered.
**
** Return values:
**    [integer] - rank of this process in the computation
**       -1     - barrier only requested; barrier was successful
**       -2     - error
*/
int CMI_VMI_CRM_Register (PUCHAR key, int numProcesses, BOOLEAN reg)
{
  SOCKET clientSock;
  int i;
  PCRM_Msg msg;
  int myIP;
  pid_t myPID;
  int myRank;
  int nodeIP;
  pid_t processPID;
  int processRank;


  /* Get our process ID. */
  myPID = getpid ();

  /*
    Register with the CRM.

    Synchronization requires the following information:
      - synchronization key (common for all processes in computation)
      - total number of processes in computation
      - shutdown port (unique per machine - sent to all other processes)

    Synchronization returns the following information:
      - a socket for future interaction with the CRM
      - the IP address of this process's machine
      - an array of CRM response messages (which can be parsed later)
  */
  if (!CRMRegister (key, numProcesses, myPID, &clientSock, &myIP, &msg)) {
    return (-2);
  }

  /* Close the socket to the CRM. */
  close (clientSock);

  /*
    Parse the CRM response messages if this is a registration request.

    Parsing requires the following information:
      - an array of CRM response messages
      - the index into the array of response messages to get info for

    Parsing returns the following information:
      - IP address of the process's machine
      - shutdown port of the process
      - rank of the process
  */
  myRank = -1;
  if (reg) {
    for (i = 0; i < msg->msg.CRMCtx.np; i++) {
      CRMParseMsg (msg, i, &nodeIP, &processPID, &processRank);

      /* Check for the current process in the list to obtain our rank. */
      if ((nodeIP == myIP) && (processPID == myPID)) {
	myRank = processRank;
      }

      /* Cache remote nodes IP address. */
      (&CMI_VMI_Procs[i])->nodeIP = nodeIP;
      (&CMI_VMI_Procs[i])->rank   = processRank;
    }
  }

  /* Free the message from CRM. */
  free (msg->msg.CRMCtx.node);
  free (msg);

  /* Synchronized successfully with the CRM. */
  return myRank;
}



/**************************************************************************
** Set up connections between all processes.
**
** Issue connection requests to all processes with a rank lower than our
** rank and wait for connection requests from processes with a rank higher
** than our rank.
**
** Each connection request contains connect data containing the rank of
** the process attempting the connection.  That way the current process
** can store the connection information in the correct slot in the
** CMI_VMI_ProcessList[].
*/
BOOLEAN CMI_VMI_Open_Connections (PUCHAR synckey)
{
  VMI_STATUS status;

  int i;

  char *uname;
  char *remotekey;

  PVMI_BUFFER connmsgbuf;
  CMI_VMI_Connect_Message_T *connmsgdata;
  CMI_VMI_Process_Info_T *proc;
  struct hostent *rhostinfo;
  PVMI_NETADDRESS raddr;

  struct timeval tp;
  long starttime;
  long nowtime;


  /*
    **********
    * Step 1 *   Set up data structures.
    **********
  */

  /* Get our username. */
  uname = getpwuid (getuid ())->pw_name;
  if (!uname) {
    DEBUG_PRINT ("Unable to get username.\n");
    return FALSE;
  }

  /* Allocate space for the remote key. */
  remotekey = malloc (strlen (synckey) + 32);
  if (!remotekey) {
    DEBUG_PRINT ("Unable to allocate memory for remote key.\n");
    return FALSE;
  }

  /* Allocate a buffer for connection message. */
  status = VMI_Buffer_Allocate (sizeof(CMI_VMI_Connect_Message_T),&connmsgbuf);
  if (!VMI_SUCCESS (status)) {
    DEBUG_PRINT ("Unable to allocate connection message buffer.\n");
    free (remotekey);
    return FALSE;
  }

  /* Set up the connection message field. */
  connmsgdata = (CMI_VMI_Connect_Message_T *) VMI_BUFFER_ADDRESS (connmsgbuf);
  connmsgdata->rank = htonl (_Cmi_mype);

  CMI_VMI_OAccept = 0;
  CMI_VMI_OReject = 0;
  CMI_VMI_OError  = 0;
  CMI_VMI_OIssue  = 0;

  CMI_VMI_IAccept  = 0;
  CMI_VMI_IReject  = 0;
  CMI_VMI_IError   = 0;
  CMI_VMI_IExpect  = 0;

  /*
    **********
    * Step 2 *   Initiate connections.
    **********

    Here we initiate a connection to each process with a rank lower than
    this process's rank.
  */
  for (i = 0; i < _Cmi_mype; i++) {
    /* Get a pointer to the process to make things easier. */
    proc = &CMI_VMI_Procs[i];

    /* Allocate a connection object */
    status = VMI_Connection_Create (&(proc->connection));
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("Unable to create connection object for process %d\n", i);
      free (remotekey);
      VMI_Buffer_Deallocate (connmsgbuf);
      return FALSE;
    }

    /* Build the remote IPV4 address. We need remote hosts name for this. */
    rhostinfo = gethostbyaddr ((const char *) &proc->nodeIP,
         sizeof (proc->nodeIP), AF_INET);
    if (!rhostinfo) {
      DEBUG_PRINT ("Error looking up host [%d.%d.%d.%d].\n",
           (proc->nodeIP >>  0) & 0xFF, (proc->nodeIP >>  8) & 0xFF,
	   (proc->nodeIP >> 16) & 0xFF, (proc->nodeIP >> 24) & 0xFF);
      free (remotekey);
      VMI_Buffer_Deallocate (connmsgbuf);
      return FALSE;
    }

    /* Construct a remote VMI key in terms of our progKey and peer's rank */
    sprintf (remotekey, "%s:%u\0", synckey, proc->rank);

    /* Allocate a remote IPv4 NETADDRESS. */
    status = VMI_Connection_Allocate_IPV4_Address (rhostinfo->h_name, 0,
         uname, remotekey, &raddr);
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("Unable to allocate remote node IPV4 address.\n");
      DEBUG_PRINT ("Error 0x%08x.\n", status);
      free (remotekey);
      VMI_Buffer_Deallocate (connmsgbuf);
      return FALSE;
    }

    /* Now bind the local and remote addresses. */
    status = VMI_Connection_Bind (*localAddress, *raddr, proc->connection);
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("Error binding connection for node %d.\n", i);
      free (remotekey);
      VMI_Buffer_Deallocate (connmsgbuf);
      return FALSE;
    }

    // Do this here to avoid a race condition where we complete the connect
    // right away and then set the state here to
    // CMI_VMI_CONNECTION_STATE_CONNECTING.
    proc->state = CMI_VMI_CONNECTION_CONNECTING;

    /* Issue the actual connection request. */
    status = VMI_Connection_Issue (proc->connection, connmsgbuf,
         (VMIConnectIssue) CMI_VMI_Connection_Response_Handler, proc);
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("Error issuing connection for process %d.\n", i);
      free (remotekey);
      VMI_Buffer_Deallocate (connmsgbuf);
      return FALSE;
    }

    /* Increment number of issued connections. */
    CMI_VMI_OIssue++;

    DEBUG_PRINT ("Issued a connection to process %d:\n", i);
    DEBUG_PRINT ("\tRank - %d\n", proc->rank);
    DEBUG_PRINT ("\tIP - [%d.%d.%d.%d].\n",
		 (proc->nodeIP >>  0) & 0xFF, (proc->nodeIP >>  8) & 0xFF,
		 (proc->nodeIP >> 16) & 0xFF, (proc->nodeIP >> 24) & 0xFF);
    DEBUG_PRINT ("\tHostname - %s\n", rhostinfo->h_name);
    DEBUG_PRINT ("\tKey - %s\n", remotekey);
  }

  /* Set the connection state to ourself to "connected". */
  (&CMI_VMI_Procs[_Cmi_mype])->state = CMI_VMI_CONNECTION_CONNECTED;

  /*
    **********
    * Step 3 *   Wait for connections.
    **********

    Now wait for all outgoing connections to complete and for all
    incoming connections to arrive.
  */

  /* Calculate how many pprocesses are supposed to connect to us. */
  CMI_VMI_IExpect = ((_Cmi_numpes - _Cmi_mype) - 1);

  DEBUG_PRINT ("This process's rank is %d.\n", _Cmi_mype);
  DEBUG_PRINT ("Issued %d connection requests.\n", CMI_VMI_OIssue);
  DEBUG_PRINT ("Expecting %d connections to arrive.\n", CMI_VMI_IExpect);

  /* Complete all connection requests and accepts. */
  gettimeofday (&tp, NULL);
  starttime = tp.tv_sec;
  nowtime   = tp.tv_sec;
  while( (((CMI_VMI_OAccept+CMI_VMI_OReject+CMI_VMI_OError)<CMI_VMI_OIssue) ||
	   ((CMI_VMI_IAccept+CMI_VMI_IReject+CMI_VMI_IError)<CMI_VMI_IExpect))
         &&
	 ((starttime + CMI_VMI_CONNECTION_TIMEOUT) > nowtime)) {
    sched_yield ();
    status = VMI_Poll ();
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("VMI_Poll() failed while waiting for connections.\n");
      DEBUG_PRINT ("Error 0x%08x\n", status);
      return FALSE;
    }
    gettimeofday (&tp, NULL);
    nowtime = tp.tv_sec;
  }

  /*
    **********
    * Step 4 *   Verify that there were no connection problems.
    **********
  */
  if (  (CMI_VMI_OReject > 0) || (CMI_VMI_OError > 0)
     || (CMI_VMI_IReject > 0) || (CMI_VMI_IError > 0)  ) {
    DEBUG_PRINT ("%d outgoing connections were rejected.\n",CMI_VMI_OReject);
    DEBUG_PRINT ("%d outgoing connections had errors.\n", CMI_VMI_OError);
    DEBUG_PRINT ("%d incoming connections were rejected.\n", CMI_VMI_IReject);
    DEBUG_PRINT ("%d incoming connections had errors.\n", CMI_VMI_IError);

    free (remotekey);
    VMI_Buffer_Deallocate (connmsgbuf);

    return FALSE;
  }

  DEBUG_PRINT ("All connections complete for process %d.\n", _Cmi_mype);

  free (remotekey);
  VMI_Buffer_Deallocate (connmsgbuf);

  /* Successfully setup connections. */
  return TRUE;
}



/**************************************************************************
**
*/
int CMI_VMI_Get_RDMA_Receive_Context ()
{
  VMI_STATUS status;

  int i;


  i = 0;
  while ((&(CMI_VMI_RDMA_Receive_Context[i]))->allocated) {
    i++;

    if (i >= CMI_VMI_MAX_RECEIVE_HANDLES) {
      i = 0;

      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }      
  }

  (&(CMI_VMI_RDMA_Receive_Context[i]))->allocated = TRUE;

  return (i);
}



#if CONVERSE_VERSION_VMI
/**************************************************************************
**
*/
void *CMI_VMI_CmiAlloc (int size)
{
  VMI_STATUS status;

  void *ptr;


  if (size < CMI_VMI_BUCKET1_SIZE) {
    status = VMI_Pool_Allocate_Buffer (CMI_VMI_Bucket1_Pool, &ptr, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");
  } else if (size < CMI_VMI_BUCKET2_SIZE) {
    status = VMI_Pool_Allocate_Buffer (CMI_VMI_Bucket2_Pool, &ptr, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");
  } else if (size < CMI_VMI_BUCKET3_SIZE) {
    status = VMI_Pool_Allocate_Buffer (CMI_VMI_Bucket3_Pool, &ptr, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");
  } else if (size < CMI_VMI_BUCKET4_SIZE) {
    status = VMI_Pool_Allocate_Buffer (CMI_VMI_Bucket4_Pool, &ptr, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");
  } else if (size < CMI_VMI_BUCKET5_SIZE) {
    status = VMI_Pool_Allocate_Buffer (CMI_VMI_Bucket5_Pool, &ptr, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");
  } else {
    ptr = malloc (size);
  }

  return (ptr);
}


/**************************************************************************
**
*/
void CMI_VMI_CmiFree (void *ptr)
{
  VMI_STATUS status;

  int size;


  size = ((int *) ptr)[0];

  if (size < CMI_VMI_BUCKET1_SIZE) {
    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_Bucket1_Pool, ptr);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
  } else if (size < CMI_VMI_BUCKET2_SIZE) {
    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_Bucket2_Pool, ptr);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
  } else if (size < CMI_VMI_BUCKET3_SIZE) {
    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_Bucket3_Pool, ptr);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
  } else if (size < CMI_VMI_BUCKET4_SIZE) {
    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_Bucket4_Pool, ptr);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
  } else if (size < CMI_VMI_BUCKET5_SIZE) {
    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_Bucket5_Pool, ptr);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
  } else {
    free (ptr);
  }
}
#endif



#if CMK_BROADCAST_SPANNING_TREE
/**************************************************************************
** This function returns the count of the number of children of the current
** process in the spanning tree for a given message.
*/
int CMI_VMI_Spanning_Children_Count (char *msg)
{
  int startrank;
  int destrank;
  int childcount;

  int i;


  childcount = 0;

  startrank = CMI_BROADCAST_ROOT (msg) - 1;
  for (i = 1; i <= CMI_VMI_BROADCAST_SPANNING_FACTOR; i++) {
    destrank = _Cmi_mype - startrank;

    if (destrank < 0) {
      destrank += _Cmi_numpes;
    }

    destrank = CMI_VMI_BROADCAST_SPANNING_FACTOR * destrank + i;

    if (destrank > (_Cmi_numpes - 1)) {
      break;
    }

    destrank += startrank;
    destrank %= _Cmi_numpes;

    childcount++;
  }

  return (childcount);
}



/**************************************************************************
**
*/
void CMI_VMI_Send_Spanning_Children (int msgsize, char *msg)
{
  VMI_STATUS status;

  PVMI_BUFFER bufHandles[1];
  PVOID addrs[1];
  ULONG sz[1];

  CMI_VMI_Handle_T handle;

  int i;

  PVMI_CACHE_ENTRY cacheentry;

  int childcount;
  int startrank;
  int destrank;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;


  DEBUG_PRINT ("CMI_VMI_Send_Spanning_Children() called.\n");

  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle.msg = msg;
    handle.msgsize = msgsize;
    handle.commhandle = NULL;
    handle.type = CMI_VMI_HANDLE_TYPE_SYNC_BROADCAST_STREAM;
    handle.data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    // If childcount is 0 here, we could exit immediately.

    handle.refcount = childcount + 1;

    startrank = CMI_BROADCAST_ROOT (msg) - 1;
    for (i = 1; i <= CMI_VMI_BROADCAST_SPANNING_FACTOR; i++) {
      destrank = _Cmi_mype - startrank;

      if (destrank < 0) {
	destrank += _Cmi_numpes;
      }

      destrank = CMI_VMI_BROADCAST_SPANNING_FACTOR * destrank + i;

      if (destrank > (_Cmi_numpes - 1)) {
	break;
      }

      destrank += startrank;
      destrank %= _Cmi_numpes;

      status = VMI_Stream_Send ((&CMI_VMI_Procs[destrank])->connection,
           bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler,
           (PVOID) &handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    while (handle.refcount > 1) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    status = VMI_Cache_Deregister (cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
  } else {
    handle.msg = msg;
    handle.msgsize = msgsize;
    handle.commhandle = NULL;
    handle.type = CMI_VMI_HANDLE_TYPE_SYNC_BROADCAST_RDMA;

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMABytesSent_Pool,
	 (PVOID) &(handle.data.rdmabroad.bytes_sent), NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    CmiSetHandler ((int *) &rendezvous_msg,
		   CMI_VMI_RDMA_Rendezvous_Handler_ID);
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) &handle;

    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    // If childcount is 0 here, we could exit immediately.

    handle.refcount = childcount + 1;

    startrank = CMI_BROADCAST_ROOT (msg) - 1;
    for (i = 1; i <= CMI_VMI_BROADCAST_SPANNING_FACTOR; i++) {
      destrank = _Cmi_mype - startrank;

      if (destrank < 0) {
	destrank += _Cmi_numpes;
      }

      destrank = CMI_VMI_BROADCAST_SPANNING_FACTOR * destrank + i;

      if (destrank > (_Cmi_numpes - 1)) {
	break;
      }

      destrank += startrank;
      destrank %= _Cmi_numpes;

      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[destrank])->connection,
	   addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    while (handle.refcount > 1) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_RDMABytesSent_Pool,
					 handle.data.rdmabroad.bytes_sent);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
  }
}
#endif   /* CMK_BROADCAST_SPANNING_TREE */



/*************************************************************************/
/*************************************************************************/
/***** E X T E R N A L   M A C H I N E   L A Y E R   R O U T I N E S *****/
/*************************************************************************/
/*************************************************************************/



/**************************************************************************
**
** argc
** argv
** startFn - the user-supplied function to run (function pointer)
** userCallsScheduler - boolean for whether ConverseInit() needs to invoke
**                      scheduler or whether user code will do this
** initReturns - boolean for whether ConverseInit() returns
*/
void ConverseInit (int argc, char **argv, CmiStartFn startFn,
		   int userCallsScheduler, int initReturns)
{
  VMI_STATUS status;

  char *a;
  int i;

  char *synckey;
  char *vmikey;
  char *initkey;

  char *vmiinlinesize;

  CMI_VMI_Process_Info_T *proc;


  DEBUG_PRINT ("ConverseInit() called.\n");

  /* Initialize the global asynchronous message count. */
  CMI_VMI_AsyncMsgCount = 0;

  /*
    **********
    * STEP 1 *   Synchronize with all processes.
    **********

    In this step, we use the CRM to synchronize with all processes.
    The synchronization key used for this step is taken from the VMI_KEY
    environment variable or from argv[0] if VMI_KEY is not defined.
  */

  /* Read VMI_PROCS environment variable for number of nodes to expect. */
  a = getenv ("VMI_PROCS");
  if (!a) {
    CmiAbort ("Environment variable VMI_PROCS not set.");
  }
  _Cmi_numpes = atoi (a);

  CMI_VMI_Procs = (CMI_VMI_Process_Info_T *)
       malloc (_Cmi_numpes * sizeof (CMI_VMI_Process_Info_T));
  for (i = 0; i < _Cmi_numpes; i++) {
    (&CMI_VMI_Procs[i])->state = CMI_VMI_CONNECTION_DISCONNECTED;
  }

  /* Initialize the CRM. */
  if (!CRMInit ()) {
    CmiAbort ("Failed to initialize CRM.");
  }

  /* Set up the synchronization key for initial interaction with CRM. */
  a = getenv ("VMI_KEY");
  if (a) {
    synckey = (char *) strdup (a);
  }
  else {
    synckey = malloc (strlen (argv[0]) + 1);
    if (!synckey) {
      CmiAbort ("Unable to allocate memory for syncronization key.");
    }

    sprintf (synckey, "%s\0", argv[0]);
  }

  DEBUG_PRINT ("The initial synchronization key is %s.\n", synckey);

  /* Register with the CRM. */
  if ((_Cmi_mype = CMI_VMI_CRM_Register (synckey, _Cmi_numpes, TRUE)) < 0) {
    CmiAbort ("Unable to synchronize with the CRM.");
  }

  DEBUG_PRINT ("This process's rank is %d of %d total processes.\n",
	       _Cmi_mype, _Cmi_numpes);

  /*
    **********
    * STEP 2 *   Initialize VMI.
    **********

    In this step, we initialize VMI.  At this point we know that all
    processes for the computation are present.  We need a unique VMI key
    for each process, so we use "[syncronization key]:[process rank]" for
    each processes's key.
  */

  /*
    Set the very short message boundary.

    Messages of less than CMI_VMI_Small_Message_Boundary bytes will be
    sent asynchronously regardless of how they were requested to be sent.
  */
  a = getenv ("CMI_VMI_SMALL_MESSAGE_BOUNDARY");
  if (a) {
    CMI_VMI_Small_Message_Boundary = atoi (a);
  } else {
    CMI_VMI_Small_Message_Boundary = CMI_VMI_SMALL_MESSAGE_BOUNDARY;
  }

  /*
    Set the short message boundary.

    Messages of greater than CMI_VMI_Medium_Message_Boundary bytes will be
    sent via RDMA instead of stream.
  */
  a = getenv ("CMI_VMI_MEDIUM_MESSAGE_BOUNDARY");
  if (a) {
    CMI_VMI_Medium_Message_Boundary = atoi (a);
  } else {
    CMI_VMI_Medium_Message_Boundary = CMI_VMI_MEDIUM_MESSAGE_BOUNDARY;
  }

  a = getenv ("CMI_VMI_RDMA_MAX_OUTSTANDING");
  if (a) {
    CMI_VMI_RDMA_Max_Outstanding = atoi (a);
  } else {
    CMI_VMI_RDMA_Max_Outstanding = CMI_VMI_RDMA_MAX_OUTSTANDING;
  }

  a = getenv ("CMI_VMI_RDMA_MAX_CHUNK");
  if (a) {
    CMI_VMI_RDMA_Max_Chunk = atoi (a);
  } else {
    CMI_VMI_RDMA_Max_Chunk = CMI_VMI_RDMA_MAX_CHUNK;
  }

  /* Set the VMI_KEY environment variable. */
  vmikey = (char *) malloc (strlen (synckey) + 32);
  if (!vmikey) {
    CmiAbort ("Unable to allocate memory for VMI_KEY environment variable.");
  }

  sprintf (vmikey, "VMI_KEY=%s:%d\0", synckey, _Cmi_mype);

  if (putenv (vmikey) == -1) {
    CmiAbort ("Unable to set VMI_KEY environment variable.");
  }

  /* Set the maximum size of inlined stream messages. */
  vmiinlinesize = (char *) malloc (50);
  if (!vmiinlinesize) {
    CmiAbort ("Unable to allocate memory for environment variable.");
  }

  sprintf (vmiinlinesize, "VMI_INLINED_DATA_SZ=%d\0", CMI_VMI_Medium_Message_Boundary);

  if (putenv (vmiinlinesize) == -1) {
    CmiAbort ("Unable to set VMI_INLINED_DATA_SZ environment variable.");
  }

  /* Initialize VMI. */
  DEBUG_PRINT ("Initializing VMI with key %s\n", vmikey);
  status = VMI_Init (0, NULL);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Init()");

  /* Create buffer pools. */
  status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_Medium_Message_Boundary,
					sizeof (PVOID),
       CMI_VMI_MESSAGE_BUFFER_POOL_PREALLOCATE,
       CMI_VMI_MESSAGE_BUFFER_POOL_GROW,
       (VMI_POOL_HANDLE | VMI_POOL_REGISTER | VMI_POOL_CLEARONCE),
       &CMI_VMI_MessageBuffer_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (sizeof (CMI_VMI_CmiCommHandle_T),
       sizeof (PVOID), CMI_VMI_CMICOMMHANDLE_POOL_PREALLOCATE,
       CMI_VMI_CMICOMMHANDLE_POOL_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_CmiCommHandle_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (sizeof (CMI_VMI_Handle_T),
       sizeof (PVOID), CMI_VMI_HANDLE_POOL_PREALLOCATE,
       CMI_VMI_HANDLE_POOL_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_Handle_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (_Cmi_numpes * sizeof (int),
       sizeof (PVOID), CMI_VMI_RDMA_BYTES_SENT_POOL_PREALLOCATE,
       CMI_VMI_RDMA_BYTES_SENT_POOL_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_RDMABytesSent_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (sizeof (CMI_VMI_RDMA_Put_Context_T),
       sizeof (PVOID), CMI_VMI_RDMA_PUT_CONTEXT_POOL_PREALLOCATE,
       CMI_VMI_RDMA_PUT_CONTEXT_POOL_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_RDMAPutContext_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_RDMA_Max_Outstanding *
       sizeof (PVMI_CACHE_ENTRY), sizeof (PVOID),
       CMI_VMI_RDMA_CACHE_ENTRY_POOL_PREALLOCATE,
       CMI_VMI_RDMA_CACHE_ENTRY_POOL_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_RDMACacheEntry_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET1_SIZE, sizeof (PVOID),
       CMI_VMI_BUCKET1_PREALLOCATE, CMI_VMI_BUCKET1_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_Bucket1_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET2_SIZE, sizeof (PVOID),
       CMI_VMI_BUCKET2_PREALLOCATE, CMI_VMI_BUCKET2_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_Bucket2_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET3_SIZE, sizeof (PVOID),
       CMI_VMI_BUCKET3_PREALLOCATE, CMI_VMI_BUCKET3_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_Bucket3_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET4_SIZE, sizeof (PVOID),
       CMI_VMI_BUCKET4_PREALLOCATE, CMI_VMI_BUCKET4_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_Bucket4_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET5_SIZE, sizeof (PVOID),
       CMI_VMI_BUCKET5_PREALLOCATE, CMI_VMI_BUCKET5_GROW, VMI_POOL_CLEARONCE,
       &CMI_VMI_Bucket5_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

  /* Initialize RDMA receive context table entries. */
  for (i = 0; i < CMI_VMI_MAX_RECEIVE_HANDLES; i++) {
    (&(CMI_VMI_RDMA_Receive_Context[i]))->allocated = FALSE;
    (&(CMI_VMI_RDMA_Receive_Context[i]))->index = i;
  }

  /* Create the FIFOs for holding local and remote messages. */
  CpvAccess (CmiLocalQueue) = CdsFifo_Create ();
  CpvAccess (CMI_VMI_RemoteQueue) = CdsFifo_Create ();

  /* Set a stream receive function. */
  VMI_STREAM_SET_RECV_FUNCTION (CMI_VMI_Stream_Receive_Handler);

  /* Set a connection accept function. */
  status = VMI_Connection_Accept_Fn (CMI_VMI_Connection_Accept_Handler);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Connection_Accept_Fn()");

  /* Set a connection disconnect function. */
  status = VMI_Connection_Disconnect_Fn(CMI_VMI_Connection_Disconnect_Handler);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Connection_Disconnect_Fn()");

  DEBUG_PRINT ("Initialized VMI successfully.\n");

  /*
    **********
    * STEP 3 *   Re-synchronize with all processes.
    **********

    In this step, we re-synchronize with all processes prior to attempting
    to set up connections.  This is to avoid a race condition in which a
    process could initialize VMI and then attempt to synchronize with a
    much slower process that has not yet had an opportunity to initialize
    VMI or set up its receive and connection accept handlers.  For this
    reason, IT IS CRITICAL that we set state such as the VMI stream
    receive function, the connection accept function, etc. at the end of
    the previous step!

    The synchronization key used for this step is the same for all processes
    and is "[synchronization key]:Initialized" (with "[synchronization key]"
    being the initial key used to synchronize with the CRM in Step 1 above).
  */

  /* Prepare the synchronization key to be used. */
  initkey = (PUCHAR) malloc (strlen (synckey) + 13);
  if (!initkey) {
    CmiAbort ("Unable to allocate space for initialization key.");
  }
  sprintf (initkey, "%s:Initialized", synckey);

  /* Re-register with the CRM. */
  if (CMI_VMI_CRM_Register (initkey, _Cmi_numpes, FALSE) < -1) {
    CmiAbort ("Unable to re-synchronize with all processes.");
  }

  DEBUG_PRINT ("Successfully re-synchronized with all initialized processes.");

  /*
    **********
    * STEP 4 *   Set up all connections.
    **********
  */
  if (!CMI_VMI_Open_Connections (synckey)) {
    CmiAbort ("Error during connection setup phase.");
  }

  /*
    **********
    * STEP 5 *   Finish.
    **********
  */

  /* Free up resources allocated to keys, etc. */
  free (synckey);
  free (vmikey);
  free (initkey);
  free (vmiinlinesize);



  DEBUG_PRINT ("ConverseInit() completed successfully.\n");

  CthInit (argv);
  ConverseCommonInit (argv);

  // This has to come here because ConverseCommonInit() must be called first.
  CMI_VMI_RDMA_Rendezvous_Handler_ID = CmiRegisterHandler ((CmiHandler) CMI_VMI_RDMA_Rendezvous_Handler);

  if (!initReturns) {
    startFn (CmiGetArgc (argv), argv);
    if (!userCallsScheduler) {
      CsdScheduler (-1);
    }
    ConverseExit ();
  }
}



/**************************************************************************
**
*/
void ConverseExit ()
{
  VMI_STATUS status;

  int i;

  BOOLEAN pending;

  struct timeval tp;
  long starttime;
  long nowtime;


  DEBUG_PRINT ("ConverseExit() called.\n");

  /* Call VMI_Poll() to encourage the network to reach quiescence. */
  for (i = 0; i < 1000000; i++) {
    sched_yield ();
    status = VMI_Poll ();
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
  }

  /*
    Issue a disconnect request to each process with a rank lower than
    this process's rank.
  */
  for (i = 0; i < _Cmi_mype; i++) {
    // Do this first to avoid a race condition where the disconnect
    // completes and then we set the state flag here.
    (&CMI_VMI_Procs[i])->state = CMI_VMI_CONNECTION_DISCONNECTING;

    status = VMI_Connection_Disconnect ((&CMI_VMI_Procs[i])->connection,
         CMI_VMI_Disconnection_Response_Handler, (PVOID) &CMI_VMI_Procs[i]);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Connection_Disconnect()");
  }

  (&CMI_VMI_Procs[_Cmi_mype])->state = CMI_VMI_CONNECTION_DISCONNECTED;

  /* Complete all disconnect requests and accepts. */
  gettimeofday (&tp, NULL);
  starttime = tp.tv_sec;
  nowtime   = tp.tv_sec;
  pending   = TRUE;
  while (pending && ((starttime + CMI_VMI_CONNECTION_TIMEOUT) > nowtime)) {
    sched_yield ();
    status = VMI_Poll ();
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");

    gettimeofday (&tp, NULL);
    nowtime = tp.tv_sec;

    pending = FALSE;
    for (i = 0; i < _Cmi_numpes; i++) {
      pending = pending ||
           ((&CMI_VMI_Procs[i])->state != CMI_VMI_CONNECTION_DISCONNECTED);
    }
  }

  if (pending) {
    DEBUG_PRINT ("Timed out while waiting for disconnects.\n");
  }

  /* Destroy buffer pools. */
  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_MessageBuffer_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_CmiCommHandle_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_Handle_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_RDMABytesSent_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_RDMAPutContext_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_RDMACacheEntry_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_Bucket1_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_Bucket2_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_Bucket3_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_Bucket4_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  status = VMI_Pool_Destroy_Buffer_Pool (CMI_VMI_Bucket5_Pool);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Destroy_Buffer_Pool()");

  /* Free all dynamically-allocated memory. */
  free (CMI_VMI_Procs);

  /* Destroy queues. */
  CdsFifo_Destroy (CpvAccess (CMI_VMI_RemoteQueue));
  CdsFifo_Destroy (CpvAccess (CmiLocalQueue));

  exit (0);
#if 0
  /* Terminate VMI. */
  SET_VMI_SUCCESS (status);
  VMI_Terminate (status);
#endif
}



/**************************************************************************
**
*/
void CmiAbort (const char *message)
{
  DEBUG_PRINT ("CmiAbort() called.\n");

  printf ("%s\n", message);
  exit (1);
}



/**************************************************************************
**
*/
void *CmiGetNonLocal (void)
{
  VMI_STATUS status;


  status = VMI_Poll ();
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");

  return (CdsFifo_Dequeue (CpvAccess (CMI_VMI_RemoteQueue)));
}



/**************************************************************************
**
*/
void CmiMemLock ()
{
  /* Empty. */
}



/**************************************************************************
**
*/
void CmiMemUnlock ()
{
  /* Empty. */
}



/**************************************************************************
**
*/
void CmiNotifyIdle ()
{
  VMI_STATUS status;


  status = VMI_Poll ();
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
}



/**************************************************************************
** done
*/
void CmiSyncSendFn (int destrank, int msgsize, char *msg)
{
  VMI_STATUS status;

  char *msgcopy;
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T handle;
  CMI_VMI_Handle_T *phandle;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;

  char *putaddr;
  int putlen;
  PVMI_RDMA_OP rdmaop;
  PVMI_CACHE_ENTRY cacheentry;
  CMI_VMI_RDMA_Put_Context_T *rdmaputctxt;


  DEBUG_PRINT ("CmiSyncSendFn() called.\n");

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (msg, 0);
#endif

  if (destrank == _Cmi_mype) {
    msgcopy = CmiAlloc (msgsize);
    memcpy (msgcopy, msg, msgsize);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);
  }
#if CMK_PERSISTENT_COMM
  else if ((CMI_VMI_Persistent_Handles != NULL) &&
    ((&CMI_VMI_Persistent_Handles[0])->data.persistent.destrank == destrank) &&
    ((&CMI_VMI_Persistent_Handles[0])->data.persistent.msgsize <= msgsize) &&
    ((&CMI_VMI_Persistent_Handles[0])->data.persistent.ready >= 2)) {
    // TODO: We should probably buffer the message instead of sending direct
    //       if the connection is not ready.
    // TODO: This code should perform the RDMA operation in chunks!
    //       Fix later once this version works.
    phandle = &CMI_VMI_Persistent_Handles[0];

    phandle->refcount++;
    phandle->msg = msg;
    phandle->msgsize = msgsize;

    putaddr = msg;
    putlen = msgsize;

    status = VMI_Cache_Register (putaddr, putlen, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    status = VMI_RDMA_Alloc_Op (&rdmaop);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

    rdmaop->numBufs = 1;
    rdmaop->buffers[0] = cacheentry->bufferHandle;
    rdmaop->addr[0] = putaddr;
    rdmaop->sz[0] = putlen;
    rdmaop->rbuffer = phandle->data.persistent.rbuf;
    rdmaop->roffset = 0;

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMAPutContext_Pool,
				       (PVOID) &rdmaputctxt, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    rdmaputctxt->cacheentry = cacheentry;
    rdmaputctxt->handle = phandle;

    phandle->data.persistent.bytes_sent = msgsize;

    status = VMI_RDMA_Put (phandle->data.persistent.connection, rdmaop,
         (PVOID) rdmaputctxt,
         (VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");

    while (phandle->refcount > 1) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }
  }
#endif   /* CMK_PERSISTENT_COMM */
  else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

    status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[destrank])->connection,
         addrs, sz, 1, msgsize);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
  } else {
    handle.refcount = 2;
    handle.msg = msg;
    handle.msgsize = msgsize;
    handle.commhandle = NULL;
    handle.type = CMI_VMI_HANDLE_TYPE_SYNC_SEND_RDMA;
    handle.data.rdma.bytes_sent = 0;

    CmiSetHandler (&rendezvous_msg, CMI_VMI_RDMA_Rendezvous_Handler_ID);
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) &handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

    status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[destrank])->connection,
	 addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

    while (handle.refcount > 1) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }
  }
}



/**************************************************************************
** done
*/
CmiCommHandle CmiAsyncSendFn (int destrank, int msgsize, char *msg)
{
  VMI_STATUS status;

  char *msgcopy;
  PVMI_BUFFER bufHandles[2];
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T *handle;

  CMI_VMI_CmiCommHandle_T *commhandle;

  PVMI_CACHE_ENTRY cacheentry;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;

  CMI_VMI_Handle_T *phandle;
  char *putaddr;
  int putlen;
  PVMI_RDMA_OP rdmaop;
  CMI_VMI_RDMA_Put_Context_T *rdmaputctxt;


  DEBUG_PRINT ("CmiAsyncSendFn() called.\n");

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (msg, 0);
#endif

  if (destrank == _Cmi_mype) {
    msgcopy = CmiAlloc (msgsize);
    memcpy (msgcopy, msg, msgsize);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);
    commhandle = NULL;
  }
#if CMK_PERSISTENT_COMM
  else if ((CMI_VMI_Persistent_Handles != NULL) &&
    ((&CMI_VMI_Persistent_Handles[0])->data.persistent.destrank == destrank) &&
    ((&CMI_VMI_Persistent_Handles[0])->data.persistent.msgsize <= msgsize) &&
    ((&CMI_VMI_Persistent_Handles[0])->data.persistent.ready >= 2)) {
    // TODO: We should probably buffer the message instead of sending direct
    //       if the connection is not ready.
    // TODO: This code should perform the RDMA operation in chunks!
    //       Fix later once this version works.
    phandle = &CMI_VMI_Persistent_Handles[0];

    phandle->refcount++;
    phandle->msg = msg;
    phandle->msgsize = msgsize;

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_CmiCommHandle_Pool,
				       (PVOID) &commhandle, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    commhandle->count = 1;
    phandle->commhandle = commhandle;
    // CANNOT DO THIS -->   CMI_VMI_AsyncMsgCount++;
    // Because CMI_VMI_RDMA_Completion_Handler() does not properly
    // decrement for the case of persistent handles.

    putaddr = msg;
    putlen = msgsize;

    status = VMI_Cache_Register (putaddr, putlen, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    status = VMI_RDMA_Alloc_Op (&rdmaop);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

    rdmaop->numBufs = 1;
    rdmaop->buffers[0] = cacheentry->bufferHandle;
    rdmaop->addr[0] = putaddr;
    rdmaop->sz[0] = putlen;
    rdmaop->rbuffer = phandle->data.persistent.rbuf;
    rdmaop->roffset = 0;

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMAPutContext_Pool,
				       (PVOID) &rdmaputctxt, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    rdmaputctxt->cacheentry = cacheentry;
    rdmaputctxt->handle = phandle;

    phandle->data.persistent.bytes_sent = msgsize;

    status = VMI_RDMA_Put (handle->data.persistent.connection, rdmaop,
         (PVOID) rdmaputctxt,
         (VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
  }
#endif   /* CMK_PERSISTENT_COMM */
  else if (msgsize < CMI_VMI_Small_Message_Boundary) {
    addrs[0] = (PVOID) msg;
    sz[0] = msgsize;

    status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[destrank])->connection,
         addrs, sz, 1, msgsize);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

    commhandle = NULL;
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    status = VMI_Pool_Allocate_Buffer (CMI_VMI_CmiCommHandle_Pool,
				       (PVOID) &commhandle, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_Handle_Pool, (PVOID) &handle,
				       NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->refcount = 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->commhandle = commhandle;
    handle->type = CMI_VMI_HANDLE_TYPE_ASYNC_SEND_STREAM;
    handle->data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

    commhandle->count = 1;
    CMI_VMI_AsyncMsgCount++;

    status = VMI_Stream_Send ((&CMI_VMI_Procs[destrank])->connection,
	 bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler,
	 (PVOID) handle, TRUE);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
  } else {
    status = VMI_Pool_Allocate_Buffer (CMI_VMI_CmiCommHandle_Pool,
				       (PVOID) &commhandle, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_Handle_Pool, (PVOID) &handle,
				       NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    handle->refcount = 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->commhandle = commhandle;
    handle->type = CMI_VMI_HANDLE_TYPE_ASYNC_SEND_RDMA;
    handle->data.rdma.bytes_sent = 0;

    CmiSetHandler ((int *) &rendezvous_msg,
		   CMI_VMI_RDMA_Rendezvous_Handler_ID);
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) &handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

    commhandle->count = 1;
    CMI_VMI_AsyncMsgCount++;

    status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[destrank])->connection,
	 addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

  }

  return ((CmiCommHandle) commhandle);
}



/**************************************************************************
** done
*/
void CmiFreeSendFn (int destrank, int msgsize, char *msg)
{
  DEBUG_PRINT ("CmiFreeSendFn() called.\n");

  if (destrank == _Cmi_mype) {
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msg);
  } else {
    CmiSyncSendFn (destrank, msgsize, msg);
    CmiFree (msg);
  }
}



/**************************************************************************
** done
*/
void CmiSyncBroadcastFn (int msgsize, char *msg)
{
  VMI_STATUS status;

  PVMI_BUFFER bufHandles[2];
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T handle;

  int i;

  PVMI_CACHE_ENTRY cacheentry;

  int childcount;
  int startrank;
  int destrank;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;


  DEBUG_PRINT ("CmiSyncBroadcastFn() called.\n");

  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle.msg = msg;
    handle.msgsize = msgsize;
    handle.commhandle = NULL;
    handle.type = CMI_VMI_HANDLE_TYPE_SYNC_BROADCAST_STREAM;
    handle.data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle.refcount = childcount + 1;

    startrank = CMI_BROADCAST_ROOT (msg) - 1;
    for (i = 1; i <= CMI_VMI_BROADCAST_SPANNING_FACTOR; i++) {
      destrank = _Cmi_mype - startrank;

      if (destrank < 0) {
	destrank += _Cmi_numpes;
      }

      destrank = CMI_VMI_BROADCAST_SPANNING_FACTOR * destrank + i;

      if (destrank > (_Cmi_numpes - 1)) {
	break;
      }

      destrank += startrank;
      destrank %= _Cmi_numpes;

      status = VMI_Stream_Send ((&CMI_VMI_Procs[destrank])->connection,
	   bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler,
	   (PVOID) &handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle.refcount = _Cmi_numpes;

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Procs[i])->connection, bufHandles,
           addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) &handle,
           TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Procs[i])->connection, bufHandles,
           addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) &handle,
           TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */

    while (handle.refcount > 1) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    status = VMI_Cache_Deregister (cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
  } else {
    handle.msg = msg;
    handle.msgsize = msgsize;
    handle.commhandle = NULL;
    handle.type = CMI_VMI_HANDLE_TYPE_SYNC_BROADCAST_RDMA;

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMABytesSent_Pool,
	 (PVOID) &(handle.data.rdmabroad.bytes_sent), NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    CmiSetHandler ((int *) &rendezvous_msg,
		   CMI_VMI_RDMA_Rendezvous_Handler_ID);
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) &handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle.refcount = childcount + 1;

    startrank = CMI_BROADCAST_ROOT (msg) - 1;
    for (i = 1; i <= CMI_VMI_BROADCAST_SPANNING_FACTOR; i++) {
      destrank = _Cmi_mype - startrank;

      if (destrank < 0) {
	destrank += _Cmi_numpes;
      }

      destrank = CMI_VMI_BROADCAST_SPANNING_FACTOR * destrank + i;

      if (destrank > (_Cmi_numpes - 1)) {
	break;
      }

      destrank += startrank;
      destrank %= _Cmi_numpes;

      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[destrank])->connection,
	   addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle.refcount = _Cmi_numpes;

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[i])->connection,
	   addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[i])->connection,
	   addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */

    while (handle.refcount > 1) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    status = VMI_Pool_Deallocate_Buffer (CMI_VMI_RDMABytesSent_Pool,
					 handle.data.rdmabroad.bytes_sent);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
  }
}



/**************************************************************************
**
*/
CmiCommHandle CmiAsyncBroadcastFn (int msgsize, char *msg)
{
  VMI_STATUS status;

  PVMI_BUFFER bufHandles[2];
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T *handle;

  CMI_VMI_CmiCommHandle_T *commhandle;

  int i;

  PVMI_CACHE_ENTRY cacheentry;

  int childcount;
  int startrank;
  int destrank;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;


  DEBUG_PRINT ("CmiAsyncBroadcastFn() called.\n");

  if (msgsize < CMI_VMI_Small_Message_Boundary) {
    addrs[0] = (PVOID) msg;
    sz[0] = msgsize;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    startrank = CMI_BROADCAST_ROOT (msg) - 1;
    for (i = 1; i <= CMI_VMI_BROADCAST_SPANNING_FACTOR; i++) {
      destrank = _Cmi_mype - startrank;

      if (destrank < 0) {
	destrank += _Cmi_numpes;
      }

      destrank = CMI_VMI_BROADCAST_SPANNING_FACTOR * destrank + i;

      if (destrank > (_Cmi_numpes - 1)) {
	break;
      }

      destrank += startrank;
      destrank %= _Cmi_numpes;

      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[destrank])->connection,
           addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[i])->connection,
           addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[i])->connection,
           addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */

    commhandle = NULL;
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    status = VMI_Pool_Allocate_Buffer (CMI_VMI_CmiCommHandle_Pool,
				       (PVOID) &commhandle, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_Handle_Pool, (PVOID) &handle,
				       NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->commhandle = commhandle;
    handle->type = CMI_VMI_HANDLE_TYPE_ASYNC_BROADCAST_STREAM;
    handle->data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle->refcount = childcount;
    commhandle->count = childcount;
    CMI_VMI_AsyncMsgCount += childcount;

    startrank = CMI_BROADCAST_ROOT (msg) - 1;
    for (i = 1; i <= CMI_VMI_BROADCAST_SPANNING_FACTOR; i++) {
      destrank = _Cmi_mype - startrank;

      if (destrank < 0) {
	destrank += _Cmi_numpes;
      }

      destrank = CMI_VMI_BROADCAST_SPANNING_FACTOR * destrank + i;

      if (destrank > (_Cmi_numpes - 1)) {
	break;
      }

      destrank += startrank;
      destrank %= _Cmi_numpes;

      status = VMI_Stream_Send ((&CMI_VMI_Procs[destrank])->connection,
           bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler,
	   (PVOID) &handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle->refcount = (_Cmi_numpes - 1);
    commhandle->count = (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Procs[i])->connection, bufHandles,
           addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) &handle,
           TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Procs[i])->connection, bufHandles,
           addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) &handle,
           TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */
  } else {
    status = VMI_Pool_Allocate_Buffer (CMI_VMI_CmiCommHandle_Pool,
				       (PVOID) &commhandle, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_Handle_Pool, (PVOID) &handle,
				       NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->commhandle = commhandle;
    handle->type = CMI_VMI_HANDLE_TYPE_ASYNC_BROADCAST_RDMA;

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMABytesSent_Pool,
	 (PVOID) &(handle->data.rdmabroad.bytes_sent), NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    CmiSetHandler ((int *) &rendezvous_msg,
		   CMI_VMI_RDMA_Rendezvous_Handler_ID);
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) &handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle->refcount = childcount;
    commhandle->count = childcount;
    CMI_VMI_AsyncMsgCount += childcount;

    startrank = CMI_BROADCAST_ROOT (msg) - 1;

    for (i = 1; i <= CMI_VMI_BROADCAST_SPANNING_FACTOR; i++) {
      destrank = _Cmi_mype - startrank;

      if (destrank < 0) {
	destrank += _Cmi_numpes;
      }

      destrank = CMI_VMI_BROADCAST_SPANNING_FACTOR * destrank + i;

      if (destrank > (_Cmi_numpes - 1)) {
	break;
      }

      destrank += startrank;
      destrank %= _Cmi_numpes;

      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[destrank])->connection,
           addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle->refcount = (_Cmi_numpes - 1);
    commhandle->count = (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[i])->connection,
           addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[i])->connection,
           addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */
  }

  return ((CmiCommHandle) commhandle);
}



/**************************************************************************
** done
*/
void CmiFreeBroadcastFn (int msgsize, char *msg)
{
  DEBUG_PRINT ("CmiFreeBroadcastFn() called.\n");

  CmiSyncBroadcastFn (msgsize, msg);

  CmiFree (msg);
}



/**************************************************************************
** done
*/
void CmiSyncBroadcastAllFn (int msgsize, char *msg)
{
  char *msgcopy;


  DEBUG_PRINT ("CmiSyncBroadcastAllFn() called.\n");

  msgcopy = CmiAlloc (msgsize);
  memcpy (msgcopy, msg, msgsize);
  CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);

  CmiSyncBroadcastFn (msgsize, msg);
}



/**************************************************************************
** done
*/
CmiCommHandle CmiAsyncBroadcastAllFn (int msgsize, char *msg)
{
  char *msgcopy;


  DEBUG_PRINT ("CmiAsyncBroadcastAllFn() called.\n");

  msgcopy = CmiAlloc (msgsize);
  memcpy (msgcopy, msg, msgsize);
  CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);

  return (CmiAsyncBroadcastFn (msgsize, msg));
}



/**************************************************************************
** done
*/
void CmiFreeBroadcastAllFn (int msgsize, char *msg)
{
  char *msgcopy;


  DEBUG_PRINT ("CmiFreeBroadcastAllFn() called.\n");

  msgcopy = CmiAlloc (msgsize);
  memcpy (msgcopy, msg, msgsize);
  CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);

  CmiFreeBroadcastFn (msgsize, msg);
}



#if CMK_MULTICAST_LIST_USE_SPECIAL_CODE
/**************************************************************************
** done
*/
void CmiSyncListSendFn(int npes, int *pes, int len, char *msg)
{
  CmiError("ListSend not implemented.");
}


/**************************************************************************
** done
*/
CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int len, char *msg)
{
  CmiError("ListSend not implemented.");
  return (CmiCommHandle) 0;
}


/**************************************************************************
** done
*/
void CmiFreeListSendFn (int npes, int *pes, int msgsize, char *msg)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *phandle;
  CMI_VMI_RDMA_Put_Context_T *rdmaputctxt;
  PVMI_RDMA_OP rdmaop;
  PVMI_CACHE_ENTRY cacheentry;
  char *putaddr;
  int putlen;
  int i;


  // This code completely ignores pes passed in as a parameter and
  // instead uses the destinations held in the persistent handle array.

  CmiAssert (npes == CMI_VMI_Persistent_Handles_Size);

  for (i = 0; i < npes; i++) {
    phandle = &CMI_VMI_Persistent_Handles[i];

    phandle->refcount++;
    phandle->msg = msg;
    phandle->msgsize = msgsize;

    putaddr = msg;
    putlen = msgsize;

    status = VMI_Cache_Register (putaddr, putlen, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    status = VMI_RDMA_Alloc_Op (&rdmaop);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

    rdmaop->numBufs = 1;
    rdmaop->buffers[0] = cacheentry->bufferHandle;
    rdmaop->addr[0] = putaddr;
    rdmaop->sz[0] = putlen;
    rdmaop->rbuffer = phandle->data.persistent.rbuf;
    rdmaop->roffset = 0;

    status = VMI_Pool_Allocate_Buffer (CMI_VMI_RDMAPutContext_Pool,
				       (PVOID) &rdmaputctxt, NULL);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

    rdmaputctxt->cacheentry = cacheentry;
    rdmaputctxt->handle = phandle;

    phandle->data.persistent.bytes_sent = msgsize;

    status = VMI_RDMA_Put (phandle->data.persistent.connection, rdmaop,
         (PVOID) rdmaputctxt,
         (VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");

    while (phandle->refcount > 1) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }
  }

  CmiFree(msg);
}
#endif



/**************************************************************************
**
*/
int CmiAllAsyncMsgsSent ()
{
  DEBUG_PRINT ("CmiAllAsyncMsgsSent() called.\n");

  return (CMI_VMI_AsyncMsgCount < 1);
}



/**************************************************************************
**
*/
int CmiAsyncMsgSent (CmiCommHandle cmicommhandle)
{
  CMI_VMI_CmiCommHandle_T *commhandle;


  DEBUG_PRINT ("CmiAsyncMsgSent() called.\n");

  commhandle = (CMI_VMI_CmiCommHandle_T *) cmicommhandle;

  if (commhandle == NULL) {
    return (TRUE);
  }

  return (commhandle->count < 1);
}



/**************************************************************************
**
*/
void CmiReleaseCommHandle (CmiCommHandle cmicommhandle)
{
  VMI_STATUS status;

  CMI_VMI_CmiCommHandle_T *commhandle;



  DEBUG_PRINT ("CmiReleaseCommHandle() called.\n");

  commhandle = (CMI_VMI_CmiCommHandle_T *) cmicommhandle;

  status = VMI_Pool_Deallocate_Buffer (CMI_VMI_CmiCommHandle_Pool, commhandle);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
}





#if CMK_PERSISTENT_COMM
/**************************************************************************
**
*/
void CmiPersistentInit ()
{
  CMI_VMI_Persistent_Request_Handler_ID =
    CmiRegisterHandler ((CmiHandler) CMI_VMI_Persistent_Request_Handler);

  CMI_VMI_Persistent_Grant_Handler_ID =
    CmiRegisterHandler ((CmiHandler) CMI_VMI_Persistent_Grant_Handler);

  CMI_VMI_Persistent_Destroy_Handler_ID =
    CmiRegisterHandler ((CmiHandler) CMI_VMI_Persistent_Destroy_Handler);
}



/**************************************************************************
**
*/
PersistentHandle CmiCreatePersistent (int destrank, int maxsize)
{ 
  VMI_STATUS status;

  CMI_VMI_Persistent_Request_Message_T request_msg;
  CMI_VMI_Handle_T *handle;

  PVOID addrs[1];
  ULONG sz[1];


  status = VMI_Pool_Allocate_Buffer (CMI_VMI_Handle_Pool, (PVOID) &handle,
				     NULL);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Allocate_Buffer()");

  handle->refcount = 1;
  handle->msg = NULL;
  handle->msgsize = -1;
  handle->commhandle = NULL;
  handle->type = CMI_VMI_HANDLE_TYPE_PERSISTENT;
  handle->data.persistent.ready = 0;
  handle->data.persistent.connection = (&CMI_VMI_Procs[destrank])->connection;
  handle->data.persistent.destrank = destrank;
  handle->data.persistent.maxsize = maxsize;
  handle->data.persistent.rbuf = NULL;
  handle->data.persistent.bytes_sent = 0;
  handle->data.persistent.rdmarecvindx = -1;

  CmiSetHandler (&request_msg, CMI_VMI_Persistent_Request_Handler_ID);
  request_msg.rank = _Cmi_mype;
  request_msg.maxsize = maxsize;
  request_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) handle;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (&request_msg, 0);
#endif

  addrs[0] = (PVOID) &request_msg;
  sz[0] = (ULONG) (sizeof (CMI_VMI_Persistent_Request_Message_T));

  status = VMI_Stream_Send_Inline ((&CMI_VMI_Procs[destrank])->connection,
       addrs, sz, 1, sizeof (CMI_VMI_Persistent_Request_Message_T));
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

  return ((PersistentHandle) handle);
}



/**************************************************************************
**
*/
void CmiUsePersistentHandle (PersistentHandle *handle_array, int n)
{
  CMI_VMI_Persistent_Handles = handle_array;
  CMI_VMI_Persistent_Handles_Size = n;
}


/**************************************************************************
**
*/
void CmiDestroyPersistent (PersistentHandle h)
{
  VMI_STATUS status;

  CMI_VMI_Persistent_Destroy_Message_T destroy_msg;
  CMI_VMI_Handle_T *handle;

  PVOID addrs[1];
  ULONG sz[1];


  handle = (CMI_VMI_Handle_T *) h;

  CmiSetHandler (&destroy_msg, CMI_VMI_Persistent_Destroy_Handler_ID);
  destroy_msg.rdmarecvindx = handle->data.persistent.rdmarecvindx;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (&destroy_msg, 0);
#endif

  addrs[0] = (PVOID) &destroy_msg;
  sz[0] = (ULONG) (sizeof (CMI_VMI_Persistent_Destroy_Message_T));

  status = VMI_Stream_Send_Inline (handle->data.persistent.connection,
       addrs, sz, 1, sizeof (CMI_VMI_Persistent_Destroy_Message_T));
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

  status = VMI_Pool_Deallocate_Buffer (CMI_VMI_Handle_Pool, handle);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Deallocate_Buffer()");
}



/**************************************************************************
**
*/
void CmiDestroyAllPersistent ()
{
  printf ("*** WARNING: CmiDestroyAllPersistent() not implemented! ***\n");
}
#endif   /* CMK_PERSISTENT_COMM */





/*************************************************************************/
/*************************************************************************/
/******************* N C S A   C R M   R O U T I N E S *******************/
/*************************************************************************/
/*************************************************************************/



/**************************************************************************
** Copyright (C) 2001 Board of Trustees of the University of Illinois
**
** This software, both binary and source, is copyrighted by The
** Board of Trustees of the University of Illinois.  Ownership
** remains with the University.  You should have received a copy
** of a licensing agreement with this software.  See the file
** "COPYRIGHT", or contact the University at this address:
**
**     National Center for Supercomputing Applications
**     University of Illinois
**     405 North Mathews Ave.
**     Urbana, IL 61801
*/

#include <linux/types.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <netinet/in.h>

#define CRM_GLOBAL
#define INTERNAL
#define LINUX

/**************************************************************************
**
*/
BOOLEAN CRMInit ()
{
  char *CRMLoc;
  char *temp;

  CRMLoc = getenv ("CRM");
  if (CRMLoc)
  {
    if ((temp = strstr (CRMLoc, ":")))
    {
      CRMPort = atoi (temp + 1);
      if (CRMPort == 0)
      {
	printf ("CRM: Invalid Port Type\n");
	return FALSE;
      }
      CRMHost = (char *) malloc (temp - CRMLoc);
      memcpy (CRMHost, CRMLoc, (size_t) (temp - CRMLoc));
    }
    else
    {
      CRMHost = strdup (CRMLoc);
      strcpy (CRMHost, CRMLoc);
      CRMPort = CRM_DEFAULT_PORT;
    }

    return TRUE;
  }
  else
  {
    printf ("CRM Environment variable not defined.\n");
  }

  return FALSE;
}

/* Socket Utility Functions */
SOCKET createSocket(char *hostName, int port, int *localAddr){

#ifdef WIN32  
  SOCKET sock;
#endif
  int sock;
  int status;
  int sockaddrLen;
  struct sockaddr_in peer;
  struct sockaddr_in local;
  struct hostent *hostEntry;
  unsigned long addr;
  
#ifdef WIN32
  sock = WSASocket(
		   AF_INET,
		   SOCK_STREAM,
		   0,
		   (LPWSAPROTOCOL_INFO) NULL,
		   0,
		   0
		   );
  if (sock == INVALID_SOCKET){
    perror("CRM: Create Socket failed.");
    exit(1);
  }
#endif

#ifdef LINUX
  sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0){
    perror("CRM: Create Socket failed.");
    exit(1);
  }
#endif
  

  
#ifdef WIN32
  ZeroMemory(&local, sizeof(local));
#endif
#ifdef LINUX
  bzero(&local, sizeof(local));
#endif
  
  local.sin_family = AF_INET;
  
  status = bind(
		sock,
#ifdef WIN32
		(struct sockaddr FAR*) &local,
#endif
#ifdef LINUX
		(struct sockaddr*) &local,
#endif
		sizeof(local)
		);
#ifdef WIN32  
  if (status == SOCKET_ERROR){
    printf("Bind error %d\n", WSAGetLastError());
    exit(1);
  }
#endif
#ifdef LINUX
  if (status < 0){
    fprintf(stderr,"Bind error.\n");
    fflush(stderr);
    exit(1);
  }
#endif
  
  hostEntry = gethostbyname(hostName);
  if (!hostEntry){
    printf("Unable to resolve hostname %s\n", hostName);
    exit(1);
  }

  memcpy((void*) &addr, (void*) hostEntry -> h_addr_list[0], hostEntry -> h_length);

#ifdef WIN32
  ZeroMemory(&peer, sizeof(peer));
#endif
#ifdef LINUX
  bzero(&peer, sizeof(peer));
#endif
  
  peer.sin_family = AF_INET;
  peer.sin_port = htons((u_short) port);
  peer.sin_addr.s_addr = addr;
  
#ifdef WIN32  
  status = WSAConnect(
		      sock,
		      (struct sockaddr FAR*) &peer,
		      sizeof(peer),
		      NULL,
		      NULL,
		      NULL,
		      NULL
		      );
  
  if (status == SOCKET_ERROR){
    printf("Connect error %d\n", WSAGetLastError());
    perror("CRM: Connect failed.");
    exit(1);
  }
  ZeroMemory(&local, sizeof(local));
#endif
  
#ifdef LINUX
  status = connect(
		   sock,
		   (struct sockaddr *) &peer,
		   sizeof(peer)
		   );
  
  if (status < 0){
    fprintf(stderr, "CRM: Connect failed.\n");
    fflush(stderr);
    perror("Connect: ");
    exit(1);
  }
  bzero(&local, sizeof(local));
#endif
  
  sockaddrLen = sizeof(local);
  status = getsockname(
		       sock,
#ifdef WIN32
		       (struct sockaddr FAR*) &local,
#endif
#ifdef LINUX
		       (struct sockaddr*) &local,
#endif
		       &sockaddrLen
		       );
  
#ifdef WIN32
  if (status == SOCKET_ERROR){
    perror("CRM: Getsockname failed.");
    exit(1);
  }
#endif
#ifdef LINUX
  if (status < 0){
    perror("CRM: GetSockName failed.");
    exit(1);
  }
#endif
  
  *localAddr = (int) local.sin_addr.s_addr;

  return sock;
}


/**************************************************************************
**
*/
BOOLEAN CRMRegister (char *key, ULONG numPE, int shutdownPort,
		     SOCKET *clientSock, int *clientAddr, PCRM_Msg *msg2)
{
  PCRM_Msg msg;
  int bytes;
  int bsend;


  /* Perform sanity checking on variables passed to CRM. */
  if (strlen(key) > MAX_STR_LENGTH){
    fprintf(stderr, "Unable to synchronize processes via CRM. Maximum supported VMI_KEY length is %d. User specified VMI_KEY is %d bytes.\n", MAX_STR_LENGTH, strlen(key));
    fflush(stderr);
    return FALSE;
  }

  if (numPE == 0){
    fprintf(stderr, "Malformed number of MPI processes (%d) specified in VMI_PROCS. Unable to synchronize using the CRM.\n", numPE);
    fflush(stderr);
    return FALSE;
  }

  /* Connect to the CRM. */
  *clientSock = createSocket (CRMHost, CRMPort, clientAddr);

  /* Create a CRM Register message structure and populate it. */
  msg = (PCRM_Msg) malloc (sizeof (CRM_Msg));
  
  msg->msgCode = htonl (CRM_MSG_REGISTER);
  msg->msg.CRMReg.np = htonl (numPE);
  msg->msg.CRMReg.shutdownPort = htonl ((u_long) shutdownPort);
  msg->msg.CRMReg.keyLength = htonl (strlen (key));
  strcpy (msg->msg.CRMReg.key, key);

  /* Send the Register message to the CRM. */
  bsend = ((sizeof (int) * 4) + strlen (key));
  bytes = CRMSend(*clientSock, (char *) &msg -> msgCode, sizeof(int));
  bytes += CRMSend(*clientSock, (char *) &msg -> msg.CRMReg.np, bsend - 4);

  if (bytes != bsend)
  {
    printf ("Error in sending Register message.\n");
  }
  
  /* Receive the return code from the CRM. */
  bytes = CRMRecv (*clientSock, (char *) msg, sizeof (int));
  if (bytes == SOCKET_ERROR)
  {
    goto sockError;
  }

  msg->msgCode = (int) ntohl ((u_long) msg->msgCode);
  
  /* Parse the return code. */
  switch (msg->msgCode)
  {
    case CRM_MSG_SUCCESS:
      /* Get the number of processors. */
      bytes = CRMRecv (*clientSock, (char *) &bsend, sizeof (int));
      if (bytes == SOCKET_ERROR)
      {
	goto sockError;
      }

      bsend = (int) ntohl ((u_long) bsend);
      msg->msg.CRMCtx.np = bsend;

      /* Get the block of nodeCtx's. */
      msg->msg.CRMCtx.node = (struct nodeCtx *) malloc (sizeof (nodeCtx) *
							msg->msg.CRMCtx.np);
      bytes = CRMRecv (*clientSock,
		       (char *) msg->msg.CRMCtx.node,
		       sizeof (nodeCtx) * msg->msg.CRMCtx.np);
      if (bytes == SOCKET_ERROR)
      {
	free (msg->msg.CRMCtx.node);
	goto sockError;
      }

      break;

    case CRM_MSG_FAILED:
      bytes = CRMRecv (*clientSock, (char *) &msg->msg.CRMErr, sizeof(errMsg));
      if (bytes == SOCKET_ERROR)
      {
	goto sockError;
      }
      msg->msg.CRMErr.errCode = (int) htonl ((u_long) msg->msg.CRMErr.errCode);
      switch (msg->msg.CRMErr.errCode)
      {
        case CRM_ERR_CTXTCONFLICT:
	  fprintf (stderr,
		   "CONFLICT: Key %s is being used by another program.\n",key);
	  fflush (stderr);
	  break;

        case CRM_ERR_INVALIDCTXT:
	  fprintf (stderr, "CRM: Unable to create context at CRM.\n");
	  fflush (stderr);
	  break;

        case CRM_ERR_TIMEOUT:
	  fprintf (stderr,
		   "TIMEOUT: Timeout waiting for other processes to join.\n");
	  fflush (stderr);
	  break;

        case CRM_ERR_OVERFLOW:
          fprintf (stderr,
		   "CRM: # of PE's and Processes spawned do not match.\n");
	  fflush (stderr);
	  break;
      }
      free (msg);
      return FALSE;

    default:
      printf ("Unknown response code 0x%08x from CRM\n", msg->msgCode);
      free (msg);
      return FALSE;
  }

  *msg2 = msg;

  return TRUE;

sockError:
  /* Free CRM msg struct */
  free (msg);
  perror ("CRM: Socket Error.");
  return FALSE;
}


/**************************************************************************
**
*/
BOOLEAN CRMParseMsg (PCRM_Msg msg, int rank, int *nodeIP,
		     int *shutdownPort, int *nodePE)
{
  PNodeCtx msgRank;


  if ((msg->msgCode) != CRM_MSG_SUCCESS)
  {
    return FALSE;
  }

  if (rank > msg->msg.CRMCtx.np)
  {
    return FALSE;
  }

  msgRank = (msg->msg.CRMCtx.node + rank);
  
  *nodeIP = msgRank->nodeIP;
  *shutdownPort = (int) ntohl ((u_long) msgRank->shutdownPort);
  *nodePE = (int) ntohl ((u_long) msgRank->nodePE);

  return TRUE;
}


/**************************************************************************
**
*/
int CRMSend (SOCKET s, char *msg, int n)
{
  int sent;
  int bsent;
  
  sent = 0;
  while (sent < n)
  {
    bsent = send (s, (const void *) msg + sent, (n - sent), 0);
    if (bsent < 0)
    {
      return bsent;
    }
    else
    {
      sent += bsent;
    }
  }

  return sent;
}


/**************************************************************************
**
*/
int CRMRecv (SOCKET s, char *msg, int n)
{
  int recvd;
  int brecv;

  recvd = 0;
  while (recvd < n)
  {
    brecv = recv (s, (void *) msg + recvd, (n - recvd), 0);
    if (brecv < 0)
    {
      return brecv;
    }
    else
    {
      recvd += brecv;
    }
  }

  return recvd;
}
