/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
**
** This code does not work correctly with glibc 2.2.92 or lower due to
** problems with Converse threads interacting with VMI's use of pthreads.
**
***************************************************************************
**
** IMPROVE STARTUP/SHUTDOWN
**    * startup via charmrun
**    * startup via improved CRM
**    * processes organize into a ring for improved normal/fault shutdown
**    * fix race condition in shutdown
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
**
** VMI HARDWARE MULTICAST SUPPORT
**    * very cool, but very complicated
*/

#include "machine.h"

/* The following are external variables used by the VMI core. */
extern USHORT VMI_DEVICE_RUNTIME;
extern PVMI_NETADDRESS localAddress;
extern VMIStreamRecv recvFn;


/* The following are variables and functions used by the Converse core. */
int _Cmi_mype;
int _Cmi_numpes;
int _Cmi_myrank = 0;

CpvDeclare (void *, CmiLocalQueue);
CpvDeclare (void *, CMI_VMI_RemoteQueue);

extern void CthInit (char **argv);
extern void ConverseCommonInit (char **argv);


/* This is the global count of outstanding asynchronous messages. */
volatile int CMI_VMI_AsyncMsgCount;


/* This is the maximum number of send and receive handles. */
int CMI_VMI_Maximum_Handles;
int CMI_VMI_Next_Handle;


/* This is the global array of all processes in the computation. */
CMI_VMI_Process_T *CMI_VMI_Processes;


/* This is the global array of send and receive handles. */
CMI_VMI_Handle_T *CMI_VMI_Handles;


int CMI_VMI_Small_Message_Boundary;
int CMI_VMI_Medium_Message_Boundary;
int CMI_VMI_RDMA_Chunk_Count;
int CMI_VMI_RDMA_Chunk_Size;



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



#if CMI_VMI_USE_MEMORY_POOL
PVMI_BUFFER_POOL CMI_VMI_Bucket1_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket2_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket3_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket4_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket5_Pool;
#endif


#if CMK_PERSISTENT_COMM
int CMI_VMI_Persistent_Request_Handler_ID;
int CMI_VMI_Persistent_Grant_Handler_ID;
int CMI_VMI_Persistent_Destroy_Handler_ID;

CMI_VMI_Handle_T *CMI_VMI_Persistent_Handles;
int CMI_VMI_Persistent_Handles_Size;
#endif   /* CMK_PERSISTENT_COMM */



/* The following are for the NCSA CRM code which we currently use. */
char *CRMHost;
int CRMPort;




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
int CMI_VMI_CRM_Register (char *key, int numProcesses, BOOLEAN reg)
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


  DEBUG_PRINT ("CMI_VMI_CRM_Register() called.\n");

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
      (&CMI_VMI_Processes[i])->node_IP = nodeIP;
      (&CMI_VMI_Processes[i])->rank    = processRank;
    }
  }

  /* Free the message from CRM. */
  free (msg->msg.CRMCtx.node);
  free (msg);

  /* Synchronized successfully with the CRM. */
  return myRank;
}








/**************************************************************************
**
*/
int CMI_VMI_Startup_CRM (char *key)
{
  VMI_STATUS status;

  int i;
  char *a;
  char *vmi_key;
  char *vmi_inlined_data_size;
  char *initialization_key;


  /*
    STEP 1 - Synchronize with all processes.

    In this step, we use the CRM to synchronize with all processes.
    The synchronization key used for this step is taken from the VMI_KEY
    environment variable or from argv[0] if VMI_KEY is not defined.
  */

  /* Read VMI_PROCS environment variable for the number of processes. */
  a = getenv ("VMI_PROCS");
  if (!a) {
    DEBUG_PRINT ("Environment variable VMI_PROCS is not set.");
    return (-1);
  }
  _Cmi_numpes = atoi (a);

  CMI_VMI_Processes = (CMI_VMI_Process_T *) malloc (_Cmi_numpes *
			             sizeof (CMI_VMI_Process_T));
  if (!CMI_VMI_Processes) {
    DEBUG_PRINT ("Unable to allocate memory for processes.");
    return (-1);
  }
  for (i = 0; i < _Cmi_numpes; i++) {
    (&CMI_VMI_Processes[i])->state = CMI_VMI_CONNECTION_DISCONNECTED;
  }

  /* Initialize the CRM. */
  if (!CRMInit ()) {
    DEBUG_PRINT ("Failed to initialize CRM.");
    return (-1);
  }

  /* Register with the CRM. */
  _Cmi_mype = CMI_VMI_CRM_Register (key, _Cmi_numpes, TRUE);
  if (_Cmi_mype < 0) {
    DEBUG_PRINT ("Unable to synchronize with the CRM.");
    return (-1);
  }

  DEBUG_PRINT ("This process is rank %d of %d processes.\n", _Cmi_mype,
	                                                     _Cmi_numpes);


  /*
    STEP 2 - Initialize VMI.

    In this step, we initialize VMI.  At this point we know that all
    processes for the computation are present.  We need a unique VMI key
    for each process, so we use "[syncronization key]:[process rank]" for
    each processes's key.
  */

  a = getenv ("CMI_VMI_SMALL_MESSAGE_BOUNDARY");
  if (a) {
    CMI_VMI_Small_Message_Boundary = atoi (a);
  } else {
    CMI_VMI_Small_Message_Boundary = CMI_VMI_SMALL_MESSAGE_BOUNDARY;
  }

  a = getenv ("CMI_VMI_MEDIUM_MESSAGE_BOUNDARY");
  if (a) {
    CMI_VMI_Medium_Message_Boundary = atoi (a);
  } else {
    CMI_VMI_Medium_Message_Boundary = CMI_VMI_MEDIUM_MESSAGE_BOUNDARY;
  }

  a = getenv ("CMI_VMI_RDMA_CHUNK_COUNT");
  if (a) {
    CMI_VMI_RDMA_Chunk_Count = atoi (a);
    if (CMI_VMI_RDMA_Chunk_Count > CMI_VMI_RDMA_CHUNK_COUNT) {
      CMI_VMI_RDMA_Chunk_Count = CMI_VMI_RDMA_CHUNK_COUNT;
    }
  } else {
    CMI_VMI_RDMA_Chunk_Count = CMI_VMI_RDMA_CHUNK_COUNT;
  }

  a = getenv ("CMI_VMI_RDMA_CHUNK_SIZE");
  if (a) {
    CMI_VMI_RDMA_Chunk_Size = atoi (a);
  } else {
    CMI_VMI_RDMA_Chunk_Size = CMI_VMI_RDMA_CHUNK_SIZE;
  }

  /* Set the VMI_KEY environment variable. */
  vmi_key = (char *) malloc (strlen (key) + 32);
  if (!vmi_key) {
    DEBUG_PRINT ("Unable to allocate memory for VMI key.");
    return (-1);
  }

  sprintf (vmi_key, "VMI_KEY=%s:%d\0", key, _Cmi_mype);

  if (putenv (vmi_key) == -1) {
    DEBUG_PRINT ("Unable to set VMI_KEY environment variable.");
    return (-1);
  }

  /* Set the maximum size of inlined stream messages. */
  vmi_inlined_data_size = (char *) malloc (32);
  if (!vmi_inlined_data_size) {
    DEBUG_PRINT ("Unable to allocate memory for VMI inlined data size.");
    return (-1);
  }

  sprintf (vmi_inlined_data_size, "VMI_INLINED_DATA_SZ=%d\0",
	   CMI_VMI_Medium_Message_Boundary);

  if (putenv (vmi_inlined_data_size) == -1) {
    DEBUG_PRINT ("Unable to set VMI_INLINED_DATA_SZ environment variable.");
    return (-1);
  }

  DEBUG_PRINT ("Initializing VMI with key %s.\n", vmi_key);

  /* Initialize VMI. */
  status = VMI_Init (0, NULL);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Init()");

#if CMI_VMI_USE_MEMORY_POOL
  /* Create buffer pools. */
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
#endif   /* CMI_VMI_USE_MEMORY_POOL */

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
    STEP 3 - Re-synchronize with all processes.

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

  /* Prepare the initialization key for re-synchronization with the CRM. */
  initialization_key = (char *) malloc (strlen (key) + 13);
  if (!initialization_key) {
    DEBUG_PRINT ("Unable to allocate space for initialization key.");
    return (-1);
  }
  sprintf (initialization_key, "%s:Initialized\0", key);

  /* Re-register with the CRM. */
  if (CMI_VMI_CRM_Register (initialization_key, _Cmi_numpes, FALSE) < -1) {
    DEBUG_PRINT ("Unable to re-synchronize with all processes.");
    return (-1);
  }

  DEBUG_PRINT ("Successfully re-synchronized with initialized processes.\n");


  /* Free memory. */
  free (vmi_key);
  free (initialization_key);
  free (vmi_inlined_data_size);

  /* Return successfully. */
  return (0);
}




/**************************************************************************
** This function is invoked asynchronously to handle an incoming connection
** request.
*/
VMI_CONNECT_RESPONSE
CMI_VMI_Connection_Accept_Handler (PVMI_CONNECT connection, PVMI_SLAB slab,
				   ULONG  data_size)
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
  if (data_size != size) {
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
  (&CMI_VMI_Processes[rank])->connection = connection;
  (&CMI_VMI_Processes[rank])->state = CMI_VMI_CONNECTION_CONNECTED;

  VMI_CONNECT_SET_RECEIVE_CONTEXT (connection, (&CMI_VMI_Processes[rank]));

  status = VMI_RDMA_Set_Publish_Callback (connection,
					  CMI_VMI_RDMA_Publish_Handler);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Publish_Callback()");

  status = VMI_RDMA_Set_Notification_Callback (connection,
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
					  USHORT size, PVOID handle,
					  VMI_CONNECT_RESPONSE status)
{
  CMI_VMI_Process_T *process;


  DEBUG_PRINT ("CMI_VMI_Connection_Response_Handler() called.\n");

  /* Cast the context to a CMI_VMI_Process_Info pointer. */
  process = (CMI_VMI_Process_T *) context;

  switch (status)
  {
    case VMI_CONNECT_RESPONSE_ACCEPT:
      DEBUG_PRINT ("Process %d accepted connection.\n", process->rank);

      /* Update the connection state. */
      process->state = CMI_VMI_CONNECTION_CONNECTED;

      VMI_CONNECT_SET_RECEIVE_CONTEXT (process->connection, process);

      status = VMI_RDMA_Set_Publish_Callback (process->connection,
					      CMI_VMI_RDMA_Publish_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Publish_Callback()");

      status = VMI_RDMA_Set_Notification_Callback (process->connection,
				    CMI_VMI_RDMA_Notification_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Notification_Callback()");

      /* Increment the count of outgoing connection accepts. */
      CMI_VMI_OAccept++;

      break;

    case VMI_CONNECT_RESPONSE_REJECT:
      DEBUG_PRINT ("Process %d rejected connection.\n", process->rank);

      /* Update the connection state. */
      process->state = CMI_VMI_CONNECTION_DISCONNECTED;

      /* Increment the count of outgoing connection rejects. */
      CMI_VMI_OReject++;

      break;

    case VMI_CONNECT_RESPONSE_ERROR:
      DEBUG_PRINT ("Error connecting to process %d [%d.%d.%d.%d].\n",
		   process->rank,
		   (process->node_IP >>  0) & 0xFF,
		   (process->node_IP >>  8) & 0xFF,
		   (process->node_IP >> 16) & 0xFF,
		   (process->node_IP >> 24) & 0xFF);

      /* Update the connection state. */
      process->state = CMI_VMI_CONNECTION_ERROR;

      /* Increment the count of outgoing connection errors. */
      CMI_VMI_OError++;

      break;

    default:
      DEBUG_PRINT ("Error connecting to process %d\n", process->rank);
      DEBUG_PRINT ("Error code 0x%08x\n", status);

      /* Update the connection state. */
      process->state = CMI_VMI_CONNECTION_ERROR;

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
void CMI_VMI_Connection_Disconnect_Handler (PVMI_CONNECT connection)
{
  CMI_VMI_Process_T *process;


  DEBUG_PRINT ("CMI_VMI_Connection_Disconnect_Handler() called.\n");

  process = (CMI_VMI_Process_T *) VMI_CONNECT_GET_RECEIVE_CONTEXT (connection);
  process->state = CMI_VMI_CONNECTION_DISCONNECTED;
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
int CMI_VMI_Open_Connections (char *key)
{
  VMI_STATUS status;

  char *username;
  char *remote_key;

  PVMI_BUFFER connect_message_buffer;
  CMI_VMI_Connect_Message_T *connect_message_data;

  int i;

  CMI_VMI_Process_T *process;
  struct hostent *remote_host;
  PVMI_NETADDRESS remote_address;

  struct timeval tp;
  long start_time;
  long now_time;


  DEBUG_PRINT ("CMI_VMI_Open_Connections() called.\n");

  /*
    Step 1 - Set up data structures
  */

  /* Get the username for this process. */
  username = (getpwuid (getuid ()))->pw_name;
  if (!username) {
    DEBUG_PRINT ("Unable to get username.\n");
    return (-1);
  }

  /* Allocate space for the remote key. */
  remote_key = malloc (strlen (key) + 32);
  if (!remote_key) {
    DEBUG_PRINT ("Unable to allocate memory for remote key.\n");
    return (-1);
  }

  /* Allocate a buffer for connection message. */
  status = VMI_Buffer_Allocate (sizeof (CMI_VMI_Connect_Message_T),
				&connect_message_buffer);
  if (!VMI_SUCCESS (status)) {
    DEBUG_PRINT ("Unable to allocate connection message buffer.\n");
    free (remote_key);
    return (-1);
  }

  /* Set up the connection message field. */
  connect_message_data = (CMI_VMI_Connect_Message_T *)
                          VMI_BUFFER_ADDRESS (connect_message_buffer);
  connect_message_data->rank = htonl (_Cmi_mype);

  CMI_VMI_OAccept = 0;
  CMI_VMI_OReject = 0;
  CMI_VMI_OError  = 0;
  CMI_VMI_OIssue  = 0;

  CMI_VMI_IAccept  = 0;
  CMI_VMI_IReject  = 0;
  CMI_VMI_IError   = 0;
  CMI_VMI_IExpect  = 0;


  /*
    Step 2 - Initiate connections.

    Here we initiate a connection to each process with a rank lower than
    this process's rank.
  */
  for (i = 0; i < _Cmi_mype; i++) {
    /* Get a pointer to the process to make things easier. */
    process = &CMI_VMI_Processes[i];

    /* Allocate a connection object */
    status = VMI_Connection_Create (&process->connection);
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("Unable to create connection object for process %d.\n", i);
      free (remote_key);
      VMI_Buffer_Deallocate (connect_message_buffer);
      return (-1);
    }

    /* Build the remote IPV4 address. We need remote hosts name for this. */
    remote_host = gethostbyaddr (&process->node_IP, sizeof (process->node_IP),
			         AF_INET);
    if (!remote_host) {
      DEBUG_PRINT ("Error looking up host [%d.%d.%d.%d].\n",
		   (process->node_IP >>  0) & 0xFF,
		   (process->node_IP >>  8) & 0xFF,
		   (process->node_IP >> 16) & 0xFF,
		   (process->node_IP >> 24) & 0xFF);
      free (remote_key);
      VMI_Buffer_Deallocate (connect_message_buffer);
      return (-1);
    }

    /* Construct a remote VMI key in terms of our progKey and peer's rank */
    sprintf (remote_key, "%s:%u\0", key, process->rank);

    /* Allocate a remote IPv4 NETADDRESS. */
    status = VMI_Connection_Allocate_IPV4_Address (remote_host->h_name, 0,
						   username, remote_key,
						   &remote_address);
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("Unable to allocate remote node IP V4 address.\n");
      DEBUG_PRINT ("Error 0x%08x.\n", status);
      free (remote_key);
      VMI_Buffer_Deallocate (connect_message_buffer);
      return (-1);
    }

    /* Now bind the local and remote addresses. */
    status = VMI_Connection_Bind (*localAddress, *remote_address,
				  process->connection);
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("Error binding connection for process %d.\n", i);
      free (remote_key);
      VMI_Buffer_Deallocate (connect_message_buffer);
      return (-1);
    }

    /*
      Do this here to avoid a race condition where we complete the connect
      right away and then set the state here to
      CMI_VMI_CONNECTION_STATE_CONNECTING.
    */
    process->state = CMI_VMI_CONNECTION_CONNECTING;

    /* Issue the actual connection request. */
    status = VMI_Connection_Issue (process->connection, connect_message_buffer,
         (VMIConnectIssue) CMI_VMI_Connection_Response_Handler, process);
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("Error issuing connection for process %d.\n", i);
      free (remote_key);
      VMI_Buffer_Deallocate (connect_message_buffer);
      return (-1);
    }

    /* Increment number of issued connections. */
    CMI_VMI_OIssue++;

    DEBUG_PRINT ("Issued a connection to process %d:\n", i);
    DEBUG_PRINT ("\tRank - %d\n", process->rank);
    DEBUG_PRINT ("\tIP - [%d.%d.%d.%d].\n", (process->node_IP >>  0) & 0xFF,
		                            (process->node_IP >>  8) & 0xFF,
		                            (process->node_IP >> 16) & 0xFF,
		                            (process->node_IP >> 24) & 0xFF);
    DEBUG_PRINT ("\tHostname - %s\n", remote_host->h_name);
    DEBUG_PRINT ("\tKey - %s\n", remote_key);
  }

  /* Set the connection state to ourself to "connected". */
  (&CMI_VMI_Processes[_Cmi_mype])->state = CMI_VMI_CONNECTION_CONNECTED;


  /*
    Step 3 - Wait for connections.

    Now wait for all outgoing connections to complete and for all
    incoming connections to arrive.
  */

  /* Calculate how many pprocesses are supposed to connect to us. */
  CMI_VMI_IExpect = ((_Cmi_numpes - _Cmi_mype) - 1);

  DEBUG_PRINT ("The rank of this process is %d.\n", _Cmi_mype);
  DEBUG_PRINT ("This process issued %d connection requests.\n",
	       CMI_VMI_OIssue);
  DEBUG_PRINT ("This process is expecting %d connections to arrive.\n",
	       CMI_VMI_IExpect);

  /* Complete all connection requests and accepts. */
  gettimeofday (&tp, NULL);
  start_time = tp.tv_sec;
  now_time   = tp.tv_sec;
  while ( (((CMI_VMI_OAccept+CMI_VMI_OReject+CMI_VMI_OError)<CMI_VMI_OIssue) ||
	   ((CMI_VMI_IAccept+CMI_VMI_IReject+CMI_VMI_IError)<CMI_VMI_IExpect))
         &&
	 ((start_time + CMI_VMI_CONNECTION_TIMEOUT) > now_time)) {
    sched_yield ();
    status = VMI_Poll ();
    if (!VMI_SUCCESS (status)) {
      DEBUG_PRINT ("VMI_Poll() failed while waiting for connections.\n");
      DEBUG_PRINT ("Error 0x%08x\n", status);
      return (-1);
    }
    gettimeofday (&tp, NULL);
    now_time = tp.tv_sec;
  }

  /*
    Step 4 - Verify that there were no connection problems.
  */
  if ( (CMI_VMI_OReject > 0) || (CMI_VMI_OError > 0) ||
       (CMI_VMI_IReject > 0) || (CMI_VMI_IError > 0) ) {
    DEBUG_PRINT ("%d outgoing connections were rejected.\n", CMI_VMI_OReject);
    DEBUG_PRINT ("%d outgoing connections had errors.\n", CMI_VMI_OError);
    DEBUG_PRINT ("%d incoming connections were rejected.\n", CMI_VMI_IReject);
    DEBUG_PRINT ("%d incoming connections had errors.\n", CMI_VMI_IError);

    free (remote_key);
    VMI_Buffer_Deallocate (connect_message_buffer);

    return (-1);
  }

  DEBUG_PRINT ("All connections are complete for process %d.\n", _Cmi_mype);

  free (remote_key);
  VMI_Buffer_Deallocate (connect_message_buffer);

  /* Successfully set up connections. */
  return (0);
}




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


  DEBUG_PRINT ("CMI_VMI_Spanning_Children_Count() called.\n");

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

  CMI_VMI_Handle_T *handle;

  int i;

  PVMI_CACHE_ENTRY cacheentry;

  int childcount;
  int startrank;
  int destrank;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;


  DEBUG_PRINT ("CMI_VMI_Send_Spanning_Children() called.\n");

  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    handle = CMI_VMI_Allocate_Handle ();

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.free_message = FALSE;
    handle->data.send.data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    if (childcount == 0) {
      return;
    }

    handle->refcount += childcount;
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

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) &handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    while (handle->refcount > 2) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    status = VMI_Cache_Deregister (cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

    handle->refcount = 0;
  } else {
    handle = CMI_VMI_Allocate_Handle ();

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMABROAD;
    handle->data.send.free_message = FALSE;
    handle->data.send.data.rdmabroad.chunk_size = CMI_VMI_RDMA_Chunk_Size;
    handle->data.send.data.rdmabroad.bytes_sent = CmiAlloc (_Cmi_numpes *
							    sizeof (int));
    handle->data.send.data.rdmabroad.cacheentry = CmiAlloc (_Cmi_numpes *
					       sizeof (PVMI_CACHE_ENTRY));
    for (i = 0; i < _Cmi_numpes; i++) {
      handle->data.send.data.rdmabroad.bytes_sent[i] = 0;
      handle->data.send.data.rdmabroad.cacheentry[i] = NULL;
    }

    CMI_VMI_MESSAGE_TYPE (&rendezvous_msg) = CMI_VMI_MESSAGE_TYPE_RENDEZVOUS;
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) handle;

    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    if (childcount == 0) {
      return;
    }

    handle->refcount += childcount;
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

      status=VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1,
				    sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    while (handle->refcount > 2) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    CmiFree (handle->data.send.data.rdmabroad.bytes_sent);
    CmiFree (handle->data.send.data.rdmabroad.cacheentry);

    handle->refcount = 0;
  }
}
#endif   /* CMK_BROADCAST_SPANNING_TREE */



/**************************************************************************
** This function is invoked asynchronously to handle an incoming message
** receive on a stream.
**
** This function is on the receive side.
*/
VMI_RECV_STATUS CMI_VMI_Stream_Receive_Handler (PVMI_CONNECT connection,
						PVMI_STREAM_RECV stream,
						VMI_STREAM_COMMAND command,
						PVOID context,
						PVMI_SLAB slab)
{
  VMI_STATUS status;

  ULONG size;
  char *msg;
  PVMI_SLAB_STATE state;

  PVMI_CACHE_ENTRY cacheentry;
  int i;
  int rank;
  int msgsize;
  VMI_virt_addr_t remote_handle_address;
  PUCHAR pubaddr;
  int pubsize;
  char *msg2;

  CMI_VMI_Handle_T *handle;


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

  if (CMI_VMI_MESSAGE_TYPE (msg) == CMI_VMI_MESSAGE_TYPE_STANDARD) {
#if CMK_BROADCAST_SPANNING_TREE
    /* Send the message to our spanning children (if any). */
    if (CMI_BROADCAST_ROOT (msg)) {
      CMI_VMI_Send_Spanning_Children (size, msg);
    }
#endif

    /* Enqueue the message into the remote queue. */
    CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), msg);
  } else {
    rank = ((CMI_VMI_Rendezvous_Message_T *) msg)->rank;
    msgsize = ((CMI_VMI_Rendezvous_Message_T *) msg)->msgsize;
    remote_handle_address = ((CMI_VMI_Rendezvous_Message_T *) msg)->context;

    CmiFree (msg);

    msg2 = (char *) CmiAlloc (msgsize);

    handle = CMI_VMI_Allocate_Handle ();

    handle->msg = msg2;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_RECEIVE;
    handle->data.receive.receive_handle_type=CMI_VMI_RECEIVE_HANDLE_TYPE_RDMA;
    handle->data.receive.data.rdma.bytes_published = 0;
    handle->data.receive.data.rdma.bytes_received = 0;
    handle->data.receive.data.rdma.chunk_size = CMI_VMI_RDMA_Chunk_Size;
    handle->data.receive.data.rdma.chunk_count = CMI_VMI_RDMA_Chunk_Count;
    handle->data.receive.data.rdma.chunks_outstanding = 0;
    handle->data.receive.data.rdma.send_index = 0;
    handle->data.receive.data.rdma.receive_index = 0;
    handle->data.receive.data.rdma.remote_handle_address =
                                                      remote_handle_address;

    /*
      This is the constant CMI_VMI_RDMA_CHUNK_COUNT because the cacheentry
      array is allocated statically within the structure.
    */
    for (i = 0; i < CMI_VMI_RDMA_CHUNK_COUNT; i++) {
      handle->data.receive.data.rdma.cacheentry[i] = NULL;
    }

    while ((handle->data.receive.data.rdma.bytes_published < handle->msgsize)
       &&  (handle->data.receive.data.rdma.chunks_outstanding <
	    handle->data.receive.data.rdma.chunk_count)) {
      pubaddr = handle->msg + handle->data.receive.data.rdma.bytes_published;
      pubsize = handle->msgsize-handle->data.receive.data.rdma.bytes_published;

      if (pubsize > handle->data.receive.data.rdma.chunk_size) {
	pubsize = handle->data.receive.data.rdma.chunk_size;
      }

      status = VMI_Cache_Register (pubaddr, pubsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

      handle->data.receive.data.rdma.cacheentry[handle->data.receive.data.rdma.send_index] = cacheentry;

      handle->data.receive.data.rdma.send_index++;
      if (handle->data.receive.data.rdma.send_index >=
	  handle->data.receive.data.rdma.chunk_count) {
	handle->data.receive.data.rdma.send_index = 0;
      }

      handle->data.receive.data.rdma.chunks_outstanding++;

      status = VMI_RDMA_Publish_Buffer (connection,
           cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) pubaddr,
           pubsize, remote_handle_address, (UINT32) handle->index);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");

      handle->data.receive.data.rdma.bytes_published += pubsize;
    }
  }

  /* Tell VMI that the slab can be discarded. */
  return (VMI_SLAB_DONE);
}


/**************************************************************************
** This function is invoked asynchronously to handle the completion of a
** send on a stream.
**
** This function is on the send side.
*/
void CMI_VMI_Stream_Completion_Handler (PVOID context, VMI_STATUS sstatus)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;


  DEBUG_PRINT ("CMI_VMI_Stream_Completion_Handler() called.\n");

  handle = (CMI_VMI_Handle_T *) context;

  CMI_VMI_AsyncMsgCount--;
  handle->refcount--;

  if (handle->refcount <= 1) {
    status = VMI_Cache_Deregister (handle->data.send.data.stream.cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

    if (handle->data.send.free_message) {
      CmiFree (handle->msg);
    }

    handle->refcount = 0;
  }
}



/**************************************************************************
** This function is invoked asynchronously to handle the completion of an
** RDMA Put for a fragment of a message (i.e., there is at least one more
** fragment left in the message after this one).
**
** This function is on the send side.
*/
void CMI_VMI_RDMA_Fragment_Handler (PVMI_RDMA_OP op, PVOID context,
				    VMI_STATUS rstatus)
{
  VMI_STATUS status;

  PVMI_CACHE_ENTRY cacheentry;


  DEBUG_PRINT ("CMI_VMI_RDMA_Fragment_Handler() called.\n");

  /* Cast the context to a cache entry. */
  cacheentry = (PVMI_CACHE_ENTRY) context;

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
void CMI_VMI_RDMA_Completion_Handler (PVMI_RDMA_OP op, PVOID context,
				      VMI_STATUS rstatus)
{
  VMI_STATUS status;

  int i;

  PVMI_CACHE_ENTRY cacheentry;
  CMI_VMI_Handle_T *handle;


  DEBUG_PRINT ("CMI_VMI_RDMA_Completion_Handler() called.\n");

  handle = (CMI_VMI_Handle_T *) context;

  /* Deallocate the RDMA op's buffer, the RDMA op, and the cache entry. */
  status = VMI_RDMA_Dealloc_Buffer (op->rbuffer);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Buffer()");

  status = VMI_RDMA_Dealloc_Op (op);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Op()");

  if (handle->data.send.send_handle_type == CMI_VMI_SEND_HANDLE_TYPE_RDMA) {
    status = VMI_Cache_Deregister (handle->data.send.data.rdma.cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
  }

  if (handle->data.send.send_handle_type ==
                                      CMI_VMI_SEND_HANDLE_TYPE_PERSISTENT) {
    status=VMI_Cache_Deregister(handle->data.send.data.persistent.cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

    handle->msg = NULL;
    handle->msgsize = -1;
  }

  /*
    We cannot de-register the cache entry for RDMA broadcasts because we
    have no way to determine the rank of the process that just completed.
    The best we can do is de-register all of the cache entries when the
    handle for the RDMA broadcast is released.
  */

  CMI_VMI_AsyncMsgCount--;
  handle->refcount--;

  if (handle->refcount <= 1) {
    if (handle->data.send.send_handle_type ==
	CMI_VMI_SEND_HANDLE_TYPE_RDMABROAD){

      for (i = 0; i < _Cmi_numpes; i++) {
	if (handle->data.send.data.rdmabroad.cacheentry[i]) {
	  status = VMI_Cache_Deregister
	            (handle->data.send.data.rdmabroad.cacheentry[i]);
	  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	}
      }

      CmiFree (handle->data.send.data.rdmabroad.bytes_sent);
      CmiFree (handle->data.send.data.rdmabroad.cacheentry);
    }

    if (handle->data.send.free_message) {
      CmiFree (handle->msg);
    }

    handle->refcount = 0;
  }
}




/**************************************************************************
** This function is invoked asynchronously to handle an RDMA publish of a
** buffer from a remote process.
**
** This function is on the send side.
*/
void CMI_VMI_RDMA_Publish_Handler (PVMI_CONNECT connection,
				   PVMI_REMOTE_BUFFER remote_buffer)
{
  VMI_STATUS status;

  PVMI_CACHE_ENTRY cacheentry;
  BOOLEAN complete_flag;
  CMI_VMI_Handle_T *handle;
  char *putaddr;
  int putlen;
  PVMI_RDMA_OP rdmaop;
  CMI_VMI_Process_T *process;
  int rank;


  DEBUG_PRINT ("CMI_VMI_RDMA_Publish_Handler() called.\n");

  /* Cast the remote buffer's local context to a send handle. */
  handle = (CMI_VMI_Handle_T *) (VMI_ADDR_CAST) remote_buffer->lctxt;

  if (handle->data.send.send_handle_type == CMI_VMI_SEND_HANDLE_TYPE_RDMA) {
    putaddr = handle->msg + handle->data.send.data.rdma.bytes_sent;
    putlen = handle->msgsize - handle->data.send.data.rdma.bytes_sent;

    if (putlen > handle->data.send.data.rdma.chunk_size) {
      putlen = handle->data.send.data.rdma.chunk_size;
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
    rdmaop->rbuffer = remote_buffer;
    rdmaop->roffset = 0;

    handle->data.send.data.rdma.bytes_sent += putlen;

    if (complete_flag) {
      handle->data.send.data.rdma.cacheentry = cacheentry;

      status = VMI_RDMA_Put (connection, rdmaop, (PVOID) handle,
	   (VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    } else {
      status = VMI_RDMA_Put (connection, rdmaop, (PVOID) cacheentry, 
	   (VMIRDMAWriteComplete) CMI_VMI_RDMA_Fragment_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    }
  } else if (handle->data.send.send_handle_type ==
	     CMI_VMI_SEND_HANDLE_TYPE_RDMABROAD) {
    process =
          (CMI_VMI_Process_T *) VMI_CONNECT_GET_RECEIVE_CONTEXT (connection);
    rank = process->rank;

    putaddr = handle->msg + handle->data.send.data.rdmabroad.bytes_sent[rank];
    putlen = handle->msgsize-handle->data.send.data.rdmabroad.bytes_sent[rank];

    if (putlen > handle->data.send.data.rdmabroad.chunk_size) {
      putlen = handle->data.send.data.rdmabroad.chunk_size;
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
    rdmaop->rbuffer = remote_buffer;
    rdmaop->roffset = 0;

    handle->data.send.data.rdmabroad.bytes_sent[rank] += putlen;

    if (complete_flag) {
      handle->data.send.data.rdmabroad.cacheentry[rank] = cacheentry;

      status = VMI_RDMA_Put (connection, rdmaop, (PVOID) handle,
	   (VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    } else {
      status = VMI_RDMA_Put (connection, rdmaop, (PVOID) cacheentry,
	   (VMIRDMAWriteComplete) CMI_VMI_RDMA_Fragment_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    }
  } else {
#if CMK_PERSISTENT_COMM
    handle->data.send.data.persistent.remote_buffer = remote_buffer;
    handle->data.send.data.persistent.ready++;
#else
    CmiAbort ("CMI_VMI_RDMA_Publish_Handler() got bad handle type.");
#endif
  }
}



/**************************************************************************
** This function is invoked asynchronously to handle the completion of an
** RDMA Put from a remote process.
**
** This function is on the receive side.
*/
void CMI_VMI_RDMA_Notification_Handler (PVMI_CONNECT connection,
					UINT32 rdma_size,
					UINT32 context,
					VMI_STATUS remote_status)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;

  PVMI_CACHE_ENTRY cacheentry;
  char *pubaddr;
  int pubsize;


  DEBUG_PRINT ("CMI_VMI_RDMA_Notification_Handler() called.\n");

  handle = &(CMI_VMI_Handles[context]);

  if (handle->data.receive.receive_handle_type ==
      CMI_VMI_RECEIVE_HANDLE_TYPE_RDMA) {
    handle->data.receive.data.rdma.bytes_received += rdma_size;
    handle->data.receive.data.rdma.chunks_outstanding--;
    cacheentry = handle->data.receive.data.rdma.cacheentry[handle->data.receive.data.rdma.receive_index];
    status = VMI_Cache_Deregister (cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

    handle->data.receive.data.rdma.receive_index++;
    if (handle->data.receive.data.rdma.receive_index >=
	handle->data.receive.data.rdma.chunk_count) {
      handle->data.receive.data.rdma.receive_index = 0;
    }

    if (handle->data.receive.data.rdma.bytes_published < handle->msgsize) {
      pubaddr = handle->msg + handle->data.receive.data.rdma.bytes_published;
      pubsize = handle->msgsize-handle->data.receive.data.rdma.bytes_published;

      if (pubsize > handle->data.receive.data.rdma.chunk_size) {
	pubsize = handle->data.receive.data.rdma.chunk_size;
      }

      status = VMI_Cache_Register (pubaddr, pubsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

      handle->data.receive.data.rdma.cacheentry[handle->data.receive.data.rdma.send_index] = cacheentry;

      handle->data.receive.data.rdma.send_index++;
      if (handle->data.receive.data.rdma.send_index >=
	  handle->data.receive.data.rdma.chunk_count) {
	handle->data.receive.data.rdma.send_index = 0;
      }

      handle->data.receive.data.rdma.chunks_outstanding++;

      status = VMI_RDMA_Publish_Buffer (connection, cacheentry->bufferHandle,
                          (VMI_virt_addr_t) (VMI_ADDR_CAST) pubaddr, pubsize,
	                handle->data.receive.data.rdma.remote_handle_address,
                                                     (UINT32) handle->index);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");

      handle->data.receive.data.rdma.bytes_published += pubsize;
    }

    if (handle->data.receive.data.rdma.bytes_received >= handle->msgsize) {
#if CMK_BROADCAST_SPANNING_TREE
      if (CMI_BROADCAST_ROOT (handle->msg)) {
	CMI_VMI_Send_Spanning_Children (handle->msgsize, handle->msg);
      }
#endif

      CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), handle->msg);

      handle->refcount = 0;
    }
#if CMK_PERSISTENT_COMM
  } else if (handle->data.receive.receive_handle_type ==
	     CMI_VMI_RECEIVE_HANDLE_TYPE_PERSISTENT) {
    cacheentry = handle->data.receive.data.persistent.cacheentry;
    status = VMI_Cache_Deregister (cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

    CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), handle->msg);

    handle->msg = (char *) CmiAlloc (handle->msgsize);

    status = VMI_Cache_Register (handle->msg, handle->msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->data.receive.data.persistent.cacheentry = cacheentry;

    status = VMI_RDMA_Publish_Buffer (connection, cacheentry->bufferHandle,
	    (VMI_virt_addr_t) (VMI_ADDR_CAST) handle->msg, handle->msgsize,
                handle->data.receive.data.persistent.remote_handle_address,
				                   (UINT32) handle->index);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
  } else {
    CmiAbort ("CMI_VMI_RDMA_Notification_Handler() got bad handle type.");
  }
#else   /* CMK_PERSISTENT_COMM */
  } else {
    CmiAbort ("CMI_VMI_RDMA_Notification_Handler() got bad handle type.");
  }
#endif   /* CMK_PERSISTENT_COMM */
}





/**************************************************************************
** argc
** argv
** start_function - the user-supplied function to run (function pointer)
** user_calls_scheduler - boolean for whether ConverseInit() should invoke
**                        the scheduler or whether user code will do it
** init_returns - boolean for whether ConverseInit() returns
*/
void ConverseInit (int argc, char **argv, CmiStartFn start_function,
		   int user_calls_scheduler, int init_returns)
{
  int rc;
  int i;
  int startup_type;
  char *a;
  char *key;


  DEBUG_PRINT ("ConverseInit() called.\n");

  /* Initialize the global asynchronous message count. */
  CMI_VMI_AsyncMsgCount = 0;

  /* Initialize the maximum number of send and receive handles. */
  a = getenv ("CMI_VMI_MAXIMUM_HANDLES");
  if (a) {
    CMI_VMI_Maximum_Handles = atoi (a);
  } else {
    CMI_VMI_Maximum_Handles = CMI_VMI_MAXIMUM_HANDLES;
  }
  CMI_VMI_Handles = (CMI_VMI_Handle_T *) malloc (CMI_VMI_Maximum_Handles *
	       			               sizeof (CMI_VMI_Handle_T));
  if (!CMI_VMI_Handles) {
    CmiAbort ("Unable to allocate memory for send and receive handles.");
  }
  for (i = 0; i < CMI_VMI_Maximum_Handles; i++) {
    (&CMI_VMI_Handles[i])->index = i;
    (&CMI_VMI_Handles[i])->refcount = 0;
  }
  CMI_VMI_Next_Handle = 0;

  /* Get the program key for this process. */
  a = getenv ("VMI_KEY");
  if (a) {
    key = strdup (a);
  } else {
    key = (char *) malloc (strlen (argv[0]) + 1);
    if (!key) {
      CmiAbort ("Unable to allocate memory for program key.");
    }
    sprintf (key, "%s\0", argv[0]);
  }

  DEBUG_PRINT ("The program key is %s.\n", key);

  /* Get the startup type. */
  a = getenv ("CMI_VMI_STARTUP_TYPE");
  if (a) {
    startup_type = atoi (a);
  } else {
    startup_type = CMI_VMI_STARTUP_TYPE_CRM;
  }

  DEBUG_PRINT ("The startup type is %d.\n", startup_type);

  /* Start up via the startup type selected. */
  switch (startup_type) {

    case CMI_VMI_STARTUP_TYPE_CRM:
      rc = CMI_VMI_Startup_CRM (key);
      break;

    default:
      CmiAbort ("An unknown startup type was specified.");
      break;
  }

  if (rc < 0) {
    CmiAbort ("Fatal error during startup phase.");
  }

  /* Open connections. */
  if (CMI_VMI_Open_Connections (key) < 0) {
    CmiAbort ("Fatal error during connection setup phase.");
  }

  /* Free memory. */
  free (key);

  /* Create the FIFOs for holding local and remote messages. */
  CpvAccess (CmiLocalQueue) = CdsFifo_Create ();
  CpvAccess (CMI_VMI_RemoteQueue) = CdsFifo_Create ();

  DEBUG_PRINT ("ConverseInit() is starting the main processing loop.\n");

  /* Initialize Converse and start the main processing loop. */
  CthInit (argv);
  ConverseCommonInit (argv);

  if (!init_returns) {
    start_function (CmiGetArgc (argv), argv);
    if (!user_calls_scheduler) {
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


  DEBUG_PRINT ("ConverseExit() called.\n");

  /* Should close all VMI connections here. */
  /* There is a race condition here because not all processes may have
     entered ConverseExit() simultaneously. */

#if CMI_VMI_USE_MEMORY_POOL
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
#endif   /* CMI_VMI_USE_MEMORY_POOL */

  /* Free memory. */
  free (CMI_VMI_Processes);
  free (CMI_VMI_Handles);

  /* Destroy queues. */
  CdsFifo_Destroy (CpvAccess (CMI_VMI_RemoteQueue));
  CdsFifo_Destroy (CpvAccess (CmiLocalQueue));

  /* VMI will not terminate properly while there are open connections. */
  exit (0);

  /* Terminate VMI. */
  SET_VMI_SUCCESS (status);
  VMI_Terminate (status);
}




/**************************************************************************
** done
*/
void CmiAbort (const char *message)
{
  DEBUG_PRINT ("CmiAbort() called.\n");

  printf ("%s\n", message);
  exit (1);
}



/**************************************************************************
** done
*/
void *CmiGetNonLocal (void)
{
  VMI_STATUS status;


  status = VMI_Poll ();
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");

  return (CdsFifo_Dequeue (CpvAccess (CMI_VMI_RemoteQueue)));
}


/**************************************************************************
** done
*/
void CmiMemLock ()
{
  /* Empty. */
}



/**************************************************************************
** done
*/
void CmiMemUnlock ()
{
  /* Empty. */
}


/**************************************************************************
** done
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

  CMI_VMI_Handle_T *handle;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;

  PVMI_CACHE_ENTRY cacheentry;
  PVMI_RDMA_OP rdmaop;


  DEBUG_PRINT ("CmiSyncSendFn() called.\n");

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (msg, 0);
#endif

  if (destrank == _Cmi_mype) {
    msgcopy = CmiAlloc (msgsize);
    memcpy (msgcopy, msg, msgsize);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);
#if CMK_PERSISTENT_COMM
  /* NOTE: The next line is an assignment AND a test for non-null! */
  } else if ((handle = &CMI_VMI_Persistent_Handles[0]) &&
	     (handle->data.send.data.persistent.destrank == destrank) &&
	     (handle->data.send.data.persistent.maxsize <= msgsize) &&
	     (handle->data.send.data.persistent.ready >= 2)) {
    handle->refcount = 1;
    handle->msg = msg;
    handle->msgsize = msgsize;

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->data.send.data.persistent.cacheentry = cacheentry;

    status = VMI_RDMA_Alloc_Op (&rdmaop);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

    rdmaop->numBufs = 1;
    rdmaop->buffers[0] = cacheentry->bufferHandle;
    rdmaop->addr[0] = msg;
    rdmaop->sz[0] = msgsize;
    rdmaop->rbuffer = handle->data.send.data.persistent.remote_buffer;
    rdmaop->roffset = 0;

    handle->refcount += 2;
    CMI_VMI_AsyncMsgCount++;

    status = VMI_RDMA_Put (handle->data.send.data.persistent.connection,
			                         rdmaop, (PVOID) handle,
		(VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");

    while (handle->refcount > 2) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    handle->refcount = 1;
#endif   /* CMK_PERSISTENT_COMM */
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

    status = VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1, msgsize);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
  } else {
    handle = CMI_VMI_Allocate_Handle ();

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMA;
    handle->data.send.free_message = FALSE;
    handle->data.send.data.rdma.chunk_size = CMI_VMI_RDMA_Chunk_Size;
    handle->data.send.data.rdma.bytes_sent = 0;
    handle->data.send.data.rdma.cacheentry = NULL;

    handle->refcount += 1;
    CMI_VMI_AsyncMsgCount += 1;

    CMI_VMI_MESSAGE_TYPE (&rendezvous_msg) = CMI_VMI_MESSAGE_TYPE_RENDEZVOUS;
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

    status = VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
		          addrs, sz, 1, sizeof (CMI_VMI_Rendezvous_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

    while (handle->refcount > 2) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    handle->refcount = 0;
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

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;

  PVMI_CACHE_ENTRY cacheentry;

  PVMI_RDMA_OP rdmaop;


  DEBUG_PRINT ("CmiAsyncSendFn() called.\n");

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (msg, 0);
#endif

  if (destrank == _Cmi_mype) {
    msgcopy = CmiAlloc (msgsize);
    memcpy (msgcopy, msg, msgsize);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);

    handle = NULL;
#if CMK_PERSISTENT_COMM
  /* NOTE: The next line is an assignment AND a test for non-null! */
  } else if ((handle = &CMI_VMI_Persistent_Handles[0]) &&
	     (handle->data.send.data.persistent.destrank == destrank) &&
	     (handle->data.send.data.persistent.maxsize <= msgsize) &&
	     (handle->data.send.data.persistent.ready >= 2)) {
    handle->refcount = 1;
    handle->msg = msg;
    handle->msgsize = msgsize;

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->data.send.data.persistent.cacheentry = cacheentry;

    status = VMI_RDMA_Alloc_Op (&rdmaop);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

    rdmaop->numBufs = 1;
    rdmaop->buffers[0] = cacheentry->bufferHandle;
    rdmaop->addr[0] = msg;
    rdmaop->sz[0] = msgsize;
    rdmaop->rbuffer = handle->data.send.data.persistent.remote_buffer;
    rdmaop->roffset = 0;

    handle->refcount += 2;
    CMI_VMI_AsyncMsgCount++;

    status = VMI_RDMA_Put (handle->data.send.data.persistent.connection,
			                         rdmaop, (PVOID) handle,
		(VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
#endif   /* CMK_PERSISTENT_COMM */
  } else if (msgsize < CMI_VMI_Small_Message_Boundary) {
    addrs[0] = (PVOID) msg;
    sz[0] = msgsize;

    status = VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1, msgsize);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

    handle = NULL;
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    handle = CMI_VMI_Allocate_Handle ();

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.free_message = FALSE;
    handle->data.send.data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = msgsize;

    handle->refcount += 1;
    CMI_VMI_AsyncMsgCount++;

    status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection,
			      bufHandles, addrs, sz, 1,
			      CMI_VMI_Stream_Completion_Handler,
			      (PVOID) handle, TRUE);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
  } else {
    handle = CMI_VMI_Allocate_Handle ();

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMA;
    handle->data.send.free_message = FALSE;
    handle->data.send.data.rdma.chunk_size = CMI_VMI_RDMA_Chunk_Size;
    handle->data.send.data.rdma.bytes_sent = 0;
    handle->data.send.data.rdma.cacheentry = NULL;

    handle->refcount += 1;
    CMI_VMI_AsyncMsgCount++;

    CMI_VMI_MESSAGE_TYPE (&rendezvous_msg) = CMI_VMI_MESSAGE_TYPE_RENDEZVOUS;
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

    status = VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1,
				    sizeof (CMI_VMI_Rendezvous_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
  }

  return ((CmiCommHandle) handle);
}



/**************************************************************************
** done
*/
void CmiFreeSendFn (int destrank, int msgsize, char *msg)
{
  VMI_STATUS status;

  char *msgcopy;

  PVMI_BUFFER bufHandles[2];
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T *handle;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;

  PVMI_CACHE_ENTRY cacheentry;

  PVMI_RDMA_OP rdmaop;


  DEBUG_PRINT ("CmiFreeSendFn() called.\n");

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (msg, 0);
#endif

  if (destrank == _Cmi_mype) {
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msg);
#if CMK_PERSISTENT_COMM
  /* NOTE: The next line is an assignment AND a test for non-null! */
  } else if ((handle = &CMI_VMI_Persistent_Handles[0]) &&
	     (handle->data.send.data.persistent.destrank == destrank) &&
	     (handle->data.send.data.persistent.maxsize <= msgsize) &&
	     (handle->data.send.data.persistent.ready >= 2)) {

    // NOTE: This code really should not free any message buffers because
    // the idea behind a persistent handle is to avoid pinning and
    // unpinning the buffer in memory, right?
    //
    // THIS NEEDS TO BE THOUGHT ABOUT AND FIXED!

    handle->refcount = 1;
    handle->msg = msg;
    handle->msgsize = msgsize;

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->data.send.data.persistent.cacheentry = cacheentry;

    status = VMI_RDMA_Alloc_Op (&rdmaop);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

    rdmaop->numBufs = 1;
    rdmaop->buffers[0] = cacheentry->bufferHandle;
    rdmaop->addr[0] = msg;
    rdmaop->sz[0] = msgsize;
    rdmaop->rbuffer = handle->data.send.data.persistent.remote_buffer;
    rdmaop->roffset = 0;

    handle->refcount += 1;
    CMI_VMI_AsyncMsgCount++;

    status = VMI_RDMA_Put (handle->data.send.data.persistent.connection,
			                         rdmaop, (PVOID) handle,
		(VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
#endif   /* CMK_PERSISTENT_COMM */
  } else if (msgsize < CMI_VMI_Small_Message_Boundary) {
    addrs[0] = (PVOID) msg;
    sz[0] = msgsize;

    status = VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1, msgsize);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

    CmiFree (msg);
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    handle = CMI_VMI_Allocate_Handle ();

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    /* Do NOT increment handle->refcount here! */
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.free_message = TRUE;
    handle->data.send.data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = msgsize;

    handle->refcount += 1;
    CMI_VMI_AsyncMsgCount++;

    status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection,
			      bufHandles, addrs, sz, 1,
			      CMI_VMI_Stream_Completion_Handler,
			      (PVOID) handle, TRUE);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
  } else {
    handle = CMI_VMI_Allocate_Handle ();

    /* Do NOT increment handle->refcount here! */
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMA;
    handle->data.send.free_message = TRUE;
    handle->data.send.data.rdma.chunk_size = CMI_VMI_RDMA_Chunk_Size;
    handle->data.send.data.rdma.bytes_sent = 0;
    handle->data.send.data.rdma.cacheentry = NULL;

    handle->refcount += 1;
    CMI_VMI_AsyncMsgCount++;

    CMI_VMI_MESSAGE_TYPE (&rendezvous_msg) = CMI_VMI_MESSAGE_TYPE_RENDEZVOUS;
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

    status = VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1,
				    sizeof (CMI_VMI_Rendezvous_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
  }
}



/**************************************************************************
**
*/
void CmiSyncBroadcastFn (int msgsize, char *msg)
{
  VMI_STATUS status;

  PVMI_BUFFER bufHandles[2];
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T *handle;

  int i;

  PVMI_CACHE_ENTRY cacheentry;

  int childcount;
  int startrank;
  int destrank;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;


  DEBUG_PRINT ("CmiSyncBroadcastFn() called.\n");

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;

  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    handle = CMI_VMI_Allocate_Handle ();

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.free_message = FALSE;
    handle->data.send.data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle->refcount += childcount;
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

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */

    while (handle->refcount > 2) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    status = VMI_Cache_Deregister (cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

    handle->refcount = 0;
  } else {
    handle = CMI_VMI_Allocate_Handle ();

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMABROAD;
    handle->data.send.free_message = FALSE;
    handle->data.send.data.rdmabroad.chunk_size = CMI_VMI_RDMA_Chunk_Size;
    handle->data.send.data.rdmabroad.bytes_sent = CmiAlloc (_Cmi_numpes *
							    sizeof (int));
    handle->data.send.data.rdmabroad.cacheentry = CmiAlloc (_Cmi_numpes *
					       sizeof (PVMI_CACHE_ENTRY));
    for (i = 0; i < _Cmi_numpes; i++) {
      handle->data.send.data.rdmabroad.bytes_sent[i] = 0;
      handle->data.send.data.rdmabroad.cacheentry[i] = NULL;
    }

    CMI_VMI_MESSAGE_TYPE (&rendezvous_msg) = CMI_VMI_MESSAGE_TYPE_RENDEZVOUS;
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle->refcount += childcount;
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

      status=VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1,
				    sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1,
				       sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1,
				       sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */

    while (handle->refcount > 2) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    for (i = 0; i < _Cmi_numpes; i++) {
      if (handle->data.send.data.rdmabroad.cacheentry[i]) {
	status = VMI_Cache_Deregister
                  (handle->data.send.data.rdmabroad.cacheentry[i]);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
      }
    }

    CmiFree (handle->data.send.data.rdmabroad.bytes_sent);
    CmiFree (handle->data.send.data.rdmabroad.cacheentry);

    handle->refcount = 0;
  }
}



/**************************************************************************
** done
*/
CmiCommHandle CmiAsyncBroadcastFn (int msgsize, char *msg)
{
  VMI_STATUS status;

  PVMI_BUFFER bufHandles[2];
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T *handle;

  int i;

  PVMI_CACHE_ENTRY cacheentry;

  int childcount;
  int startrank;
  int destrank;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;


  DEBUG_PRINT ("CmiAsyncBroadcastFn() called.\n");

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;

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

      status=VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */

    handle = NULL;
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    handle = CMI_VMI_Allocate_Handle ();

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.free_message = FALSE;
    handle->data.send.data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle->refcount += childcount;
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

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */
  } else {
    handle = CMI_VMI_Allocate_Handle ();

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMABROAD;
    handle->data.send.free_message = FALSE;
    handle->data.send.data.rdmabroad.chunk_size = CMI_VMI_RDMA_Chunk_Size;
    handle->data.send.data.rdmabroad.bytes_sent = CmiAlloc (_Cmi_numpes *
							    sizeof (int));
    handle->data.send.data.rdmabroad.cacheentry = CmiAlloc (_Cmi_numpes *
					       sizeof (PVMI_CACHE_ENTRY));
    for (i = 0; i < _Cmi_numpes; i++) {
      handle->data.send.data.rdmabroad.bytes_sent[i] = 0;
      handle->data.send.data.rdmabroad.cacheentry[i] = NULL;
    }

    CMI_VMI_MESSAGE_TYPE (&rendezvous_msg) = CMI_VMI_MESSAGE_TYPE_RENDEZVOUS;
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle->refcount += childcount;
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

      status=VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1,
				    sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1,
				       sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1,
				       sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */
  }

  return ((CmiCommHandle) handle);
}



/**************************************************************************
**
*/
void CmiFreeBroadcastFn (int msgsize, char *msg)
{
  VMI_STATUS status;

  PVMI_BUFFER bufHandles[2];
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T *handle;

  int i;

  PVMI_CACHE_ENTRY cacheentry;

  int childcount;
  int startrank;
  int destrank;

  CMI_VMI_Rendezvous_Message_T rendezvous_msg;


  DEBUG_PRINT ("CmiFreeBroadcastFn() called.\n");

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;

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

      status=VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */

    CmiFree (msg);
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    handle = CMI_VMI_Allocate_Handle ();

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    /* Do NOT increment handle->refcount here! */
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.free_message = TRUE;
    handle->data.send.data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle->refcount += childcount;
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

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection,
				bufHandles, addrs, sz, 1,
				CMI_VMI_Stream_Completion_Handler,
				(PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */
  } else {
    handle = CMI_VMI_Allocate_Handle ();

    /* Do NOT increment handle->refcount here! */
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMABROAD;
    handle->data.send.free_message = TRUE;
    handle->data.send.data.rdmabroad.chunk_size = CMI_VMI_RDMA_Chunk_Size;
    handle->data.send.data.rdmabroad.bytes_sent = CmiAlloc (_Cmi_numpes *
							    sizeof (int));
    handle->data.send.data.rdmabroad.cacheentry = CmiAlloc (_Cmi_numpes *
					       sizeof (PVMI_CACHE_ENTRY));
    for (i = 0; i < _Cmi_numpes; i++) {
      handle->data.send.data.rdmabroad.bytes_sent[i] = 0;
      handle->data.send.data.rdmabroad.cacheentry[i] = NULL;
    }

    CMI_VMI_MESSAGE_TYPE (&rendezvous_msg) = CMI_VMI_MESSAGE_TYPE_RENDEZVOUS;
    rendezvous_msg.rank = _Cmi_mype;
    rendezvous_msg.msgsize = msgsize;
    rendezvous_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) handle;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&rendezvous_msg, 0);
#endif

    addrs[0] = (PVOID) &rendezvous_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Rendezvous_Message_T));

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (msg, (_Cmi_mype + 1));

    childcount = CMI_VMI_Spanning_Children_Count (msg);

    handle->refcount += childcount;
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

      status=VMI_Stream_Send_Inline((&CMI_VMI_Processes[destrank])->connection,
				    addrs, sz, 1,
				    sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else   /* CMK_BROADCAST_SPANNING_TREE */
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1,
				       sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection,
				       addrs, sz, 1,
				       sizeof (CMI_VMI_Rendezvous_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif   /* CMK_BROADCAST_SPANNING_TREE */
  }
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

#if CMK_BROADCAST_SPANNING_TREE
  if (msgsize < CMI_VMI_Small_Message_Boundary) {
    CmiSyncBroadcastFn (msgsize, msg);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msg);
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    msgcopy = CmiAlloc (msgsize);
    memcpy (msgcopy, msg, msgsize);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);

    CmiFreeBroadcastFn (msgsize, msg);
  } else {
    CmiSyncBroadcastFn (msgsize, msg);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msg);
  }
#else   /* CMK_BROADCAST_SPANNING_TREE */
  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    msgcopy = CmiAlloc (msgsize);
    memcpy (msgcopy, msg, msgsize);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);

    CmiFreeBroadcastFn (msgsize, msg);
  } else {
    CmiSyncBroadcastFn (msgsize, msg);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msg);
  }
#endif   /* CMK_BROADCAST_SPANNING_TREE */
}



/**************************************************************************
** done
*/
int CmiAllAsyncMsgsSent ()
{
  DEBUG_PRINT ("CmiAllAsyncMsgsSent() called.\n");

  return (CMI_VMI_AsyncMsgCount < 1);
}



/**************************************************************************
** done
*/
int CmiAsyncMsgSent (CmiCommHandle commhandle)
{
  CMI_VMI_Handle_T *handle;


  DEBUG_PRINT ("CmiAsyncMsgSent() called.\n");

  handle = (CMI_VMI_Handle_T *) commhandle;

  if (handle) {
    return (handle->refcount <= 1);
  } else {
    return (TRUE);
  }
}



/**************************************************************************
** done
*/
void CmiReleaseCommHandle (CmiCommHandle commhandle)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;

  int i;


  DEBUG_PRINT ("CmiReleaseCommHandle() called.\n");

  handle = (CMI_VMI_Handle_T *) commhandle;

  if (handle) {
    handle->refcount--;

    if (handle->refcount <= 1) {
      if (handle->data.send.send_handle_type == 
	  CMI_VMI_SEND_HANDLE_TYPE_STREAM) {
	status=VMI_Cache_Deregister (handle->data.send.data.stream.cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
      }

      if (handle->data.send.send_handle_type ==
	  CMI_VMI_SEND_HANDLE_TYPE_RDMABROAD) {
	for (i = 0; i < _Cmi_numpes; i++) {
	  if (handle->data.send.data.rdmabroad.cacheentry[i]) {
	    status = VMI_Cache_Deregister
	              (handle->data.send.data.rdmabroad.cacheentry[i]);
	    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	  }
	}

	CmiFree (handle->data.send.data.rdmabroad.bytes_sent);
	CmiFree (handle->data.send.data.rdmabroad.cacheentry);
      }

      if (handle->data.send.free_message) {
	CmiFree (handle->msg);
      }

      handle->refcount = 0;
    }
  }
}





#if CMK_PERSISTENT_COMM
/**************************************************************************
** done
*/
void CmiPersistentInit ()
{
  DEBUG_PRINT ("CmiPersistentInit() called.\n");

  CMI_VMI_Persistent_Request_Handler_ID =
    CmiRegisterHandler ((CmiHandler) CMI_VMI_Persistent_Request_Handler);

  CMI_VMI_Persistent_Grant_Handler_ID =
    CmiRegisterHandler ((CmiHandler) CMI_VMI_Persistent_Grant_Handler);

  CMI_VMI_Persistent_Destroy_Handler_ID =
    CmiRegisterHandler ((CmiHandler) CMI_VMI_Persistent_Destroy_Handler);
}



/**************************************************************************
** done
*/
PersistentHandle CmiCreatePersistent (int destrank, int maxsize)
{
  VMI_STATUS status;

  CMI_VMI_Persistent_Request_Message_T request_msg;
  CMI_VMI_Handle_T *handle;

  PVOID addrs[1];
  ULONG sz[1];


  DEBUG_PRINT ("CmiCreatePersitsent() called.\n");

  handle = CMI_VMI_Allocate_Handle ();

  handle->msg = NULL;
  handle->msgsize = -1;
  handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
  handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_PERSISTENT;
  handle->data.send.free_message = FALSE;
  handle->data.send.data.persistent.ready = 0;
  handle->data.send.data.persistent.connection =
                                (&CMI_VMI_Processes[destrank])->connection;
  handle->data.send.data.persistent.destrank = destrank;
  handle->data.send.data.persistent.maxsize = maxsize;
  handle->data.send.data.persistent.remote_buffer = NULL;
  handle->data.send.data.persistent.rdma_receive_index = -1;

  CMI_VMI_MESSAGE_TYPE (&request_msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;
  CmiSetHandler (&request_msg, CMI_VMI_Persistent_Request_Handler_ID);
  request_msg.rank = _Cmi_mype;
  request_msg.maxsize = maxsize;
  request_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) handle;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (&request_msg, 0);
#endif

  addrs[0] = (PVOID) &request_msg;
  sz[0] = (ULONG) (sizeof (CMI_VMI_Persistent_Request_Message_T));

  status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[destrank])->connection,
		 addrs, sz, 1, sizeof (CMI_VMI_Persistent_Request_Message_T));
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

  return ((PersistentHandle) handle);
}


/**************************************************************************
** done
*/
void CmiUsePersistentHandle (PersistentHandle *handle_array, int array_size)
{
  DEBUG_PRINT ("CmiUsePersistentHandle() called.\n");

  CMI_VMI_Persistent_Handles = (CMI_VMI_Handle_T *) handle_array;
  CMI_VMI_Persistent_Handles_Size = array_size;
}


/**************************************************************************
**
*/
void CmiDestroyPersistent (PersistentHandle phandle)
{
  VMI_STATUS status;

  CMI_VMI_Persistent_Destroy_Message_T destroy_msg;
  CMI_VMI_Handle_T *handle;

  PVOID addrs[1];
  ULONG sz[1];


  DEBUG_PRINT ("CmiDestroyPersistent() called.\n");

  handle = (CMI_VMI_Handle_T *) phandle;

  CMI_VMI_MESSAGE_TYPE (&destroy_msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;
  CmiSetHandler (&destroy_msg, CMI_VMI_Persistent_Destroy_Handler_ID);
  destroy_msg.rdma_receive_index =
                 handle->data.send.data.persistent.rdma_receive_index;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (&destroy_msg, 0);
#endif

  addrs[0] = (PVOID) &destroy_msg;
  sz[0] = (ULONG) (sizeof (CMI_VMI_Persistent_Destroy_Message_T));

  status = VMI_Stream_Send_Inline(handle->data.send.data.persistent.connection,
                  addrs, sz, 1, sizeof (CMI_VMI_Persistent_Destroy_Message_T));
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

  handle->refcount = 0;
}



/**************************************************************************
**
*/
void CmiDestroyAllPersistent ()
{
  int i;

  CMI_VMI_Handle_T *handle;


  DEBUG_PRINT ("CmiDestroyAllPersistent() called.\n");

  for (i = 0; i < CMI_VMI_Maximum_Handles; i++) {
    handle = &CMI_VMI_Handles[i];

    if ((handle->refcount > 0) &&
	(handle->handle_type == CMI_VMI_HANDLE_TYPE_SEND) &&
	(handle->data.send.send_handle_type ==
	 CMI_VMI_SEND_HANDLE_TYPE_PERSISTENT)) {
      CmiDestroyPersistent ((PersistentHandle) handle);
    }
  }
}
#endif   /* CMK_PERSISTENT_COMM */





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
  VMI_virt_addr_t remote_handle_address;

  PVMI_CACHE_ENTRY cacheentry;
  CMI_VMI_Handle_T *handle;

  CMI_VMI_Persistent_Grant_Message_T grant_msg;

  PVOID addrs[1];
  ULONG sz[1];


  DEBUG_PRINT ("CMI_VMI_Persistent_Request_Handler() called.\n");

  rank = ((CMI_VMI_Persistent_Request_Message_T *) msg)->rank;
  maxsize = ((CMI_VMI_Persistent_Request_Message_T *) msg)->maxsize;
  remote_handle_address =
                   ((CMI_VMI_Persistent_Request_Message_T *) msg)->context;

  CmiFree (msg);

  handle = CMI_VMI_Allocate_Handle ();

  msg2 = (char *) CmiAlloc (maxsize);

  status = VMI_Cache_Register (msg2, maxsize, &cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

  handle->msg = msg2;
  handle->msgsize = maxsize;
  handle->handle_type = CMI_VMI_HANDLE_TYPE_RECEIVE;
  handle->data.receive.receive_handle_type =
                                      CMI_VMI_RECEIVE_HANDLE_TYPE_PERSISTENT;
  handle->data.receive.data.persistent.cacheentry = cacheentry;
  handle->data.receive.data.persistent.remote_handle_address =
                                                       remote_handle_address;

  status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[rank])->connection,
       cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg2,
       maxsize, remote_handle_address, (UINT32) handle->index);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");

  CMI_VMI_MESSAGE_TYPE (&grant_msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;
  CmiSetHandler (&grant_msg, CMI_VMI_Persistent_Grant_Handler_ID);
  grant_msg.context = (VMI_virt_addr_t) (VMI_ADDR_CAST) remote_handle_address;
  grant_msg.rdma_receive_index = handle->index;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (&grant_msg, 0);
#endif

  addrs[0] = (PVOID) &grant_msg;
  sz[0] = (ULONG) (sizeof (CMI_VMI_Persistent_Grant_Message_T));

  status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[rank])->connection,
	       addrs, sz, 1, sizeof (CMI_VMI_Persistent_Grant_Message_T));
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
}


/**************************************************************************
** This is a Converse handler.
*/
void CMI_VMI_Persistent_Grant_Handler (char *msg)
{
  CMI_VMI_Handle_T *handle;


  DEBUG_PRINT ("CMI_VMI_Persistent_Grant_Handler() called.\n");

  handle = (CMI_VMI_Handle_T *) (VMI_ADDR_CAST)
            ((CMI_VMI_Persistent_Grant_Message_T *) msg)->context;
  handle->data.send.data.persistent.rdma_receive_index =
            ((CMI_VMI_Persistent_Grant_Message_T *) msg)->rdma_receive_index;

  CmiFree (msg);

  handle->data.send.data.persistent.ready++;
}


/**************************************************************************
** This is a Converse handler.
*/
void CMI_VMI_Persistent_Destroy_Handler (char *msg)
{
  VMI_STATUS status;
  int handle_index;

  CMI_VMI_Handle_T *handle;


  DEBUG_PRINT ("CMI_VMI_Persistent_Destroy_Handler() called.\n");

  handle_index =
       ((CMI_VMI_Persistent_Destroy_Message_T *) msg)->rdma_receive_index;

  CmiFree (msg);

  handle = &CMI_VMI_Handles[handle_index];

  CmiFree (handle->msg);
  CmiFree (handle->data.receive.data.persistent.cacheentry);

  handle->refcount = 0;
}
#endif   /* CMK_PERSISTENT_COMM */




#if CMK_MULTICAST_LIST_USE_SPECIAL_CODE
/**************************************************************************
** done
*/
void CmiSyncListSendFn (int npes, int *pes, int len, char *msg)
{
  DEBUG_PRINT ("CmiSyncListSendFn() called.\n");

  CmiError ("ListSend not implemented.");
}


/**************************************************************************
** done
*/
CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int len, char *msg)
{
  DEBUG_PRINT ("CmiAsyncListSendFn() called.\n");

  CmiError ("ListSend not implemented.");
  return ((CmiCommHandle) NULL);
}


/**************************************************************************
** done
*/
void CmiFreeListSendFn (int npes, int *pes, int msgsize, char *msg)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;
  PVMI_RDMA_OP rdmaop;
  PVMI_CACHE_ENTRY cacheentry;
  char *putaddr;
  int putlen;
  int i;


  DEBUG_PRINT ("CmiFreeListSendFn() called.\n");

  /*
    NOTE: This code completely ignores pes passed in as a parameter and
    instead uses the destinations held in the persistent handle array.
  */

  CmiAssert (npes == CMI_VMI_Persistent_Handles_Size);

  status = VMI_Cache_Register (msg, msgsize, &cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

  for (i = 0; i < CMI_VMI_Persistent_Handles_Size; i++) {
    handle = &CMI_VMI_Persistent_Handles[i];

    // WARNING: Check the use of refcount in this code!!

    handle->refcount++;
    handle->msg = msg;
    handle->msgsize = msgsize;

    handle->data.send.data.persistent.cacheentry = cacheentry;

    status = VMI_RDMA_Alloc_Op (&rdmaop);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

    rdmaop->numBufs = 1;
    rdmaop->buffers[0] = cacheentry->bufferHandle;
    rdmaop->addr[0] = msg;
    rdmaop->sz[0] = msgsize;
    rdmaop->rbuffer = handle->data.send.data.persistent.remote_buffer;
    rdmaop->roffset = 0;

    CMI_VMI_AsyncMsgCount++;

    status = VMI_RDMA_Put (handle->data.send.data.persistent.connection,
			                         rdmaop, (PVOID) handle,
	        (VMIRDMAWriteComplete) CMI_VMI_RDMA_Completion_Handler);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");

    while (handle->refcount > 1) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }
  }

  status = VMI_Cache_Deregister (cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

  CmiFree (msg);
}
#endif



/**************************************************************************
**
*/
CMI_VMI_Handle_T *CMI_VMI_Allocate_Handle ()
{
  VMI_STATUS status;

  int i;
  int j;


  DEBUG_PRINT ("CMI_VMI_Allocate_Handle() called.\n");

  i = CMI_VMI_Next_Handle;
  j = CMI_VMI_Next_Handle;
  while ((&CMI_VMI_Handles[i])->refcount > 0) {
    i++;

    if (i >= CMI_VMI_Maximum_Handles) {
      i = 0;
    }

    if (i == j) {
      i = CMI_VMI_Maximum_Handles;
      CMI_VMI_Maximum_Handles *= 2;
      CMI_VMI_Handles = (CMI_VMI_Handle_T *) realloc (CMI_VMI_Handles,
		 CMI_VMI_Maximum_Handles * sizeof (CMI_VMI_Handle_T));
      for (j = i; j < CMI_VMI_Maximum_Handles; j++) {
	(&CMI_VMI_Handles[j])->index = j;
	(&CMI_VMI_Handles[j])->refcount = 0;
      }
    }
  }

  (&CMI_VMI_Handles[i])->refcount = 1;
  CMI_VMI_Next_Handle = (i + 1);
  if (CMI_VMI_Next_Handle >= CMI_VMI_Maximum_Handles) {
    CMI_VMI_Next_Handle = 0;
  }

  return (&CMI_VMI_Handles[i]);
}






#if CMI_VMI_USE_MEMORY_POOL
/**************************************************************************
**
*/
void *CMI_VMI_CmiAlloc (int size)
{
  VMI_STATUS status;

  void *ptr;


  DEBUG_PRINT ("CMI_VMI_CmiAlloc() (memory pool version) called.\n");

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


  DEBUG_PRINT ("CMI_VMI_CmiFree() (memory pool version) called.\n");

  size = (((int *) ptr)[0]) + sizeof (int) + sizeof (int);

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
#else   /* CMI_VMI_USE_MEMORY_POOL */
/**************************************************************************
**
*/
void *CMI_VMI_CmiAlloc (int size)
{
  DEBUG_PRINT ("CMI_VMI_CmiAlloc() (simple version) called.\n");

  return (malloc (size));
}


/**************************************************************************
**
*/
void CMI_VMI_CmiFree (void *ptr)
{
  DEBUG_PRINT ("CMI_VMI_CmiFree() (simple version) called.\n");

  free (ptr);
}
#endif   /* CMI_VMI_USE_MEMORY_POOL */









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


  DEBUG_PRINT ("CRMInit() called.\n");

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


/**************************************************************************
**
*/
SOCKET createSocket(char *hostName, int port, int *localAddr)
{
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


  DEBUG_PRINT ("createSocket() called.\n");
  
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


  DEBUG_PRINT ("CRMRegister() called.\n");

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


  DEBUG_PRINT ("CRMParseMsg() called.\n");

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


  DEBUG_PRINT ("CRMSend() called.\n");
  
  sent = 0;
  while (sent < n)
  {
    bsent = send (s, (const void *) (msg + sent), (n - sent), 0);
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


  DEBUG_PRINT ("CRMRecv() called.\n");

  recvd = 0;
  while (recvd < n)
  {
    brecv = recv (s, (void *) (msg + recvd), (n - recvd), 0);
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
