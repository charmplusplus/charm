/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
**
** This code does not work correctly with glibc 2.2.92 or lower due to
** problems with Converse threads interacting with VMI's use of pthreads.
** To check which version is on a system:
**
**    nm -a /lib/libc.so.6 | grep 'A GLIBC_'
*/

/** @file
 * VMI machine layer
 * @ingroup Machine
 * @{
 */

#include "machine.h"

/* The following are external variables used by the VMI core. */
extern USHORT VMI_DEVICE_RUNTIME;
extern PVMI_NETADDRESS localAddress;
extern VMIStreamRecv recvFn;

/* The following are variables and functions used by the Converse core. */
int _Cmi_numpes;
int _Cmi_mype;
int _Cmi_myrank = 0;

CpvDeclare (void *, CmiLocalQueue);
CpvDeclare (void *, CMI_VMI_RemoteQueue);

extern void CthInit (char **argv);
extern void ConverseCommonInit (char **argv);

/* Global variables. */
char *CMI_VMI_Username;
char *CMI_VMI_Program_Key;
int CMI_VMI_Startup_Type;
int CMI_VMI_WAN_Latency;
int CMI_VMI_Cluster;
int CMI_VMI_Probe_Clusters;
int CMI_VMI_Memory_Pool;
int CMI_VMI_Terminate_VMI_Hack;
int CMI_VMI_Connection_Timeout;
int CMI_VMI_Maximum_Handles;
int CMI_VMI_Small_Message_Boundary;
int CMI_VMI_Medium_Message_Boundary;
int CMI_VMI_Eager_Protocol;
int CMI_VMI_Eager_Interval;
int CMI_VMI_Eager_Threshold;
int CMI_VMI_Eager_Short_Pollset_Size_Maximum;
int CMI_VMI_Eager_Short_Slots;
int CMI_VMI_Eager_Long_Buffers;
int CMI_VMI_Eager_Long_Buffer_Size;

volatile int CMI_VMI_Message_Receive_Count;
volatile int CMI_VMI_AsyncMsgCount;
volatile int CMI_VMI_Barrier_Count;

int CMI_VMI_Charmrun_Socket;
char CMI_VMI_Charmrun_IP[1024];
int CMI_VMI_Charmrun_Port;

int CMI_VMI_CRM_Socket;
char *CMI_VMI_CRM_Hostname;
int CMI_VMI_CRM_Port;

CMI_VMI_Process_T *CMI_VMI_Processes;
CMI_VMI_Process_T **CMI_VMI_Eager_Short_Pollset;
int CMI_VMI_Eager_Short_Pollset_Size;

CMI_VMI_Handle_T *CMI_VMI_Handles;
int CMI_VMI_Next_Handle;

int CMI_VMI_Latency_Vectors_Received;
BOOLEAN CMI_VMI_Cluster_Mapping_Received;

PVMI_BUFFER_POOL CMI_VMI_Bucket1_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket2_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket3_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket4_Pool;
PVMI_BUFFER_POOL CMI_VMI_Bucket5_Pool;



/**************************************************************************
** This function is the entry point for all Converse and Charm++ codes.
** 
** argc
** argv
** start_function - the user-supplied function to run (function pointer)
** user_calls_scheduler - boolean for whether ConverseInit() should invoke
**                        the scheduler or whether user code will do it
** init_returns - boolean for whether ConverseInit() returns
*/
void ConverseInit (int argc, char **argv, CmiStartFn start_function, int user_calls_scheduler, int init_returns)
{
  int rc;
  int i;
  int j;


  DEBUG_PRINT ("ConverseInit() called.\n");

  /* Get a default program key from argv[0]. */
  if (!(CMI_VMI_Program_Key = strdup (argv[0]))) {
    CmiAbort ("Unable to allocate memory for the program key.");
  }

  /* Initialize global variables. */
  CMI_VMI_Startup_Type                     = CMI_VMI_STARTUP_TYPE_UNKNOWN;
  CMI_VMI_WAN_Latency                      = CMI_VMI_WAN_LATENCY;
  CMI_VMI_Cluster                          = CMI_VMI_CLUSTER_UNKNOWN;
  CMI_VMI_Probe_Clusters                   = CMI_VMI_PROBE_CLUSTERS;
  CMI_VMI_Memory_Pool                      = CMI_VMI_MEMORY_POOL;
  CMI_VMI_Terminate_VMI_Hack               = CMI_VMI_TERMINATE_VMI_HACK;
  CMI_VMI_Connection_Timeout               = CMI_VMI_CONNECTION_TIMEOUT;
  CMI_VMI_Maximum_Handles                  = CMI_VMI_MAXIMUM_HANDLES;
  CMI_VMI_Small_Message_Boundary           = CMI_VMI_SMALL_MESSAGE_BOUNDARY;
  CMI_VMI_Medium_Message_Boundary          = CMI_VMI_MEDIUM_MESSAGE_BOUNDARY;
  CMI_VMI_Eager_Protocol                   = CMI_VMI_EAGER_PROTOCOL;
  CMI_VMI_Eager_Interval                   = CMI_VMI_EAGER_INTERVAL;
  CMI_VMI_Eager_Threshold                  = CMI_VMI_EAGER_THRESHOLD;
  CMI_VMI_Eager_Short_Pollset_Size_Maximum = CMI_VMI_EAGER_SHORT_POLLSET_SIZE_MAXIMUM;
  CMI_VMI_Eager_Short_Slots                = CMI_VMI_EAGER_SHORT_SLOTS;
  CMI_VMI_Eager_Long_Buffers               = CMI_VMI_EAGER_LONG_BUFFERS;
  CMI_VMI_Eager_Long_Buffer_Size           = CMI_VMI_EAGER_LONG_BUFFER_SIZE;

  CMI_VMI_Message_Receive_Count = 0;
  CMI_VMI_AsyncMsgCount = 0;
  CMI_VMI_Barrier_Count = 0;

  CMI_VMI_Cluster_Mapping_Received = FALSE;

  /* Read global variable values from the environment. */
  CMI_VMI_Read_Environment ();

  /* Set up the process array and initialize. */
  CMI_VMI_Processes = (CMI_VMI_Process_T *) (malloc (_Cmi_numpes * sizeof (CMI_VMI_Process_T)));
  if (!CMI_VMI_Processes) {
    CmiAbort ("Unable to allocate memory for process array.");
  }

  CMI_VMI_Eager_Short_Pollset = (CMI_VMI_Process_T **) (malloc (_Cmi_numpes * sizeof (CMI_VMI_Process_T *)));
  if (!CMI_VMI_Eager_Short_Pollset) {
    CmiAbort ("Unable to allocate memory for eager pollset array.");
  }

  for (i = 0; i < _Cmi_numpes; i++) {
    (&CMI_VMI_Processes[i])->connection_state = CMI_VMI_CONNECTION_DISCONNECTED;
    (&CMI_VMI_Processes[i])->cluster = CMI_VMI_CLUSTER_UNKNOWN;

    (&CMI_VMI_Processes[i])->latency_vector = NULL;

    (&CMI_VMI_Processes[i])->normal_short_count;
    (&CMI_VMI_Processes[i])->normal_long_count;
    (&CMI_VMI_Processes[i])->eager_short_count;
    (&CMI_VMI_Processes[i])->eager_long_count;

    for (j = 0; j < CMI_VMI_Eager_Short_Slots; j++) {
      (&CMI_VMI_Processes[i])->eager_short_send_handles[j] = NULL;
      (&CMI_VMI_Processes[i])->eager_short_receive_handles[j] = NULL;
    }

    (&CMI_VMI_Processes[i])->eager_short_send_size = 0;
    (&CMI_VMI_Processes[i])->eager_short_send_index = 0;
    (&CMI_VMI_Processes[i])->eager_short_send_credits_available = 0;

    (&CMI_VMI_Processes[i])->eager_short_receive_size = 0;
    (&CMI_VMI_Processes[i])->eager_short_receive_index = 0;
    (&CMI_VMI_Processes[i])->eager_short_receive_dirty = 0;
    (&CMI_VMI_Processes[i])->eager_short_receive_credits_replentish = 0;

    CMI_VMI_Eager_Short_Pollset[i] = (CMI_VMI_Process_T *) NULL;

    for (j = 0; j < CMI_VMI_Eager_Long_Buffers; j++) {
      (&CMI_VMI_Processes[i])->eager_long_send_handles[j] = NULL;
      (&CMI_VMI_Processes[i])->eager_long_receive_handles[j] = NULL;
    }

    (&CMI_VMI_Processes[i])->eager_long_send_size = 0;
    (&CMI_VMI_Processes[i])->eager_long_receive_size = 0;
  }

  CMI_VMI_Eager_Short_Pollset_Size = 0;

  /* Set up the send/receive handle array and initialize. */
  CMI_VMI_Handles = (CMI_VMI_Handle_T *) (malloc (CMI_VMI_Maximum_Handles * sizeof (CMI_VMI_Handle_T)));
  if (!CMI_VMI_Handles) {
    CmiAbort ("Unable to allocate memory for handle array.");
  }

  for (i = 0; i < CMI_VMI_Maximum_Handles; i++) {
    (&CMI_VMI_Handles[i])->index = i;
    (&CMI_VMI_Handles[i])->refcount = 0;
  }

  CMI_VMI_Next_Handle = 0;

  /* Print out debug information if compiled with debug support. */
  DEBUG_PRINT ("The program key is %s.\n", key);
  DEBUG_PRINT ("The startup type is %d.\n", CMI_VMI_Startup_Type);

  /* Start up via the startup type selected. */
  switch (CMI_VMI_Startup_Type)
  {
    case CMI_VMI_STARTUP_TYPE_CRM:
      rc = CMI_VMI_Startup_CRM ();
      break;

    case CMI_VMI_STARTUP_TYPE_CHARMRUN:
      rc = CMI_VMI_Startup_Charmrun ();
      break;

    default:
      CmiAbort ("An unknown startup type was specified.");
      break;
  }

  if (rc < 0) {
    CmiAbort ("There was a fatal error during the startup phase.");
  }

  /* Initialize VMI. */
  rc = CMI_VMI_Initialize_VMI ();

  if (rc < 0) {
    CmiAbort ("There was a fatal error during VMI initialization.");
  }

  DEBUG_PRINT ("VMI was initialized successfully.\n");

  /*
    Create the FIFOs for holding local and remote messages.

    NOTE: FIFO creation must happen at this point due to a race condition
          where some processes may open their connections and start sending
          messages before all of the other processes are started, and we
          must be able to deal with this situation.
  */
  CpvAccess (CmiLocalQueue) = CdsFifo_Create ();
  CpvAccess (CMI_VMI_RemoteQueue) = CdsFifo_Create ();

  /* Open connections. */
  rc = CMI_VMI_Open_Connections ();

  if (rc < 0) {
    CmiAbort ("Fatal error during connection setup phase.");
  }

  /* Probe the cluster mapping by requesting all-to-all latencies (if requested). */
  if (CMI_VMI_Probe_Clusters) {
    if (_Cmi_mype == 0) {
      CmiProbeLatencies ();
      CMI_VMI_Compute_Cluster_Mapping ();
      CMI_VMI_Distribute_Cluster_Mapping ();
    } else {
      CMI_VMI_Wait_Cluster_Mapping ();
    }
  }

  DEBUG_PRINT ("ConverseInit() is starting the main processing loop.\n");

  /* Initialize Converse and start the main processing loop. */
  CthInit (argv);
  ConverseCommonInit (argv);

  /* Set up CmiNotifyIdle() to be called when processor goes idle. */
  CcdCallOnConditionKeep (CcdPROCESSOR_STILL_IDLE, (CcdVoidFn) CmiNotifyIdle, NULL);

  if (!init_returns) {
    start_function (CmiGetArgc (argv), argv);
    if (!user_calls_scheduler) {
      CsdScheduler (-1);
    }
    ConverseExit ();
  }
}



/**************************************************************************
** This function is the exit point for all Converse and Charm++ codes.
*/
void ConverseExit ()
{
  VMI_STATUS status;

  int i;


  DEBUG_PRINT ("ConverseExit() called.\n");

  /* Signal the charmrun terminal that the computation has ended (if necessary). */
  if (CMI_VMI_Startup_Type == CMI_VMI_STARTUP_TYPE_CHARMRUN) {
    CMI_VMI_Charmrun_Message_Header_T hdr;
    int rc;

    hdr.msg_len = htonl (0);
    strcpy (hdr.msg_type, "ending");

    rc = CMI_VMI_Socket_Send (CMI_VMI_Charmrun_Socket, (const void *) &hdr, (int) sizeof (CMI_VMI_Charmrun_Message_Header_T));
    if (rc < 0) {
      DEBUG_PRINT ("Error sending to charmrun.\n");
    }

    /* Do NOT close CMI_VMI_Charmrun_Socket here or charmrun will die! */
  }

  /* ConverseCommonExit() shuts down CCS and closes Projections logs. */
  ConverseCommonExit ();

  /* Barrier to ensure that all processes are ready to shut down. */
  CmiBarrier ();

  for (i = 0; i < 100000; i++) {
    status = VMI_Poll ();
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
  }

  /* If a clean VMI termination is requested, do it. */
  if (!CMI_VMI_Terminate_VMI_Hack) {
    CMI_VMI_Close_Connections ();

    for (i = 0; i < 100000; i++) {
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    CMI_VMI_Terminate_VMI ();
  }

  /* Free resources and exit. */
  CdsFifo_Destroy (CpvAccess (CMI_VMI_RemoteQueue));
  CdsFifo_Destroy (CpvAccess (CmiLocalQueue));

  for (i = 0; i < _Cmi_numpes; i++) {
    if ((&CMI_VMI_Processes[i])->latency_vector) {
      free ((&CMI_VMI_Processes[i])->latency_vector);
    }
  }

  free (CMI_VMI_Handles);
  free (CMI_VMI_Eager_Short_Pollset);
  free (CMI_VMI_Processes);
  free (CMI_VMI_Program_Key);
  free (CMI_VMI_Username);

  exit (0);
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
void CmiNotifyIdle ()
{
  VMI_STATUS status;

  CMI_VMI_Process_T *process;
  CMI_VMI_Handle_T *handle;

  int index;
  char *msg;
  CMI_VMI_Eager_Short_Slot_Footer_T *footer;
  int credits_temp;

  CMI_VMI_Credit_Message_T credit_msg;
  PVOID addrs[1];
  ULONG sz[1];

  int i;


  DEBUG_PRINT ("CmiNotifyIdle() called.\n");

  if (CMI_VMI_Eager_Protocol) {
    /*
      Check to see if any processes have a large number of outstanding eager short credits.

      Normally, eager short credits are replentished on the sender when we send a message
      to it.  If we do not communicate frequently with the sender, this does not happen
      automatically and we need to explicitly send credit updates.
    */
    for (i = 0; i < _Cmi_numpes; i++) {
      process = &CMI_VMI_Processes[i];

      if (process->eager_short_receive_credits_replentish >= (0.75 * CMI_VMI_Eager_Short_Slots)) {
	CMI_VMI_MESSAGE_TYPE (&credit_msg) = CMI_VMI_MESSAGE_TYPE_CREDIT;
	CMI_VMI_MESSAGE_CREDITS (&credit_msg) = process->eager_short_receive_credits_replentish;

#if CMK_BROADCAST_SPANNING_TREE
	CMI_SET_BROADCAST_ROOT (&credit_msg, 0);
#endif

	addrs[0] = (PVOID) &credit_msg;
	sz[0] = (ULONG) (sizeof (CMI_VMI_Credit_Message_T));

	status = VMI_Stream_Send_Inline (process->connection, addrs, sz, 1, sizeof (CMI_VMI_Credit_Message_T));
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

	process->eager_short_receive_credits_replentish = 0;
      }
    }

    /*
      Check to see if any processes are communicating with us frequently.
      These processes are candidates for eager communications.
    */
    if (CMI_VMI_Message_Receive_Count > CMI_VMI_Eager_Interval) {
      for (i = 0; i < _Cmi_numpes; i++) {
	if ((CMI_VMI_Eager_Short_Pollset_Size < CMI_VMI_Eager_Short_Pollset_Size_Maximum) &&
	    ((&CMI_VMI_Processes[i])->normal_short_count > CMI_VMI_Eager_Threshold) &&
	    ((&CMI_VMI_Processes[i])->eager_short_receive_size == 0) &&
	    (VMI_CONNECT_ONE_WAY_LATENCY ((&CMI_VMI_Processes[i])->connection) < CMI_VMI_WAN_Latency)) {
	  CMI_VMI_Eager_Short_Setup (i);
	}

	if (((&CMI_VMI_Processes[i])->normal_long_count > CMI_VMI_Eager_Threshold) &&
	    ((&CMI_VMI_Processes[i])->eager_long_receive_size == 0)) {
	  CMI_VMI_Eager_Long_Setup (i, CMI_VMI_Eager_Long_Buffer_Size);
	}

	(&CMI_VMI_Processes[i])->normal_short_count = 0;
	(&CMI_VMI_Processes[i])->normal_long_count = 0;
	(&CMI_VMI_Processes[i])->eager_short_count = 0;
	(&CMI_VMI_Processes[i])->eager_long_count = 0;
      }

      CMI_VMI_Message_Receive_Count = 0;
    }
  }

  /* Pump the message loop. */
  status = VMI_Poll ();
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
}



/**************************************************************************
**
*/
void CmiMemLock ()
{
  DEBUG_PRINT ("CmiMemLock() called.\n");

  /* Empty. */
}



/**************************************************************************
**
*/
void CmiMemUnlock ()
{
  DEBUG_PRINT ("CmiMemUnlock() called.\n");

  /* Empty. */
}



/**************************************************************************
** This is our implementation of CmiPrintf().  For the case where the code
** was started by charmrun, we must use the charmrun protocol to send
** output back to the charmrun terminal.  Otherwise, we can simply send
** output to the program's stdout (which is automatically redirected to
** a socket that is attached to the right place).
**
** NOTE: When sending to the charmrun terminal, an explicit NULL must be
** included at the end of the message buffer.  The charmrun terminal reuses
** its buffer, so if a terminating NULL is not sent, the tail of any
** previous larger-sized message is printed after the shorter message sent
** here.
*/
void CmiPrintf (const char *format, ...)
{
  DEBUG_PRINT ("CmiPrintf() called.\n");

  if (CMI_VMI_Startup_Type == CMI_VMI_STARTUP_TYPE_CHARMRUN) {
    CMI_VMI_Charmrun_Message_Header_T hdr;
    va_list args;
    char *temp_str;
    int rc;

    va_start (args, format);
    vasprintf (&temp_str, format, args);

    hdr.msg_len = htonl (strlen (temp_str) + 1);
    strcpy (hdr.msg_type, "print");

    rc = CMI_VMI_Socket_Send (CMI_VMI_Charmrun_Socket, (const void *) &hdr, sizeof (CMI_VMI_Charmrun_Message_Header_T));
    if (rc < 0) {
      DEBUG_PRINT ("Error sending to charmrun.\n");
    }
    rc = CMI_VMI_Socket_Send (CMI_VMI_Charmrun_Socket, temp_str, ((strlen (temp_str)) + 1));
    if (rc < 0) {
      DEBUG_PRINT ("Error sending to charmrun.\n");
    }

    free (temp_str);
  } else {
    va_list args;
    va_start (args, format);
    vprintf (format, args);
    fflush (stdout);
    va_end (args);
  }
}



/**************************************************************************
** See comments for CmiPrintf() above.
*/
void CmiError (const char *format, ...)
{
  DEBUG_PRINT ("CmiError() called.\n");

  if (CMI_VMI_Startup_Type == CMI_VMI_STARTUP_TYPE_CHARMRUN) {
    CMI_VMI_Charmrun_Message_Header_T hdr;
    va_list args;
    char *temp_str;
    int rc;

    va_start (args, format);
    vasprintf (&temp_str, format, args);

    hdr.msg_len = htonl (strlen (temp_str) + 1);
    strcpy (hdr.msg_type, "printerr");

    rc = CMI_VMI_Socket_Send (CMI_VMI_Charmrun_Socket, (const void *) &hdr, sizeof (CMI_VMI_Charmrun_Message_Header_T));
    if (rc < 0) {
      DEBUG_PRINT ("Error sending to charmrun.\n");
    }
    rc = CMI_VMI_Socket_Send (CMI_VMI_Charmrun_Socket, temp_str, ((strlen (temp_str)) + 1));
    if (rc < 0) {
      DEBUG_PRINT ("Error sending to charmrun.\n");
    }

    free (temp_str);
  } else {
    va_list args;
    va_start (args, format);
    vfprintf (stderr, format, args);
    fflush (stdout);
    va_end (args);
  }
}



/**************************************************************************
** This is a simple barrier function, similar to the one implemented in the
** net-linux-gm machine layer.  This routine assumes that there are few
** messages in flight; I have not tested extensively with many outstanding
** messages and there could very well be some nasty race conditions.
**
** This routine was implemented to allow clocks to be synchronized at
** program startup time, so Projections timeline views do not show message
** sends that appear after the corresponding message deliveries.
**
** THIS CODE ASSUMES THAT CMI_VMI_Barrier_Count IS INITIALIZED TO 0
** DURING ConverseInit()!  We cannot initialize it in this function due
** to a race condition where PE 0 might invoke CmiBarrer() much later
** than other nodes in the computation.  In this case, it might have
** already seen barrier messages coming in from the other nodes and
** counted them in the stream receive handler prior to invoking
** CmiBarrier().
**
** TODO: This routine should use spanning trees if this machine layer has
** been configured for that type of message broadcast.
*/
void CmiBarrier ()
{
  VMI_STATUS status;

  CMI_VMI_Barrier_Message_T barrier_msg;
  PVOID addrs[1];
  ULONG sz[1];

  int i;


  DEBUG_PRINT ("CmiBarrier() called.\n");

  /* PE 0 coordinates the barrier. */
  if (_Cmi_mype == 0) {
    /* Wait until all processes send us a barrier message. */
    while (CMI_VMI_Barrier_Count < (_Cmi_numpes - 1)) {
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    /* Reset the barrier count immediately to set up next barrier operation. */
    CMI_VMI_Barrier_Count = 0;

    /* Send a barrier message to each process to signal that barrier is finished. */
    CMI_VMI_MESSAGE_TYPE (&barrier_msg) = CMI_VMI_MESSAGE_TYPE_BARRIER;
    CMI_VMI_MESSAGE_CREDITS (&barrier_msg) = 0;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&barrier_msg, 0);
#endif

    addrs[0] = (PVOID) &barrier_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Barrier_Message_T));

    for (i = 1; i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection, addrs, sz, 1, sizeof (CMI_VMI_Barrier_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
  } else {
    /* Send a barrier message to PE 0. */
    CMI_VMI_MESSAGE_TYPE (&barrier_msg) = CMI_VMI_MESSAGE_TYPE_BARRIER;
    CMI_VMI_MESSAGE_CREDITS (&barrier_msg) = 0;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&barrier_msg, 0);
#endif

    addrs[0] = (PVOID) &barrier_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Barrier_Message_T));

    status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[0])->connection, addrs, sz, 1, sizeof (CMI_VMI_Barrier_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

    /* Wait until PE 0 notifies us that barrier is finished. */
    while (CMI_VMI_Barrier_Count < 1) {
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    /* Reset the barrier count immediately to set up next barrier operation. */
    CMI_VMI_Barrier_Count = 0;
  }
}



/**************************************************************************
**
*/
void CmiBarrierZero ()
{
  DEBUG_PRINT ("CmiBarrierZero() called.\n");

  CmiBarrier ();
}



/**************************************************************************
**
*/
void CmiSyncSendFn (int destrank, int msgsize, char *msg)
{
  VMI_STATUS status;

  char *msgcopy;

  CMI_VMI_Process_T *process;

  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T *handle;

  PVMI_CACHE_ENTRY cacheentry;

  CMI_VMI_Publish_Message_T publish_msg;

  void *context;

  int index;

  PVMI_RDMA_OP rdmaop;

  int offset;

  CMI_VMI_Eager_Short_Slot_Footer_T footer;


  DEBUG_PRINT ("CmiSyncSendFn() called.\n");

  if (destrank == _Cmi_mype) {
    msgcopy = CmiAlloc (msgsize);
    memcpy (msgcopy, msg, msgsize);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);
    return;
  }

  process = &CMI_VMI_Processes[destrank];

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (msg, 0);
#endif

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;
  CMI_VMI_MESSAGE_CREDITS (msg) = process->eager_short_receive_credits_replentish;
  process->eager_short_receive_credits_replentish = 0;

  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    if (process->eager_short_send_credits_available > 0) {
      index = process->eager_short_send_index;
      handle = process->eager_short_send_handles[index];

      memcpy (handle->msg, msg, msgsize);

      footer.msgsize = msgsize;
      footer.sentinel = CMI_VMI_EAGER_SHORT_SENTINEL_DATA;
      memcpy (handle->msg + msgsize, &footer, sizeof (CMI_VMI_Eager_Short_Slot_Footer_T));

      handle->msgsize = msgsize;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;

      cacheentry = handle->data.send.data.eager_short.cacheentry;
      rdmaop = handle->data.send.data.eager_short.rdmaop;

      offset = handle->data.send.data.eager_short.offset;
      offset += (CMI_VMI_Medium_Message_Boundary - msgsize);

      rdmaop->numBufs = 1;
      rdmaop->buffers[0] = cacheentry->bufferHandle;
      rdmaop->addr[0] = handle->msg;
      rdmaop->sz[0] = msgsize + sizeof (CMI_VMI_Eager_Short_Slot_Footer_T);
      rdmaop->rbuffer = handle->data.send.data.eager_short.remote_buffer;
      rdmaop->roffset = offset;
      rdmaop->notify = FALSE;

      status = VMI_RDMA_Put (process->connection, rdmaop, (PVOID) NULL, (VMIRDMACompleteNotification) NULL);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");

      process->eager_short_send_index = ((index + 1) % process->eager_short_send_size);
      process->eager_short_send_credits_available -= 1;
    } else {
      addrs[0] = (PVOID) msg;
      sz[0] = (ULONG) msgsize;

      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[destrank])->connection, addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
  } else {
    if (CMI_VMI_Eager_Protocol) {
      context = CONTEXTFIELD (msg);
      if (context) {
	cacheentry = CMI_VMI_CacheEntry_From_Context (context);
      } else {
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }
    } else {
      context = NULL;
      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
    }

    index = process->eager_long_send_size - 1;
    handle = process->eager_long_send_handles[index];

    if ((process->eager_long_send_size > 0) && (msgsize < handle->data.send.data.eager_long.maxsize)) {
      process->eager_long_send_size = index;

      handle->msg = msg;
      handle->msgsize = msgsize;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
      handle->data.send.data.eager_long.cacheentry = cacheentry;

      status = VMI_RDMA_Alloc_Op (&rdmaop);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

      rdmaop->numBufs = 1;
      rdmaop->buffers[0] = cacheentry->bufferHandle;
      rdmaop->addr[0] = handle->msg;
      rdmaop->sz[0] = msgsize;
      rdmaop->rbuffer = handle->data.send.data.eager_long.remote_buffer;
      rdmaop->roffset = 0;
      rdmaop->notify = TRUE;

      CMI_VMI_AsyncMsgCount += 1;
      handle->refcount += 1;

      status = VMI_RDMA_Put (process->connection, rdmaop, (PVOID) handle, (VMIRDMACompleteNotification) CMI_VMI_RDMA_Put_Completion_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    } else {
      handle = CMI_VMI_Handle_Allocate ();

      handle->refcount += 1;
      handle->msg = msg;
      handle->msgsize = msgsize;
      handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
      handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMAGET;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
      handle->data.send.data.rdmaget.cacheentry = cacheentry;

      CMI_VMI_AsyncMsgCount += 1;
      handle->refcount += 1;

      publish_msg.type = CMI_VMI_PUBLISH_TYPE_GET;

      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[destrank])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
				        (UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
				        (ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }

    while (handle->refcount > 2) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    if (!context) {
      status = VMI_Cache_Deregister (cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
    }

    CMI_VMI_Handle_Deallocate (handle);
  }
}



/**************************************************************************
**
*/
CmiCommHandle CmiAsyncSendFn (int destrank, int msgsize, char *msg)
{
  VMI_STATUS status;

  char *msgcopy;

  CMI_VMI_Process_T *process;

  PVMI_BUFFER bufHandles[2];
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T *handle;

  PVMI_CACHE_ENTRY cacheentry;

  CMI_VMI_Publish_Message_T publish_msg;

  void *context;

  int index;

  PVMI_RDMA_OP rdmaop;

  int offset;

  CMI_VMI_Eager_Short_Slot_Footer_T footer;


  DEBUG_PRINT ("CmiAsyncSendFn() called.\n");

  if (destrank == _Cmi_mype) {
    msgcopy = CmiAlloc (msgsize);
    memcpy (msgcopy, msg, msgsize);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);
    return ((CmiCommHandle) NULL);
  }

  process = &CMI_VMI_Processes[destrank];

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (msg, 0);
#endif

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;
  CMI_VMI_MESSAGE_CREDITS (msg) = process->eager_short_receive_credits_replentish;
  process->eager_short_receive_credits_replentish = 0;

  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    if (process->eager_short_send_credits_available > 0) {
      index = process->eager_short_send_index;
      handle = process->eager_short_send_handles[index];

      memcpy (handle->msg, msg, msgsize);

      footer.msgsize = msgsize;
      footer.sentinel = CMI_VMI_EAGER_SHORT_SENTINEL_DATA;
      memcpy (handle->msg + msgsize, &footer, sizeof (CMI_VMI_Eager_Short_Slot_Footer_T));

      handle->msgsize = msgsize;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;

      cacheentry = handle->data.send.data.eager_short.cacheentry;
      rdmaop = handle->data.send.data.eager_short.rdmaop;

      offset = handle->data.send.data.eager_short.offset;
      offset += (CMI_VMI_Medium_Message_Boundary - msgsize);

      rdmaop->numBufs = 1;
      rdmaop->buffers[0] = cacheentry->bufferHandle;
      rdmaop->addr[0] = handle->msg;
      rdmaop->sz[0] = msgsize + sizeof (CMI_VMI_Eager_Short_Slot_Footer_T);
      rdmaop->rbuffer = handle->data.send.data.eager_short.remote_buffer;
      rdmaop->roffset = offset;
      rdmaop->notify = FALSE;

      status = VMI_RDMA_Put (process->connection, rdmaop, (PVOID) NULL, (VMIRDMACompleteNotification) NULL);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");

      process->eager_short_send_index = ((index + 1) % process->eager_short_send_size);
      process->eager_short_send_credits_available -= 1;

      handle = NULL;
    } else if (msgsize < CMI_VMI_Small_Message_Boundary) {
      addrs[0] = (PVOID) msg;
      sz[0] = msgsize;

      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[destrank])->connection, addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

      handle = NULL;
    } else {
      if (CMI_VMI_Eager_Protocol) {
	context = CONTEXTFIELD (msg);
	if (context) {
	  cacheentry = CMI_VMI_CacheEntry_From_Context (context);
	} else {
	  status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
	}
      } else {
	context = NULL;
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }

      handle = CMI_VMI_Handle_Allocate ();

      handle->refcount += 1;
      handle->msg = msg;
      handle->msgsize = msgsize;
      handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
      handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
      handle->data.send.data.stream.cacheentry = cacheentry;

      bufHandles[0] = cacheentry->bufferHandle;
      addrs[0] = (PVOID) msg;
      sz[0] = msgsize;

      CMI_VMI_AsyncMsgCount += 1;
      handle->refcount += 1;

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
  } else {
    if (CMI_VMI_Eager_Protocol) {
      context = CONTEXTFIELD (msg);
      if (context) {
	cacheentry = CMI_VMI_CacheEntry_From_Context (context);
      } else {
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }
    } else {
      context = NULL;
      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
    }

    index = process->eager_long_send_size - 1;
    handle = process->eager_long_send_handles[index];

    if ((process->eager_long_send_size > 0) && (msgsize < handle->data.send.data.eager_long.maxsize)) {
      process->eager_long_send_size = index;

      handle->msg = msg;
      handle->msgsize = msgsize;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
      handle->data.send.data.eager_long.cacheentry = cacheentry;

      status = VMI_RDMA_Alloc_Op (&rdmaop);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

      rdmaop->numBufs = 1;
      rdmaop->buffers[0] = cacheentry->bufferHandle;
      rdmaop->addr[0] = handle->msg;
      rdmaop->sz[0] = msgsize;
      rdmaop->rbuffer = handle->data.send.data.eager_long.remote_buffer;
      rdmaop->roffset = 0;
      rdmaop->notify = TRUE;

      CMI_VMI_AsyncMsgCount += 1;
      handle->refcount += 1;

      status = VMI_RDMA_Put (process->connection, rdmaop, (PVOID) handle, (VMIRDMACompleteNotification) CMI_VMI_RDMA_Put_Completion_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    } else {
      handle = CMI_VMI_Handle_Allocate ();

      handle->refcount += 1;
      handle->msg = msg;
      handle->msgsize = msgsize;
      handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
      handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMAGET;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
      handle->data.send.data.rdmaget.cacheentry = cacheentry;

      handle->refcount += 1;
      CMI_VMI_AsyncMsgCount += 1;

      publish_msg.type = CMI_VMI_PUBLISH_TYPE_GET;

      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[destrank])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
				        (UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
				        (ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
  }

  return ((CmiCommHandle) handle);
}



/**************************************************************************
**
*/
void CmiFreeSendFn (int destrank, int msgsize, char *msg)
{
  VMI_STATUS status;

  char *msgcopy;

  CMI_VMI_Process_T *process;

  PVMI_BUFFER bufHandles[2];
  PVOID addrs[2];
  ULONG sz[2];

  CMI_VMI_Handle_T *handle;

  PVMI_CACHE_ENTRY cacheentry;

  CMI_VMI_Publish_Message_T publish_msg;

  void *context;

  int index;

  PVMI_RDMA_OP rdmaop;

  int offset;

  CMI_VMI_Eager_Short_Slot_Footer_T footer;


  DEBUG_PRINT ("CmiFreeSendFn() called.\n");

  if (destrank == _Cmi_mype) {
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msg);
    return;
  }

  process = &CMI_VMI_Processes[destrank];

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (msg, 0);
#endif

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;
  CMI_VMI_MESSAGE_CREDITS (msg) = process->eager_short_receive_credits_replentish;
  process->eager_short_receive_credits_replentish = 0;

  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    if (process->eager_short_send_credits_available > 0) {
      index = process->eager_short_send_index;
      handle = process->eager_short_send_handles[index];

      memcpy (handle->msg, msg, msgsize);

      footer.msgsize = msgsize;
      footer.sentinel = CMI_VMI_EAGER_SHORT_SENTINEL_DATA;
      memcpy (handle->msg + msgsize, &footer, sizeof (CMI_VMI_Eager_Short_Slot_Footer_T));

      handle->msgsize = msgsize;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;

      cacheentry = handle->data.send.data.eager_short.cacheentry;
      rdmaop = handle->data.send.data.eager_short.rdmaop;

      offset = handle->data.send.data.eager_short.offset;
      offset += (CMI_VMI_Medium_Message_Boundary - msgsize);

      rdmaop->numBufs = 1;
      rdmaop->buffers[0] = cacheentry->bufferHandle;
      rdmaop->addr[0] = handle->msg;
      rdmaop->sz[0] = msgsize + sizeof (CMI_VMI_Eager_Short_Slot_Footer_T);
      rdmaop->rbuffer = handle->data.send.data.eager_short.remote_buffer;
      rdmaop->roffset = offset;
      rdmaop->notify = FALSE;

      status = VMI_RDMA_Put (process->connection, rdmaop, (PVOID) NULL, (VMIRDMACompleteNotification) NULL);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");

      process->eager_short_send_index = ((index + 1) % process->eager_short_send_size);
      process->eager_short_send_credits_available -= 1;

      CmiFree (msg);
    } else if (msgsize < CMI_VMI_Small_Message_Boundary) {
      addrs[0] = (PVOID) msg;
      sz[0] = msgsize;

      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[destrank])->connection, addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");

      CmiFree (msg);
    } else {
      if (CMI_VMI_Eager_Protocol) {
	context = CONTEXTFIELD (msg);
	if (context) {
	  cacheentry = CMI_VMI_CacheEntry_From_Context (context);
	} else {
	  status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
	}
      } else {
	context = NULL;
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }

      handle = CMI_VMI_Handle_Allocate ();

      /* Do NOT increment handle->refcount here! */
      handle->msg = msg;
      handle->msgsize = msgsize;
      handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
      handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_FREE;
      handle->data.send.data.stream.cacheentry = cacheentry;

      bufHandles[0] = cacheentry->bufferHandle;
      addrs[0] = (PVOID) msg;
      sz[0] = msgsize;

      handle->refcount += 1;
      CMI_VMI_AsyncMsgCount += 1;

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
  } else {
    if (CMI_VMI_Eager_Protocol) {
      context = CONTEXTFIELD (msg);
      if (context) {
	cacheentry = CMI_VMI_CacheEntry_From_Context (context);
      } else {
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }
    } else {
      context = NULL;
      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
    }

    index = process->eager_long_send_size - 1;
    handle = process->eager_long_send_handles[index];

    if ((process->eager_long_send_size > 0) && (msgsize < handle->data.send.data.eager_long.maxsize)) {
      process->eager_long_send_size = index;

      handle->msg = msg;
      handle->msgsize = msgsize;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_FREE;
      handle->data.send.data.eager_long.cacheentry = cacheentry;

      status = VMI_RDMA_Alloc_Op (&rdmaop);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

      rdmaop->numBufs = 1;
      rdmaop->buffers[0] = cacheentry->bufferHandle;
      rdmaop->addr[0] = handle->msg;
      rdmaop->sz[0] = msgsize;
      rdmaop->rbuffer = handle->data.send.data.eager_long.remote_buffer;
      rdmaop->roffset = 0;
      rdmaop->notify = TRUE;

      CMI_VMI_AsyncMsgCount += 1;
      /* Do NOT increment handle->refcount here! */

      status = VMI_RDMA_Put (process->connection, rdmaop, (PVOID) handle, (VMIRDMACompleteNotification) CMI_VMI_RDMA_Put_Completion_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Put()");
    } else {
      handle = CMI_VMI_Handle_Allocate ();

      /* Do NOT increment handle->refcount here! */
      handle->msg = msg;
      handle->msgsize = msgsize;
      handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
      handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMAGET;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_FREE;
      handle->data.send.data.rdmaget.cacheentry = cacheentry;

      handle->refcount += 1;
      CMI_VMI_AsyncMsgCount += 1;

      publish_msg.type = CMI_VMI_PUBLISH_TYPE_GET;

      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[destrank])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
				        (UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
				        (ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
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

  CMI_VMI_Publish_Message_T publish_msg;

  void *context;


  DEBUG_PRINT ("CmiSyncBroadcastFn() called.\n");

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;
  CMI_VMI_MESSAGE_CREDITS (msg) = 0;

  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    if (CMI_VMI_Eager_Protocol) {
      context = CONTEXTFIELD (msg);
      if (context) {
	cacheentry = CMI_VMI_CacheEntry_From_Context (context);
      } else {
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }
    } else {
      context = NULL;
      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
    }

    handle = CMI_VMI_Handle_Allocate ();

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
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

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#else
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#endif

    while (handle->refcount > 2) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    if (!context) {
      status = VMI_Cache_Deregister (cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
    }

    CMI_VMI_Handle_Deallocate (handle);
  } else {
    if (CMI_VMI_Eager_Protocol) {
      context = CONTEXTFIELD (msg);
      if (context) {
	cacheentry = CMI_VMI_CacheEntry_From_Context (context);
      } else {
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }
    } else {
      context = NULL;
      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
    }

    handle = CMI_VMI_Handle_Allocate ();

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type=CMI_VMI_SEND_HANDLE_TYPE_RDMABROADCAST;
    handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
    handle->data.send.data.rdmabroadcast.cacheentry = cacheentry;

    publish_msg.type = CMI_VMI_PUBLISH_TYPE_GET;

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

      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[destrank])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
#else
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[i])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[i])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
#endif

    while (handle->refcount > 2) {
      sched_yield ();
      status = VMI_Poll ();
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
    }

    if (!context) {
      status = VMI_Cache_Deregister (cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
    }

    CMI_VMI_Handle_Deallocate (handle);
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

  int i;

  PVMI_CACHE_ENTRY cacheentry;

  int childcount;
  int startrank;
  int destrank;

  CMI_VMI_Publish_Message_T publish_msg;

  void *context;


  DEBUG_PRINT ("CmiAsyncBroadcastFn() called.\n");

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;
  CMI_VMI_MESSAGE_CREDITS (msg) = 0;

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

      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[destrank])->connection, addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else
    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection, addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection, addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif

    handle = NULL;
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    if (CMI_VMI_Eager_Protocol) {
      context = CONTEXTFIELD (msg);
      if (context) {
	cacheentry = CMI_VMI_CacheEntry_From_Context (context);
      } else {
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }
    } else {
      context = NULL;
      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
    }

    handle = CMI_VMI_Handle_Allocate ();

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
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

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#else
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#endif
  } else {
    if (CMI_VMI_Eager_Protocol) {
      context = CONTEXTFIELD (msg);
      if (context) {
	cacheentry = CMI_VMI_CacheEntry_From_Context (context);
      } else {
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }
    } else {
      context = NULL;
      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
    }

    handle = CMI_VMI_Handle_Allocate ();

    handle->refcount += 1;
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_RDMABROADCAST;
    handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
    handle->data.send.data.rdmabroadcast.cacheentry = cacheentry;

    publish_msg.type = CMI_VMI_PUBLISH_TYPE_GET;

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

      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[destrank])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
#else
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[i])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[i])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
#endif
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

  CMI_VMI_Publish_Message_T publish_msg;

  void *context;


  DEBUG_PRINT ("CmiFreeBroadcastFn() called.\n");

  CMI_VMI_MESSAGE_TYPE (msg) = CMI_VMI_MESSAGE_TYPE_STANDARD;
  CMI_VMI_MESSAGE_CREDITS (msg) = 0;

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

      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[destrank])->connection, addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#else
    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection, addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection, addrs, sz, 1, msgsize);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
    }
#endif

    CmiFree (msg);
  } else if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    if (CMI_VMI_Eager_Protocol) {
      context = CONTEXTFIELD (msg);
      if (context) {
	cacheentry = CMI_VMI_CacheEntry_From_Context (context);
      } else {
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }
    } else {
      context = NULL;
      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
    }

    handle = CMI_VMI_Handle_Allocate ();

    /* Do NOT increment handle->refcount here! */
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_FREE;
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

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#else
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
#endif
  } else {
    if (CMI_VMI_Eager_Protocol) {
      context = CONTEXTFIELD (msg);
      if (context) {
	cacheentry = CMI_VMI_CacheEntry_From_Context (context);
      } else {
	status = VMI_Cache_Register (msg, msgsize, &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
      }
    } else {
      context = NULL;
      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");
    }

    handle = CMI_VMI_Handle_Allocate ();

    /* Do NOT increment handle->refcount here! */
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type=CMI_VMI_SEND_HANDLE_TYPE_RDMABROADCAST;
    handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_FREE;
    handle->data.send.data.rdmabroadcast.cacheentry = cacheentry;

    publish_msg.type = CMI_VMI_PUBLISH_TYPE_GET;

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

      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[destrank])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
#else
    handle->refcount += (_Cmi_numpes - 1);
    CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

    for (i = 0; i < _Cmi_mype; i++) {
      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[i])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }

    for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[i])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
#endif
  }
}



/**************************************************************************
**
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
**
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
** The idea here is that for short messages, just do a sync broadcast
** since internally all of the sends for short messages happen
** synchronously anyway and these are pretty quick due to high memory
** bus bandwidth; for medium messages, copy the message and enqueue the
** copy locally and then call FreeBroadcast to send the message to the
** other processes asynchronously (and free the message at an idle point
** in the future when doing periodic resource cleanup); for large messages,
** send synchronously to all other processes and then enqueue the actual
** message to avoid copying it.
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
#else
  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    msgcopy = CmiAlloc (msgsize);
    memcpy (msgcopy, msg, msgsize);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msgcopy);

    CmiFreeBroadcastFn (msgsize, msg);
  } else {
    CmiSyncBroadcastFn (msgsize, msg);
    CdsFifo_Enqueue (CpvAccess (CmiLocalQueue), msg);
  }
#endif
}



/**************************************************************************
**
*/
int CmiAsyncMsgSent (CmiCommHandle commhandle)
{
  CMI_VMI_Handle_T *handle;


  DEBUG_PRINT ("CmiAsyncMsgSent() called.\n");

  if (commhandle) {
    handle = (CMI_VMI_Handle_T *) commhandle;
    return (handle->refcount <= 2);
  }

  return (TRUE);
}



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
void CmiReleaseCommHandle (CmiCommHandle commhandle)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;

  void *context;

  int i;


  DEBUG_PRINT ("CmiReleaseCommHandle() called.\n");

  if (commhandle) {
    handle = (CMI_VMI_Handle_T *) commhandle;
    handle->refcount -= 1;

    if (handle->refcount <= 1) {
      if (handle->data.send.send_handle_type == CMI_VMI_SEND_HANDLE_TYPE_STREAM) {
	if (CMI_VMI_Eager_Protocol) {
	  context = CONTEXTFIELD (handle->msg);
	  if (!context) {
	    status = VMI_Cache_Deregister (handle->data.send.data.stream.cacheentry);
	    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	  }
	} else {
	  context = NULL;
	  status = VMI_Cache_Deregister (handle->data.send.data.stream.cacheentry);
	  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	}
      }

      if (handle->data.send.send_handle_type == CMI_VMI_SEND_HANDLE_TYPE_RDMAGET) {
	if (CMI_VMI_Eager_Protocol) {
	  context = CONTEXTFIELD (handle->msg);
	  if (!context) {
	    status = VMI_Cache_Deregister (handle->data.send.data.rdmaget.cacheentry);
	    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	  }
	} else {
	  context = NULL;
	  status = VMI_Cache_Deregister (handle->data.send.data.rdmaget.cacheentry);
	  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	}
      }

      if (handle->data.send.send_handle_type == CMI_VMI_SEND_HANDLE_TYPE_RDMABROADCAST) {
	if (CMI_VMI_Eager_Protocol) {
	  context = CONTEXTFIELD (handle->msg);
	  if (!context) {
	    status = VMI_Cache_Deregister (handle->data.send.data.rdmabroadcast.cacheentry);
	    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	  }
	} else {
	  context = NULL;
	  status = VMI_Cache_Deregister (handle->data.send.data.rdmabroadcast.cacheentry);
	  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	}
      }

      if (handle->data.send.send_handle_type == CMI_VMI_SEND_HANDLE_TYPE_EAGER_LONG) {
	if (CMI_VMI_Eager_Protocol) {
	  context = CONTEXTFIELD (handle->msg);
	  if (!context) {
	    status = VMI_Cache_Deregister (handle->data.send.data.eager_long.cacheentry);
	    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	  }
	} else {
	  context = NULL;
	  status = VMI_Cache_Deregister (handle->data.send.data.eager_long.cacheentry);
	  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
	}
      }

      if (handle->data.send.message_disposition == CMI_VMI_MESSAGE_DISPOSITION_FREE) {
	CmiFree (handle->msg);
      }

      CMI_VMI_Handle_Deallocate (handle);
    }
  }
}



/**************************************************************************
** This code must call VMI_Poll() to ensure forward progress of the message
** pumping loop.
*/
void *CmiGetNonLocal (void)
{
  VMI_STATUS status;

  CMI_VMI_Process_T *process;
  CMI_VMI_Handle_T *handle;

  int index;
  char *msg;
  CMI_VMI_Eager_Short_Slot_Footer_T *footer;
  int credits_temp;

  CMI_VMI_Credit_Message_T credit_msg;
  PVOID addrs[1];
  ULONG sz[1];

  int i;


  DEBUG_PRINT ("CmiGetNonLocal() called.\n");

  /* Check the eager pollset to see if any new messages have arrived. */
  for (i = 0; i < CMI_VMI_Eager_Short_Pollset_Size; i++) {
    /* Get the next process in the eager short pollset. */
    process = CMI_VMI_Eager_Short_Pollset[i];

    /* Examine the footer of the process's next eager short handle. */
    index = process->eager_short_receive_index;
    handle = process->eager_short_receive_handles[index];
    footer = handle->data.receive.data.eager_short.footer;

    /* Get data out of the eager short handle (and any after it). */
    while (footer->sentinel == CMI_VMI_EAGER_SHORT_SENTINEL_DATA) {
      /* Get a pointer to the start of the message data. */
      msg = (char *) ((void *) handle->data.receive.data.eager_short.footer - footer->msgsize);

      /* Deal with any eager send credits send with the message. */
      //credits_temp = CMI_VMI_MESSAGE_CREDITS (msg);
      //process->eager_short_send_credits_available += credits_temp;

      /* Set up the Converse memory fields prior to the message data. */
      SIZEFIELD (msg) = footer->msgsize;
      REFFIELD (msg) = 1;
      CONTEXTFIELD (msg) = handle;

      /* Mark the message footer as "received". */
      footer->sentinel = CMI_VMI_EAGER_SHORT_SENTINEL_RECEIVED;

      process->eager_short_count += 1;
      CMI_VMI_Message_Receive_Count += 1;

      /* Enqueue the message. */
      //CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), msg);
      CMI_VMI_Common_Receive (process->rank, footer->msgsize, msg);

      /* Examine the footer of the process's next eager short handle. */
      index = (index + 1) % process->eager_short_receive_size;
      process->eager_short_receive_index = index;
      handle = process->eager_short_receive_handles[index];
      footer = handle->data.receive.data.eager_short.footer;
    }
  }

  status = VMI_Poll ();
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");

  return (CdsFifo_Dequeue (CpvAccess (CMI_VMI_RemoteQueue)));
}



/**************************************************************************
**
*/
void CmiProbeLatencies ()
{
  VMI_STATUS status;

  CMI_VMI_Latency_Vector_Request_Message_T request_msg;
  PVOID addrs[1];
  ULONG sz[1];

  int i;


  DEBUG_PRINT ("CmiProbeLatencies() called.\n");

  CMI_VMI_Latency_Vectors_Received = 0;

  /* Send a latency request message to every process except ourself. */
  CMI_VMI_MESSAGE_TYPE (&request_msg) = CMI_VMI_MESSAGE_TYPE_LATENCY_VECTOR_REQUEST;
  CMI_VMI_MESSAGE_CREDITS (&request_msg) = 0;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (&request_msg, 0);
#endif

  addrs[0] = (PVOID) &request_msg;
  sz[0] = (ULONG) (sizeof (CMI_VMI_Latency_Vector_Request_Message_T));

  for (i = 0; i < _Cmi_mype; i++) {
    status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection, addrs, sz, 1, sizeof (CMI_VMI_Latency_Vector_Request_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
  }

  for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
    status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[i])->connection, addrs, sz, 1, sizeof (CMI_VMI_Latency_Vector_Request_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
  }

  (&CMI_VMI_Processes[_Cmi_mype])->latency_vector = (unsigned long *) malloc (_Cmi_numpes * sizeof (unsigned long));
  for (i = 0; i < _Cmi_numpes; i++) {
    if (i == _Cmi_mype) {
      (&CMI_VMI_Processes[_Cmi_mype])->latency_vector[i] = 0;
    } else {
      (&CMI_VMI_Processes[_Cmi_mype])->latency_vector[i] = VMI_CONNECT_ONE_WAY_LATENCY ((&CMI_VMI_Processes[i])->connection);
    }
  }

  while (CMI_VMI_Latency_Vectors_Received < (_Cmi_numpes - 1)) {
    status = VMI_Poll ();
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
  }
}



/**************************************************************************
**
*/
unsigned long CmiGetLatency (int process1, int process2)
{
  if ((&CMI_VMI_Processes[process1])->latency_vector) {
    return ((&CMI_VMI_Processes[process1])->latency_vector[process2]);
  } else {
    return (CMI_VMI_LATENCY_UNKNOWN);
  }
}



/**************************************************************************
**
*/
int CmiGetCluster (int process)
{
  return ((&CMI_VMI_Processes[process])->cluster);
}



#if CMK_PERSISTENT_COMM
/**************************************************************************
** done
*/
void CmiPersistentInit ()
{
  DEBUG_PRINT ("CmiPersistentInit() called.\n");
}



/**************************************************************************
** done
*/
PersistentHandle CmiCreatePersistent (int destrank, int maxsize)
{
  VMI_STATUS status;

  CMI_VMI_Persistent_Request_Message_T request_msg;
  PVOID addrs[1];
  ULONG sz[1];


  DEBUG_PRINT ("CmiCreatePersistent() called.\n");

  if (CMI_VMI_Eager_Protocol) {
    CMI_VMI_MESSAGE_TYPE (&request_msg) = CMI_VMI_MESSAGE_TYPE_PERSISTENT_REQUEST;
    CMI_VMI_MESSAGE_CREDITS (&request_msg) = 0;

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT (&request_msg, 0);
#endif

    request_msg.maxsize = maxsize;

    addrs[0] = (PVOID) &request_msg;
    sz[0] = (ULONG) (sizeof (CMI_VMI_Persistent_Request_Message_T));

    status = VMI_Stream_Send_Inline ((&CMI_VMI_Processes[destrank])->connection, addrs, sz, 1, sizeof (CMI_VMI_Persistent_Request_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send_Inline()");
  }

  return ((PersistentHandle) NULL);
}



/**************************************************************************
** done
*/
void CmiUsePersistentHandle (PersistentHandle *handle_array, int array_size)
{
  DEBUG_PRINT ("CmiUserPersistentHandle() called.\n");
}



/**************************************************************************
** done
*/
void CmiDestroyPersistent (PersistentHandle phandle)
{
  DEBUG_PRINT ("CmiDestroyPersistent() called.\n");
}



/**************************************************************************
** done
*/
void CmiDestroyAllPersistent ()
{
  DEBUG_PRINT ("CmiDestroyAllPersistent() called.\n");
}



/**************************************************************************
**
*/
PersistentReq CmiCreateReceiverPersistent (int maxsize)
{
  PersistentReq request;


  DEBUG_PRINT ("CmiCreateReceiverPersistent() called.\n");

  request.pe = _Cmi_mype;
  request.maxBytes = maxsize;

  return (request);
}



/**************************************************************************
**
*/
PersistentHandle CmiRegisterReceivePersistent (PersistentReq request)
{
  DEBUG_PRINT ("CmiRegisterReceiverPersistent() called.\n");

  if (CMI_VMI_Eager_Protocol) {
    CMI_VMI_Eager_Short_Setup (request.pe);

    if (request.maxBytes > CMI_VMI_Medium_Message_Boundary) {
      if (request.maxBytes < CMI_VMI_Eager_Long_Buffer_Size) {
	CMI_VMI_Eager_Long_Setup (request.pe, CMI_VMI_Eager_Long_Buffer_Size);
      } else {
	CMI_VMI_Eager_Long_Setup (request.pe, request.maxBytes);
      }
    }
  }

  return ((PersistentHandle) NULL);
}
#endif   /* CMK_PERSISTENT_COMM */



/**************************************************************************
**
*/
void CMI_VMI_Read_Environment ()
{
  char *value;

  int dummy1;
  int dummy2;


  DEBUG_PRINT ("CMI_VMI_Read_Environment() called.\n");

  /* Get the username for this process. */
  if (value = (getpwuid (getuid ()))->pw_name) {
    if (!(CMI_VMI_Username = strdup (value))) {
      CmiAbort ("Unable to allocate memory for the username.");
    }
  } else {
    CmiAbort ("Unable to get the username for this process.");
  }

  /* Get the program key. */
  if (value = getenv ("VMI_KEY")) {
    /* Free the default value established in ConverseInit() from argv[0]. */
    free (CMI_VMI_Program_Key);

    if (!(CMI_VMI_Program_Key = strdup (value))) {
      CmiAbort ("Unable to allocate memory for the program key.");
    }
  }

  /* Get the total number of processes in the computation. */
  if (value = getenv ("VMI_PROCS")) {
    _Cmi_numpes = atoi (value);
  } else {
    CmiAbort ("Unable to determine the number of processes in the computation (VMI_PROCS).");
  }

  /* Get parameters for runtime behavior that override default values. */
  if (value = getenv ("CMI_VMI_WAN_LATENCY")) {
    CMI_VMI_WAN_Latency = atoi (value);
  }

  if (value = getenv ("CMI_VMI_CLUSTER")) {
    CMI_VMI_Cluster = atoi (value);
  }

  if (value = getenv ("CMI_VMI_PROBE_CLUSTERS")) {
    CMI_VMI_Probe_Clusters = atoi (value);
  }

  if (value = getenv ("CMI_VMI_MEMORY_POOL")) {
    CMI_VMI_Memory_Pool = atoi (value);
  }

  if (value = getenv ("CMI_VMI_TERMINATE_VMI_HACK")) {
    CMI_VMI_Terminate_VMI_Hack = atoi (value);
  }

  if (value = getenv ("CMI_VMI_CONNECTION_TIMEOUT")) {
    CMI_VMI_Connection_Timeout = atoi (value);
  }

  if (value = getenv ("CMI_VMI_MAXIMUM_HANDLES")) {
    CMI_VMI_Maximum_Handles = atoi (value);
  }

  if (value = getenv ("CMI_VMI_SMALL_MESSAGE_BOUNDARY")) {
    CMI_VMI_Small_Message_Boundary = atoi (value);
  }

  if (value = getenv ("CMI_VMI_MEDIUM_MESSAGE_BOUNDARY")) {
    CMI_VMI_Medium_Message_Boundary = atoi (value);

    /*
      If the medium message boundary is greater than 65536 bytes,
      then reset it to 65536.  This is because the eager short
      footer sends the eager short msgsize as an unsigned short
      which is limited to a maximum value of 65536.

      (If this limit is unacceptable in the future, the best
      strategy is to use different boundary variables for
      "normal" and "eager" messages.)
    */
    if (CMI_VMI_Medium_Message_Boundary > 65536) {
      CMI_VMI_Medium_Message_Boundary = 65536;
    }
  }

  if (value = getenv ("CMI_VMI_EAGER_PROTOCOL")) {
    CMI_VMI_Eager_Protocol = atoi (value);
  }

  if (value = getenv ("CMI_VMI_EAGER_INTERVAL")) {
    CMI_VMI_Eager_Interval = atoi (value);
  }

  if (value = getenv ("CMI_VMI_EAGER_THRESHOLD")) {
    CMI_VMI_Eager_Threshold = atoi (value);
  }

  if (value = getenv ("CMI_VMI_EAGER_SHORT_POLLSET_SIZE_MAXIMUM")) {
    CMI_VMI_Eager_Short_Pollset_Size_Maximum = atoi (value);
  }

  if (value = getenv ("CMI_VMI_EAGER_SHORT_SLOTS")) {
    CMI_VMI_Eager_Short_Slots = atoi (value);
  }

  if (value = getenv ("CMI_VMI_EAGER_LONG_BUFFERS")) {
    CMI_VMI_Eager_Long_Buffers = atoi (value);
  }

  if (value = getenv ("CMI_VMI_EAGER_LONG_BUFFER_SIZE")) {
    CMI_VMI_Eager_Long_Buffer_Size = atoi (value);
  }

  /* Figure out the startup type. */
  value = getenv ("CRM");
  if (value) {
    CMI_VMI_Startup_Type = CMI_VMI_STARTUP_TYPE_CRM;
    if (strstr (value, ":")) {
      CMI_VMI_CRM_Hostname = strdup (value);
      dummy1 = 0;
      while (CMI_VMI_CRM_Hostname[dummy1] != ':') {
	dummy1 += 1;
      }
      CMI_VMI_CRM_Hostname[dummy1] = 0;
      CMI_VMI_CRM_Port = atoi (value + dummy1 + 1);
    } else {
      CMI_VMI_CRM_Hostname = strdup (value);
      CMI_VMI_CRM_Port = CMI_VMI_CRM_PORT;
    }
    return;
  }

  value = getenv ("NETSTART");
  if (value) {
    CMI_VMI_Startup_Type = CMI_VMI_STARTUP_TYPE_CHARMRUN;
    sscanf (value, "%d%s%d%d%d", &_Cmi_mype, CMI_VMI_Charmrun_IP, &CMI_VMI_Charmrun_Port, &dummy1, &dummy2);
    return;
  }
}



/**************************************************************************
**
*/
int CMI_VMI_Startup_CRM ()
{
  pid_t myPID;

  struct hostent *host_ent;

  struct sockaddr_in serv_addr;
  int rc;

  CMI_VMI_CRM_Register_Message_T msg_register;

  int msg_code;
  int msg_error;

  int msg_numpes;

  CMI_VMI_CRM_Nodeblock_Message_T *msg_nodeblock;

  char crm_ip[1024];

  struct sockaddr_in local;
  socklen_t sockaddr_len;
  int myIP;

  int i;


  DEBUG_PRINT ("CMI_VMI_Startup_CRM() called.\n");

  myPID = getpid ();

  CMI_VMI_CRM_Socket = socket (AF_INET, SOCK_STREAM, 0);
  if (CMI_VMI_CRM_Socket < 0) {
    DEBUG_PRINT ("Error opening socket to CRM.\n");
    return (-1);
  }

  host_ent = gethostbyname (CMI_VMI_CRM_Hostname);
  if (!host_ent) {
    DEBUG_PRINT ("Error in gethostbyname() while contacting CRM.\n");
    return (-1);
  }

  strcpy (crm_ip, inet_ntoa (*((struct in_addr *) host_ent->h_addr_list[0])));

  //memset ((void *) &serv_addr, 0, sizeof (serv_addr));
  memset ((void *) &serv_addr, 0, sizeof (struct sockaddr_in));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = inet_addr (crm_ip);
  serv_addr.sin_port = htons (CMI_VMI_CRM_Port);

  //rc = connect (CMI_VMI_CRM_Socket, (struct sockaddr *) &serv_addr, sizeof (serv_addr));
  rc = connect (CMI_VMI_CRM_Socket, (struct sockaddr *) &serv_addr, sizeof (struct sockaddr_in));
  if (rc < 0) {
    DEBUG_PRINT ("Error connecting to CRM.\n");
    return (rc);
  }

  memset ((void *) &local, 0, sizeof (struct sockaddr_in));
  sockaddr_len = sizeof (struct sockaddr_in);
  rc = getsockname (CMI_VMI_CRM_Socket, (struct sockaddr *) &local, &sockaddr_len);
  if (rc < 0) {
    DEBUG_PRINT ("Error getting local TCP/IP address while synchronizing with CRM.\n");
    close (CMI_VMI_CRM_Socket);
    return (rc);
  }
  myIP = (int) local.sin_addr.s_addr;

  msg_code = htonl (CMI_VMI_CRM_MESSAGE_REGISTER);

  rc = CMI_VMI_Socket_Send (CMI_VMI_CRM_Socket, (const void *) &msg_code, sizeof (int));
  if (rc < 0) {
    DEBUG_PRINT ("Error sending to CRM.\n");
    close (CMI_VMI_CRM_Socket);
    return (rc);
  }

  msg_register.numpes = htonl (_Cmi_numpes);
  msg_register.cluster = htonl (CMI_VMI_Cluster);
  msg_register.node_context = htonl (myPID);
  msg_register.key_length = htonl (strlen (CMI_VMI_Program_Key));
  strcpy ((char *) &msg_register.key, CMI_VMI_Program_Key);

  rc = CMI_VMI_Socket_Send (CMI_VMI_CRM_Socket, (const void *) &msg_register, ((4 * sizeof (int)) + strlen (CMI_VMI_Program_Key)));
  if (rc < 0) {
    DEBUG_PRINT ("Error sending to CRM.\n");
    close (CMI_VMI_CRM_Socket);
    return (rc);
  }

  rc = CMI_VMI_Socket_Receive (CMI_VMI_CRM_Socket, &msg_code, sizeof (int));
  if (rc < 0) {
    DEBUG_PRINT ("Error receiveing from CRM.\n");
    close (CMI_VMI_CRM_Socket);
    return (rc);
  }

  msg_code = ntohl (msg_code);

  switch (msg_code)
  {
    case CMI_VMI_CRM_MESSAGE_SUCCESS:
      rc = CMI_VMI_Socket_Receive (CMI_VMI_CRM_Socket, &msg_numpes, sizeof (int));
      if (rc < 0) {
	DEBUG_PRINT ("Error receiveing from CRM.\n");
	close (CMI_VMI_CRM_Socket);
	return (rc);
      }

      msg_numpes = ntohl (msg_numpes);

      msg_nodeblock = malloc (msg_numpes * sizeof (CMI_VMI_CRM_Nodeblock_Message_T));
      if (!msg_nodeblock) {
	DEBUG_PRINT ("Unable to allocate memory to receive nodeblock from CRM.\n");
	close (CMI_VMI_CRM_Socket);
	return (-1);
      }

      rc = CMI_VMI_Socket_Receive (CMI_VMI_CRM_Socket, msg_nodeblock, msg_numpes * sizeof (CMI_VMI_CRM_Nodeblock_Message_T));
      if (rc < 0) {
	DEBUG_PRINT ("Error receiveing from CRM.\n");
	close (CMI_VMI_CRM_Socket);
	return (rc);
      }

      _Cmi_mype = -1;
      for (i = 0; i < msg_numpes; i++) {
	(&CMI_VMI_Processes[i])->rank = i;
	(&CMI_VMI_Processes[i])->node_IP = (&msg_nodeblock[i])->node_IP;
	(&CMI_VMI_Processes[i])->cluster = ntohl ((&msg_nodeblock[i])->cluster);

	(&msg_nodeblock[i])->node_context = ntohl ((&msg_nodeblock[i])->node_context);

	if (((&msg_nodeblock[i])->node_IP == myIP) && ((&msg_nodeblock[i])->node_context) == myPID) {
	  _Cmi_mype = i;
	}
      }

      free (msg_nodeblock);

      close (CMI_VMI_CRM_Socket);

      return (0);

      break;

    case CMI_VMI_CRM_MESSAGE_FAILURE:
      rc = CMI_VMI_Socket_Receive (CMI_VMI_CRM_Socket, &msg_error, sizeof (int));
      if (rc < 0) {
	DEBUG_PRINT ("Error receiveing from CRM.\n");
	close (CMI_VMI_CRM_Socket);
	return (rc);
      }

      msg_error = ntohl (msg_error);

      switch (msg_error)
      {
        case CMI_VMI_CRM_ERROR_CONFLICT:
	  CmiPrintf ("Error synchronizing with CRM (key/# PE conflict).\n");
	  break;

        case CMI_VMI_CRM_ERROR_TIMEOUT:
	  CmiPrintf ("Error synchronizing with CRM (timeout).\n");
	  break;

        default:
	  CmiPrintf ("Error synchronizing with CRM (unknown problem).\n");
	  break;
      }

      return (-1);

      close (CMI_VMI_CRM_Socket);

      break;

    default:
      printf ("Unknown message code received from CRM.\n");
      return (-1);
      break;
  }

  /* Should never get here! */
  return (-2);
}



/**************************************************************************
**
*/
int CMI_VMI_Startup_Charmrun ()
{
  pid_t myPID;

  CMI_VMI_Charmrun_Message_Header_T hdr;
  CMI_VMI_Charmrun_Register_Message_T msg_register;
  int msg_numnodes;
  CMI_VMI_Charmrun_Nodeblock_Message_T *msg_nodeblock;

  struct sockaddr_in serv_addr;
  int rc;

  int i;
  int j;


  DEBUG_PRINT ("CMI_VMI_Startup_Charmrun() called.\n");

  myPID = getpid ();

  CMI_VMI_Charmrun_Socket = socket (AF_INET, SOCK_STREAM, 0);
  if (CMI_VMI_Charmrun_Socket < 0) {
    DEBUG_PRINT ("Error opening socket to charmrun.\n");
    return (-1);
  }

  //bzero ((char *) &serv_addr, sizeof (serv_addr));
  memset ((void *) &serv_addr, 0, sizeof (struct sockaddr_in));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = inet_addr (CMI_VMI_Charmrun_IP);
  serv_addr.sin_port = htons (CMI_VMI_Charmrun_Port);

  rc = connect (CMI_VMI_Charmrun_Socket, (struct sockaddr *) &serv_addr, sizeof (serv_addr));
  if (rc < 0) {
    DEBUG_PRINT ("Error connecting to charmrun.\n");
    return (rc);
  }

  hdr.msg_len = htonl (sizeof (CMI_VMI_Charmrun_Register_Message_T));
  strcpy (hdr.msg_type, "initnode");

  msg_register.node_number = htonl (_Cmi_mype);   /* the rank of this PE */
  msg_register.numpes = htonl (0);                /* ignored */
  msg_register.dataport = htonl (myPID);          /* not used by vmi-linux -- must not be 0! */
  msg_register.mach_id = htonl (0);               /* not used by vmi-linux */
  msg_register.node_IP = htonl (0);               /* ignored */

  rc = CMI_VMI_Socket_Send (CMI_VMI_Charmrun_Socket, (const void *) &hdr, sizeof (CMI_VMI_Charmrun_Message_Header_T));
  if (rc < 0) {
    DEBUG_PRINT ("Error sending to charmrun.\n");
    return (rc);
  }
  rc = CMI_VMI_Socket_Send (CMI_VMI_Charmrun_Socket, (const void *) &msg_register, sizeof (CMI_VMI_Charmrun_Register_Message_T));
  if (rc < 0) {
    DEBUG_PRINT ("Error sending to charmrun.\n");
    return (rc);
  }

  rc = CMI_VMI_Socket_Receive (CMI_VMI_Charmrun_Socket, (void *) &hdr, sizeof (CMI_VMI_Charmrun_Message_Header_T));
  if (rc < 0) {
    DEBUG_PRINT ("Error receiving from charmrun.\n");
    return (rc);
  }
  rc = CMI_VMI_Socket_Receive (CMI_VMI_Charmrun_Socket, (void *) &msg_numnodes, sizeof (int));
  if (rc < 0) {
    DEBUG_PRINT ("Error receiving from charmrun.\n");
    return (rc);
  }

  msg_numnodes = ntohl (msg_numnodes);

  msg_nodeblock = (CMI_VMI_Charmrun_Nodeblock_Message_T *) malloc (msg_numnodes * sizeof (CMI_VMI_Charmrun_Nodeblock_Message_T));

  rc = CMI_VMI_Socket_Receive (CMI_VMI_Charmrun_Socket, (void *) msg_nodeblock, (msg_numnodes * sizeof (CMI_VMI_Charmrun_Nodeblock_Message_T)));
  if (rc < 0) {
    DEBUG_PRINT ("Error receiving from charmrun.\n");
    return (rc);
  }

  for (i = 0; i < msg_numnodes; i++) {
    (&CMI_VMI_Processes[i])->node_IP = (&msg_nodeblock[i])->node_IP;
    (&CMI_VMI_Processes[i])->rank = i;
  }

  free (msg_nodeblock);

  /* Return successfully. */
  return (0);
}



/**************************************************************************
** This function initializes VMI.  It assumes that we know our rank in the
** computation (i.e., _Cmi_mype is set).
**
** We need a unique VMI key for each process, so we use
** "[syncronization key]:[process rank]" for each processes's key.
** This enables us to figure out each process's key later when we connect
** to that process.
*/
int CMI_VMI_Initialize_VMI ()
{
  VMI_STATUS status;

  char *vmi_key;
  char *vmi_inlined_data_size;


  DEBUG_PRINT ("CMI_VMI_Initialize_VMI() called.\n");

  /* Set the VMI_KEY environment variable. */
  vmi_key = (char *) malloc ((strlen (CMI_VMI_Program_Key)) + 32);
  if (!vmi_key) {
    DEBUG_PRINT ("Unable to allocate memory for VMI key.");
    return (-1);
  }

  sprintf (vmi_key, "VMI_KEY=%s:%d\0", CMI_VMI_Program_Key, _Cmi_mype);

  if (putenv (vmi_key) < 0) {
    DEBUG_PRINT ("Unable to set VMI_KEY environment variable.");
    return (-1);
  }

  /* Set the maximum size of inlined stream messages. */
  vmi_inlined_data_size = (char *) malloc (32);
  if (!vmi_inlined_data_size) {
    DEBUG_PRINT ("Unable to allocate memory for VMI inlined data size.");
    return (-1);
  }

  sprintf (vmi_inlined_data_size, "VMI_INLINED_DATA_SZ=%d\0", CMI_VMI_Medium_Message_Boundary);

  if (putenv (vmi_inlined_data_size) < 0) {
    DEBUG_PRINT ("Unable to set VMI_INLINED_DATA_SZ environment variable.");
    return (-1);
  }

  DEBUG_PRINT ("Initializing VMI with key %s.\n", vmi_key);

  /* Initialize VMI. */
  status = VMI_Init (0, NULL);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Init()");

  /* Set a connection accept function. */
  status = VMI_Connection_Accept_Fn (CMI_VMI_Connection_Handler);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Connection_Accept_Fn()");

  /* Set a connection disconnect function. */
  status = VMI_Connection_Disconnect_Fn (CMI_VMI_Disconnection_Handler);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Connection_Disconnect_Fn()");

  /* Set a stream receive function. */
  VMI_STREAM_SET_RECV_FUNCTION (CMI_VMI_Stream_Notification_Handler);

  /* Create buffer pools. */
  if (CMI_VMI_Memory_Pool) {
    status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET1_SIZE, sizeof (PVOID), CMI_VMI_BUCKET1_PREALLOCATE,
					  CMI_VMI_BUCKET1_GROW, VMI_POOL_CLEARONCE, &CMI_VMI_Bucket1_Pool);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

    status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET2_SIZE, sizeof (PVOID), CMI_VMI_BUCKET2_PREALLOCATE,
					  CMI_VMI_BUCKET2_GROW, VMI_POOL_CLEARONCE, &CMI_VMI_Bucket2_Pool);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

    status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET3_SIZE, sizeof (PVOID), CMI_VMI_BUCKET3_PREALLOCATE,
					  CMI_VMI_BUCKET3_GROW, VMI_POOL_CLEARONCE, &CMI_VMI_Bucket3_Pool);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

    status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET4_SIZE, sizeof (PVOID), CMI_VMI_BUCKET4_PREALLOCATE,
					  CMI_VMI_BUCKET4_GROW, VMI_POOL_CLEARONCE, &CMI_VMI_Bucket4_Pool);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");

    status = VMI_Pool_Create_Buffer_Pool (CMI_VMI_BUCKET5_SIZE, sizeof (PVOID), CMI_VMI_BUCKET5_PREALLOCATE,
					  CMI_VMI_BUCKET5_GROW, VMI_POOL_CLEARONCE, &CMI_VMI_Bucket5_Pool);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Pool_Create_Buffer_Pool()");
  }

  /* Free memory. */
  free (vmi_inlined_data_size);
  free (vmi_key);

  /* Return successfully. */
  return (0);
}



/**************************************************************************
**
*/
int CMI_VMI_Terminate_VMI ()
{
  VMI_STATUS status;


  DEBUG_PRINT ("CMI_VMI_Terminate_VMI() called.\n");

  /* Release memory used by buffer pools. */
  if (CMI_VMI_Memory_Pool) {
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
  }

  /* Terminate VMI. */
  SET_VMI_SUCCESS (status);
  VMI_Terminate (status);


  /* Return successfully. */
  return (0);
}



/**************************************************************************
**
*/
int CMI_VMI_Socket_Send (int sockfd, const void *msg, int size)
{
  int sent;
  int rc;


  DEBUG_PRINT ("CMI_VMI_Socket_Send() called.\n");

  sent = 0;
  while (sent < size) {
    rc = send (sockfd, (const void *) (msg + sent), (size - sent), 0);
    if (rc < 0) {
      return (rc);
    } else {
      sent += rc;
    }
  }

  return (sent);
}



/**************************************************************************
**
*/
int CMI_VMI_Socket_Receive (int sockfd, void *msg, int size)
{
  int received;
  int rc;


  DEBUG_PRINT ("CMI_VMI_Socket_Receive() called.\n");

  received = 0;
  while (received < size) {
    rc = recv (sockfd, (void *) (msg + received), (size - received), 0);
    if (rc < 0) {
      return (rc);
    } else {
      received += rc;
    }
  }

  return (received);
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
int CMI_VMI_Open_Connections ()
{
  VMI_STATUS status;

  char *remote_key;

  PVMI_BUFFER connect_message_buffer;
  CMI_VMI_Connect_Message_T *connect_message_data;

  int i;

  //CMI_VMI_Process_T *process;

  struct timeval tp;
  long start_time;
  long now_time;
  BOOLEAN pending;


  DEBUG_PRINT ("CMI_VMI_Open_Connections() called.\n");

  /* Allocate space for the remote key. */
  remote_key = malloc ((strlen (CMI_VMI_Program_Key)) + 32);
  if (!remote_key) {
    DEBUG_PRINT ("Unable to allocate memory for remote key.\n");
    return (-1);
  }

  /* Allocate a buffer for connection message. */
  status = VMI_Buffer_Allocate (sizeof (CMI_VMI_Connect_Message_T), &connect_message_buffer);
  if (!VMI_SUCCESS (status)) {
    DEBUG_PRINT ("Unable to allocate connection message buffer.\n");
    free (remote_key);
    return (-1);
  }

  /* Set up the connection message field. */
  connect_message_data = (CMI_VMI_Connect_Message_T *) VMI_BUFFER_ADDRESS (connect_message_buffer);
  connect_message_data->rank = htonl (_Cmi_mype);

  /* Open connections to every process with a lower rank than ours. */
  for (i = 0; i < _Cmi_mype; i++) {
    /* Construct a remote VMI key in terms of the program key and peer's rank */
    sprintf (remote_key, "%s:%u\0", CMI_VMI_Program_Key, (&CMI_VMI_Processes[i])->rank);

    CMI_VMI_Open_Connection (i, remote_key, connect_message_buffer);

    DEBUG_PRINT ("Issued a connection to process %d:\n", i);
    DEBUG_PRINT ("\tRank - %d\n", (&CMI_VMI_Processes[i])->rank);
    DEBUG_PRINT ("\tKey - %s\n", remote_key);
    DEBUG_PRINT ("\tHostname - %s\n", remote_host->h_name);
    DEBUG_PRINT ("\tIP - [%d.%d.%d.%d].\n", ((process->node_IP >>  0) & 0xFF),
		                            ((process->node_IP >>  8) & 0xFF),
		                            ((process->node_IP >> 16) & 0xFF),
		                            ((process->node_IP >> 24) & 0xFF));
  }

  /* Set the connection state to ourself to "connected". */
  (&CMI_VMI_Processes[_Cmi_mype])->connection_state = CMI_VMI_CONNECTION_CONNECTED;

  /* Wait for connections.  */
  gettimeofday (&tp, NULL);
  start_time = tp.tv_sec;
  now_time   = tp.tv_sec;
  pending    = TRUE;

  while (pending && ((start_time + CMI_VMI_Connection_Timeout) > now_time)) {
    sched_yield ();
    status = VMI_Poll ();
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");

    gettimeofday (&tp, NULL);
    now_time = tp.tv_sec;

    pending = FALSE;
    for (i = 0; i < _Cmi_numpes; i++) {
      pending = pending || ((&CMI_VMI_Processes[i])->connection_state != CMI_VMI_CONNECTION_CONNECTED);
    }

    /* Every 30 seconds, retry any outgoing connections that had errors. */
    if (pending && ((now_time - start_time) % 30 == 0)) {
      for (i = 0; i < _Cmi_mype; i++) {
	if ((&CMI_VMI_Processes[i])->connection_state == CMI_VMI_CONNECTION_ERROR) {
	  sprintf (remote_key, "%s:%u\0", CMI_VMI_Program_Key, (&CMI_VMI_Processes[i])->rank);
	  CMI_VMI_Open_Connection (i, remote_key, connect_message_buffer);
	}
      }
    }
  }

  /* Free memory. */
  free (remote_key);
  VMI_Buffer_Deallocate (connect_message_buffer);

  /* Verify that there were no connection problems. */
  if (pending) {
    DEBUG_PRINT ("There were connection errors for process %d.\n", _Cmi_mype);
    return (-1);
  }

  DEBUG_PRINT ("All connections are open for process %d.\n", _Cmi_mype);

  return (0);
}



/**************************************************************************
**
*/
int CMI_VMI_Open_Connection (int remote_rank, char *remote_key, PVMI_BUFFER connect_message_buffer)
{
  VMI_STATUS status;

  CMI_VMI_Process_T *process;
  struct hostent *remote_host;
  PVMI_NETADDRESS remote_address;


  DEBUG_PRINT ("CMI_VMI_Open_Connection() called.\n");

  /* Get a pointer to the process to make things easier. */
  process = &CMI_VMI_Processes[remote_rank];

  /* Allocate a connection object */
  status = VMI_Connection_Create (&process->connection);
  if (!VMI_SUCCESS (status)) {
    DEBUG_PRINT ("Unable to create connection to process %d.\n", remote_rank);
    return (-1);
  }

  /* Build the remote IPV4 address. We need remote hosts name for this. */
  remote_host = gethostbyaddr (&process->node_IP, sizeof (process->node_IP), AF_INET);
  if (!remote_host) {
    DEBUG_PRINT ("Error looking up host [%d.%d.%d.%d].\n", ((process->node_IP >>  0) & 0xFF),
		                                           ((process->node_IP >>  8) & 0xFF),
		                                           ((process->node_IP >> 16) & 0xFF),
		                                           ((process->node_IP >> 24) & 0xFF));
    return (-1);
  }

  /* Allocate a remote IPv4 NETADDRESS. */
  status = VMI_Connection_Allocate_IPV4_Address (remote_host->h_name, 0, CMI_VMI_Username, remote_key, &remote_address);
  if (!VMI_SUCCESS (status)) {
    DEBUG_PRINT ("Unable to allocate remote node IP V4 address.\n");
    return (-1);
  }

  /* Now bind the local and remote addresses. */
  status = VMI_Connection_Bind (*localAddress, *remote_address, process->connection);
  if (!VMI_SUCCESS (status)) {
    DEBUG_PRINT ("Error binding connection for process %d.\n", i);
    return (-1);
  }

  /*
    Do this here to avoid a race condition where we complete the connect right
    away and then set the state here to CMI_VMI_CONNECTION_STATE_CONNECTING.
  */
  process->connection_state = CMI_VMI_CONNECTION_CONNECTING;

  /* Issue the actual connection request. */
  status = VMI_Connection_Issue (process->connection, connect_message_buffer, (VMIConnectIssue) CMI_VMI_Connection_Response_Handler, process);
  if (!VMI_SUCCESS (status)) {
    DEBUG_PRINT ("Error issuing connection for process %d.\n", i);
    return (-1);
  }
}



/**************************************************************************
** This function is invoked asynchronously to handle an incoming connection
** request.
*/
VMI_CONNECT_RESPONSE CMI_VMI_Connection_Handler (PVMI_CONNECT connection, PVMI_SLAB slab, ULONG data_size)
{
  VMI_STATUS status;

  CMI_VMI_Connect_Message_T *data;
  ULONG rank;
  ULONG size;
  PVMI_SLAB_STATE state;


  DEBUG_PRINT ("CMI_VMI_Connection_Handler() called.\n");

  /* Initialize the number of bytes expected from the connect data. */
  size = sizeof (CMI_VMI_Connect_Message_T);

  /* Make sure we received the expected number of bytes. */
  if (data_size != size) {
    return VMI_CONNECT_RESPONSE_ERROR;
  }

  /* Allocate connection data structure. */
  data = (CMI_VMI_Connect_Message_T *) malloc (size);
  if (!data) {
    return VMI_CONNECT_RESPONSE_ERROR;
  }

  /* Save the slab state prior to reading. */
  status = VMI_Slab_Save_State (slab, &state);
  if (!VMI_SUCCESS (status)) {
    free (data);
    return VMI_CONNECT_RESPONSE_ERROR;
  }

  /* Copy connect data. */
  status = VMI_Slab_Copy_Bytes (slab, size, data);
  if (!VMI_SUCCESS (status)) {
    VMI_Slab_Restore_State (slab, state);
    free (data);
    return VMI_CONNECT_RESPONSE_ERROR;
  }

  /* Get rank of connecting process. */
  rank = ntohl (data->rank);

  DEBUG_PRINT ("Accepting a connection request from rank %u.\n", rank);

  /* Update the connection state. */
  (&CMI_VMI_Processes[rank])->connection = connection;
  (&CMI_VMI_Processes[rank])->connection_state = CMI_VMI_CONNECTION_CONNECTED;


  VMI_CONNECT_SET_RECEIVE_CONTEXT (connection, (&CMI_VMI_Processes[rank]));

  /* A bug in VMI 2.1 prevents the following three calls from returning a proper return code. */
  status = VMI_RDMA_Set_Publish_Callback (connection, (VMIRDMABuffer) CMI_VMI_RDMA_Publish_Handler);
  /* CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Publish_Callback()"); */

  status = VMI_RDMA_Set_Put_Notification_Callback (connection, CMI_VMI_RDMA_Put_Notification_Handler);
  /* CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Put_Notification_Callback()"); */

  status = VMI_RDMA_Set_Get_Notification_Callback (connection, CMI_VMI_RDMA_Get_Notification_Handler);
  /* CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Get_Notification_Callback()"); */

  /* Free the connect data buffer. */
  free (data);

  /* Accepted the connection. */
  return VMI_CONNECT_RESPONSE_ACCEPT;
}



/**************************************************************************
** This function is invoked asynchronously to handle a process's response
** to our connection request.
*/
void CMI_VMI_Connection_Response_Handler (PVOID context, PVOID response, USHORT size, PVOID handle, VMI_CONNECT_RESPONSE sstatus)
{
  VMI_STATUS status;

  CMI_VMI_Process_T *process;


  DEBUG_PRINT ("CMI_VMI_Connection_Response_Handler() called.\n");

  /* Cast the context to a CMI_VMI_Process_Info pointer. */
  process = (CMI_VMI_Process_T *) context;

  switch (sstatus)
  {
    case VMI_CONNECT_RESPONSE_ACCEPT:
      DEBUG_PRINT ("Process %d accepted connection.\n", process->rank);

      /* Update the connection state. */
      process->connection_state = CMI_VMI_CONNECTION_CONNECTED;

      VMI_CONNECT_SET_RECEIVE_CONTEXT (process->connection, process);

      /* A bug in VMI 2.1 prevents the following three calls from returning a proper return code. */
      status = VMI_RDMA_Set_Publish_Callback (process->connection, CMI_VMI_RDMA_Publish_Handler);
      /* CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Set_Publish_Callback()"); */

      status = VMI_RDMA_Set_Put_Notification_Callback (process->connection, CMI_VMI_RDMA_Put_Notification_Handler);
      /* CMI_VMI_CHECK_SUCCESS (status,"VMI_RDMA_Set_Put_Notification_Callback()"); */

      status = VMI_RDMA_Set_Get_Notification_Callback (process->connection, CMI_VMI_RDMA_Get_Notification_Handler);
      /* CMI_VMI_CHECK_SUCCESS (status,"VMI_RDMA_Set_Get_Notification_Callback()"); */

      break;

    case VMI_CONNECT_RESPONSE_REJECT:
      DEBUG_PRINT ("Process %d rejected connection.\n", process->rank);

      /* Update the connection state. */
      process->connection_state = CMI_VMI_CONNECTION_DISCONNECTED;

      break;

    case VMI_CONNECT_RESPONSE_ERROR:
      DEBUG_PRINT ("Error connecting to process %d [%d.%d.%d.%d].\n", process->rank, ((process->node_IP >>  0) & 0xFF),
		                                                                     ((process->node_IP >>  8) & 0xFF),
		                                                                     ((process->node_IP >> 16) & 0xFF),
		                                                                     ((process->node_IP >> 24) & 0xFF));

      /* Update the connection state. */
      process->connection_state = CMI_VMI_CONNECTION_ERROR;

      break;

    default:
      DEBUG_PRINT ("Error connecting to process %d\n", process->rank);
      DEBUG_PRINT ("Error code 0x%08x\n", status);

      /* Update the connection state. */
      process->connection_state = CMI_VMI_CONNECTION_ERROR;

      break;
  }

  /* Deallocate the connection receive context. */
  VMI_Buffer_Deallocate ((PVMI_BUFFER) context);
}



/**************************************************************************
**
*/
int CMI_VMI_Close_Connections ()
{
  VMI_STATUS status;

  struct timeval tp;
  long start_time;
  long now_time;
  BOOLEAN pending;

  int i;


  DEBUG_PRINT ("CMI_VMI_Close_Connections() called.\n");

  /* Issue a disconnect request to each process with a lower rank. */
  for (i = 0; i < _Cmi_mype; i++) {
    (&CMI_VMI_Processes[i])->connection_state = CMI_VMI_CONNECTION_DISCONNECTING;

    status = VMI_Connection_Disconnect ((&CMI_VMI_Processes[i])->connection, CMI_VMI_Disconnection_Response_Handler, (PVOID) (&CMI_VMI_Processes[i]));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Connection_Disconnect()");
  }

  /* Set the connection state to ourself to "disconnected". */
  (&CMI_VMI_Processes[_Cmi_mype])->connection_state = CMI_VMI_CONNECTION_DISCONNECTED;

  /* Wait until all processes are disconnected or timeout occurs. */
  gettimeofday (&tp, NULL);
  start_time = tp.tv_sec;
  now_time   = tp.tv_sec;
  pending    = TRUE;

  while (pending && ((start_time + CMI_VMI_Connection_Timeout) > now_time)) {
    sched_yield ();
    status = VMI_Poll ();
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");

    gettimeofday (&tp, NULL);
    now_time = tp.tv_sec;

    pending = FALSE;
    for (i = 0; i < _Cmi_numpes; i++) {
      pending = (pending || ((&CMI_VMI_Processes[i])->connection_state != CMI_VMI_CONNECTION_DISCONNECTED));
    }
  }

  /* Return the return code. */
  if (pending) {
    return (-1);
  }

  return (0);
}



/**************************************************************************
** This function is invoked asynchronously to handle an incoming disconnect
** request.
*/
void CMI_VMI_Disconnection_Handler (PVMI_CONNECT connection)
{
  CMI_VMI_Process_T *process;


  DEBUG_PRINT ("CMI_VMI_Disconnection_Handler() called.\n");

  process = (CMI_VMI_Process_T *) VMI_CONNECT_GET_RECEIVE_CONTEXT (connection);
  process->connection_state = CMI_VMI_CONNECTION_DISCONNECTED;
}



/**************************************************************************
** This function is invoked asynchronously to handle a process's response
** to our disconnection request.
*/
void CMI_VMI_Disconnection_Response_Handler (PVMI_CONNECT connection, PVOID context, VMI_STATUS sstatus)
{
  CMI_VMI_Process_T *process;


  DEBUG_PRINT ("CMI_VMI_Disconnection_Response_Handler() called.\n");

  process = (CMI_VMI_Process_T *) context;
  process->connection_state = CMI_VMI_CONNECTION_DISCONNECTED;
}



/**************************************************************************
**
*/
void CMI_VMI_Reply_Latencies (int sourcerank)
{
  VMI_STATUS status;

  CMI_VMI_Latency_Vector_Reply_Message_T *reply_msg;
  int reply_msgsize;

  PVMI_CACHE_ENTRY cacheentry;

  PVMI_BUFFER bufHandles[1];
  PVOID addrs[1];
  ULONG sz[1];

  CMI_VMI_Handle_T *handle;

  int i;


  reply_msgsize = CmiMsgHeaderSizeBytes + (_Cmi_numpes * sizeof (unsigned long));
  reply_msg = (CMI_VMI_Latency_Vector_Reply_Message_T *) CmiAlloc (reply_msgsize);

  CMI_VMI_MESSAGE_TYPE (reply_msg) = CMI_VMI_MESSAGE_TYPE_LATENCY_VECTOR_REPLY;
  CMI_VMI_MESSAGE_CREDITS (reply_msg) = 0;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (reply_msg, 0);
#endif

  for (i = 0; i < _Cmi_numpes; i++) {
    if (i == _Cmi_mype) {
      reply_msg->latency[i] = 0;
    } else {
      reply_msg->latency[i] = VMI_CONNECT_ONE_WAY_LATENCY ((&CMI_VMI_Processes[i])->connection);
    }
  }

  status = VMI_Cache_Register (reply_msg, reply_msgsize, &cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

  handle = CMI_VMI_Handle_Allocate ();

  /* Do NOT increment handle->refcount here! */
  handle->msg = (char *) reply_msg;
  handle->msgsize = reply_msgsize;
  handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
  handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
  handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_FREE;
  handle->data.send.data.stream.cacheentry = cacheentry;

  bufHandles[0] = cacheentry->bufferHandle;
  addrs[0] = (PVOID) reply_msg;
  sz[0] = reply_msgsize;

  handle->refcount += 1;
  CMI_VMI_AsyncMsgCount += 1;

  status = VMI_Stream_Send ((&CMI_VMI_Processes[sourcerank])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
}



/**************************************************************************
**
*/
void CMI_VMI_Compute_Cluster_Mapping ()
{
  int i;
  int j;
  int cluster;
  BOOLEAN found_flag;


  for (i = 0; i < _Cmi_numpes; i++) {
    (&CMI_VMI_Processes[i])->cluster = CMI_VMI_CLUSTER_UNKNOWN;
  }

  cluster = 0;
  for (i = 0; i < _Cmi_numpes; i++) {
    found_flag = FALSE;
    for (j = 0; j < _Cmi_numpes; j++) {
      if ((&CMI_VMI_Processes[i])->latency_vector[j] < CMI_VMI_WAN_Latency) {
	if ((&CMI_VMI_Processes[j])->cluster == CMI_VMI_CLUSTER_UNKNOWN) {
	  (&CMI_VMI_Processes[j])->cluster = cluster;
	  found_flag = TRUE;
	}
      }
    }
    if (found_flag) {
      cluster += 1;
    }
  }
}



/**************************************************************************
**
*/
void CMI_VMI_Distribute_Cluster_Mapping ()
{
  VMI_STATUS status;

  CMI_VMI_Cluster_Mapping_Message_T *mapping_msg;
  int mapping_msgsize;

  PVMI_CACHE_ENTRY cacheentry;

  PVMI_BUFFER bufHandles[1];
  PVOID addrs[1];
  ULONG sz[1];

  CMI_VMI_Handle_T *handle;

  int i;


  mapping_msgsize = CmiMsgHeaderSizeBytes + (_Cmi_numpes * sizeof (int));
  mapping_msg = (CMI_VMI_Cluster_Mapping_Message_T *) CmiAlloc (mapping_msgsize);

  CMI_VMI_MESSAGE_TYPE (mapping_msg) = CMI_VMI_MESSAGE_TYPE_CLUSTER_MAPPING;
  CMI_VMI_MESSAGE_CREDITS (mapping_msg) = 0;

#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT (mapping_msg, 0);
#endif

  for (i = 0; i < _Cmi_numpes; i++) {
    mapping_msg->cluster[i] = (&CMI_VMI_Processes[i])->cluster;
  }

  status = VMI_Cache_Register (mapping_msg, mapping_msgsize, &cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

  handle = CMI_VMI_Handle_Allocate ();

  /* Do NOT increment handle->refcount here! */
  handle->msg = (char *) mapping_msg;
  handle->msgsize = mapping_msgsize;
  handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
  handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
  handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_FREE;
  handle->data.send.data.stream.cacheentry = cacheentry;

  bufHandles[0] = cacheentry->bufferHandle;
  addrs[0] = (PVOID) mapping_msg;
  sz[0] = mapping_msgsize;

  handle->refcount += (_Cmi_numpes - 1);
  CMI_VMI_AsyncMsgCount += (_Cmi_numpes - 1);

  for (i = 0; i < _Cmi_mype; i++) {
    status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
  }

  for (i = (_Cmi_mype + 1); i < _Cmi_numpes; i++) {
    status = VMI_Stream_Send ((&CMI_VMI_Processes[i])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
  }
}



/**************************************************************************
**
*/
void CMI_VMI_Wait_Cluster_Mapping ()
{
  VMI_STATUS status;


  while (!CMI_VMI_Cluster_Mapping_Received) {
    status = VMI_Poll ();
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Poll()");
  }
}



/**************************************************************************
**
*/
void *CMI_VMI_CmiAlloc (int request_size)
{
  VMI_STATUS status;

  int size;
  void *ptr;


  DEBUG_PRINT ("CMI_VMI_CmiAlloc() (memory pool version) called.\n");

  if (CMI_VMI_Eager_Protocol) {
    size = request_size + (sizeof (CMI_VMI_Memory_Chunk_T) - sizeof (CmiChunkHeader));
  } else {
    size = request_size;
  }

  if (CMI_VMI_Memory_Pool) {
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
  } else {
    ptr = malloc (size);
  }

  if (CMI_VMI_Eager_Protocol) {
    ptr += sizeof (CMI_VMI_Memory_Chunk_T);
    CONTEXTFIELD (ptr) = NULL;

    ptr -= sizeof (CmiChunkHeader);
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

  void *context;

  CMI_VMI_Process_T *process;
  CMI_VMI_Handle_T *handle;

  CMI_VMI_Eager_Short_Slot_Footer_T *footer;
  int sender_rank;
  int credits_temp;
  int index;

  PVMI_CACHE_ENTRY cacheentry;
  char *publish_buffer;
  int buffer_size;

  CMI_VMI_Publish_Message_T publish_msg;


  DEBUG_PRINT ("CMI_VMI_CmiFree() (memory pool version) called.\n");

  ptr += sizeof (CmiChunkHeader);

  if (CMI_VMI_Eager_Protocol) {
    size = SIZEFIELD (ptr) + sizeof (CMI_VMI_Memory_Chunk_T);
    context = CONTEXTFIELD (ptr);
  } else {
    size = SIZEFIELD (ptr);
    context = NULL;
  }

  if (context) {
    handle = (CMI_VMI_Handle_T *) context;

    if (handle->data.receive.receive_handle_type == CMI_VMI_RECEIVE_HANDLE_TYPE_EAGER_SHORT) {
      sender_rank = handle->data.receive.data.eager_short.sender_rank;
      process = &CMI_VMI_Processes[sender_rank];

      footer = handle->data.receive.data.eager_short.footer;
      footer->sentinel = CMI_VMI_EAGER_SHORT_SENTINEL_FREE;

      credits_temp = 0;
      index = process->eager_short_receive_dirty;
      handle = process->eager_short_receive_handles[index];
      footer = handle->data.receive.data.eager_short.footer;
      while (footer->sentinel == CMI_VMI_EAGER_SHORT_SENTINEL_FREE) {
	footer->sentinel = CMI_VMI_EAGER_SHORT_SENTINEL_READY;
	credits_temp += 1;

	index = (index + 1) % process->eager_short_receive_size;
	handle = process->eager_short_receive_handles[index];
	footer = handle->data.receive.data.eager_short.footer;
      }

      process->eager_short_receive_dirty = index;
      process->eager_short_receive_credits_replentish += credits_temp;
    } else {
      sender_rank = handle->data.receive.data.eager_long.sender_rank;
      process = &CMI_VMI_Processes[sender_rank];

      publish_buffer = handle->msg;
      buffer_size = handle->data.receive.data.eager_long.maxsize;
      cacheentry = handle->data.receive.data.eager_long.cacheentry;

      /* Fill in the publish data which will be sent to the sender. */
      publish_msg.type = CMI_VMI_PUBLISH_TYPE_EAGER_LONG;

      /* Publish the eager buffer to the sender. */
      status = VMI_RDMA_Publish_Buffer (process->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) publish_buffer,
					(UINT32) buffer_size, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index,
					(PVOID) &publish_msg, (ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
  } else {
    if (CMI_VMI_Eager_Protocol) {
      ptr -= sizeof (CMI_VMI_Memory_Chunk_T);
    } else {
      ptr -= sizeof (CmiChunkHeader);
    }

    if (CMI_VMI_Memory_Pool) {
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
    } else {
      free (ptr);
    }
  }
}



/**************************************************************************
**
*/
PVMI_CACHE_ENTRY CMI_VMI_CacheEntry_From_Context (void *context)
{
  CMI_VMI_Handle_T *handle;


  DEBUG_PRINT ("CMI_VMI_CacheEntry_From_Context() called.\n");

  handle = (CMI_VMI_Handle_T *) context;

  switch (handle->data.receive.receive_handle_type)
  {
    case CMI_VMI_RECEIVE_HANDLE_TYPE_EAGER_SHORT:
      return (handle->data.receive.data.eager_short.cacheentry);
      break;

    case CMI_VMI_RECEIVE_HANDLE_TYPE_EAGER_LONG:
      return (handle->data.receive.data.eager_long.cacheentry);
      break;

    default:
      CmiAbort ("CMI_VMI_CacheEntry_From_Context() called on invalid handle type.");
      break;
  }
}



/**************************************************************************
**
*/
CMI_VMI_Handle_T *CMI_VMI_Handle_Allocate ()
{
  VMI_STATUS status;

  int i;
  int j;


  DEBUG_PRINT ("CMI_VMI_Handle_Allocate() called.\n");

  i = CMI_VMI_Next_Handle;
  j = CMI_VMI_Next_Handle;
  while ((&CMI_VMI_Handles[i])->refcount > 0) {
    i = ((i + 1) % CMI_VMI_Maximum_Handles);

    if (i == j) {
      i = CMI_VMI_Maximum_Handles;
      CMI_VMI_Maximum_Handles *= 2;
      CMI_VMI_Handles = (CMI_VMI_Handle_T *) realloc (CMI_VMI_Handles, CMI_VMI_Maximum_Handles * sizeof (CMI_VMI_Handle_T));
      for (j = i; j < CMI_VMI_Maximum_Handles; j++) {
	(&CMI_VMI_Handles[j])->index = j;
	(&CMI_VMI_Handles[j])->refcount = 0;
      }
    }
  }

  (&CMI_VMI_Handles[i])->refcount = 1;

  CMI_VMI_Next_Handle = ((i + 1) % CMI_VMI_Maximum_Handles);

  return (&CMI_VMI_Handles[i]);
}



/**************************************************************************
**
*/
void CMI_VMI_Handle_Deallocate (CMI_VMI_Handle_T *handle)
{
  DEBUG_PRINT ("CMI_VMI_Handle_Deallocate() called.\n");

  handle->refcount = 0;
}



/**************************************************************************
**
*/
void CMI_VMI_Eager_Short_Setup (int sender_rank)
{
  VMI_STATUS status;

  CMI_VMI_Process_T *process;
  CMI_VMI_Handle_T *handle;

  int slot_size;
  int buffer_size;
  int i;
  int index;

  char *publish_buffer;
  char *eager_buffer;

  PVMI_CACHE_ENTRY cacheentry;

  CMI_VMI_Eager_Short_Slot_Footer_T *footer;

  CMI_VMI_Publish_Message_T publish_msg;


  DEBUG_PRINT ("CMI_VMI_Eager_Short_Setup() called.\n");

  /* Get a pointer to the sender's process table entry. */
  process = &CMI_VMI_Processes[sender_rank];

  /* Compute the size of each slot and the size of the total buffer. */
  slot_size = sizeof (CMI_VMI_Memory_Chunk_T) + CMI_VMI_Medium_Message_Boundary + sizeof (CMI_VMI_Eager_Short_Slot_Footer_T);
  buffer_size = CMI_VMI_Eager_Short_Slots * slot_size;

  /* Allocate the eager buffer. */
  publish_buffer = CmiAlloc (buffer_size);

  /* Register the eager buffer so it can be published. */
  status = VMI_Cache_Register (publish_buffer, buffer_size, &cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

  /* Allocate handles into buffer slots and set them up. */
  for (i = 0; i < CMI_VMI_Eager_Short_Slots; i++) {
    handle = CMI_VMI_Handle_Allocate ();

    eager_buffer = publish_buffer + (slot_size * i);
    footer = (CMI_VMI_Eager_Short_Slot_Footer_T *) (eager_buffer + sizeof (CMI_VMI_Memory_Chunk_T) + CMI_VMI_Medium_Message_Boundary);

    handle->refcount += 1;
    handle->msg = NULL;
    handle->msgsize = -1;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_RECEIVE;
    handle->data.receive.receive_handle_type = CMI_VMI_RECEIVE_HANDLE_TYPE_EAGER_SHORT;
    handle->data.receive.data.eager_short.sender_rank = sender_rank;
    handle->data.receive.data.eager_short.publish_buffer = publish_buffer;
    handle->data.receive.data.eager_short.cacheentry = cacheentry;
    handle->data.receive.data.eager_short.eager_buffer = eager_buffer;
    handle->data.receive.data.eager_short.footer = footer;

    footer->msgsize = -1;
    footer->sentinel = CMI_VMI_EAGER_SHORT_SENTINEL_READY;

    index = process->eager_short_receive_size;
    process->eager_short_receive_handles[index] = handle;
    process->eager_short_receive_size += 1;
  }

  /* Add the sender process to the poll set. */
  index = CMI_VMI_Eager_Short_Pollset_Size;
  CMI_VMI_Eager_Short_Pollset[index] = &CMI_VMI_Processes[sender_rank];
  CMI_VMI_Eager_Short_Pollset_Size += 1;

  /* Fill in the publish data which will be sent to the sender. */
  publish_msg.type = CMI_VMI_PUBLISH_TYPE_EAGER_SHORT;

  /* Publish the eager buffer to the sender. */
  status = VMI_RDMA_Publish_Buffer (process->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) publish_buffer,
				    (UINT32) buffer_size, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) 0, (PVOID) &publish_msg,
				    (ULONG) sizeof (CMI_VMI_Publish_Message_T));
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
}



/**************************************************************************
**
*/
void CMI_VMI_Eager_Long_Setup (int sender_rank, int maxsize)
{
  VMI_STATUS status;

  CMI_VMI_Process_T *process;
  CMI_VMI_Handle_T *handle;

  int i;
  int index;

  char *publish_buffer;

  PVMI_CACHE_ENTRY cacheentry;

  CMI_VMI_Publish_Message_T publish_msg;


  DEBUG_PRINT ("CMI_VMI_Eager_Long_Setup() called.\n");

  /* Get a pointer to the sender's process table entry. */
  process = &CMI_VMI_Processes[sender_rank];

  for (i = 0; i < CMI_VMI_Eager_Long_Buffers; i++) {
    publish_buffer = CmiAlloc (maxsize);

    status = VMI_Cache_Register (publish_buffer, maxsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle = CMI_VMI_Handle_Allocate ();

    handle->refcount += 1;
    handle->msg = publish_buffer;
    handle->msgsize = -1;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_RECEIVE;
    handle->data.receive.receive_handle_type = CMI_VMI_RECEIVE_HANDLE_TYPE_EAGER_LONG;
    handle->data.receive.data.eager_long.sender_rank = sender_rank;
    handle->data.receive.data.eager_long.maxsize = maxsize;
    handle->data.receive.data.eager_long.cacheentry = cacheentry;

    index = process->eager_long_receive_size;
    process->eager_long_receive_handles[index] = handle;
    process->eager_long_receive_size += 1;

    /* Fill in the publish data which will be sent to the sender. */
    publish_msg.type = CMI_VMI_PUBLISH_TYPE_EAGER_LONG;

    /* Publish the eager buffer to the sender. */
    status = VMI_RDMA_Publish_Buffer (process->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) publish_buffer,
				      (UINT32) maxsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index,
				      (PVOID) &publish_msg, (ULONG) sizeof (CMI_VMI_Publish_Message_T));
    CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
  }
}



/**************************************************************************
** This function is invoked asynchronously to handle an incoming message
** receive on a stream.
**
** This function is on the receive side.
*/
VMI_RECV_STATUS CMI_VMI_Stream_Notification_Handler (PVMI_CONNECT connection, PVMI_STREAM_RECV stream, VMI_STREAM_COMMAND command, PVOID context, PVMI_SLAB slab)
{
  VMI_STATUS status;

  ULONG msgsize;
  char *msg;
  PVMI_SLAB_STATE state;

  CMI_VMI_Process_T *process;


  DEBUG_PRINT ("CMI_VMI_Stream_Notification_Handler() called.\n");

  /* Save the slab state. */
  status = VMI_Slab_Save_State (slab, &state);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Slab_Save_State()");

  msgsize = VMI_SLAB_BYTES_REMAINING (slab);

  msg = CmiAlloc (msgsize);

  /* Copy the message body into the message buffer. */
  status = VMI_Slab_Copy_Bytes (slab, msgsize, msg);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Slab_Copy_Bytes()");

  /* Restore the slab state. */
  status = VMI_Slab_Restore_State (slab, state);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Slab_Restore_State()");

  process = (CMI_VMI_Process_T *) VMI_CONNECT_GET_RECEIVE_CONTEXT (connection);
  process->normal_short_count += 1;
  CMI_VMI_Message_Receive_Count += 1;

  CMI_VMI_Common_Receive (process->rank, msgsize, msg);

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

  void *mem_context;


  DEBUG_PRINT ("CMI_VMI_Stream_Completion_Handler() called.\n");

  handle = (CMI_VMI_Handle_T *) context;

  CMI_VMI_AsyncMsgCount -= 1;
  handle->refcount -= 1;

  if (handle->refcount <= 1) {
    if (CMI_VMI_Eager_Protocol) {
      mem_context = CONTEXTFIELD (handle->msg);
      if (!mem_context) {
	status = VMI_Cache_Deregister (handle->data.send.data.stream.cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
      }
    } else {
      mem_context = NULL;
      status = VMI_Cache_Deregister (handle->data.send.data.stream.cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
    }

    if (handle->data.send.message_disposition == CMI_VMI_MESSAGE_DISPOSITION_FREE) {
      CmiFree (handle->msg);
    }

#if CMK_BROADCAST_SPANNING_TREE
    if (handle->data.send.message_disposition == CMI_VMI_MESSAGE_DISPOSITION_ENQUEUE) {
      CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), handle->msg);
    }
#endif

    CMI_VMI_Handle_Deallocate (handle);
  }
}



/**************************************************************************
** This function is invoked asynchronously to handle an RDMA publish of a
** buffer from a remote process.
**
** This function is on the receive side.
**
**
** REMEMBER: We can cast the remote buffer's local context to a send handle:
**
**   handle = (CMI_VMI_Handle_T *) (VMI_ADDR_CAST) remote_buffer->lctxt;
**
** as long as the publisher (sender) specifies an address IN OUR ADDRESS
** SPACE as the 5th argument to VMI_RDMA_Publish_Buffer().
*/
void CMI_VMI_RDMA_Publish_Handler (PVMI_CONNECT connection, PVMI_REMOTE_BUFFER remote_buffer, PVMI_SLAB publish_data, ULONG publish_data_size)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;
  PVMI_CACHE_ENTRY cacheentry;
  PVMI_RDMA_OP rdmaop;
  ULONG msgsize;
  char *msg;

  PVMI_BUFFER_OP msgop;
  CMI_VMI_Publish_Message_T *publish_msg;

  CMI_VMI_Process_T *process;
  int slot_size;
  int i;
  int offset;
  int index;


  DEBUG_PRINT ("CMI_VMI_RDMA_Publish_Handler() called.\n");

  process = (CMI_VMI_Process_T *) VMI_CONNECT_GET_RECEIVE_CONTEXT (connection);

  msgop = VMI_SLAB_GET_BUFFEROPS (publish_data);
  publish_msg = (CMI_VMI_Publish_Message_T *) VMI_BUFFEROP_GET_ADDRESS (msgop);

  switch (publish_msg->type)
  {
    case CMI_VMI_PUBLISH_TYPE_GET:
      msgsize = remote_buffer->rsz;
      msg = CmiAlloc (msgsize);

      status = VMI_Cache_Register (msg, msgsize, &cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

      status = VMI_RDMA_Alloc_Op (&rdmaop);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

      rdmaop->numBufs = 1;
      rdmaop->buffers[0] = cacheentry->bufferHandle;
      rdmaop->addr[0] = msg;
      rdmaop->sz[0] = msgsize;
      rdmaop->rbuffer = remote_buffer;
      rdmaop->roffset = 0;
      rdmaop->notify = TRUE;

      handle = CMI_VMI_Handle_Allocate ();

      handle->refcount += 1;
      handle->msg = msg;
      handle->msgsize = msgsize;
      handle->handle_type = CMI_VMI_HANDLE_TYPE_RECEIVE;
      handle->data.receive.receive_handle_type = CMI_VMI_RECEIVE_HANDLE_TYPE_RDMAGET;
      handle->data.receive.data.rdmaget.cacheentry = cacheentry;
      handle->data.receive.data.rdmaget.process = (void *) process;

      status = VMI_RDMA_Get (connection, rdmaop, (PVOID) handle, CMI_VMI_RDMA_Get_Completion_Handler);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Get()");

      break;

    case CMI_VMI_PUBLISH_TYPE_EAGER_SHORT:
      slot_size = sizeof (CMI_VMI_Memory_Chunk_T) + CMI_VMI_Medium_Message_Boundary + sizeof (CMI_VMI_Eager_Short_Slot_Footer_T);

      for (i = 0; i < CMI_VMI_Eager_Short_Slots; i++) {
	offset = (i * slot_size) + sizeof (CMI_VMI_Memory_Chunk_T);

	msg = CmiAlloc (CMI_VMI_Medium_Message_Boundary + sizeof (CMI_VMI_Eager_Short_Slot_Footer_T));

	status = VMI_Cache_Register (msg, CMI_VMI_Medium_Message_Boundary + sizeof (CMI_VMI_Eager_Short_Slot_Footer_T), &cacheentry);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

	status = VMI_RDMA_Alloc_Op (&rdmaop);
	CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Alloc_Op()");

	handle = CMI_VMI_Handle_Allocate ();

	handle->refcount += 1;
	handle->msg = msg;
	handle->msgsize = -1;
	handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
	handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_EAGER_SHORT;
	handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
	handle->data.send.data.eager_short.remote_buffer = remote_buffer;
	handle->data.send.data.eager_short.offset = offset;
	handle->data.send.data.eager_short.cacheentry = cacheentry;
	handle->data.send.data.eager_short.rdmaop = rdmaop;

	index = process->eager_short_send_size;
	process->eager_short_send_handles[index] = handle;
	process->eager_short_send_size += 1;
	process->eager_short_send_credits_available += 1;
      }

      break;

    case CMI_VMI_PUBLISH_TYPE_EAGER_LONG:
      handle = CMI_VMI_Handle_Allocate ();

      handle->refcount += 1;
      handle->msg = NULL;
      handle->msgsize = -1;
      handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
      handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_EAGER_LONG;
      handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_NONE;
      handle->data.send.data.eager_long.maxsize = remote_buffer->rsz;
      handle->data.send.data.eager_long.remote_buffer = remote_buffer;
      handle->data.send.data.eager_long.cacheentry = NULL;

      index = process->eager_long_send_size;
      process->eager_long_send_handles[index] = handle;
      process->eager_long_send_size += 1;

      break;
  }
}



/**************************************************************************
** done
*/
void CMI_VMI_RDMA_Put_Notification_Handler (PVMI_CONNECT connection, UINT32 rdma_size, UINT32 context, VMI_STATUS remote_status)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;
  CMI_VMI_Process_T *process;

  PVMI_CACHE_ENTRY cacheentry;
  PVMI_RDMA_OP rdmaop;

  char *publish_buffer;

  CMI_VMI_Publish_Message_T publish_msg;

  int buffer_size;

  char *msg;


  DEBUG_PRINT ("CMI_VMI_RDMA_Put_Notification_Handler() called.\n");

  /* Cast the context into an index into the handle array. */
  handle = &CMI_VMI_Handles[context];

  msg = handle->msg;
  SIZEFIELD (msg) = rdma_size;
  REFFIELD (msg) = 1;
  CONTEXTFIELD (msg) = handle;

  process = (CMI_VMI_Process_T *) VMI_CONNECT_GET_RECEIVE_CONTEXT (connection);
  process->eager_long_count += 1;
  CMI_VMI_Message_Receive_Count += 1;

  //CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), handle->msg);
  CMI_VMI_Common_Receive (process->rank, rdma_size, msg);
}



/**************************************************************************
**
*/
void CMI_VMI_RDMA_Put_Completion_Handler (PVMI_RDMA_OP rdmaop, PVOID context, VMI_STATUS remote_status)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;

  void *mem_context;


  DEBUG_PRINT ("CMI_VMI_RDMA_Put_Completion_Handler() called.\n");

  handle = (CMI_VMI_Handle_T *) context;

  CMI_VMI_AsyncMsgCount -= 1;
  handle->refcount -= 1;

  status = VMI_RDMA_Dealloc_Buffer (rdmaop->rbuffer);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Buffer()");

  status = VMI_RDMA_Dealloc_Op (rdmaop);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Op()");

  if (handle->refcount <= 1) {
    /* Do not need to check for CMI_VMI_Eager_Protocol here because Put is only used for eager. */
    mem_context = CONTEXTFIELD (handle->msg);
    if (!mem_context) {
      status = VMI_Cache_Deregister (handle->data.send.data.eager_long.cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
    }

    if (handle->data.send.message_disposition == CMI_VMI_MESSAGE_DISPOSITION_FREE) {
      CmiFree (handle->msg);
    }

    CMI_VMI_Handle_Deallocate (handle);
  }
}



/**************************************************************************
** This function is invoked asynchronously to handle the completion of an
** RDMA Get from a remote process.
**
** This function is on the send side.
*/
void CMI_VMI_RDMA_Get_Notification_Handler (PVMI_CONNECT connection, UINT32 context, VMI_STATUS remote_status)
{
  VMI_STATUS status;

  CMI_VMI_Handle_T *handle;

  void *mem_context;


  DEBUG_PRINT ("CMI_VMI_RDMA_Get_Notification_Handler() called.\n");

  handle = &(CMI_VMI_Handles[context]);

  if (CMI_VMI_Eager_Protocol) {
    mem_context = CONTEXTFIELD (handle->msg);
    if (!mem_context) {
      status = VMI_Cache_Deregister (handle->data.send.data.rdmaget.cacheentry);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
    }
  } else {
    mem_context = NULL;
    status = VMI_Cache_Deregister (handle->data.send.data.rdmaget.cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");
  }
  
  handle->refcount -= 1;
  CMI_VMI_AsyncMsgCount -= 1;

  if (handle->refcount <= 1) {
    if (handle->data.send.message_disposition == CMI_VMI_MESSAGE_DISPOSITION_FREE) {
      CmiFree (handle->msg);
    }

    CMI_VMI_Handle_Deallocate (handle);
  }
}



/**************************************************************************
** This function is on the receive side.
*/
void CMI_VMI_RDMA_Get_Completion_Handler (PVMI_RDMA_OP rdmaop, PVOID context, VMI_STATUS sstatus)
{
  VMI_STATUS status;

  ULONG msgsize;
  char *msg;

  CMI_VMI_Handle_T *handle;
  CMI_VMI_Process_T *process;

  int credits_temp;


  DEBUG_PRINT ("CMI_VMI_RDMA_Get_Completion_Handler() called.\n");

  handle = (CMI_VMI_Handle_T *) context;

  msg = handle->msg;
  msgsize = handle->msgsize;

  status = VMI_Cache_Deregister (handle->data.receive.data.rdmaget.cacheentry);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Deregister()");

  status = VMI_RDMA_Dealloc_Buffer (rdmaop->rbuffer);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Buffer()");

  status = VMI_RDMA_Dealloc_Op (rdmaop);
  CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Dealloc_Op()");

  process = (CMI_VMI_Process_T *) handle->data.receive.data.rdmaget.process;
  process->normal_long_count += 1;
  CMI_VMI_Message_Receive_Count += 1;

  CMI_VMI_Common_Receive (process->rank, msgsize, msg);

#if 0

  /* Deal with any eager send credits send with the message. */
  credits_temp = CMI_VMI_MESSAGE_CREDITS (msg);
  process->eager_short_send_credits_available += credits_temp;

#if CMK_BROADCAST_SPANNING_TREE
  if (CMI_BROADCAST_ROOT (msg)) {
    /* Message is enqueued into CMI_VMI_RemoteQueue when send to all spanning children finishes. */
    CMI_VMI_Send_Spanning_Children (msgsize, msg);
  } else {
    CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), msg);
  }
#else
  CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), msg);
#endif

#endif

  CMI_VMI_Handle_Deallocate (handle);
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

  CMI_VMI_Publish_Message_T publish_msg;


  DEBUG_PRINT ("CMI_VMI_Send_Spanning_Children() called.\n");

  childcount = CMI_VMI_Spanning_Children_Count (msg);

  if (childcount == 0) {
    CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), msg);
    return;
  }

  if (msgsize < CMI_VMI_Medium_Message_Boundary) {
    handle = CMI_VMI_Handle_Allocate ();

    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    /* Do NOT increment handle->refcount here! */
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type = CMI_VMI_SEND_HANDLE_TYPE_STREAM;
    handle->data.send.message_disposition = CMI_VMI_MESSAGE_DISPOSITION_ENQUEUE;
    handle->data.send.data.stream.cacheentry = cacheentry;

    bufHandles[0] = cacheentry->bufferHandle;
    addrs[0] = (PVOID) msg;
    sz[0] = (ULONG) msgsize;

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

      status = VMI_Stream_Send ((&CMI_VMI_Processes[destrank])->connection, bufHandles, addrs, sz, 1, CMI_VMI_Stream_Completion_Handler, (PVOID) handle, TRUE);
      CMI_VMI_CHECK_SUCCESS (status, "VMI_Stream_Send()");
    }
  } else {
    status = VMI_Cache_Register (msg, msgsize, &cacheentry);
    CMI_VMI_CHECK_SUCCESS (status, "VMI_Cache_Register()");

    handle = CMI_VMI_Handle_Allocate ();

    /* Do NOT increment handle->refcount here! */
    handle->msg = msg;
    handle->msgsize = msgsize;
    handle->handle_type = CMI_VMI_HANDLE_TYPE_SEND;
    handle->data.send.send_handle_type=CMI_VMI_SEND_HANDLE_TYPE_RDMABROADCAST;
    handle->data.send.message_disposition=CMI_VMI_MESSAGE_DISPOSITION_ENQUEUE;
    handle->data.send.data.rdmabroadcast.cacheentry = cacheentry;

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

      publish_msg.type = CMI_VMI_PUBLISH_TYPE_GET;

      status = VMI_RDMA_Publish_Buffer ((&CMI_VMI_Processes[destrank])->connection, cacheentry->bufferHandle, (VMI_virt_addr_t) (VMI_ADDR_CAST) msg,
					(UINT32) msgsize, (VMI_virt_addr_t) (VMI_ADDR_CAST) NULL, (UINT32) handle->index, (PVOID) &publish_msg,
					(ULONG) sizeof (CMI_VMI_Publish_Message_T));
      CMI_VMI_CHECK_SUCCESS (status, "VMI_RDMA_Publish_Buffer()");
    }
  }
}
#endif   /* CMK_BROADCAST_SPANNING_TREE */



/**************************************************************************
**
*/
void CMI_VMI_Common_Receive (int sourcerank, int msgsize, char *msg)
{
  CMI_VMI_Process_T *process;

  int credits_temp;

  CMI_VMI_Persistent_Request_Message_T *persistent_request_msg;
  CMI_VMI_Latency_Vector_Reply_Message_T *latency_vector_reply_msg;
  CMI_VMI_Cluster_Mapping_Message_T *mapping_msg;

  int i;


  process = &CMI_VMI_Processes[sourcerank];

  /* Deal with any eager send credits sent with the message. */
  credits_temp = CMI_VMI_MESSAGE_CREDITS (msg);
  process->eager_short_send_credits_available += credits_temp;

  switch (CMI_VMI_MESSAGE_TYPE (msg))
  {
    case CMI_VMI_MESSAGE_TYPE_STANDARD:
#if CMK_BROADCAST_SPANNING_TREE
      if (CMI_BROADCAST_ROOT (msg)) {
        /* Message is enqueued after send to spanning children completes. */
	CMI_VMI_Send_Spanning_Children (msgsize, msg);
      } else {
	CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), msg);
      }
#else
      CdsFifo_Enqueue (CpvAccess (CMI_VMI_RemoteQueue), msg);
#endif
      break;

    case CMI_VMI_MESSAGE_TYPE_BARRIER:
      CMI_VMI_Barrier_Count++;
      CmiFree (msg);
      break;

    case CMI_VMI_MESSAGE_TYPE_PERSISTENT_REQUEST:
      persistent_request_msg = (CMI_VMI_Persistent_Request_Message_T *) msg;
      CMI_VMI_Eager_Short_Setup (sourcerank);
      if (persistent_request_msg->maxsize > CMI_VMI_Medium_Message_Boundary) {
	if (persistent_request_msg->maxsize < CMI_VMI_Eager_Long_Buffer_Size) {
	  CMI_VMI_Eager_Long_Setup (sourcerank, CMI_VMI_Eager_Long_Buffer_Size);
	} else {
	  CMI_VMI_Eager_Long_Setup (sourcerank, persistent_request_msg->maxsize);
	}
      }
      CmiFree (msg);
      break;

    case CMI_VMI_MESSAGE_TYPE_CREDIT:
      CmiFree (msg);
      break;

    case CMI_VMI_MESSAGE_TYPE_LATENCY_VECTOR_REQUEST:
      CMI_VMI_Reply_Latencies (sourcerank);
      CmiFree (msg);
      break;

    case CMI_VMI_MESSAGE_TYPE_LATENCY_VECTOR_REPLY:
      latency_vector_reply_msg = (CMI_VMI_Latency_Vector_Reply_Message_T *) msg;
      (&CMI_VMI_Processes[sourcerank])->latency_vector = (unsigned long *) malloc (_Cmi_numpes * sizeof (unsigned long));
      for (i = 0; i < _Cmi_numpes; i++) {
	(&CMI_VMI_Processes[sourcerank])->latency_vector[i] = latency_vector_reply_msg->latency[i];
      }
      CMI_VMI_Latency_Vectors_Received += 1;
      CmiFree (msg);
      break;

    case CMI_VMI_MESSAGE_TYPE_CLUSTER_MAPPING:
      mapping_msg = (CMI_VMI_Cluster_Mapping_Message_T *) msg;
      for (i = 0; i < _Cmi_numpes; i++) {
	(&CMI_VMI_Processes[i])->cluster = mapping_msg->cluster[i];
      }
      CMI_VMI_Cluster_Mapping_Received = TRUE;
      CmiFree (msg);
      break;

    case CMI_VMI_MESSAGE_TYPE_UNKNOWN:
      break;

    default:
      break;
  }
}

/*@}*/
