/**
 ** Ammasso implementation of Converse NET version
 ** Contains Ammasso specific code for:
 **   CmiMachineInit()
 **   CmiNotifyIdle()
 **   DeliverViaNetwork()
 **   CommunicationServer()
 **   CmiMachineExit()
 **
 ** Written By:    Isaac Dooley      idooley2@uiuc.edu
 ** 03/12/05       Esteban Pauli     etpauli2@uiuc.edu
 **                Filippo Gioachin  gioachin@uiuc.ed
 **                David Kunzman     kunzman2@uiuc.edu
 **
 ** Change Log:
 **   03/12/05 : DMK : Initial Version
 **
 ** Todo List:
 **
 **/


////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines, Types, etc. ////////////////////////////////////////////////////////////////////////////

#define AMMASSO_PORT        2583

#define AMMASSO_BUFSIZE            (1024 * 16)
#define AMMASSO_NUMMSGBUFS_PER_QP  4
#define AMMASSO_BUF_IN_USE         0
#define AMMASSO_BUF_NOT_IN_USE     1

#define AMMASSO_CTRLTYPE_READY     1
#define AMMASSO_CTRLTYPE_ACK       2

#define AMMASSO_CTRLMSG_LEN        7
#define CtrlHeader_Construct(buf, ctrlType)  { *((int*)buf) = Cmi_charmrun_pid;                               \
                                               *((short*)((char*)buf + sizeof(int))) = contextBlock->myNode;  \
                                               *((char*)buf + sizeof(int) + sizeof(short)) = ctrlType;        \
                                             }

#define CtrlHeader_GetCharmrunPID(buf)   (*((int*)buf))
#define CtrlHeader_GetNode(buf)          (*((short*)((char*)buf + sizeof(int))))
#define CtrlHeader_GetCtrlType(buf)      (*((char*)buf + sizeof(int) + sizeof(short)))


typedef struct __cmi_idle_state {
  char none;
} CmiIdleState;


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

  // Bufer Pool
  char*  bufferPool;
  CmiNodeLock bufferPoolLock;

} mycb_t;

// Global instance of the mycb_t structure to be used throughout this machine layer
mycb_t *contextBlock = NULL;


#define AMMASSO_STATS   1
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
  AMMASSO_STATS_VARS(sendDataOnQP)

  AMMASSO_STATS_VARS(AsynchronousEventHandler)
  AMMASSO_STATS_VARS(CompletionEventHandler)
  AMMASSO_STATS_VARS(ProcessMessage)
  AMMASSO_STATS_VARS(processAmmassoControlMessage)

} AmmassoStats;

AmmassoStats __stats;

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

#elif

#define AMMASSO_STATS_START(event)
#define AMMASSO_STATS_END(event)
#define AMMASSO_STATS_DISPLAY(event)

#endif



////////////////////////////////////////////////////////////////////////////////////////////////////
// Function ProtoTypes /////////////////////////////////////////////////////////////////////////////

void CmiMachineInit(char **argv);

void CmiAmmassoNodeAddressesStoreHandler(int pe, struct sockaddr_in *addr, int port);

void AmmassoDoIdle();
void CmiNotifyIdle(void);
static CmiIdleState* CmiNotifyGetState(void);
static void CmiNotifyBeginIdle(CmiIdleState *s);
static void CmiNotifyStillIdle(CmiIdleState *s);

void sendAck(OtherNode node);
int getQPSendBuffer(OtherNode node, char force);
int sendDataOnQP(char* data, int len, OtherNode node, char force);
void DeliverViaNetwork(OutgoingMsg msg, OtherNode otherNode, int rank, unsigned int broot);
static void CommunicationServer(int withDelayMs, int where);
void CmiMachineExit();

void AsynchronousEventHandler(cc_rnic_handle_t rnic, cc_event_record_t *eventRecord, void *cb);
void CompletionEventHandler(cc_rnic_handle_t rnic, cc_cq_handle_t cq, void *cb);

void CmiAmmassoOpenQueuePairs();

void processAmmassoControlMessage(char* msg, int len, char *needAck);
void ProcessMessage(char* msg, int len, char *needAck);
//int PollForMessage(cc_cq_handle_t cq);

OtherNode getNodeFromQPId(cc_qp_id_t qp_id);
OtherNode getNodeFromQPHandle(cc_qp_handle_t qp);

void establishQPConnection(OtherNode node, int reuseQPFlag);
void reestablishQPConnection(OtherNode node);
void closeQPConnection(OtherNode node, int destroyQPFlag);

// NOTE: I couldn't find functions similar to these in the CCIL API, if found, use the API ones instead.  They
//   are basically just utility functions.
char* cc_status_to_string(cc_status_t errorCode);
char* cc_conn_error_to_string(cc_connect_status_t errorCode);
void displayQueueQuery(cc_qp_handle_t qp, cc_qp_query_attrs_t *attrs);
char* cc_qp_state_to_string(cc_qp_state_t qpState);
char* cc_event_id_to_string(cc_event_id_t id);
char* cc_connect_status_to_string(cc_connect_status_t status);


////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Bodies /////////////////////////////////////////////////////////////////////////////////


/* CmiMachineInit()
 *   This is called as the node is starting up.  It does some initialization of the machine layer.
 */
void CmiMachineInit(char **argv) {

  char buf[128];
  cc_status_t rtn;

  AMMASSO_STATS_START(MachineInit)

  MACHSTATE(3, "CmiMachineInit() - INFO: (***** Ammasso Specific*****) - Called... Initializing RNIC...");
  MACHSTATE1(3, "CmiMachineInit() - INFO: Cmi_charmrun_pid = %d", Cmi_charmrun_pid);

  // Allocate a context block that will be used throughout this machine layer
  if (contextBlock != NULL) {
    MACHSTATE(3, "CmiMachineInit() - ERROR: contextBlock != NULL");
    sprintf(buf, "CmiMachineInit() - ERROR: contextBlock != NULL");
    CmiAbort(buf);
  }
  contextBlock = (mycb_t*)malloc(sizeof(mycb_t));
  if (contextBlock == NULL) {
    MACHSTATE(3, "CmiMachineInit() - ERROR: Unable to malloc memory for contextBlock");
    sprintf(buf, "CmiMachineInit() - ERROR: Unable to malloc memory for contextBlock");
    CmiAbort(buf);
  }

  // Initialize the contextBlock by zero-ing everything out and then setting anything special
  memset(contextBlock, 0, sizeof(mycb_t));
  contextBlock->rnic = -1;

  MACHSTATE(3, "CmiMachineInit() - INFO: (PRE-OPEN_RNIC)");

  // Check to see if in stand-alone mode
  if (Cmi_charmrun_pid != 0) {

    // Try to Open the RNIC
    //   TODO : Look-up the difference between CC_PBL_PAGE_MODE and CC_PBL_BLOCK_MODE
    //   TODO : Would a call to cc_rnic_enum or cc_rnic_query do any good here?
    rtn = cc_rnic_open(0, CC_PBL_PAGE_MODE, contextBlock, &(contextBlock->rnic));
    if (rtn != CC_OK) {
      MACHSTATE2(3, "CmiMachineInit() - ERROR: Unable to open RNIC: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "CmiMachineInit() - ERROR: Unable to open RNIC: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "CmiMachineInit() - INFO: (PRE-SET-ASYNC-HANDLER)");

    // Set the asynchronous event handler function
    rtn = cc_eh_set_async_handler(contextBlock->rnic, AsynchronousEventHandler, contextBlock);
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "CmiMachineInit() - ERROR: Unable to set Asynchronous Handler: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "CmiMachineInit() - ERROR: Unable to set Asynchronous Handler: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "CmiMachineInit() - INFO: (PRE-SET-CE-HANDLER)");

    // Set the Completion Event Handler
    contextBlock->eh_id = 0;
    rtn = cc_eh_set_ce_handler(contextBlock->rnic, CompletionEventHandler, &(contextBlock->eh_id));
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "CmiMachineInit() - ERROR: Unable to set Completion Event Handler: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "CmiMachineInit() - ERROR: Unable to set Completion Event Handler: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "CmiMachineInit() - INFO: (PRE-PD-ALLOC)");

    // Allocate the Protection Domain
    rtn = cc_pd_alloc(contextBlock->rnic, &(contextBlock->pd_id));
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "CmiMachineInit() - ERROR: Unable to allocate Protection Domain: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "CmiMachineInit() - ERROR: Unable to allocate Protection Domain: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "CmiMachineInit() - INFO: RNIC Open For Business!!!");

    // DMK : TODO : Allocate memory for the buffer pool here and register the memory with the RNIC

  } else {  // Otherwise, not in stand-alone mode

    // Flag the rnic variable as invalid
    contextBlock->rnic = -1;
  }

  MACHSTATE(3, "CmiMachineInit() - INFO: Completed Successfully !!!");

  AMMASSO_STATS_END(MachineInit)
}


void CmiAmmassoNodeAddressesStoreHandler(int pe, struct sockaddr_in *addr, int port) {

  // DMK : NOTE : The hope is that this can be used to request the RMDA addresses of the other nodes after the
  //              initial addresses from charmrun are given to the node.  Get the address here, use that to request
  //              the RDMA address, use the RDMA address to create the QP connection (in establishQPConnection(), which
  //              only subtracts one from the address at the moment... the way our cluster is setup).

  MACHSTATE1(3, "CmiNodeAddressesStoreHandler() - INFO: pe = %d", pe);
  MACHSTATE1(3, "                                       addr = { sin_family = %d,", addr->sin_family);
  MACHSTATE1(3, "                                                sin_port = %d,", addr->sin_port);
  MACHSTATE4(3, "                                                sin_addr.s_addr = %d.%d.%d.%d }", (addr->sin_addr.s_addr & 0xFF), ((addr->sin_addr.s_addr >> 8) & 0xFF), ((addr->sin_addr.s_addr >> 16) & 0xFF), ((addr->sin_addr.s_addr >> 24) & 0xFF));
  MACHSTATE1(3, "                                       port = %d", port);
}


void AmmassoDoIdle() {

  int i;

  AMMASSO_STATS_START(AmmassoDoIdle)

  MACHSTATE(3, "AmmassoDoIdle() - INFO: Called...");
  //
  //for (i = 0; i < contextBlock->numNodes; i++) {
  //
  //  // Skip self since don't have a connection to self
  //  if (i == contextBlock->myNode) continue;
  //
  //  MACHSTATE1(3, "AmmassoDoIdle() - INFO: Calling displayQueueQuery() on QP for node %d", i);
  //  displayQueueQuery(nodes[i].qp, &(nodes[i].qp_attrs));
  //}

  for (i = 0; i < contextBlock->numNodes; i++) {
    if (i == contextBlock->myNode) continue;
    CompletionEventHandler(contextBlock->rnic, nodes[i].recv_cq, &(nodes[i]));
    CompletionEventHandler(contextBlock->rnic, nodes[i].send_cq, &(nodes[i]));
  }

  AMMASSO_STATS_END(AmmassoDoIdle)
}

void CmiNotifyIdle(void) {
  AmmassoDoIdle();
}

static CmiIdleState* CmiNotifyGetState(void) {
  return NULL;
}

static void CmiNotifyBeginIdle(CmiIdleState *s) {
  AmmassoDoIdle();
}

static void CmiNotifyStillIdle(CmiIdleState *s) {
  AmmassoDoIdle();
}

void sendAck(OtherNode node) {

  char buf[AMMASSO_CTRLMSG_LEN];

  MACHSTATE1(3, "sendAck() - Ammasso - INFO: Called... sending ACK to node %d", node->myNode);

  // Create and send an ACK message to the specified QP/Connection/Node
  CtrlHeader_Construct(buf, AMMASSO_CTRLTYPE_ACK);
  sendDataOnQP(buf, AMMASSO_CTRLMSG_LEN, node, 1);  // This is an ACK so set the force flag
}

int getQPSendBuffer(OtherNode node, char force) {

  int rtnBufIndex, i;

  AMMASSO_STATS_START(getQPSendBuffer)

  MACHSTATE1(3, "getQPSendBuffer() - Ammasso - Called (send to node %d)...", node->myNode);

  while (1) {

    rtnBufIndex = -1;

    MACHSTATE(3, "getQPSendBuffer() - INFO: Pre-sendBufLock");
    #if CMK_SHARED_VARS_UNAVAILABLE
      while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
    #endif
    CmiLock(node->sendBufLock);
    
    // If force is set, let the message use any of the available send buffers.  Otherwise, there can only
    // be AMMASSO_NUMMSGBUFS_PER_QP message outstanding (haven't gotten an ACK for) so wait for that.
    // VERY IMPORTANT !!! Only use force for sending ACKs !!!!!!!!  If this is done, it is ensured that there
    // will always be at least one buffer available when the force code is executed
    if (node->connectionState == QP_CONN_STATE_CONNECTED) {
      if (force) {
        rtnBufIndex = node->send_UseIndex;
        node->send_UseIndex++;
        if (node->send_UseIndex >= AMMASSO_NUMMSGBUFS_PER_QP * 2)
          node->send_UseIndex = 0;
      } else {
        if (node->send_InUseCounter < AMMASSO_NUMMSGBUFS_PER_QP) {
          rtnBufIndex = node->send_UseIndex;
          node->send_InUseCounter++;
          node->send_UseIndex++;
          if (node->send_UseIndex >= AMMASSO_NUMMSGBUFS_PER_QP * 2)
            node->send_UseIndex = 0;
        }
      }
    }
 
    /*
    // Attempt to grab one of the buffers
    for (int i = 0; i < AMMASSO_NUMMSGBUFS_PER_QP * 2; i++)
      if (node->send_bufFree[i] != AMMASSO_BUF_IN_USE) {
        rtnBufIndex = i;
        node->send_bufFree = AMMASSO_BUF_IN_USE;
        break;
      }
    */
        
    CmiUnlock(node->sendBufLock);
    MACHSTATE3(3, "getQPSendBuffer() - INFO: Post-sendBufLock - rtnBufIndex = %d, node->connectionState = %d, node->send_UseIndex = %d", rtnBufIndex, node->connectionState, node->send_UseIndex);

    if (rtnBufIndex >= 0) {
      break;
    } else {

      //usleep(1);

      CompletionEventHandler(contextBlock->rnic, node->recv_cq, node);
      CompletionEventHandler(contextBlock->rnic, node->send_cq, node);
    }
  }

  //// Increment the send_UseIndex counter so another buffer will be used next time
  //node->send_UseIndex++;
  //if (node->send_UseIndex >= AMMASSO_NUMMSGBUFS_PER_QP * 2)
  //  node->send_UseIndex = 0;

  MACHSTATE1(3, "getQPSendBuffer() - Ammasso - Finished (returning buffer index: %d)", rtnBufIndex);

  AMMASSO_STATS_END(getQPSendBuffer)

  return rtnBufIndex;

}

// NOTE: The force parameter can be thought of as an "is ACK" control message flag (see comments in getQPSendBuffer())
int sendDataOnQP(char* data, int len, OtherNode node, char force) {

  char buf[256];
  int sendBufIndex;
  int toSendLength;
  cc_status_t rtn;
  cc_uint32_t WRsPosted;
  char isFirst = 1;
  char *origMsgStart = data;

  AMMASSO_STATS_START(sendDataOnQP)

  //CompletionEventHandler(contextBlock->rnic, node->recv_cq, node);
  CompletionEventHandler(contextBlock->rnic, node->send_cq, node);

  MACHSTATE2(3, "sendDataOnQP() - Ammasso - INFO: Called (send to node %d, len = %d)...", node->myNode, len);

  // DMK : For each message that is fragmented, attach another DGRAM header to it, (keeping in mind that the control
  //       messages are no where near large enough for this to occur).

  while (len > 0) {

    MACHSTATE(3, "sendDataOnQP() - Ammasso - INFO: Sending Fragment...");

    // Get a free send buffer (NOTE: This call will block until a send buffer is free)
    sendBufIndex = getQPSendBuffer(node, force);

    // Copy the contents (up to AMMASSO_BUFSIZE worth) of data into the send buffer
    if (isFirst) {

      // In this case, the toSendLength includes the DGRAM header that was part of the original message
      toSendLength = ((len > AMMASSO_BUFSIZE) ? (AMMASSO_BUFSIZE) : (len));  // MIN of len and AMMASSO_BUFSIZE
      memcpy(node->send_buf + (sendBufIndex * AMMASSO_BUFSIZE), data, toSendLength);

    } else {

      // In this case, the toSendLength does not include the DGRAM header size since toSendLength represents the amount of
      //   data in the message being sent and this time around the DGRAM header is being constructed
      toSendLength = ((len > AMMASSO_BUFSIZE - DGRAM_HEADER_SIZE) ? (AMMASSO_BUFSIZE - DGRAM_HEADER_SIZE) : (len));  // MIN of len and AMMASSO_BUFSIZE
      memcpy(node->send_buf + (sendBufIndex * AMMASSO_BUFSIZE) + DGRAM_HEADER_SIZE, data, toSendLength);
      
      // This dgram header is the same except for the sequence number, so copy the original and just modify the sequence number
      // NOTE: If the message is large enough that fragmentation needs to happen, the send_next_lock is already owned by the
      //       thread executing this code.
      // TODO: This code is hacked up a bit to get this working in a hurry, come back later and find a more graceful way to handle this and
      //       keep sendDataOnQP() generic (this should be in DeliverViaNetwork()).
      memcpy(node->send_buf + (sendBufIndex * AMMASSO_BUFSIZE), origMsgStart, DGRAM_HEADER_SIZE);
      ((DgramHeader*)(node->send_buf + (sendBufIndex * AMMASSO_BUFSIZE)))->seqno = node->send_next;
      node->send_next = ((node->send_next+1) & DGRAM_SEQNO_MASK);  // Increase the sequence number
    }

    // Set the data length in the SGL (Note: this is safe to reset like this because the buffer is of length
    //   AMMASSO_BUFSIZE and toSendLength has a maximum value of AMMASSO_BUFSIZE).
    (node->send_sgl[sendBufIndex]).length = toSendLength + ((isFirst) ? (0) : (DGRAM_HEADER_SIZE));

    {
      int j;
      MACHSTATE1(3, "sendDataOnQP() - INFO: sendBufIndex = %d", sendBufIndex);
      MACHSTATE1(3, "                       toSendLength = %d", toSendLength);
      MACHSTATE1(3, "                       node->send_sgl = %p", node->send_sgl);
      MACHSTATE1(3, "                       &(node->send_sgl[sendBufIndex]) = %p", &(node->send_sgl[sendBufIndex]));
      MACHSTATE1(3, "                       sizeof(cc_data_addr_t) = %d", sizeof(cc_data_addr_t));
      MACHSTATE1(3, "                       node->send_sgl + (sizeof(cc_data_addr_t) * sendBufIndex) = %p", node->send_sgl + (sizeof(cc_data_addr_t) * sendBufIndex));
      MACHSTATE(3, "                       Raw Data:");
      for (j = 0; j < toSendLength; j++) {
        MACHSTATE2(3, "                          ((char*)((node->send_sgl[sendBufIndex]).to))[%d] = %02x", j, ((char*)((node->send_sgl[sendBufIndex]).to))[j]);
      }
    }

    rtn = cc_qp_post_sq(contextBlock->rnic, node->qp, &(node->sq_wr[sendBufIndex]), 1, &(WRsPosted));
    if (rtn != CC_OK || WRsPosted != 1) {

      MACHSTATE(3, "sendDataOnQP() - INFO: Pre-sendBufLock");
      #if CMK_SHARED_VARS_UNAVAILABLE
        while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
      #endif
      CmiLock(node->sendBufLock);

      //// Free the buffer lock to indicate that the buffer is no longer in use
      //node->send_bufFree = AMMASSO_BUF_NOT_IN_USE;
      
      // TODO : Find a way to recover from this (the counter being used, send_InUseCounter, is not good enough)... for now, abort...

      CmiUnlock(node->sendBufLock);
      MACHSTATE(3, "sendDataOnQP() - INFO: Post-sendBufLock");

      // Let the user know that an error occured
      MACHSTATE3(3, "sendDataOnQP() - Ammasso - ERROR: Unable to send data to node %d: %d, \"%s\"", node->myNode, rtn, cc_status_to_string(rtn));
      sprintf(buf, "sendDataOnQP() - Ammasso - ERROR: Unable to send data to node %d: %d, \"%s\"\n", node->myNode, rtn, cc_status_to_string(rtn));
      //CmiAbort("Unable to send packet - See Machine Layer Debug File for more details.");
      CmiAbort(buf);
    }

    // Update the data and len variables for the next while (if fragmenting is needed
    data += toSendLength;
    len -= toSendLength;
    isFirst = 0;
  }

  AMMASSO_STATS_END(sendDataOnQP)
}


/* DeliverViaNetwork()
 *
 */
void DeliverViaNetwork(OutgoingMsg msg, OtherNode otherNode, int rank, unsigned int broot) {

  cc_status_t rtn;
  cc_stag_index_t stag;
  cc_data_addr_t sgl;
  cc_sq_wr_t wr;
  cc_uint32_t WRsPosted;

  AMMASSO_STATS_START(DeliverViaNetwork)
  AMMASSO_STATS_START(DeliverViaNetwork_pre_lock)

  MACHSTATE(3, "DeliverViaNetwork() - Ammasso - INFO: Called...");

  // Wait for the send_buf to be free
  // DMK : TODO : Make this a real lock
  // DMK : NOTE : IMPORTANT : It is wise to notice that the version of CmiLock() when CMK_SHARED_VARS_UNAVAILABLE is used
  //                          does NOT block.  Instead, it only increments the CmiNodeLock (which is an int).  If we
  //                          move on to CMK_SHARED_VARS_POSIX_THREADS_SMP, CmiNodeLock() will block... but until then
  //                          we have to do the blocking. 
  //while(otherNode->sendBufLock || otherNode->connectionState != QP_CONN_STATE_CONNECTED) {
  //  MACHSTATE1(3, "DeliverViaNetwork() - INFO - Waiting for sendBufLock or connection to re-establish (sendBufLock = %d)...", otherNode->sendBufLock);
  //  usleep(1000); // DMK : TODO : FIXME : For now, just spin loop, sleeping 1ms each loop... change to grab from already allocated or pin the message memory
  //}
  //
  //MACHSTATE1(3, "DeliverViaNetwork() - Ammasso - INFO: setting sendBufLock (%p)...", otherNode);
  //CmiLock(otherNode->sendBufLock);

  //displayQueueQuery(otherNode->qp, &(otherNode->qp_attrs));
  //{
  //  rtn = cc_qp_query(contextBlock->rnic, otherNode->qp, otherNode->send_buf);
  //  if (rtn != CC_OK) {
  //    MACHSTATE2(3, "DeliverViaNetwork() - Ammasso - WARNING: Unable to query QP: %d, \"%s\"", rtn, cc_status_to_string(rtn));
  //  } else {
  //    MACHSTATE2(3, "DeliverViaNetwork() - Ammasso - INFO: QP.qp_state = %d, \"%s\"",
  //                ((cc_qp_query_attrs_t*)(otherNode->send_buf))->qp_state,
  //                cc_qp_state_to_string(((cc_qp_query_attrs_t*)(otherNode->send_buf))->qp_state)
  //              );
  //  }
  //}

  // Increase the reference count on the outgoing message so the memory for the message is not free'd
  // DMK : NOTE : We don't need to do this since the message data is being copied into the send_buf, the OutgoingMsg
  //              can be free'd ASAP
  //msg->refcount++;  // This will be decreased in the completion handler

  // Create the header for the packet
  //DgramHeaderMake(otherNode->send_buf, rank, msg->src, Cmi_charmrun_pid, otherNode->send_next, broot);

  //MACHSTATE(3, "DeliverViaNetwork() - INFO: Pre-send_next_lock");
  //#if CMK_SHARED_VARS_UNAVAILABLE
  //  while (otherNode->send_next_lock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
  //#endif
  //CmiLock(otherNode->send_next_lock);

  AMMASSO_STATS_END(DeliverViaNetwork_pre_lock)
  AMMASSO_STATS_START(DeliverViaNetwork_lock)

  MACHSTATE(3, "DeliverViaNetwork() - INFO: Pre-send_next_lock");
  #if CMK_SHARED_VARS_UNAVAILABLE
    while (otherNode->send_next_lock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
  #endif
  CmiLock(otherNode->send_next_lock);

  AMMASSO_STATS_END(DeliverViaNetwork_lock)
  AMMASSO_STATS_START(DeliverViaNetwork_post_lock)

  DgramHeaderMake(msg->data, rank, msg->src, Cmi_charmrun_pid, otherNode->send_next, broot);  // Set DGram Header Fields In-Place
  otherNode->send_next = ((otherNode->send_next+1) & DGRAM_SEQNO_MASK);  // Increase the sequence number

  //CmiUnlock(otherNode->send_next_lock);
  //MACHSTATE(3, "DeliverViaNetwork() - INFO: Post-send_next_lock");

  //{
  //  int j;
  //  MACHSTATE(3, "DeliverViaNetwork() - INFO: msg->data:");
  //  for (j = 0; j < 24; j++) {
  //    MACHSTATE2(3, "                            msg->data[%3d] = %d", j, msg->data[j]);
  //  }
  //}

  // Copy the data in
  //memcpy(otherNode->send_buf + DGRAM_HEADER_SIZE, msg->data + DGRAM_HEADER_SIZE, msg->size);        // Data

  //{
  //  int j = 0;
  //  MACHSTATE(3, "DeliverViaNetwork() - INFO: Raw Message Data:");
  //  MACHSTATE1(3, "DeliverViaNetwork() - INFO: msg->size = %d", msg->size);
  //  for (j = 0; j < msg->size + DGRAM_HEADER_SIZE + 24; j++) {
  //    MACHSTATE2(3, "                              otherNode->send_buf[%03d] = [%02x]", j, otherNode->send_buf[j]);
  //  }
  //}
  
  MACHSTATE1(3, "DeliverViaNetwork() - INFO: Sending message to  node %d", otherNode->myNode);
  MACHSTATE1(3, "DeliverViaNetwork() - INFO:                     rank %d", rank);
  MACHSTATE1(3, "DeliverViaNetwork() - INFO:                    broot %d", broot);

  // Attempt send the data in the buffer
  // Post the Work Request to the Send Queue
  // DMK : TODO : Update this code to handle CCERR_QP_TOO_MANY_WRS_POSTED errors (pause breifly and retry)
  // DMK : TODO : FIXME : Modify the size of the buffer so only the used portion of the buffer is sent... This will
  //                      be a custom Work Request when the buffer pool is implemented... I.e. - The pool will always know
  //                      what size the memory buffer is and the WR's SGL will contain the lenghth of the buffer (how much
  //                      of the buffer should be passed to the other side of the connection.
  //otherNode->send_sgl.length = msg->size;  // BAD HACK WARNING !!!
  //rtn = cc_qp_post_sq(contextBlock->rnic, otherNode->qp, &(otherNode->sq_wr), 1, &(WRsPosted));
  //if (rtn != CC_OK || WRsPosted != 1) {
  //  // Free the sendBufLock
  //  CmiUnlock(otherNode->sendBufLock);
  //  #if CMK_SHARED_VARS_UNAVAILABLE
  //  // DMK : NOTE : Since the locks aren't really locks... but we have multiple threads (PE + ammasso), we need CmiLock() to
  //  //              actually block and CmiUnlock() to clear the lock.
  //  #endif
  //
  //  // Let the user know what happened
  //  MACHSTATE2(3, "DeliverViaNetwork() - Ammasso - ERROR - Unable post Work Request to Send Queue: %d, \"%s\"", rtn, cc_status_to_string(rtn));
  //  displayQueueQuery(otherNode->qp, &(otherNode->qp_attrs));
  //}

  AMMASSO_STATS_START(DeliverViaNetwork_send)
  sendDataOnQP(msg->data, msg->size, otherNode, 0);  // These are never forced (not control message of type ACKs)
  AMMASSO_STATS_END(DeliverViaNetwork_send)

  CmiUnlock(otherNode->send_next_lock);
  MACHSTATE(3, "DeliverViaNetwork() - INFO: Post-send_next_lock");


  // DMK : NOTE : I left this in as an example of how to retister the memory with the RNIC on the fly.  Since the ccil
  //              library has a bug which causes it not to de-pin memory that we unregister, it will probably be a better
  //              idea to fragment a message that is too large for a single buffer from the buffer pool.
  /***************************************************************
  // DMK : TODO : This is an important optimization area.  This is registering the memory where the outgoing message
  //              is located with the RNIC.  One balancing act that we will need to do is the cost of copying messages
  //              into memory regions already allocated VS. the cost of registering the memory.  The cost of registering
  //              memory might be constant as the memory doesn't have to traversed.  If the cost of doing a memcpy on a
  //              small message, since memcpy traverses the memory range, is less than the cost of registering the
  //              memory with the RNIC, it would be better to copy the message into a pre-registered memory location.
 
  // Start by registering the memory of the outgoing message with the RNIC
  rtn = cc_nsmr_register_virt(contextBlock->rnic,
                              CC_ADDR_TYPE_VA_BASED,
                              msg->data + DGRAM_HEADER_SIZE,
                              msg->size - DGRAM_HEADER_SIZE,
                              contextBlock->pd_id,
                              0, 0,
                              CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE,
                              &stag
                             );
  if (rtn != CC_OK) {
    // Let the user know what happened
    MACHSTATE2(3, "DeliverViaNetwork() - Ammasso - ERROR - Unable to register OutgoingMsg memory with RNIC: %d, \"%s\"", rtn, cc_status_to_string(rtn));
    return;
  }

  // Setup the Scatter/Gather List
  sgl.stag = stag;
  sgl.to = (cc_uint64_t)(unsigned long)(msg->data + DGRAM_HEADER_SIZE);
  sgl.length = msg->size - DGRAM_HEADER_SIZE;

  // Create the Work Request
  wr.wr_id = (cc_uint64_t)(unsigned long)&(wr);
  wr.wr_type = CC_WR_TYPE_SEND;
  wr.wr_u.send.local_sgl.sge_count = 1;
  wr.wr_u.send.local_sgl.sge_list = &sgl;
  wr.signaled = 1;

  // Post the Work Request to the Send Queue
  // DMK : TODO : Update this code to handle CCERR_QP_TOO_MANY_WRS_POSTED errors (pause breifly and retry)
  rtn = cc_qp_post_sq(contextBlock->rnic, otherNode->qp, &(wr), 1, &(WRsPosted));
  if (rtn != CC_OK || WRsPosted != 1) {
    // Let the user know what happened
    MACHSTATE2(3, "DeliverViaNetwork() - Ammasso - ERROR - Unable post Work Request to Send Queue: %d, \"%s\"", rtn, cc_status_to_string(rtn));
    displayQueueQuery(otherNode->qp, &(otherNode->qp_attrs));
  }
  ***************************************************************/

  MACHSTATE(3, "DeliverViaNetwork() - Ammasso - INFO: Completed.");

  AMMASSO_STATS_END(DeliverViaNetwork_post_lock)
  AMMASSO_STATS_END(DeliverViaNetwork)
}



/****************************************************************************
 *                                                                          
 * CheckSocketsReady
 *
 * Checks both sockets to see which are readable and which are writeable.
 * We check all these things at the same time since this can be done for
 * free with ``select.'' The result is stored in global variables, since
 * this is essentially global state information and several routines need it.
 *
 ***************************************************************************/

int CheckSocketsReady(int withDelayMs)
{   
  int nreadable;
  CMK_PIPE_DECL(withDelayMs);

  CmiStdoutAdd(CMK_PIPE_SUB);
  if (Cmi_charmrun_fd!=-1) CMK_PIPE_ADDREAD(Cmi_charmrun_fd);

  nreadable=CMK_PIPE_CALL();
  ctrlskt_ready_read = 0;
  dataskt_ready_read = 0;
  dataskt_ready_write = 0;
  
  if (nreadable == 0) {
    MACHSTATE(2, "} CheckSocketsReady (nothing readable)");
    return nreadable;
  }
  if (nreadable==-1) {
    CMK_PIPE_CHECKERR();
    MACHSTATE(3, "} CheckSocketsReady (INTERRUPTED!)");
    return CheckSocketsReady(0);
  }
  CmiStdoutCheck(CMK_PIPE_SUB);
  if (Cmi_charmrun_fd!=-1) 
          ctrlskt_ready_read = CMK_PIPE_CHECKREAD(Cmi_charmrun_fd);
  MACHSTATE(3, "} CheckSocketsReady");
  return nreadable;
}

/***********************************************************************
 * CommunicationServer()
 * 
 * This function does the scheduling of the tasks related to the
 * message sends and receives. 
 * It first check the charmrun port for message, and poll the gm event
 * for send complete and outcoming messages.
 *
 ***********************************************************************/

// NOTE: Always called from interrupt
static void ServiceCharmrun_nolock() {

  int again = 1;

  MACHSTATE(2, "ServiceCharmrun_nolock begin {");

  while (again) {
    again = 0;

    CheckSocketsReady(0);
    if (ctrlskt_ready_read) { ctrl_getone(); again=1; }
    if (CmiStdoutNeedsService()) { CmiStdoutService(); }
  }

  MACHSTATE(2, "} ServiceCharmrun_nolock end");
}

void processAmmassoControlMessage(char* msg, int len, char *needAck) {

  int nodeIndex, ctrlType, i;
  OtherNode node;

  AMMASSO_STATS_START(processAmmassoControlMessage)

  // Check the message
  if (len != AMMASSO_CTRLMSG_LEN || CtrlHeader_GetCharmrunPID(msg) != Cmi_charmrun_pid) {
    // Let the user know what happened
    MACHSTATE(3, "processAmmassoControlMessage() - Ammasso - ERROR: Received unknown packet... ignoring...");

    MACHSTATE(3, "processAmmassoControlMessage() - Ammasso - Raw Message Data:");
    MACHSTATE1(3, "                                              len = %d", len);
    for (i = 0; i < len; i++) {
      MACHSTATE2(3, "                                              msg[%d] = %02x", i, msg[i]);
    }

    return;
  }

  nodeIndex = CtrlHeader_GetNode(msg);
  ctrlType = CtrlHeader_GetCtrlType(msg);

  // Verify the control parameters are ok
  if (nodeIndex < 0 || nodeIndex >= contextBlock->numNodes || (ctrlType != AMMASSO_CTRLTYPE_READY && ctrlType != AMMASSO_CTRLTYPE_ACK)) {
    // Let the user know what happened
    MACHSTATE(3, "processAmmassoControlMessage() - Ammasso - ERROR: Received unknown packet... ignoring...");

    MACHSTATE(3, "processAmmassoControlMessage() - Ammasso - Raw Message Data:");
    MACHSTATE1(3, "                                              len = %d", len);
    for (i = 0; i < len; i++) {
      MACHSTATE2(3, "                                              msg[%d] = %02x", i, msg[i]);
    }

    return;
  }

  // Get a pointer to the other node structure for the node the control message came from
  node = &(nodes[nodeIndex]);

  // Perform an action based on the control message type
  switch (ctrlType) {

    case AMMASSO_CTRLTYPE_READY:

      // Decrement the node ready count by one
      contextBlock->nodeReadyCount--;
      MACHSTATE1(3, "processAmmassoControlMessage() - Ammasso - INFO: Received READY packet... still waiting for %d more...", contextBlock->nodeReadyCount);
    
      *needAck = 1;

      //#if CMK_SHARED_VARS_UNAVAILABLE
      //  while (node->sendBufLock != 0) { usleep(10000); } // Since CmiLock() is not really a lock, actually wait
      //#endif
      //CmiLock(node->sendBufLock);

      // Mark the send buffer that was used as usable again
      //node->send_bufFree[node->sendAckCounter] = AMMASSO_BUF_NOT_IN_USE;
      //node->send_InUseCounter--;
      //node->send_AckIndex++;
      //if (node->send_AckIndex >= AMMASSO_NUMMSGBUFS_PER_QP)
      //  node->sendAckIndex = 0;

      //CmiUnlock(node->sendBufLock);

      break;

    case AMMASSO_CTRLTYPE_ACK:

      // DMK: Because the send buffers are used in order and the completions for this connection are completed in order (which
      //      means that the acks are sent back in order), a simple counter on this end will do for keeping track of which
      //      send buffer was used for the message this ACK is for.

      MACHSTATE1(3, "processAmmassoControlMessage() - Ammasso - INFO: Received ACK from node %d", nodeIndex);

      // NOTE: Don't send and ACK for an ACK

      MACHSTATE(3, "processAmmassoControlMessage() - INFO: Pre-sendBufLock");
      #if CMK_SHARED_VARS_UNAVAILABLE
        while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
      #endif
      CmiLock(node->sendBufLock);

      // Mark the send buffer that was used as usable again
      //node->send_bufFree[node->sendAckCounter] = AMMASSO_BUF_NOT_IN_USE;
      node->send_InUseCounter--;
      //node->send_AckIndex++;
      //if (node->send_AckIndex >= AMMASSO_NUMMSGBUFS_PER_QP)
      //  node->sendAckIndex = 0;

      CmiUnlock(node->sendBufLock);
      MACHSTATE(3, "processAmmassoControlMessage() - INFO: Post-sendBufLock");

      break;
  }

  AMMASSO_STATS_END(processAmmassoControlMessage)
}

void ProcessMessage(char* msg, int len, char *needAck) {

  int rank, srcPE, seqNum, magicCookie, size, i;
  unsigned int broot;
  unsigned char checksum;
  OtherNode fromNode;
  char *newMsg;

  AMMASSO_STATS_START(ProcessMessage)

  MACHSTATE(3, "ProcessMessage() - INFO: Called...");

  // Begin by indicating that the message does not need an ACK, set needAck if we find out an ACK is needed
  *needAck = 0;

  {
    MACHSTATE1(3, "ProcessMessage() - INFO: msg = %p", msg);
    int j;
    for (j = 0; j < DGRAM_HEADER_SIZE + 24; j++) {
      MACHSTATE2(3, "ProcessMessage() - INFO: msg[%d] = %02x", j, msg[j]);
    }
  }

  // Check to see if the message length is too short
  if (len < DGRAM_HEADER_SIZE) {

    /*
    // Check to see if the packet is the READY packet
    // NOTE: This is done here so the if statement that would normally check for the READY packet will only
    //       be executed if the common-case execution path for handling messages considers the message to be
    //       invalid.  This is done here so the common case is not slowed by the check for the READY packet.
    // Message needs to have length 6 (int + short), the charmrun pid, and a valid node number
    if ((len == 6) && (*((int*)(msg)) == Cmi_charmrun_pid) && (*((short*)(msg + 4)) >= 0 && *((short*)(msg + 4)) < contextBlock->numNodes)) {

      // This is a READY packet so decrease the number of READY packets we are waiting for
      contextBlock->nodeReadyCount--;
      MACHSTATE1(3, "ProcessMessage() - Ammasso - INFO: Received READY packet... still waiting for %d more...", contextBlock->nodeReadyCount);
      return;

    } else {

      MACHSTATE(3, "ProcessMessage() - Ammasso - ERROR: Received a message that was too short... Ignoring...");
      CmiPrintf("[%d] Received a message that was too short (len: %d)... ignoring...\n", CmiMyPe(), len);
      return;
    }
    */
    processAmmassoControlMessage(msg, len, needAck);
    return;
  }

  // Get the header fields of the message
  DgramHeaderBreak(msg, rank, srcPE, magicCookie, seqNum, broot);

  MACHSTATE(3, "ProcessMessage() - INFO: Message Contents:");
  MACHSTATE1(3, "                           rank = %d", rank);
  MACHSTATE1(3, "                           srcPE = %d", srcPE);
  MACHSTATE1(3, "                           magicCookie = %d", magicCookie);
  MACHSTATE1(3, "                           seqNum = %d", seqNum);
  MACHSTATE1(3, "                           broot = %d", broot);

#ifdef CMK_USE_CHECKSUM

  // Check the checksum
  checksum = computeCheckSum(msg, len);
  if (checksum != 0) {
    MACHSTATE1(3, "ProcessMessage() - Ammasso - ERROR: Received message with bad checksum (%d)... ignoring...", checksum);
    CmiPrintf("[%d] ProcessMessage() - Ammasso - ERROR: Received message with bad checksum (%d)... ignoring...\n", CmiMyPe(), checksum);
    return;
  }

#else

  // Check the magic cookie for correctness
  if (magicCookie != (Cmi_charmrun_pid & DGRAM_MAGIC_MASK)) {
    MACHSTATE(3, "ProcessMessage() - Ammasso - ERROR: Received message with a bad magic cookie... ignoring...");
    CmiPrintf("[%d] ProcessMessage() - Ammasso - ERROR: Received message with a bad magic cookie... ignoring...\n", CmiMyPe());
    return;
  }

#endif

  // Get the OtherNode structure for the node this message was sent from
  fromNode = nodes_by_pe[srcPE];

  MACHSTATE1(3, "ProcessMessage() - INFO: Message from node %d...", fromNode->myNode);

  //MACHSTATE(3, "ProcessMessage() - INFO: Pre-recv_expect_lock");
  //#if CMK_SHARED_VARS_UNAVAILABLE
  //  while (fromNode->recv_expect_lock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
  //#endif
  //CmiLock(fromNode->recv_expect_lock);

  // Check the sequence number of the message
  if (seqNum == (fromNode->recv_expect)) {
    // The expected sequence number was received so setup the next one
    fromNode->recv_expect = ((seqNum+1) & DGRAM_SEQNO_MASK);
  } else {
    MACHSTATE(3, "ProcessMessage() - Ammasso - ERROR: Received a message with a bad sequence number... ignoring...");
    CmiPrintf("[%d] ProcessMessage() - Ammasso - ERROR: Received a message witha bad sequence number... ignoring...\n", CmiMyPe());
    return;
  }

  //CmiUnlock(fromNode->recv_expect_lock);
  //MACHSTATE(3, "ProcessMessage() - INFO: Post-recv_expect_lock");

  newMsg = fromNode->asm_msg;

  // Check to see if this is the start of the message (i.e. - if the message was fragmented, if this is the first
  //   packet of the message) or the entire message.  Only the first packet's header information will be copied into
  //   the asm_buf buffer.
  if (newMsg == NULL) {

    // Allocate memory to hold the new message
    size = CmiMsgHeaderGetLength(msg);
    newMsg = (char*)CmiAlloc(size);
    _MEMCHECK(newMsg);
    
    // Verify the message size
    if (len > size) {
      MACHSTATE2(3, "ProcessMessage() - Ammasso - ERROR: Message size mismatch (size: %d != len: %d)", size, len);
      CmiPrintf("[%d] ProcessMessage() - Ammasso - ERROR: Message size mismatch (size: %d != len: %d)\n", CmiMyPe(), size, len);
      CmiAbort("Message Size Mismatch");
    }

    // Copy the message into the memory location and setup the fromNode structure accordingly
    //memcpy(newMsg, msg + DGRAM_HEADER_SIZE, size);
    memcpy(newMsg, msg, len);
    fromNode->asm_rank = rank;
    fromNode->asm_total = size;
    fromNode->asm_fill = len;
    fromNode->asm_msg = newMsg;

  // Otherwise, this packet is a continuation of the overall message so append it to the last packet
  } else {

    size = len - DGRAM_HEADER_SIZE;

    // Make sure there is enough room in the asm_msg buffer (this should always be true because of the alloc in the true
    //   portion of this if statement).
    if (fromNode->asm_fill + size > fromNode->asm_total) {
      MACHSTATE(3, "ProcessMessage() - Ammasso - ERROR: Message size mismatch");
      CmiPrintf("[%d] ProcessMessage() - Ammasso - ERROR: Message size mismatch", CmiMyPe());
      CmiAbort("Message Size Mismatch");
    }

    // Copy the message into the asm_msg buffer
    memcpy(newMsg + fromNode->asm_fill, msg + DGRAM_HEADER_SIZE, size);
    fromNode->asm_fill += size;
  }

  MACHSTATE2(3, "ProcessMessage() - Ammasso - INFO: Message copied into asm_buf (asm_fill = %d, asm_total = %d)...", fromNode->asm_fill, fromNode->asm_total);

  *needAck = 1;

  // Check to see if a full packet has been received
  if (fromNode->asm_fill == fromNode->asm_total) {

    MACHSTATE(3, "ProcessMessage() - Ammasso - INFO: Pushing message...");

    // React to the message based on its rank
    switch (rank) {

      case DGRAM_BROADCAST:

        MACHSTATE1(3, "ProcessMessage() - Ammasso - INFO: Broadcast - _Cmi_mynodesize = %d", _Cmi_mynodesize);

        // Make a copy of the message for all the PEs on this node except for zero, zero gets the original
        for (i = 1; i < _Cmi_mynodesize; i++)
          CmiPushPE(i, CopyMsg(newMsg, fromNode->asm_total));
        CmiPushPE(0, newMsg);
        break;

#if CMK_NODE_QUEUE_AVAILABLE
      case DGRAM_NODEBROADCAST:
      case DGRAM_NODEMESSAGE:
        CmiPushNode(newMsg);
        break;
#endif

      default:
        CmiPushPE(rank, newMsg);
    }

    MACHSTATE(3, "ProcessMessage() - Ammasso - INFO: NULLing asm_msg...");

    // Clear the message buffer
    fromNode->asm_msg = NULL;
  }

  MACHSTATE(3, "ProcessMessage() - Ammasso - INFO: Checking for re-broadcast");

  // If this packet is part of a broadcast, pass it on to the next nodes
  #if CMK_BROADCAST_SPANNING_TREE

      if (rank == DGRAM_BROADCAST
    #if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
    #endif
	 ) SendSpanningChildren(NULL, 0, len, msg, broot, rank);
  
  #elif CMK_BROADCAST_HYPERCUBE

      if (rank == DGRAM_BROADCAST
    #if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
    #endif
	 ) {
             MACHSTATE(3, "ProcessMessage() - INFO: Calling SendHypercube()...");
             SendHypercube(NULL, 0, len, msg, broot, rank);
           }
  #endif

  AMMASSO_STATS_END(ProcessMessage)
}

// DMK : NOTE : I attempted to put this is a single function to be called by many places but it was getting
//              way too messy... I'll leave the code here for now in hopes of being able to do this in the future.
/*************************************************
// NOTE: Return non-zero if a message was received, zero otherwise
int PollForMessage(cc_cq_handle_t cq) {

  cc_status_t rtn;
  cc_wc_t wc;
  int tmp, fromNode;
  cc_qp_query_attrs_t qpAttr;

  // Poll the Completion Queue for Work Completions
  rtn = cc_cq_poll(contextBlock->rnic, cq, &wc);

  // Just return 0 if there are no Work Completions
  if (rtn == CCERR_CQ_EMPTY) return 0;

  // Let the user know if there was an error
  if (rtn != CC_OK) {
    MACHSTATE2(3, "PollForMessage() - Ammasso - ERROR: Unable to poll Completion Queue: %d, \"%s\"", rtn, cc_status_to_string(rtn));
  }

  // Let the user know if there was an error
  if (wc.status != CC_OK) {
    MACHSTATE2(3, "PollForMessage() - Ammasso - ERROR: Work Completion Status: %d, \"%s\"", wc.status, cc_status_to_string(wc.status));
  }

  // Depending on the WC.type, react accordingly
  switch (wc.wr_type) {

    // wc.wr_type == CC_WR_TYPE_RDMA_READ
    case CC_WR_TYPE_RDMA_READ:

      // DMK : TODO : Fill this in

      break;

    // wc.wr_type == CC_WR_TYPE_RDMA_WRITE
    case CC_WR_TYPE_RDMA_WRITE:

      // DMK : TODO : Fill this in

      break;

    // wc.wr_type == CC_WR_TYPE_SEND  (Message was sent)
    case CC_WR_TYPE_SEND:

      // This will be called as a result of a cc_qp_post_sq()'s Send Work Request in DeliverViaNetwork finishing
      // Free the STag that was created
      rtn = cc_stag_dealloc(contextBlock->rnic, wc.stag);

      break;

    // wc.wr_type == CC_WR_TYPE_RECV  (Message was received)
    case CC_WR_TYPE_RECV:

      // Check for CCERR_FLUSHED which means "we're shutting down" according to the example code
      if (wc.status == CCERR_FLUSHED)
        break;

      //// Make sure that node is defined
      //if (node == NULL) {
      //  MACHSTATE(3, "PollForMessage() - WARNING: Received message but node == NULL... ignoring...");
      //  break;
      //}

      // HORRIBLE HACK!!! -- In the OtherNode structure, a myNode was placed directly after the rq_wr structure.  This
      //   was done so this function could tell from which node the received message came.
      fromNode = *((int*)((char*)wc.wr_id + sizeof(cc_rq_wr_t)));
      //fromNode = node->myNode;

      {
        int j;
        MACHSTATE1(3, "PollForMessage() - INFO: fromNode = %d", fromNode);
        MACHSTATE1(3, "PollForMessage() - INFO: wc.status = %d", wc.status);
        MACHSTATE1(3, "PollForMessage() - INFO: wc.bytes_rcvd = %d", wc.bytes_rcvd);
        MACHSTATE1(3, "PollForMessage() - INFO: wc.wr_id = %d", wc.wr_id);
        //MACHSTATE1(3, "PollForMessage() - INFO: &(nodes[0].rq_wr) = %p", &(nodes[0].rq_wr));
        //MACHSTATE1(3, "PollForMessage() - INFO: &(nodes[1].rq_wr) = %p", &(nodes[1].rq_wr));
        //MACHSTATE(3, "PollForMessage() - INFO: Raw Message Data:");
        //for (j = 0; j < wc.bytes_rcvd; j++) {
        //  MACHSTATE1(3, "                         [%02x]", nodes[fromNode].recv_buf[j]);
        //}
      }

      // Process the Message
      ProcessMessage(nodes[fromNode].recv_buf, wc.bytes_rcvd);
 
      // Re-Post the receive buffer
      tmp = 0;
      rtn = cc_qp_post_rq(contextBlock->rnic, nodes[fromNode].qp, &(nodes[fromNode].rq_wr), 1, &tmp);
      if (rtn != CC_OK) {
        // Let the user know what happened
        MACHSTATE2(3, "PollForMessage() - Ammasso - ERROR: Unable to Post Work Request to Queue Pair: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      }

      break;

    // default
    default:
      MACHSTATE1(3, "PollForMessage() - Ammasso - WARNING - Unknown WC.wr_type: %d", wc.wr_type);
      break;

  } // end switch (wc.wr_type)

  return 1;
}
****************************************************/


static void CommunicationServer_nolock(int withDelayMs) {

  MACHSTATE(2, "CommunicationServer_nolock start {");

  // DMK : TODO : In spare time (here), check for messages and/or completions and/or errors
  //while(PollForMessage(contextBlock->send_cq));  // Clear all the completed sends
  //while(PollForMessage(contextBlock->recv_cq));  // Keep looping while there are still messages that have been received

  MACHSTATE(2, "}CommunicationServer_nolock end");
}


/* CommunicationServer()
 * where:
 *   0: from smp thread
 *   1: from interrupt
 *   2: from worker thread
 */
static void CommunicationServer(int withDelayMs, int where) {

  //// Check to see if we are running in standalone mode... if so, do nothing
  //if (Cmi_charmrun_pid == 0)
  //  return;

  MACHSTATE2(2, "CommunicationServer(%d) from %d {",withDelayMs, where);

  // Check to see if this call is from an interrupt
  if (where == 1) {
    
    // Don't service charmrum if converse exits, this fixed a hang bug
    if (!machine_initiated_shutdown)
      ServiceCharmrun_nolock();
    
    // Don't process any further for interrupts
    return;
  }

  CmiCommLock();
  CommunicationServer_nolock(withDelayMs);
  CmiCommUnlock();

#if CMK_IMMEDIATE_MSG
  if (where == 0)
  CmiHandleImmediate();
#endif

  MACHSTATE(2,"} CommunicationServer");
}



/* CmiMachineExit()
 *
 */
void CmiMachineExit(void) {

  char buf[128];
  cc_status_t rtn;
  int i;

  MACHSTATE(3, "CmiMachineExit() - INFO: Called...");

  // DMK - This is a sleep to help keep the output from the stat displays below separated in the program output
  if (contextBlock->myNode)
    sleep(contextBlock->myNode);

  AMMASSO_STATS_DISPLAY(MachineInit)

  AMMASSO_STATS_DISPLAY(DeliverViaNetwork)
  AMMASSO_STATS_DISPLAY(DeliverViaNetwork_pre_lock)
  AMMASSO_STATS_DISPLAY(DeliverViaNetwork_lock)
  AMMASSO_STATS_DISPLAY(DeliverViaNetwork_post_lock)
  AMMASSO_STATS_DISPLAY(DeliverViaNetwork_send)

  AMMASSO_STATS_DISPLAY(getQPSendBuffer)
  AMMASSO_STATS_DISPLAY(sendDataOnQP)

  AMMASSO_STATS_DISPLAY(AsynchronousEventHandler)
  AMMASSO_STATS_DISPLAY(CompletionEventHandler)
  AMMASSO_STATS_DISPLAY(ProcessMessage)
  AMMASSO_STATS_DISPLAY(processAmmassoControlMessage)

  // TODO: It would probably be a good idea to make a "closing connection" control message so all of the nodes
  //       can agree to close the connections in a graceful way when they are all done.  Similar to the READY packet
  //       but for closing so the closing is graceful (and the other node does not try to reconnect).  This barrier
  //       would go here.

  // Check to see if in stand-alone mode
  if (Cmi_charmrun_pid != 0 && contextBlock->rnic != -1) {

    // Do Clean-up for each of the OtherNode structures (the connections and related data)
    for (i = 0; i < contextBlock->numNodes; i++) {

      // Close all the connections and destroy all the QPs (and related data structures)
      closeQPConnection(nodes + i, 1);  // The program is closing so destroy the QPs

      // Destroy the locks
      CmiDestroyLock(nodes[i].sendBufLock);
      CmiDestroyLock(nodes[i].send_next_lock);
      CmiDestroyLock(nodes[i].recv_expect_lock);
    }

    // DMK : TODO : Clean up the buffer pool here (unregister the memory from the RNIC and free it)

    //// Close the RNIC interface
    rtn = cc_rnic_close(contextBlock->rnic);
    if (rtn != CC_OK) {
      MACHSTATE2(3, "CmiMachineExit() - ERROR: Unable to close the RNIC: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "CmiMachineExit() - ERROR: Unable to close the RNIC: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "CmiMachineExit() - INFO: RNIC Closed.");
  }
}



OtherNode getNodeFromQPId(cc_qp_id_t qp_id) {

  int i;

  // DMK : FIXME : This is a horrible linear search through the nodes table to find the node structure that has this
  //               qp_id but currently this is the fastest way that I know how to do this... In the future to speed it up.
  for (i = 0; i < contextBlock->numNodes; i++)
    if (nodes[i].qp_id == qp_id)
      return (nodes + i);

  return NULL;
}

OtherNode getNodeFromQPHandle(cc_qp_handle_t qp) {

  int i;

  // DMK : FIXME : This is a horrible linear search through the nodes table to find the node structure that has this
  //               qp_id but currently this is the fastest way that I know how to do this... In the future to speed it up.
  for (i = 0; i < contextBlock->numNodes; i++)
    if (nodes[i].qp == qp)
      return (nodes + i);

  return NULL;
}


void AsynchronousEventHandler(cc_rnic_handle_t rnic, cc_event_record_t *er, void *cb) {

  int nodeNumber;
  OtherNode node;
  cc_ep_handle_t connReqEP;
  cc_status_t rtn;
  char buf[16];
  cc_qp_modify_attrs_t modAttrs;

  AMMASSO_STATS_START(AsynchronousEventHandler)

  MACHSTATE2(3, "AsynchronousEventHandler() - INFO: Called... event_id = %d, \"%s\"", er->event_id, cc_event_id_to_string(er->event_id));

  // Do a couple of checks... the reasons for these stem from some example code
  if (er->rnic_handle != contextBlock->rnic) {
    MACHSTATE(3, "AsynchronousEventHandler() - WARNING: er->rnic_handle != contextBlock->rnic");
  }
  if (er->rnic_user_context != contextBlock) {
    MACHSTATE(3, "AsynchronousEventHandler() - WARNING: er->rnic_user_context != contextBlock");
  }

  // Based on the er->event_id, do something about it
  switch (er->event_id) {

    // er->event_id == CCAE_LLP_CLOSE_COMPLETE
    case CCAE_LLP_CLOSE_COMPLETE:
      MACHSTATE(3, "AsynchronousEventHandler() - INFO: Connection Closed.");

      // Get the OtherNode structure for the other node
      MACHSTATE2(3, "AsynchronousEventHandler() - INFO: er->resource_indicator = %d (CC_RES_IND_QP: %d)", er->resource_indicator, CC_RES_IND_QP);
      node = getNodeFromQPId(er->resource_id.qp_id);
      if (node == NULL) {
        MACHSTATE(3, "AsynchronousEventHandler() - ERROR: Unable to find QP from QP ID... Unable to create/recover connection");
        break;
      }

      MACHSTATE(3, "AsynchronousEventHandler() - INFO: Pre-sendBufLock");
      #if CMK_SHARED_VARS_UNAVAILABLE
        while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
      #endif
      CmiLock(node->sendBufLock);

      // Set the state of the connection to closed
      node->connectionState = QP_CONN_STATE_CONNECTION_CLOSED;
  
      CmiUnlock(node->sendBufLock);
      MACHSTATE(3, "AsynchronousEventHandler() - INFO: Post-sendBufLock");
    
      break;

    // er->event_id == CCAE_CONNECTION_REQUEST
    case CCAE_CONNECTION_REQUEST:
      MACHSTATE3(3, "AsynchronousEventHandler() - INFO: Incomming Connection Request -> %s:%d, \"%s\"",
                    inet_ntoa(*(struct in_addr*) &(er->event_data.connection_request.laddr)),
                    ntohs(er->event_data.connection_request.lport),
                    er->event_data.connection_request.private_data
                );

      connReqEP = er->event_data.connection_request.cr_handle;

      // NOTE : The "+ 5" comes from the private data string starting with "node "
      // Get the index of the node that is making the request
      nodeNumber = atoi(er->event_data.connection_request.private_data + 5);
      if (nodeNumber < 0 || nodeNumber >= contextBlock->numNodes) {

        // Refuse the connection and log the rejection
        MACHSTATE1(3, "AsynchronousEventHandler() - WARNING: Unknown entity attempting to connect (node %d)... rejecting connection.", nodeNumber);
        cc_cr_reject(contextBlock->rnic, connReqEP);

      } else {

        // Keep a copy of the end point handle
        nodes[nodeNumber].cr = connReqEP;

        // Accept the connection
        sprintf(buf, "node %d", contextBlock->myNode);
        rtn = cc_cr_accept(contextBlock->rnic, connReqEP, nodes[nodeNumber].qp, strlen(buf) + 1, buf);
        if (rtn != CC_OK) {

          // Let the user know what happened
          MACHSTATE1(3, "AsynchronousEventHandler() - Ammasso - WARNING: Unable to accept connection from node %d", nodeNumber);
	} else {
  
          MACHSTATE1(3, "AsynchronousEventHandler() - Ammasso - INFO: Accepted Connection from node %d", nodeNumber);

          MACHSTATE(3, "AsynchronousEventHandler() - INFO: Pre-sendBufLock");
          #if CMK_SHARED_VARS_UNAVAILABLE
            while (nodes[nodeNumber].sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
          #endif
          CmiLock(nodes[nodeNumber].sendBufLock);

          // Indicate that this connection has been made only if is it the first time the connection was made (don't count re-connects)
          if (nodes[nodeNumber].connectionState == QP_CONN_STATE_PRE_CONNECT)
            (contextBlock->outstandingConnectionCount)--;
          nodes[nodeNumber].connectionState = QP_CONN_STATE_CONNECTED;

          CmiUnlock(nodes[nodeNumber].sendBufLock);
          MACHSTATE(3, "AsynchronousEventHandler() - INFO: Post-sendBufLock");

          MACHSTATE1(3, "AsynchronousEventHandler() - Connected to node %d", nodes[nodeNumber].myNode);
	}
      }

      break;

    // er->event_id == CCAE_ACTIVE_CONNECT_RESULTS
    case CCAE_ACTIVE_CONNECT_RESULTS:
      MACHSTATE(3, "AsynchronousEventHandler() - INFO: Connection Results");

      // Get the OtherNode structure for the other node
      MACHSTATE2(3, "AsynchronousEventHandler() - INFO: er->resource_indicator = %d (CC_RES_IND_QP: %d)", er->resource_indicator, CC_RES_IND_QP);
      node = getNodeFromQPId(er->resource_id.qp_id);
      if (node == NULL) {
        MACHSTATE(3, "AsynchronousEventHandler() - ERROR: Unable to find QP from QP ID... Unable to create/recover connection");
        break;
      }

      // Check to see if the connection was established or not
      if (er->event_data.active_connect_results.status != CC_CONN_STATUS_SUCCESS) {

        MACHSTATE(3, "                                     Connection Failed.");
        MACHSTATE1(3, "                                      - status: \"%s\"", cc_connect_status_to_string(er->event_data.active_connect_results.status));
        MACHSTATE1(3, "                                      - private_data_length = %d", er->event_data.active_connect_results.private_data_length);
        displayQueueQuery(node->qp, &(node->qp_attrs));

        // Attempt to reconnect (try again... don't give up... you can do it!)
        reestablishQPConnection(node);

      } else { // Connection was a success

        MACHSTATE(3, "                                     Connection Success...");
        MACHSTATE2(3, "                                     l -> %s:%d", inet_ntoa(*(struct in_addr*) &(er->event_data.active_connect_results.laddr)), ntohs(er->event_data.active_connect_results.lport));
        MACHSTATE2(3, "                                     r -> %s:%d", inet_ntoa(*(struct in_addr*) &(er->event_data.active_connect_results.raddr)), ntohs(er->event_data.active_connect_results.rport));
        MACHSTATE1(3, "                                     private_data -> \"%s\"", er->event_data.active_connect_results.private_data);

        MACHSTATE(3, "AsynchronousEventHandler() - INFO: Pre-sendBufLock");
        #if CMK_SHARED_VARS_UNAVAILABLE
          while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
        #endif
        CmiLock(node->sendBufLock);

        // Indicate that this connection has been made only if it is the first time the connection was made (don't count re-connects)
        if (node->connectionState == QP_CONN_STATE_PRE_CONNECT)
          (contextBlock->outstandingConnectionCount)--;
        node->connectionState = QP_CONN_STATE_CONNECTED;

        CmiUnlock(node->sendBufLock);
        MACHSTATE(3, "AsynchronousEventHandler() - INFO: Post-sendBufLock");

        MACHSTATE1(3, "AsynchronousEventHandler() - Connected to node %d", node->myNode);
      }

      break;

    // er->event_id is set to a value that indicates the QP will transition into the TERMINATE state
      // Remotely Detected Errors
    case CCAE_TERMINATE_MESSAGE_RECEIVED:
      // LLP Errors
    case CCAE_LLP_SEGMENT_SIZE_INVALID:
    case CCAE_LLP_INVALID_CRC:
    case CCAE_LLP_BAD_FPDU:
      // Remote Operation Errors
    case CCAE_INVALID_DDP_VERSION:
    case CCAE_INVALID_RDMA_VERSION:
    case CCAE_UNEXPECTED_OPCODE:
    case CCAE_INVALID_DDP_QUEUE_NUMBER:
    case CCAE_RDMA_READ_NOT_ENABLED:
    case CCAE_NO_L_BIT:
      // Remote Protection Errors (not associated with RQ)
    case CCAE_TAGGED_INVALID_STAG:
    case CCAE_TAGGED_BASE_BOUNDS_VIOLATION:
    case CCAE_TAGGED_ACCESS_RIGHTS_VIOLATION:
    case CCAE_TAGGED_INVALID_PD:
    case CCAE_WRAP_ERROR:
      // Remote Closing Error
    case CCAE_BAD_LLP_CLOSE:
      // Remote Protection Errors (associated with RQ)
    case CCAE_INVALID_MSN_RANGE:
    case CCAE_INVALID_MSN_GAP:
      // IRRQ Proection Errors
    case CCAE_IRRQ_INVALID_STAG:
    case CCAE_IRRQ_BASE_BOUNDS_VIOLATION:
    case CCAE_IRRQ_ACCESS_RIGHTS_VIOLATION:
    case CCAE_IRRQ_INVALID_PD:
    case CCAE_IRRQ_WRAP_ERROR:  // For these, the VERBS
    case CCAE_IRRQ_OVERFLOW:    // spec is 100% clear as to
    case CCAE_IRRQ_MSN_GAP:     // what to do... but I think
    case CCAE_IRRQ_MSN_RANGE:   // this is correct... DMK
      // Local Errors - ??? Not 100% sure about these (they are written differently in cc_ae.h than in the verbs spec... but I think they go here)
    case CCAE_CQ_SQ_COMPLETION_OVERFLOW:
    case CCAE_CQ_RQ_COMPLETION_ERROR:
    case CCAE_QP_SRQ_WQE_ERROR:
    
    // er->event_id is set to a value that indicates the QP will transition into the ERROR state
      // LLP Errors
    case CCAE_LLP_CONNECTION_LOST:
    case CCAE_LLP_CONNECTION_RESET:
      // Remote Closing Error
    case CCAE_BAD_CLOSE:
      // Local Errors
    case CCAE_QP_LOCAL_CATASTROPHIC_ERROR:
      MACHSTATE3(3, "AsynchronousEventHandler() - WARNING: Connection Error \"%s\" - er->resource_indicator = %d (CC_RES_IND_QP: %d)", cc_event_id_to_string(er->event_id), er->resource_indicator, CC_RES_IND_QP);

      // Figure out which QP went down
      node = getNodeFromQPId(er->resource_id.qp_id);
      if (node == NULL) {
        MACHSTATE(3, "AsynchronousEventHandler() - ERROR: Unable to find QP from QP ID... Unable to recover connection");
        break;
      }

      MACHSTATE(3, "AsynchronousEventHandler() - INFO: Pre-sendBufLock");
      #if CMK_SHARED_VARS_UNAVAILABLE
        while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
      #endif
      CmiLock(node->sendBufLock);

      // Indicate that the connection was lost or will be lost in the very near future (depending on the er->event_id)
      node->connectionState = QP_CONN_STATE_CONNECTION_LOST;

      CmiUnlock(node->sendBufLock);
      MACHSTATE(3, "AsynchronousEventHandler() - INFO: Post-sendBufLock");

      MACHSTATE1(3, "AsynchronousEventHandler() -        Connection ERROR Occured - node %d", node->myNode);
      displayQueueQuery(node->qp, &(node->qp_attrs));

      // Attempt to bring the connection back to life
      reestablishQPConnection(node);

      break;

    // er->event_id == ???
    default:
      MACHSTATE1(3, "AsynchronousEventHandler() - WARNING - Unknown/Unexpected Asynchronous Event: er->event_id = %d", er->event_id);
      break;

  } // end switch (er->event_id)

  AMMASSO_STATS_END(AsynchronousEventHandler)
}


void CompletionEventHandler(cc_rnic_handle_t rnic, cc_cq_handle_t cq, void *cb) {

  OtherNode node = (OtherNode)cb;
  cc_status_t rtn;
  cc_wc_t wc;
  int tmp;
  cc_rq_wr_t *rq_wr;
  char* recvBuf;
  char needAck;

  AMMASSO_STATS_START(CompletionEventHandler)

  //MACHSTATE(3, "CompletionEventHandler() - Called...");

  //// Reset the request notification type
  //rtn = cc_cq_request_notification(contextBlock->rnic, cq, CC_CQ_NOTIFICATION_TYPE_NEXT);
  //if (rtn != CC_OK) {
  //  // Let the user know what happened
  //  MACHSTATE2(3, "CompletionEventHandler() - Ammasso - WARNING - Unable to reset CQ request notification type: %d, \"%s\"", rtn, cc_status_to_string(rtn));
  //}

  // Keep polling the Completion Queue until it is empty of Work Completions
  while (1) {

    //if (PollForMessage(cq) == 0) break;

    // Poll the Completion Queue
    rtn = cc_cq_poll(contextBlock->rnic, cq, &wc);
    //if (rtn == CCERR_CQ_EMPTY) break;
    if (rtn != CC_OK) break;

    // Let the user know if there was an error
    if (wc.status != CC_OK) {
      MACHSTATE2(3, "CompletionEventHandler() - Ammasso - WARNING - WC status not CC_OK: %d, \"%s\"", wc.status, cc_status_to_string(wc.status));

      // DMK : TODO : Add code here that will recover from a Work Request Gone Wrong... i.e. - if a connection is lost, the
      //              WRs that are still pending in the queues will complete in failure with a FLUSHED error status... these
      //              WRs will have to be reposed after the QP is back into the IDLE state in reestablishQPConnection (or later).
    }

    // Depending on the WC.type, react accordingly
    switch (wc.wr_type) {

      // wc.wr_type == CC_WR_TYPE_RDMA_READ
      case CC_WR_TYPE_RDMA_READ:
        // DMK : TODO : Fill this in
        break;

      // wc.wr_type == CC_WR_TYPE_RDMA_WRITE
      case CC_WR_TYPE_RDMA_WRITE:
        // DMK : TODO : Fill this in
        break;

      // wc.wr_type == CC_WR_TYPE_SEND
      case CC_WR_TYPE_SEND:

        // This will be called as a result of a cc_qp_post_sq()'s Send Work Request in DeliverViaNetwork finishing

        // DMK : NOTE : I left this here as an example (if message memory is registered with the RNIC on the fly
        //              in DeliverViaNetwork(), the memory would have to be un-registered/un-pinned here).
        // Free the STag that was created
        //rtn = cc_stag_dealloc(contextBlock->rnic, wc.stag);

        // Unlock the send_buf so it can be used again
        // DMK : TODO : Make this a real lock
        //MACHSTATE1(3, "CompletionEventHandler() - INFO: Send completed... clearing sendBufLock for next send (%p)", node);
        //CmiUnlock(node->sendBufLock);

        MACHSTATE1(3, "CompletionEventHandler() - INFO: Send completed with node %d... still waiting for acknowledge...", node->myNode);

        break;

      // wc.wr_type == CC_WR_TYPE_RECV
      case CC_WR_TYPE_RECV:

        // Check for CCERR_FLUSHED which mean "we're shutting down" according to the example code
        if (wc.status == CCERR_FLUSHED) {
          displayQueueQuery(node->qp, &(node->qp_attrs));
          break;
	}

        // Make sure that node is defined
        if (node == NULL) {
          MACHSTATE(3, "CompletionEventHandler() - WARNING: Received message but node == NULL... ignoring...");
          break;
        }

        displayQueueQuery(node->qp, &(node->qp_attrs));

        {
          int j;
          MACHSTATE1(3, "CompletionEventHandler() - INFO: fromNode = %d", node->myNode);
          MACHSTATE1(3, "CompletionEventHandler() - INFO: wc.status = %d", wc.status);
          MACHSTATE1(3, "CompletionEventHandler() - INFO: wc.bytes_rcvd = %d", wc.bytes_rcvd);
          MACHSTATE1(3, "CompletionEventHandler() - INFO: node = %p", node);
          //MACHSTATE1(3, "CompletionEventHandler() - INFO: &(nodes[0]) = %p", &(nodes[0]));
          //MACHSTATE1(3, "CompletionEventHandler() - INFO: &(nodes[1]) = %p", &(nodes[1]));
          //MACHSTATE1(3, "CompletionEventHandler() - INFO: nodes[0].recv_buf = %p", nodes[0].recv_buf);
          //MACHSTATE1(3, "CompletionEventHandler() - INFO: nodes[1].recv_buf = %p", nodes[1].recv_buf);
          //MACHSTATE(3, "CompletionEventHandler() - INFO: Raw Message Data:");
          //for (j = 0; j < /* wc.bytes_rcvd */ DGRAM_HEADER_SIZE; j++) {
          //  //MACHSTATE1(3, "                         [%02x]", node->recv_buf[j]);
          //  MACHSTATE2(3, "                         [0:%02x, 1:%02x]", nodes[0].recv_buf[j], nodes[1].recv_buf[j]);
	  //}
        }

        // Get the address of the receive buffer used
        rq_wr = (cc_rq_wr_t*)wc.wr_id;
        
        // Process the Message
        ProcessMessage((char*)(rq_wr->local_sgl.sge_list[0].to), wc.bytes_rcvd, &needAck);
        //ProcessMessage(node->recv_buf, wc.bytes_rcvd);

        // Re-Post the receive buffer
        // DMK : NOTE : I'm doing this after processing the message because I'm not 100% sure if the RNIC is "smart" enought
        //              to not reuse the buffer if still in this event handler.
        tmp = 0;
        rtn = cc_qp_post_rq(contextBlock->rnic, node->qp, rq_wr, 1, &tmp);
        if (rtn != CC_OK || tmp != 1) {
          // Let the user know what happened
          MACHSTATE2(3, "CompletionEventHandler() - Ammasso - ERROR: Unable to Post Work Request to Queue Pair: %d, \"%s\"", rtn, cc_status_to_string(rtn));
        }

        // Send and ACK to indicate this node is ready to receive another message
        if (needAck)
          sendAck(node);

        break;

      // default
      default:
	MACHSTATE1(3, "CompletionEventHandler() - Ammasso - WARNING - Unknown WC.wr_type: %d", wc.wr_type);
	break;

    } // end switch (wc.wr_type)
  } // end while (1)

  AMMASSO_STATS_END(CompletionEventHandler)
}


// NOTE: DMK: The code here follows from open_tcp_sockets() in machine-tcp.c.
void CmiAmmassoOpenQueuePairs() {

  char buf[128];
  int i, myNode, numNodes, keepWaiting;
  cc_qp_create_attrs_t qpCreateAttrs;
  cc_status_t rtn;
  cc_inet_addr_t address;
  cc_inet_port_t port;


  MACHSTATE1(3, "CmiAmmassoOpenQueuePairs() - INFO: Called... (Cmi_charmrun_pid = %d)", Cmi_charmrun_pid);

  // Check for stand-alone mode... no connections needed
  if (Cmi_charmrun_pid == 0) return;

  if (nodes == NULL) {
    MACHSTATE(3, "CmiAmmassoOpenQueuePairs() - WARNING: nodes = NULL");
    return;
  }
  MACHSTATE1(3, "CmiAmmassoOpenQueuePairs() - INFO: nodes = %p (remove this line)", nodes);

  // DMK : FIXME : At this point, CmiMyNode() seems to be returning 0 on any node while _Cmi_mynode is
  // !!!!!!!!!!!   returning the correct value.  However, _Cmi_mynode and _Cmi_numnodes may not work with
  // !!!!!!!!!!!   SMP.  Resolve this issue and fix this code.  For now, using _Cmi_mynode and _Cmi_numnodes.
  //myNode = CmiMyNode();
  //numNodes = CmiNumNodes();
  contextBlock->myNode = myNode = _Cmi_mynode;
  contextBlock->numNodes = numNodes = _Cmi_numnodes;
  contextBlock->outstandingConnectionCount = contextBlock->numNodes - 1;  // No connection with self
  contextBlock->nodeReadyCount = contextBlock->numNodes - 1;              // No ready packet from self

  MACHSTATE2(3, "CmiAmmassoOpenQueuePairs() - INFO: myNode = %d, numNodes = %d", myNode, numNodes);

  // Loop through all of the PEs
  //   Begin setting up the Queue Pairs (common code for both "server" and "client")
  //   For nodes with a lower PE, set up a "server" connection (accept)
  //   For nodes with a higher PE, connect as "client" (connect)

  // Loop through all the nodes in the setup
  for (i = 0; i < numNodes; i++) {

    // Setup any members of this nodes OtherNode structure that need setting up
    nodes[i].myNode = i;
    nodes[i].connectionState = QP_CONN_STATE_PRE_CONNECT;
    nodes[i].sendBufLock = CmiCreateLock();
    nodes[i].send_next_lock = CmiCreateLock();
    nodes[i].recv_expect_lock = CmiCreateLock();
    nodes[i].send_UseIndex = 0;
    nodes[i].send_InUseCounter = 0;

    // If you walk around talking to yourself people will look at you all funny-like.  Try not to do that.
    if (i == myNode) continue;

    // Establish the Connection
    establishQPConnection(nodes + i, 0); // Don't reuse the QP (there isn't one yet) 

  } // end for (i < numNodes)


  // Need to block here until all the connections for this node are made
  MACHSTATE(3, "CmiAmmassoOpenQueuePairs() - INFO: Waiting for all connections to be established...");
  while (contextBlock->outstandingConnectionCount > 0) {

    usleep(1000);

    for (i = 0; i < contextBlock->numNodes; i++) {
      if (i == contextBlock->myNode) continue;
      CompletionEventHandler(contextBlock->rnic, nodes[i].recv_cq, &(nodes[i]));
      CompletionEventHandler(contextBlock->rnic, nodes[i].send_cq, &(nodes[i]));
    }
  }
  MACHSTATE(3, "CmiAmmassoOpenQueuePairs() - INFO: All Connections have been established... Continuing");

  // Pause a little so both ends of the connection have time to receive and process the asynchronous events
  usleep(800000); // 800ms

  MACHSTATE(3, "CmiAmmassoOpenQueuePairs() - INFO: Sending ready to all neighboors...");

  // Send all the ready packets
  for (i = 0; i < numNodes; i++) {
    int tmp;
    char buf[24];

    if (i == myNode) continue;  // Skip self

    MACHSTATE1(3, "CmiAmmassoOpenQueuePairs() - INFO: Sending READY to node %d", i);

    // Create the READY control message and send it
    CtrlHeader_Construct(buf, AMMASSO_CTRLTYPE_READY);
    sendDataOnQP(buf, AMMASSO_CTRLMSG_LEN, &(nodes[i]), 0);  // Don't force ready packets

    /*
    // Set the sendBufLock
    CmiLock(nodes[i].sendBufLock);

    // Setup the message
    *(  (int*)(nodes[i].send_buf    )) = Cmi_charmrun_pid;   // Send the charmrun PID
    *((short*)(nodes[i].send_buf + 4)) = myNode;             // Send this node's number

    // Post the send
    nodes[i].send_sgl.length = 6;   // DMK : TODO : FIXME : Change this later when multiple buffers are supported
    rtn = cc_qp_post_sq(contextBlock->rnic, nodes[i].qp, &(nodes[i].sq_wr), 1, &tmp);
    if (rtn != CC_OK || tmp != 1) {
      // Free the sendBufLock
      CmiUnlock(nodes[i].sendBufLock);

      // Let the user know what happened
      MACHSTATE1(3, "CmiAmmassoOpenQueuePairs() - ERROR: Unable to send READY packet to node %d.", i);
    }
    */

    // Note: The code in the completion event handler will unlock sendBufLock again after the packet has actually been sent
  }

  // DMK : NOTE : I don't think this is really needed... (leaving for now just incase I find out it really is)
  //// Wait until all the ready packets have been sent
  //while (keepWaiting) {
  //  int j;
  //
  //  // Assume we won't find a lock that is set
  //  keepWaiting = 0;
  //
  //  // Check all the locks, if one is set, sleep and try again
  //  for (j = 0; i < numNodes; j++)
  //    if (nodes[j].sendBufLock) {
  //      keepWaiting = 1;
  //      usleep(10000); // sleep 10ms
  //      break;
  //    }
  //}

  MACHSTATE(3, "CmiAmmassoOpenQueuePairs() - INFO: All ready packets sent to neighboors...");

  // Need to block here until all of the ready packets have been received
  // NOTE : Because this is a fully connection graph of connections between the nodes, this will block all the nodes
  //        until all the nodes are ready (and all the PEs since there is a node barrier in the run pe function that
  //        all the threads execute... the thread executing this is one of those so it has to reach that node barrier
  //        before any of the other can start doing much of anything).
  MACHSTATE(3, "CmiAmmassoOpenQueuePairs() - INFO: Waiting for all neighboors to be ready...");
  while (contextBlock->nodeReadyCount > 0) {
    usleep(10000);  // Sleep 10ms
    
    for (i = 0; i < contextBlock->numNodes; i++) {
      if (i == contextBlock->myNode) continue;
      CompletionEventHandler(contextBlock->rnic, nodes[i].recv_cq, &(nodes[i]));
      CompletionEventHandler(contextBlock->rnic, nodes[i].send_cq, &(nodes[i]));
    }
  }
  MACHSTATE(3, "CmiAmmassoOpenQueuePairs() - INFO: All neighboors ready...");

  MACHSTATE(3, "CmiAmmassoOpenQueuePairs() - INFO: Finished.");
}



// NOTE: When reestablishing a connection, the QP can be reused so don't recreate a new one (reuseQPFlag = 1).
//       When openning the connection for the first time, there is no QP so create one (reuseQPFlag = 0).
// DMK : TODO : Fix the comment and parameter (I've been playing with what reuseQPFlag actually does and got
//              tired of updating comments)... update the comment when this is finished).
void establishQPConnection(OtherNode node, int reuseQPFlag) {

  char buf[128];
  cc_qp_create_attrs_t qpCreateAttrs;
  cc_status_t rtn;
  int i;
  cc_uint32_t numWRsPosted;

  MACHSTATE1(3, "establishQPConnection() - INFO: Called for node %d...", node->myNode);

  ///// Shared "Client" and "Server" Code /////

    MACHSTATE(3, "establishQPConnection() - INFO: (PRE-RECV-CQ-CREATE)");

    // Create the Completion Queue
    node->recv_cq_depth = AMMASSO_NUMMSGBUFS_PER_QP * 2;
    rtn = cc_cq_create(contextBlock->rnic, &(node->recv_cq_depth), contextBlock->eh_id, node, &(node->recv_cq));
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to create RECV Completion Queue: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to create RECV Completion Queue: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "establishQPConnection() - INFO: (PRE-RECV-CQ-REQUEST-NOTIFICATION)");

    // Setup the Request Notification Type
    //rtn = cc_cq_request_notification(contextBlock->rnic, node->recv_cq, CC_CQ_NOTIFICATION_TYPE_NEXT);
    rtn = cc_cq_request_notification(contextBlock->rnic, node->recv_cq, CC_CQ_NOTIFICATION_TYPE_NEXT);
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to set RECV CQ Notification Type: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to set RECV CQ Notification Type: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "establishQPConnection() - INFO: (PRE-SEND-CQ-CREATE)");

    // Create the Completion Queue
    node->send_cq_depth = AMMASSO_NUMMSGBUFS_PER_QP * 2;
    rtn = cc_cq_create(contextBlock->rnic, &(node->send_cq_depth), contextBlock->eh_id, node, &(node->send_cq));
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to create SEND Completion Queue: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to create SEND Completion Queue: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "establishQPConnection() - INFO: (PRE-SEND-CQ-REQUEST-NOTIFICATION)");

    // Setup the Request Notification Type
    //rtn = cc_cq_request_notification(contextBlock->rnic, node->send_cq, CC_CQ_NOTIFICATION_TYPE_NEXT);
    rtn = cc_cq_request_notification(contextBlock->rnic, node->send_cq, CC_CQ_NOTIFICATION_TYPE_NEXT);
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to set SEND CQ Notification Type: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to set SEND CQ Notification Type: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "establishQPConnection() - INFO: (PRE-QP-CREATE)");

    // Create the Queue Pair
    // Set some initial Create Queue Pair Attributes that will be reused for all Queue Pairs Created
    qpCreateAttrs.sq_cq = node->send_cq;   // Set the Send Queue's Completion Queue
    qpCreateAttrs.rq_cq = node->recv_cq;   // Set the Request Queue's Completion Queue
    qpCreateAttrs.sq_depth = AMMASSO_NUMMSGBUFS_PER_QP * 2;
    qpCreateAttrs.rq_depth = AMMASSO_NUMMSGBUFS_PER_QP * 2;
    qpCreateAttrs.srq = 0;                         //
    qpCreateAttrs.rdma_read_enabled = 1;           //
    qpCreateAttrs.rdma_write_enabled = 1;          //
    qpCreateAttrs.rdma_read_response_enabled = 1;  //
    qpCreateAttrs.mw_bind_enabled = 0;             //
    qpCreateAttrs.zero_stag_enabled = 0;           //
    qpCreateAttrs.send_sgl_depth = 1;              //
    qpCreateAttrs.recv_sgl_depth = 1;              //
    qpCreateAttrs.rdma_write_sgl_depth = 1;        //
    qpCreateAttrs.ord = 1;                         //
    qpCreateAttrs.ird = 1;                         //
    qpCreateAttrs.pdid = contextBlock->pd_id;      // Set the Protection Domain
    qpCreateAttrs.user_context = node;             // Set the User Context Block that will be passed into function calls

    rtn = cc_qp_create(contextBlock->rnic, &qpCreateAttrs, &(node->qp), &(node->qp_id));
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to create Queue Pair: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to create Queue Pair: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    // Since the QP was just created (or re-created), reset the sequence number and any other variables that need reseting
    node->send_InUseCounter = 0;
    node->send_UseIndex = 0;
    node->sendBufLock = 0;
    node->send_next = 0;
    node->send_next_lock = CmiCreateLock();
    node->recv_expect = 0;
    node->recv_expect_lock = CmiCreateLock();

  if (!reuseQPFlag) {

    MACHSTATE(3, "establishQPConnection() - INFO: (PRE-NSMR-REGISTER-VIRT QP-QUERY-ATTRS)");

    // Attempt to register the qp_attrs member of the OtherNode structure with the RNIC so the Queue Pair's state can be queried
    rtn = cc_nsmr_register_virt(contextBlock->rnic,
                                CC_ADDR_TYPE_VA_BASED,
                                (cc_byte_t*)(&(node->qp_attrs)),
                                sizeof(cc_qp_query_attrs_t),
                                contextBlock->pd_id,
                                0, 0,
                                CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE,
                                &(node->qp_attrs_stag_index)
                               );
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to register memory region for Queue Pair Query Attributes: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to register memory region for Queue Pair Query Attributes: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "establishQPConnection() - INFO: (PRE-NSMR-REGISTER-VIRT RECV-BUF)");

    // Attempt to get some memory for the buffers
    node->recv_buf = (char*)CmiAlloc(AMMASSO_BUFSIZE * AMMASSO_NUMMSGBUFS_PER_QP * 2);
    if (node->recv_buf == NULL) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE(3, "establishQPConnection() - ERROR: Unable to allocate memory for RECV buffers");
      sprintf(buf, "establishQPConnection() - ERROR: Unable to allocate memory for RECV buffers");
      CmiAbort(buf);
    }

    // Attempt to register a memory region for receiving (NOTE: memory will be pinned in memory as a result of this call)
    rtn = cc_nsmr_register_virt(contextBlock->rnic,                      // RNIC Handle
                                CC_ADDR_TYPE_VA_BASED,                   // Next parameter is a virtual address
                                node->recv_buf,                          // Virtual address for start of memory region
                                AMMASSO_BUFSIZE * AMMASSO_NUMMSGBUFS_PER_QP * 2, // Size of memory region
                                contextBlock->pd_id,                     // The Protection Domain
                                0,                                       // Flag: "If true, remote access is enabled with this STag"
                                0,                                       // Remote Access Flag
                                CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE,  // Access Control Flags
                                &(node->recv_stag_index)                 // Pointer to STag Index
                               );
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to register memory region for receive buffer: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to register memory region for receive buffer: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }
    
    // Allocate memory for the WRs and SGLs
    node->rq_wr = (cc_rq_wr_t*)CmiAlloc(sizeof(cc_rq_wr_t) * AMMASSO_NUMMSGBUFS_PER_QP * 2);
    if (node->rq_wr == NULL) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE(3, "establishQPConnection() - ERROR: Unable to allocate memory for RECV buffer's WRs");
      sprintf(buf, "establishQPConnection() - ERROR: Unable to allocate memory for RECV buffer's WRs");
      CmiAbort(buf);
    }

    node->recv_sgl = (cc_data_addr_t*)CmiAlloc(sizeof(cc_data_addr_t) * AMMASSO_NUMMSGBUFS_PER_QP * 2);
    if (node->rq_wr == NULL) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE(3, "establishQPConnection() - ERROR: Unable to allocate memory for RECV buffer's SGLs");
      sprintf(buf, "establishQPConnection() - ERROR: Unable to allocate memory for RECV buffer's SGLs");
      CmiAbort(buf);
    }

    for (i = 0; i < AMMASSO_NUMMSGBUFS_PER_QP * 2; i++) {

      // Setup the SGEs for the RECVs
      (node->recv_sgl[i]).length = AMMASSO_BUFSIZE;
      (node->recv_sgl[i]).stag = node->recv_stag_index;
      (node->recv_sgl[i]).to = (cc_uint64_t)(unsigned long)(node->recv_buf + (AMMASSO_BUFSIZE * i));
    
      // Setup single WR to reuse for untagged recvs
      (node->rq_wr[i]).wr_id = (cc_uint64_t)(unsigned long) &(node->rq_wr[i]);
      (node->rq_wr[i]).local_sgl.sge_count = 1;   // TODO : Make this more flexible
      (node->rq_wr[i]).local_sgl.sge_list = &(node->recv_sgl[i]);
    }
    
    MACHSTATE(3, "establishQPConnection() - INFO: (PRE-NSMR-REGISTER-VIRT SEND-BUF)");

    /*
    // Attempt to get memory for the send buffer busy flags
    node->send_bufFree = (char*)CmiAlloc(sizeof(char) * AMMASSO_NUMMSGBUFS_PER_QP);
    if (node->recv_buf == NULL) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE(3, "establishQPConnection() - ERROR: Unable to allocate memory for SEND buffer flags");
      sprintf(buf, "establishQPConnection() - ERROR: Unable to allocate memory for SEND buffer flags");
      CmiAbort(buf);
    }

    // Set the flags so the buffers are initially marked as "not-in-use"
    for (i = 0; i < AMMASSO_NUMMSGBUFS_PER_QP * 2; i++)
      (node->send_bufFree[i]) = 1;
    */

    // Attempt to get some memory for the buffers
    node->send_buf = (char*)CmiAlloc(AMMASSO_BUFSIZE * AMMASSO_NUMMSGBUFS_PER_QP * 2);
    if (node->recv_buf == NULL) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE(3, "establishQPConnection() - ERROR: Unable to allocate memory for SEND buffers");
      sprintf(buf, "establishQPConnection() - ERROR: Unable to allocate memory for SEND buffers");
      CmiAbort(buf);
    }

    // Attempt to register a memory region for sending (NOTE: memory will be pinned in memory as a result of this call)
    rtn = cc_nsmr_register_virt(contextBlock->rnic,
                                CC_ADDR_TYPE_VA_BASED,
                                node->send_buf,
                                AMMASSO_BUFSIZE * AMMASSO_NUMMSGBUFS_PER_QP * 2,
                                contextBlock->pd_id,
                                0,
                                0,
                                CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE,
                                &(node->send_stag_index)
                               );
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to register memory region for send buffer: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to register memory region for send buffer: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    // Allocate memory for the WRs and SGLs
    node->sq_wr = (cc_sq_wr_t*)CmiAlloc(sizeof(cc_sq_wr_t) * AMMASSO_NUMMSGBUFS_PER_QP * 2);
    if (node->sq_wr == NULL) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE(3, "establishQPConnection() - ERROR: Unable to allocate memory for SEND buffer's WRs");
      sprintf(buf, "establishQPConnection() - ERROR: Unable to allocate memory for SEND buffer's WRs");
      CmiAbort(buf);
    }

    node->send_sgl = (cc_data_addr_t*)CmiAlloc(sizeof(cc_data_addr_t) * AMMASSO_NUMMSGBUFS_PER_QP * 2);
    if (node->rq_wr == NULL) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE(3, "establishQPConnection() - ERROR: Unable to allocate memory for SEND buffer's SGLs");
      sprintf(buf, "establishQPConnection() - ERROR: Unable to allocate memory for SEND buffer's SGLs");
      CmiAbort(buf);
    }

    for (i = 0; i < AMMASSO_NUMMSGBUFS_PER_QP * 2; i++) {

      // Setup single SGE for untagged sends
      (node->send_sgl[i]).length = AMMASSO_BUFSIZE;
      (node->send_sgl[i]).stag = node->send_stag_index;
      (node->send_sgl[i]).to = (cc_uint64_t)(unsigned long)(node->send_buf + (AMMASSO_BUFSIZE * i));

      // Setup single WR to reuse for untagged sends
      (node->sq_wr[i]).wr_type = CC_WR_TYPE_SEND;
      (node->sq_wr[i]).wr_id = (cc_uint64_t)(unsigned long)&(node->sq_wr[i]);
      (node->sq_wr[i]).wr_u.send.local_sgl.sge_count = 1;  // TODO : Make this more flexible
      (node->sq_wr[i]).wr_u.send.local_sgl.sge_list = &(node->send_sgl[i]);
      (node->sq_wr[i]).signaled = 1; // ((i % AMMASSO_NUMMSGBUFS_PER_QP == 0) ? (1) : (0));   // Tagged ???
    }

    // DMK : NOTE : This was originally from one of the Ammasso Examples.  We don't need it here but I'm leaving it in as an
    //              example of how to register a memory region for RMDA activity.
    /***************************************************
    MACHSTATE(3, "establishQPConnection() - INFO: (PRE-NSMR-REGISTER-VIRT RDMA-BUF)");

    // Attempt to register a memory region for RMDA (NOTE: memory will be pinned in memory as a result of this call)
    rtn = cc_nsmr_register_virt(contextBlock->rnic,
                                CC_ADDR_TYPE_VA_BASED,
                                node->rdma_buf,
                                sizeof(node->rdma_buf),
                                contextBlock->pd_id,
                                0,
                                0,
                                CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE | CC_ACF_REMOTE_READ | CC_ACF_REMOTE_WRITE,
                                &(node->rdma_stag_index)
                               );
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to register memory region for RDMA buffer: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to register memory region for RDMA buffer: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }
    ****************************************************/

  } // end if (!reuseQPFlag)


  MACHSTATE1(3, "establishQPConnection() - INFO: (PRE-QP-POST-RECV : %d)", AMMASSO_NUMMSGBUFS_PER_QP);

  for (i = 0; i < AMMASSO_NUMMSGBUFS_PER_QP * 2; i++) {
    numWRsPosted = 0;
    rtn = cc_qp_post_rq(contextBlock->rnic, node->qp, &(node->rq_wr[i]), 1, &numWRsPosted);
    if (rtn != CC_OK || numWRsPosted != 1) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to post to RECV RQ: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to post to RECV RQ: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }
  }


  ///// "Server" Specific /////
  if (node->myNode < contextBlock->myNode) {

    int count = 64;
    char value[64];
    int j;

    MACHSTATE(3, "establishQPConnection() - INFO: Starting \"Server\" Code...");

    // Setup the address
    rtn = cc_rnic_getconfig(contextBlock->rnic, CC_GETCONFIG_ADDRS, &count, &value);
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to get local address for to listen: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to get local address for to listen: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    //MACHSTATE1(3, "establishQPConnection() - count = %4d", count);
    //for (j = 0; j < count; j++) {
    //  MACHSTATE2(3, "establishQPConnection() - value[%d] = %4d", j, (int)value[j]);
    //}
      
    // Setup the Address
    // DMK : TODO : FIXME : Fix this code so that it handles host-network/big-little endian ordering
    *(((char*)&(node->address)) + 0) = value[0];
    *(((char*)&(node->address)) + 1) = value[1];
    *(((char*)&(node->address)) + 2) = value[2];
    *(((char*)&(node->address)) + 3) = value[3];

    // Setup the Port
    node->port = htons(AMMASSO_PORT + node->myNode);

    MACHSTATE4(3, "establishQPConnection() - Using Address (Hex) 0x%02X 0x%02X 0x%02X 0x%02X", ((node->address >> 24) & 0xFF), ((node->address >> 16) & 0xFF), ((node->address >> 8) & 0xFF), (node->address & 0xFF));
    MACHSTATE4(3, "                                        (Dec) %4d %4d %4d %4d", ((node->address >> 24) & 0xFF), ((node->address >> 16) & 0xFF), ((node->address >> 8) & 0xFF), (node->address & 0xFF));
    MACHSTATE2(3, "                                   Port (Hex) 0x%02X 0x%02X", ((node->port >> 8) & 0xFF), (node->port & 0xFF));

    // Listen for an incomming connection (NOTE: This call will return immediately; when a connection attempt is
    //   made by a "client", the asynchronous handler will be called.)
    rtn = cc_ep_listen_create(contextBlock->rnic, node->address, &(node->port), 3, contextBlock, &(node->ep));
    if (rtn != CC_OK) {

      // Attempt to close the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE3(3, "establishQPConnection() - ERROR: Unable to listen for incomming connection for node %d: %d, \"%s\"", node->myNode, rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to listen for incomming connection for node %d: %d, \"%s\"", node->myNode, rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(3, "establishQPConnection() - Listening...");
  }


  ///// "Client" Specific /////
  if (node->myNode > contextBlock->myNode) {

    // A one-time sleep that should give the passive side QPs time to post the listens before the active sides start trying to connect
    if (node->myNode == contextBlock->myNode + 1 || reuseQPFlag)  // Only do once if all the connections are being made for the first time, do this for all
      usleep(400000);  // Sleep 400ms                             // connections if reconnecting so the other RNIC has time to setup the listen

    MACHSTATE(3, "establishQPConnection() - INFO: Starting \"Client\" Code...");

    // Setup the Address
    // DMK : TODO : FIXME : Fix this code so that it handles host-network/big-little endian ordering
    *(((char*)&(node->address)) + 0) =  *(((char*)&(node->addr.sin_addr.s_addr)) + 0);
    *(((char*)&(node->address)) + 1) =  *(((char*)&(node->addr.sin_addr.s_addr)) + 1);
    *(((char*)&(node->address)) + 2) =  *(((char*)&(node->addr.sin_addr.s_addr)) + 2);
    *(((char*)&(node->address)) + 3) = (*(((char*)&(node->addr.sin_addr.s_addr)) + 3)) - 1;

    // Setup the Port
    node->port = htons(AMMASSO_PORT + contextBlock->myNode);

    MACHSTATE4(3, "establishQPConnection() - Using Address (Hex) 0x%02X 0x%02X 0x%02X 0x%02X", ((node->address >> 24) & 0xFF), ((node->address >> 16) & 0xFF), ((node->address >> 8) & 0xFF), (node->address & 0xFF));
    MACHSTATE4(3, "                                        (Dec) %4d %4d %4d %4d", ((node->address >> 24) & 0xFF), ((node->address >> 16) & 0xFF), ((node->address >> 8) & 0xFF), (node->address & 0xFF));
    MACHSTATE2(3, "                                   Port (Hex) 0x%02X 0x%02X", ((node->port >> 8) & 0xFF), (node->port & 0xFF));

    // Attempt to make a connection to a "server" (NOTE: This call will return immediately; when the connection
    //   to the "server" is established, the asynchronous handler will be called.)
    sprintf(buf, "node %d", contextBlock->myNode);
    rtn = cc_qp_connect(contextBlock->rnic, node->qp, node->address, node->port, strlen(buf)+1, buf);
    if (rtn != CC_OK) {

      // Attempt to clise the RNIC
      cc_rnic_close(contextBlock->rnic);

      // Let the user know what happened and bail
      MACHSTATE2(3, "establishQPConnection() - ERROR: Unable to request a connection: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "establishQPConnection() - ERROR: Unable to request a connection: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }
  }
}


// NOTE: This will be called in the event of a connection error
void reestablishQPConnection(OtherNode node) {

  cc_status_t rtn;
  char buf[16];
  cc_qp_modify_attrs_t modAttrs;
  cc_wc_t wc;

  MACHSTATE1(3, "reestablishQPConnection() - INFO: For node %d: Clearing Outstanding WRs...", node->myNode);

  // Drain the RECV completion Queue (if a connection is lost, all pending Work Requests are completed with a FLUSHED error)
  while (1) {
    // Pool for a message
    rtn = cc_cq_poll(contextBlock->rnic, node->recv_cq, &wc);
    if (rtn == CCERR_CQ_EMPTY) break;

    // DMK : TODO : FIXME : Something should be done with the WRs that are pulled off so they can be reissued
  }  

  // Drain the SEND completion Queue (if a connection is lost, all pending Work Requests are completed with a FLUSHED error)
  while (1) {
    // Pool for a message
    rtn = cc_cq_poll(contextBlock->rnic, node->send_cq, &wc);
    if (rtn == CCERR_CQ_EMPTY) break;

    // DMK : TODO : FIXME : Something should be done with the WRs that are pulled off so they can be reissued
  }  

  MACHSTATE1(3, "reestablishQPConnection() - INFO: For node %d: Waiting for QP to enter ERROR state...", node->myNode);

  do {

    // Query the QP's state
    rtn = cc_qp_query(contextBlock->rnic, node->qp, &(node->qp_attrs));
    if (rtn != CC_OK) {
      MACHSTATE2(3, "AsynchronousEventHandler() - ERROR: Unable to Query Queue Pair (l): %d, \"%s\"", rtn, cc_status_to_string(rtn));
      break;
    }

    // Check to see if the state is ERROR, if so, break from the loop... otherwise, keep waiting, it will be soon
    if (node->qp_attrs.qp_state == CC_QP_STATE_ERROR)
      break;
    else
      usleep(1000); // 1ms

  } while (1);

  MACHSTATE2(3, "reestablishQPConnection() - INFO: Finished waiting node %d: QP state = \"%s\"...", node->myNode, cc_qp_state_to_string(node->qp_attrs.qp_state));
  MACHSTATE1(3, "reestablishQPConnection() - INFO: Attempting to transition QP into IDLE state for node %d", node->myNode);

  // Transition the Queue Pair from ERROR into IDLE state
  modAttrs.llp_ep = node->ep;
  modAttrs.next_qp_state = CC_QP_STATE_IDLE;
  modAttrs.ord = CC_QP_NO_ATTR_CHANGE;
  modAttrs.ird = CC_QP_NO_ATTR_CHANGE;
  modAttrs.sq_depth = CC_QP_NO_ATTR_CHANGE;
  modAttrs.rq_depth = CC_QP_NO_ATTR_CHANGE;
  modAttrs.stream_message_buffer = NULL;
  modAttrs.stream_message_length = 0;
  rtn = cc_qp_modify(contextBlock->rnic, node->qp, &modAttrs);
  if (rtn != CC_OK) {
    // Let the user know what happened
    MACHSTATE2(3, "reestablishQPConnection() - ERROR: Unable to Modify QP State: %d, \"%s\"", rtn, cc_status_to_string(rtn));
  }

  rtn = cc_qp_query(contextBlock->rnic, node->qp, &(node->qp_attrs));
  if (rtn != CC_OK) {
    MACHSTATE2(3, "reestablishQPConnection() - ERROR: Unable to Query Queue Pair (1): %d, \"%s\"", rtn, cc_status_to_string(rtn));
  }
  MACHSTATE2(3, "reestablishQPConnection() - INFO: Transition results for node %d: QP state = \"%s\"...", node->myNode, cc_qp_state_to_string(node->qp_attrs.qp_state));

  closeQPConnection(node, 0);      // Close the connection but do not destroy the QP
  establishQPConnection(node, 1);  // Reopen the connection and reuse the QP that has already been created
}


// NOTE: When reestablishing a connection, the QP can be reused so don't destroy it and create a new one (destroyQPFlag = 0).
//       When closing the connection because the application is terminating, destroy the QP (destroyQPFlat != 0).
// DMK : TODO : Fix the comment and parameter (I've been playing with what destroyQPFlag actually does and got
//              tired of updating comments)... update the comment when this is finished).
void closeQPConnection(OtherNode node, int destroyQPFlag) {

  MACHSTATE(3, "closeQPConnection() - INFO: Called...");

  /*
  // Close the Completion Queues
  cc_qp_destroy(contextBlock->rnic, node->qp);
  cc_cq_destroy(contextBlock->rnic, node->send_cq);
  cc_cq_destroy(contextBlock->rnic, node->recv_cq);

  if (destroyQPFlag) {
    // De-Register Memory with the RNIC
    cc_stag_dealloc(contextBlock->rnic, node->qp_attrs_stag_index);
    cc_stag_dealloc(contextBlock->rnic, node->recv_stag_index);
    cc_stag_dealloc(contextBlock->rnic, node->send_stag_index);
    //cc_stag_dealloc(contextBlock->rnic, node->rdma_stag_index);

    CmiFree(node->recv_buf);
    CmiFree(node->rq_wr);
    CmiFree(node->recv_sgl);

    CmiFree(node->send_buf);
    CmiFree(node->sq_wr);
    CmiFree(node->send_sgl);
    //CmiFree(node->send_bufFree);
  }
  */
}


char* cc_status_to_string(cc_status_t errorCode) {

  switch (errorCode) {

    case CC_OK: return "OK";
    case CCERR_INSUFFICIENT_RESOURCES:       return "Insufficient Resources";
    case CCERR_INVALID_MODIFIER:             return "Invalid Modifier";
    case CCERR_INVALID_MODE:                 return "Invalid Mode";
    case CCERR_IN_USE:                       return "In Use";
    case CCERR_INVALID_RNIC:                 return "Invalid RNIC";
    case CCERR_INTERRUPTED_OPERATION:        return "Interrupted Operation";
    case CCERR_INVALID_EH:                   return "Invalid EH";
    case CCERR_INVALID_CQ:                   return "Invalid CQ";
    case CCERR_CQ_EMPTY:                     return "CQ Empty";
    case CCERR_NOT_IMPLEMENTED:              return "Not Implemented";
    case CCERR_CQ_DEPTH_TOO_SMALL:           return "CQ Depth Too Small";
    case CCERR_PD_IN_USE:                    return "PD In Use";
    case CCERR_INVALID_PD:                   return "Invalid PD";
    case CCERR_INVALID_SRQ:                  return "Invalid SRQ";
    case CCERR_INVALID_ADDRESS:              return "Invalid Address";
    case CCERR_INVALID_NETMASK:              return "Invalid Netmask";
    case CCERR_INVALID_QP:                   return "Invalid QP";
    case CCERR_INVALID_QP_STATE:             return "Invalid QP State";
    case CCERR_TOO_MANY_WRS_POSTED:          return "Too Many WRs Posted";
    case CCERR_INVALID_WR_TYPE:              return "Invalid WR Type";
    case CCERR_INVALID_SGL_LENGTH:           return "Invalid SGL Length";
    case CCERR_INVALID_SQ_DEPTH:             return "Invalid SQ Depth";
    case CCERR_INVALID_RQ_DEPTH:             return "Invalid RQ Depth";
    case CCERR_INVALID_ORD:                  return "Invalid ORD";
    case CCERR_INVALID_IRD:                  return "Invalid IRD";
    case CCERR_QP_ATTR_CANNOT_CHANGE:        return "QP_ATTR_CANNON_CHANGE";
    case CCERR_INVALID_STAG:                 return "Invalid STag";
    case CCERR_QP_IN_USE:                    return "QP In Use";
    case CCERR_OUTSTANDING_WRS:              return "Outstanding WRs";
    // case CCERR_MR_IN_USE:   NOTE : "CCERR_MR_IN_USE = CCERR_STAG_IN_USE" in "cc_status.h"
    case CCERR_STAG_IN_USE:                  return "STag In Use";
    case CCERR_INVALID_STAG_INDEX:           return "Invalid STag Index";
    case CCERR_INVALID_SGL_FORMAT:           return "Invalid SGL Format";
    case CCERR_ADAPTER_TIMEOUT:              return "Adapter Timeout";
    case CCERR_INVALID_CQ_DEPTH:             return "Invalid CQ Depth";
    case CCERR_INVALID_PRIVATE_DATA_LENGTH:  return "Invalid Private Data Length";
    case CCERR_INVALID_EP:                   return "Invalid EP";
    case CCERR_FLUSHED:                      return "Flushed";
    case CCERR_INVALID_WQE:                  return "Invalid WQE";
    case CCERR_LOCAL_QP_CATASTROPHIC_ERROR:  return "Local QP Catastrophic Error";
    case CCERR_REMOTE_TERMINATION_ERROR:     return "Remote Termination Error";
    case CCERR_BASE_AND_BOUNDS_VIOLATION:    return "Base and Bounds Violation";
    case CCERR_ACCESS_VIOLATION:             return "Access Violation";
    case CCERR_INVALID_PD_ID:                return "Invalid PD ID";
    case CCERR_WRAP_ERROR:                   return "Wrap Error";
    case CCERR_INV_STAG_ACCESS_ERROR:        return "Invalid STag Access Error";
    case CCERR_ZERO_RDMA_READ_RESOURCES:     return "Zero RDMA Read Resources";
    case CCERR_QP_NOT_PRIVILEGED:            return "QP Not Privileged";
    case CCERR_STAG_STATE_NOT_INVALID:       return "STag State Not Invalid";  // ???
    case CCERR_INVALID_PAGE_SIZE:            return "Invalid Page Size";
    case CCERR_INVALID_BUFFER_SIZE:          return "Invalid Buffer Size";
    case CCERR_INVALID_PBE:                  return "Invalid PBE";
    case CCERR_INVALID_FBO:                  return "Invalid FBO";
    case CCERR_INVALID_LENGTH:               return "Invalid Length";
    case CCERR_INVALID_ACCESS_RIGHTS:        return "Invalid Access Rights";
    case CCERR_PBL_TOO_BIG:                  return "PBL Too Big";
    case CCERR_INVALID_VA:                   return "Invalid VA";
    case CCERR_INVALID_REGION:               return "Invalid Region";
    case CCERR_INVALID_WINDOW:               return "Invalid Window";
    case CCERR_TOTAL_LENGTH_TOO_BIG:         return "Total Length Too Big";
    case CCERR_INVALID_QP_ID:                return "Invalid QP ID";
    case CCERR_ADDR_IN_USE:                  return "Address In Use";
    case CCERR_ADDR_NOT_AVAIL:               return "Address Not Available";
    case CCERR_NET_DOWN:                     return "Network Down";
    case CCERR_NET_UNREACHABLE:              return "Network Unreachable";
    case CCERR_CONN_ABORTED:                 return "Connection Aborted";
    case CCERR_CONN_RESET:                   return "Connection Reset";
    case CCERR_NO_BUFS:                      return "No Buffers";
    case CCERR_CONN_TIMEDOUT:                return "Connection Timed-Out";
    case CCERR_CONN_REFUSED:                 return "Connection Refused";
    case CCERR_HOST_UNREACHABLE:             return "Host Unreachable";
    case CCERR_INVALID_SEND_SGL_DEPTH:       return "Invalid Send SGL Depth";
    case CCERR_INVALID_RECV_SGL_DEPTH:       return "Invalid Receive SGL Depth";
    case CCERR_INVALID_RDMA_WRITE_SGL_DEPTH: return "Ivalid RDMA Write SGL Depth";
    case CCERR_INSUFFICIENT_PRIVILEGES:      return "Insufficient Privileges";
    case CCERR_STACK_ERROR:                  return "Stack Error";
    case CCERR_INVALID_VERSION:              return "Invalid Version";
    case CCERR_INVALID_MTU:                  return "Invalid MTU";
    case CCERR_INVALID_IMAGE:                return "Invalid Image";
    case CCERR_PENDING:                      return "(PENDING: Internal to Adapter... Hopefully you aren't reading this...)";   /* not an error; user internally by adapter */
    case CCERR_DEFER:                        return "(DEFER: Internal to Adapter... Hopefully you aren't reading this...)";     /* not an error; used internally by adapter */
    case CCERR_FAILED_WRITE:                 return "Failed Write";
    case CCERR_FAILED_ERASE:                 return "Failed Erase";
    case CCERR_FAILED_VERIFICATION:          return "Failed Verification";
    case CCERR_NOT_FOUND:                    return "Not Found";
    default:                                 return "Unknown Error Code";
  }
}

// NOTE: Letting these string be separate incase different information should be
//   returned that what cc_status_to_string() would return
char* cc_conn_error_to_string(cc_connect_status_t errorCode) {

  switch (errorCode) {
    case CC_CONN_STATUS_SUCCESS:          return "Success";
    case CC_CONN_STATUS_NO_MEM:           return "No Memory";
    case CC_CONN_STATUS_TIMEDOUT:         return "Timed-Out";
    case CC_CONN_STATUS_REFUSED:          return "Refused";
    case CC_CONN_STATUS_NETUNREACH:       return "Network Unreachable";
    case CC_CONN_STATUS_HOSTUNREACH:      return "Host Unreachable";
    case CC_CONN_STATUS_INVALID_RNIC:     return "Invalid RNIC";
    case CC_CONN_STATUS_INVALID_QP:       return "Invalid QP";
    case CC_CONN_STATUS_INVALID_QP_STATE: return "Invalid QP State";
    case CC_CONN_STATUS_REJECTED:         return "Rejected";
    default:                              return (cc_status_to_string((cc_status_t)errorCode));
  }
}

void displayQueueQuery(cc_qp_handle_t qp, cc_qp_query_attrs_t *attrs) {

  //cc_qp_query_attrs_t attr;
  cc_status_t rtn;
  char buf[1024];

  OtherNode node = getNodeFromQPHandle(qp);
  if (node != NULL) {
    MACHSTATE1(3, "displayQueueQuery() - Called for node %d", node->myNode);
  } else {
    MACHSTATE(3, "displayQueueQuery() - Called for unknown node");
  }

  // Query the Queue for its Attributes
  rtn = cc_qp_query(contextBlock->rnic, qp, attrs);
  if (rtn != CC_OK) {
    // Let the user know what happened
    MACHSTATE2(3, "displayQueueQuery() - ERROR: Unable to query queue: %d, \"%s\"", rtn, cc_status_to_string(rtn));
    return;
  }

  // Output the results of the Query
  // DMK : TODO : For now I'm only putting in the ones that I care about... add more later or as needed
  MACHSTATE2(3, "displayQueueQuery() - qp_state = %d, \"%s\"", attrs->qp_state, cc_qp_state_to_string(attrs->qp_state));
  if (attrs->terminate_message_length > 0) {
    memcpy(buf, attrs->terminate_message, attrs->terminate_message_length);
    buf[attrs->terminate_message_length] = '\0';
    MACHSTATE1(3, "displayQueueQuery() - terminate_message = \"%s\"", buf);
  } else {
    MACHSTATE(3, "displayQueueQuery() - terminate_message = NULL");
  }
}

char* cc_qp_state_to_string(cc_qp_state_t qpState) {

  switch (qpState) {
    case CC_QP_STATE_IDLE:       return "IDLE";
    case CC_QP_STATE_CONNECTING: return "CONNECTED";
    case CC_QP_STATE_RTS:        return "RTS";
    case CC_QP_STATE_CLOSING:    return "CLOSING";
    case CC_QP_STATE_TERMINATE:  return "TERMINATE";
    case CC_QP_STATE_ERROR:      return "ERROR";
    default:                     return "unknown";
  }
}

char* cc_event_id_to_string(cc_event_id_t id) {

  switch(id) {
    case CCAE_REMOTE_SHUTDOWN:                 return "Remote Shutdown";
    case CCAE_ACTIVE_CONNECT_RESULTS:          return "Active Connect Results";
    case CCAE_CONNECTION_REQUEST:              return "Connection Request";
    case CCAE_LLP_CLOSE_COMPLETE:              return "LLP Close Complete";
    case CCAE_TERMINATE_MESSAGE_RECEIVED:      return "Terminate Message Received";
    case CCAE_LLP_CONNECTION_RESET:            return "LLP Connection Reset";
    case CCAE_LLP_CONNECTION_LOST:             return "LLP Connection Lost";
    case CCAE_LLP_SEGMENT_SIZE_INVALID:        return "Segment Size Invalid";
    case CCAE_LLP_INVALID_CRC:                 return "LLP Invalid CRC";
    case CCAE_LLP_BAD_FPDU:                    return "LLP Bad FPDU";
    case CCAE_INVALID_DDP_VERSION:             return "Invalid DDP Version";
    case CCAE_INVALID_RDMA_VERSION:            return "Invalid RMDA Version";
    case CCAE_UNEXPECTED_OPCODE:               return "Unexpected Opcode";
    case CCAE_INVALID_DDP_QUEUE_NUMBER:        return "Invalid DDP Queue Number";
    case CCAE_RDMA_READ_NOT_ENABLED:           return "RDMA Read Not Enabled";
    case CCAE_RDMA_WRITE_NOT_ENABLED:          return "RDMA Write Not Enabled";
    case CCAE_RDMA_READ_TOO_SMALL:             return "RDMA Read Too Small";
    case CCAE_NO_L_BIT:                        return "No L Bit";
    case CCAE_TAGGED_INVALID_STAG:             return "Tagged Invalid STag";
    case CCAE_TAGGED_BASE_BOUNDS_VIOLATION:    return "Tagged Base Bounds Violation";
    case CCAE_TAGGED_ACCESS_RIGHTS_VIOLATION:  return "Tagged Access Rights Violation";
    case CCAE_TAGGED_INVALID_PD:               return "Tagged Invalid PD";
    case CCAE_WRAP_ERROR:                      return "Wrap Error";
    case CCAE_BAD_CLOSE:                       return "Bad Close";
    case CCAE_BAD_LLP_CLOSE:                   return "Bad LLP Close";
    case CCAE_INVALID_MSN_RANGE:               return "Invalid MSN Range";
    case CCAE_INVALID_MSN_GAP:                 return "Invalid MSN Gap";
    case CCAE_IRRQ_OVERFLOW:                   return "IRRQ Overflow";
    case CCAE_IRRQ_MSN_GAP:                    return "IRRQ MSG Gap";
    case CCAE_IRRQ_MSN_RANGE:                  return "IRRQ MSN Range";
    case CCAE_IRRQ_INVALID_STAG:               return "IRRQ Invalid STag";
    case CCAE_IRRQ_BASE_BOUNDS_VIOLATION:      return "IRRQ Base Bounds Violation";
    case CCAE_IRRQ_ACCESS_RIGHTS_VIOLATION:    return "IRRQ Access Rights Violation";
    case CCAE_IRRQ_INVALID_PD:                 return "IRRQ Invalid PD";
    case CCAE_IRRQ_WRAP_ERROR:                 return "IRRQ Wrap Error";
    case CCAE_CQ_SQ_COMPLETION_OVERFLOW:       return "CQ SQ Completion Overflow";
    case CCAE_CQ_RQ_COMPLETION_ERROR:          return "CQ RQ Completion Overflow";
    case CCAE_QP_SRQ_WQE_ERROR:                return "QP SRQ WQE Error";
    case CCAE_QP_LOCAL_CATASTROPHIC_ERROR:     return "QP Local Catastrophic Error";
    case CCAE_CQ_OVERFLOW:                     return "CQ Overflow";
    case CCAE_CQ_OPERATION_ERROR:              return "CQ Operation Error";
    case CCAE_SRQ_LIMIT_REACHED:               return "SRQ Limit Reached";
    case CCAE_QP_RQ_LIMIT_REACHED:             return "QP RQ Limit Reached";
    case CCAE_SRQ_CATASTROPHIC_ERROR:          return "SRQ Catastrophic Error";
    case CCAE_RNIC_CATASTROPHIC_ERROR:         return "RNIC Catastrophic Error";
    default:                                   return "Unknown Event ID";
  }
}

char* cc_connect_status_to_string(cc_connect_status_t status) {

  switch (status) {
    case CC_CONN_STATUS_SUCCESS:     return "Success";
    case CC_CONN_STATUS_TIMEDOUT:    return "Timedout";
    case CC_CONN_STATUS_REFUSED:     return "Refused";
    case CC_CONN_STATUS_NETUNREACH:  return "Network Unreachable";
    default:                         return "Unknown";
  }
}
