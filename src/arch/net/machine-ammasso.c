/**
 ** Ammasso implementation of Converse NET version
 ** Contains Ammasso specific
 ** code for:
 ** CmiMachineInit()
 ** CmiCommunicationInit()
 ** CmiNotifyIdle()
 ** DeliverViaNetwork()
 ** CommunicationServer()
 ** CmiMachineExit()
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

#ifndef ALIGN8
#define ALIGN8(x)   (int)((~7)&((x)+7))
#endif

/* DYNAMIC ALLOCATOR: Limit of the allowed pinned memory */
#define MAX_PINNED_MEMORY   100000000
/* DYNAMIC ALLOCATOR END */

#define WASTE_TIME 600
// In order to use CC_POST_CHECK, the last argument to cc_qp_post_sq must be "nWR"
#define CC_POST_CHECK(routine,args,nodeTo) {\
        int retry=10000000; \
        while (ammasso_check_post_err(routine args, #routine, __LINE__, &nWR, nodeTo, retry) == 1) { \
          int i; \
          retry += WASTE_TIME; \
          for (i=0; i<=WASTE_TIME; ++i) { retry --; } \
        } \
      }

#define CC_CHECK(routine,args) \
        ammasso_check_err(routine args, #routine, __LINE__);

static void ammasso_check_err(cc_status_t returnCode,const char *routine,int line) {
  if (returnCode!=CC_OK) {
    char buf[128];
    char *errMsg = cc_status_to_string(returnCode);

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE3(5,"Fatal CC error while executing %s at %s:%d\n", routine, __FILE__, line);
    MACHSTATE2(5,"  Description: %d, %s\n",returnCode,errMsg);
    sprintf(buf,"Fatal CC error while executing %s at %s:%d\n"
	    "  Description: %d, %s\n", routine, __FILE__, line,returnCode,errMsg);
    CmiAbort(buf);
  }
}

// We pass the pointer used in the cc_qp_post_sq call as last parameter, since
// when this function is called, cc_pq_post_sq has already been called
static int ammasso_check_post_err(cc_status_t returnCode,const char *routine,int line, int *nWR, int nodeTo, int retry) {
  if (returnCode == CCERR_TOO_MANY_WRS_POSTED && *nWR != 1 && retry>0) {
    cc_wc_t wc;
    // drain the send completion queue and retry
    while (cc_cq_poll(contextBlock->rnic, nodes[nodeTo].send_cq, &wc) == CC_OK) {
      MACHSTATE1(5, "Error posting send request - INFO: Send completed with node %d... now waiting for acknowledge...", nodeTo);
    }
    MACHSTATE(5, "Error posting send request - Retrying...");
    return 1;
  }

  if (returnCode != CC_OK || *nWR != 1) {
    char buf[128];
    char *errMsg = cc_status_to_string(returnCode);

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE3(5,"Fatal CC error while executing %s at %s:%d\n", routine, __FILE__, line);
    MACHSTATE3(5,"  Description: %d, %s (nWR = %d)\n",returnCode,errMsg,nWR);
    sprintf(buf,"Fatal CC error while executing %s at %s:%d\n"
	    "  Description: %d, %s (nWR = %d)\n", routine, __FILE__, line,returnCode,errMsg,*nWR);
    CmiAbort(buf);
  }
  return 0;
}


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
AmmassoToken *getQPSendToken(OtherNode node);
int sendDataOnQP(char* data, int len, OtherNode node, char flags);
void DeliverViaNetwork(OutgoingMsg msg, OtherNode otherNode, int rank, unsigned int broot, int copy);
static void CommunicationServer(int withDelayMs, int where);
void CmiMachineExit();

void AsynchronousEventHandler(cc_rnic_handle_t rnic, cc_event_record_t *eventRecord, void *cb);
void CheckRecvBufForMessage(OtherNode node);
//void CompletionEventHandler(cc_rnic_handle_t rnic, cc_cq_handle_t cq, void *cb);
//void CompletionEventHandlerWithAckFlag(cc_rnic_handle_t rnic, cc_cq_handle_t cq, void *cb, int breakOnAck);

void CmiAmmassoOpenQueuePairs();

void processAmmassoControlMessage(char* msg, int len, Tailer *tail, OtherNode from);
int ProcessMessage(char* msg, int len, Tailer *tail, OtherNode from);

OtherNode getNodeFromQPId(cc_qp_id_t qp_id);
OtherNode getNodeFromQPHandle(cc_qp_handle_t qp);

void establishQPConnection(OtherNode node, int reuseQPFlag);
void reestablishQPConnection(OtherNode node);
void closeQPConnection(OtherNode node, int destroyQPFlag);

void BufferAlloc(int n);
void TokenAlloc(int n);
void RequestTokens(OtherNode node, int n);
void GrantTokens(OtherNode node, int n);
void RequestReleaseTokens(OtherNode node, int n);
void ReleaseTokens(OtherNode node, int n);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Bodies /////////////////////////////////////////////////////////////////////////////////

/* Callbacks used by the DYNAMIC ALLOCATOR */
void AllocatorCheck () {
  int i, limit;
  char buf[24];
  for (i=0; i<contextBlock->numNodes; ++i) {
    if (i==contextBlock->myNode) continue;
    limit = nodes[i].num_sendTokens - nodes[i].max_used_tokens - 10;
    CmiPrintf("[%d] AllocatorCheck called: node %d, limit %d\n",CmiMyPe(),i,limit);
    if (limit > 0) {
      ReleaseTokens(&nodes[i], limit);
      CmiPrintf("[%d] Releasing %d tokens to %d\n", CmiMyPe(), limit, i);
      nodes[i].max_used_tokens = 0;
    }
  }
}
/* DYNAMIC ALLOCATOR END */

void BufferAlloc(int n) {
  int i;
  char buf[128];
  cc_stag_index_t newStagIndex;
  AmmassoBuffer *newBuffers;

  MACHSTATE1(3, "Allocating %d new Receive Buffers",n);

  // Try to allocate the memory for n receiving buffers
  newBuffers = (AmmassoBuffer*) CmiAlloc(n*sizeof(AmmassoBuffer));

  if (newBuffers == NULL) {

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE(5, "BufferAlloc() - ERROR: Unable to allocate memory for RECV buffers");
    sprintf(buf, "BufferAlloc() - ERROR: Unable to allocate memory for RECV buffers");
    CmiAbort(buf);
  }

  contextBlock->pinnedMemory += n*sizeof(AmmassoBuffer);
  CC_CHECK(cc_nsmr_register_virt,(contextBlock->rnic,
				  CC_ADDR_TYPE_VA_BASED,
				  (cc_byte_t*)newBuffers,
				  n*sizeof(AmmassoBuffer),
				  contextBlock->pd_id,
				  0, 0,
				  CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE | CC_ACF_REMOTE_WRITE,
				  &newStagIndex)
	   );

  for (i=0; i<n; ++i) {
    newBuffers[i].tail.length = 0;
    newBuffers[i].next = &(newBuffers[i+1]);
    newBuffers[i].stag = newStagIndex;
  }
  newBuffers[n-1].next = NULL;
  if (contextBlock->freeRecvBuffers == NULL) {
    contextBlock->freeRecvBuffers = newBuffers;
  } else {
    contextBlock->last_freeRecvBuffers->next = newBuffers;
  }
  contextBlock->last_freeRecvBuffers = &newBuffers[n-1];
  contextBlock->num_freeRecvBuffers += n;
}

void TokenAlloc(int n) {
  int i;
  char buf[128];
  cc_stag_index_t newStagIndex;
  AmmassoToken *sendToken, *tokenScanner;
  cc_data_addr_t *sendSgl;
  AmmassoBuffer *sendBuffer;

  MACHSTATE1(3, "Allocating %d new Tokens",n);

  // Try to allocate the memory for n sending buffers
  sendBuffer = (AmmassoBuffer*) CmiAlloc(n*sizeof(AmmassoBuffer));

  if (sendBuffer == NULL) {

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE(5, "TokenAlloc() - ERROR: Unable to allocate memory for SEND buffers");
    sprintf(buf, "TokenAlloc() - ERROR: Unable to allocate memory for SEND buffers");
    CmiAbort(buf);
  }

  contextBlock->pinnedMemory += n*sizeof(AmmassoBuffer);
  CC_CHECK(cc_nsmr_register_virt,(contextBlock->rnic,
				  CC_ADDR_TYPE_VA_BASED,
				  (cc_byte_t*)sendBuffer,
				  n*sizeof(AmmassoBuffer),
				  contextBlock->pd_id,
				  0, 0,
				  CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE,
				  &newStagIndex)
	   );

  // Allocate the send tokens
  sendToken = (AmmassoToken*) CmiAlloc(n*ALIGN8(sizeof(AmmassoToken)));

  if (sendToken == NULL) {

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE(5, "TokenAlloc() - ERROR: Unable to allocate memory for send TOKEN buffers");
    sprintf(buf, "TokenAlloc() - ERROR: Unable to allocate memory for send TOKEN buffers");
    CmiAbort(buf);
  }

  sendSgl = (cc_data_addr_t*) CmiAlloc(n*ALIGN8(sizeof(cc_data_addr_t)));

  if (sendSgl == NULL) {

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE(5, "TokenAlloc() - ERROR: Unable to allocate memory for send SGL buffers");
    sprintf(buf, "TokenAlloc() - ERROR: Unable to allocate memory for send SGL buffers");
    CmiAbort(buf);
  }

  tokenScanner = sendToken;
  for (i=0; i<n; ++i) {
    sendSgl->stag = newStagIndex;
    sendSgl->length = AMMASSO_BUFSIZE + sizeof(Tailer);
    sendSgl->to = (unsigned long)&(sendBuffer[i]);
    tokenScanner->wr.wr_id = (unsigned long)tokenScanner;
    tokenScanner->wr.wr_type = CC_WR_TYPE_RDMA_WRITE;
    tokenScanner->wr.wr_u.rdma_write.local_sgl.sge_count = 1;
    tokenScanner->wr.wr_u.rdma_write.local_sgl.sge_list = sendSgl;
    tokenScanner->wr.signaled = 1;
    tokenScanner->localBuf = (AmmassoBuffer*)&(sendBuffer[i]);
    LIST_ENQUEUE(contextBlock->,freeTokens,tokenScanner);
    sendSgl = (cc_data_addr_t*)(((char*)sendSgl)+ALIGN8(sizeof(cc_data_addr_t)));
    tokenScanner = (AmmassoToken*)(((char*)tokenScanner)+ALIGN8(sizeof(AmmassoToken)));
  }
}

void RequestTokens(OtherNode node, int n) {
  char buf[24];
  *((int*)buf) = n;
  sendDataOnQP(buf, sizeof(int), node, AMMASSO_MOREBUFFERS);
}

void GrantTokens(OtherNode node, int n) {
  int i;
  char *buf;
  AmmassoBuffer *buffer;
  AmmassoBuffer *prebuffer;
  AmmassoTokenDescription *tokenDesc;
  if (node->pending != NULL) return;
  if (n*sizeof(AmmassoTokenDescription) + sizeof(int) > AMMASSO_BUFSIZE) {
    n = (AMMASSO_BUFSIZE-sizeof(int)) / sizeof(AmmassoTokenDescription);
  }
  if (contextBlock->num_freeRecvBuffers < n) {
    int quantity = (n - contextBlock->num_freeRecvBuffers + 1023) & (~1023);
    BufferAlloc(quantity);
  }
  buf = (char*) CmiAlloc(n*sizeof(AmmassoTokenDescription) + sizeof(int));
  *((int*)buf) = n;
  tokenDesc = (AmmassoTokenDescription*)(buf+sizeof(int));
  buffer = contextBlock->freeRecvBuffers;
  for (i=0; i<n; ++i) {
    tokenDesc[i].stag = buffer->stag;
    tokenDesc[i].to = (unsigned long)buffer;
    prebuffer = buffer;
    buffer = buffer->next;
  }
  node->pending = contextBlock->freeRecvBuffers;
  node->last_pending = prebuffer;
  node->num_pending = n;
  prebuffer->next = NULL;
  contextBlock->num_freeRecvBuffers -= n;
  contextBlock->freeRecvBuffers = buffer;
  sendDataOnQP(buf, n*sizeof(AmmassoTokenDescription) + sizeof(int), node, AMMASSO_ALLOCATE);
  CmiFree(buf);
}

void RequestReleaseTokens(OtherNode node, int n) {
  char buf[24];
  *((int*)buf) = n;
  sendDataOnQP(buf, sizeof(int), node, AMMASSO_RELEASE);
}

void ReleaseTokens(OtherNode node, int n) {
  int i, nWR;
  AmmassoToken *token;
  AmmassoBuffer *tokenBuf;
  cc_data_addr_t *tokenSgl;

  if (node->num_sendTokens < n) n = node->num_sendTokens - 1;
  if (n <= 0) return;
  token = node->sendTokens;
  tokenBuf = token->localBuf;

  tokenBuf->tail.length = 1;
  tokenBuf->tail.ack = 0;  // do not send any ACK with this message
  tokenBuf->tail.flags = AMMASSO_RELEASED;

  // Setup the local SGL
  tokenSgl = token->wr.wr_u.rdma_write.local_sgl.sge_list;
  tokenSgl->length = sizeof(Tailer);
  tokenSgl->to = (unsigned long)&tokenBuf->tail;
  token->wr.wr_u.rdma_write.remote_to = (unsigned long)&token->remoteBuf->tail;

  CC_POST_CHECK(cc_qp_post_sq,(contextBlock->rnic, node->qp, &token->wr, 1, &nWR),node->myNode);

  if (contextBlock->freeTokens == NULL) {
    contextBlock->freeTokens = node->sendTokens;
  } else {
    contextBlock->last_freeTokens->next = node->sendTokens;
  }
  for (i=1; i<n; ++i) token = token->next;
  contextBlock->last_freeTokens = token;
  node->sendTokens = token->next;
  token->next = NULL;
  contextBlock->num_freeTokens += n;
  node->num_sendTokens -= n;
}

/* CmiMachineInit()
 *   This is called as the node is starting up.  It does some initialization of the machine layer.
 */
void CmiMachineInit(char **argv) {

  char buf[128];
  cc_status_t rtn;

  AMMASSO_STATS_INIT

  AMMASSO_STATS_START(MachineInit)

  MACHSTATE(2, "CmiMachineInit() - INFO: (***** Ammasso Specific*****) - Called... Initializing RNIC...");
  MACHSTATE1(1, "CmiMachineInit() - INFO: Cmi_charmrun_pid = %d", Cmi_charmrun_pid);


  //CcdCallOnConditionKeep(CcdPERIODIC, (CcdVoidFn)periodicFunc, NULL);


  // Allocate a context block that will be used throughout this machine layer
  if (contextBlock != NULL) {
    MACHSTATE(5, "CmiMachineInit() - ERROR: contextBlock != NULL");
    sprintf(buf, "CmiMachineInit() - ERROR: contextBlock != NULL");
    CmiAbort(buf);
  }
  contextBlock = (mycb_t*)malloc(sizeof(mycb_t));
  if (contextBlock == NULL) {
    MACHSTATE(5, "CmiMachineInit() - ERROR: Unable to malloc memory for contextBlock");
    sprintf(buf, "CmiMachineInit() - ERROR: Unable to malloc memory for contextBlock");
    CmiAbort(buf);
  }

  // Initialize the contextBlock by zero-ing everything out and then setting anything special
  memset(contextBlock, 0, sizeof(mycb_t));
  contextBlock->rnic = -1;

  MACHSTATE(1, "CmiMachineInit() - INFO: (PRE-OPEN_RNIC)");

  // Check to see if in stand-alone mode
  if (Cmi_charmrun_pid != 0) {

    // Try to Open the RNIC
    //   TODO : Look-up the difference between CC_PBL_PAGE_MODE and CC_PBL_BLOCK_MODE
    //   TODO : Would a call to cc_rnic_enum or cc_rnic_query do any good here?
    rtn = cc_rnic_open(0, CC_PBL_PAGE_MODE, contextBlock, &(contextBlock->rnic));
    if (rtn != CC_OK) {
      MACHSTATE2(5, "CmiMachineInit() - ERROR: Unable to open RNIC: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "CmiMachineInit() - ERROR: Unable to open RNIC: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(1, "CmiMachineInit() - INFO: (PRE-SET-ASYNC-HANDLER)");

    // Set the asynchronous event handler function
    CC_CHECK(cc_eh_set_async_handler,(contextBlock->rnic, AsynchronousEventHandler, contextBlock));

    /*
    MACHSTATE(3, "CmiMachineInit() - INFO: (PRE-SET-CE-HANDLER)");

    // Set the Completion Event Handler
    contextBlock->eh_id = 0;
    CC_CHECK(cc_eh_set_ce_handler,(contextBlock->rnic, CompletionEventHandler, &(contextBlock->eh_id)));
    */

    MACHSTATE(1, "CmiMachineInit() - INFO: (PRE-PD-ALLOC)");

    // Allocate the Protection Domain
    CC_CHECK(cc_pd_alloc,(contextBlock->rnic, &(contextBlock->pd_id)));

    MACHSTATE(1, "CmiMachineInit() - INFO: RNIC Open For Business!!!");

  } else {  // Otherwise, not in stand-alone mode

    // Flag the rnic variable as invalid
    contextBlock->rnic = -1;
  }

  MACHSTATE(2, "CmiMachineInit() - INFO: Completed Successfully !!!");

  AMMASSO_STATS_END(MachineInit)
}

void CmiCommunicationInit(char **argv)
{
}

void CmiAmmassoNodeAddressesStoreHandler(int pe, struct sockaddr_in *addr, int port) {

  // DMK : NOTE : The hope is that this can be used to request the RMDA addresses of the other nodes after the
  //              initial addresses from charmrun are given to the node.  Get the address here, use that to request
  //              the RDMA address, use the RDMA address to create the QP connection (in establishQPConnection(), which
  //              only subtracts one from the address at the moment... the way our cluster is setup).

  MACHSTATE1(2, "CmiNodeAddressesStoreHandler() - INFO: pe = %d", pe);
  MACHSTATE1(1, "                                       addr = { sin_family = %d,", addr->sin_family);
  MACHSTATE1(1, "                                                sin_port = %d,", addr->sin_port);
  MACHSTATE4(1, "                                                sin_addr.s_addr = %d.%d.%d.%d }", (addr->sin_addr.s_addr & 0xFF), ((addr->sin_addr.s_addr >> 8) & 0xFF), ((addr->sin_addr.s_addr >> 16) & 0xFF), ((addr->sin_addr.s_addr >> 24) & 0xFF));
  MACHSTATE1(1, "                                       port = %d", port);
}


void AmmassoDoIdle() {

  int i;
  cc_wc_t wc;

  AMMASSO_STATS_START(AmmassoDoIdle)

  /* DYNAMIC ALLOCATOR: Callbacks */
  /*if (contextBlock->conditionRegistered == 0) {
    CcdCallOnConditionKeep(CcdPERIODIC_1s, (CcdVoidFn) AllocatorCheck, NULL);
    //CcdCallFnAfter((CcdVoidFn) AllocatorCheck, NULL, 100);
    contextBlock->conditionRegistered = 1;
    }*/
  /* DYNAMIC ALLOCATOR END */

  for (i = 0; i < contextBlock->numNodes; i++) {
    if (i == contextBlock->myNode) continue;
    CheckRecvBufForMessage(&(nodes[i]));
    while (cc_cq_poll(contextBlock->rnic, nodes[i].send_cq, &wc) == CC_OK) {
      MACHSTATE1(3, "AmmassoDoIdle() - INFO: Send completed with node %d... now waiting for acknowledge...", i);
    }
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

/* NOTE: if the ack overflows, we cannot use this method of sending, but we need
   to send a special message for the purpose */
void sendAck(OtherNode node) {

  int nWR;

  AMMASSO_STATS_START(sendAck)

  MACHSTATE2(3, "sendAck() - Ammasso - INFO: Called... sending ACK %d to node %d", *node->remoteAck, node->myNode);

  if (*node->remoteAck < ACK_MASK) {

    // Send an ACK message to the specified QP/Connection/Node
    CC_POST_CHECK(cc_qp_post_sq,(contextBlock->rnic, node->qp, node->ack_sq_wr, 1, &nWR),node->myNode);

  } else {
    // Rare case: happens only after days of run! In this case, do not update
    // directly the ack, but zero it on the other side, wait one second to be
    // safe, and then send a regular message with the ACK
    AmmassoToken *token;
    AmmassoBuffer *tokenBuf;
    int tmp_ack = *node->remoteAck;
    *node->remoteAck = 0;
    CC_POST_CHECK(cc_qp_post_sq,(contextBlock->rnic, node->qp, node->ack_sq_wr, 1, &nWR),node->myNode);

    sleep(1);

    token = getQPSendToken(node);
    tokenBuf = token->localBuf;
    tokenBuf->tail.ack = tmp_ack;
    tokenBuf->tail.flags = ACK_WRAPPING;
    tokenBuf->tail.length = 1; // So it will be seen by the receiver
    token->wr.wr_u.rdma_write.local_sgl.sge_list->length = sizeof(Tailer);
    token->wr.wr_u.rdma_write.local_sgl.sge_list->to = (unsigned long)&tokenBuf->tail;
    token->wr.wr_u.rdma_write.remote_to = (unsigned long)&token->remoteBuf->tail;
    CC_POST_CHECK(cc_qp_post_sq,(contextBlock->rnic, node->qp, &token->wr, 1, &nWR),node->myNode);
    LIST_ENQUEUE(node->,usedTokens,token);
    node->max_used_tokens = (node->num_usedTokens>node->max_used_tokens)?node->num_usedTokens:node->max_used_tokens;
    *node->remoteAck = tmp_ack & ACK_MASK;
  }

  node->messagesNotYetAcknowledged = 0;

  AMMASSO_STATS_END(sendAck)
}

/* NOTE, even in SMP versions, only the communication server should be sending
   messages out, thus no locking should be necessary */
/* This function returns the token usable for next communication. If no token is
   available, it blocks until one becomes available */
AmmassoToken *getQPSendToken(OtherNode node) {
  AmmassoToken *token;
  int i;
  cc_wc_t wc;
  ammasso_ack_t newAck;
  while (node->connectionState != QP_CONN_STATE_CONNECTED ||
	 node->sendTokens == NULL) {
    // Try to see if an ACK has been sent directly, so we free some tokens The
    // direct token will never be greater than ACK_MASK (by protocol
    // definition), so we do not need to wrap around
    MACHSTATE(3, "getQPSendBuffer() - INFO: No tokens available");
    if (*node->directAck > node->localAck) {
      newAck = *node->directAck;
      for (i=node->localAck; i<newAck; ++i) {
	LIST_DEQUEUE(node->,usedTokens,token);
	LIST_ENQUEUE(node->,sendTokens,token);
      }
      node->localAck = newAck;
    }

    CheckRecvBufForMessage(node);

    while (cc_cq_poll(contextBlock->rnic, node->send_cq, &wc) == CC_OK) {
      MACHSTATE1(3, "getQPSendBuffer() - INFO: Send completed with node %d... now waiting for acknowledge...", node->myNode);
    }
  }
  LIST_DEQUEUE(node->,sendTokens,token);
  return token;
}

/*
int getQPSendBuffer(OtherNode node, char force) {

  int rtnBufIndex, i;
  cc_wc_t wc;

  AMMASSO_STATS_START(getQPSendBuffer)

  MACHSTATE1(3, "getQPSendBuffer() - Ammasso - Called (send to node %d)...", node->myNode);

  while (1) {

    AMMASSO_STATS_START(getQPSendBuffer_loop)

    rtnBufIndex = -1;

    AMMASSO_STATS_START(getQPSendBuffer_lock)

    MACHSTATE(3, "getQPSendBuffer() - INFO: Pre-sendBufLock");
    #if CMK_SHARED_VARS_UNAVAILABLE
      while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
    #endif
    CmiLock(node->sendBufLock);

    AMMASSO_STATS_END(getQPSendBuffer_lock)
    
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
 
    CmiUnlock(node->sendBufLock);
    MACHSTATE3(3, "getQPSendBuffer() - INFO: Post-sendBufLock - rtnBufIndex = %d, node->connectionState = %d, node->send_UseIndex = %d", rtnBufIndex, node->connectionState, node->send_UseIndex);

    if (rtnBufIndex >= 0) {

      AMMASSO_STATS_END(getQPSendBuffer_loop)
      break;

    } else {

      //usleep(1);

      AMMASSO_STATS_START(getQPSendBuffer_CEH)

      CheckRecvBufForMessage(node);
      //CompletionEventHandlerWithAckFlag(contextBlock->rnic, node->recv_cq, node, 1);

      //CompletionEventHandler(contextBlock->rnic, node->send_cq, node);
      while (cc_cq_poll(contextBlock->rnic, node->send_cq, &wc) == CC_OK) {
	MACHSTATE1(3, "getQPSendBuffer() - INFO: Send completed with node %d... now waiting for acknowledge...", nextNode);
      }


      AMMASSO_STATS_END(getQPSendBuffer_CEH)

      AMMASSO_STATS_END(getQPSendBuffer_loop)
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
*/

// NOTE: The force parameter can be thought of as an "is ACK" control message flag (see comments in getQPSendBuffer())
int sendDataOnQP(char* data, int len, OtherNode node, char flags) {

  AmmassoToken *sendBufToken;
  AmmassoBuffer *tokenBuf;
  cc_data_addr_t *tokenSgl;
  int toSendLength;
  cc_wc_t wc;
  cc_uint32_t nWR;
  char isFirst = 1;
  char *origMsgStart = data;
  char *sendBufBegin;

  int origSize = len;

  if (origSize <= 1024) {
    AMMASSO_STATS_START(sendDataOnQP_1024)
  } else if (origSize <= 2048) { 
    AMMASSO_STATS_START(sendDataOnQP_2048)
  } else if (origSize <= 4096) { 
    AMMASSO_STATS_START(sendDataOnQP_4096)
  } else if (origSize <= 16384) { 
    AMMASSO_STATS_START(sendDataOnQP_16384)
  } else {
    AMMASSO_STATS_START(sendDataOnQP_over)
  }

  AMMASSO_STATS_START(sendDataOnQP)

  //CompletionEventHandler(contextBlock->rnic, node->recv_cq, node);
  //CompletionEventHandler(contextBlock->rnic, node->send_cq, node);
  while (cc_cq_poll(contextBlock->rnic, node->send_cq, &wc) == CC_OK) {
    MACHSTATE1(3, "sendDataOnQP() - INFO: Send completed with node %d... now waiting for acknowledge...", node->myNode);
  }


  MACHSTATE2(2, "sendDataOnQP() - Ammasso - INFO: Called (send to node %d, len = %d)...", node->myNode, len);

  // Assert that control messages will not be fragmented
  CmiAssert(flags==0 || len<=AMMASSO_BUFSIZE);

  // DMK : For each message that is fragmented, attach another DGRAM header to
  // it, (keeping in mind that the control messages are no where near large
  // enough for this to occur).

  while (len > 0) {

    AMMASSO_STATS_START(sendDataOnQP_pre_send)  

    // Get a free send buffer (NOTE: This call will block until a send buffer is free)
    sendBufToken = getQPSendToken(node);
    tokenBuf = sendBufToken->localBuf;
    // Enqueue the token to the used queue immediately, so it is safe to be
    // interrupted by other calls
    LIST_ENQUEUE(node->,usedTokens,sendBufToken);
    node->max_used_tokens = (node->num_usedTokens>node->max_used_tokens)?node->num_usedTokens:node->max_used_tokens;

    // Copy the contents (up to AMMASSO_BUFSIZE worth) of data into the send buffer

    // The toSendLength includes the DGRAM header size. If the chunk sent is not
    // the first, the initial DGRAM_HEADER_SIZE bytes need to be contructed from
    // the DGRAM header of the original message (instead of just being copied
    // together with the message itself)

    if (isFirst) {

      toSendLength = len > AMMASSO_BUFSIZE ? AMMASSO_BUFSIZE : len;  // MIN of len and AMMASSO_BUFSIZE
      sendBufBegin = tokenBuf->buf + AMMASSO_BUFSIZE - ALIGN8(toSendLength);

      memcpy(sendBufBegin, data, toSendLength);

      MACHSTATE1(1, "sendDataOnQP() - Ammasso - INFO: Sending 1st Fragment - toSendLength = %d...", toSendLength);

    } else {

      toSendLength = len > (AMMASSO_BUFSIZE - DGRAM_HEADER_SIZE) ? AMMASSO_BUFSIZE : (len+DGRAM_HEADER_SIZE);  // MIN of len and AMMASSO_BUFSIZE
      sendBufBegin = tokenBuf->buf + AMMASSO_BUFSIZE - ALIGN8(toSendLength);

      memcpy(sendBufBegin+DGRAM_HEADER_SIZE, data, toSendLength-DGRAM_HEADER_SIZE);
      
      // This dgram header is the same of the original message, except for the
      // sequence number, so copy the original and just modify the sequence
      // number.

      // NOTE: If the message is large enough that fragmentation needs to
      // happen, the send_next_lock is already owned by the thread executing
      // this code.
      memcpy(sendBufBegin, origMsgStart, DGRAM_HEADER_SIZE);

      ((DgramHeader*)sendBufBegin)->seqno = node->send_next;
      node->send_next = ((node->send_next+1) & DGRAM_SEQNO_MASK);  // Increase the sequence number

      MACHSTATE1(1, "sendDataOnQP() - Ammasso - INFO: Sending Continuation Fragment - toSendLength = %d...", toSendLength);
    }

    // Write the size of the message at the end of the buffer, with the ack and
    // flags
    tokenBuf->tail.length = toSendLength;
    node->messagesNotYetAcknowledged = 0;
    tokenBuf->tail.ack = *node->remoteAck;
    if (*node->remoteAck > ACK_MASK) {
      // Rare case of ACK wrapping
      tokenBuf->tail.ack = 0;
      // in the rare case that we didn't send an ack with this message because
      // the ack just wrapped around (rare case), send a full ACK message. It is
      // safe to send it now, since the queues are consistent between sender and
      // receiver. The fact that this ack is effectively sent before the regular
      // message is not important, since on the other side it will be discovered
      // only after this one is received
      sendAck(node);
    }
    tokenBuf->tail.flags = flags;

    // Setup the local SGL
    tokenSgl = sendBufToken->wr.wr_u.rdma_write.local_sgl.sge_list;
    tokenSgl->length = ALIGN8(toSendLength) + sizeof(Tailer);
    tokenSgl->to = (unsigned long)sendBufBegin;
    sendBufToken->wr.wr_u.rdma_write.remote_to = (unsigned long)(((char*)sendBufToken->remoteBuf)+AMMASSO_BUFSIZE-ALIGN8(toSendLength));

    // The remote_to and remote_stag are already fixed part of the token
 
    AMMASSO_STATS_END(sendDataOnQP_pre_send)  
    AMMASSO_STATS_START(sendDataOnQP_send)

    MACHSTATE(3, "sendDataOnQP() - Ammasso - INFO: Enqueuing RDMA Write WR...");

    MACHSTATE1(1, "sendDataOnQP() - Ammasso - INFO: tokenSgl->to = %p", tokenSgl->to);
    MACHSTATE1(1, "sendDataOnQP() - Ammasso - INFO: sendBufToken->wr.wr_u.rdma_write.remote_to = %p", sendBufToken->wr.wr_u.rdma_write.remote_to);
    MACHSTATE1(1, "sendDataOnQP() - Ammasso - INFO: tail.ack = %d", tokenBuf->tail.ack);
    MACHSTATE1(1, "sendDataOnQP() - Ammasso - INFO: tail.flags = %d", tokenBuf->tail.flags);

    CC_POST_CHECK(cc_qp_post_sq,(contextBlock->rnic, node->qp, &sendBufToken->wr, 1, &nWR),node->myNode);
    
    MACHSTATE(1, "sendDataOnQP() - Ammasso - INFO: RDMA Write WR Enqueue Completed");

    AMMASSO_STATS_END(sendDataOnQP_send)
    AMMASSO_STATS_START(sendDataOnQP_post_send)  

    // Update the data and len variables for the next while (if fragmenting is needed)
    data += toSendLength;
    len -= toSendLength;
    if (isFirst == 0) {
      data -= DGRAM_HEADER_SIZE;
      len += DGRAM_HEADER_SIZE;
    }
    isFirst = 0;

    AMMASSO_STATS_END(sendDataOnQP_post_send)  
  }

  AMMASSO_STATS_END(sendDataOnQP)

  if (origSize <= 1024) {
    AMMASSO_STATS_END(sendDataOnQP_1024)
  } else if (origSize <= 2048) { 
    AMMASSO_STATS_END(sendDataOnQP_2048)
  } else if (origSize <= 4096) { 
    AMMASSO_STATS_END(sendDataOnQP_4096)
  } else if (origSize <= 16384) { 
    AMMASSO_STATS_END(sendDataOnQP_16384)
  } else {
    AMMASSO_STATS_END(sendDataOnQP_over)
  }

}


/* DeliverViaNetwork()
 *
 */
void DeliverViaNetwork(OutgoingMsg msg, OtherNode otherNode, int rank, unsigned int broot, int copy) {

  cc_status_t rtn;
  cc_stag_index_t stag;
  cc_data_addr_t sgl;
  cc_sq_wr_t wr;
  cc_uint32_t WRsPosted;

  AMMASSO_STATS_START(DeliverViaNetwork)

  MACHSTATE(2, "DeliverViaNetwork() - Ammasso - INFO: Called...");

  // We don't need to do this since the message data is being copied into the
  // send_buf, the OutgoingMsg can be free'd ASAP

  // The lock will be already held by the calling function in machine.c
  // (CommLock)

  AMMASSO_STATS_START(DeliverViaNetwork_post_lock)

  DgramHeaderMake(msg->data, rank, msg->src, Cmi_charmrun_pid, otherNode->send_next, broot);  // Set DGram Header Fields In-Place
  otherNode->send_next = ((otherNode->send_next+1) & DGRAM_SEQNO_MASK);  // Increase the sequence number

  MACHSTATE1(1, "DeliverViaNetwork() - INFO: Sending message to  node %d", otherNode->myNode);
  MACHSTATE1(1, "DeliverViaNetwork() - INFO:                     rank %d", rank);
  MACHSTATE1(1, "DeliverViaNetwork() - INFO:                    broot %d", broot);

  AMMASSO_STATS_START(DeliverViaNetwork_send)

  sendDataOnQP(msg->data, msg->size, otherNode, 0);

  AMMASSO_STATS_END(DeliverViaNetwork_send)

    //CmiUnlock(otherNode->send_next_lock);
  MACHSTATE(1, "DeliverViaNetwork() - INFO: Post-send_next_lock");


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

  MACHSTATE(2, "DeliverViaNetwork() - Ammasso - INFO: Completed.");

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

void processAmmassoControlMessage(char* msg, int len, Tailer *tail, OtherNode from) {

  int nodeIndex, ctrlType, i, n, nWR;
  AmmassoToken *token, *pretoken;
  AmmassoBuffer *tokenBuf;
  cc_data_addr_t *tokenSgl;
  OtherNode node;
  AmmassoTokenDescription *tokenDesc;

  AMMASSO_STATS_START(processAmmassoControlMessage)

  // Do not check the message, the flags field was set, and this is enough

  // Perform an action based on the control message type
  switch (tail->flags) {

  case AMMASSO_READY:

    // Decrement the node ready count by one
    contextBlock->nodeReadyCount--;
    MACHSTATE1(3, "processAmmassoControlMessage() - Ammasso - INFO: Received READY packet... still waiting for %d more...", contextBlock->nodeReadyCount);

    break;

  case AMMASSO_ALLOCATE: // Sent by the receiver to allocate more tokens

    token = getQPSendToken(from);
    tokenBuf = token->localBuf;

    tokenBuf->tail.length = 1;
    tokenBuf->tail.ack = 0;  // do not send any ACK with this message
    tokenBuf->tail.flags = AMMASSO_ALLOCATED;

    // Setup the local SGL
    tokenSgl = token->wr.wr_u.rdma_write.local_sgl.sge_list;
    tokenSgl->length = sizeof(Tailer);
    tokenSgl->to = (unsigned long)&tokenBuf->tail;
    token->wr.wr_u.rdma_write.remote_to = (unsigned long)&token->remoteBuf->tail;

    CC_POST_CHECK(cc_qp_post_sq,(contextBlock->rnic, node->qp, &token->wr, 1, &nWR),node->myNode);

    LIST_ENQUEUE(from->,usedTokens,token);
    from->max_used_tokens = (from->num_usedTokens>from->max_used_tokens)?from->num_usedTokens:from->max_used_tokens;
    // add the new tokens at the end of usedTokens (right after the one which
    // which we are sending back the confirmation)
    n = *((int*)msg);
    if (contextBlock->num_freeTokens < n) {
      int quantity = (n - contextBlock->num_freeTokens + 1023) & (~1023);
      BufferAlloc(quantity);
    }
    token = contextBlock->freeTokens;
    tokenDesc = (AmmassoTokenDescription*)(msg+sizeof(int));
    for (i=0; i<n; ++i) {
      token->remoteBuf = (AmmassoBuffer*)tokenDesc[i].to;
      token->wr.wr_u.rdma_write.remote_stag = tokenDesc[i].stag;
      pretoken = token;
      token = token->next;
    }
    from->last_usedTokens->next = contextBlock->freeTokens;
    from->last_usedTokens = pretoken;
    pretoken->next = NULL;
    contextBlock->freeTokens = token;
    from->num_usedTokens += n;
    contextBlock->num_freeTokens -= n;

    break;

  case AMMASSO_ALLOCATED: // Sent by the sender to conferm the token allocation

    // link the pending buffers at the end of those allocated (right after the
    // one with which this message came)
    from->last_recv_buf->next = from->pending;
    from->last_recv_buf = from->last_pending;
    from->last_pending = NULL;
    from->num_recv_buf += from->num_pending;
    *from->remoteAck += from->num_pending;
    from->num_pending = 0;

    break;

  case AMMASSO_MOREBUFFERS: // Sent by the sender to ask for more tokens

    /* DYNAMIC ALLOCATOR: Grant what requested */
    n = *((int*)msg);
    GrantTokens(from, n);
    /* DYNAMIC ALLOCATOR END */

    break;

  case AMMASSO_RELEASE: // Sent by the receiver to request back tokens

    /* DYNAMIC ALLOCATOR: Ignore the request of the receiver to return buffers */
    /* DYNAMIC ALLOCATOR END */

    break;

  case AMMASSO_RELEASED: // Sent by the sender to release tokens

    // This secondLastRecvBuf is set in CheckRecvBufForMessage
    from->secondLastRecvBuf->next = NULL;
    tokenBuf = from->last_recv_buf;
    from->last_recv_buf = from->secondLastRecvBuf;
    from->num_recv_buf--;
    LIST_ENQUEUE(contextBlock->,freeRecvBuffers,tokenBuf);
    n = *((int*)msg);
    for (i=1; i<n; ++i) {
      LIST_DEQUEUE(from->,recv_buf,tokenBuf);
      LIST_ENQUEUE(contextBlock->,freeRecvBuffers,tokenBuf);
    }

    break;

  case ACK_WRAPPING:

    // The ACK has already been accounted in ProcessMessage, just mask it
    from->localAck &= ACK_MASK;

    break;

  default:
    MACHSTATE1(5, "processAmmassoControlMessage() - Ammasso -INFO: Received control message with invalid flags: %d", tail->flags);
    CmiAbort("Invalid control message received");
  }

  AMMASSO_STATS_END(processAmmassoControlMessage)
}

int ProcessMessage(char* msg, int len, Tailer *tail, OtherNode from) {

  int rank, srcPE, seqNum, magicCookie, size, i;
  unsigned int broot;
  unsigned char checksum;
  OtherNode fromNode;
  char *newMsg;
  int needAck;

  AMMASSO_STATS_START(ProcessMessage)

  MACHSTATE(2, "ProcessMessage() - INFO: Called...");
  MACHSTATE2(1, "ProcessMessage() - INFO: tail - ack=%d, flags=%d", tail->ack, tail->flags);

  // Decide whether a direct ACK will be needed or not, based on how many
  // messages the other side has sent us, and we haven't acknowledged yet
  if (2*from->messagesNotYetAcknowledged > from->num_recv_buf) {
    needAck = 1;
  } else {
    needAck = 0;
  }

  // This message contains an ACK as all messages, parse it. Do not worry about
  // wrap around of the ACK, since when this happen a special control message
  // ACK_WRAPPING is received
  if (tail->ack > from->localAck) {
    AmmassoToken *token;
    for (i=from->localAck; i<tail->ack; ++i) {
      LIST_DEQUEUE(from->,usedTokens,token);
      LIST_ENQUEUE(from->,sendTokens,token);
    }
    from->localAck = tail->ack;
  }

  {
    MACHSTATE1(1, "ProcessMessage() - INFO: msg = %p", msg);
    int j;
    for (j = 0; j < DGRAM_HEADER_SIZE + 24; j++) {
      MACHSTATE2(1, "ProcessMessage() - INFO: msg[%d] = %02x", j, msg[j]);
    }
  }

  if (tail->flags != 0) {
    processAmmassoControlMessage(msg, len, tail, from);
    return needAck;
  }

  // Get the header fields of the message
  DgramHeaderBreak(msg, rank, srcPE, magicCookie, seqNum, broot);

  MACHSTATE(1, "ProcessMessage() - INFO: Message Contents:");
  MACHSTATE1(1, "                           rank = %d", rank);
  MACHSTATE1(1, "                           srcPE = %d", srcPE);
  MACHSTATE1(1, "                           magicCookie = %d", magicCookie);
  MACHSTATE1(1, "                           seqNum = %d", seqNum);
  MACHSTATE1(1, "                           broot = %d", broot);

#ifdef CMK_USE_CHECKSUM

  // Check the checksum
  checksum = computeCheckSum(msg, len);
  if (checksum != 0) {
    MACHSTATE1(5, "ProcessMessage() - Ammasso - ERROR: Received message with bad checksum (%d)... ignoring...", checksum);
    CmiPrintf("[%d] ProcessMessage() - Ammasso - ERROR: Received message with bad checksum (%d)... ignoring...\n", CmiMyPe(), checksum);
    return needAck;
  }

#else

  // Check the magic cookie for correctness
  if (magicCookie != (Cmi_charmrun_pid & DGRAM_MAGIC_MASK)) {
    MACHSTATE(5, "ProcessMessage() - Ammasso - ERROR: Received message with a bad magic cookie... ignoring...");
    CmiPrintf("[%d] ProcessMessage() - Ammasso - ERROR: Received message with a bad magic cookie... ignoring...\n", CmiMyPe());

    {
      CmiPrintf("ProcessMessage() - INFO: Cmi_charmrun_pid = %d\n", Cmi_charmrun_pid);
      CmiPrintf("ProcessMessage() - INFO: node->recv_UseIndex = %d\n", nodes_by_pe[srcPE]->recv_UseIndex);
      CmiPrintf("ProcessMessage() - INFO: msg = %p\n", msg);
      int j;
      for (j = 0; j < DGRAM_HEADER_SIZE + 24; j++) {
        CmiPrintf("ProcessMessage() - INFO: msg[%d] = %02x\n", j, msg[j]);
      }
    }
    return needAck;
  }

#endif

  // Get the OtherNode structure for the node this message was sent from
  fromNode = nodes_by_pe[srcPE];

  CmiAssert(fromNode == from);

  MACHSTATE1(1, "ProcessMessage() - INFO: Message from node %d...", fromNode->myNode);

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
    MACHSTATE(5, "ProcessMessage() - Ammasso - ERROR: Received a message with a bad sequence number... ignoring...");
    CmiPrintf("[%d] ProcessMessage() - Ammasso - ERROR: Received a message witha bad sequence number... ignoring...\n", CmiMyPe());
    return needAck;
  }

  //CmiUnlock(fromNode->recv_expect_lock);
  //MACHSTATE(3, "ProcessMessage() - INFO: Post-recv_expect_lock");

  newMsg = fromNode->asm_msg;

  // Check to see if this is the start of the message (i.e. - if the message was
  // fragmented, if this is the first packet of the message) or the entire
  // message. Only the first packet's header information will be copied into the
  // asm_buf buffer.
  if (newMsg == NULL) {

    // Allocate memory to hold the new message
    size = CmiMsgHeaderGetLength(msg);
    newMsg = (char*)CmiAlloc(size);
    _MEMCHECK(newMsg);
    
    // Verify the message size
    if (len > size) {
      MACHSTATE2(5, "ProcessMessage() - Ammasso - ERROR: Message size mismatch (size: %d != len: %d)", size, len);
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
      MACHSTATE(5, "ProcessMessage() - Ammasso - ERROR: Message size mismatch");
      CmiPrintf("[%d] ProcessMessage() - Ammasso - ERROR: Message size mismatch", CmiMyPe());
      CmiAbort("Message Size Mismatch");
    }

    // Copy the message into the asm_msg buffer
    memcpy(newMsg + fromNode->asm_fill, msg + DGRAM_HEADER_SIZE, size);
    fromNode->asm_fill += size;
  }

  MACHSTATE2(1, "ProcessMessage() - Ammasso - INFO: Message copied into asm_buf (asm_fill = %d, asm_total = %d)...", fromNode->asm_fill, fromNode->asm_total);

  // Check to see if a full packet has been received
  if (fromNode->asm_fill == fromNode->asm_total) {

    MACHSTATE(1, "ProcessMessage() - Ammasso - INFO: Pushing message...");

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

    MACHSTATE(1, "ProcessMessage() - Ammasso - INFO: NULLing asm_msg...");

    // Clear the message buffer
    fromNode->asm_msg = NULL;
  }

  MACHSTATE(1, "ProcessMessage() - Ammasso - INFO: Checking for re-broadcast");

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
  return needAck;
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

  int i;

  MACHSTATE(2, "CommunicationServer_nolock start {");

  // DMK : TODO : In spare time (here), check for messages and/or completions and/or errors
  //while(PollForMessage(contextBlock->send_cq));  // Clear all the completed sends
  //while(PollForMessage(contextBlock->recv_cq));  // Keep looping while there are still messages that have been received

  for (i = 0; i < contextBlock->numNodes; i++) {
    if (i == contextBlock->myNode) continue;
    //CompletionEventHandlerWithAckFlag(contextBlock->rnic, nodes[i].recv_cq, &(nodes[i]), 1);
    //CompletionEventHandler(contextBlock->rnic, nodes[i].send_cq, &(nodes[i]));
    CheckRecvBufForMessage(&(nodes[i]));
  }

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

  AMMASSO_STATS_START(CommunicationServer)

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
  inProgress[CmiMyRank()] += 1;
  CommunicationServer_nolock(withDelayMs);
  CmiCommUnlock();
  inProgress[CmiMyRank()] -= 1;

#if CMK_IMMEDIATE_MSG
  if (where == 0)
  CmiHandleImmediate();
#endif

  MACHSTATE(2,"} CommunicationServer");

  AMMASSO_STATS_END(CommunicationServer)
}



/* CmiMachineExit()
 *
 */
void CmiMachineExit(void) {

  char buf[128];
  cc_status_t rtn;
  int i;

  MACHSTATE(2, "CmiMachineExit() - INFO: Called...");

  // DMK - This is a sleep to help keep the output from the stat displays below separated in the program output
  if (contextBlock->myNode)
    usleep(10000*contextBlock->myNode);

  AMMASSO_STATS_DISPLAY(MachineInit)

  AMMASSO_STATS_DISPLAY(AmmassoDoIdle)

  AMMASSO_STATS_DISPLAY(DeliverViaNetwork)
  AMMASSO_STATS_DISPLAY(DeliverViaNetwork_pre_lock)
  AMMASSO_STATS_DISPLAY(DeliverViaNetwork_lock)
  AMMASSO_STATS_DISPLAY(DeliverViaNetwork_post_lock)
  AMMASSO_STATS_DISPLAY(DeliverViaNetwork_send)

  AMMASSO_STATS_DISPLAY(getQPSendBuffer)
  AMMASSO_STATS_DISPLAY(getQPSendBuffer_lock)
  AMMASSO_STATS_DISPLAY(getQPSendBuffer_CEH)
  AMMASSO_STATS_DISPLAY(getQPSendBuffer_loop)

  AMMASSO_STATS_DISPLAY(sendDataOnQP)
  AMMASSO_STATS_DISPLAY(sendDataOnQP_pre_send)
  AMMASSO_STATS_DISPLAY(sendDataOnQP_send)
  AMMASSO_STATS_DISPLAY(sendDataOnQP_post_send)

  AMMASSO_STATS_DISPLAY(sendDataOnQP_1024)
  AMMASSO_STATS_DISPLAY(sendDataOnQP_2048)
  AMMASSO_STATS_DISPLAY(sendDataOnQP_4096)
  AMMASSO_STATS_DISPLAY(sendDataOnQP_16384)
  AMMASSO_STATS_DISPLAY(sendDataOnQP_over)

  AMMASSO_STATS_DISPLAY(AsynchronousEventHandler)
  AMMASSO_STATS_DISPLAY(CompletionEventHandler)
  AMMASSO_STATS_DISPLAY(ProcessMessage)
  AMMASSO_STATS_DISPLAY(processAmmassoControlMessage)

  AMMASSO_STATS_DISPLAY(sendAck)

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
      MACHSTATE2(5, "CmiMachineExit() - ERROR: Unable to close the RNIC: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      sprintf(buf, "CmiMachineExit() - ERROR: Unable to close the RNIC: %d, \"%s\"", rtn, cc_status_to_string(rtn));
      CmiAbort(buf);
    }

    MACHSTATE(2, "CmiMachineExit() - INFO: RNIC Closed.");
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

  int nodeNumber, i;
  OtherNode node;
  cc_ep_handle_t connReqEP;
  cc_status_t rtn;
  char buf[64];
  cc_qp_modify_attrs_t modAttrs;
  AmmassoPrivateData *priv;
  AmmassoToken *token;

  AMMASSO_STATS_START(AsynchronousEventHandler)

  MACHSTATE2(2, "AsynchronousEventHandler() - INFO: Called... event_id = %d, \"%s\"", er->event_id, cc_event_id_to_string(er->event_id));

  // Do a couple of checks... the reasons for these stem from some example code
  if (er->rnic_handle != contextBlock->rnic) {
    MACHSTATE(5, "AsynchronousEventHandler() - WARNING: er->rnic_handle != contextBlock->rnic");
  }
  if (er->rnic_user_context != contextBlock) {
    MACHSTATE(5, "AsynchronousEventHandler() - WARNING: er->rnic_user_context != contextBlock");
  }

  // Based on the er->event_id, do something about it
  switch (er->event_id) {

    // er->event_id == CCAE_LLP_CLOSE_COMPLETE
    case CCAE_LLP_CLOSE_COMPLETE:
      MACHSTATE(1, "AsynchronousEventHandler() - INFO: Connection Closed.");

      // Get the OtherNode structure for the other node
      MACHSTATE2(1, "AsynchronousEventHandler() - INFO: er->resource_indicator = %d (CC_RES_IND_QP: %d)", er->resource_indicator, CC_RES_IND_QP);
      node = getNodeFromQPId(er->resource_id.qp_id);
      if (node == NULL) {
        MACHSTATE(5, "AsynchronousEventHandler() - ERROR: Unable to find QP from QP ID... Unable to create/recover connection");
        break;
      }

      MACHSTATE(1, "AsynchronousEventHandler() - INFO: Pre-sendBufLock");
      #if CMK_SHARED_VARS_UNAVAILABLE
        while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
      #endif
      CmiLock(node->sendBufLock);

      // Set the state of the connection to closed
      node->connectionState = QP_CONN_STATE_CONNECTION_CLOSED;
  
      CmiUnlock(node->sendBufLock);
      MACHSTATE(1, "AsynchronousEventHandler() - INFO: Post-sendBufLock");
    
      break;

    // er->event_id == CCAE_CONNECTION_REQUEST
    case CCAE_CONNECTION_REQUEST:

      MACHSTATE2(1, "AsynchronousEventHandler() - INFO: Incomming Connection Request -> %s:%d",
                    inet_ntoa(*(struct in_addr*) &(er->event_data.connection_request.laddr)),
                    ntohs(er->event_data.connection_request.lport)
                );

      connReqEP = er->event_data.connection_request.cr_handle;

      priv = (AmmassoPrivateData*)er->event_data.connection_request.private_data;

      nodeNumber = priv->node;
 
      MACHSTATE1(3, "AsynchronousEventHandler() - INFO: Connection Request from node %d", nodeNumber);

      if (nodeNumber < 0 || nodeNumber >= contextBlock->numNodes) {

        // Refuse the connection and log the rejection
        MACHSTATE1(1, "AsynchronousEventHandler() - WARNING: Unknown entity attempting to connect (node %d)... rejecting connection.", nodeNumber);
        cc_cr_reject(contextBlock->rnic, connReqEP);

      } else {


        { int j;
          for (j = 0; j < 16; j++) {
            MACHSTATE2(3, "                                      private_data[%d] = %02X", j, ((char*)er->event_data.connection_request.private_data)[j]);
	  }
	}

        // Grab the remote stag and to, by the protocol, all the tokens are
        // consecutive in memory
	for (i=0; i<(AMMASSO_INITIAL_BUFFERS/(contextBlock->numNodes-1)); ++i) {
	  LIST_DEQUEUE(contextBlock->,freeTokens,token);
	  token->wr.wr_u.rdma_write.remote_stag = priv->stag;
	  token->wr.wr_u.rdma_write.remote_to = priv->to + (i * sizeof(AmmassoBuffer));
	  token->remoteBuf = (AmmassoBuffer*)(priv->to + (i * sizeof(AmmassoBuffer)));
	  LIST_ENQUEUE(nodes[nodeNumber].,sendTokens,token);
	}

	nodes[nodeNumber].ack_sq_wr->wr_u.rdma_write.remote_to = priv->ack_to;
	nodes[nodeNumber].ack_sq_wr->wr_u.rdma_write.remote_stag = priv->stag;

        MACHSTATE2(1, "AsynchronousEventHandler() - INFO: tokens starting from %p, stag = %d",priv->to,priv->stag);

        // Keep a copy of the end point handle
        nodes[nodeNumber].cr = connReqEP;

        // Accept the connection
        priv = (AmmassoPrivateData*)buf;
        priv->node = contextBlock->myNode;
        priv->stag = nodes[nodeNumber].recv_buf->stag;
        priv->to = (cc_uint64_t)nodes[nodeNumber].recv_buf;
	priv->ack_to = (cc_uint64_t)nodes[nodeNumber].directAck;

        MACHSTATE1(1, "                                          node = %d", priv->node);
        MACHSTATE1(1, "                                          stag = %d", priv->stag);
        MACHSTATE1(1, "                                          to = %p", priv->to);
        
        { int j;
          MACHSTATE2(3, "                                      buf = %p, priv = %p", buf, priv);
          for (j = 0; j < 16; j++) {
            MACHSTATE2(3, "                                      ((char*)priv)[%d] = %02X", j, ((char*)priv)[j]);
	  }
	}

        MACHSTATE(1, "AsynchronousEventHandler() - Ammasso - INFO: Accepting Connection...");

        rtn = cc_cr_accept(contextBlock->rnic, connReqEP, nodes[nodeNumber].qp, sizeof(AmmassoPrivateData), (cc_uint8_t*)priv);
        if (rtn != CC_OK) {

          // Let the user know what happened
          MACHSTATE1(3, "AsynchronousEventHandler() - Ammasso - WARNING: Unable to accept connection from node %d", nodeNumber);

	} else {
  
          MACHSTATE1(3, "AsynchronousEventHandler() - Ammasso - INFO: Accepted Connection from node %d", nodeNumber);

          MACHSTATE(1, "AsynchronousEventHandler() - INFO: Pre-sendBufLock");
          #if CMK_SHARED_VARS_UNAVAILABLE
            while (nodes[nodeNumber].sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
          #endif
          CmiLock(nodes[nodeNumber].sendBufLock);

          // Indicate that this connection has been made only if is it the first time the connection was made (don't count re-connects)
          if (nodes[nodeNumber].connectionState == QP_CONN_STATE_PRE_CONNECT)
            (contextBlock->outstandingConnectionCount)--;
          nodes[nodeNumber].connectionState = QP_CONN_STATE_CONNECTED;

          CmiUnlock(nodes[nodeNumber].sendBufLock);
          MACHSTATE(1, "AsynchronousEventHandler() - INFO: Post-sendBufLock");

          MACHSTATE1(1, "AsynchronousEventHandler() - Connected to node %d", nodes[nodeNumber].myNode);
	}
      }

      break;

    // er->event_id == CCAE_ACTIVE_CONNECT_RESULTS
    case CCAE_ACTIVE_CONNECT_RESULTS:
      MACHSTATE(1, "AsynchronousEventHandler() - INFO: Connection Results");

      // Get the OtherNode structure for the other node
      MACHSTATE2(1, "AsynchronousEventHandler() - INFO: er->resource_indicator = %d (CC_RES_IND_QP: %d)", er->resource_indicator, CC_RES_IND_QP);
      node = getNodeFromQPId(er->resource_id.qp_id);
      if (node == NULL) {
        MACHSTATE(5, "AsynchronousEventHandler() - ERROR: Unable to find QP from QP ID... Unable to create/recover connection");
        break;
      }

      // Check to see if the connection was established or not
      if (er->event_data.active_connect_results.status != CC_CONN_STATUS_SUCCESS) {

        MACHSTATE(5, "                                     Connection Failed.");
        MACHSTATE1(5, "                                      - status: \"%s\"", cc_connect_status_to_string(er->event_data.active_connect_results.status));
        MACHSTATE1(5, "                                      - private_data_length = %d", er->event_data.active_connect_results.private_data_length);
        displayQueueQuery(node->qp, &(node->qp_attrs));

        // Attempt to reconnect (try again... don't give up... you can do it!)
        reestablishQPConnection(node);

      } else { // Connection was a success

        MACHSTATE(3, "                                     Connection Success...");
        MACHSTATE2(1, "                                     l -> %s:%d", inet_ntoa(*(struct in_addr*) &(er->event_data.active_connect_results.laddr)), ntohs(er->event_data.active_connect_results.lport));
        MACHSTATE2(1, "                                     r -> %s:%d", inet_ntoa(*(struct in_addr*) &(er->event_data.active_connect_results.raddr)), ntohs(er->event_data.active_connect_results.rport));
        MACHSTATE4(1, "                                     private_data_length = %d (%d, %d, %d)", er->event_data.active_connect_results.private_data_length, sizeof(int), sizeof(cc_stag_t), sizeof(cc_uint64_t));

        priv = (AmmassoPrivateData*)((char*)er->event_data.active_connect_results.private_data);

        { int j;
	  MACHSTATE2(1, "                                      private_data = %p, priv = %p", er->event_data.active_connect_results.private_data, priv);
          for (j = 0; j < 16; j++) {
            MACHSTATE3(1, "                                      private_data[%d] = %02X (priv:%02X)", j, ((char*)(er->event_data.active_connect_results.private_data))[j], ((char*)priv)[j]);
	  }
	}

        // Grab the remote stag and to, by the protocol, all the tokens are
        // consecutive in memory
	for (i=0; i<(AMMASSO_INITIAL_BUFFERS/(contextBlock->numNodes-1)); ++i) {
	  LIST_DEQUEUE(contextBlock->,freeTokens,token);
	  token->wr.wr_u.rdma_write.remote_stag = priv->stag;
	  token->wr.wr_u.rdma_write.remote_to = priv->to + (i * sizeof(AmmassoBuffer));
	  token->remoteBuf = (AmmassoBuffer*)(priv->to + (i * sizeof(AmmassoBuffer)));
	  LIST_ENQUEUE(node->,sendTokens,token);
	}

	node->ack_sq_wr->wr_u.rdma_write.remote_to = priv->ack_to;
	node->ack_sq_wr->wr_u.rdma_write.remote_stag = priv->stag;

	MACHSTATE1(3, "                                          node = %d", priv->node);
	MACHSTATE1(3, "                                          stag = %d", priv->stag);
	MACHSTATE1(3, "                                          to = %p", priv->to);
        
        MACHSTATE2(1, "AsynchronousEventHandler() - INFO: tokens from %p, stag = %d",priv->to,priv->stag);

        MACHSTATE(1, "AsynchronousEventHandler() - INFO: Pre-sendBufLock");
        #if CMK_SHARED_VARS_UNAVAILABLE
          while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
        #endif
        CmiLock(node->sendBufLock);

        // Indicate that this connection has been made only if it is the first time the connection was made (don't count re-connects)
        if (node->connectionState == QP_CONN_STATE_PRE_CONNECT)
          (contextBlock->outstandingConnectionCount)--;
        node->connectionState = QP_CONN_STATE_CONNECTED;

        CmiUnlock(node->sendBufLock);
        MACHSTATE(1, "AsynchronousEventHandler() - INFO: Post-sendBufLock");

        MACHSTATE1(1, "AsynchronousEventHandler() - Connected to node %d", node->myNode);
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
      MACHSTATE3(5, "AsynchronousEventHandler() - WARNING: Connection Error \"%s\" - er->resource_indicator = %d (CC_RES_IND_QP: %d)", cc_event_id_to_string(er->event_id), er->resource_indicator, CC_RES_IND_QP);
      CmiPrintf("AsynchronousEventHandler() - WARNING: Connection Error \"%s\" - er->resource_indicator = %d (CC_RES_IND_QP: %d)\n", cc_event_id_to_string(er->event_id), er->resource_indicator, CC_RES_IND_QP);

      // Figure out which QP went down
      node = getNodeFromQPId(er->resource_id.qp_id);
      if (node == NULL) {
        MACHSTATE(5, "AsynchronousEventHandler() - ERROR: Unable to find QP from QP ID... Unable to recover connection");
        break;
      }

      MACHSTATE(1, "AsynchronousEventHandler() - INFO: Pre-sendBufLock");
      #if CMK_SHARED_VARS_UNAVAILABLE
        while (node->sendBufLock != 0) { usleep(1); } // Since CmiLock() is not really a lock, actually wait
      #endif
      CmiLock(node->sendBufLock);

      // Indicate that the connection was lost or will be lost in the very near future (depending on the er->event_id)
      node->connectionState = QP_CONN_STATE_CONNECTION_LOST;

      CmiUnlock(node->sendBufLock);
      MACHSTATE(1, "AsynchronousEventHandler() - INFO: Post-sendBufLock");

      MACHSTATE1(1, "AsynchronousEventHandler() -        Connection ERROR Occured - node %d", node->myNode);
      displayQueueQuery(node->qp, &(node->qp_attrs));

      // Attempt to bring the connection back to life
      reestablishQPConnection(node);

      break;

    // er->event_id == ???
    default:
      MACHSTATE1(5, "AsynchronousEventHandler() - WARNING - Unknown/Unexpected Asynchronous Event: er->event_id = %d", er->event_id);
      break;

  } // end switch (er->event_id)

  AMMASSO_STATS_END(AsynchronousEventHandler)
}

void CheckRecvBufForMessage(OtherNode node) {

  int needAck;
  unsigned int len;
  AmmassoBuffer *current;

  MACHSTATE1(2, "CheckRecvBufForMessage() - INFO: Called... (node->recv_buf = %p)...", node->recv_buf);

  // Process all messages, identified by a length not zero
  while ((len = node->recv_buf->tail.length) != 0) {

    MACHSTATE1(2, "                                           (len = %d)...", len);

    // Start by zero-ing out the length of the message so it isn't picked up again
    node->recv_buf->tail.length = 0;
    (*node->remoteAck) ++;
    node->messagesNotYetAcknowledged ++;

    current = node->recv_buf;
    // Move the recv_buf back at the end of the receiving queue. This works also
    // if the queue if formed by a single element
    node->last_recv_buf->next = node->recv_buf;
    node->last_recv_buf = node->recv_buf;
    node->recv_buf = node->recv_buf->next;
    node->last_recv_buf->next = NULL;

    // Process the messsage, NOTE that the message start is aligned to 8 bytes!
    needAck = ProcessMessage(&(current->buf[AMMASSO_BUFSIZE - ALIGN8(len)]), len, &current->tail, node);

    // If an ACK is needed in response to this message, send one
    if (needAck) sendAck(node);
  }

}

/*
void CompletionEventHandler(cc_rnic_handle_t rnic, cc_cq_handle_t cq, void *cb) {
  CompletionEventHandlerWithAckFlag(rnic, cq, cb, 0);
}

void CompletionEventHandlerWithAckFlag(cc_rnic_handle_t rnic, cc_cq_handle_t cq, void *cb, int breakOnAck) {

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
          //for (j = 0; j < DGRAM_HEADER_SIZE; j++) {
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

        // Check to see if the function should return (stop polling for messages)
        if (breakOnAck && (!needAck))   // NOTE: Should only not need an ACK if ACK arrived or error
          return;

        break;

      // default
      default:
	MACHSTATE1(3, "CompletionEventHandler() - Ammasso - WARNING - Unknown WC.wr_type: %d", wc.wr_type);
	break;

    } // end switch (wc.wr_type)
  } // end while (1)

  AMMASSO_STATS_END(CompletionEventHandler)
}
*/

// NOTE: DMK: The code here follows from open_tcp_sockets() in machine-tcp.c.
void CmiAmmassoOpenQueuePairs() {

  char buf[128];
  int i, myNode, numNodes, keepWaiting;
  int buffersPerNode;
  cc_qp_create_attrs_t qpCreateAttrs;
  cc_status_t rtn;
  cc_inet_addr_t address;
  cc_inet_port_t port;
  AmmassoBuffer *sendBuffer;
  AmmassoBuffer *bufferScanner;
  AmmassoToken *newTokens, *tokenScanner;
  cc_data_addr_t *newSgls;
  cc_stag_index_t newStagIndex;
  ammasso_ack_t *ack_location;


  MACHSTATE1(2, "CmiAmmassoOpenQueuePairs() - INFO: Called... (Cmi_charmrun_pid = %d)", Cmi_charmrun_pid);

  // Check for stand-alone mode... no connections needed
  if (Cmi_charmrun_pid == 0) return;

  if (nodes == NULL) {
    MACHSTATE(5, "CmiAmmassoOpenQueuePairs() - WARNING: nodes = NULL");
    return;
  }
  MACHSTATE1(1, "CmiAmmassoOpenQueuePairs() - INFO: nodes = %p (remove this line)", nodes);

  // DMK : FIXME : At this point, CmiMyNode() seems to be returning 0 on any node while _Cmi_mynode is
  // !!!!!!!!!!!   returning the correct value.  However, _Cmi_mynode and _Cmi_numnodes may not work with
  // !!!!!!!!!!!   SMP.  Resolve this issue and fix this code.  For now, using _Cmi_mynode and _Cmi_numnodes.
  //myNode = CmiMyNode();
  //numNodes = CmiNumNodes();
  contextBlock->myNode = myNode = _Cmi_mynode;
  contextBlock->numNodes = numNodes = _Cmi_numnodes;
  contextBlock->outstandingConnectionCount = contextBlock->numNodes - 1;  // No connection with self
  contextBlock->nodeReadyCount = contextBlock->numNodes - 1;              // No ready packet from self
  contextBlock->conditionRegistered = 0;

  MACHSTATE2(1, "CmiAmmassoOpenQueuePairs() - INFO: myNode = %d, numNodes = %d", myNode, numNodes);

  CmiAssert(sizeof(AmmassoBuffer) == (sizeof(AmmassoBuffer)&(~63)));

  // Try to allocate the memory for the receiving buffers
  contextBlock->freeRecvBuffers = (AmmassoBuffer*) CmiAlloc(AMMASSO_INITIAL_BUFFERS*sizeof(AmmassoBuffer) + (contextBlock->numNodes-1)*sizeof(ammasso_ack_t));

  ack_location = (ammasso_ack_t*)&(contextBlock->freeRecvBuffers[AMMASSO_INITIAL_BUFFERS]);
  if (contextBlock->freeRecvBuffers == NULL) {

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE(5, "CmiAmmassoOpenQueuePairs() - ERROR: Unable to allocate memory for RECV buffers");
    sprintf(buf, "CmiAmmassoOpenQueuePairs() - ERROR: Unable to allocate memory for RECV buffers");
    CmiAbort(buf);
  }

  contextBlock->pinnedMemory = AMMASSO_INITIAL_BUFFERS*sizeof(AmmassoBuffer) + (contextBlock->numNodes-1)*sizeof(ammasso_ack_t);
  CC_CHECK(cc_nsmr_register_virt,(contextBlock->rnic,
				  CC_ADDR_TYPE_VA_BASED,
				  (cc_byte_t*)contextBlock->freeRecvBuffers,
				  AMMASSO_INITIAL_BUFFERS*sizeof(AmmassoBuffer) + (contextBlock->numNodes-1)*sizeof(ammasso_ack_t),
				  contextBlock->pd_id,
				  0, 0,
				  CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE | CC_ACF_REMOTE_WRITE,
				  &newStagIndex)
	   );

  for (i=0; i<AMMASSO_INITIAL_BUFFERS; ++i) {
    contextBlock->freeRecvBuffers[i].tail.length = 0;
    contextBlock->freeRecvBuffers[i].next = &(contextBlock->freeRecvBuffers[i+1]);
    contextBlock->freeRecvBuffers[i].stag = newStagIndex;
  }
  contextBlock->freeRecvBuffers[AMMASSO_INITIAL_BUFFERS-1].next = NULL;
  contextBlock->last_freeRecvBuffers = &contextBlock->freeRecvBuffers[AMMASSO_INITIAL_BUFFERS-1];

  buffersPerNode = AMMASSO_INITIAL_BUFFERS / (contextBlock->numNodes-1);

  // distribute all the buffers allocated to the different processors, together
  // the the buffer where to receive the directly sent ACK
  bufferScanner = contextBlock->freeRecvBuffers;
  contextBlock->freeRecvBuffers = contextBlock->freeRecvBuffers[(contextBlock->numNodes-1)*buffersPerNode-1].next;
  contextBlock->num_freeRecvBuffers = AMMASSO_INITIAL_BUFFERS - (contextBlock->numNodes-1)*buffersPerNode;
  for (i=0; i<contextBlock->numNodes; ++i) {
    if (i == contextBlock->myNode) continue;
    nodes[i].num_recv_buf = buffersPerNode;
    nodes[i].recv_buf = bufferScanner;
    bufferScanner[buffersPerNode-1].next = NULL;
    nodes[i].last_recv_buf = &(bufferScanner[buffersPerNode-1]);
    nodes[i].pending = NULL;
    bufferScanner += buffersPerNode;  // move forward of buffersPerNode buffers (of size sizeof(AmmassoBuffer))
    nodes[i].directAck = ack_location;
    ack_location++;
  }

  // Try to allocate the memory for the sending buffers, together with the
  // buffers from where the direct ACK will be sent
  sendBuffer = (AmmassoBuffer*) CmiAlloc(AMMASSO_INITIAL_BUFFERS*sizeof(AmmassoBuffer) + (contextBlock->numNodes-1)*sizeof(ammasso_ack_t));

  if (sendBuffer == NULL) {

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE(5, "CmiAmmassoOpenQueuePairs() - ERROR: Unable to allocate memory for SEND buffers");
    sprintf(buf, "CmiAmmassoOpenQueuePairs() - ERROR: Unable to allocate memory for SEND buffers");
    CmiAbort(buf);
  }

  contextBlock->pinnedMemory += AMMASSO_INITIAL_BUFFERS*sizeof(AmmassoBuffer) + (contextBlock->numNodes-1)*sizeof(ammasso_ack_t);
  CC_CHECK(cc_nsmr_register_virt,(contextBlock->rnic,
				  CC_ADDR_TYPE_VA_BASED,
				  (cc_byte_t*)sendBuffer,
				  AMMASSO_INITIAL_BUFFERS*sizeof(AmmassoBuffer) + (contextBlock->numNodes-1)*sizeof(ammasso_ack_t),
				  contextBlock->pd_id,
				  0, 0,
				  CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE,
				  &newStagIndex)
	   );

  contextBlock->freeTokens = NULL;
  contextBlock->num_freeTokens = 0;

  // Allocate the send tokens, together with the tokens for the ACK buffers
  newTokens = (AmmassoToken*) CmiAlloc((AMMASSO_INITIAL_BUFFERS+contextBlock->numNodes-1)*ALIGN8(sizeof(AmmassoToken)));

  if (newTokens == NULL) {

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE(5, "CmiAmmassoOpenQueuePairs() - ERROR: Unable to allocate memory for SEND buffers");
    sprintf(buf, "CmiAmmassoOpenQueuePairs() - ERROR: Unable to allocate memory for SEND buffers");
    CmiAbort(buf);
  }

  newSgls = (cc_data_addr_t*) CmiAlloc((AMMASSO_INITIAL_BUFFERS+contextBlock->numNodes-1)*ALIGN8(sizeof(cc_data_addr_t)));

  if (newSgls == NULL) {

    // Attempt to close the RNIC
    cc_rnic_close(contextBlock->rnic);

    // Let the user know what happened and bail
    MACHSTATE(5, "CmiAmmassoOpenQueuePairs() - ERROR: Unable to allocate memory for SEND buffers");
    sprintf(buf, "CmiAmmassoOpenQueuePairs() - ERROR: Unable to allocate memory for SEND buffers");
    CmiAbort(buf);
  }

  contextBlock->num_freeTokens = 0;
  contextBlock->last_freeTokens = NULL;
  contextBlock->freeTokens = NULL;
  tokenScanner = newTokens;
  for (i=0; i<AMMASSO_INITIAL_BUFFERS; ++i) {
    newSgls->stag = newStagIndex;
    newSgls->length = AMMASSO_BUFSIZE + sizeof(Tailer);
    newSgls->to = (unsigned long)&(sendBuffer[i]);
    tokenScanner->wr.wr_id = (unsigned long)tokenScanner;
    tokenScanner->wr.wr_type = CC_WR_TYPE_RDMA_WRITE;
    tokenScanner->wr.wr_u.rdma_write.local_sgl.sge_count = 1;
    tokenScanner->wr.wr_u.rdma_write.local_sgl.sge_list = newSgls;
    tokenScanner->wr.signaled = 1;
    tokenScanner->localBuf = (AmmassoBuffer*)&(sendBuffer[i]);
    LIST_ENQUEUE(contextBlock->,freeTokens,tokenScanner);
    newSgls = (cc_data_addr_t*)(((char*)newSgls)+ALIGN8(sizeof(cc_data_addr_t)));
    tokenScanner = (AmmassoToken*)(((char*)tokenScanner)+ALIGN8(sizeof(AmmassoToken)));
  }

  /* At this point, newSgls, tokenScanner, and ack_location point to the first
     element to be used for the ACK buffers */

  // Setup the ack_sq_wr for all nodes
  for (i=0; i<contextBlock->numNodes; ++i) {
    if (i == contextBlock->myNode) continue;
    newSgls->stag = newStagIndex;
    newSgls->length = sizeof(ammasso_ack_t);
    newSgls->to = (unsigned long)ack_location;
    nodes[i].remoteAck = ack_location;
    tokenScanner->wr.wr_id = (unsigned long)tokenScanner;
    tokenScanner->wr.wr_type = CC_WR_TYPE_RDMA_WRITE;
    tokenScanner->wr.wr_u.rdma_write.local_sgl.sge_count = 1;
    tokenScanner->wr.wr_u.rdma_write.local_sgl.sge_list = newSgls;
    tokenScanner->wr.signaled = 1;
    nodes[i].ack_sq_wr = &tokenScanner->wr;
    newSgls = (cc_data_addr_t*)(((char*)newSgls)+ALIGN8(sizeof(cc_data_addr_t)));
    tokenScanner = (AmmassoToken*)(((char*)tokenScanner)+ALIGN8(sizeof(AmmassoToken)));
    ack_location++;
  }

  // Loop through all of the PEs
  //   Begin setting up the Queue Pairs (common code for both "server" and "client")
  //   For nodes with a lower PE, set up a "server" connection (accept)
  //   For nodes with a higher PE, connect as "client" (connect)

  // Loop through all the nodes in the setup
  for (i = 0; i < numNodes; i++) {

    // Setup any members of this nodes OtherNode structure that need setting up
    nodes[i].myNode = i;
    nodes[i].connectionState = QP_CONN_STATE_PRE_CONNECT;
    nodes[i].messagesNotYetAcknowledged = 0;
    nodes[i].usedTokens = NULL;
    nodes[i].last_usedTokens = NULL;
    nodes[i].num_usedTokens = 0;
    nodes[i].localAck = 0;
    nodes[i].sendBufLock = CmiCreateLock();
    nodes[i].send_next_lock = CmiCreateLock();
    nodes[i].recv_expect_lock = CmiCreateLock();
    nodes[i].max_used_tokens = 0;
    //nodes[i].send_UseIndex = 0;
    //nodes[i].send_InUseCounter = 0;
    //nodes[i].recv_UseIndex = 0;

    // If you walk around talking to yourself people will look at you all funny-like.  Try not to do that.
    if (i == myNode) continue;

    *nodes[i].remoteAck = 0;

    // Establish the Connection
    establishQPConnection(nodes + i, 0); // Don't reuse the QP (there isn't one yet) 

  } // end for (i < numNodes)


  // Need to block here until all the connections for this node are made
  MACHSTATE(2, "CmiAmmassoOpenQueuePairs() - INFO: Waiting for all connections to be established...");
  while (contextBlock->outstandingConnectionCount > 0) {

    usleep(1000);

    for (i = 0; i < contextBlock->numNodes; i++) {
      if (i == contextBlock->myNode) continue;
      //CompletionEventHandler(contextBlock->rnic, nodes[i].recv_cq, &(nodes[i]));
      //CompletionEventHandler(contextBlock->rnic, nodes[i].send_cq, &(nodes[i]));
      CheckRecvBufForMessage(&(nodes[i]));
    }
  }
  MACHSTATE(1, "CmiAmmassoOpenQueuePairs() - INFO: All Connections have been established... Continuing");

  // Pause a little so both ends of the connection have time to receive and process the asynchronous events
  //usleep(800000); // 800ms
  sleep(1);

  MACHSTATE(1, "CmiAmmassoOpenQueuePairs() - INFO: Sending ready to all neighboors...");

  // Send all the ready packets
  for (i = 0; i < numNodes; i++) {
    int tmp;
    char buf[24];

    if (i == myNode) continue;  // Skip self

    MACHSTATE1(1, "CmiAmmassoOpenQueuePairs() - INFO: Sending READY to node %d", i);

    // Send a READY control message to the node, give a non-null length
    sendDataOnQP(buf, 1, &(nodes[i]), AMMASSO_READY);

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

  MACHSTATE(1, "CmiAmmassoOpenQueuePairs() - INFO: All ready packets sent to neighboors...");

  // Need to block here until all of the ready packets have been received
  // NOTE : Because this is a fully connection graph of connections between the nodes, this will block all the nodes
  //        until all the nodes are ready (and all the PEs since there is a node barrier in the run pe function that
  //        all the threads execute... the thread executing this is one of those so it has to reach that node barrier
  //        before any of the other can start doing much of anything).
  MACHSTATE(2, "CmiAmmassoOpenQueuePairs() - INFO: Waiting for all neighboors to be ready...");
  while (contextBlock->nodeReadyCount > 0) {
    usleep(10000);  // Sleep 10ms
    
    for (i = 0; i < contextBlock->numNodes; i++) {
      if (i == contextBlock->myNode) continue;
      //CompletionEventHandler(contextBlock->rnic, nodes[i].recv_cq, &(nodes[i]));
      //CompletionEventHandler(contextBlock->rnic, nodes[i].send_cq, &(nodes[i]));
      CheckRecvBufForMessage(&(nodes[i]));
    }
  }
  MACHSTATE(1, "CmiAmmassoOpenQueuePairs() - INFO: All neighboors ready...");

  MACHSTATE(2, "CmiAmmassoOpenQueuePairs() - INFO: Finished.");
}



// NOTE: When reestablishing a connection, the QP can be reused so don't recreate a new one (reuseQPFlag = 1).
//       When openning the connection for the first time, there is no QP so create one (reuseQPFlag = 0).
// DMK : TODO : Fix the comment and parameter (I've been playing with what reuseQPFlag actually does and got
//              tired of updating comments)... update the comment when this is finished).
void establishQPConnection(OtherNode node, int reuseQPFlag) {

  cc_qp_create_attrs_t qpCreateAttrs;
  cc_status_t rtn;
  int i;
  cc_uint32_t numWRsPosted;

  MACHSTATE1(2, "establishQPConnection() - INFO: Called for node %d...", node->myNode);

  ///// Shared "Client" and "Server" Code /////

  MACHSTATE(1, "establishQPConnection() - INFO: (PRE-RECV-CQ-CREATE)");

  // Create the Completion Queue, just create a fake one since with rdma writes it is not used
  node->recv_cq_depth = 1;
  CC_CHECK(cc_cq_create,(contextBlock->rnic, &(node->recv_cq_depth), contextBlock->eh_id, node, &(node->recv_cq)));

  MACHSTATE(1, "establishQPConnection() - INFO: (PRE-SEND-CQ-CREATE)");

  // Create the Completion Queue
  //node->send_cq_depth = AMMASSO_NUMMSGBUFS_PER_QP * 4;
  node->send_cq_depth = AMMASSO_BUFFERS_INFLY;
  CC_CHECK(cc_cq_create,(contextBlock->rnic, &(node->send_cq_depth), contextBlock->eh_id, node, &(node->send_cq)));

  MACHSTATE(1, "establishQPConnection() - INFO: (PRE-QP-CREATE)");

  // Create the Queue Pair
  // Set some initial Create Queue Pair Attributes that will be reused for all Queue Pairs Created
  qpCreateAttrs.sq_cq = node->send_cq;   // Set the Send Queue's Completion Queue
  qpCreateAttrs.rq_cq = node->recv_cq;   // Set the Request Queue's Completion Queue
  qpCreateAttrs.sq_depth = node->send_cq_depth;
  qpCreateAttrs.rq_depth = node->recv_cq_depth;
  qpCreateAttrs.srq = 0;
  qpCreateAttrs.rdma_read_enabled = 1;
  qpCreateAttrs.rdma_write_enabled = 1;
  qpCreateAttrs.rdma_read_response_enabled = 1;
  qpCreateAttrs.mw_bind_enabled = 0;
  qpCreateAttrs.zero_stag_enabled = 0;
  qpCreateAttrs.send_sgl_depth = 1;
  qpCreateAttrs.recv_sgl_depth = 1;
  qpCreateAttrs.rdma_write_sgl_depth = 1;
  qpCreateAttrs.ord = 1;
  qpCreateAttrs.ird = 1;
  qpCreateAttrs.pdid = contextBlock->pd_id;      // Set the Protection Domain
  qpCreateAttrs.user_context = node;             // Set the User Context Block that will be passed into function calls

  CC_CHECK(cc_qp_create,(contextBlock->rnic, &qpCreateAttrs, &(node->qp), &(node->qp_id)));

  // Since the QP was just created (or re-created), reset the sequence number and any other variables that need reseting
  //node->send_InUseCounter = 0;
  //node->send_UseIndex = 0;
  node->sendBufLock = 0;
  node->send_next = 0;
  node->send_next_lock = CmiCreateLock();
  node->recv_expect = 0;
  node->recv_expect_lock = CmiCreateLock();
  node->recv_UseIndex = 0;

  if (!reuseQPFlag) {

    MACHSTATE(1, "establishQPConnection() - INFO: (PRE-NSMR-REGISTER-VIRT QP-QUERY-ATTRS)");

    // Attempt to register the qp_attrs member of the OtherNode structure with the RNIC so the Queue Pair's state can be queried
    contextBlock->pinnedMemory += sizeof(cc_qp_query_attrs_t);
    CC_CHECK(cc_nsmr_register_virt,(contextBlock->rnic,
				    CC_ADDR_TYPE_VA_BASED,
				    (cc_byte_t*)(&(node->qp_attrs)),
				    sizeof(cc_qp_query_attrs_t),
				    contextBlock->pd_id,
				    0, 0,
				    CC_ACF_LOCAL_READ | CC_ACF_LOCAL_WRITE,
				    &(node->qp_attrs_stag_index))
	     );
  }

  ///// "Server" Specific /////
  if (node->myNode < contextBlock->myNode) {

    int count = 64;
    char value[64];
    int j;

    MACHSTATE(1, "establishQPConnection() - INFO: Starting \"Server\" Code...");

    // Setup the address
    CC_CHECK(cc_rnic_getconfig,(contextBlock->rnic, CC_GETCONFIG_ADDRS, &count, &value));

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

    MACHSTATE4(1, "establishQPConnection() - Using Address (Hex) 0x%02X 0x%02X 0x%02X 0x%02X", ((node->address >> 24) & 0xFF), ((node->address >> 16) & 0xFF), ((node->address >> 8) & 0xFF), (node->address & 0xFF));
    MACHSTATE4(1, "                                        (Dec) %4d %4d %4d %4d", ((node->address >> 24) & 0xFF), ((node->address >> 16) & 0xFF), ((node->address >> 8) & 0xFF), (node->address & 0xFF));
    MACHSTATE2(1, "                                   Port (Hex) 0x%02X 0x%02X", ((node->port >> 8) & 0xFF), (node->port & 0xFF));

    /* Listen for an incomming connection (NOTE: This call will return
       immediately; when a connection attempt is made by a "client", the
       asynchronous handler will be called.) */
    CC_CHECK(cc_ep_listen_create,(contextBlock->rnic, node->address, &(node->port), 3, contextBlock, &(node->ep)));

    MACHSTATE(1, "establishQPConnection() - Listening...");
  }


  ///// "Client" Specific /////
  if (node->myNode > contextBlock->myNode) {
    int first;
    AmmassoPrivateData priv;

    // A one-time sleep that should give the passive side QPs time to post the listens before the active sides start trying to connect
    if (node->myNode == contextBlock->myNode + 1 || reuseQPFlag)  // Only do once if all the connections are being made for the first time, do this for all
      usleep(400000);  // Sleep 400ms                             // connections if reconnecting so the other RNIC has time to setup the listen

    MACHSTATE(1, "establishQPConnection() - INFO: Starting \"Client\" Code...");

    // Setup the Address
    // DMK : TODO : FIXME : Fix this code so that it handles host-network/big-little endian ordering
    *(((char*)&(node->address)) + 0) =  *(((char*)&(node->addr.sin_addr.s_addr)) + 0);
    *(((char*)&(node->address)) + 1) =  *(((char*)&(node->addr.sin_addr.s_addr)) + 1);
    *(((char*)&(node->address)) + 2) =  *(((char*)&(node->addr.sin_addr.s_addr)) + 2);
    *(((char*)&(node->address)) + 3) = (*(((char*)&(node->addr.sin_addr.s_addr)) + 3)) - 1;

    // Setup the Port
    node->port = htons(AMMASSO_PORT + contextBlock->myNode);

    MACHSTATE4(1, "establishQPConnection() - Using Address (Hex) 0x%02X 0x%02X 0x%02X 0x%02X", ((node->address >> 24) & 0xFF), ((node->address >> 16) & 0xFF), ((node->address >> 8) & 0xFF), (node->address & 0xFF));
    MACHSTATE4(1, "                                        (Dec) %4d %4d %4d %4d", ((node->address >> 24) & 0xFF), ((node->address >> 16) & 0xFF), ((node->address >> 8) & 0xFF), (node->address & 0xFF));
    MACHSTATE2(1, "                                   Port (Hex) 0x%02X 0x%02X", ((node->port >> 8) & 0xFF), (node->port & 0xFF));

    /* Attempt to make a connection to a "server" (NOTE: This call will return
       immediately; when the connection to the "server" is established, the
       asynchronous handler will be called.) */

    // by allocation protocol, the stags will be the same
    priv.node = contextBlock->myNode;
    priv.stag = node->recv_buf->stag;
    priv.to = (cc_uint64_t)node->recv_buf;
    priv.ack_to = (cc_uint64_t)node->directAck;

    CC_CHECK(cc_qp_connect,(contextBlock->rnic, node->qp, node->address, node->port, sizeof(AmmassoPrivateData), (cc_uint8_t*)&priv));

  }
}


// NOTE: This will be called in the event of a connection error
void reestablishQPConnection(OtherNode node) {

  cc_status_t rtn;
  char buf[16];
  cc_qp_modify_attrs_t modAttrs;
  cc_wc_t wc;

  MACHSTATE1(2, "reestablishQPConnection() - INFO: For node %d: Clearing Outstanding WRs...", node->myNode);

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

  MACHSTATE1(1, "reestablishQPConnection() - INFO: For node %d: Waiting for QP to enter ERROR state...", node->myNode);

  do {

    // Query the QP's state
    rtn = cc_qp_query(contextBlock->rnic, node->qp, &(node->qp_attrs));
    if (rtn != CC_OK) {
      MACHSTATE2(5, "AsynchronousEventHandler() - ERROR: Unable to Query Queue Pair (l): %d, \"%s\"", rtn, cc_status_to_string(rtn));
      break;
    }

    // Check to see if the state is ERROR, if so, break from the loop... otherwise, keep waiting, it will be soon
    if (node->qp_attrs.qp_state == CC_QP_STATE_ERROR)
      break;
    else
      usleep(1000); // 1ms

  } while (1);

  MACHSTATE2(1, "reestablishQPConnection() - INFO: Finished waiting node %d: QP state = \"%s\"...", node->myNode, cc_qp_state_to_string(node->qp_attrs.qp_state));
  MACHSTATE1(1, "reestablishQPConnection() - INFO: Attempting to transition QP into IDLE state for node %d", node->myNode);

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
    MACHSTATE2(5, "reestablishQPConnection() - ERROR: Unable to Modify QP State: %d, \"%s\"", rtn, cc_status_to_string(rtn));
  }

  rtn = cc_qp_query(contextBlock->rnic, node->qp, &(node->qp_attrs));
  if (rtn != CC_OK) {
    MACHSTATE2(5, "reestablishQPConnection() - ERROR: Unable to Query Queue Pair (1): %d, \"%s\"", rtn, cc_status_to_string(rtn));
  }
  MACHSTATE2(1, "reestablishQPConnection() - INFO: Transition results for node %d: QP state = \"%s\"...", node->myNode, cc_qp_state_to_string(node->qp_attrs.qp_state));

  closeQPConnection(node, 0);      // Close the connection but do not destroy the QP
  establishQPConnection(node, 1);  // Reopen the connection and reuse the QP that has already been created
}


// NOTE: When reestablishing a connection, the QP can be reused so don't destroy it and create a new one (destroyQPFlag = 0).
//       When closing the connection because the application is terminating, destroy the QP (destroyQPFlat != 0).
// DMK : TODO : Fix the comment and parameter (I've been playing with what destroyQPFlag actually does and got
//              tired of updating comments)... update the comment when this is finished).
void closeQPConnection(OtherNode node, int destroyQPFlag) {

  MACHSTATE(2, "closeQPConnection() - INFO: Called...");

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
    MACHSTATE1(2, "displayQueueQuery() - Called for node %d", node->myNode);
  } else {
    MACHSTATE(2, "displayQueueQuery() - Called for unknown node");
  }

  // Query the Queue for its Attributes
  rtn = cc_qp_query(contextBlock->rnic, qp, attrs);
  if (rtn != CC_OK) {
    // Let the user know what happened
    MACHSTATE2(5, "displayQueueQuery() - ERROR: Unable to query queue: %d, \"%s\"", rtn, cc_status_to_string(rtn));
    return;
  }

  // Output the results of the Query
  // DMK : TODO : For now I'm only putting in the ones that I care about... add more later or as needed
  MACHSTATE2(1, "displayQueueQuery() - qp_state = %d, \"%s\"", attrs->qp_state, cc_qp_state_to_string(attrs->qp_state));
  if (attrs->terminate_message_length > 0) {
    memcpy(buf, attrs->terminate_message, attrs->terminate_message_length);
    buf[attrs->terminate_message_length] = '\0';
    MACHSTATE1(1, "displayQueueQuery() - terminate_message = \"%s\"", buf);
  } else {
    MACHSTATE(1, "displayQueueQuery() - terminate_message = NULL");
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
