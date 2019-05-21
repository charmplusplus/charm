/* Support for Direct Nocopy API (Generic Implementation)
 * Specific implementations are in arch/layer/machine-onesided.{h,c}
 */
#include "converse.h"
#include <algorithm>

// Methods required to keep the Nocopy Direct API functional on non-LRTS layers
#if !CMK_USE_LRTS
void CmiSetRdmaCommonInfo(void *info, const void *ptr, int size) {
}

int CmiGetRdmaCommonInfoSize() {
  return 0;
}
#endif

#if !CMK_ONESIDED_IMPL
/* Support for generic implementation */

// Function Pointer to Acknowledement handler function for the Direct API
RdmaAckCallerFn ncpyDirectAckHandlerFn;

// An Rget initiator PE sends this message to the target PE that will be the source of the data
typedef struct _converseRdmaMsg {
  char cmicore[CmiMsgHeaderSizeBytes];
} ConverseRdmaMsg;

static int get_request_handler_idx;
static int put_data_handler_idx;

// Invoked when this PE has to send a large array for an Rget
static void getRequestHandler(ConverseRdmaMsg *getReqMsg){

  NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)((char *)(getReqMsg) + sizeof(ConverseRdmaMsg));

  resetNcpyOpInfoPointers(ncpyOpInfo);

  ncpyOpInfo->freeMe = CMK_DONT_FREE_NCPYOPINFO;

  // Get is implemented internally using a call to Put
  CmiIssueRput(ncpyOpInfo);
}

// Invoked when this PE receives a large array as the target of an Rput or the initiator of an Rget
static void putDataHandler(ConverseRdmaMsg *payloadMsg) {

  NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)((char *)payloadMsg + sizeof(ConverseRdmaMsg));

  resetNcpyOpInfoPointers(ncpyOpInfo);

  // copy the received messsage into the user's destination address
  memcpy((char *)ncpyOpInfo->destPtr,
         (char *)payloadMsg + sizeof(ConverseRdmaMsg) + ncpyOpInfo->ncpyOpInfoSize,
         std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize));

  // Invoke the destination ack
  ncpyOpInfo->ackMode = CMK_DEST_ACK; // Only invoke the destination ack
  ncpyOpInfo->freeMe  = CMK_DONT_FREE_NCPYOPINFO;
  ncpyDirectAckHandlerFn(ncpyOpInfo);
}

// Rget/Rput operations are implemented as normal converse messages
// This method is invoked during converse initialization to initialize these message handlers
void CmiOnesidedDirectInit(void) {
  get_request_handler_idx = CmiRegisterHandler((CmiHandler)getRequestHandler);
  put_data_handler_idx = CmiRegisterHandler((CmiHandler)putDataHandler);
}

void CmiSetDirectNcpyAckHandler(RdmaAckCallerFn fn) {
  ncpyDirectAckHandlerFn = fn;
}

void CmiIssueRget(NcpyOperationInfo *ncpyOpInfo) {

  int ncpyOpInfoSize = ncpyOpInfo->ncpyOpInfoSize;

  // Send a ConverseRdmaMsg to other PE requesting it to send the array
  ConverseRdmaMsg *getReqMsg = (ConverseRdmaMsg *)CmiAlloc(sizeof(ConverseRdmaMsg) + ncpyOpInfoSize);

  // copy the additional Info into the getReqMsg
  memcpy((char *)getReqMsg + sizeof(ConverseRdmaMsg),
         (char *)ncpyOpInfo,
         ncpyOpInfoSize);

  CmiSetHandler(getReqMsg, get_request_handler_idx);
  CmiSyncSendAndFree(ncpyOpInfo->srcPe, sizeof(ConverseRdmaMsg) + ncpyOpInfoSize, getReqMsg);

  // free original ncpyOpinfo
  CmiFree(ncpyOpInfo);
}

void CmiIssueRput(NcpyOperationInfo *ncpyOpInfo) {

  int ncpyOpInfoSize = ncpyOpInfo->ncpyOpInfoSize;
  int size = ncpyOpInfo->srcSize;

  // Send a ConverseRdmaMsg to the other PE sending the array
  ConverseRdmaMsg *payloadMsg = (ConverseRdmaMsg *)CmiAlloc(sizeof(ConverseRdmaMsg) + ncpyOpInfoSize + size);

  // copy the ncpyOpInfo into the recvMsg
  memcpy((char *)payloadMsg + sizeof(ConverseRdmaMsg),
         (char *)ncpyOpInfo,
         ncpyOpInfoSize);

  // copy the large array into the recvMsg
  memcpy((char *)payloadMsg + sizeof(ConverseRdmaMsg) + ncpyOpInfoSize,
         ncpyOpInfo->srcPtr,
         size);

  // Invoke the source ack
  ncpyOpInfo->ackMode = CMK_SRC_ACK; // only invoke the source ack

  ncpyDirectAckHandlerFn(ncpyOpInfo);

  CmiSetHandler(payloadMsg, put_data_handler_idx);
  CmiSyncSendAndFree(ncpyOpInfo->destPe,
                     sizeof(ConverseRdmaMsg) + ncpyOpInfoSize + size,
                     payloadMsg);
}

void CmiSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode) {}

void CmiDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode) {}

#else

// Support for sending an ack message for the Entry Method API

RdmaEMAckCallerFn ncpyEMAckHandlerFn; // P2P API ack handler function

RdmaAckCallerFn ncpyEMBcastAckHandlerFn; // BCast API ack handler function

RdmaAckCallerFn ncpyEMBcastPostAckHandlerFn; // BCast API ack handler function

static int invoke_entry_method_ack_handler_idx, ncpy_bcast_ack_handler_idx;
static int ncpy_bcast_post_handler_idx;

// Ack Message is typically used in case of reverse operation (when a reverse put is used instead of a get)
typedef struct _ackEntryMethodMsg{
  char cmicore[CmiMsgHeaderSizeBytes];
  void *ref;
} ackEntryMethodMsg;

// Handler invoked on receiving a ackEntryMethodMsg
// This handler invokes the ncpyEMAckHandler on the receiver side
static void ackEntryMethodHandler(ackEntryMethodMsg *msg) {
  // Invoke the charm handler
  ncpyEMAckHandlerFn(CmiMyPe(), msg->ref);
}

// This handler invokes the ncpyEMBcastAckHandler on the source (root node or intermediate nodes)
static void bcastAckHandler(ackEntryMethodMsg *msg) {
  ncpyEMBcastAckHandlerFn(msg->ref);
}

static void bcastPostAckArrayHandler(ackEntryMethodMsg *msg) {
  ncpyEMBcastPostAckHandlerFn(msg->ref);
}

// Method to create a ackEntryMethodMsg and send it
void CmiInvokeRemoteAckHandler(int pe, void *ref) {
  ackEntryMethodMsg *msg = (ackEntryMethodMsg *)CmiAlloc(sizeof(ackEntryMethodMsg));
  msg->ref = ref;

  CmiSetHandler(msg, invoke_entry_method_ack_handler_idx);
  CmiSyncSendAndFree(pe, sizeof(ackEntryMethodMsg), msg);
}

// Register converse handler for invoking ack on reverse operation
void CmiOnesidedDirectInit(void) {
  invoke_entry_method_ack_handler_idx = CmiRegisterHandler((CmiHandler)ackEntryMethodHandler);
  ncpy_bcast_ack_handler_idx = CmiRegisterHandler((CmiHandler)bcastAckHandler);

  ncpy_bcast_post_handler_idx = CmiRegisterHandler((CmiHandler)bcastPostAckArrayHandler);
}

void CmiSetEMNcpyAckHandler(RdmaEMAckCallerFn fn, RdmaAckCallerFn bcastFn, RdmaAckCallerFn bcastArrayFn) {
  // set the EM Ack caller function
  ncpyEMAckHandlerFn = fn;

  // set the EM Bcast Ack caller function
  ncpyEMBcastAckHandlerFn = bcastFn;

  // set the EM Bcast Post Ack caller function
  ncpyEMBcastPostAckHandlerFn = bcastArrayFn;
}

void CmiInvokeBcastAckHandler(int pe, void *ref) {

  ackEntryMethodMsg *msg = (ackEntryMethodMsg *)CmiAlloc(sizeof(ackEntryMethodMsg));
  msg->ref = ref;

  CmiSetHandler(msg, ncpy_bcast_ack_handler_idx);
  CmiSyncSendAndFree(pe, sizeof(ackEntryMethodMsg), msg);
}

void CmiInvokeBcastPostAckHandler(int pe, void *ref) {
  ackEntryMethodMsg *msg = (ackEntryMethodMsg *)CmiAlloc(sizeof(ackEntryMethodMsg));
  msg->ref = ref;

  CmiSetHandler(msg, ncpy_bcast_post_handler_idx);
  CmiSyncSendAndFree(pe, sizeof(ackEntryMethodMsg), msg);
}
#endif
