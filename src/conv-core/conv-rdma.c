/* Support for Direct Nocopy API (Generic Implementation)
 * Specific implementations are in arch/layer/machine-onesided.{h,c}
 */
#include "converse.h"
#if !CMK_ONESIDED_DIRECT_IMPL
/* Support for generic implementation */

// Function Pointer to Acknowledement handler function for the Direct API
RdmaSingleAckCallerFn ncpyAckHandlerFn;

// This message is received on an Rget recipient (Rget is performed on this PE by some other PE)
typedef struct _getRequestMsg {
  char cmicore[CmiMsgHeaderSizeBytes];
  int srcPe; /* Source processor */
  int tgtPe; /* Target processor */
  int size; /* size of the source buffer */
  char *srcAddr; /* Source Address */
  char *tgtAddr; /* Target Address */
  int ackSize;  /* Number of bytes occupied by the ack */
} getRequestMsg;

// This message is received on an Rget initiator(when this PE performs an Rget) or
// an Rput recipient (Rput is performed on this PE by some other PE)
typedef struct _receiveDataMsg {
  char cmicore[CmiMsgHeaderSizeBytes];
  int pe; /* Source processor */
  int size; /* size of the buffer */
  char *tgtAddr; /* Target Address */
  char *ref; /* Reference Address used for invoking acks*/
} receiveDataMsg;

static int get_request_handler_idx;
static int put_data_handler_idx;

// Invoked when this PE has to send a large array for an Rget
static void getRequestHandler(getRequestMsg *reqMsg){
  void *srcAck = (char *)reqMsg + sizeof(getRequestMsg);
  void *tgtAck = (char *)srcAck + reqMsg->ackSize;
  // Get is implemented internally using a call to Put
  CmiIssueRput(reqMsg->tgtAddr,
               NULL,
               tgtAck,
               reqMsg->ackSize,
               reqMsg->tgtPe,
               reqMsg->srcAddr,
               NULL,
               srcAck,
               reqMsg->ackSize,
               reqMsg->srcPe,
               reqMsg->size);
}

static void putDataHandler(receiveDataMsg *recvMsg) {
  // copy the received messsage into the user's target address
  memcpy(recvMsg->tgtAddr, (char *)recvMsg + sizeof(receiveDataMsg), recvMsg->size);

  // Invoke the target ack
  void *tarAck = (char *)recvMsg + sizeof(receiveDataMsg) + recvMsg->size;
  ncpyAckHandlerFn(tarAck, recvMsg->pe, recvMsg->tgtAddr);
}

void CmiOnesidedDirectInit() {
  get_request_handler_idx = CmiRegisterHandler((CmiHandler)getRequestHandler);
  put_data_handler_idx = CmiRegisterHandler((CmiHandler)putDataHandler);
}

void CmiSetRdmaSrcInfo(void *info, const void *ptr, int size) {}
void CmiSetRdmaTgtInfo(void *info, const void *ptr, int size) {}

void CmiSetRdmaNcpyAck(RdmaSingleAckCallerFn fn) {
  ncpyAckHandlerFn = fn;
}

int CmiHasNativeRdmaSupport() {
  return 0;
}

void CmiIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  int size) {

  // Send a getRequestMsg to other PE requesting it to send the array
  getRequestMsg *getReqMsg = (getRequestMsg *)CmiAlloc(sizeof(getRequestMsg) + srcAckSize + tgtAckSize);
  getReqMsg->srcPe = srcPe;
  getReqMsg->tgtPe = tgtPe;
  getReqMsg->size = size;
  getReqMsg->srcAddr = (char *)srcAddr;
  getReqMsg->tgtAddr = (char *)tgtAddr;

  CmiAssert(srcAckSize == tgtAckSize);
  getReqMsg->ackSize = srcAckSize;

  // copy the source ack into the getReqMsg
  memcpy((char *)getReqMsg + sizeof(getRequestMsg), srcAck, srcAckSize);

  // copy the target ack into the getReqMsg
  memcpy((char *)getReqMsg + sizeof(getRequestMsg) + srcAckSize, tgtAck, tgtAckSize);

  CmiSetHandler(getReqMsg, get_request_handler_idx);
  CmiSyncSendAndFree(srcPe, sizeof(getRequestMsg) + srcAckSize + tgtAckSize, getReqMsg);
}

void CmiIssueRput(
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  int size) {

  // Send a receiveDataMsg to the other PE sending the array
  receiveDataMsg *recvMsg = (receiveDataMsg *)CmiAlloc(sizeof(receiveDataMsg) + size + tgtAckSize);

  // copy the large array into the recvMsg
  memcpy((char *)recvMsg + sizeof(receiveDataMsg), srcAddr, size);

  // Invoke the source ack
  ncpyAckHandlerFn(srcAck, srcPe, srcAddr);

  // copy the target ack into the recvMsg
  memcpy((char *)recvMsg + sizeof(receiveDataMsg) + size, tgtAck, tgtAckSize);

  recvMsg->pe = tgtPe;
  recvMsg->size = size;
  recvMsg->tgtAddr = (char *)tgtAddr;

  CmiSetHandler(recvMsg, put_data_handler_idx);
  CmiSyncSendAndFree(tgtPe, sizeof(receiveDataMsg) + size + tgtAckSize, recvMsg);
}

void CmiReleaseSourceResource(void *info, int pe) {}
void CmiReleaseTargetResource(void *info, int pe) {}
#endif
