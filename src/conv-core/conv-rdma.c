/* Support for Direct Nocopy API (Generic Implementation)
 * Specific implementations are in arch/layer/machine-onesided.{h,c}
 */
#include "converse.h"

#if CMK_USE_CMA
extern int cma_works;
#endif

// Methods required to keep the Nocopy Direct API functional on non-LRTS layers
#if !CMK_USE_LRTS
void CmiSetRdmaCommonInfo(void *info, const void *ptr, int size) {
}

int CmiGetRdmaCommonInfoSize() {
  return 0;
}
#endif

#if !CMK_ONESIDED_DIRECT_IMPL
/* Support for generic implementation */

// Function Pointer to Acknowledement handler function for the Direct API
RdmaSingleAckCallerFn ncpyAckHandlerFn;

// An Rget initiator PE sends this message to the target PE that will be the source of the data
typedef struct _getRequestMsg {
  char cmicore[CmiMsgHeaderSizeBytes];
  int srcPe; /* Source processor */
  int destPe; /* Destination processor */
  int size; /* size of the source buffer */
  char *srcAddr; /* Source Address */
  char *destAddr; /* Destination Address */
  int ackSize;  /* Number of bytes occupied by the ack */
} getRequestMsg;

// This is a header for RDMA payloads transferred as normal converse messages,
// delivered to the PE holding the destination buffer (Rget initiator or Rput target)
typedef struct _rdmaPayloadMsg {
  char cmicore[CmiMsgHeaderSizeBytes];
  int pe; /* Source processor */
  int size; /* size of the buffer */
  char *destAddr; /* Destination Address */
  char *ref; /* Reference Address used for invoking acks*/
} rdmaPayloadMsg;

static int get_request_handler_idx;
static int put_data_handler_idx;

// Invoked when this PE has to send a large array for an Rget
static void getRequestHandler(getRequestMsg *reqMsg){
  void *srcAck = (char *)reqMsg + sizeof(getRequestMsg);
  void *destAck = (char *)srcAck + reqMsg->ackSize;
  // Get is implemented internally using a call to Put
  CmiIssueRput(reqMsg->destAddr,
               NULL,
               destAck,
               reqMsg->ackSize,
               reqMsg->destPe,
               NULL,
               reqMsg->srcAddr,
               NULL,
               srcAck,
               reqMsg->ackSize,
               reqMsg->srcPe,
               NULL,
               reqMsg->size);
}

// Invoked when this PE receives a large array as the target of an Rput or the initiator of an Rget
static void putDataHandler(rdmaPayloadMsg *recvMsg) {
  // copy the received messsage into the user's destination address
  memcpy(recvMsg->destAddr, (char *)recvMsg + sizeof(rdmaPayloadMsg), recvMsg->size);

  // Invoke the destination ack
  void *destAck = (char *)recvMsg + sizeof(rdmaPayloadMsg) + recvMsg->size;
  ncpyAckHandlerFn(destAck, recvMsg->pe, recvMsg->destAddr);
}

// Rget/Rput operations are implemented as normal converse messages
// This method is invoked during converse initialization to initialize these message handlers
void CmiOnesidedDirectInit(void) {
  get_request_handler_idx = CmiRegisterHandler((CmiHandler)getRequestHandler);
  put_data_handler_idx = CmiRegisterHandler((CmiHandler)putDataHandler);
}

void CmiSetRdmaNcpyAck(RdmaSingleAckCallerFn fn) {
  ncpyAckHandlerFn = fn;
}

void CmiIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  unsigned short int *srcMode,
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  unsigned short int *destMode,
  int size) {

#if CMK_USE_CMA
  // check if remote PE is on the same physical node
  if(cma_works && CmiPeOnSamePhysicalNode(srcPe, destPe)) {

    CmiIssueRgetUsingCMA(srcAddr, srcInfo, srcPe,
                         destAddr, destInfo, destPe,
                         size);

    // directy invoke the acks
    ncpyAckHandlerFn(srcAck, srcPe, srcAddr);
    ncpyAckHandlerFn(destAck, destPe, destAddr);
    return;
  }
#endif

  // Send a getRequestMsg to other PE requesting it to send the array
  getRequestMsg *getReqMsg = (getRequestMsg *)CmiAlloc(sizeof(getRequestMsg) + srcAckSize + destAckSize);
  getReqMsg->srcPe = srcPe;
  getReqMsg->destPe = destPe;
  getReqMsg->size = size;
  getReqMsg->srcAddr = (char *)srcAddr;
  getReqMsg->destAddr = (char *)destAddr;

  CmiAssert(srcAckSize == destAckSize);
  getReqMsg->ackSize = srcAckSize;

  // copy the source ack into the getReqMsg
  memcpy((char *)getReqMsg + sizeof(getRequestMsg), srcAck, srcAckSize);

  // copy the destination ack into the getReqMsg
  memcpy((char *)getReqMsg + sizeof(getRequestMsg) + srcAckSize, destAck, destAckSize);

  CmiSetHandler(getReqMsg, get_request_handler_idx);
  CmiSyncSendAndFree(srcPe, sizeof(getRequestMsg) + srcAckSize + destAckSize, getReqMsg);
}

void CmiIssueRput(
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  unsigned short int *destMode,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  unsigned short int *srcMode,
  int size) {

#if CMK_USE_CMA
  // check if remote PE is on the same physical node
  if(cma_works && CmiPeOnSamePhysicalNode(srcPe, destPe)) {

    CmiIssueRputUsingCMA(destAddr, destInfo, destPe,
                         srcAddr, srcInfo, srcPe,
                         size);

    // directy invoke the acks
    ncpyAckHandlerFn(srcAck, srcPe, srcAddr);
    ncpyAckHandlerFn(destAck, destPe, destAddr);
    return;
  }
#endif

  // Send a rdmaPayloadMsg to the other PE sending the array
  rdmaPayloadMsg *recvMsg = (rdmaPayloadMsg *)CmiAlloc(sizeof(rdmaPayloadMsg) + size + destAckSize);

  // copy the large array into the recvMsg
  memcpy((char *)recvMsg + sizeof(rdmaPayloadMsg), srcAddr, size);

  // copy the destination ack into the recvMsg
  memcpy((char *)recvMsg + sizeof(rdmaPayloadMsg) + size, destAck, destAckSize);

  // Invoke the source ack
  ncpyAckHandlerFn(srcAck, srcPe, srcAddr);

  recvMsg->pe = destPe;
  recvMsg->size = size;
  recvMsg->destAddr = (char *)destAddr;

  CmiSetHandler(recvMsg, put_data_handler_idx);
  CmiSyncSendAndFree(destPe, sizeof(rdmaPayloadMsg) + size + destAckSize, recvMsg);
}

void CmiSetRdmaSrcInfo(void *info, const void *ptr, int size, unsigned short int mode) {}
void CmiSetRdmaDestInfo(void *info, const void *ptr, int size, unsigned short int mode) {}

void CmiReleaseSourceResource(const void *ptr, void *info, int pe, unsigned short int mode) {}
void CmiReleaseDestinationResource(const void *ptr, void *info, int pe, unsigned short int mode) {}
#endif
