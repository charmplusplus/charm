/* Support for Direct Nocopy API (Generic Implementation)
 * Specific implementations are in arch/layer/machine-onesided.{h,c}
 */
#include "converse.h"

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
RdmaAckCallerFn ncpyAckHandlerFn;

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

  ncpyOpInfo->freeMe = 0;

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
         ncpyOpInfo->srcSize);

  // Invoke the destination ack
  ncpyOpInfo->ackMode = 2;
  ncpyOpInfo->freeMe  = 0;
  ncpyAckHandlerFn(ncpyOpInfo);
  //ncpyAckHandlerFn(destAck, recvMsg->pe, recvMsg->destAddr);
}

// Rget/Rput operations are implemented as normal converse messages
// This method is invoked during converse initialization to initialize these message handlers
void CmiOnesidedDirectInit(void) {
  get_request_handler_idx = CmiRegisterHandler((CmiHandler)getRequestHandler);
  put_data_handler_idx = CmiRegisterHandler((CmiHandler)putDataHandler);
}

void CmiSetRdmaNcpyAck(RdmaAckCallerFn fn) {
  ncpyAckHandlerFn = fn;
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
  ncpyOpInfo->ackMode = 1;

  ncpyAckHandlerFn(ncpyOpInfo);

  CmiSetHandler(payloadMsg, put_data_handler_idx);
  CmiSyncSendAndFree(ncpyOpInfo->destPe,
                     sizeof(ConverseRdmaMsg) + ncpyOpInfoSize + size,
                     payloadMsg);
}

void CmiSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode) {}

void CmiDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode) {}
#endif
