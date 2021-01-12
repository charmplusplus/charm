#include "cmirdmautils.h"
#include "converse.h" // for CmiAbort usage to avoid undeclared warning

#include <stdio.h>
#include <string.h>
#include <limits>

int getNcpyOpInfoTotalSize(
  int srcLayerSize,
  int srcAckSize,
  int destLayerSize,
  int destAckSize) {
  return sizeof(NcpyOperationInfo) + srcLayerSize + destLayerSize + srcAckSize + destAckSize;
}

void setNcpyOpInfo(
    const void *srcPtr,
    char *srcLayerInfo,
    int srcLayerSize,
    char *srcAck,
    int srcAckSize,
    size_t srcSize,
    unsigned short int srcRegMode,
    unsigned short int srcDeregMode,
    unsigned short int isSrcRegistered,
    int srcPe,
    const void *srcRef,
    const void *destPtr,
    char *destLayerInfo,
    int destLayerSize,
    char *destAck,
    int destAckSize,
    size_t destSize,
    unsigned short int destRegMode,
    unsigned short int destDeregMode,
    unsigned short int isDestRegistered,
    int destPe,
    const void *destRef,
    int rootNode,
    NcpyOperationInfo *ncpyOpInfo) {

  char *base = (char *)ncpyOpInfo + sizeof(NcpyOperationInfo);

  ncpyOpInfo->srcLayerInfo = NULL;
  ncpyOpInfo->destLayerInfo = NULL;
  ncpyOpInfo->srcAck = NULL;
  ncpyOpInfo->destAck = NULL;

  // memcpy srcLayerInfo
  if(srcLayerInfo != NULL && srcLayerSize != 0) {
    memcpy(base, srcLayerInfo, srcLayerSize);
    ncpyOpInfo->srcLayerInfo = base;
    base = base + srcLayerSize;
  }

  // memcpy srcAckInfo
  if(srcAck != NULL && srcAckSize != 0) {
    memcpy(base, srcAck, srcAckSize);
    ncpyOpInfo->srcAck = base;
    base = base + srcAckSize;
  }

  // memcpy destLayerInfo
  if(srcLayerInfo != NULL && destLayerSize != 0) {
    memcpy(base, destLayerInfo, destLayerSize);
    ncpyOpInfo->destLayerInfo = base;
    base = base + destLayerSize;
  }

  // memcpy destAck Info
  if(destAck != NULL && destAckSize != 0) {
    memcpy(base, destAck, destAckSize);
    ncpyOpInfo->destAck = base;
  }

  ncpyOpInfo->srcPtr = srcPtr;
  ncpyOpInfo->srcPe = srcPe;
  ncpyOpInfo->srcRef = srcRef;
  CmiAssert(srcLayerSize <= std::numeric_limits<unsigned short int>::max());
  ncpyOpInfo->srcLayerSize = (short int)srcLayerSize;
  CmiAssert(srcAckSize <= std::numeric_limits<unsigned short int>::max());
  ncpyOpInfo->srcAckSize = (short int)srcAckSize;
  ncpyOpInfo->srcSize = srcSize;
  CmiAssert(srcRegMode <= std::numeric_limits<unsigned char>::max());
  ncpyOpInfo->srcRegMode = (unsigned char)srcRegMode;
  CmiAssert(srcDeregMode <= std::numeric_limits<unsigned char>::max());
  ncpyOpInfo->srcDeregMode = (unsigned char)srcDeregMode;
  CmiAssert(isSrcRegistered <= std::numeric_limits<unsigned char>::max());
  ncpyOpInfo->isSrcRegistered = (unsigned char)isSrcRegistered;

  ncpyOpInfo->destPtr = destPtr;
  ncpyOpInfo->destPe = destPe;
  ncpyOpInfo->destRef = destRef;
  CmiAssert(destLayerSize <= std::numeric_limits<unsigned short int>::max());
  ncpyOpInfo->destLayerSize = (unsigned short int)destLayerSize;
  CmiAssert(destAckSize <= std::numeric_limits<unsigned short int>::max());
  ncpyOpInfo->destAckSize = (unsigned short int)destAckSize;
  ncpyOpInfo->destSize = destSize;
  CmiAssert(destRegMode <= std::numeric_limits<unsigned char>::max());
  ncpyOpInfo->destRegMode = (unsigned char)destRegMode;
  CmiAssert(destDeregMode <= std::numeric_limits<unsigned char>::max());
  ncpyOpInfo->destDeregMode = (unsigned char)destDeregMode;
  CmiAssert(isDestRegistered <= std::numeric_limits<unsigned char>::max());
  ncpyOpInfo->isDestRegistered = (unsigned char)isDestRegistered;

  ncpyOpInfo->opMode  = CMK_DIRECT_API; // default operation mode is CMK_DIRECT_API
  ncpyOpInfo->ackMode = CMK_SRC_DEST_ACK; // default ack mode is CMK_SRC_DEST_ACK
  ncpyOpInfo->freeMe  = CMK_FREE_NCPYOPINFO; // default ack mode is CMK_FREE_NCPYOPINFO

  ncpyOpInfo->rootNode = rootNode;

  ncpyOpInfo->ncpyOpInfoSize = (unsigned short int)(sizeof(NcpyOperationInfo) + srcLayerSize + destLayerSize + srcAckSize + destAckSize);
}


void resetNcpyOpInfoPointers(NcpyOperationInfo *ncpyOpInfo) {

  char *base = (char *)ncpyOpInfo + sizeof(NcpyOperationInfo);

  if(ncpyOpInfo->srcLayerInfo) {
    ncpyOpInfo->srcLayerInfo = base;
    base = base + ncpyOpInfo->srcLayerSize;
  }

  if(ncpyOpInfo->srcAck) {
    ncpyOpInfo->srcAck = base;
    base = base + ncpyOpInfo->srcAckSize;
  }

  if(ncpyOpInfo->destLayerInfo) {
    ncpyOpInfo->destLayerInfo = base;
    base = base + ncpyOpInfo->destLayerSize;
  }

  if(ncpyOpInfo->destAck) {
    ncpyOpInfo->destAck = base;
  }

}

void setReverseModeForNcpyOpInfo(NcpyOperationInfo *ncpyOpInfo) {
  switch(ncpyOpInfo->opMode) {
    case CMK_EM_API          : ncpyOpInfo->opMode = CMK_EM_API_REVERSE;
                               break;
    case CMK_DIRECT_API      : // Do nothing
                               break;
    case CMK_BCAST_EM_API    : ncpyOpInfo->opMode = CMK_BCAST_EM_API_REVERSE;
                               break;
    default                  : CmiAbort("Unknown opcode");
                               break;
  }
}
