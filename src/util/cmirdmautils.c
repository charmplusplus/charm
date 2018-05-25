#include "cmirdmautils.h"

#include <stdio.h>
#include <string.h>

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
    int srcPe,
    const void *srcRef,
    const void *destPtr,
    char *destLayerInfo,
    int destLayerSize,
    char *destAck,
    int destAckSize,
    int destPe,
    const void *destRef,
    int size,
    NcpyOperationInfo *ncpyOpInfo) {

  // memcpy srcLayerInfo
  memcpy((char *)ncpyOpInfo + sizeof(NcpyOperationInfo), srcLayerInfo, srcLayerSize);
  ncpyOpInfo->srcLayerInfo = (char *)ncpyOpInfo + sizeof(NcpyOperationInfo);
  // memcpy srcAckInfo
  memcpy(ncpyOpInfo->srcLayerInfo + srcLayerSize, srcAck, srcAckSize);
  ncpyOpInfo->srcAck = ncpyOpInfo->srcLayerInfo + srcLayerSize;

  // memcpy destLayerInfo
  memcpy(ncpyOpInfo->srcAck + srcAckSize, destLayerInfo, destLayerSize);
  ncpyOpInfo->destLayerInfo = ncpyOpInfo->srcAck + srcAckSize;

  // memcpy destAck Info
  memcpy(ncpyOpInfo->destLayerInfo + destLayerSize, destAck, destAckSize);
  ncpyOpInfo->destAck = ncpyOpInfo->destLayerInfo + destLayerSize;

  ncpyOpInfo->srcPtr = srcPtr;
  ncpyOpInfo->srcPe = srcPe;
  ncpyOpInfo->srcRef = srcRef;
  ncpyOpInfo->srcLayerSize = srcLayerSize;
  ncpyOpInfo->srcAckSize = srcAckSize;
  ncpyOpInfo->origSrcLayerInfoPtr = srcLayerInfo;

  ncpyOpInfo->destPtr = destPtr;
  ncpyOpInfo->destPe = destPe;
  ncpyOpInfo->destRef = destRef;
  ncpyOpInfo->destLayerSize = destLayerSize;
  ncpyOpInfo->destAckSize = destAckSize;
  ncpyOpInfo->origDestLayerInfoPtr = destLayerInfo;

  ncpyOpInfo->ackMode = 0;
  ncpyOpInfo->freeMe  = 1;

  ncpyOpInfo->ncpyOpInfoSize = sizeof(NcpyOperationInfo) + srcLayerSize + destLayerSize + srcAckSize + destAckSize;
  ncpyOpInfo->size = size;
}


void resetNcpyOpInfoPointers(NcpyOperationInfo *ncpyOpInfo) {
  ncpyOpInfo->srcLayerInfo = (char *)ncpyOpInfo + sizeof(NcpyOperationInfo);

  ncpyOpInfo->srcAck = (char *)(ncpyOpInfo->srcLayerInfo) + ncpyOpInfo->srcLayerSize;

  ncpyOpInfo->destLayerInfo = (char *)(ncpyOpInfo->srcAck) + ncpyOpInfo->srcAckSize;

  ncpyOpInfo->destAck = (char *)(ncpyOpInfo->destLayerInfo) + ncpyOpInfo->destLayerSize;
}
