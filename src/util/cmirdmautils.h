#ifndef _CKRDMAUTILS_H
#define _CKRDMAUTILS_H

#include "conv-header.h"

// Structure that can be used across layers
typedef struct ncpystruct{

  // Used in the MPI layer
#if CMK_CONVERSE_MPI
  char core[CmiMsgHeaderSizeBytes];
  int tag;
#endif

  const void *srcPtr;
  int srcPe;
  char *srcLayerInfo;
  int srcLayerSize;
  char *srcAck;
  int srcAckSize;
  int srcSize;
  unsigned short int srcMode;
  unsigned short int isSrcRegistered;
  const void *srcRef;

  const void *destPtr;
  int destPe;
  char *destLayerInfo;
  int destLayerSize;
  char *destAck;
  int destAckSize;
  int destSize;
  unsigned short int destMode;
  unsigned short int isDestRegistered;
  const void *destRef;

  // Variables used for ack handling
  int ackMode; // 0 for call both src and dest acks
               // 1 for call just src ack
               // 2 for call just dest ack
  int freeMe; // 1 for free, 0 for do not free

  int ncpyOpInfoSize;

}NcpyOperationInfo;

int getNcpyOpInfoTotalSize(
  int srcLayerSize,
  int srcAckSize,
  int destLayerSize,
  int destAckSize);

void setNcpyOpInfo(
  const void *srcPtr,
  char *srcLayerInfo,
  int srcLayerSize,
  char *srcAck,
  int srcAckSize,
  int srcSize,
  unsigned short int srcMode,
  unsigned short int isSrcRegistered,
  int srcPe,
  const void *srcRef,
  const void *destPtr,
  char *destLayerInfo,
  int destLayerSize,
  char *destAck,
  int destAckSize,
  int destSize,
  unsigned short int destMode,
  unsigned short int isdestRegistered,
  int destPe,
  const void *destRef,
  NcpyOperationInfo *ncpyOpInfo);

void resetNcpyOpInfoPointers(NcpyOperationInfo *ncpyOpInfo);
#endif
