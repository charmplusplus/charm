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

  unsigned char opMode; // CMK_DIRECT_API for p2p direct api
                        // CMK_DIRECT_API_REVERSE for p2p direct api with inverse operation
                        // CMK_EM_API for p2p entry method api
                        // CMK_EM_API_REVERSE for p2p entry method api with inverse operation

  // Variables used for ack handling
  unsigned char ackMode; // CMK_SRC_DEST_ACK for call both src and dest acks
                         // CMK_SRC_ACK for call just src ack
                         // CMK_DEST_ACK for call just dest ack

  unsigned char freeMe;  // CMK_FREE_NCPYOPINFO in order to free NcpyOperationInfo
                         // CMK_DONT_FREE_NCPYOPINFO in order to not free NcpyOperationInfo

  void *refPtr;

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

void setReverseModeForNcpyOpInfo(NcpyOperationInfo *ncpyOpInfo);
#endif
