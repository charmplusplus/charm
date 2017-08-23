#ifndef _CONV_RDMA_H
#define _CONV_RDMA_H

typedef void (*RdmaSingleAckCallerFn)(void *cbPtr, int pe, const void *ptr);
typedef void (*RdmaAckCallerFn)(void *token);

void *CmiSetRdmaAck(RdmaAckCallerFn fn, void *token);
void CmiSetRdmaInfo(void *dest, int destPE, int numOps);
void CmiSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE);
int CmiGetRdmaOpInfoSize();
int CmiGetRdmaGenInfoSize();

int CmiGetRdmaInfoSize(int numOps);
void CmiSetRdmaRecvInfo(void *dest, int numOps, void *msg, void *rdmaInfo, int msgSize);
void CmiSetRdmaRecvOpInfo(void *dest, void *buffer, void *src_ref, int size, int opIndex, void *rdmaInfo);
int CmiGetRdmaOpRecvInfoSize();
int CmiGetRdmaGenRecvInfoSize();
int CmiGetRdmaRecvInfoSize(int numOps);

void CmiIssueRgets(void *recv, int pe);

/* Support for Direct API */
void CmiSetRdmaSrcInfo(void *info, const void *ptr, int size);
void CmiSetRdmaTgtInfo(void *info, const void *ptr, int size);
void CmiSetRdmaNcpyAck(RdmaSingleAckCallerFn fn);
int CmiHasNativeRdmaSupport();

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
  int size);

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
  int size);

void CmiReleaseSourceResource(void *info, int pe);
void CmiReleaseTargetResource(void *info, int pe);
#endif
