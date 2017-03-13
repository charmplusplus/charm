#ifndef _CONV_RDMA_H
#define _CONV_RDMA_H

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
#endif
