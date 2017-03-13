#ifndef NETLRTS_MACHINE_ONESIDED_H
#define NETLRTS_MACHINE_ONESIDED_H

/*
 * None of these should ever be called, since this header is only included
 * if CMK_ONESIDED_IMPL is 1, and for netlrts that is currently only true for
 * multicore builds, which always do a direct memcpy of the source buffer
 * to a message allocated by the receiver.
 *
 * TODO: RDMA support for non-multicore netlrts builds.
 */

void LrtsSetRdmaInfo(void *dest, int destPE, int numOps) {
  CmiAbort("Should never reach here!");
}

void LrtsSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE) {
  CmiAbort("Should never reach here!");
}

int LrtsGetRdmaOpInfoSize() {
  CmiAbort("Should never reach here!");
}

int LrtsGetRdmaGenInfoSize() {
  CmiAbort("Should never reach here!");
}

int LrtsGetRdmaInfoSize(int numOps) {
  CmiAbort("Should never reach here!");
}

void LrtsSetRdmaRecvInfo(void *dest, int numOps, void *charmMsg, void *rdmaInfo, int msgSize) {
  CmiAbort("Should never reach here!");
}

void LrtsSetRdmaRecvOpInfo(void *dest, void *buffer, void *src_ref, int size, int opIndex, void *rdmaRecv) {
  CmiAbort("Should never reach here!");
}

int LrtsGetRdmaOpRecvInfoSize() {
  CmiAbort("Should never reach here!");
}

int LrtsGetRdmaGenRecvInfoSize() {
  CmiAbort("Should never reach here!");
}

int LrtsGetRdmaRecvInfoSize(int numOps) {
  CmiAbort("Should never reach here!");
}

void LrtsIssueRgets(void *recv, int pe) {
  CmiAbort("Should never reach here!");
}

#endif /* NETLRTS_MACHINE_ONESIDED_H */
