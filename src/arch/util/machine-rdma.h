#ifndef _MACHINE_RDMA_H_
#define _MACHINE_RDMA_H_

/*Function Pointer to Acknowledgement Handler*/
typedef void (*RdmaAckHandlerFn)(void *token);

/* Support for Nocopy Direct API */
typedef struct _cmi_common_rdma_info {
#if CMK_USE_CMA
  pid_t pid;
#elif defined _MSC_VER
  char empty;
#endif
} CmiCommonRdmaInfo_t;

/* Set the generic converse/LRTS information */
void CmiSetRdmaCommonInfo(void *info, const void *ptr, int size) {
#if CMK_USE_CMA
  CmiCommonRdmaInfo_t *cmmInfo = (CmiCommonRdmaInfo_t *)info;
  cmmInfo->pid = getpid();
#endif
}

int CmiGetRdmaCommonInfoSize() {
#if CMK_USE_CMA
  return sizeof(CmiCommonRdmaInfo_t);
#else
  return 0; // If CMK_USE_CMA is false, sizeof(CmiCommonRdmaInfo_t) is 1 (size of an empty structure in C++)
            // However, 0 is returned since CMK_COMMON_NOCOPY_DIRECT_BYTES is set to 0 when CMK_USE_CMA is false
            // because the offset (returned by CmiGetRdmaCommonInfoSize) should equal CMK_COMMON_NOCOPY_DIRECT_BYTES
#endif
}

#if CMK_USE_CMA
#include <sys/uio.h> // for struct iovec
extern int cma_works;
int readShmCma(pid_t, char*, char*, size_t);
int writeShmCma(pid_t, char *, char *, size_t);

// These methods are also used by the generic layer implementation of the Direct API
void CmiIssueRgetUsingCMA(
  const void* srcAddr,
  void *srcInfo,
  int srcPe,
  const void* destAddr,
  void *destInfo,
  int destPe,
  size_t size) {

  // get remote process id
  CmiCommonRdmaInfo_t *remoteCommInfo = (CmiCommonRdmaInfo_t *)srcInfo;
  pid_t pid = remoteCommInfo->pid;
  readShmCma(pid, (char *)destAddr, (char *)srcAddr, size);
}

void CmiIssueRputUsingCMA(
  const void* destAddr,
  void *destInfo,
  int destPe,
  const void* srcAddr,
  void *srcInfo,
  int srcPe,
  size_t size) {

  // get remote process id
  CmiCommonRdmaInfo_t *remoteCommInfo = (CmiCommonRdmaInfo_t *)destInfo;
  pid_t pid = remoteCommInfo->pid;
  writeShmCma(pid, (char *)srcAddr, (char *)destAddr, size);
}
#endif


// Function Pointer to the acknowledement handler function for the Direct API
extern RdmaAckHandlerFn ncpyDirectAckHandlerFn;

typedef struct _cmi_rdma_direct_ack {
  const void *srcAddr;
  int srcPe;
  const void *destAddr;
  int destPe;
  int ackSize;
} CmiRdmaDirectAck;

void CmiInvokeNcpyAck(void *ack) {
  ncpyDirectAckHandlerFn(ack);
}

#endif
