#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/uio.h>

#define MAX_CMA_RW_COUNT 2147479552

// Method checks if process has permissions to use CMA
int CmiInitCma() {
  char buffer   = '0';
  int cma_works = 0;
  int fd;

  // determine permissions
  fd = open("/proc/sys/kernel/yama/ptrace_scope", O_RDONLY);
  if (0 <= fd) {
    if (read (fd, &buffer, 1) != 1) {
      CmiAbort("CMA> reading /proc/sys/kernel/yama/ptrace_scope failed!");
    }
    close(fd);
  }

  if('0' != buffer) {
#if defined PR_SET_PTRACER
    // set ptrace scope to allow attach
    int ret = prctl (PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    if (0 == ret) {
      cma_works = 1;
    }
#endif
  } else {
    cma_works = 1;
  }
  return cma_works;
}

// Method to display thresholds for regular messaging using CMA
void CmiDisplayCMAThresholds(int cma_min_threshold, int cma_max_threshold) {
  if(_Cmi_mynode == 0) {
      if(cma_min_threshold > cma_max_threshold) {
        CmiAbort("CMA size thresholds incorrect! Values should satisfy cma_min_threshold <= cma_min_threshold condition");
      }
      CmiPrintf("Charm++> CMA enabled for within node transfers of messages(sized between %d & %d bytes)\n", cma_min_threshold, cma_max_threshold);
  }
}

/* Used to read from a list of remote buffers and write into a
 * list of local buffers over SHM using the cma library
 */
int readShmCma(
  pid_t remotePid,
  char *localAddr,
  char *remoteAddr,
  size_t totalBytes) {

  int numOps = 1;
  size_t toBeReadTotal = totalBytes, nread_total = 0, offset = 0;
  struct iovec local, remote;

  while(toBeReadTotal) {
    local.iov_base = (char *)localAddr + offset;
    remote.iov_base = (char *)remoteAddr + offset;

    ssize_t toBeRead  = (toBeReadTotal > MAX_CMA_RW_COUNT) ? MAX_CMA_RW_COUNT : toBeReadTotal;
    local.iov_len  = toBeRead;
    remote.iov_len  = toBeRead;

    ssize_t nread = process_vm_readv(remotePid, &local, numOps, &remote, numOps, 0);
    if(nread != toBeRead) {
      CmiAbort("process_vm_readv failed! Size to be read:%zu bytes, Size actually read:%zd bytes (negative value indicates error), error no:%d, error message:%s!\n", toBeRead, nread, errno, strerror(errno));
      return errno;
    }
    // successful operation (nread == toBeRead)
    toBeReadTotal -= toBeRead;
    offset += toBeRead;
  }
  // all reading is complete
  return 0;
}

/* Used to write into a list of remote buffers by reading from a
 * list of local buffers over SHM using the cma library
 */
int writeShmCma(
  pid_t remotePid,
  char *localAddr,
  char *remoteAddr,
  size_t totalBytes) {

  int numOps = 1;
  size_t toBeWrittenTotal = totalBytes, nwriteTotal = 0, offset = 0;
  struct iovec local, remote;

  while(toBeWrittenTotal) {
    local.iov_base = (char *)localAddr + offset;
    remote.iov_base = (char *)remoteAddr + offset;

    ssize_t toBeWritten = (toBeWrittenTotal > MAX_CMA_RW_COUNT) ? MAX_CMA_RW_COUNT : toBeWrittenTotal;
    local.iov_len = toBeWritten;
    remote.iov_len = toBeWritten;

    ssize_t nwrite = process_vm_writev(remotePid, &local, numOps, &remote, numOps, 0);
    if(nwrite != toBeWritten) {
      CmiAbort("process_vm_writev failed! Size to be written:%zu bytes, Size actually written:%zd bytes (negative value indicates error), error no:%d, error message:%s!\n", toBeWritten, nwrite, errno, strerror(errno));
      return errno;
    }
    // successful operation (nwrite == toBeWritten)
    toBeWrittenTotal -= toBeWritten;
    offset += toBeWritten;
  }
  // all writing is complete
  return 0;
}

// Metadata of a buffer that is to be sent to another intra-host process over CMA
typedef struct _cma_src_buffer_info {
  int srcPE;
  pid_t srcPid;
  void *srcAddr;
  int size;
}CmaSrcBufferInfo_t;

// Method invoked on receiving a CMK_CMA_MD_MSG
// This method uses the buffer metadata to perform a CMA read. It also modifies *sizePtr & *msgPtr to
// point to the buffer message
void handleOneCmaMdMsg(int *sizePtr, char **msgPtr) {
  struct iovec local, remote;
  char *destAddr;

  // Get buffer metadata
  CmaSrcBufferInfo_t *bufInfo = (CmaSrcBufferInfo_t *)(*msgPtr + CmiMsgHeaderSizeBytes);

  // Allocate a buffer to hold the buffer
  destAddr = (char *)CmiAlloc(bufInfo->size);

  // Perform CMA read into destAddr
  readShmCma(bufInfo->srcPid,
             destAddr,
             (char *)bufInfo->srcAddr,
             bufInfo->size);

  // Send the buffer md msg back as an ack msg to signal CMA read completion in order to free buffers
  // on the source process
  CMI_CMA_MSGTYPE(*msgPtr) = CMK_CMA_ACK_MSG;

  CmiInterSendNetworkFunc(bufInfo->srcPE,
                          CmiMyPartition(),
                          CmiMsgHeaderSizeBytes + sizeof(CmaSrcBufferInfo_t),
                          *msgPtr,
                          P2P_SYNC);

  // Reassign *msgPtr to the buffer
  *msgPtr = destAddr;
  // Reassign *sizePtr to the size of the buffer
  *sizePtr = local.iov_len;
}


// Method invoked on receiving CMK_CMA_ACK_MSG
// This method frees the buffer and the received buffer ack msg
void handleOneCmaAckMsg(int size, void *msg) {

  // Get buffer metadata
  CmaSrcBufferInfo_t *bufInfo = (CmaSrcBufferInfo_t *)((char *)msg + CmiMsgHeaderSizeBytes);

  // Free the large buffer sent
  CmiFree(bufInfo->srcAddr);

  // Free this ack message
  CmiFree(msg);
}

// Method invoked to send the buffer via CMA
// This method creates a buffer metadata msg from a buffer and modifies the *msgPtr and *sizePtr to point to
// the buffer metadata msg.
void CmiSendMessageCma(char **msgPtr, int *sizePtr) {

  // Send buffer metadata instead of original msg
  // Buffer metadata msg consists of pid, addr, size for the other process to perform a read through CMA
  char *cmaBufMdMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes + sizeof(CmaSrcBufferInfo_t));
  CmaSrcBufferInfo_t *bufInfo = (CmaSrcBufferInfo_t *)(cmaBufMdMsg + CmiMsgHeaderSizeBytes);
  bufInfo->srcPE  = CmiMyPe();
  bufInfo->srcPid = getpid();
  bufInfo->srcAddr = *msgPtr;
  bufInfo->size    = *sizePtr;

  // Tag this message as a CMA buffer md message
  CMI_CMA_MSGTYPE(cmaBufMdMsg) = CMK_CMA_MD_MSG;

  // Reassign *sizePtr to store the size of the buffer md msg
  *sizePtr = CmiMsgHeaderSizeBytes + sizeof(CmaSrcBufferInfo_t);

  // Reassign *msgPtr to point to the buffer metadata msg
  *msgPtr = (char *)cmaBufMdMsg;
}
