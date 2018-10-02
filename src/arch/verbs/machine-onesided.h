#ifndef VERBS_MACHINE_ONESIDED_H
#define VERBS_MACHINE_ONESIDED_H

typedef struct _cmi_verbs_rzv_rdma_pointer {
  struct ibv_mr *mr;
  uint32_t key;
}CmiVerbsRdmaPtr_t;

/* Compiler checks to ensure that CMK_NOCOPY_DIRECT_BYTES in conv-common.h
 * is set to sizeof(CmiVerbsRdmaPtr_t). CMK_NOCOPY_DIRECT_BYTES is used in
 * ckrdma.h to reserve bytes for source or destination metadata info.           */
#define DUMB_STATIC_ASSERT(test) typedef char sizeCheckAssertion[(!!(test))*2-1]

/* Machine specific metadata information required to register a buffer and perform
 * an RDMA operation with a remote buffer. This metadata information is used to perform
 * registration and a PUT operation when the remote buffer wants to perform a GET with an
 * unregistered buffer. Similary, the metadata information is used to perform registration
 * and a GET operation when the remote buffer wants to perform a PUT with an unregistered
 * buffer.*/
typedef struct _cmi_verbs_rdma_reverse_op {
  const void *destAddr;
  int destPe;
  int destMode;
  const void *srcAddr;
  int srcPe;
  int srcMode;

  struct ibv_mr *rem_mr;
  uint32_t rem_key;
  int ackSize;
  int size;
} CmiVerbsRdmaReverseOp_t;

/* Check the value of CMK_NOCOPY_DIRECT_BYTES if the compiler reports an
 * error with the message "the size of an array must be greater than zero" */
DUMB_STATIC_ASSERT(sizeof(CmiVerbsRdmaPtr_t) == CMK_NOCOPY_DIRECT_BYTES);

// Function Declarations
// Set the machine specific information for a nocopy pointer
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode);

void postRdma(
  uint64_t local_addr,
  uint32_t local_rkey,
  uint64_t remote_addr,
  uint32_t remote_rkey,
  int size,
  int peNum,
  uint64_t rdmaPacket,
  int opcode);

// Perform an RDMA Get call into the local destination address from the remote source address
void LrtsIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  unsigned short int *srcMode,
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  unsigned short int *destMode,
  int size);

// Perform an RDMA Put call into the remote destination address from the local source address
void LrtsIssueRput(
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  unsigned short int *destMode,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  unsigned short int *srcMode,
  int size);

void registerDirectMemory(void *info, const void *addr, int size);

#endif /* VERBS_MACHINE_ONESIDED_H */
