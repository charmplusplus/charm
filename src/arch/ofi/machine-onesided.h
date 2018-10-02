/* Support for the Ncpy Entry Method API */
#ifndef OFI_MACHINE_ONESIDED_H
#define OFI_MACHINE_ONESIDED_H

// Signature of the callback function to be called on rdma direct operation completion
typedef void (*ofiCallbackFn)(struct fi_cq_tagged_entry *e, OFIRequest *req);

inline void process_onesided_completion_ack(struct fi_cq_tagged_entry *e, OFIRequest *req);
inline void process_onesided_reg_and_put(struct fi_cq_tagged_entry *e, OFIRequest *req);
inline void process_onesided_reg_and_get(struct fi_cq_tagged_entry *e, OFIRequest *req);

/* Support for Nocopy Direct API */
// Structure representing the machine specific information for a source or destination buffer used in the direct API
typedef struct _cmi_ofi_rzv_rdma_pointer {
  uint64_t       key;
  struct fid_mr  *mr;
}CmiOfiRdmaPtr_t;

/* Compiler checks to ensure that CMK_NOCOPY_DIRECT_BYTES in conv-common.h
 * is set to sizeof(CmiOfiRdmaPtr_t). CMK_NOCOPY_DIRECT_BYTES is used in
 * ckrdma.h to reserve bytes for source or destination metadata info.           */
#define DUMB_STATIC_ASSERT(test) typedef char sizeCheckAssertion[(!!(test))*2-1]

/* Check the value of CMK_NOCOPY_DIRECT_BYTES if the compiler reports an
 * error with the message "the size of an array must be greater than zero" */
DUMB_STATIC_ASSERT(sizeof(CmiOfiRdmaPtr_t) == CMK_NOCOPY_DIRECT_BYTES);

// Structure to track the progress of an RDMA read or write call
typedef struct _cmi_ofi_rzv_rdma_completion {
  void *ack_info;
  int  completion_count;
}CmiOfiRdmaComp_t;

// Set the machine specific information for a nocopy pointer
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode);

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

// Method invoked to deregister memory handle
void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode);

#endif /* OFI_MACHINE_ONESIDED_H */
