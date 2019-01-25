/* This header is included in conv-config.h, which is included in the
 * machine layer implementations through converse.h
 */

// Use CMA for intra node shared memory communication on all machines where it is supported, except for Multicore
#define CMK_USE_CMA                    (CMK_HAS_CMA && !CMK_MULTICORE && !CMK_BLUEGENEQ)

#if CMK_USE_CMA

#ifndef CMK_CMA_MIN
#define CMK_CMA_MIN                    1024
#endif

#ifndef CMK_CMA_MAX
#define CMK_CMA_MAX                    131072
#endif

#endif // end of CMK_USE_CMA

// Converse Message header contains msgtype which is set to one of these message types
enum cmiCMAMsgType {
  // CMK_REG_NOCMA_MSG refers to a message which contains the payload being sent
  CMK_REG_NO_CMA_MSG=0,

  // CMK_CMA_MD_MSG refers to a message which contains payload metadata (pe, pid, address, size) without the payload
  // This message is used by the receiving process (running on the same physical host) to perform a CMA read operation
  CMK_CMA_MD_MSG=1,

  // CMK_CMA_ACK_MSG refers to a message which contains payload metadata (pe, pid, address, size) without the payload
  // This message is sent by the receiving process to the sending process to signal the completion of the CMA operation in order
  // to free the payload buffer
  CMK_CMA_ACK_MSG=2,
};

#if CMK_USE_CMA
#undef  CMK_COMMON_NOCOPY_DIRECT_BYTES // previous definition is in conv-mach-common.h
#define CMK_COMMON_NOCOPY_DIRECT_BYTES sizeof(pid_t)
#endif
