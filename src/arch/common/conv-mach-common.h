/* This header consists of declarations and definitions that are common to all
 * machine layer implementations and architectures. This header is included in
 * conv-config.h, which is included in the machine layer implementations through
 * converse.h.
 */

/* CMK_BUFFER_REG always registers the buffer by treating network and non-network transfers
 * equally. It avoids an extra small message to register, as done in the case of CMK_BUFFER_UNREG
 */
#define CMK_BUFFER_REG                 0

/* CMK_BUFFER_UNREG is the default mode which doesn't register the buffer and avoids
 * non-network registration by registering only when required i.e only when this buffer
 * is involved in an RDMA operation using the network. (either as a source or destination).
 * The registeration on demand is performed by sending a small message to the remote process
 * in order to register the remote buffer. Zerocopy operations using this mode use the inverse
 * operation mechanism, where along with the remote registration, the RDMA operation is performed
 * by the remote process (which is not the initiator) to avoid sending another small message back to
 * the initiator.
 */
#define CMK_BUFFER_UNREG               1

/* CMK_BUFFER_PREREG is a mode that is to be supported in the future. It is intended to allow users
 * to pass buffers allocated through CmiAlloc (which uses mempools on networks requiring explicit
 * registration).
 */
#define CMK_BUFFER_PREREG              2

/* CMK_BUFFER_NOREG is used in those cases where there is no registration or setting of machine specific
 * information required. It is intended for use cases where the Ncpy objects are created on remote processes
 * to just store basic information about remote buffers without registering them.
 */
#define CMK_BUFFER_NOREG               3

// default of CMK_COMMON_NOCOPY_DIRECT_BYTES assumes no CMA support
// Refined for lrts layers with CMA support inside lrts-common.h
#define CMK_COMMON_NOCOPY_DIRECT_BYTES 0
