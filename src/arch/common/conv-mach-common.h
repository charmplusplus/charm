/* This header consists of declarations and definitions that are common to all
 * machine layer implementations and architectures. This header is included in
 * conv-config.h, which is included in the machine layer implementations through
 * converse.h.
 */

// Enum for registration modes used in the Zerocopy API
// TODO:Convert to typed enum post C++ conversion
enum ncpyRegModes {
  /* CMK_BUFFER_REG always registers the buffer by treating network and non-network transfers
   * equally. It avoids an extra small message to register, as done in the case of CMK_BUFFER_UNREG
   */
  CMK_BUFFER_REG      = 0,

  /* CMK_BUFFER_UNREG is the default mode which doesn't register the buffer and avoids
   * non-network registration by registering only when required i.e only when this buffer
   * is involved in an RDMA operation using the network. (either as a source or destination).
   * The registeration on demand is performed by sending a small message to the remote process
   * in order to register the remote buffer. Zerocopy operations using this mode use the inverse
   * operation mechanism, where along with the remote registration, the RDMA operation is performed
   * by the remote process (which is not the initiator) to avoid sending another small message back to
   * the initiator.
   */
  CMK_BUFFER_UNREG    = 1,

  /* CMK_BUFFER_PREREG is a mode that is to be supported in the future. It is intended to allow users
   * to pass buffers allocated through CmiAlloc (which uses mempools on networks requiring explicit
   * registration).
   */
  CMK_BUFFER_PREREG   = 2,

  /* CMK_BUFFER_NOREG is used in those cases where there is no registration or setting of machine specific
   * information required. It is intended for use cases where the Ncpy objects are created on remote processes
   * to just store basic information about remote buffers without registering them.
   */
  CMK_BUFFER_NOREG    = 3
};

// default of CMK_COMMON_NOCOPY_DIRECT_BYTES assumes no CMA support
// Refined for lrts layers with CMA support inside lrts-common.h
#define CMK_COMMON_NOCOPY_DIRECT_BYTES 0

// Enum for the type of zerocopy operation
// TODO: Convert to typed enum post C++ conversion
enum ncpyOperationMode {
  CMK_DIRECT_API          = 0,
  CMK_EM_API              = 1,
  CMK_EM_API_REVERSE      = 2,
  CMK_BCAST_EM_API        = 3,
  CMK_BCAST_EM_API_REVERSE= 4,
  CMK_READONLY_BCAST      = 5
};

// Enum for the method of acknowledglement handling after the completion of a zerocopy operation
// TODO: Convert to typed enum post C++ conversion
enum ncpyAckMode {
  CMK_SRC_DEST_ACK       = 0,
  CMK_SRC_ACK            = 1,
  CMK_DEST_ACK           = 2
};

// Enum to determine if a NcpyOperationInfo can be freed upon completion
// TODO: Convert to a bool variable post C++ conversion
enum ncpyFreeNcpyOpInfoMode {
  CMK_FREE_NCPYOPINFO           = 0,
  CMK_DONT_FREE_NCPYOPINFO           = 1
};

// Enum for the type of converse message
// TODO: Convert to a bool variable post C++ conversion
enum cmiZCMsgType {
  CMK_REG_NO_ZC_MSG = 0,
  CMK_ZC_P2P_SEND_MSG = 1,
  CMK_ZC_P2P_RECV_MSG = 2,
  CMK_ZC_P2P_RECV_DONE_MSG = 3,
  CMK_ZC_BCAST_SEND_MSG = 4,
  CMK_ZC_BCAST_RECV_MSG = 5,
  CMK_ZC_BCAST_RECV_DONE_MSG = 6,
  CMK_ZC_BCAST_RECV_ALL_DONE_MSG = 7
};
