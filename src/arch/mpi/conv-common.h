
#define CMK_USE_LRTS                                      1

#define CMK_HAS_PARTITION                                1
#define CMK_HAS_INTEROP                                    1

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0

#define CMI_MPI_TRACE_USEREVENTS                           0

#define CMK_STACKSIZE_DEFAULT                              65536

#define CMK_HANDLE_SIGUSR                                  1

#if CMK_ERROR_CHECKING
#define CMK_MSG_HEADER_EXT_    CmiUInt2 rank, hdl,xhdl,info, stratid, redID; CmiInt4 root; unsigned char cksum, magic;
#else
#define CMK_MSG_HEADER_EXT_    CmiUInt2 rank, hdl,xhdl,info, stratid, redID; CmiInt4 root; 
#endif
#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT    { CMK_MSG_HEADER_EXT_ }
#define CMK_MSG_HEADER_BIGSIM_    { CMK_MSG_HEADER_EXT_ CMK_BIGSIM_FIELDS }

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 1
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

#define CMK_SSH_IS_A_COMMAND                               0
#define CMK_SSH_NOT_NEEDED                                 1

#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1

#define CMK_CCS_AVAILABLE                                  1

#define NODE_0_IS_CONVHOST                                 1

#define CMK_IMMEDIATE_MSG				   1
#define CMK_MACHINE_PROGRESS_DEFINED                       1

#define CMK_LB_CPUTIMER					   0

#if CMK_CXX_MPI_BINDINGS==0
#define MPICH_IGNORE_CXX_SEEK //for build issues with mpich + intel
#endif
