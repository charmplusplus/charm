
#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0

#define CMK_HANDLE_SIGUSR                                  1

//#define CMK_ENABLE_ASYNC_PROGRESS                          1

#if CMK_ENABLE_ASYNC_PROGRESS
#define CMK_MSG_HEADER_EXT_    CmiUInt2 rank, hdl,xhdl,info, stratid; unsigned char cksum, magic; int root, size, dstnode; CmiUInt2 redID, padding; char work[6*sizeof(uintptr_t)]; 
#else
#define CMK_MSG_HEADER_EXT_    CmiUInt2 rank, hdl,xhdl,info, stratid; unsigned char cksum, magic; int root, size; CmiUInt2 redID, padding; 
#endif
#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT    { CMK_MSG_HEADER_EXT_ }
#define CMK_MSG_HEADER_BIGSIM_    { CMK_MSG_HEADER_EXT_ CMK_BIGSIM_FIELDS }

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 0
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

#define CMK_RSH_IS_A_COMMAND                               0
#define CMK_RSH_NOT_NEEDED                                 1
#define CMK_RSH_USE_REMSH                                  0

#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1

#define CMK_CCS_AVAILABLE                                  1

#define NODE_0_IS_CONVHOST                                 1

//#define CMK_IMMEDIATE_MSG				   1
#define CMK_MACHINE_PROGRESS_DEFINED                       0

//#define CMI_DIRECT_MANY_TO_MANY_DEFINED                    0

#define CMK_PERSISTENT_COMM                                0

#define  CMI_DIRECT_MANY_TO_MANY_DEFINED                   1
