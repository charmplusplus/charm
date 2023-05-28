#define CMK_USE_LRTS                                      1

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1

#define CMK_HAS_PARTITION                                  1
#define CMK_HAS_INTEROP                                    1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0

#define CMK_HANDLE_SIGUSR                                  1

//#define  DELTA_COMPRESS                                     1
#if DELTA_COMPRESS
#define CMK_MSG_HEADER_EXT_     CmiUInt8 persistRecvHandler; CmiUInt4 compressStart; int root, size; CmiUInt2 rank, hdl,xhdl,info,redID,padding,compress_flag,xxhdl; unsigned char cksum, magic; CmiUInt1 zcMsgType:4, cmaMsgType:2, nokeep:1;
#else
#define CMK_MSG_HEADER_EXT_    int root, size; CmiUInt2 rank, hdl,xhdl,info,redID,padding; unsigned char cksum, magic; CmiUInt1 zcMsgType:4, cmaMsgType:2, nokeep:1;
#endif

#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT    { CMK_MSG_HEADER_EXT_ }

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 0
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

#define CMK_SSH_IS_A_COMMAND                               0
#define CMK_SSH_NOT_NEEDED                                 1

#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1

#define NODE_0_IS_CONVHOST                                 1

//#define CMK_IMMEDIATE_MSG				   1
#define CMK_MACHINE_PROGRESS_DEFINED                       0

#undef CMK_HAS_FDATASYNC_FUNC
#define CMK_HAS_FDATASYNC_FUNC                             0

//#define CMI_DIRECT_MANY_TO_MANY_DEFINED                    0

#define CMK_PERSISTENT_COMM                                0

#define  CMI_DIRECT_MANY_TO_MANY_DEFINED                   1

#define CMK_USE_COMMON_LOCK                                1

#define CMK_CONVERSE_MPI                                   0

#define CMK_CONVERSE_PAMI                                  1
