
#define CMK_USE_LRTS                                      1

#define CMK_HAS_PARTITION                                  1
#define CMK_HAS_INTEROP                                    1

#define CMK_CONVERSE_UGNI                                  1

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0

#define CMI_MACH_TRACE_USEREVENTS                          0

#define  CMK_DIRECT                                             0

//#define  DELTA_COMPRESS                                     1

#define CMK_HANDLE_SIGUSR                                  0

#if DELTA_COMPRESS
#if CMK_ERROR_CHECKING
#define CMK_MSG_HEADER_EXT_    CmiUInt4 size; CmiUInt2 seq; unsigned char cksum, magic; CmiUInt2 rank,hdl,xhdl,info,type,redID; CmiInt4 root; CmiUInt4 compressStart; CmiUInt2 compress_flag,xxhdl; CmiUInt8 persistRecvHandler; 
#else
#define CMK_MSG_HEADER_EXT_    CmiUInt4 size; CmiUInt4 seq; CmiUInt2 rank,hdl,xhdl,info,type,redID; CmiInt4 root; CmiUInt4 compressStart; CmiUInt2 compress_flag,xxhdl; CmiUInt8 persistRecvHandler; 
#endif
#else 
#if CMK_ERROR_CHECKING
#define CMK_MSG_HEADER_EXT_    CmiUInt4 size; CmiUInt2 seq; unsigned char cksum, magic; CmiUInt2 rank,hdl,xhdl,info,type,redID; CmiInt4 root;  
#else
#define CMK_MSG_HEADER_EXT_    CmiUInt4 size; CmiUInt4 seq; CmiUInt2 rank,hdl,xhdl,info,type,redID; CmiInt4 root;  
#endif
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

#define CMK_USE_OOB                                        0

#define CMK_IMMEDIATE_MSG				   1
#define CMK_MACHINE_PROGRESS_DEFINED                       1

#define CMK_LB_CPUTIMER					   0

#define CMK_USE_COMMON_LOCK                                1

#define CMK_ONESIDED_IMPL 			 	 1

#define CMK_CMA_MIN                                        65536

#define CMK_CMA_MAX                                        1048576

#define CMK_ONESIDED_DIRECT_IMPL                           1

#define CMK_NOCOPY_DIRECT_BYTES                           16

#define CMK_REG_REQUIRED                                   1
