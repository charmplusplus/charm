
#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1
#define CMK_CMIDELIVERS_USE_SPECIAL_CODE                   0

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0
#define CMK_CMIPRINTF_IS_JUST_PRINTF                       1

#define CMK_HANDLE_SIGUSR                                  1

#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
/* Type of the message tells whether it is a statically or dynamically allocated message, 0 for dynamic and 1 for static */
#define CMK_MSG_HEADER_EXT    { unsigned char type, rank; CmiUInt2 root, hdl,xhdl,info,d3; }
#define CMK_MSG_HEADER_BLUEGENE    { CmiUInt2 rank, root, hdl,xhdl,info,d3; int nd, n; double rt; CmiUInt2 tID, hID; char t; int msgID; int srcPe;}

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 0
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

#define CMK_REDUCTION_USES_COMMON_CODE                     1
#define CMK_REDUCTION_USES_SPECIAL_CODE                    0

#define CMK_RSH_IS_A_COMMAND                               0
#define CMK_RSH_NOT_NEEDED                                 1
#define CMK_RSH_USE_REMSH                                  0

#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1
#define CMK_SPANTREE_USE_SPECIAL_CODE                      0

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1
#define CMK_VECTOR_SEND_USES_SPECIAL_CODE                  0

#define CMK_CCS_AVAILABLE                                  1

#define NODE_0_IS_CONVHOST                                 1
#define CONVERSE_VERSION_ELAN                              1

#define  CMK_PERSISTENT_COMM                               0

#define CMK_IMMEDIATE_MSG				   1
