
#define CMK_CCS_AVAILABLE                                  1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         1
#define CMK_CMIPRINTF_IS_JUST_PRINTF                       0

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1
#define CMK_CMIDELIVERS_USE_SPECIAL_CODE                   0

#define CMK_HANDLE_SIGUSR                                  1

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 1
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

#define CMK_MSG_HEADER_BASIC  { CmiUInt2 d0,d1,d2,d3,d4,d5,hdl,d7; }
#define CMK_MSG_HEADER_EXT    { CmiUInt2 d0,d1,d2,d3,d4,d5,hdl,xhdl,info,d9,da,db; }
#define CMK_MSG_HEADER_BLUEGENE    { CmiUInt2 d0,d1,d2,d3,d4,d5,hdl,xhdl,info,d9,da,db; int nd, n; double rt; CmiUInt2 tID, hID; char t; int msgID; int srcPe;}

#define CMK_REDUCTION_USES_COMMON_CODE                     1
#define CMK_REDUCTION_USES_SPECIAL_CODE                    0

#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1
#define CMK_SPANTREE_USE_SPECIAL_CODE                      0

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1
#define CMK_VECTOR_SEND_USES_SPECIAL_CODE                  0

#define CMK_IMMEDIATE_MSG				   1

