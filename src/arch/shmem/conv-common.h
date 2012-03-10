#ifndef _CONV_COMMON_H
#define _CONV_COMMON_H

#define CONVERSE_VERSION_SHMEM                             1

#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT_   char gap[56]; CmiUInt2 hdl,xhdl,info,stratid,root,redID,padding2,padding3;
#define CMK_MSG_HEADER_EXT       { CMK_MSG_HEADER_EXT_ }
#define CMK_MSG_HEADER_BIGSIM_  {CMK_MSG_HEADER_EXT_ CMK_BIGSIM_FIELDS}

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 0
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                0

#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0

#define CMK_CCS_AVAILABLE                                  1

#define CMK_HANDLE_SIGUSR                                  1

#define NODE_0_IS_CONVHOST                                 1

#endif
