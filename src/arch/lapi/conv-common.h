
#define CMK_CONVERSE_LAPI                                  1

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0

#define CMK_MSG_HEADER_FIELDS CmiUInt2 rank,hdl,xhdl,info,stratid,redID; int root, size, srcpe, seqno;

#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT    { CMK_MSG_HEADER_FIELDS }
#define CMK_MSG_HEADER_BIGSIM_    { CMK_MSG_HEADER_FIELDS CMK_BIGSIM_FIELDS }

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 1
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1

/* definitions specific of lapi */
/* CmiReference seems not to work at all, the other broadcast have not been fully checked */
#define CMK_BROADCAST_SPANNING_TREE                        0
#define CMK_BROADCAST_HYPERCUBE                            0
#define CMK_BROADCAST_USE_CMIREFERENCE                     0

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1

#define CMK_CCS_AVAILABLE                                  1

#define NODE_0_IS_CONVHOST                                 1

#define CMK_IMMEDIATE_MSG				   1
#define CMK_MACHINE_PROGRESS_DEFINED                       1
