#define CMK_USE_LRTS	                                   1

#define	CMK_HAS_PARTITION		                   1

#define CMK_NET_VERSION					   1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         1

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1

#define CMK_HANDLE_SIGUSR                                  1

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 1
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

/* the first 4 fields of the header are defined in machine-dgram.c
   and are used for the udp retransmission protocol implementation.
   Parameters stratid and root are for the communication library.
   Stratid is the stratgy id and root is used in the broadcast.
*/
#define CMK_MSG_HEADER_BASIC   CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT_    CmiUInt2 d0,d1,d2,d3,hdl,stratid,xhdl,info,redID,rank; CmiInt4 root, size;
#define CMK_MSG_HEADER_EXT       { CMK_MSG_HEADER_EXT_ }
#define CMK_MSG_HEADER_BIGSIM_  { CMK_MSG_HEADER_EXT_ CMK_BIGSIM_FIELDS }

#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1

#define CMK_IMMEDIATE_MSG				   1

#define CMK_PERSISTENT_COMM				   0

#define CMK_OBJECT_QUEUE_AVAILABLE		   	   0
#define CMK_USE_SPECIAL_MESSAGE_QUEUE_CHECK                1

#define CMK_BARRIER_USE_COMMON_CODE                        1

#define CMK_MACHINE_PROGRESS_DEFINED                       1

#define NODE_0_IS_CONVHOST                                 0

/* call cpu timer for LB */
#define CMK_LB_CPUTIMER					   0

#define CMK_HAS_SIZE_IN_MSGHDR                             1
