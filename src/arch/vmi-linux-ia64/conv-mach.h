#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_IA64                                           1

#define CMK_MEMORY_PAGESIZE                                16384




#if 0
/************************************************************************/


#define CMK_USE_GM                                         0


/*
#define CMK_CCS_AVAILABLE                                  1
*/
#define CMK_CCS_AVAILABLE                                  0 

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1
#define CMK_CMIDELIVERS_USE_SPECIAL_CODE                   0

/*
#define CMK_CMIPRINTF_IS_A_BUILTIN                         1
#define CMK_CMIPRINTF_IS_JUST_PRINTF                       0
*/
#define CMK_CMIPRINTF_IS_A_BUILTIN                         0 
#define CMK_CMIPRINTF_IS_JUST_PRINTF                       1 

#define CMK_HANDLE_SIGUSR                                  1

/*
#define CMK_MALLOC_USE_GNU_MALLOC                          1
#define CMK_MALLOC_USE_OS_BUILTIN                          0
*/
#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1


/*
#define CMK_MSG_HEADER_BASIC  { CmiUInt2 d0,d1,d2,d3,d4,d5,hdl,d7; }
#define CMK_MSG_HEADER_EXT    { CmiUInt2 d0,d1,d2,d3,d4,d5,hdl,xhdl,info,d9,da,db; }
*/

#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT    { CmiUInt2 rank,root,hdl,xhdl,info,d3; }

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 1
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1


#define CMK_REDUCTION_USES_COMMON_CODE                     1
#define CMK_REDUCTION_USES_SPECIAL_CODE                    0


#define CMK_CONV_HOST_WANT_CSH                             1



#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1
#define CMK_SPANTREE_USE_SPECIAL_CODE                      0




#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1
#define CMK_VECTOR_SEND_USES_SPECIAL_CODE                  0


#define CMK_USE_HP_MAIN_FIX                                0
#define CMK_DONT_USE_HP_MAIN_FIX                           1

/*
#define CMK_WEB_MODE                                       1
*/

#define CMK_LBDB_OFF					   0

#include "conv-mach-opt.h"
#endif   // 0


#endif
