#ifndef _CONV_HEADER_H
#define _CONV_HEADER_H

#include "conv-config.h"

/******** CMI: TYPE DEFINITIONS ********/
typedef CMK_TYPEDEF_INT1      CmiInt1;
typedef CMK_TYPEDEF_INT2      CmiInt2;
typedef CMK_TYPEDEF_INT4      CmiInt4;
typedef CMK_TYPEDEF_INT8      CmiInt8;
typedef CMK_TYPEDEF_UINT1     CmiUInt1;
typedef CMK_TYPEDEF_UINT2     CmiUInt2;
typedef CMK_TYPEDEF_UINT4     CmiUInt4;
typedef CMK_TYPEDEF_UINT8     CmiUInt8;
#if CMK___int128_t_DEFINED
typedef __int128_t            CmiInt16;
typedef __uint128_t           CmiUInt16;
#elif CMK___int128_DEFINED
typedef __int128              CmiInt16;
typedef __uint128     CmiUInt16;
#endif

#if defined(CMK_CUSTOM_FP_FORMAT)
typedef CMK_TYPEDEF_FLOAT4    CmiFloat4;
typedef CMK_TYPEDEF_FLOAT8    CmiFloat8;
#else
typedef float                 CmiFloat4;
typedef double                CmiFloat8;
#endif

typedef void  *CmiCommHandle;
typedef void (*CmiHandler)(void *msg);
typedef void (*CmiHandlerEx)(void *msg,void *userPtr);



typedef struct CMK_MSG_HEADER_BASIC CmiMsgHeaderBasic;
typedef struct CMK_MSG_HEADER_EXT   CmiMsgHeaderExt;

#define CmiMsgHeaderSizeBytes (sizeof(CmiMsgHeaderBasic))
#define CmiExtHeaderSizeBytes (sizeof(CmiMsgHeaderExt))

/* all common extra fields in BigSim message header */
#define CMK_BIGSIM_FIELDS  CmiInt4 nd,n; double rt; CmiInt2 tID, hID; char t, flag; CmiInt2 ref; CmiInt4 msgID, srcPe;

#ifndef CmiReservedHeaderSize
typedef struct CMK_MSG_HEADER_BIGSIM_   CmiBlueGeneMsgHeader;
#define CmiBlueGeneMsgHeaderSizeBytes (sizeof(CmiBlueGeneMsgHeader))
#if CMK_BIGSIM_CHARM
#  define CmiReservedHeaderSize   CmiBlueGeneMsgHeaderSizeBytes
#else
#  define CmiReservedHeaderSize   CmiExtHeaderSizeBytes
#endif
#endif

#endif
