#ifndef _CONV_CPATH_H
#define _CONV_CPATH_H

#ifdef __cplusplus
extern "C" {
#endif

/* For CmiRegisterHandler and CmiHandler */
#include "converse.h"

typedef struct
{
  int    seqno;
  short  creator;
  short  startfn;
  short  mapfn;
  short  nsizes;
  int    sizes[13];
}
CPath;

#define CPathArrayDimensions(a) ((a)->nsizes)
#define CPathArrayDimension(a,n) ((a)->sizes[n])

#define CPATH_WILD (-1)

typedef unsigned int (*CPathMapFn)(CPath *path, int *indices);
typedef void (*CPathReduceFn)(int nelts,void *updateme,void *inputme);

#define CPathRegisterMapper(x)   CmiRegisterHandler((CmiHandler)(x))
#define CPathRegisterThreadFn(x) CmiRegisterHandler((CmiHandler)(x))
#define CPathRegisterReducer(x)  CmiRegisterHandler((CmiHandler)(x))

void CPathMakeArray(CPath *path, int startfn, int mapfn, ...);
void CPathMakeThread(CPath *path, int startfn, int pe);

void  CPathSend(int key, ...);
void *CPathRecv(int key, ...);
void  CPathReduce(int key, ...);

void CPathMsgDecodeBytes(void *msg, int *len, void *bytes);
void CPathMsgDecodeReduction(void *msg,int *vecsize,int *eltsize,void *bytes);
void CPathMsgFree(void *msg);

#define CPATH_ALL    (-1)
#define CPATH_END      0
#define CPATH_DEST     1
#define CPATH_DESTELT  2
#define CPATH_TAG      3
#define CPATH_TAGS     4
#define CPATH_TAGVEC   5
#define CPATH_BYTES    6
#define CPATH_OVER     7
#define CPATH_REDUCER  8
#define CPATH_REDBYTES 9

#ifdef __cplusplus
}
#endif

#endif
