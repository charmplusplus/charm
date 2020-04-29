#ifndef __CKARRAYMAP_H
#define __CKARRAYMAP_H

#include "CkArrayMap.decl.h"

/******************* Map object ******************/

extern CkGroupID _defaultArrayMapID;
extern CkGroupID _fastArrayMapID;

/**
\addtogroup CkArray
*/
/*@{*/

#include "ckarrayoptions.h"

#if CMK_CHARMPY

extern int (*ArrayMapProcNumExtCallback)(int, int, const int *);

class ArrayMapExt: public CkArrayMap {
public:
  ArrayMapExt(void *impl_msg);

  static void __ArrayMapExt(void *impl_msg, void *impl_obj_void) {
    new (impl_obj_void) ArrayMapExt(impl_msg);
  }

  static void __entryMethod(void *impl_msg, void *impl_obj_void) {
    //fprintf(stderr, "ArrayMapExt:: entry method invoked\n");
    ArrayMapExt *obj = static_cast<ArrayMapExt *>(impl_obj_void);
    CkMarshallMsg *impl_msg_typed = (CkMarshallMsg *)impl_msg;
    char *impl_buf = impl_msg_typed->msgBuf;
    PUP::fromMem implP(impl_buf);
    int msgSize; implP|msgSize;
    int ep; implP|ep;
    int dcopy_start; implP|dcopy_start;
    GroupMsgRecvExtCallback(obj->thisgroup.idx, ep, msgSize, impl_buf+(3*sizeof(int)),
                            dcopy_start);
  }

  int procNum(int arrayHdl, const CkArrayIndex &element) {
    return ArrayMapProcNumExtCallback(thisgroup.idx, element.getDimension(), element.data());
    //fprintf(stderr, "[%d] ArrayMapExt - procNum is %d\n", CkMyPe(), pe);
  }
};

#endif

/*@}*/

/**
\addtogroup CkArrayImpl
\brief Migratable Chare Arrays: Implementation classes.
*/
/*@{*/
static inline CkGroupID CkCreatePropMap(void)
{
  return CProxy_PropMap::ckNew();
}

extern void _propMapInit(void);

#endif
