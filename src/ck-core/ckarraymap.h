#ifndef __CKARRAYMAP_H
#define __CKARRAYMAP_H

#include "CkArrayMap.decl.h"

#include "cklocation.h"

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

class RRMapObj : public CkArrayMapObj {
PUPable_decl(RRMapObj);
public:
  RRMapObj() {}
  RRMapObj(CkMigrateMessage* m) {}

  int homePe(const CkArrayIndex& i) const;
};

class DefaultArrayMapObj : public CkArrayMapObj {
PUPable_decl(DefaultArrayMapObj);
protected:
  int totalChares, blockSize, firstSet, remainder;
public:
  DefaultArrayMapObj()
      : totalChares(0), blockSize(0), firstSet(0), remainder(0) {}
  DefaultArrayMapObj(CkMigrateMessage* m) {}

  void setArrayOptions(const CkArrayOptions& opts);
  int homePe(const CkArrayIndex& i) const;
  void pup(PUP::er& p);
};

class FastArrayMapObj : public DefaultArrayMapObj {
PUPable_decl(FastArrayMapObj);
public:
  FastArrayMapObj() {}
  FastArrayMapObj(CkMigrateMessage* m) : DefaultArrayMapObj(m) {}

  void setArrayOptions(const CkArrayOptions& opts);
  int homePe(const CkArrayIndex& i) const;
};

/*@}*/

/**
\addtogroup CkArrayImpl
\brief Migratable Chare Arrays: Implementation classes.
*/
/*@{*/
#if 0
static inline CkGroupID CkCreatePropMap(void)
{
  return CProxy_PropMap::ckNew();
}

extern void _propMapInit(void);
#endif

#endif
