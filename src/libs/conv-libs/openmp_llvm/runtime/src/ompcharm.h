#ifndef OMP_CHARM_H
#define OMP_CHARM_H
#include "converse.h"
#include "conv-config.h"
#include "kmp.h"
typedef struct KMP_ALIGN_CACHE ompConverseMsg {
  char core[CmiMsgHeaderSizeBytes];
  void *data;
} OmpConverseMsg;

typedef OmpConverseMsg* OmpCharmMsg;
CpvExtern(int, OmpHandlerIdx);
#define CharmOMPDebug(...) //CmiPrintf(__VA_ARGS__)
#endif
