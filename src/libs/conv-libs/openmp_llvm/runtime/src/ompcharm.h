#ifndef OMP_CHARM_H
#define OMP_CHARM_H
#include "converse.h"
#include "conv-config.h"
#include "kmp.h"
typedef struct KMP_ALIGN_CACHE ompConvMsg_base {
  char core[CmiMsgHeaderSizeBytes];
  void *data;
} ompConvMsg_base_t;

typedef union KMP_ALIGN_CACHE ompConverseMsg {
  ompConvMsg_base_t convMsg;
  double convMsg_align;
  char pad[KMP_PAD(ompConvMsg_base_t , CACHE_LINE)];
} OmpConverseMsg;

CpvExtern(int, OmpHandlerIdx);
#define CharmOMPDebug(...) //CmiPrintf(__VA_ARGS__)
// the intial ratio of OpenMP tasks in local list and work-stealing taskqueue
#define INITIAL_RATIO 2
#define windowSize 4
#endif
