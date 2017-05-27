#ifndef OMP_CHARM_H
#define OMP_CHARM_H
#include "converse.h"
#include "conv-config.h"
#include "kmp.h"

CpvExtern(int, curNumThreads);
CpvExtern(int, OmpHandlerIdx);
extern void StealTask();
#define CharmOMPDebug(...) // CmiPrintf(__VA_ARGS__)
// the intial ratio of OpenMP tasks in local list and work-stealing taskqueue
#define INITIAL_RATIO 2
#define windowSize 16
#endif
