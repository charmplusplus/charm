#include "OmpCharm.decl.h"
#include "ompcharm.h"

extern void* __kmp_launch_worker(void *);
CpvDeclare(int, curNumThreads);
CpvDeclare(int, OmpHandlerIdx);
static void RegisterOmpHdlrs() {
    CpvInitialize(int, OmpHandlerIdx); 
    CpvAccess(OmpHandlerIdx) = CmiRegisterHandler((CmiHandler)__kmp_launch_worker);
}

extern "C" int CmiGetCurKnownOmpThreads() {
  return CpvAccess(curNumThreads);
}

#include "OmpCharm.def.h"
