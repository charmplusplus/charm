#include "OmpCharm.decl.h"
#include "ompcharm.h"

extern void* __kmp_launch_worker(void *);
CpvDeclare(int, prevGtid);
CpvDeclare(unsigned int, localRatio);
CpvDeclare(unsigned int*, localRatioArray);
CpvDeclare(unsigned int, ratioIdx);
CpvDeclare(bool, ratioInit);
CpvDeclare(unsigned int, ratioSum);
CpvDeclare(int, curNumThreads);
extern int __kmp_get_global_thread_id_reg();
static void RegisterOmpHdlrs() {
    CpvInitialize(int, prevGtid);
    CpvInitialize(unsigned int, localRatio);
    CpvInitialize(unsigned int, ratioIdx);
    CpvInitialize(unsigned int, ratioSum);
    CpvInitialize(bool, ratioInit);
    CpvInitialize(unsigned int*, localRatioArray);
    CpvInitialize(int, curNumThreads);
    CpvAccess(localRatioArray) = (unsigned int*) __kmp_allocate(sizeof(unsigned int) * windowSize);
    memset(CpvAccess(localRatioArray), 0, sizeof (unsigned int) * windowSize);
    CpvAccess(localRatio) = 0;
    CpvAccess(ratioIdx) = 0;
    CpvAccess(ratioSum) = 0;
    CpvAccess(ratioInit) = false;
    CpvAccess(prevGtid) = -2;
    __kmp_get_global_thread_id_reg();
}

extern "C" int CmiGetCurKnownOmpThreads() {
  return CpvAccess(curNumThreads);
}

#include "OmpCharm.def.h"
