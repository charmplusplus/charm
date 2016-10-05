#include "OmpCharm.decl.h"
#include "ompcharm.h"

extern void* __kmp_launch_worker(void *);

CpvDeclare(int, OmpHandlerIdx);
static void RegisterOmpHdlrs() {
    CpvInitialize(int, OmpHandlerIdx); 
    CpvAccess(OmpHandlerIdx) = CmiRegisterHandler((CmiHandler)__kmp_launch_worker);
}

#include "OmpCharm.def.h"
