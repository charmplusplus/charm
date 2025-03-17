#include <string.h>
#include "idlework.h"

Main::Main(CkArgMsg *m) {
    CkPrintf("[MAIN] Creating Test Array.\n");
    CProxy_Test testProxy = CProxy_Test::ckNew(1);
}

Test::Test() {
    thisProxy.registerIdleWork();
}

void Test::registerIdleWork() {
    CkPrintf("[TEST] Registering Idle Work.\n");
    CkCallWhenIdle(CkIndex_Test::idleProgress(), this);
}

bool Test::idleProgress() {
    static bool calledBefore = false; 
    CkPrintf("[TEST] Idle Work Called, CalledBefore=%d.\n", (int)calledBefore);
    calledBefore = !calledBefore;
    if (!calledBefore) {
        CkExit();
    }
    return calledBefore;
}

#include "idlework.def.h"

