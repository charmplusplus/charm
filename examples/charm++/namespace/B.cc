#include "B.h"

namespace Base { namespace Derived {
    B::B() {}
    B::B(CkMigrateMessage *m) {}
    void B::quit() {
        CkExit();
    }
} }

#include "B.def.h"
