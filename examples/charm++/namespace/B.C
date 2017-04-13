#include "B.h"

namespace Base { namespace Derived {
    B::B() {}
    B::B(CkMigrateMessage *m) {}
    void B::quit() {
        CkAbort("Expected C::quit to be called as a virtual method");
    }
} }

#include "B.def.h"
