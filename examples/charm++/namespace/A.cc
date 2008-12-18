#include "A.h"

namespace Base {
    A::A() {}
    A::A(CkMigrateMessage *m) {}
    void A::sayHello() {
        CkPrintf("Hello\n");
    }
}

#include "A.def.h"
