#include "B.h"
#include "Inherit.h"

Driver::Driver(CkArgMsg* args) {
    CProxy_B b = CProxy_B::ckNew();
    b.sayHello();
    b.quit();
}

#include "Inherit.def.h"
