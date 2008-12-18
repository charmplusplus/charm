#include "C.h"
#include "Inherit.h"

Driver::Driver(CkArgMsg* args) {
    using Base::Derived::CProxy_B;
    CProxy_B b = CProxy_B::ckNew();
    b.sayHello();
    CProxy_C c = CProxy_C::ckNew();
    c.quit();
}

#include "Inherit.def.h"
