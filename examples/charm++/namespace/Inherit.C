#include "C.h"
#include "Inherit.h"

Driver::Driver(CkArgMsg* args) {
    using Base::Derived::CProxy_B;
    CProxy_B b = CProxy_B::ckNew();
    b.sayHello();
    b = CProxy_C::ckNew();
    b.quit();
}

#include "Inherit.def.h"
