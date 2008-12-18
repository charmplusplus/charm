#ifndef __B_H__
#define __B_H__
#include "A.h"
#include "B.decl.h"

namespace Base { namespace Derived {
    class B : public CBase_B {
      public: 
        B();
        B(CkMigrateMessage *m);
        void quit();
    };
} }

#endif
