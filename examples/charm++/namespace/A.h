#ifndef __A_H__
#define __A_H__
#include "A.decl.h"

namespace Base {
    class A : public CBase_A {
      public: 
        A(CkMigrateMessage *m);
        A();
        virtual void sayHello();
    };
}

#endif
