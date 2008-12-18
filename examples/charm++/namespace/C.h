#ifndef __C_H__
#define __C_H__
#include "B.h"
#include "C.decl.h"

class C : public CBase_C {
  public: 
    C();
    C(CkMigrateMessage *m);
};

#endif
