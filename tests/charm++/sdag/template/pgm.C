#include "pgm.decl.h"
#define CK_TEMPLATES_ONLY
#include "pgm.def.h"
#undef CK_TEMPLATES_ONLY

#include <iostream>

class main : public CBase_main {
  main_SDAG_CODE;
  CProxy_tchare<int> cint;
  CProxy_tchare<double> cdouble;
public:
  main(CkArgMsg *m) {
    delete m;
    thisProxy.run();
  }
};

template <typename T>
class tchare : public CBase_tchare<T> {
  tchare_SDAG_CODE;
  T val;
  tchare(T arg) : val(arg) { }
};

#include "pgm.def.h"
