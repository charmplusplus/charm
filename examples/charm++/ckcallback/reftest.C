#include "reftest.decl.h"

/*readonly*/ size_t magic_number;
/*readonly*/ size_t max_iter;

class Main : public CBase_Main {
public:
  Main_SDAG_CODE
  size_t order { 0 };

  Main(CkArgMsg* msg) {
    delete msg;
    magic_number = 29;
    max_iter = 1000;
    thisProxy.caller();
    thisProxy.recv_in_order();
  }

  void caller() {
    for (auto order = 0; order < max_iter; order++) {
      CkCallback cb(CkIndex_Main::callee(NULL), thisProxy);
      cb.setRefnum(order + magic_number);
      CProxy_Bounce::ckNew(cb);
    }
  }
};

class Bounce : public CBase_Bounce {
public:
  // send the callback back
  Bounce(CkCallback cb) {
    cb.send();
  }
};

#include "reftest.def.h"
