#include "pgm.decl.h"

/*readonly*/ CProxy_Main mainProxy;

struct TestMsg : public CMessage_TestMsg { };

struct Main : CBase_Main {
  int numToSend;

  Main(CkArgMsg* m)
    : numToSend(40)
  {
    delete m;

    mainProxy = thisProxy;

    CProxy_Chare1 c1 = CProxy_Chare1::ckNew();

    for (int i = 0; i < numToSend; i += 4) {
      c1.prioMarshalling(10, & CkEntryOptions().setPriority(-1) );

      TestMsg* msg = new (8*sizeof(int)) TestMsg;
      *(int*)CkPriorityPtr(msg) = -2;
      CkSetQueueing(msg, CK_QUEUEING_IFIFO);
      c1.prioMessage(msg);

      CkEntryOptions opts;
      opts.setPriority(-1);
      opts.setQueueing(CK_QUEUEING_ILIFO);
      c1.prioVoidMessage(&opts);

      c1.normalPrio(20);
    }
  }

  void finished() {
    if (--numToSend == 0) CkExit();
  }
};

struct Chare1 : CBase_Chare1 {
  Chare1() {
    CkPrintf("Chare: created\n");
  }
  void prioMarshalling(int test) {
    CkPrintf("prioMarshalling arrived\n");
    mainProxy.finished();
  }
  void prioMessage(TestMsg* msg) {
    CkPrintf("prioMessage arrived\n");
    mainProxy.finished();
    delete msg;
  }
  void prioVoidMessage() {
    CkPrintf("prioVoidMessage arrived\n");
    mainProxy.finished();
  }
  void normalPrio(int test2) {
    CkPrintf("normalPrio arrived\n");
    mainProxy.finished();
  }
};

#include "pgm.def.h"
