#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;


class Main : public CBase_Main {

  public:
    Main(CkArgMsg* m) {

      //Process command-line arguments
      nElements = 5;
      if (m->argc > 1) nElements = atoi(m->argv[1]);
      delete m;

      //Start the computation
      CkPrintf("Running Hello on %d processors for %d elements\n", CkNumPes(), nElements);
      mainProxy = thisProxy;

      CProxy_Hello arr = CProxy_Hello::ckNew(nElements);
      char *msg = "Hello from Main";
      arr[0].saySomething(strlen(msg) + 1, msg, -1);
    };

    void done(void) {
      CkPrintf("All done\n");
      CkExit();
    };
};


class Hello : public CBase_Hello {

  // Declare the CkIndex_Hello class as a friend of this class so that the accelerated
  //   entry methods can access the member variables of this class
  friend class CkIndex_Hello;

  public:

    Hello() { }
    Hello(CkMigrateMessage *m) {}
    ~Hello() { }
  
    void saySomething_callback() {
      if (thisIndex < nElements - 1) {
        char msgBuf[128];
        int msgLen = sprintf(msgBuf, "Hello from %d", thisIndex) + 1;
        thisProxy[thisIndex+1].saySomething(msgLen, msgBuf, thisIndex);
      } else {
        mainProxy.done();
      }
    }
};


#include "hello.def.h"
