#include "iotest.decl.h"

/* readonly */ CkGroupID mgr;

class Main : public CBase_Main {
  Main_SDAG_CODE

  CProxy_test testers;
  int n;
  Ck::IO::File f;
public:
  Main(CkArgMsg *m) {
    n = atoi(m->argv[1]);
    Ck::IO::Options opts;
    opts.peStripe = 200;
    opts.writeStripe = 1;
    CkCallback opened(CkIndex_Main::ready(NULL), thisProxy);
    opened.setRefnum(5);
    Ck::IO::open("test", opened, opts);

    CkPrintf("Main ran\n");
    thisProxy.run();
  }

};

struct test : public CBase_test {
  test(Ck::IO::Session token) {
    char out[11];
    sprintf(out, "%9d\n", thisIndex);
    Ck::IO::write(token, out, 10, 10*thisIndex);
  }
  test(CkMigrateMessage *m) {}
};


#include "iotest.def.h"
