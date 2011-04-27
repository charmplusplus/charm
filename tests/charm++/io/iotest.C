#include "iotest.decl.h"

/* readonly */ CkGroupID mgr;

class Main : public CBase_Main {
  CProxy_test testers;
  int n;
public:
  Main(CkArgMsg *m) {
    n = atoi(m->argv[1]);
    mgr = Ck::IO::CProxy_Manager::ckNew();
    CkPrintf("Main created group\n");
    Ck::IO::Manager *iomgr = (Ck::IO::Manager *)CkLocalBranch(mgr);
    Ck::IO::Options opts;
    opts.peStripe = 200;
    opts.writeStripe = 1;
    iomgr->prepareOutput("test", 10*n, CkCallback(CkIndex_Main::ready(NULL), thisProxy),
			 CkCallback(), opts);
    CkPrintf("Main ran\n");
  }

  void ready(Ck::IO::FileReadyMsg *m) {
    testers = CProxy_test::ckNew(m->token, n);
    CkCallback cb(CkIndex_Main::test_written(), thisProxy);
    CkStartQD(cb);
    CkPrintf("Main saw file ready\n");
  }

  void test_written() {
    CkPrintf("Main see write done\n");
    // Read file and validate contents

    CkExit();
  }
};

struct test : public CBase_test {
  test(Ck::IO::Token token) {
    char out[11];
    sprintf(out, "%10d", thisIndex);
    ((Ck::IO::Manager *)CkLocalBranch(mgr))->write(token, out, 10, 10*thisIndex);
  }
  test(CkMigrateMessage *m) {}
};


#include "iotest.def.h"
