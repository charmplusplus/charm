#include "iotest.decl.h"

/* readonly */ CkGroupID mgr;

class Main : public CBase_Main {
  CProxy_test testers;
  int n;
  Ck::IO::File f;
public:
  Main(CkArgMsg *m) {
    n = atoi(m->argv[1]);
    Ck::IO::Options opts;
    opts.peStripe = 200;
    opts.writeStripe = 1;

    Ck::IO::open("test", CkCallback(CkIndex_Main::ready(NULL), thisProxy), opts);

    CkPrintf("Main ran\n");
  }

  void ready(Ck::IO::FileReadyMsg *m) {
    f = m->file;
    Ck::IO::startSession(f, 10*n, 0,
                         CkCallback(CkIndex_Main::start_write(0), thisProxy),
			 CkCallback(CkIndex_Main::test_written(), thisProxy));
    delete m;
    CkPrintf("Main saw file ready\n");
  }

  void start_write(Ck::IO::SessionReadyMsg *m) {
    testers = CProxy_test::ckNew(m->session, n);
    CkPrintf("Main saw session ready\n");
  }

  void test_written() {
    CkPrintf("Main saw write done\n");
    // Read file and validate contents

    CkExit();
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
