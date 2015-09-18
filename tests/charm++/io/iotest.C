#include "iotest.decl.h"
#include <vector>

class Main : public CBase_Main {
  Main_SDAG_CODE

  CProxy_test testers;
  int n, numdone;
  std::vector<Ck::IO::File> f;
public:
  Main(CkArgMsg *m) {
    numdone = 0;
    n = atoi(m->argv[1]);

    f.resize(6);
    for (int i = 0; i < f.size(); ++i)
      thisProxy.run(4*i);

    CkPrintf("Main ran\n");
    delete m;
  }

  void iterDone() {
    numdone++;
    if (numdone == f.size())
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
