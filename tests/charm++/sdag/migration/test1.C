#include <cstdlib>
#include <cmath>

#include "test1.decl.h"

/*readonly*/ CProxy_Main mainProxy;
using namespace std;

int nrows = 5;
int ncols = 5;

class Main : public CBase_Main {
public:
  CProxy_Cell arr;

  Main(CkArgMsg* m) {
    delete m;

    CkPrintf("Running Parallel on %d processors for %d elements\n",
	     CkNumPes(), nrows * ncols);

    mainProxy = thishandle;

    arr = CProxy_Cell::ckNew(nrows, ncols);
    arr.finished(100);
    arr.process();
  }

  void end() {
    CkPrintf("Test was successful!\n");
    CkExit();
  }

};

class Cell : public CBase_Cell {
  Cell_SDAG_CODE

public:
  int val;

  Cell() : val(thisIndex.x * 10 + thisIndex.y) { }

  void pup(PUP::er &p) {
    CkPrintf("called PUP for cell %s\n", p.isPacking() ? "packing" : "unpacking or sizing");
    p | val;
  }

  Cell(CkMigrateMessage *m) { }

  // It is not currently safe to call migrateMe() from inside an SDAG
  // entry method, so we wrap the call to migrateMe() in a non-SDAG
  // entry method for now. See redmine issue #480 for more details.
  void callMigrateMe(int pe) {
    migrateMe(pe);
  }
};

#include "test1.def.h"
