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
};

#include "test1.def.h"
