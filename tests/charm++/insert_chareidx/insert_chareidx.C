// Array elements inserted on the PE of a plain chare (Driver), from that
// chare's entry method, then destroyed or migrated away. An element constructed
// while currentChareIdx names Driver must not take Driver's chare_objs slot, so
// Driver must still receive its messages afterwards.

#include "insert_chareidx.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Elem elemProxy;

static const int numElems = 8;
static const int numPings = 5;

class Main : public CBase_Main {
  CProxy_Driver driver;
  int pongs = 0;

public:
  Main(CkArgMsg* m) {
    delete m;
    mainProxy = thisProxy;
    elemProxy = CProxy_Elem::ckNew();
    driver = CProxy_Driver::ckNew();
    driver.insertSome(numElems);
  }

  void inserted() {
    CkStartQD(CkCallback(CkIndex_Main::constructed(), thisProxy));
  }

  void constructed() {
    for (int i = 0; i < numElems; i++) elemProxy[i].leave();
    CkStartQD(CkCallback(CkIndex_Main::left(), thisProxy));
  }

  void left() {
    for (int i = 0; i < numPings; i++) driver.ping();
  }

  void pong(int count) {
    pongs++;
    if (pongs < numPings) return;
    if (count != numPings)
      CkAbort("insert_chareidx: Driver counted %d pings, expected %d\n", count, numPings);
    CkPrintf("insert_chareidx: PASS on %d PEs\n", CkNumPes());
    CkExit();
  }
};

class Driver : public CBase_Driver {
  int pings = 0;

public:
  Driver() {}

  void insertSome(int k) {
    for (int i = 0; i < k; i++) elemProxy[i].insert(CkMyPe(), CkMyPe());
    elemProxy.doneInserting();
    mainProxy.inserted();
  }

  void ping() { mainProxy.pong(++pings); }
};

class Elem : public CBase_Elem {
  int insertPe;

public:
  Elem(int pe) : insertPe(pe) {
    if (CkMyPe() != insertPe) CkAbort("Elem %d constructed on PE %d, not %d\n", thisIndex, CkMyPe(), insertPe);
  }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}

  void pup(PUP::er& p) { p | insertPe; }

  // Odd elements migrate to the next PE, even ones are destroyed.
  void leave() {
    if (CkNumPes() > 1 && thisIndex % 2 == 1)
      migrateMe((CkMyPe() + 1) % CkNumPes());
    else
      ckDestroy();
  }
};

#include "insert_chareidx.def.h"
