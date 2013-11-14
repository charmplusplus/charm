#include "sdagCase.decl.h"

#define NUM_ELMS 40

/*readonly*/ CProxy_Array arr;
/*readonly*/ CProxy_Main mainProxy;

#include <map>

using namespace std;

struct Counts {
  int a, b, c;
  bool done;

  Counts() : a(0), b(0), c(0), done(false) { }
};

struct Main : public CBase_Main {
  int numCompleted;
  map<int, Counts> counts;

  Main(CkArgMsg* m) {
    delete m;
    arr = CProxy_Array::ckNew(NUM_ELMS);
    numCompleted = 0;
    mainProxy = thisProxy;
    CkStartQD(CkCallback(CkIndex_Main::sendMsgs(), thisProxy));
  }
  void sendMsgs() {
    srand(10129829);
    while (numCompleted < NUM_ELMS) {
      for (int i = 0; i < NUM_ELMS; i++) {
        if (!counts[i].done) {
          if (rand() % 10 != 1) {
            switch (rand() % 3) {
            case 0:
              arr[i].a();
              counts[i].a++;
              break;
            case 1:
              arr[i].b();
              counts[i].b++;
              break;
            case 2:
              arr[i].c();
              counts[i].c++;
              break;
            }
          } else {
            // send done msg
            counts[i].done = true;
            numCompleted++;
            arr[i].recvStats(counts[i].a, counts[i].b, counts[i].c);
          }
        }
      }
    }
    CkPrintf("in loop, numCompleted = %d\n", numCompleted);
    CkStartQD(CkCallback(CkIndex_Array::d(), arr));
  }
  void finished() {
    CkExit();
  }
};

struct Array : public CBase_Array {
  Array_SDAG_CODE

  bool recvMore;
  int acount, bcount, ccount, dcount;

  Array(CkMigrateMessage*) { }

  Array() {
    recvMore = true;
    acount = bcount = ccount = dcount = 0;
    thisProxy[thisIndex].main();
  }
};

#include "sdagCase.def.h"
