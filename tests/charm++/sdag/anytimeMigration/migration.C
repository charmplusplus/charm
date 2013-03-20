#include "migration.decl.h"

#include <string.h>

#define NUM_ELEMS 500

struct Msg : public CMessage_Msg {
  int data;
  char* arr;

  Msg(int data, char* src)
    : data(data) {
    strcpy(arr, src);
  }
};

struct Main : public CBase_Main {
  Main(CkArgMsg *m) {
    CkPrintf("running SDAG migration test\n");
    CProxy_Test testProxy = CProxy_Test::ckNew(NUM_ELEMS);
    testProxy.wrapper(100, 200);
    for (int i = 0; i < NUM_ELEMS; i++) {
      char str[100];
      sprintf(str, "test %d", i);
      Msg* m = new (strlen(str) + 1) Msg(i, str);
      testProxy[i].method2(i * 2, i * 2 + 1);
      testProxy[i].method3(m);
      testProxy[i].methodA();
    }
    CkStartQD(CkCallback(CkIndex_Main::finished(), thisProxy));
  }

  void finished() {
    CkPrintf("SDAG migration test ran successfully!\n");
    CkExit();
  }
};

struct Test : public CBase_Test {
  Test_SDAG_CODE

  int data;

  Test()
    : data(thisIndex) {
    //CkPrintf("%d: constructing Test element %d\n", CkMyPe(), thisIndex);
    thisProxy[thisIndex].run();
    //thisProxy[thisIndex].method1();
  }
  Test(CkMigrateMessage*) { }

  void pup(PUP::er& p) {
    p | data;
  }

  void doMigrate(int pe) {
    if (pe != CkMyPe()) {
      //CkPrintf("calling migrateMe on Test[%d] to %d from %d\n", thisIndex, pe, CkMyPe());
      migrateMe(pe);
    }
  }

  void ckJustMigrated() {
    //CkPrintf("just migrated called on pe %d\n", CkMyPe());
    thisProxy[thisIndex].method1();
    thisProxy[thisIndex].methodB();
  }
};

#include "migration.def.h"
