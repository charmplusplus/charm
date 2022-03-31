#include "charm++.h"
#include "idlework.decl.h"

class Test : public CBase_Test {
public:
  Test(CkMigrateMessage *m) {}
  Test();
  bool idleProgress();
  void registerIdleWork();
};

class Main : public CBase_Main {
public:
  Main(CkArgMsg *);
};
