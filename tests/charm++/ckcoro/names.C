// See names.ci: compiled to check the code charmxi generates, not run.
#include "names.decl.h"

int coroutine;

class inbox : public CMessage_inbox {};

class Names : public CBase_Names {
  Names_SDAG_CODE
 public:
  Names() {}
  Names(CkMigrateMessage *m) : CBase_Names(m) {}
  void take(inbox *m) { delete m; }
  void put(int) {}
};

#include "names.def.h"
