#include "MiniNet.decl.h"

CkChareID mainhandle;

class main : public Chare {
public:
  main(CkArgMsg *m);
  main(CkMigrateMessage *) {};
};
