#include "Pgm.decl.h"

CkChareID mainhandle;

class main : public Chare {
public:
  main(CkArgMsg *m);
  main(CkMigrateMessage *) {};
  void performRefinements();
  void readMesh(char *filename);
};
