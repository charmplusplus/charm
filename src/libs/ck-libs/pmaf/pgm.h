#include "Pgm.decl.h"

CkChareID mainhandle;

class main : public Chare {
  int numChunks;
  char filename[30];
public:
  main(CkArgMsg *m);
  main(CkMigrateMessage *) {};
  void readMesh();
};
