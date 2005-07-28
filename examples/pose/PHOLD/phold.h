#include "PHOLD.decl.h"

CkChareID mainhandle;

class main : public Chare {
  int *map;
public:
  main(CkArgMsg *m);
  main(CkMigrateMessage *) {};
  void buildMap(int numObjs, int dist);
  int getAnbr(int numObjs, int locale, int dest);
};
