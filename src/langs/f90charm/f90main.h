
#include "f90main.decl.h"

class f90main: public Chare
{
  int count;
  public:
  f90main(CkArgMsg *);
  void done();
};

