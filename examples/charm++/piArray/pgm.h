#include "pgm.decl.h"

class main : public Chare
{
private:
  int responders;
  int  count;
  int ns, nc;
  double starttime, endtime;
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m);
  void results(int c);
};

class piPart : public ArrayElement1D 
{
public:
  piPart(CkMigrateMessage *m) {}
  piPart(void);
  void compute(int ns);
};
