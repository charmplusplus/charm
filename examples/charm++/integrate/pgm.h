#include "pgm.decl.h"

class main : public Chare
{
private:
  int count; //Number of partial results that have not arrived yet
  double integral; //Partial value of function integral
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m);
  void results(double partialIntegral);
};

class integrate : public Chare 
{
private:

public:
  integrate(CkMigrateMessage *m) {}
  integrate(double xStart,double dx, int nSlices);
};

