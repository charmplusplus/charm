// pgm.decl.h is generated from pgm.ci
#include "pgm.decl.h"

// define Charm++'s main Chare
// Main Chare only lives on Processor 0
class main : public CBase_main
{
public:
  main(CkArgMsg *m);
  void results(CkReductionMsg *msg);
};

// define type of an array of Chares
class integrate : public CBase_integrate 
{
public:
  integrate(CkMigrateMessage *m) {}
  integrate(double xStart,double dx, int nSlices);
};

