#include "allGather.hh"

#include "user.decl.h"
class start : public CBase_start
{
private:
  CProxy_simBox sim;
  CProxy_AllGather AllGather;

public:
  start(CkArgMsg* msg);

  void fini();
};

class simBox : public CBase_simBox
{
private:
  CProxy_start startProxy;
  int k;
  int n;
  int d;
  long int* data;
  long int* result;

public:
  simBox(CProxy_start startProxy, int k, int n, int d);

  void begin(CProxy_AllGather AllGather);

  void done(allGatherMsg* msg);
};
