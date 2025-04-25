#include "user.decl.h"

#include "allGather.hh"

class start : public CBase_start
{
private:
  int n;
  int k;
  int x;
  int y;
  CProxy_simBox sim;
  CProxy_AllGather AllGather;

public:
  start(CkArgMsg* msg);

  void fini(int numDone);
};

class simBox : public CBase_simBox
{
private:
  CProxy_start startProxy;
  int k;
  int n;
  int x;
  int y;
  long int* data;
  long int* result;

public:
  simBox(CProxy_start startProxy, int k, int n, int x, int y);

  void begin(CProxy_AllGather AllGather);

  void done(allGatherMsg* msg);
};
