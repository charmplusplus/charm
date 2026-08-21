#ifndef __MAIN_H__
#define __MAIN_H__

#include "defs.h"

//central controller chare
class Main : public CBase_Main {
  private:
    Main_SDAG_CODE
    double startBenchmarkTime;

  public:
    Main(CkArgMsg* msg);
    Main(CkMigrateMessage* msg);
    void pup(PUP::er &p);
};
#endif
