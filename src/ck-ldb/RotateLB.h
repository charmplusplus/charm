#ifndef _ROTATELB_H_
#define _ROTATELB_H_

#include "CentralLB.h"

void CreateRotateLB ();

class RotateLB : public CBase_RotateLB
{
  public:
    RotateLB (const CkLBOptions &opt);
    RotateLB (CkMigrateMessage *m) : CBase_RotateLB (m) { };

    void work(LDStats *stats);
    void pup (PUP::er &p) { }

  private:
    bool QueryBalanceNow (int step);
};

#endif /* _ROTATELB_H_ */
