#ifndef _ROTATELB_H_
#define _ROTATELB_H_

#include "CentralLB.h"

void CreateRotateLB ();

class RotateLB : public CentralLB
{
  public:
    RotateLB (const CkLBOptions &opt);
    RotateLB (CkMigrateMessage *m) : CentralLB (m) { };

    void work(LDStats *stats);

    void pup (PUP::er &p) { CentralLB::pup(p); }

  private:
    CmiBool QueryBalanceNow (int step);
};

#endif /* _ROTATELB_H_ */
