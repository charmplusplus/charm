#ifndef _BLOCKLB_H_
#define _BLOCKLB_H_

#include "CentralLB.h"

void CreateBlockLB ();

class BlockLB : public CentralLB
{
  public:
    BlockLB (const CkLBOptions &opt);
    BlockLB (CkMigrateMessage *m) : CentralLB (m) { };

    void work (LDStats *stats);

    void pup (PUP::er &p) { CentralLB::pup(p); }

  private:
    CmiBool QueryBalanceNow (int step);
};

#endif /* _BLOCKLB_H_ */
