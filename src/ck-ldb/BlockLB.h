#ifndef _BLOCKLB_H_
#define _BLOCKLB_H_

#include "CentralLB.h"

void CreateBlockLB ();

class BlockLB : public CBase_BlockLB
{
  public:
    BlockLB (const CkLBOptions &opt);
    BlockLB (CkMigrateMessage *m) : CBase_BlockLB (m) { };

    void work (LDStats *stats);
    void pup (PUP::er &p) { }

  private:
    bool QueryBalanceNow (int step);
};

#endif /* _BLOCKLB_H_ */
