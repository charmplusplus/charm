#ifndef _WSLB_H_
#define _WSLB_H_

#include "NeighborLB.h"
#include "WSLB.decl.h"

void CreateWSLB();

class WSLB : public NeighborLB {
public:
  WSLB();
private:
  CmiBool QueryBalanceNow(int step) { return CmiTrue; };
  virtual int num_neighbors() {
    return (CkNumPes() > 5) ? 4 : (CkNumPes()-1);
  };
  virtual void neighbors(int* _n) {
    CkPrintf("[%d] Saving neighbors\n",CkMyPe());
    const int me = CkMyPe();
    const int npe = CkNumPes();
    if (npe > 1)
      _n[0] = (me + npe - 1) % npe;
    if (npe > 2)
      _n[1] = (me + 1) % npe;
    if (npe > 3)
      _n[2] = (me + 2) % npe;
    if (npe > 4)
      _n[3] = (me + npe - 2) % npe;
  };

  NLBMigrateMsg* Strategy(NeighborLB::LDStats* stats, int count);
};

#endif /* _WSLB_H_ */
