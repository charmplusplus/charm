/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _NEIGHBORCOMMLB_H_
#define _NEIGHBORCOMMLB_H_

#include <math.h>

#include "NborBaseLB.h"
#include "NeighborCommLB.decl.h"

void CreateNeighborCommLB();

class NeighborCommLB : public CBase_NeighborCommLB {
public:
  NeighborCommLB(const CkLBOptions &);
  NeighborCommLB(CkMigrateMessage *m):CBase_NeighborCommLB(m) {}
private:
  bool QueryBalanceNow(int step) { return true; };
  virtual int max_neighbors() {
    return (CkNumPes() > 5) ? 4 : (CkNumPes()-1);
  };
  virtual int num_neighbors() {
    return (CkNumPes() > 5) ? 4 : (CkNumPes()-1);
  };
  virtual void neighbors(int* _n) {
    const int me = CkMyPe();
    const int npe = CkNumPes();
    if (npe > 1)
      _n[0] = (me + npe - 1) % npe;
    if (npe > 2)
      _n[1] = (me + 1) % npe;

    int bigstep = (npe - 1) / 3 + 1;
    if (bigstep == 1) bigstep++;

    if (npe > 3)
      _n[2] = (me + bigstep) % npe;
    if (npe > 4)
      _n[3] = (me + npe - bigstep) % npe;
  };

  LBMigrateMsg* Strategy(NborBaseLB::LDStats* stats, int n_nbrs);
};

#endif /* _NeighborCommLB_H_ */

/*@}*/
