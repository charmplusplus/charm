/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _NEIGHBORLB_H_
#define _NEIGHBORLB_H_

#include <math.h>

#include "NborBaseLB.h"
#include "NeighborLB.decl.h"

void CreateNeighborLB();

class NeighborLB : public CBase_NeighborLB {
public:
  NeighborLB(const CkLBOptions &);
  NeighborLB(CkMigrateMessage *m):CBase_NeighborLB(m) {}
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

#endif /* _NeighborLB_H_ */

/*@}*/
