/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _RANDCENTLB_H_
#define _RANDCENTLB_H_

#include "CentralLB.h"
#include "RandCentLB.decl.h"

void CreateRandCentLB();

class RandCentLB : public CBase_RandCentLB {
public:
  RandCentLB(const CkLBOptions &opt);
  RandCentLB(CkMigrateMessage *m) : CBase_RandCentLB(m) { lbname = "RandCentLB"; }
  void pup(PUP::er &p){ }

  void work(LDStats* stats);
private:
  bool QueryBalanceNow(int step);
};

#endif /* _RANDCENTLB_H_ */

/*@}*/
