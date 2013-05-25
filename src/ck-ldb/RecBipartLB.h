/** \file RecBipartLB.h
 *  Author: Swapnil Ghike
 *  Date Created:
 *  E-mail:ghike2@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#ifndef _RECBIPARTLB_H_
#define _RECBIPARTLB_H_

#include "CentralLB.h"
#include "RecBipartLB.decl.h"

class RecBipartLB : public CentralLB {
  public:
    RecBipartLB(const CkLBOptions &opt);
    RecBipartLB(CkMigrateMessage *m) : CentralLB (m) { };

    void work(LDStats *stats);
    void pup(PUP::er &p) { CentralLB::pup(p); }

  private:
    bool QueryBalanceNow(int _step);
};

#endif /* _RECBIPARTLB_H_ */

/*@}*/
