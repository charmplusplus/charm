/** \file ScotchTopoLB.h
 *  Authors: Abhinav S Bhatele (bhatele@illinois.edu)
 *           Sebastien Fourestier (fouresti@labri.fr)
 *  Date Created: November 25th, 2010
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#ifndef _SCOTCHLB_H_
#define _SCOTCHLB_H_

#include "CentralLB.h"
#include "ScotchTopoLB.decl.h"

void CreateScotchTopoLB();

class ScotchTopoLB : public CentralLB {
  public:
    ScotchTopoLB(const CkLBOptions &opt);
    ScotchTopoLB(CkMigrateMessage *m) : CentralLB (m) { };

    void work(LDStats *stats);
    void pup(PUP::er &p) { CentralLB::pup(p); }

  private:
    CmiBool QueryBalanceNow(int _step);
};

#endif /* _GRAPHPARTLB_H_ */

/*@}*/
