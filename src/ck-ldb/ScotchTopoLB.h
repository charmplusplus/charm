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

class ScotchTopoLB : public CBase_ScotchTopoLB {
  public:
    ScotchTopoLB(const CkLBOptions &opt);
    ScotchTopoLB(CkMigrateMessage *m) : CBase_ScotchTopoLB (m) { };

    void work(LDStats *stats);
    void pup(PUP::er &p) {  }

  private:
    bool QueryBalanceNow(int _step);
};

#endif /* _GRAPHPARTLB_H_ */

/*@}*/
