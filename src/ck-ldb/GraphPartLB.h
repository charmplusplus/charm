/** \file GraphPartLB.h
 *  Author: Abhinav S Bhatele
 *  Date Created: September 3rd, 2010
 *  E-mail: bhatele@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */
/*@{*/

#ifndef _GRAPHPARTLB_H_
#define _GRAPHPARTLB_H_

#include "CentralLB.h"
#include "GraphPartLB.decl.h"

void CreateGraphPartLB();

class GraphPartLB : public CentralLB {
  public:
    GraphPartLB(const CkLBOptions &opt);
    GraphPartLB(CkMigrateMessage *m) : CentralLB (m) { };

    void work(LDStats *stats);
    void pup(PUP::er &p) { CentralLB::pup(p); }

  private:
    CmiBool QueryBalanceNow(int _step);
};

#endif /* _GRAPHPARTLB_H_ */

/*@}*/
