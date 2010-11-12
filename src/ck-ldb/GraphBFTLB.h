/** \file GraphBFTLB.h
 *  Author: Abhinav S Bhatele
 *  Date Created: November 10th, 2010
 *  E-mail: bhatele@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#ifndef _GRAPHBFTLB_H_
#define _GRAPHBFTLB_H_

#include "CentralLB.h"
#include "GraphBFTLB.decl.h"

void CreateGraphBFTLB();

class GraphBFTLB : public CentralLB {
  public:
    GraphBFTLB(const CkLBOptions &opt);
    GraphBFTLB(CkMigrateMessage *m) : CentralLB (m) { };

    void work(LDStats *stats);
    void pup(PUP::er &p) { CentralLB::pup(p); }

  private:
    CmiBool QueryBalanceNow(int _step);
};

#endif /* _GRAPHBFTLB_H_ */

/*@}*/
