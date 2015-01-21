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

class GraphBFTLB : public CBase_GraphBFTLB {
  public:
    GraphBFTLB(const CkLBOptions &opt);
    GraphBFTLB(CkMigrateMessage *m) : CBase_GraphBFTLB (m) { };

    void work(LDStats *stats);
    void pup(PUP::er &p) { }

  private:
    bool QueryBalanceNow(int _step);
};

#endif /* _GRAPHBFTLB_H_ */

/*@}*/
