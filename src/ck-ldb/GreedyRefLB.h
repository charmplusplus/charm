/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GREEDYREFLB_H_
#define _GREEDYREFLB_H_

#include "CentralLB.h"
#include "GreedyLB.h"
#include "Refiner.h"
#include "GreedyRefLB.decl.h"

void CreateGreedyRefLB();

class GreedyRefLB : public GreedyLB {
public:
  GreedyRefLB();
  GreedyRefLB(CkMigrateMessage *m):GreedyLB(m) {}
private:
  void work(CentralLB::LDStats* stats, int count);
};

#endif /* _GREEDYREFLB_H_ */

/*@}*/
