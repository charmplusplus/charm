/** \file ScotchRefineLB.h
 *  Date Created: November 25th, 2010
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#ifndef _SCOTCHREFINELB_H_
#define _SCOTCHREFINELB_H_

#include "CentralLB.h"
#include "ScotchRefineLB.decl.h"

void CreateScotchRefineLB();

class ScotchRefineLB : public CBase_ScotchRefineLB {
  public:
    ScotchRefineLB(const CkLBOptions &opt);
    ScotchRefineLB(CkMigrateMessage *m) : CBase_ScotchRefineLB (m) { };

    void work(LDStats *stats);
    void pup(PUP::er &p) {  }

  private:
    bool QueryBalanceNow(int _step);
};

#endif /* _SCOTCHREFINELB_H_ */

/*@}*/
