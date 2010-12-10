/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _TEAMLB_H_
#define _TEAMLB_H_

#include "CentralLB.h"
#include "TeamLB.decl.h"

void CreateTeamLB();
BaseLB * AllocateTeamLB();

class TeamLB : public CentralLB {
  public:
    TeamLB(const CkLBOptions &);
    TeamLB(CkMigrateMessage *m):CentralLB(m) { lbname = "TeamLB"; }

    void work(LDStats* stats);
    void pup(PUP::er &p) { CentralLB::pup(p); }

  private:
    int teamSize;
    int numberTeams;

    CmiBool QueryBalanceNow(int step) { return CmiTrue; }
};

#endif /* _TEAMLB_H_ */

/*@}*/
