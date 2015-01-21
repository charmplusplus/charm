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

class TeamLB : public CBase_TeamLB {
  public:
    TeamLB(const CkLBOptions &);
    TeamLB(CkMigrateMessage *m):CBase_TeamLB(m) { lbname = "TeamLB"; }

    void work(LDStats* stats);
    void pup(PUP::er &p) {  }

  private:
    int teamSize;
    int numberTeams;

    bool QueryBalanceNow(int step) { return true; }
};

#endif /* _TEAMLB_H_ */

/*@}*/
