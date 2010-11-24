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
private:
	int teamSize;
	int numberTeams;
  CmiBool QueryBalanceNow(int step) { return CmiTrue; }
  void work(LDStats* stats);
};

#define WEIGHTED 1
#define MULTI_CONSTRAINT 2

#endif /* _TEAMLB_H_ */

/*@}*/
