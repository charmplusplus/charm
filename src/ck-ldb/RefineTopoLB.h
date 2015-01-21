#ifndef _REFINETOPOLB_H_
#define _REINETOPOLB_H_

#include "CentralLB.h"
#include "TopoLB.h"
#include "topology.h"

#ifndef INFTY
#define INFTY 999999999
#endif

void CreateTopoLB ();

class RefineTopoLB : public CBase_RefineTopoLB
{
  public:
    RefineTopoLB (const CkLBOptions &opt);
    RefineTopoLB (CkMigrateMessage *m) : CBase_RefineTopoLB (m) { };
  
    void work (LDStats *stats);
    void pup (PUP::er &p) { }
    	
    //LBTopolog *topo;
  
  protected:
    double getCpartHopBytes(int cpart, int proc, int count);
    double findSwapGain(int cpart1, int cpart2, int n_pes);
    //double getInterMedHopBytes(CentralLB::LDStats *stats,int count, int *newmap);
    bool QueryBalanceNow (int step);
    void updateCommUA(int count);
}; 


#endif /* _TOPOLB_H_ */
