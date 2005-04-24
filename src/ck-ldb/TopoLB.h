#ifndef _TOPOLB_H_
#define _TOPOLB_H_

#include "CentralLB.h"
#include "topology.h"

#ifndef INFTY
#define INFTY 999999999
#endif

void CreateTopoLB ();

class TopoLB : public CentralLB
{
  public:
    TopoLB (const CkLBOptions &opt);
    TopoLB (CkMigrateMessage *m) : CentralLB (m) { };
  
    void work (CentralLB::LDStats *stats, int count);
    void pup (PUP::er &p) { CentralLB::pup(p); }
    	
    LBTopology			*topo;
  
  private:

    double **dist;
    double **comm;
    double *commUA;
    double **hopBytes;
    bool *pfree;
    bool *cfree;
    int *assign;
    
    void computePartitions(CentralLB::LDStats *stats,int count,int *newmap);
    void allocateDataStructures(int num_procs);
    void freeDataStructures(int num_procs);
    void initDataStructures(CentralLB::LDStats *stats,int count,int *newmap);
    void printDataStructures(int num_procs, int num_objs, int *newmap);
    double getHopBytes(CentralLB::LDStats *stats,int count,CkVec<int>obj_to_proc);
    
    CmiBool QueryBalanceNow (int step);
}; 


#endif /* _TOPOLB_H_ */
