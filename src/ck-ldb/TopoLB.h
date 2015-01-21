#ifndef _TOPOLB_H_
#define _TOPOLB_H_

#include "CentralLB.h"
#include "topology.h"

#ifndef INFTY
#define INFTY 999999999
#endif





extern "C" void METIS_PartGraphRecursive (int*, int*, int*, int*, int*, int*,
					  int*, int*, int*, int*, int*);

extern "C" void METIS_PartGraphKway (int*, int*, int*, int*, int*, int*,
				     int*, int*, int*, int*, int*);

extern "C" void METIS_PartGraphVKway (int*, int*, int*, int*, int*, int*,
				      int*, int*, int*, int*, int*);

extern "C" void METIS_WPartGraphRecursive (int*, int*, int*, int*,
					   int*, int*, int*, int*,
					   float*, int*, int*, int*);

extern "C" void METIS_WPartGraphKway (int*, int*, int*, int*,
				      int*, int*, int*, int*,
				      float*, int*, int*, int*);

extern "C" void METIS_mCPartGraphRecursive (int*, int*, int*, int*,
					    int*, int*, int*, int*,
					    int*, int*, int*, int*);

extern "C" void METIS_mCPartGraphKway (int*, int*, int*, int*, int*,
				       int*, int*, int*, int*, int*,
				       int*, int*, int*);






void CreateTopoLB ();

class TopoLB : public CBase_TopoLB
{
  public:
    TopoLB (const CkLBOptions &opt);
    TopoLB (CkMigrateMessage *m) : CBase_TopoLB (m) { };
  
    void work (LDStats *stats);
   // void work_fromFile (char *filename);
    void pup (PUP::er &p) { }
    	
    LBTopology			*topo;
  
  protected:

    double **dist;
    double **comm;
    double *commUA;
    double **hopBytes;
    bool *pfree;
    bool *cfree;
    int *assign;
    double total_comm;
    
    virtual void computePartitions(CentralLB::LDStats *stats,int count,int *newmap);
    virtual void allocateDataStructures(int num_procs);
    virtual void freeDataStructures(int num_procs);
    virtual void initDataStructures(CentralLB::LDStats *stats,int count,int *newmap);
    virtual void printDataStructures(int num_procs, int num_objs, int *newmap);
    virtual double getHopBytes(CentralLB::LDStats *stats,int count,CkVec<int>obj_to_proc);
    virtual double getHopBytesNew(int *assign_map, int count);
    void performMapping(int *newmap, int count);
    
    bool QueryBalanceNow (int step);
}; 


#endif /* _TOPOLB_H_ */
