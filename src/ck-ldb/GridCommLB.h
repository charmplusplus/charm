#ifndef _GRIDCOMMLB_H_
#define _GRIDCOMMLB_H_

#include "CentralLB.h"

void CreateGridCommLB ();

class PE_Data_T
{
  public:
    CmiBool available;
    int cluster;
    int num_objs;
    int num_lan_objs;
    int num_lan_msgs;
    int num_wan_objs;
    int num_wan_msgs;
};

class Object_Data_T
{
  public:
    CmiBool migratable;
    int cluster;
    int from_pe;
    int to_pe;
    int num_lan_msgs;
    int num_wan_msgs;
};

class GridCommLB : public CentralLB
{
  public:
    GridCommLB (const CkLBOptions &);
    GridCommLB (CkMigrateMessage *m);

    void work (CentralLB::LDStats *stats, int count);

    void pup (PUP::er &p) { CentralLB::pup (p); }

  private:
    int Get_Cluster (int pe);
    int Find_Maximum_WAN_Object (int cluster);
    int Find_Minimum_WAN_PE (int cluster);
    void Assign_Object_To_PE (int target_object, int target_pe);
    CmiBool QueryBalanceNow (int step);

    int Num_PEs;
    int Num_Objects;
    PE_Data_T *PE_Data;
    Object_Data_T *Object_Data;
};

#endif
