#ifndef _GRIDCOMMREFINELB_H_
#define _GRIDCOMMREFINELB_H_

#include <limits.h>
#include "CentralLB.h"

#define CK_LDB_GRIDCOMMREFINELB_TOLERANCE 1.10

#ifndef MAXINT
#define MAXINT 2147483647
#endif

void CreateGridCommRefineLB ();

class PE_Data_T
{
  public:
    bool available;
    int cluster;
    int num_objs;
    int num_lan_objs;
    int num_lan_msgs;
    int num_wan_objs;
    int num_wan_msgs;
    double relative_speed;
    double scaled_load;
};

class Object_Data_T
{
  public:
    bool migratable;
    int cluster;
    int from_pe;
    int to_pe;
    int num_lan_msgs;
    int num_wan_msgs;
    double load;
};

class GridCommRefineLB : public CBase_GridCommRefineLB
{
  public:
    GridCommRefineLB (const CkLBOptions &);
    GridCommRefineLB (CkMigrateMessage *msg);

    bool QueryBalanceNow (int step);
    void work (LDStats *stats);
    void pup (PUP::er &p) { }

  private:
    int Get_Cluster (int pe);
    void Initialize_PE_Data (CentralLB::LDStats *stats);
    int Available_PE_Count ();
    int Compute_Number_Of_Clusters ();
    void Initialize_Object_Data (CentralLB::LDStats *stats);
    void Examine_InterObject_Messages (CentralLB::LDStats *stats);
    void Place_Objects_On_PEs ();
    void Remap_Objects_To_PEs (int cluster);
    int Find_Maximum_WAN_Object (int pe);
    int Find_Minimum_WAN_PE (int cluster);
    void Remove_Object_From_PE (int target_object, int target_pe);
    void Assign_Object_To_PE (int target_object, int target_pe);

    int Num_PEs;
    int Num_Objects;
    int Num_Clusters;
    PE_Data_T *PE_Data;
    Object_Data_T *Object_Data;
    double CK_LDB_GridCommRefineLB_Tolerance;
};

#endif
