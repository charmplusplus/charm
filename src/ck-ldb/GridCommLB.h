#ifndef _GRIDCOMMLB_H_
#define _GRIDCOMMLB_H_

#include <limits.h>
#include <math.h>
#include "CentralLB.h"

#define CK_LDB_GRIDCOMMLB_MODE 0
#define CK_LDB_GRIDCOMMLB_BACKGROUND_LOAD 1
#define CK_LDB_GRIDCOMMLB_LOAD_TOLERANCE 0.10

#ifndef MAXINT
#define MAXINT 2147483647
#endif

#ifndef MAXDOUBLE
#define MAXDOUBLE 1e10
#endif

void CreateGridCommLB ();

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

class GridCommLB : public CBase_GridCommLB
{
  public:
    GridCommLB (const CkLBOptions &);
    GridCommLB (CkMigrateMessage *msg);

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
    void Map_NonMigratable_Objects_To_PEs ();
    void Map_Migratable_Objects_To_PEs (int cluster);
    int Find_Maximum_Object (int cluster);
    int Find_Minimum_PE (int cluster);
    void Assign_Object_To_PE (int target_object, int target_pe);

    int CK_LDB_GridCommLB_Mode;
    int CK_LDB_GridCommLB_Background_Load;
    double CK_LDB_GridCommLB_Load_Tolerance;

    int Num_PEs;
    int Num_Objects;
    int Num_Clusters;
    PE_Data_T *PE_Data;
    Object_Data_T *Object_Data;
};

#endif
