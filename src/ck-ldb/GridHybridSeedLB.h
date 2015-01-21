#ifndef _GRIDHYBRIDSEEDLB_H_
#define _GRIDHYBRIDSEEDLB_H_

#include "CentralLB.h"

#define CK_LDB_GRIDHYBRIDSEEDLB_MODE 0
#define CK_LDB_GRIDHYBRIDSEEDLB_BACKGROUND_LOAD 1
#define CK_LDB_GRIDHYBRIDSEEDLB_LOAD_TOLERANCE 0.10

#ifndef MAXINT
#define MAXINT 2147483647
#endif

#ifndef MAXDOUBLE
#define MAXDOUBLE 1e10
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

void CreateGridHybridSeedLB ();

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
    int secondary_index;
};

class Cluster_Data_T
{
  public:
    int num_pes;
    double total_cpu_power;
    double scaled_cpu_power;
};

class GridHybridSeedLB : public CBase_GridHybridSeedLB
{
  public:
    GridHybridSeedLB (const CkLBOptions &opt);
    GridHybridSeedLB (CkMigrateMessage *msg);

    bool QueryBalanceNow (int step);
    void work (LDStats *stats);
    void pup (PUP::er &p) { }

  private:
    int Get_Cluster (int pe);
    void Initialize_PE_Data (CentralLB::LDStats *stats);
    int Available_PE_Count ();
    int Compute_Number_Of_Clusters ();
    void Initialize_Object_Data (CentralLB::LDStats *stats);
    int Compute_Migratable_Object_Count ();
    void Initialize_Cluster_Data ();
    void Initialize_Communication_Matrix (CentralLB::LDStats *stats);
    void Partition_Objects_Into_Clusters (CentralLB::LDStats *stats);
    void Examine_InterObject_Messages (CentralLB::LDStats *stats);
    void Map_NonMigratable_Objects_To_PEs ();
    int Find_Maximum_Object (int cluster);
    int Find_Maximum_Border_Object (int cluster);
    int Find_Maximum_Object_From_Seeds (int pe);
    int Find_Maximum_Border_Object_From_Seeds (int pe);
    int Compute_Communication_Events (int obj1, int obj2);
    int Find_Minimum_PE (int cluster);
    void Assign_Object_To_PE (int target_object, int target_pe);

    int CK_LDB_GridHybridSeedLB_Mode;
    int CK_LDB_GridHybridSeedLB_Background_Load;
    double CK_LDB_GridHybridSeedLB_Load_Tolerance;

    int Num_PEs;
    int Num_Objects;
    int Num_Migratable_Objects;
    int Num_Clusters;
    PE_Data_T *PE_Data;
    Object_Data_T *Object_Data;
    Cluster_Data_T *Cluster_Data;
    int *Migratable_Objects;
    int **Communication_Matrix;
};

#endif
