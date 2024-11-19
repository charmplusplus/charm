/*Distributed Graph Refinement Strategy*/
#ifndef _DISTLB_H_
#define _DISTLB_H_

#include "charm++.h"
#include "BaseLB.h"
#include "CentralLB.h"
#include "DistBaseLB.h"
#include "ckgraph.h"
#include "TopoManager.h"

#include "topology.h"
#include "ckheap.h"

#include <vector>
#include <unordered_map>
#include<queue>

#include "DiffusionLB.decl.h"

void CreateDiffusionLB();

/// for backward compatibility
typedef LBMigrateMsg NLBMigrateMsg;

class DiffusionLB : public CBase_DiffusionLB {
public:
    DiffusionLB_SDAG_CODE 
    DiffusionLB(const CkLBOptions &);
    DiffusionLB(CkMigrateMessage *m);
    ~DiffusionLB();
    void MigratedHelper(LDObjHandle h, int waitBarrier);
    void Migrated(LDObjHandle h, int waitBarrier=1);
    void createCommList();
    void AddNeighbor(int node);
    void findNeighbors(int do_again);
    void proposeNbor(int nborId);
    void okayNbor(int agree, int nborId);
    void statsAssembled();
    void startStrategy();
//    void notifyNeighbor(int isNbor, int node);
    void doneNborExng();
    void sortArr(long arr[], int n, int *nbors);

    //void ReceiveLoadInfo(int itr, double load, int node);
    void MigrationDone();  // Call when migration is complete
    void ReceiveStats(CkMarshalledCLBStatsMessage &&data);

    //void LoadTransfer(double load, int initPE, int objId);
    void LoadReceived(int objId, int fromPE);
    //void PseudoLoad(int itr, double load, int node);    

    //void MigrationInfo(int to, int from);

    // TODO: Can be made private or not entry methods as well.
    void MigrationEnded();
    void DoneNodeLB();

    void ResumeClients(CkReductionMsg *msg);
    void ResumeClients(int balancing);
    void CallResumeClients();
    void PrintDebugMessage(int len, double* result);
    void LoadMetaInfo(LDObjHandle h, double load);
protected:
    virtual bool QueryBalanceNow(int) { return true; };  

private:
    CProxy_DiffusionLB thisProxy;

    // aggregate load received
    int itr; // iteration count
    int temp_itr;
    int pick;
    int notif;
    int statsReceived, rank0_acks;
    int loadReceived;
    int do_again = 1;
    int round, requests_sent;
    int thisNode;
    int *nbors;
    CLBStatsMsg* statsmsg;
    CkMarshalledCLBStatsMessage *marshmsg;
    CLBStatsMsg** statsList;
    BaseLB::LDStats* nodeStats; 
    DistBaseLB::LDStats* myStats;
    std::vector<int> numObjects;
    std::vector<int> prefixObjects;
    std::vector<double> pe_load;
    std::vector<double> pe_loadBefore;
    
    std::vector<int> neighbors; // Neighbors which the node uses to make load balancing decisions
    std::vector<double> loadNeighbors;
    std::vector<int> sendToNeighbors; // Neighbors to which curr node has to send load.
    int toSend;
    std::unordered_map<int, int> neighborPos;  // nodes position in the neighbors vector
    int neighborCount;
    std::unordered_map<int, int> neighborPosReceive;  // nodes position in the neighbors vector
    std::vector<double> toSendLoad;
    std::vector<double> toReceiveLoad;
    int edgeCount;
    std::vector<int> edge_indices;
    const DistBaseLB::LDStats *statsData;

    // Information to send to neighbor nodes
    LDObjKey* nodeKeys;
    double my_load;
    double my_loadAfterTransfer;

    // Load information received from neighbor node
    std::vector<std::vector<LDObjKey>> neighborKeys;
    int londReceived;

    double b_load;
    double avgLoadNeighbor;  // Average load of the neighbor group
    
    // heap
    int* obj_arr;
    int* gain_val;
    std::vector<CkVertex> objs;
    std::vector<int> obj_heap;
    std::vector<int> heap_pos;

    // usability
    int myspeed;
    int rank0PE;
    int nodeSize;
    int numNodes;
//    std::unordered_map<int, int> peNodes;
//    std::vector<int> nodes;

    int actualSend;
    std::vector<bool> balanced;
    
    // migration
    int total_migrates;
    int total_migratesActual;
    int migrates_completed;
    int migrates_expected;
    std::vector<MigrateInfo*> migrateInfo;
    LBMigrateMsg* msg;
    std::vector<int> migratedTo;
    std::vector<int> migratedFrom;
    bool entered;
    int finalBalancing;

    //stats
    double strat_end_time;
    double start_lb_time;
    double end_lb_time;
    // debug messages vars and functions.
    // load
    double maxB;
    double minB;
    double avgB;
    double maxA;
    double minA;
    double avgA;
    int maxPEB;
    int maxPEA;
    int minPEB;
    int minPEA;
    // communication
    double internalBefore;
    double externalBefore;
    double internalAfter;
    double externalAfter;
    double internalBeforeFinal;
    double externalBeforeFinal;
    double internalAfterFinal;
    double externalAfterFinal;
    int receivedStats;
    int migrates; // number of objects migrated across node
    int migratesNode; // number of objects migrated within node   

    // Cascading migrations
    std::vector<LDObjHandle> objectHandles;
    std::vector<double> objectLoads;
    int FindObjectHandle(LDObjHandle h);
    void CascadingMigration(LDObjHandle h, double load);

    // helper functions
    bool preprocessingDone;
    int GetPENumber(int& obj_id);
    void preprocess(const int explore_limit);
    bool AggregateToSend();
    
    void Strategy(const DistBaseLB::LDStats* const stats);
    
    // Heap functions
    void InitializeObjHeap(BaseLB::LDStats *stats, int* obj_arr,int size, int* gain_val);
    
    // topo aware neighbors are populated.
    void ComputeNeighbors();
    double  avgNborLoad();
    double averagePE();
    int findNborIdx(int node);
    void PseudoLoadBalancing();
    void LoadBalancing();
    CLBStatsMsg* AssembleStats();
    void AddToList(CLBStatsMsg* m, int rank);
    void BuildStats();
};

#endif /* _DistributedLB_H_ */

