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

#include "GraphRefinementDist.decl.h"

void CreateGraphRefinementDist();

/// for backward compatibility
typedef LBMigrateMsg NLBMigrateMsg;

class GraphRefinementDist : public CBase_GraphRefinementDist {
public:
    GraphRefinementDist(const CkLBOptions &);
    GraphRefinementDist(CkMigrateMessage *m);
    ~GraphRefinementDist();
    static void staticAtSync(void*);
    static void staticMigrated(void* me, LDObjHandle h, int waitBarrier);
    void Migrated(LDObjHandle h, int waitBarrier=1);
    void PEStarted();
    void AddNeighbor(int node);
    void AtSync(void); // Everything is at the PE barrier
    void ProcessAtSync(void);

    void ReceiveLoadInfo(double load, int node);
    void MigrationDone(int balancing);  // Call when migration is complete
    void ReceiveStats(CkMarshalledCLBStatsMessage &data);

    void LoadTransfer(double load, int initPE, int objId);
    void LoadReceived(int objId, int fromPE);
    
    void MigrationInfo(int to, int from);

    // TODO: Can be made private or not entry methods as well.
    void MigrationEnded();
    void DoneNodeLB();

    void ResumeClients(CkReductionMsg *msg);
    void ResumeClients(int balancing);
    void CallResumeClients();
    void PrintDebugMessage(int len, double* msg);
protected:
    virtual bool QueryBalanceNow(int) { return true; };  

private:
    CProxy_GraphRefinementDist thisProxy;

    // aggregate load received
    int statsReceived;
    int loadReceived;
    CLBStatsMsg** statsList;
    BaseLB::LDStats* nodeStats; 
    DistBaseLB::LDStats* myStats;
    std::vector<int> numObjects;
    std::vector<int> prefixObjects;
    std::vector<double> loadPE;
    std::vector<double> loadPEBefore;
    
    std::vector<int> neighbors; // Neighbors which the node uses to make load balancing decisions
    std::vector<double> loadNeighbors;
    std::vector<int> sendToNeighbors; // Neighbors to which curr node has to send load.
    int toSend;
    std::unordered_map<int, int> neighborPos;  // nodes position in the neighbors vector
    int neighborCount;

    // Information to send to neighbor nodes
    LDObjKey* nodeKeys;
    double my_load;

    // Load information received from neighbor node
    std::vector<std::vector<LDObjKey>> neighborKeys;
    int londReceived;

    double b_load;
    double avgLoadNeighbor;  // Average load of the neighbor group
    
    // heap
    int* obj_arr;
    int* gain_val;
    std::vector<Vertex> objs;
    std::vector<int> obj_heap;
    std::vector<int> heap_pos;

    // usability
    int n_objs;
    int myspeed;
    int nodeFirst;
    int nodeSize;
    int numNodes;
    std::unordered_map<int, int> peNodes;
    std::vector<int> nodes;
    
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

    // helper functions
    int GetPENumber(int& obj_id);
    std::vector<TCoord> coord_table;
    std::vector<TCoord*> peToCoords;
    bool preprocessingDone;
    std::vector< std::vector<TCoord*> > closest_coords;
    void preprocess(const int explore_limit);
    
    void InitLB(const CkLBOptions &);
    void Strategy(const BaseLB::LDStats* const stats);
    
    // Heap functions
    void InitializeObjHeap(BaseLB::LDStats *stats, int* obj_arr,int size, int* gain_val);
    
    // topo aware neighbors are populated.
    void ComputeNeighbors();
    double  average();
    double averagePE();
    void LoadBalancing();
    CLBStatsMsg* AssembleStats();
    void AddToList(CLBStatsMsg* m, int rank);
    void BuildStats();
};

#endif /* _DistributedLB_H_ */

