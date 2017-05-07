#ifndef _GRAPH_KL_LB_H
#define _GRAPH_KL_LB_H

#include "CentralLB.h"
#include "GraphKLLB.decl.h"
#include "ckgraph.h"
#include <unordered_map>

void CreateGraphKLLB();
BaseLB * AllocateGraphKLLB();

class GraphKLLB : public CBase_GraphKLLB {
 public:
    GraphKLLB(const CkLBOptions &);
    GraphKLLB(CkMigrateMessage *m):CBase_GraphKLLB(m) { lbname = "GraphKLLB"; }
    void work(LDStats* stats);
 private:
    bool QueryBalanceNow(int step);
    void InitializeObjs(LDStats* stats, int* obj_arr, int* gain_val);
    void InitializeObjHeap(LDStats* stats, int* obj_arr,int size, int* gain_val);
      
    void RefineGraphKLLB(int* obj_arr,std::unordered_map<int,int>& Ra,std::unordered_map<int,int>& Rb,
    int idx,int start,int size,float part_load,float total_load,int new_part_sz,int part_size, int* gain_val);
    void setup_RefineGraphKLLB(int* obj_arr,std::unordered_map<int,int>& Ra,
        std::unordered_map<int,int>& Rb,int idx,int size,float part_load,float total_load,int part_size, int* gain_val);
    void RefineGraphKLLB_setgains(int* obj_arr,std::unordered_map<int,int>& Ra,std::unordered_map<int,int>& Rb, int idx,int size);
    void Update_NeighborsGain(int source,int dest,std::unordered_map<int,int>& Ra,std::unordered_map<int,int>& Rb, int* gain_val);
      
    void TransferObjToProc(LDStats* stats, ProcInfo& p, Vertex& v);
    Vertex& PickObjFromQueue(int q_idx);
    ProcInfo& PickProcFromHeap(int q_idx, int& pe_idx);
    void AdjustObjQueues(int q_idx);
    void AdjustProcHeap(int q_idx, int pe_idx);

    void RecursiveBisection(LDStats* stats, int* obj_arr, int size, int start,
    double load, int* gain_val, int* part, int part_size);
    void Gain_Update(int* gain_update,int* gain_val);
    void UpdateVertexNeighbor(int v_idx, int* gain_val,int* gain_update,std::unordered_map<int,int>& Rb);
    void ClearDatastructure();

    int n_pes;
    int n_objs;
      
    // Total load of all the objects
    double total_vload;

    std::vector<ProcInfo>  procs;

    std::vector<Vertex> objs;
    std::vector<int> obj_heap;
    std::vector<int> heap_pos;
    ObjGraph *ogr;
    int filled_queues;
    bool flag;

};

#endif
