#include <charm++.h>

#if CMK_LBDB_ON

#include "CkLists.h"
#include "HeapCentLB.h"
#include "HeapCentLB.def.h"

void CreateHeapCentLB()
{
  //  CkPrintf("[%d] creating HeapCentLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_HeapCentLB::ckNew();
  //  CkPrintf("[%d] created HeapCentLB %d\n",CkMyPe(),loadbalancer);
}

HeapCentLB::HeapCentLB()
{
  if (CkMyPe()==0)
    CkPrintf("[%d] HeapCentLB created\n",CkMyPe());
}

CmiBool HeapCentLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

void HeapCentLB::Heapify(HeapData *heap, int node, int heapSize)
{
  int left = 2*node+1;
  int right = 2*node+2;
  int smallest;

  if (left <= heapSize && heap[left].load > heap[node].load)
    smallest = left;
  else smallest = node;
  
  if (right <= heapSize && heap[right].load > heap[smallest].load) 
    smallest = right;

  if (smallest != node) {
    HeapData obj;
    obj = heap[node];
    heap[node] = heap[smallest];
    heap[smallest] = obj;
    Heapify(heap, smallest, heapSize);
  }    
}


//Inserts object into appropriate sorted position in the object array
void HeapCentLB::InsertObject(HeapData *objData, int index)
{
  HeapData key = objData[index];
  
  int i;
  for(i = index-1; i >= 0 && objData[i].load < key.load; i--)
    objData[i+1] = objData[i];
  objData[i+1] = key;
}

HeapCentLB::HeapData* 
HeapCentLB::BuildObjectArray(CentralLB::LDStats* stats, 
                             int count, int *objCount)
{
  HeapData *objData;

  *objCount = 0;
  int pe, obj;
  for (pe = 0; pe < count; pe++)
    for (obj = 0; obj < stats[pe].n_objs; obj++)
      if (stats[pe].objData[obj].migratable == CmiTrue) (*objCount)++; 

  objData  = new HeapData[*objCount];
  *objCount = 0; 
  for(pe=0; pe < count; pe++)
    for(obj=0; obj < stats[pe].n_objs; obj++)
      if (stats[pe].objData[obj].migratable == CmiTrue) {
        objData[*objCount].load = 
          stats[pe].objData[obj].wallTime * stats[pe].pe_speed;
        objData[*objCount].pe = pe;
        objData[*objCount].id = obj;
        InsertObject(objData, (*objCount)++);
      }

  return objData;
}

HeapCentLB::HeapData* 
HeapCentLB::BuildCpuArray(CentralLB::LDStats* stats, 
                          int count, int *peCount)
{
  HeapData           *data;
  CentralLB::LDStats *peData;
  
  *peCount = 0;
  int pe, obj;
  for (pe = 0; pe < count; pe++)
    if (stats[pe].available == CmiTrue) (*peCount)++;

  data = new HeapData[*peCount];
  
  *peCount = 0;
  for (pe=0; pe < count; pe++) {
    data[*peCount].load = 0.0;
    peData = &(stats[pe]);

    for (obj = 0; obj < peData->n_objs; obj++) 
      if (peData->objData[obj].migratable == CmiFalse) 
        data[*peCount].load -= 
          peData->objData[obj].wallTime * peData->pe_speed;
        
     if (peData->available == CmiTrue) {
      data[*peCount].load += 
        (peData->total_walltime - peData->bg_walltime) * peData->pe_speed;
      data[*peCount].pe = data[*peCount].id = pe;
      InsertObject(data, *peCount);
      (*peCount)++;
    }
  }
  
  return data;
}

CLBMigrateMsg* HeapCentLB::Strategy(CentralLB::LDStats* stats, int count)
{
  CkVector migrateInfo;
  int      obj, heapSize, objCount;
  HeapData *cpuData = BuildCpuArray(stats, count, &heapSize);
  HeapData *objData = BuildObjectArray(stats, count, &objCount);

  //  CkPrintf("[%d] HeapCentLB strategy\n",CkMyPe());

  heapSize--;
  HeapData maxCpu;  
  for (obj=0; obj < objCount; obj++) {
    // Operation of extracting the the least loaded processor
    // from the heap
    maxCpu = cpuData[0];
    cpuData[0] = cpuData[heapSize];
    heapSize--;
    Heapify(cpuData, 0, heapSize);    

    // Increment the time of the least loaded processor by the cpuTime of
    // the `heaviest' object
    maxCpu.load -= objData[obj].load;

    //Insert object into migration queue if necessary
    const int dest = maxCpu.pe;
    const int pe   = objData[obj].pe;
    const int id   = objData[obj].id;
    if (dest != pe) {
      //      CkPrintf("[%d] Obj %d migrating from %d to %d\n",
      //         CkMyPe(),obj,pe,dest);
      MigrateInfo *migrateMe = new MigrateInfo;
      migrateMe->obj = stats[pe].objData[id].handle;
      migrateMe->from_pe = pe;
      migrateMe->to_pe = dest;
      migrateInfo.push_back((void*)migrateMe);
    }

    //Insert the least loaded processor with load updated back into the heap
    heapSize++;
    int location = heapSize;
    while (location>0 && cpuData[(location-1)/2].load < maxCpu.load) {
      cpuData[location] = cpuData[(location-1)/2];
      location = (location-1)/2;
    }
    cpuData[location] = maxCpu;
  }

  int migrate_count=migrateInfo.size();
  CkPrintf("HeapCentLB migrating %d elements\n",migrate_count);
  CLBMigrateMsg* msg = new(&migrate_count,1) CLBMigrateMsg;
  msg->n_moves = migrate_count;
  for(int i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  return msg;
};

#endif






