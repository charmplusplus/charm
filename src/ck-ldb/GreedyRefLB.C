#include <charm++.h>

#if CMK_LBDB_ON

#include "CkLists.h"
#include "GreedyRefLB.h"
#include "GreedyRefLB.def.h"

void CreateGreedyRefLB()
{
  //  CkPrintf("[%d] creating GreedyRefLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_GreedyRefLB::ckNew();
  //  CkPrintf("[%d] created GreedyRefLB %d\n",CkMyPe(),loadbalancer);
}

GreedyRefLB::GreedyRefLB()
{
  if (CkMyPe()==0)
    CkPrintf("[%d] GreedyRefLB created\n",CkMyPe());
}

CmiBool GreedyRefLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

void GreedyRefLB::Heapify(HeapData *heap, int node, int heapSize)
{
  int left = 2*node+1;
  int right = 2*node+2;
  int largest;

  if (left <= heapSize && heap[left].cpuTime < heap[node].cpuTime)
    largest = left;
  else largest = node;
	
  if (right <= heapSize && heap[right].cpuTime < heap[largest].cpuTime) 
    largest = right;

  if (largest != node) {
    HeapData obj;
    obj = heap[node];
    heap[node] = heap[largest];
    heap[largest] = obj;
    Heapify(heap, largest, heapSize);
  }
		
}


CLBMigrateMsg* GreedyRefLB::Strategy(CentralLB::LDStats* stats, int count)
{
  int pe,obj;
  //  CkPrintf("[%d] GreedyRefLB strategy\n",CkMyPe());

  CkVector migrateInfo;

  int totalObjs = 0;
  HeapData *cpuData = new HeapData[count];
  HeapData *objData;

  for (pe=0; pe < count; pe++) {
    totalObjs += stats[pe].n_objs;
    cpuData[pe].cpuTime = 0.;
    cpuData[pe].pe = cpuData[pe].id = pe;
  }

  objData = new HeapData[totalObjs];
  int objCount = 0;
  for(pe=0; pe < count; pe++) {
    //    CkPrintf("[%d] PE %d : %d Objects : %d Communication\n",
    //	     CkMyPe(),pe,stats[pe].n_objs,stats[pe].n_comm);

    for(obj=0; obj < stats[pe].n_objs; obj++, objCount++) {
      objData[objCount].cpuTime = stats[pe].objData[obj].cpuTime;
      objData[objCount].pe = pe;
      objData[objCount].id = obj;
    }
  }

  for (obj=1; obj < totalObjs; obj++) {
    HeapData key = objData[obj];
    int i = obj-1;
    while (i >=0 && objData[i].cpuTime < key.cpuTime) {
      objData[i+1] = objData[i];
      i--;
    }
    objData[i+1] = key;
  }
  
  // Build the refine data structure, and use it for storing the info
  // from the heap
  int** from_procs = Refiner::AllocProcs(count, stats);
  for(pe=0;pe<count;pe++)
    for(obj=0;obj < stats[pe].n_objs; obj++)
      from_procs[pe][obj] = 0;

  int heapSize = count-1;
  HeapData minCpu;	
  for (obj=0; obj < totalObjs; obj++) {
    // Operation of extracting the minimum(the least loaded processor)
    // from the heap
    minCpu = cpuData[0];
    cpuData[0] = cpuData[heapSize];
    heapSize--;
    Heapify(cpuData, 0, heapSize);		

    // Increment the time of the least loaded processor by the cpuTime of
    // the `heaviest' object
    minCpu.cpuTime += objData[obj].cpuTime;

    //Insert object into migration queue if necessary
    const int dest = minCpu.pe;
    const int pe   = objData[obj].pe;
    const int id   = objData[obj].id;
    from_procs[pe][id] = dest;

    //Insert the least loaded processor with load updated back into the heap
    heapSize++;
    int location = heapSize;
    while (location>0 && cpuData[(location-1)/2].cpuTime > minCpu.cpuTime) {
      cpuData[location] = cpuData[(location-1)/2];
      location = (location-1)/2;
    }
    cpuData[location] = minCpu;
  }

  int initial_migrates=0;
  for(pe=0;pe < count; pe++) {
    for(obj=0;obj<stats[pe].n_objs;obj++) {
      if (from_procs[pe][obj] == -1)
	CkPrintf("From_Proc was unassigned!\n");
      if (from_procs[pe][obj] != pe)
	initial_migrates++;
    }
  }
  CkPrintf("Initially migrating %d objects\n",initial_migrates);

  // Get a new buffer to refine into
  int** to_procs = Refiner::AllocProcs(count,stats);

  Refiner refiner(1.01);  // overload tolerance=1.05
  
  refiner.Refine(count,stats,from_procs,to_procs);

  // Report on the output
  for(pe=0;pe < count; pe++) {
    for(obj=0;obj<stats[pe].n_objs;obj++) {
      if (from_procs[pe][obj] != to_procs[pe][obj]) {
	CkPrintf("Refinement moved obj %d orig %d from %d to %d\n",
		 obj,pe,from_procs[pe][obj],to_procs[pe][obj]);
      }
    }
  }

  // Save output
  for(pe=0;pe < count; pe++) {
    for(obj=0;obj<stats[pe].n_objs;obj++) {
      if (from_procs[pe][obj] == -1)
	CkPrintf("From_Proc was unassigned!\n");
      if (to_procs[pe][obj] == -1)
	CkPrintf("To_Proc was unassigned!\n");
      
      if (to_procs[pe][obj] != pe) {
	//	CkPrintf("[%d] Obj %d migrating from %d to %d\n",
	//		 CkMyPe(),obj,pe,to_procs[pe][obj]);
	MigrateInfo *migrateMe = new MigrateInfo;
	migrateMe->obj = stats[pe].objData[obj].handle;
	migrateMe->from_pe = pe;
	migrateMe->to_pe = to_procs[pe][obj];
	migrateInfo.push_back((void*)migrateMe);
      }
    }
  }

  int migrate_count=migrateInfo.size();
  CkPrintf("GreedyRefLB migrating %d elements\n",migrate_count);
  CLBMigrateMsg* msg = new(&migrate_count,1) CLBMigrateMsg;
  msg->n_moves = migrate_count;
  for(int i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  // Free the refine buffers
  Refiner::FreeProcs(from_procs);
  Refiner::FreeProcs(to_procs);
  
  return msg;
};

#endif
