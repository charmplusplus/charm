/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <charm++.h>

#if CMK_LBDB_ON

#include "cklists.h"
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

CmiBool GreedyRefLB::Compare(double x, double y, GreedyCmp cmp)
{
  int test = ((cmp == GT) ? (x > y) : (x < y));
  if (test) return CmiTrue;
  else return CmiFalse;
}


void GreedyRefLB::Heapify(HeapData *heap, int node, int heapSize, GreedyCmp cmp)
{
  int left = 2*node+1;
  int right = 2*node+2;
  int xchange;

  //heap[left].load > heap[node].load)
  if (left <= heapSize && Compare(heap[left].load, heap[node].load, cmp))
	  xchange = left;
  else xchange = node;
  //heap[right].load > heap[xchange].load
  if (right <= heapSize && Compare(heap[right].load, heap[xchange].load, cmp)) 
    xchange = right;

  if (xchange != node) {
    HeapData obj;
    obj = heap[node];
    heap[node] = heap[xchange];
    heap[xchange] = obj;
    Heapify(heap, xchange, heapSize, cmp);
  }    
}

void GreedyRefLB::BuildHeap(HeapData *data, int heapSize, GreedyCmp cmp)
{
	int i;
	for(i=heapSize/2; i >= 0; i--)
		Heapify(data, i, heapSize, cmp);
}

void GreedyRefLB::HeapSort(HeapData *data, int heapSize, GreedyCmp cmp)
{
	int i;
	HeapData key;

	BuildHeap(data, heapSize, cmp);
	for (i=heapSize; i > 0; i--) {
		key = data[0];
		data[0] = data[i];
		data[i] = key;
		heapSize--;
		Heapify(data, 0, heapSize, cmp);
	}
}

GreedyRefLB::HeapData* 
GreedyRefLB::BuildObjectArray(CentralLB::LDStats* stats, 
									int count, int *objCount)
{
  HeapData *objData;

  *objCount = 0;
  int pe, obj;
  for (pe = 0; pe < count; pe++)
	  *objCount += stats[pe].n_objs;
    //for (obj = 0; obj < stats[pe].n_objs; obj++)
      //if (stats[pe].objData[obj].migratable == CmiTrue) (*objCount)++; 

  objData  = new HeapData[*objCount];
  *objCount = 0; 
  for(pe=0; pe < count; pe++)
	for(obj=0; obj < stats[pe].n_objs; obj++) {
      //if (stats[pe].objData[obj].migratable == CmiTrue) {
        objData[*objCount].load = 
          stats[pe].objData[obj].wallTime * stats[pe].pe_speed;
        objData[*objCount].pe = pe;
        objData[*objCount].id = obj;
		(*objCount)++;
    }
  HeapSort(objData, *objCount-1, GT);
  return objData;
}

GreedyRefLB::HeapData* 
GreedyRefLB::BuildCpuArray(CentralLB::LDStats* stats, int count, int *peCount)
{
  HeapData           *data;
  CentralLB::LDStats *peData;
  
  *peCount = 0;
  int pe;
  for (pe = 0; pe < count; pe++)
    if (stats[pe].available == CmiTrue) (*peCount)++;

  data = new HeapData[*peCount];
  
  *peCount = 0;
  for (pe=0; pe < count; pe++) {
    data[*peCount].load = 0.0;
    peData = &(stats[pe]);

    if (peData->available == CmiTrue) {

	/*			 
	for (obj = 0; obj < peData->n_objs; obj++) 
		 if (peData->objData[obj].migratable == CmiFalse) 
				 data[*peCount].load -= 
					 peData->objData[obj].wallTime * peData->pe_speed;
	*/
       
	 data[*peCount].load += peData->bg_walltime * peData->pe_speed;
	//		 (peData->total_walltime - peData->bg_walltime) * peData->pe_speed;
	 data[*peCount].pe = data[*peCount].id = pe;
	 (*peCount)++;
	}
  }
  BuildHeap(data, *peCount-1, LT);
  return data;
}



CLBMigrateMsg* GreedyRefLB::Strategy(CentralLB::LDStats* stats, int count)
{
  CkVec<MigrateInfo*> migrateInfo;
  int      pe, obj, heapSize, objCount;
  HeapData *cpuData = BuildCpuArray(stats, count, &heapSize);
  HeapData *objData = BuildObjectArray(stats, count, &objCount);

  //  CkPrintf("[%d] GreedyRefLB strategy\n",CkMyPe());

  // Build the refine data structure, and use it for storing the info
  // from the heap
  int** from_procs = Refiner::AllocProcs(count, stats);
  for(pe=0;pe<count;pe++)
    for(obj=0;obj < stats[pe].n_objs; obj++)
      from_procs[pe][obj] = 0;

  heapSize--;
  HeapData minCpu;  
  for (obj=0; obj < objCount; obj++) {
    // Operation of extracting the the least loaded processor
    // from the heap
    minCpu = cpuData[0];
    cpuData[0] = cpuData[heapSize];
    heapSize--;
    Heapify(cpuData, 0, heapSize, LT);    

    // Increment the time of the least loaded processor by the cpuTime of
    // the `heaviest' object
    minCpu.load += objData[obj].load;

    //Insert object into migration queue if necessary
    const int dest = minCpu.pe;
    const int pe   = objData[obj].pe;
    const int id   = objData[obj].id;
    from_procs[pe][id] = dest;

    //Insert the least loaded processor with load updated back into the heap
    heapSize++;
    int location = heapSize;
    while (location>0 && cpuData[(location-1)/2].load > minCpu.load) {
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
  //  CkPrintf("[%d] Obj %d migrating from %d to %d\n",
  //     CkMyPe(),obj,pe,to_procs[pe][obj]);
  MigrateInfo *migrateMe = new MigrateInfo;
  migrateMe->obj = stats[pe].objData[obj].handle;
  migrateMe->from_pe = pe;
  migrateMe->to_pe = to_procs[pe][obj];
  migrateInfo.insertAtEnd(migrateMe);
      }
    }
  }

  int migrate_count=migrateInfo.length();
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
