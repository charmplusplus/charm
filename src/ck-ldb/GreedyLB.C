/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>

#if CMK_LBDB_ON

#include "cklists.h"
#include "GreedyLB.h"

void CreateGreedyLB()
{
  //  CkPrintf("[%d] creating GreedyLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_GreedyLB::ckNew();
  //  CkPrintf("[%d] created GreedyLB %d\n",CkMyPe(),loadbalancer);
}

static void lbinit(void) {
//        LBSetDefaultCreate(CreateGreedyLB);        
  LBRegisterBalancer("GreedyLB", CreateGreedyLB, "always assign the heaviest obj onto lightest loaded processor.");
}

#include "GreedyLB.def.h"

GreedyLB::GreedyLB()
{
  lbname = "GreedyLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] GreedyLB created\n",CkMyPe());
}

CmiBool GreedyLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

CmiBool  GreedyLB::Compare(double x, double y, HeapCmp cmp)
{
  const int test =  ((cmp == GT) ? (x > y) : (x < y));

  if (test) return CmiTrue; 
  else return CmiFalse;
}


void GreedyLB::Heapify(HeapData *heap, int node, int heapSize, HeapCmp cmp)
{
  int left = 2*node+1;
  int right = 2*node+2;
  int xchange;

  //heap[left].load > heap[node].load)
  if (left <= heapSize &&  Compare(heap[left].load, heap[node].load, cmp))
    xchange = left;
  else xchange = node;
  //heap[right].load > heap[xchange].load) 
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

void GreedyLB::BuildHeap(HeapData *data, int heapSize, HeapCmp cmp)
{
	int i;
	for(i=heapSize/2; i >= 0; i--)
		Heapify(data, i, heapSize, cmp);
}

void GreedyLB::HeapSort(HeapData *data, int heapSize, HeapCmp cmp)
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

GreedyLB::HeapData* 
GreedyLB::BuildObjectArray(CentralLB::LDStats* stats, 
                             int count, int *objCount)
{
  HeapData *objData;

  *objCount = 0;
  int obj;
  *objCount += stats->n_objs;

//for (obj = 0; obj < stats[pe].n_objs; obj++)
//if (stats[pe].objData[obj].migratable == CmiTrue) (*objCount)++; 

  objData  = new HeapData[*objCount];
  *objCount = 0; 
  for(obj=0; obj < stats->n_objs; obj++) {
//      if (stats[pe].objData[obj].migratable == CmiTrue) {
	int pe = stats->from_proc[obj];
        objData[*objCount].load = 
          stats->objData[obj].wallTime * stats->procs[pe].pe_speed;
        objData[*objCount].pe = pe;
        objData[*objCount].id = obj;
        (*objCount)++;
  }
  
  HeapSort(objData, *objCount-1, GT);
  return objData;
}

GreedyLB::HeapData* 
GreedyLB::BuildCpuArray(CentralLB::LDStats* stats, 
                          int count, int *peCount)
{
  HeapData           *data;
  CentralLB::ProcStats *peData;
  
  *peCount = 0;
  int pe;
  for (pe = 0; pe < count; pe++)
    if (stats->procs[pe].available == CmiTrue) (*peCount)++;

  data = new HeapData[*peCount];
  
  *peCount = 0;
  for (pe=0; pe < count; pe++) {
    data[*peCount].load = 0.0;
    peData = &(stats->procs[pe]);
 
    if (peData->available == CmiTrue) 
	{
/*  Use of the migratable flag is not yet defined
	  for (int obj = 0; obj < peData->n_objs; obj++) 
	  { 
		if (peData->objData[obj].migratable == CmiFalse) 
			data[*peCount].load -= 
				peData->objData[obj].wallTime * peData->pe_speed;
	  }
*/        
	  data[*peCount].load += peData->bg_walltime * peData->pe_speed;
// (peData->total_walltime - peData->bg_walltime) * peData->pe_speed;
      data[*peCount].pe = data[*peCount].id = pe;
      (*peCount)++;
    }
  }
  BuildHeap(data, *peCount-1, LT);
  return data;
}

void GreedyLB::work(CentralLB::LDStats* stats, int count)
{
  int      obj, heapSize, objCount;
  HeapData *cpuData = BuildCpuArray(stats, count, &heapSize);
  HeapData *objData = BuildObjectArray(stats, count, &objCount);

  //  CkPrintf("[%d] GreedyLB strategy\n",CkMyPe());

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
    if (dest != pe) {
      stats->to_proc[id] = dest;
      //      CkPrintf("[%d] Obj %d migrating from %d to %d\n",
      //         CkMyPe(),obj,pe,dest);
/*
      MigrateInfo *migrateMe = new MigrateInfo;
      migrateMe->obj = stats->objData[id].data.handle;
      migrateMe->from_pe = pe;
      migrateMe->to_pe = dest;
      migrateInfo.insertAtEnd(migrateMe);
*/
    }

    //Insert the least loaded processor with load updated back into the heap
    heapSize++;
    int location = heapSize;
    while (location>0 && cpuData[(location-1)/2].load > minCpu.load) {
      cpuData[location] = cpuData[(location-1)/2];
      location = (location-1)/2;
    }
    cpuData[location] = minCpu;
  }

  delete [] cpuData;
  delete [] objData;
}

LBMigrateMsg * GreedyLB::createMigrateMsg(LDStats* stats,int count)
{
  int i;
  CkVec<MigrateInfo*> migrateInfo;
  for (i=0; i<stats->n_objs; i++) {
    LDObjData &objData = stats->objData[i];
    int frompe = stats->from_proc[i];
    int tope = stats->to_proc[i];
    if (frompe != tope) {
      //      CkPrintf("[%d] Obj %d migrating from %d to %d\n",
      //         CkMyPe(),obj,pe,dest);
      MigrateInfo *migrateMe = new MigrateInfo;
      migrateMe->obj = objData.handle;
      migrateMe->from_pe = frompe;
      migrateMe->to_pe = tope;
      migrateInfo.insertAtEnd(migrateMe);
    }
  }

  int migrate_count=migrateInfo.length();
  CkPrintf("GreedyLB migrating %d elements\n",migrate_count);
  LBMigrateMsg* msg = new(&migrate_count,1) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  return msg;
};

#endif


/*@}*/




