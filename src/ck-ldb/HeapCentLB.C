#include <charm++.h>

#if CMK_LBDB_ON

#if CMK_STL_USE_DOT_H
#include <deque.h>
#include <queue.h>
#else
#include <deque>
#include <queue>
#endif

#include "HeapCentLB.h"
#include "HeapCentLB.def.h"

#if CMK_STL_USE_DOT_H
template class deque<CentralLB::MigrateInfo>;
#else
template class std::deque<CentralLB::MigrateInfo>;
#endif

void CreateHeapCentLB()
{
  CkPrintf("[%d] creating HeapCentLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_HeapCentLB::ckNew();
  CkPrintf("[%d] created RandCentLB %d\n",CkMyPe(),loadbalancer);
}

HeapCentLB::HeapCentLB()
{
  CkPrintf("[%d] RandCentLB created\n",CkMyPe());
}

CmiBool HeapCentLB::QueryBalanceNow(int _step)
{
  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

void Heapify(HeapData *heap, int node, int heapSize)
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


CLBMigrateMsg* HeapCentLB::Strategy(CentralLB::LDStats* stats, int count)
{
  CkPrintf("[%d] HeapCentLB strategy\n",CkMyPe());

#if CMK_STL_USE_DOT_H
  queue<MigrateInfo> migrateInfo;
#else
  std::queue<MigrateInfo> migrateInfo;
#endif
	
	int totalObjs = 0;
	HeapData *cpuData = new HeapData[count];
	HeapData *objData;

	for (int pe=0; pe < count; pe++) {
		totalObjs += stats[pe].n_objs;
		cpuData[pe].cpuTime = 0.;
		cpuData[pe].pe = cpuData[pe].id = pe;
	}

	objData = new HeapData[totalObjs];
	int objCount = 0;
  for(int pe=0; pe < count; pe++) {
    CkPrintf("[%d] PE %d : %d Objects : %d Communication\n",
	     CkMyPe(),pe,stats[pe].n_objs,stats[pe].n_comm);

    for(int obj=0; obj < stats[pe].n_objs; obj++, objCount++) {

			objData[objCount].cpuTime = stats[pe].objData[obj].cpuTime;
			objData[objCount].pe = pe;
			objData[objCount].id = obj;
		}
	}

	for (int obj=1; obj < totalObjs; obj++) {
		HeapData key = objData[obj];
		int i = obj-1;
		while (i >=0 && objData[i].cpuTime < key.cpuTime) {
			objData[i+1] = objData[i];
			i--;
		}
		objData[i+1] = key;
	}

	int heapSize = count-1;
	HeapData minCpu;	
	for (int obj=0; obj < totalObjs; obj++) {

		//Operation of extracting the minimum(the least loaded processor) from the heap
		minCpu = cpuData[0];
		cpuData[0] = cpuData[heapSize];
		heapSize--;
		Heapify(cpuData, 0, heapSize);		

		//Increment the time of the least loaded processor by the cpuTime of the `heaviest' object
		minCpu.cpuTime += objData[obj].cpuTime;

    //Insert object into migration queue if necessary
		const int dest = minCpu.pe;
		const int pe   = objData[obj].pe;
		const int id   = objData[obj].id;
		if (dest != pe) {
			CkPrintf("[%d] Obj %d migrating from %d to %d\n",
							 CkMyPe(),obj,pe,dest);
			MigrateInfo migrateMe;
			migrateMe.obj = stats[pe].objData[id].handle;
			migrateMe.from_pe = pe;
			migrateMe.to_pe = dest;
			migrateInfo.push(migrateMe);
		}
		
		//Insert the least loaded processor with load updated back into the heap
		heapSize++;
		int location = heapSize;
		while (location>0 && cpuData[(location-1)/2].cpuTime > minCpu.cpuTime) {
			cpuData[location] = cpuData[(location-1)/2];
			location = (location-1)/2;
		}
		cpuData[location] = minCpu;
	}
  int migrate_count=migrateInfo.size();
  CLBMigrateMsg* msg = new(&migrate_count,1) CLBMigrateMsg;
  msg->n_moves = migrate_count;
  for(int i=0; i < migrate_count; i++) {
    msg->moves[i] = migrateInfo.front();
    migrateInfo.pop();
  }
  return msg;
};

#endif
