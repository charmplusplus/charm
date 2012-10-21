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

/*
 status:
  * support processor avail bitvector
  * support nonmigratable attrib
      nonmigratable object load is added to its processor's background load
      and the nonmigratable object is not taken in the objData array
*/

#include "charm++.h"


#include "cklists.h"
#include "GreedyLB.h"

CreateLBFunc_Def(GreedyLB, "always assign the heaviest obj onto lightest loaded processor.")

GreedyLB::GreedyLB(const CkLBOptions &opt): CentralLB(opt)
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

        int origSize = heapSize;
	BuildHeap(data, heapSize, cmp);
        for (i=heapSize; i > 0; i--) {
               key = data[0];
               data[0] = data[i];
               data[i] = key;
               heapSize--;
               Heapify(data, 0, heapSize, cmp);
	}
	// after HeapSort, the data are in reverse order
        for (i=0; i<(origSize+1)/2; i++) {
          key = data[i];
          data[i] = data[origSize-i];
          data[origSize-i] = key;
        }
}

GreedyLB::HeapData* 
GreedyLB::BuildObjectArray(BaseLB::LDStats* stats, 
                             int count, int *objCount)
{
  HeapData *objData;
  int obj;

//for (obj = 0; obj < stats[pe].n_objs; obj++)
//if (stats[pe].objData[obj].migratable == CmiTrue) (*objCount)++; 

  objData  = new HeapData[stats->n_objs];
  *objCount = 0; 
  for(obj=0; obj < stats->n_objs; obj++) {
    LDObjData &oData = stats->objData[obj];
    int pe = stats->from_proc[obj];
    if (!oData.migratable) {
      if (!stats->procs[pe].available) 
        CmiAbort("GreedyLB cannot handle nonmigratable object on an unavial processor!\n");
      continue;
    }
    objData[*objCount].load = oData.wallTime * stats->procs[pe].pe_speed;
    objData[*objCount].pe = pe;
    objData[*objCount].id = obj;
    (*objCount)++;
  }
  
  HeapSort(objData, *objCount-1, GT);
/*
for (int i=0; i<*objCount; i++)
  CmiPrintf("%f ", objData[i].load);
CmiPrintf("\n");
*/
  return objData;
}

GreedyLB::HeapData* 
GreedyLB::BuildCpuArray(BaseLB::LDStats* stats, 
                          int count, int *peCount)
{
  int pe;

  *peCount = 0;
  for (pe = 0; pe < count; pe++)
    if (stats->procs[pe].available) (*peCount)++;
  HeapData *data = new HeapData[*peCount];
  int *map = new int[count];
  
  *peCount = 0;
  for (pe=0; pe < count; pe++) {
    CentralLB::ProcStats &peData = stats->procs[pe];
 
    map[pe] = -1;
    if (peData.available) 
    {
    	data[*peCount].load = 0.0;
      data[*peCount].load += peData.bg_walltime;
      data[*peCount].pe = data[*peCount].id = pe;
      map[pe] = *peCount;
      (*peCount)++;
    }
  }

  // take non migratbale object load as background load
  for (int obj = 0; obj < stats->n_objs; obj++) 
  { 
      LDObjData &oData = stats->objData[obj];
      if (!oData.migratable)  {
        int pe = stats->from_proc[obj];
        pe = map[pe];
        if (pe==-1) 
          CmiAbort("GreedyLB: nonmigratable object on an unavail processor!\n");
        data[pe].load += oData.wallTime;
      }
  }

  // considering cpu speed
  for (pe = 0; pe<*peCount; pe++)
    data[pe].load *= stats->procs[data[pe].pe].pe_speed;

  BuildHeap(data, *peCount-1, LT);     // minHeap
  delete [] map;
  return data;
}

void GreedyLB::work(LDStats* stats)
{
  int  obj, heapSize, objCount;
  int n_pes = stats->nprocs();

  HeapData *cpuData = BuildCpuArray(stats, n_pes, &heapSize);
  HeapData *objData = BuildObjectArray(stats, n_pes, &objCount);

  if (_lb_args.debug()>1) 
    CkPrintf("[%d] In GreedyLB strategy\n",CkMyPe());
  heapSize--;
  int nmoves = 0;
  for (obj=0; obj < objCount; obj++) {
    HeapData minCpu;  
    // Operation of extracting the least loaded processor
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
      nmoves ++;
      if (_lb_args.debug()>2) 
        CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),objData[obj].id,pe,dest);
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

  if (_lb_args.debug()>0) 
    CkPrintf("[%d] %d objects migrating.\n", CkMyPe(), nmoves);

  delete [] objData;
  delete [] cpuData;
}

#include "GreedyLB.def.h"

/*@}*/




