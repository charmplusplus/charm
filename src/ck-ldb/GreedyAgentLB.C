/**
 * \addtogroup CkLdb
*/
/*@{*/

/*
 status:
  * support processor avail bitvector
  * support nonmigratable attrib
*/

#include <LBSimulation.h>
#include "GreedyAgentLB.h"

#define LOAD_OFFSET 0.05

CreateLBFunc_Def(GreedyAgentLB,"always assign the heaviest obj onto lightest loaded processor taking into account the topology")

/*static void lbinit(void) {
  LBRegisterBalancer("GreedyAgentLB", 
                     CreateGreedyAgentLB, 
                     AllocateGreedyAgentLB, 
                     "always assign the heaviest obj onto lightest loaded processor.");
}
*/
#include "GreedyAgentLB.def.h"

GreedyAgentLB::GreedyAgentLB(const CkLBOptions &opt): CBase_GreedyAgentLB(opt)
{
  lbname = "GreedyAgentLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] GreedyAgentLB created\n",CkMyPe());
}

bool GreedyAgentLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return true;
}

bool  GreedyAgentLB::Compare(double x, double y, HeapCmp cmp)
{
  const int test =  ((cmp == GT) ? (x > y) : (x < y));

  if (test) return true; 
  else return false;
}


void GreedyAgentLB::Heapify(HeapData *heap, int node, int heapSize, HeapCmp cmp)
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

void GreedyAgentLB::BuildHeap(HeapData *data, int heapSize, HeapCmp cmp)
{
	int i;
	for(i=heapSize/2; i >= 0; i--)
		Heapify(data, i, heapSize, cmp);
}

void GreedyAgentLB::HeapSort(HeapData *data, int heapSize, HeapCmp cmp)
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

GreedyAgentLB::HeapData* 
GreedyAgentLB::BuildObjectArray(CentralLB::LDStats* stats, 
                             int count, int *objCount)
{
  HeapData *objData;
  int obj;

//for (obj = 0; obj < stats[pe].n_objs; obj++)
//if (stats[pe].objData[obj].migratable == true) (*objCount)++; 

  objData  = new HeapData[stats->n_objs];
  *objCount = 0; 
  for(obj=0; obj < stats->n_objs; obj++) {
    LDObjData &oData = stats->objData[obj];
    int pe = stats->from_proc[obj];
    if (!oData.migratable) {
      if (!stats->procs[pe].available) 
        CmiAbort("GreedyAgentLB cannot handle nonmigratable object on an unavial processor!\n");
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

GreedyAgentLB::HeapData* 
GreedyAgentLB::BuildCpuArray(CentralLB::LDStats* stats, 
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
 
    data[*peCount].load = 0.0;
    map[pe] = -1;
    if (peData.available) 
    {
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
          CmiAbort("GreedyAgentLB: nonmigratable object on an unavail processor!\n");
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

void GreedyAgentLB::work(LDStats* stats)
{
  int  i, obj, heapSize, objCount;
  int n_pes = stats->nprocs();

  int *pemap = new int [n_pes];
  HeapData *cpuData = BuildCpuArray(stats, n_pes, &heapSize);
  HeapData *objData = BuildObjectArray(stats, n_pes, &objCount);
	
 	int max_neighbors=0;
 
 	//int simprocs = LBSimulation::simProcs;
	//CkPrintf("\nnum of procs:%d\n",simprocs);
	

	CkPrintf("num procs in stats:%d\n", n_pes);
	topologyAgent = new TopologyAgent(stats, n_pes);

	max_neighbors = topologyAgent->topo->max_neighbors();
	
  if (_lb_args.debug()) CkPrintf("In GreedyAgentLB strategy\n",CkMyPe());

  heapSize--;
	
	HeapData *minCpu = new HeapData[n_pes];
	double minLoad = 0.0;
	double loadThreshold = 0.0;
	int *trialpes = new int[n_pes + 1];
	int *trialmap = new int[n_pes];
	int *existing_map = new int[objCount];
	Agent::Elem *preferList;
	
	for(i=0;i<objCount;i++)
		existing_map[i]=-1;

	int extractIndex=0;

	//stats->makeCommHash();

	CkPrintf("before assigning objects...objcount:%d\n",objCount);
  for (obj=0; obj < objCount; obj++) {
    //HeapData minCpu;  
    // Operation of extracting the the least loaded processor
    // from the heap
    //int extractIndex=0;
		
			CkPrintf("obj count:%d\n",obj);
		for(i = 0; i <= n_pes; i++)
			trialpes[i]=-1;

		if(extractIndex==0)
			minLoad = cpuData[0].load;
		else
			minLoad = minCpu[0].load;

				//if(minLoad < 0.0)
		//	loadThreshold = minLoad*(1-LOAD_OFFSET);
		//else
		loadThreshold = minLoad*(1+LOAD_OFFSET);
		
		//CkPrintf("minload :%lf , threshold:%lf , heapSize:%d\n",minLoad,loadThreshold,heapSize);
		//We can do better by extracting from the heap only the incremental load nodes
	 	//after we have assigned the preferred node in the previous step
		//....as the others are still with us..
	
		//CkPrintf("heapsize before :%d\n",heapSize);
		/*CkPrintf("heap stats...\n");
		for(int t=0;t<=heapSize;t++)
			CkPrintf("..pe:%d,load:%f..",cpuData[t].pe,cpuData[t].load);
		*/
		while(1){
			if(cpuData[0].load > loadThreshold)
				break;
			minCpu[extractIndex]=cpuData[0];
			extractIndex++;
    	cpuData[0]=cpuData[heapSize];
    	heapSize--;
			if(heapSize==-1)
				break;
    	Heapify(cpuData, 0, heapSize, LT);    
		}
		//CkPrintf("after extracting loop....extractindex:%d,heapsize:%d\n",extractIndex,heapSize);
		//CkPrintf("trialpes...\n");
		int trialLen = 0;
		if(obj!=0){
			trialLen = max_neighbors*max_neighbors;
			if(trialLen > extractIndex)
				trialLen = extractIndex;
		}
		else
			trialLen = extractIndex;
			
		for(i=0;i<trialLen;i++){
			trialpes[i]=minCpu[i].pe;
			trialmap[minCpu[i].pe]=i;
		}
		preferList = topologyAgent->my_preferred_procs(existing_map,objData[obj].id,trialpes,1);
    // Increment the time of the least loaded processor by the cpuTime of
    // the `heaviest' object
		// Assign the object to first processor in the preferList...we may change this
		// and assign by comparing the object load with topology comm cost
		int minIndex = trialmap[preferList[0].pe];
		/*int s=0;
		for(s=0;s<trialLen;s++)
			if(minCpu[s].pe == preferList[0].pe){
				minIndex = s;
				break;
			}
		*/
		//CkPrintf("first element of prefer list...%d,%d...\n",minIndex,minCpu[minIndex].pe);
		
		//if(s==extractIndex)
			//CmiAbort("Seems as if Agent has returned corrupt value");

		const int dest = minCpu[minIndex].pe;
		const int id   = objData[obj].id;

		//CkPrintf("chk before load updation\n");
    minCpu[minIndex].load += objData[obj].load;
		//CkPrintf("chk within updation.\n");
		existing_map[id]=minCpu[minIndex].pe;
		
    //Insert object into migration queue if necessary
    //const int dest = minCpu[minIndex].pe;
    const int pe   = objData[obj].pe;
    //const int id   = objData[obj].id;
    if (dest != pe) {
      stats->to_proc[id] = dest;
      if (_lb_args.debug()>1) 
        CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),obj,pe,dest);
    }

    //Insert all the extracted processors (one with load updated) back into the heap
    /*int cnt=0;
		while(cnt<extractIndex){
			heapSize++;
    	int location = heapSize;
    	while (location>0 && cpuData[(location-1)/2].load > minCpu[cnt].load) {
      	cpuData[location] = cpuData[(location-1)/2];
      	location = (location-1)/2;
    	}
    	cpuData[location] = minCpu[cnt];
			cnt++;
		}*/
		
		heapSize++;
    extractIndex--;
		int location = heapSize;
    while (location>0 && cpuData[(location-1)/2].load > minCpu[minIndex].load) {
     	cpuData[location] = cpuData[(location-1)/2];
     	location = (location-1)/2;
    }
    cpuData[location] = minCpu[minIndex];

		for(int r=minIndex;r<extractIndex;r++)
			minCpu[r] = minCpu[r+1];
	}

  delete [] cpuData;
  delete [] objData;
	delete [] minCpu;
}



/*@}*/




