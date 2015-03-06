/**************************************************************************
** Amit Sharma (asharma6@uiuc.edu)
** November 23, 2004
**
** This is a topology conscious load balancer.
** It migrates objects to new processors based on the topology in which the processors are connected.
****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include "TopoCentLB.decl.h"
#include "TopoCentLB.h"

#define alpha PER_MESSAGE_SEND_OVERHEAD_DEFAULT  /*Startup time per message, seconds*/
#define beta PER_BYTE_SEND_OVERHEAD_DEFAULT     /*Long-message time per byte, seconds*/
#define DEG_THRES 0.50

//#define MAX_EDGE
//#define RAND_COMM
#define make_mapping 0

CreateLBFunc_Def(TopoCentLB,"Balance objects based on the network topology")


/*static void lbinit (void)
{
  LBRegisterBalancer ("TopoCentLB",
		      CreateTopoCentLB,
		      AllocateTopoCentLB,
		      "Balance objects based on the network topology");
}*/


TopoCentLB::TopoCentLB(const CkLBOptions &opt) : CBase_TopoCentLB (opt)
{
  lbname = "TopoCentLB";
  if (CkMyPe () == 0) {
    CkPrintf ("[%d] TopoCentLB created\n",CkMyPe());
  }
}


bool TopoCentLB::QueryBalanceNow (int _step)
{
  return true;
}

TopoCentLB::~TopoCentLB(){
	if(topo) delete topo;
}

/*This routine partitions the task graph minimizing the communication and balancing the object load on all partitions*/
/*It uses METIS library to accomplish that*/
void TopoCentLB::computePartitions(CentralLB::LDStats *stats,int count,int *newmap)
{
	
  int numobjs = stats->n_objs;
	int i, j, m;

  // allocate space for the computing data
  double *objtime = new double[numobjs];
  int *objwt = new int[numobjs];
  int *origmap = new int[numobjs];
  LDObjHandle *handles = new LDObjHandle[numobjs];
  
	for(i=0;i<numobjs;i++) {
    objtime[i] = 0.0;
    objwt[i] = 0;
    origmap[i] = 0;
  }

  //Prepare compute loads for METIS library
  for (i=0; i<stats->n_objs; i++) {
    LDObjData &odata = stats->objData[i];
    if (!odata.migratable) 
      CmiAbort("MetisLB doesnot dupport nonmigratable object.\n");
    int frompe = stats->from_proc[i];
    origmap[i] = frompe;
    objtime[i] = odata.wallTime*stats->procs[frompe].pe_speed;
    handles[i] = odata.handle;
  }

  // to convert the weights on vertices to integers
  double max_objtime = objtime[0];
  for(i=0; i<numobjs; i++) {
    if(max_objtime < objtime[i])
      max_objtime = objtime[i];
  }
	int maxobj=0;
	int totalwt=0;
  double ratio = 1000.0/max_objtime;
  for(i=0; i<numobjs; i++) {
      objwt[i] = (int)(objtime[i]*ratio);
			if(maxobj<objwt[i])
				maxobj=objwt[i];
			totalwt+=objwt[i];
  }
	
  int **comm = new int*[numobjs];
  for (i=0; i<numobjs; i++) {
    comm[i] = new int[numobjs];
    for (j=0; j<numobjs; j++)  {
      comm[i][j] = 0;
    }
  }

  //Prepare communication for METIS library
  const int csz = stats->n_comm;
  for(i=0; i<csz; i++) {
      LDCommData &cdata = stats->commData[i];
      //if(cdata.from_proc() || cdata.receiver.get_type() != LD_OBJ_MSG)
        //continue;
			if(!cdata.from_proc() && cdata.receiver.get_type() == LD_OBJ_MSG){
      	int senderID = stats->getHash(cdata.sender);
      	int recverID = stats->getHash(cdata.receiver.get_destObj());
      	CmiAssert(senderID < numobjs);
      	CmiAssert(recverID < numobjs);
				comm[senderID][recverID] += cdata.messages;
      	comm[recverID][senderID] += cdata.messages;
				//Use bytes or messages -- do i include messages for objlist too...??
			}
			else if (cdata.receiver.get_type() == LD_OBJLIST_MSG) {
				//CkPrintf("in objlist..\n");
        int nobjs;
        LDObjKey *objs = cdata.receiver.get_destObjs(nobjs);
        int senderID = stats->getHash(cdata.sender);
        for (j=0; j<nobjs; j++) {
           int recverID = stats->getHash(objs[j]);
           if((senderID == -1)||(recverID == -1))
              if (_lb_args.migObjOnly()) continue;
              else CkAbort("Error in search\n");
           comm[senderID][recverID] += cdata.messages;
           comm[recverID][senderID] += cdata.messages;
        }
			}
		}

// ignore messages sent from an object to itself
  for (i=0; i<numobjs; i++)
    comm[i][i] = 0;

  // construct the graph in CSR format
  int *xadj = new int[numobjs+1];
  int numedges = 0;
  for(i=0;i<numobjs;i++) {
    for(j=0;j<numobjs;j++) {
      if(comm[i][j] != 0)
        numedges++;
    }
  }
  int *adjncy = new int[numedges];
  int *edgewt = new int[numedges];
	int factor = 10;
  xadj[0] = 0;
  int count4all = 0;
  for (i=0; i<numobjs; i++) {
    for (j=0; j<numobjs; j++) { 
      if (comm[i][j] != 0) { 
        adjncy[count4all] = j;
        edgewt[count4all++] = comm[i][j]/factor;
      }
    }
    xadj[i+1] = count4all;
  }

  //Call METIS routine
  int wgtflag = 3; // Weights both on vertices and edges
  int numflag = 0; // C Style numbering
  int options[5];
  int edgecut;
  options[0] = 0;

  if (count < 1) {
    CkPrintf("error: Number of Pe less than 1!");
  }
  else if (count == 1) {
   	for(m=0;m<numobjs;m++) 
			newmap[i] = origmap[i];
  }
  else {
		/*
  	if (count > 8)
			METIS_PartGraphKway(&numobjs, xadj, adjncy, objwt, edgewt, 
			    &wgtflag, &numflag, &count, options, 
			    &edgecut, newmap);
	  else
			METIS_PartGraphRecursive(&numobjs, xadj, adjncy, objwt, edgewt, 
				 &wgtflag, &numflag, &count, options, 
				 &edgecut, newmap);
		*/
	 	METIS_PartGraphRecursive(&numobjs, xadj, adjncy, objwt, edgewt,
                                 &wgtflag, &numflag, &count, options,
                                 &edgecut, newmap);
  }
	 
 
  //Debugging code: Checking load on each partition
  if(_lb_args.debug() >=2){
	  int total=0;
	  int *chkwt = new int[count];
	  for(i=0;i<count;i++)
		  chkwt[i]=0;
	  for(i=0;i<numobjs;i++){
	  	chkwt[newmap[i]] += objwt[i];
		  total += objwt[i];
	  }
	  for(i=0;i<count;i++)
		  CkPrintf("%d -- %d\n",i,chkwt[i]);
	  CkPrintf("Totalwt of all partitions after call to METIS:%d, Avg is %d\n",total,total/count);
  }

  //Clean up all the variables allocated in this routine
  for(i=0;i<numobjs;i++)
    delete[] comm[i];
  delete[] comm;
  delete[] objtime;
  delete[] xadj;
  delete[] adjncy;
  delete[] objwt;
  delete[] edgewt;
	delete[] handles;
  delete[] origmap;

}

int TopoCentLB::findMaxObjs(int *map,int totalobjs,int count)
{
	int *max_num = new int[count];
	int i;
	int maxobjs=0;
	
	for(i=0;i<count;i++)
		max_num[i]=0;
		
	for(i=0;i<totalobjs;i++)
		max_num[map[i]]++;
	
	for(i=0;i<count;i++)
		if(max_num[i]>maxobjs)
			maxobjs = max_num[i];
	
	delete[] max_num;

	return maxobjs;
}

void TopoCentLB::Heapify(HeapNode *heap, int node, int heapSize)
{
  int left = 2*node+1;
  int right = 2*node+2;
  int xchange;
	
  if (left < heapSize && (heap[left].key > heap[node].key))
    xchange = left;
  else 
		xchange = node;
  
  if (right < heapSize && (heap[right].key > heap[xchange].key))
    xchange = right;

  if (xchange != node) {
    HeapNode tmp;
    tmp = heap[node];
    heap[node] = heap[xchange];
    heap[xchange] = tmp;
		heapMapping[heap[node].node]=node;
		heapMapping[heap[xchange].node]=xchange;
    Heapify(heap,xchange,heapSize);
  }    
}


TopoCentLB::HeapNode TopoCentLB::extractMax(HeapNode *heap,int *heapSize){

	if(*heapSize < 1)
		CmiAbort("Empty Heap passed to extractMin!\n");

	HeapNode max = heap[0];
	heap[0] = heap[*heapSize-1];
	heapMapping[heap[0].node]=0;
	*heapSize = *heapSize - 1;
	Heapify(heap,0,*heapSize);
	return max;
}

void TopoCentLB::BuildHeap(HeapNode *heap,int heapSize){
	for(int i=heapSize/2; i >= 0; i--)
		Heapify(heap,i,heapSize);
}

void TopoCentLB :: increaseKey(HeapNode *heap,int i,double wt){
	if(wt != -1.00){
		#ifdef MAX_EDGE
			if(wt>heap[i].key)
				heap[i].key = wt;
		#else
			heap[i].key += wt;
		#endif
	}
	int parent = (i-1)/2;
	
	if(heap[parent].key >= heap[i].key)
		return;
	else {
		HeapNode tmp = heap[parent];
		heap[parent] = heap[i];
		heap[i] = tmp;
		heapMapping[heap[parent].node]=parent;
		heapMapping[heap[i].node]=i;
		increaseKey(heap,parent,-1.00);
	}
}

/*This routine implements the algorithm used to produce the partition-processor mapping*/
/*The algorithm uses an idea similar to the standard MST algorithm*/
void TopoCentLB :: calculateMST(PartGraph *partgraph,LBTopology *topo,int *proc_mapping,int max_comm_part) {

  int *inHeap;
  double *keys;
  int count = partgraph->n_nodes;
  int i=0,j=0;

  //Arrays needed for keeping information
  inHeap = new int[partgraph->n_nodes];
  keys = new double[partgraph->n_nodes];

  int *assigned_procs = new int[count];

  hopCount = new double*[count];
  for(i=0;i<count;i++){
    proc_mapping[i]=-1;
    assigned_procs[i]=0;
    hopCount[i] = new double[count];
    for(j=0;j<count;j++)
      hopCount[i][j] = 0;
  }

  //Call a topology routine to fill up hopCount
  topo->get_pairwise_hop_count(hopCount);
	
  int max_neighbors = topo->max_neighbors();
	
  HeapNode *heap = new HeapNode[partgraph->n_nodes];
  heapMapping = new int[partgraph->n_nodes];
	
  int heapSize = 0;

  for(i=0;i<partgraph->n_nodes;i++){
    heap[i].key = 0.00;
    heap[i].node = i;
    keys[i] = 0.00;
    inHeap[i] = 1;
    heapMapping[i]=i;
  }

  //Assign the max comm partition first
  heap[max_comm_part].key = 1.00;
	
  heapSize = partgraph->n_nodes;
  BuildHeap(heap,heapSize);

  int k=0,comm_cnt=0,m=0;
  int *commParts = new int[partgraph->n_nodes];
	
  //srand(count);

  while(heapSize > 0){

    /****Phase1: Extracting appropriate partition from heap****/

    HeapNode max = extractMax(heap,&heapSize);
    inHeap[max.node] = 0;

    for(i=0;i<partgraph->n_nodes;i++){
      commParts[i]=-1;
      PartGraph::Edge wt = partgraph->edges[max.node][i];
      if(wt == 0)
	continue;
      if(inHeap[i]){
#ifdef MAX_EDGE
	if(wt>keys[i])
	  keys[i]=wt;
#else
	keys[i] += wt;
#endif
        /*This part has been COMMENTED out for optimizing the code: we handle the updation using heapMapping*/
        /*array instead of searching for node in the heap everytime*/

	//Update in the heap too
	//First, find where this node is..in the heap
	/*for(j=0;j<heapSize;j++)
	  if(heap[j].node == i)
	  break;
	  if(j==heapSize)
	  CmiAbort("Some error in heap...\n");*/
	increaseKey(heap,heapMapping[i],wt);
      }
    }
		 
    /*Phase2: ASSIGNING partition to processor*/
		
    //Special case
    if(heapSize == partgraph->n_nodes-1){ //Assign max comm partition to 0th proc in the topology
      proc_mapping[max.node]=0;
      assigned_procs[0]=1;
      continue;
    }
		
    m=0;

    comm_cnt=0;

    double min_cost=-1;
    int min_cost_index=-1;
    double cost=0;
    int p=0;
    //int q=0;

    for(k=0;k<partgraph->n_nodes;k++){
      if(!inHeap[k] && partgraph->edges[k][max.node]){
	commParts[comm_cnt]=k;
	comm_cnt++;
      }
    }

    //Optimized the loop by commenting out the get_hop_count code and getting all the hop counts initially
    for(m=0;m<count;m++){
      if(!assigned_procs[m]){
	cost=0;
	for(p=0;p<comm_cnt;p++){
	  //if(!hopCount[proc_mapping[commParts[p]]][m])
	  //hopCount[proc_mapping[commParts[p]]][m]=topo->get_hop_count(proc_mapping[commParts[p]],m);
	  cost += hopCount[proc_mapping[commParts[p]]][m]*partgraph->edges[commParts[p]][max.node];
	}
	if(min_cost==-1 || cost<min_cost){
	  min_cost=cost;
	  min_cost_index=m;
	}
      }
    }

    proc_mapping[max.node]=min_cost_index;
    assigned_procs[min_cost_index]=1;
  }

  //clear up memory
  delete[] inHeap;
  delete[] keys;
  delete[] assigned_procs;
  delete[] heap;
  delete[] commParts;
}


void TopoCentLB :: work(LDStats *stats)
{
  int proc;
  int i,j;
  int n_pes = stats->nprocs();
	
  if (_lb_args.debug() >= 2) {
    CkPrintf("In TopoCentLB Strategy...\n");
  }
  
  // Make sure that there is at least one available processor.
  for (proc = 0; proc < n_pes; proc++) {
    if (stats->procs[proc].available) {
      break;
    }
  }

  if (proc == n_pes) {
    CmiAbort ("TopoCentLB: no available processors!");
  }

  
  removeNonMigratable(stats, n_pes);
  int *newmap = new int[stats->n_objs];


  if(make_mapping)
    computePartitions(stats, n_pes, newmap);
  else {
    //mapping taken from previous algo
    for(i=0;i<stats->n_objs;i++) {
      newmap[i]=stats->from_proc[i];
    }
  }

  //Debugging Code
  if(_lb_args.debug() >=2){
    CkPrintf("Map obtained from partitioning:\n");
    for(i=0;i<stats->n_objs;i++)
      CkPrintf(" %d,%d ",i,newmap[i]);
  }

  int max_objs = findMaxObjs(newmap,stats->n_objs, n_pes);
	
  partgraph = new PartGraph(n_pes, max_objs);

  //Fill up the partition graph - first fill the nodes and then, the edges

  for(i=0;i<stats->n_objs;i++)
    {
      PartGraph::Node* n = &partgraph->nodes[newmap[i]];
      n->obj_list[n->num_objs]=i;
      n->num_objs++;
    }

  int *addedComm=new int[n_pes];
  
  stats->makeCommHash();
  
  int max_comm_part=-1;
	
  double max_comm=0;

  //Try putting random amount of communication on the partition graph edges to see if things work fine
  //This also checks the running time of the algorithm since number of edges is high than in a practical scenario
#ifdef RAND_COMM
  for(i = 0; i < n_pes; i++) {
    for(j = i+1; j < n_pes; j++) {
      int val;
      if(rand()%5==0)
	val=0;
      else
	val= rand()%1000;
				
      partgraph->edges[i][j] = val;
      partgraph->edges[j][i] = val;
			
      partgraph->nodes[i].comm += val;
      partgraph->nodes[j].comm += val;
			
      if(partgraph->nodes[i].comm > max_comm){
	max_comm = partgraph->nodes[i].comm;
	max_comm_part = i;
      }
      if(partgraph->nodes[j].comm > max_comm){
	max_comm = partgraph->nodes[j].comm;
	max_comm_part = j;
      }
    }
  }
#else
  //Adding communication to the partition graph edges
  for(i=0;i<stats->n_comm;i++)
    {
      //DO I consider other comm too....i.e. to or from a processor
      LDCommData &cdata = stats->commData[i];
      if(!cdata.from_proc() && cdata.receiver.get_type() == LD_OBJ_MSG){
    	int senderID = stats->getHash(cdata.sender);
    	int recverID = stats->getHash(cdata.receiver.get_destObj());
	CmiAssert(senderID < stats->n_objs);
	CmiAssert(recverID < stats->n_objs);
		
	if(newmap[senderID]==newmap[recverID])
	  continue;
	
	if(partgraph->edges[newmap[senderID]][newmap[recverID]] == 0){
	  partgraph->nodes[newmap[senderID]].degree++;
	  partgraph->nodes[newmap[recverID]].degree++;
	}
		
	partgraph->edges[newmap[senderID]][newmap[recverID]] += cdata.bytes;
	partgraph->edges[newmap[recverID]][newmap[senderID]] += cdata.bytes;
			
	partgraph->nodes[newmap[senderID]].comm += cdata.bytes;
	partgraph->nodes[newmap[recverID]].comm += cdata.bytes;

	//Keeping track of maximum communiacting partition
	if(partgraph->nodes[newmap[senderID]].comm > max_comm){
	  max_comm = partgraph->nodes[newmap[senderID]].comm;
	  max_comm_part = newmap[senderID];
	}
	if(partgraph->nodes[newmap[recverID]].comm > max_comm){
	  max_comm = partgraph->nodes[newmap[recverID]].comm;
	  max_comm_part = newmap[recverID];
	}
      }
      else if(cdata.receiver.get_type() == LD_OBJLIST_MSG) {
	int nobjs;
    	LDObjKey *objs = cdata.receiver.get_destObjs(nobjs);
	int senderID = stats->getHash(cdata.sender);
	for(j = 0; j < n_pes; j++)
	  addedComm[j]=0;
	for (j=0; j<nobjs; j++) {
	  int recverID = stats->getHash(objs[j]);
	  if((senderID == -1)||(recverID == -1))
	    if (_lb_args.migObjOnly()) continue;
	    else CkAbort("Error in search\n");
					
	  if(newmap[senderID]==newmap[recverID])
	    continue;
	
	  if(partgraph->edges[newmap[senderID]][newmap[recverID]] == 0){
	    partgraph->nodes[newmap[senderID]].degree++;
	    partgraph->nodes[newmap[recverID]].degree++;
	  }

	  //Communication added only once for a message sent to many objects on a single processor
	  if(!addedComm[newmap[recverID]]){
	    partgraph->edges[newmap[senderID]][newmap[recverID]] += cdata.bytes;
	    partgraph->edges[newmap[recverID]][newmap[senderID]] += cdata.bytes;
	
	    partgraph->nodes[newmap[senderID]].comm += cdata.bytes;
	    partgraph->nodes[newmap[recverID]].comm += cdata.bytes;

	    if(partgraph->nodes[newmap[senderID]].comm > max_comm){
	      max_comm = partgraph->nodes[newmap[senderID]].comm;
	      max_comm_part = newmap[senderID];
	    }
	    if(partgraph->nodes[newmap[recverID]].comm > max_comm){
	      max_comm = partgraph->nodes[newmap[recverID]].comm;
	      max_comm_part = newmap[recverID];
	    }
	    //bytesComm[newmap[senderID]][newmap[recverID]] += cdata.bytes;
	    //bytesComm[newmap[recverID]][newmap[senderID]] += cdata.bytes;
	    addedComm[newmap[recverID]]=1;
	  }
	}
      }

    }
#endif
	
  int *proc_mapping = new int[n_pes];
	
  delete [] addedComm;
		
  LBtopoFn topofn;

  //Parsing the command line input for getting the processor topology
  char *lbcopy = strdup(_lbtopo);
  char *ptr = strchr(lbcopy, ':');
  if (ptr!=NULL)
    ptr = strtok(lbcopy, ":");
  else
    ptr=lbcopy;

  topofn = LBTopoLookup(ptr);
  if (topofn == NULL) {
    char str[1024];
    CmiPrintf("TopoCentLB> Fatal error: Unknown topology: %s. Choose from:\n", ptr);
    printoutTopo();
    sprintf(str, "TopoCentLB> Fatal error: Unknown topology: %s", ptr);
    CmiAbort(str);
  }
  
  topo = topofn(n_pes);

  //Call the core routine to produce the partition processor mapping
  calculateMST(partgraph,topo,proc_mapping,max_comm_part);
  //Returned partition graph is a Maximum Spanning Tree -- converted in above function itself

  //Debugging code: Result of mapping partition graph onto processor graph
  if (_lb_args.debug()>1) {
    CkPrintf("Resultant mapping..(partition,processor)\n");
    for(i = 0; i < n_pes; i++)
      CkPrintf("%d,%d\n",i,proc_mapping[i]);
  }

  //Store the result in the load balancing database
  int pe;
  PartGraph::Node* n;
  for(i = 0; i < n_pes; i++){
    pe = proc_mapping[i];
    n = &partgraph->nodes[i];
    for(j=0;j<n->num_objs;j++){
      stats->to_proc[n->obj_list[j]] = pe;
      if (_lb_args.debug()>1) 
        CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),n->obj_list[j],stats->from_proc[n->obj_list[j]],pe);
    }
  }

  delete[] newmap;
  delete[] proc_mapping;
  //Delete hopCount
  for(i = 0; i < n_pes; i++)
    delete[] hopCount[i];

  delete[] hopCount;
  delete[] heapMapping;
	
  delete partgraph;
}

#include "TopoCentLB.def.h"
