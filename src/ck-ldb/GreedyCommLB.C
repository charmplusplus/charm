/**
 * \addtogroup CkLdb
*/
/*@{*/

/*
Status:
  * support processor avail bitvector
  * support nonmigratable attrib
  * support background load

  rewritten by Gengbin Zheng to use the new load balancer database and hash table;
  modified to recognize the nonmigratable attrib of an object 
  by Gengbin Zheng, 7/28/2003
*/

#include "elements.h"
#include "ckheap.h"
#include "GreedyCommLB.h"
#include "manager.h"

CreateLBFunc_Def(GreedyCommLB, "Greedy algorithm which takes communication graph into account")

void GreedyCommLB::init()
{
    lbname = (char*)"GreedyCommLB";
    alpha = _lb_args.alpha();
    beta = _lb_args.beta();
    manager_init();
}

GreedyCommLB::GreedyCommLB(const CkLBOptions &opt): CBase_GreedyCommLB(opt)
{
    init();
    if (CkMyPe() == 0)
	CkPrintf("[%d] GreedyCommLB created\n",CkMyPe());
}

GreedyCommLB::GreedyCommLB(CkMigrateMessage *m):CBase_GreedyCommLB(m) {
    init();
}

bool GreedyCommLB::QueryBalanceNow(int _step)
{
    //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
    return true;
}

// assign id to processor pe, load including both computation and communication
void GreedyCommLB::alloc(int pe,int id,double load){
    //  CkPrintf("alloc %d ,%d\n",pe,id);
    assigned_array[id] = 1;
    processors[pe].load += load;
}

// communication cost when obj id is put on proc pe
double GreedyCommLB::compute_com(LDStats* stats, int id, int pe){
    int j,com_data=0,com_msg=0;
    double total_time;
    graph * ptr;
    
    ptr = object_graph[id].next;
    
    for(j=0;(j<2*nobj)&&(ptr != NULL);j++,ptr=ptr->next){
	int destObj = ptr->id;
	if(assigned_array[destObj] == 0)  // this obj has not been assigned
	    continue;
        if (stats->to_proc[destObj] == pe)    // this obj is assigned to same pe
	    continue;
	com_data += ptr->data;
	com_msg += ptr->nmsg;
    }
    
    total_time = alpha*com_msg + beta*com_data;
    return total_time;
}

// update all communicating processors after assigning obj id on proc pe
void GreedyCommLB::update(LDStats* stats, int id, int pe){
    graph * ptr = object_graph[id].next;
    
    for(int j=0;(j<2*nobj)&&(ptr != NULL);j++,ptr=ptr->next){
	int destObj = ptr->id;
	if(assigned_array[destObj] == 0)  // this obj has not been assigned
	    continue;
	int destPe = stats->to_proc[destObj];
        if (destPe == pe)                     // this obj is assigned to same pe
	    continue;
	int com_data = ptr->data;
	int com_msg = ptr->nmsg;
        double com_time = alpha*com_msg + beta*com_data;
        processors[destPe].load += com_time;
    }
}

// add comm between obj x and y
// two direction
void GreedyCommLB::add_graph(int x, int y, int data, int nmsg){
    graph * ptr, *temp;
    
    ptr = &(object_graph[x]);  
    
    temp = new graph;
    
    temp->id = y;
    temp->data = data;
    temp->nmsg = nmsg;
    temp->next = ptr->next;
    
    ptr->next = temp;
    
    ptr = &(object_graph[y]);  

    temp = new graph;
    
    temp->id = x;
    temp->data = data;
    temp->nmsg = nmsg;
    temp->next = ptr->next;
    
    ptr->next = temp;
}
  
static void init_data(int *assign, graph * object_graph, int l, int b){
    for(int obj=0;obj < b;obj++)
	assign[obj] = 0;

    for(int j=0;j<b;j++){
	object_graph[j].data = 0;
	object_graph[j].nmsg = 0;
	object_graph[j].next = NULL;
    }
}

void GreedyCommLB::work(LDStats* stats)
{
    int pe,obj,com;
    ObjectRecord *x;
    int i;
    
    if (_lb_args.debug()) CkPrintf("In GreedyCommLB strategy\n",CkMyPe());
    npe = stats->nprocs();
    nobj = stats->n_objs;

    // nmigobj is calculated as the number of migratable objects
    // ObjectHeap maxh is of size nmigobj
    nmigobj = stats->n_migrateobjs;

    stats->makeCommHash();

    assigned_array = new int[nobj];

    object_graph = new graph[nobj];

    init_data(assigned_array,object_graph,npe,nobj);

#define MAXDOUBLE   1e10;

    // processor heap
    processors = new processorInfo[npe];
    for (int p=0; p<npe; p++) {
      processors[p].Id = p;
      processors[p].backgroundLoad = stats->procs[p].bg_walltime;
      processors[p].computeLoad = 0;
      processors[p].pe_speed = stats->procs[p].pe_speed;
      if (!stats->procs[p].available) {
        processors[p].load = MAXDOUBLE;
      }
      else {
        processors[p].load = 0;
        if (!_lb_args.ignoreBgLoad())
          processors[p].load = processors[p].backgroundLoad;
      }
    }


    // assign communication graph
    for(com =0; com< stats->n_comm;com++) {
         int xcoord=0,ycoord=0;
	 LDCommData &commData = stats->commData[com];
	 if((!commData.from_proc())&&(commData.recv_type()==LD_OBJ_MSG))
	 {
		xcoord = stats->getHash(commData.sender);
		ycoord = stats->getHash(commData.receiver.get_destObj());
		if((xcoord == -1)||(ycoord == -1))
		    if (_lb_args.ignoreBgLoad() || stats->complete_flag==0) continue;
		    else CkAbort("Error in search\n");
		add_graph(xcoord,ycoord,commData.bytes, commData.messages);
	 }
         else if (commData.recv_type()==LD_OBJLIST_MSG) {
		int nobjs;
		LDObjKey *objs = commData.receiver.get_destObjs(nobjs);
		xcoord = stats->getHash(commData.sender);
		for (int i=0; i<nobjs; i++) {
		  ycoord = stats->getHash(objs[i]);
		  if((xcoord == -1)||(ycoord == -1))
		    if (_lb_args.migObjOnly()) continue;
		    else CkAbort("Error in search\n");
//printf("Multicast: %d => %d %d %d\n", xcoord, ycoord, commData.bytes, commData.messages);
		  add_graph(xcoord,ycoord,commData.bytes, commData.messages);
		}
         }
    }

    // only build heap with migratable objects, 
    // mapping nonmigratable objects to the same processors
    ObjectHeap maxh(nmigobj+1);
    for(obj=0; obj < stats->n_objs; obj++) {
      LDObjData &objData = stats->objData[obj];
      int onpe = stats->from_proc[obj];
      if (!objData.migratable) {
        if (!stats->procs[onpe].available) {
	  CmiAbort("Load balancer is not be able to move a nonmigratable object out of an unavailable processor.\n");
        }
        alloc(onpe, obj, objData.wallTime);
	update(stats, obj, onpe);	     // update communication cost on other pes
      }
      else {
        x = new ObjectRecord;
        x->id = obj;
        x->pos = obj;
        x->val = objData.wallTime;
        x->pe = onpe;
        maxh.insert(x);
      }
    }

    minHeap *lightProcessors = new minHeap(npe);
    for (i=0; i<npe; i++)
      if (stats->procs[i].available)
        lightProcessors->insert((InfoRecord *) &(processors[i]));

    int id,maxid,minpe=0;
    double temp,total_time,min_temp;
    // for(pe=0;pe < count;pe++)
    //	CkPrintf("avail for %d = %d\n",pe,stats[pe].available);

    double *pe_comm = new double[npe];
    for (int i=0; i<npe; i++) pe_comm[i] = 0.0;

    for(id = 0;id<nmigobj;id++){
	x  = maxh.deleteMax();

	maxid = x->id;

        processorInfo *donor = (processorInfo *) lightProcessors->deleteMin();
	CmiAssert(donor);
	int first_avail_pe = donor->Id;
	temp = compute_com(stats, maxid, first_avail_pe);
	min_temp = temp;
	//total_time = temp + alloc_array[first_avail_pe][nobj];
	total_time = temp + donor->load; 
	minpe = first_avail_pe;
	
    	// search all procs for best
 	// optimization: only search processors that it communicates
	// and the minimum of all others
        CkVec<int> commPes;
        graph * ptr = object_graph[maxid].next;
    
 	// find out all processors that this obj communicates
	double commload = 0.0;			// total comm load
        for(int com=0;(com<2*nobj)&&(ptr != NULL);com++,ptr=ptr->next){
	  int destObj = ptr->id;
	  if(assigned_array[destObj] == 0)  // this obj has not been assigned
	    continue;
	  int destPe = stats->to_proc[destObj];
	  if(stats->procs[destPe].available == 0) continue;
	  
	  double cload = alpha*ptr->nmsg + beta*ptr->data;
	  pe_comm[destPe] += cload;
	  commload += cload;

          int exist = 0;
	  for (int pp=0; pp<commPes.size(); pp++)
            if (destPe == commPes[pp]) { exist=1; break; }    // duplicated
	  if (!exist) commPes.push_back(destPe);
        }

	int k;
	for(k = 0; k < commPes.size(); k++){
	    pe = commPes[k];
            processorInfo *commpe = (processorInfo *) &processors[pe];
	    
	    temp = commload - pe_comm[pe];
	    
	    //CkPrintf("check id = %d, processor = %d,com = %lf, pro = %lf, comp=%lf\n", maxid,pe,temp,alloc_array[pe][nobj],total_time);
	    if(total_time > (temp + commpe->load)){
		minpe = pe;
		total_time = temp + commpe->load;
		min_temp = temp;
	    }
	}
	/* CkPrintf("check id = %d, processor = %d, obj = %lf com = %lf, pro = %lf, comp=%lf\n", maxid,minpe,x->load,min_temp,alloc_array[minpe][nobj],total_time); */
	
	//    CkPrintf("before 2nd alloc\n");
        stats->assign(maxid, minpe);
	
	alloc(minpe, maxid, x->val + min_temp);

	// now that maxid assigned to minpe, update other pes load
	update(stats, maxid, minpe);

        // update heap
 	lightProcessors->insert(donor);
	for(k = 0; k < commPes.size(); k++) {
	    pe = commPes[k];
            processorInfo *commpe = (processorInfo *) &processors[pe];
            lightProcessors->update(commpe);
	    pe_comm[pe] = 0.0;			// clear
        }

	delete x;
    }
    
    // free up memory
    delete [] pe_comm;

    delete [] processors;
    delete [] assigned_array;

    delete lightProcessors;

    for(int oindex= 0; oindex < nobj; oindex++){
      graph * ptr = &object_graph[oindex];
      ptr = ptr->next;
      
      while(ptr != NULL){
	graph *cur = ptr;
	ptr = ptr->next;
	delete cur;
      }
    }
    delete [] object_graph;

}

#include "GreedyCommLB.def.h"

/*@}*/

