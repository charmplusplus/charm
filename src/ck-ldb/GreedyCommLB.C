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
Status:
  * support processor avail bitvector
  * support nonmigratable attrib
  * support background load

  rewritten by Gengbin Zheng to use the new load balancer database and hash table;
  modified to recognize the nonmigratable attrib of an object 
  by Gengbin Zheng, 7/28/2003
*/

#include <charm++.h>
#include <stdio.h>

#include "cklists.h"
#include "GreedyCommLB.h"
#include "manager.h"

CreateLBFunc_Def(GreedyCommLB, "Greedy algorithm which takes communication graph into account");

void GreedyCommLB::init()
{
    lbname = (char*)"GreedyCommLB";
    alpha = _lb_args.alpha();
    beeta = _lb_args.beeta();
    manager_init();
}

GreedyCommLB::GreedyCommLB(const CkLBOptions &opt): CentralLB(opt)
{
    init();
    if (CkMyPe() == 0)
	CkPrintf("[%d] GreedyCommLB created\n",CkMyPe());
}

GreedyCommLB::GreedyCommLB(CkMigrateMessage *m):CentralLB(m) {
    init();
}

CmiBool GreedyCommLB::QueryBalanceNow(int _step)
{
    //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
    return CmiTrue;
}

// assign id to processor pe, load including both computation and communication
void GreedyCommLB::alloc(int pe,int id,double load){
    //  CkPrintf("alloc %d ,%d\n",pe,id);
    alloc_array[npe][id] = 1.0;
    alloc_array[pe][id] = load;
    alloc_array[pe][nobj] += load;
}

// communication cost when obj id is put on proc pe
double GreedyCommLB::compute_com(int id, int pe){
    int j,com_data=0,com_msg=0;
    double total_time;
    graph * ptr;
    
    ptr = object_graph[id].next;
    
    for(j=0;(j<2*nobj)&&(ptr != NULL);j++,ptr=ptr->next){
	int destObj = ptr->id;
	if(alloc_array[npe][destObj] == 0.0)  // this obj has not been assigned
	    continue;
        if (stats->to_proc[destObj] == pe)    // this obj is assigned to same pe
	    continue;
	com_data += ptr->data;
	com_msg += ptr->nmsg;
    }
    
    total_time = alpha*com_msg + beeta*com_data;
    return total_time;
}

// update all communicating processors after assigning obj id on proc pe
void GreedyCommLB::update(int id, int pe){
    graph * ptr = object_graph[id].next;
    
    for(int j=0;(j<2*nobj)&&(ptr != NULL);j++,ptr=ptr->next){
	int destObj = ptr->id;
	if(alloc_array[npe][destObj] == 0.0)  // this obj has not been assigned
	    continue;
	int destPe = stats->to_proc[destObj];
        if (destPe == pe)                     // this obj is assigned to same pe
	    continue;
	int com_data = ptr->data;
	int com_msg = ptr->nmsg;
        double com_time = alpha*com_msg + beeta*com_data;
        alloc_array[destPe][destObj] += com_time;
        alloc_array[destPe][nobj] += com_time;
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
  
static void init_data(double **a, graph * object_graph, int l, int b){
    int i,j;
    
    for(i=0;i<l+1;i++)
	for(j=0;j<b+1;j++)
	    a[i][j] = 0.0;
    
    for(j=0;j<b;j++){
	object_graph[j].data = 0;
	object_graph[j].nmsg = 0;
	object_graph[j].next = NULL;
    }
}

void GreedyCommLB::work(CentralLB::LDStats* _stats, int count)
{
    int pe,obj,com;
    ObjectRecord *x;
    
    if (_lb_args.debug()) CkPrintf("In GreedyCommLB strategy\n",CkMyPe());
    stats = _stats; 
    npe = count;
    nobj = stats->n_objs;

    // nmigobj is calculated as the number of migratable objects
    // ObjectHeap maxh is of size nmigobj
    nmigobj = stats->n_migrateobjs;

    stats->makeCommHash();
    
    alloc_array = new double *[count+1];
    for(pe=0;pe <= count;pe++)
	alloc_array[pe] = new double[nobj +1];

    object_graph = new graph[nobj];

    init_data(alloc_array,object_graph,npe,nobj);

    // assign communication graph
    for(com =0; com< stats->n_comm;com++) {
         int xcoord=0,ycoord=0;
	 LDCommData &commData = stats->commData[com];
	 if((!commData.from_proc())&&(commData.recv_type()==LD_OBJ_MSG))
	 {
		xcoord = stats->getHash(commData.sender);
		ycoord = stats->getHash(commData.receiver.get_destObj());
		if((xcoord == -1)||(ycoord == -1))
		    if (_lb_args.ignoreBgLoad()) continue;
		    else CkAbort("Error in search\n");
		add_graph(xcoord,ycoord,commData.bytes, commData.messages);
	 }
    }

    // assign background load
    if (!_lb_args.ignoreBgLoad())
      for(pe=0;pe <= count;pe++)
          alloc_array[pe][nobj] = stats->procs[pe].bg_walltime;

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
	update(obj, onpe);	     // update communication cost on other pes
      }
      else {
        x = new ObjectRecord;
        x->id = obj;
        x->pos = obj;
        x->load = objData.wallTime;
        x->pe = onpe;
        maxh.insert(x);
      }
    }


    int id,maxid,spe=0,minpe=0,mpos;
    double temp,total_time,min_temp;
    /*
    for(pe=0;pe < count;pe++)
	CkPrintf("avail for %d = %d\n",pe,stats[pe].available);
    */
    pe = -1;
    for (int p=0; p<count; p++) {
      if (!stats->procs[p].available) continue;
      if (pe == -1 || alloc_array[p][nobj] < alloc_array[pe][nobj]) pe = p;
    }
    if (pe==-1) CmiAbort("LB Panic: No processor is available!");

    int first_avail_pe = pe;

    x  = maxh.deleteMax();
    maxid = x->id;
    spe = x->pe;
    mpos = x->pos;
    delete x;
    //  CkPrintf("before alloc firstpe = %d\n",pe);
    alloc(pe,maxid,stats->objData[mpos].wallTime);
    stats->assign(maxid, pe);
    if(pe != spe){
	//    CkPrintf("**Moving from %d to %d\n",spe,pe);
	CmiAssert(stats->from_proc[mpos] == spe);
	stats->to_proc[mpos] = pe;
    }

    for(id = 1;id<nmigobj;id++){
	x  = maxh.deleteMax();

	maxid = x->id;
	spe = x->pe;
	mpos = x->pos;
	temp = compute_com(maxid,first_avail_pe);
	min_temp = temp;
	total_time = temp + alloc_array[first_avail_pe][nobj];
	minpe = first_avail_pe;
	
    	// search all procs for best
	for(int k = 1; k < count; k++){
	    pe = (k+first_avail_pe)%count;
	    if(stats->procs[pe].available == 0) continue;
	    
	    temp = compute_com(maxid,pe);
	    
	    /*  CkPrintf("check id = %d, processor = %d,com = %lf, pro = %lf, comp=%lf\n", maxid,pe,temp,alloc_array[pe][nobj],total_time); */
	    if(total_time > (temp + alloc_array[pe][nobj])){
		minpe = pe;
		total_time = temp + alloc_array[pe][nobj];
		min_temp = temp;
	    }
	}
	/* CkPrintf("check id = %d, processor = %d, obj = %lf com = %lf, pro = %lf, comp=%lf\n", maxid,minpe,x->load,min_temp,alloc_array[minpe][nobj],total_time); */
	
	//    CkPrintf("before 2nd alloc\n");
        stats->assign(maxid, minpe);
	
	alloc(minpe,maxid,x->load + min_temp);

	// now that maxid assigned to minpe, update other pes load
	update(maxid, minpe);

/*
	if(minpe != spe){
	    //      CkPrintf("**Moving from %d to %d\n",spe,minpe);
	    CmiAssert(stats->from_proc[mpos] == spe);
	    stats->to_proc[mpos] = minpe;
	}
*/
	delete x;
    }
    
    // free up memory
    for(pe=0;pe <= count;pe++)
	delete [] alloc_array[pe];
    delete [] alloc_array;

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

