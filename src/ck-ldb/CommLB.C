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

  rewritten by Gengbin Zheng to use the new load balancer database and hash table;
  modified to recognize the nonmigratable attrib of an object 
  by Gengbin Zheng, 7/28/2003
*/

#include <charm++.h>
#include <stdio.h>

#if CMK_LBDB_ON

#include "cklists.h"

#include "CommLB.h"

#define alpha PER_MESSAGE_SEND_OVERHEAD  /*Startup time per message, seconds*/
#define beeta PER_BYTE_SEND_OVERHEAD     /*Long-message time per byte, seconds*/

CreateLBFunc_Def(CommLB);

static void lbinit(void) {
  LBRegisterBalancer("CommLB", CreateCommLB, "Greedy algorithm which takes communication graph into account");
}

#include "CommLB.def.h"

#include "manager.h"

CommLB::CommLB(const CkLBOptions &opt): CentralLB(opt)
{
    lbname = "CommLB";
    if (CkMyPe() == 0)
	CkPrintf("[%d] CommLB created\n",CkMyPe());
    manager_init();
}

CommLB::CommLB(CkMigrateMessage *m):CentralLB(m) {
    lbname = "CommLB";
    manager_init();
}

CmiBool CommLB::QueryBalanceNow(int _step)
{
    //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
    return CmiTrue;
}

// assign id to processor pe, load including both computation and communication
void CommLB::alloc(int pe,int id,double load){
    //  CkPrintf("alloc %d ,%d\n",pe,id);
    alloc_array[npe][id] = 1.0;
    alloc_array[pe][id] = load;
    alloc_array[pe][nobj] += load;
}

double CommLB::compute_com(int id, int pe){
    int j,com_data=0,com_msg=0;
    double total_time;
    graph * ptr;
    
    ptr = object_graph[id].next;
    
    for(j=0;(j<2*nobj)&&(ptr != NULL);j++,ptr=ptr->next){
	int destObj = ptr->id;
	if(alloc_array[npe][destObj] == 0.0)  // this obj has not been assigned
	    continue;
	if(alloc_array[pe][destObj] > 0.0)    // this obj is assigned to same pe
	    continue;
	com_data += ptr->data;
	com_msg += ptr->nmsg;
    }
    
    total_time = alpha*com_msg + beeta*com_data;
    return total_time;
}

void CommLB::update(int id, int pe){
    graph * ptr = object_graph[id].next;
    
    for(int j=0;(j<2*nobj)&&(ptr != NULL);j++,ptr=ptr->next){
	int destObj = ptr->id;
	if(alloc_array[npe][destObj] == 0.0)  // this obj has been assigned
	    continue;
	if(alloc_array[pe][destObj] > 0.0)    // this obj is assigned to same pe
	    continue;
	int destPe = stats->to_proc[destObj];
	int com_data = ptr->data;
	int com_msg = ptr->nmsg;
        double total_time = alpha*com_msg + beeta*com_data;
        alloc_array[destPe][nobj] += total_time;
    }
}

// add comm between obj x and y
// two direction
void CommLB::add_graph(int x, int y, int data, int nmsg){
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
  
void init(double **a, graph * object_graph, int l, int b){
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

void CommLB::work(CentralLB::LDStats* _stats, int count)
{
    int pe,obj,com;
    ObjectRecord *x;
    
    //  CkPrintf("[%d] CommLB strategy\n",CkMyPe());
    stats = _stats; 
    npe = count;
    nobj = stats->n_objs;

    // nmigobj is calculated as the number of migratable objects
    // ObjectHeap maxh is of size nmigobj
    nmigobj = stats->n_migrateobjs;

    stats->makeCommHash();
    
    alloc_array = new double *[count+1];
    
    object_graph = new graph[nobj];

    for(pe=0;pe <= count;pe++)
	alloc_array[pe] = new double[nobj +1];

    init(alloc_array,object_graph,npe,nobj);

    // only build heap with migratable objects, 
    // mapping nonmigratable objects to the same processors
    ObjectHeap maxh(nmigobj+1);
    for(obj=0; obj < stats->n_objs; obj++) {
      LDObjData &objData = stats->objData[obj];
      int onpe = stats->from_proc[obj];
      if (!objData.migratable) {
        alloc(onpe, obj, objData.wallTime);
        if (!stats->procs[onpe].available) {
	  CmiAbort("Load balancer is not be able to move a nonmigratable object out of an unavailable processor.\n");
        }
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

    int xcoord=0,ycoord=0;

    for(com =0; com< stats->n_comm;com++) {
	 LDCommData &commData = stats->commData[com];
	 if((!commData.from_proc())&&(commData.recv_type()==LD_OBJ_MSG))
	 {
		xcoord = stats->getHash(commData.sender);
		ycoord = stats->getHash(commData.receiver.get_destObj());
		if((xcoord == -1)||(ycoord == -1))
		    if (_lb_ignoreBgLoad) continue;
		    else CkAbort("Error in search\n");
		add_graph(xcoord,ycoord,commData.bytes, commData.messages);
	 }
    }

    int id,maxid,spe=0,minpe=0,mpos;
    double temp,total_time,min_temp;
    /*
    for(pe=0;pe < count;pe++)
	CkPrintf("avail for %d = %d\n",pe,stats[pe].available);
    */
    for(pe=0;pe < count;pe++)
	if(stats->procs[pe].available == 1)
	    break;

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
	
	for(pe = first_avail_pe +1; pe < count; pe++){
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
	
	// now that maxid assigned to minpe, update other pes load
	update(maxid, minpe);

	//    CkPrintf("before 2nd alloc\n");
	alloc(minpe,maxid,x->load + min_temp);
        stats->assign(maxid, minpe);
	
	if(minpe != spe){
	    //      CkPrintf("**Moving from %d to %d\n",spe,minpe);
	    CmiAssert(stats->from_proc[mpos] == spe);
	    stats->to_proc[mpos] = minpe;
	}
	delete x;
    }
    
    // free up memory
    for(pe=0;pe <= count;pe++)
	delete alloc_array[pe];
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


#endif

/*@}*/

