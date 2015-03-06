/**
 * \addtogroup CkLdb
*/
/*@{*/

/*
  status:
  * knows nonmigratable attribe
  * doesnot support processor avail bitvector

  rewritten by Gengbin Zheng to use the new load balancer database and comm hash table;
  modified to recognize the nonmigratable attrib of an object 
  by Gengbin Zheng, 7/28/2003

*/

#include "CommLB.h"

#define alpha 35e-6
#define beta 8.5e-9

#define LOWER_FACTOR 0.33
#define UPPER_FACTOR 0.67
#define MAX_WEIGHT 5.0

CreateLBFunc_Def(CommLB, "another variation of CommLB")

CommLB::CommLB(const CkLBOptions &opt): CBase_CommLB(opt)
{
  if (CkMyPe() == 0)
    CkPrintf("[%d] CommLB created\n",CkMyPe());
  lbname = "CommLB";
}

bool CommLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return true;
}

void CommLB::alloc(int pe , int id, double load, int nmsg, int nbyte){
  alloc_array[npe][id].load = 1.0;
  alloc_array[pe][id].load = load;
  alloc_array[pe][id].nmsg = nmsg;
  alloc_array[pe][id].nbyte = nbyte;
  alloc_array[pe][nobj].load += load;
  alloc_array[pe][nobj].nmsg += nmsg;
  alloc_array[pe][nobj].nbyte += nbyte;
}

double CommLB::compute_cost(int id, int pe, int n_alloc, int &com_msg, int &com_data){
  int j;
  double total_cost, com_cost, weight=0.0;
  graph * ptr;
  double bound1,bound2;

  bound1 = LOWER_FACTOR * nobj;
  bound2 = UPPER_FACTOR * nobj;

  if(n_alloc <= (int)bound1)
    weight = MAX_WEIGHT;
  else if((n_alloc > (int)bound1)&&(n_alloc < (int)bound2))
    weight = (bound2 - n_alloc)/(bound2 - bound1) * (MAX_WEIGHT - 1) + 1;
  else if(n_alloc >= (int)bound2)
    weight = 1.0;

//  weight = MAX_WEIGHT;
  ptr = object_graph[id].next;

  com_msg = 0;
  com_data = 0;
  for(j=0;(j<2*nobj)&&(ptr != NULL);j++,ptr=ptr->next){
    if(alloc_array[npe][ptr->id].load == 0.0)
      continue;
    if(alloc_array[pe][ptr->id].load > 0.0)
      continue;
    com_data += ptr->data;
    com_msg += ptr->nmsg;
  }
  com_cost = weight * (alpha*(com_msg + alloc_array[pe][nobj].nmsg) + beta*(com_data + alloc_array[pe][nobj].nbyte));
//  CkPrintf("%d, %d \n",com_data,com_msg);
  total_cost = alloc_array[pe][nobj].load + com_cost;
  return total_cost;
}

void CommLB::add_graph(int x, int y, int data, int nmsg){
  graph * ptr, *temp;

//  CkPrintf("Add graph : %d,%d", data, nmsg);
  ptr = &(object_graph[x]);  
  for(;ptr->next != NULL; ptr = ptr->next);
  
  temp = new graph;
  
  temp->id = y;
  temp->data = data;
  temp->nmsg = nmsg;
  temp->next = NULL;

  ptr->next = temp;

  ptr = &(object_graph[y]);  
  for(;ptr->next != NULL; ptr = ptr->next);
  
  temp = new graph;
  
  temp->id = x;
  temp->data = data;
  temp->nmsg = nmsg;
  temp->next = NULL;

  ptr->next = temp;
}
  
 
void init(alloc_struct **a, graph * object_graph, int l, int b){
  int i,j;

  for(i=0;i<l+1;i++)
    for(j=0;j<b+1;j++){
      a[i][j].load = 0.0;
      a[i][j].nbyte = 0;
      a[i][j].nmsg = 0;
    }
      
  for(j=0;j<b;j++){
    object_graph[j].data = 0;
    object_graph[j].nmsg = 0;
    object_graph[j].next = NULL;
  }
}

void CommLB::work(LDStats* stats)
{
  int pe,obj,com;
  double mean_load =0.0;
  ObjectRecord *x;

  //  CkPrintf("[%d] CommLB strategy\n",CkMyPe());

  nobj = stats->n_objs;
  npe = stats->nprocs();

  stats->makeCommHash();

  alloc_array = new alloc_struct *[npe + 1];

  object_graph = new graph[nobj];
  
  for(pe = 0; pe <= npe; pe++)
    alloc_array[pe] = new alloc_struct[nobj +1];

  init(alloc_array,object_graph,npe,nobj);

  ObjectHeap maxh(nobj+1);
  for(obj=0; obj < nobj; obj++) {
      LDObjData &objData = stats->objData[obj];
      int onpe = stats->from_proc[obj];
      x = new ObjectRecord;
      x->id = obj;
      x->pos = obj;
      x->val = objData.wallTime;
      x->pe = onpe;
      maxh.insert(x);
      mean_load += objData.wallTime;
  }
  mean_load /= npe;

  int xcoord=0,ycoord=0;

  for(com =0; com< stats->n_comm;com++) {
      LDCommData &commData = stats->commData[com];
      if((!commData.from_proc())&&(commData.recv_type()==LD_OBJ_MSG)){
	xcoord = stats->getHash(commData.sender); 
	ycoord = stats->getHash(commData.receiver.get_destObj());
	if((xcoord == -1)||(ycoord == -1))
	  if (_lb_args.ignoreBgLoad()) continue;
	  else CkAbort("Error in search\n");
	add_graph(xcoord,ycoord,commData.bytes, commData.messages);	
      }
  }
  
  int id,maxid,spe=0,minpe=0,mpos;
  double temp_cost,min_cost;

  pe = 0;
  x  = maxh.deleteMax();
  maxid = x->id;
  spe = x->pe;
  mpos = x->pos;
  
  alloc(pe,maxid,stats->objData[mpos].wallTime,0,0);
  if(pe != spe){
    //      CkPrintf("**Moving from %d to %d\n",spe,pe);
    CmiAssert(stats->from_proc[mpos] == spe);
    stats->to_proc[mpos] = pe;
  }

  int out_msg,out_byte,min_msg,min_byte;

  for(id = 1;id<nobj;id++){
    x  = maxh.deleteMax();   

    maxid = x->id;
    spe = x->pe;
    mpos = x->pos;
    LDObjData &objData = stats->objData[mpos];

    if (!objData.migratable) {
      if (!stats->procs[spe].available) {
	  CmiAbort("Load balancer is not be able to move a nonmigratable object out of an unavailable processor.\n");
      }
      temp_cost = compute_cost(maxid,spe,id,out_msg,out_byte);
      alloc(spe, maxid, x->val, out_msg, out_byte);
      continue;
    }

    for(pe =0; pe < npe; pe++)
      if((alloc_array[pe][nobj].load <= mean_load)||(id >= UPPER_FACTOR*nobj))
	break;
    CmiAssert(pe < npe);

    temp_cost = compute_cost(maxid,pe,id,out_msg,out_byte);
    min_cost = temp_cost;
    minpe = pe;
    min_msg = out_msg;
    min_byte = out_byte;
    pe++;
    for(; pe < npe; pe++) {
      if((alloc_array[pe][nobj].load > mean_load) && (id < UPPER_FACTOR*nobj))
	continue;
      temp_cost = compute_cost(maxid,pe,id,out_msg,out_byte);
      if(min_cost > temp_cost){
	minpe = pe;
	min_cost = temp_cost;
	min_msg = out_msg;
	min_byte = out_byte;
      }
    }
    CmiAssert(minpe < npe);

    alloc(minpe, maxid, x->val, min_msg, min_byte);

    if(minpe != spe){
      //      CkPrintf("**Moving from %d to %d\n",spe,minpe);
      CmiAssert(stats->from_proc[mpos] == spe);
      stats->to_proc[mpos] = minpe;
    }
  }
}

#include "CommLB.def.h"

/*@}*/


