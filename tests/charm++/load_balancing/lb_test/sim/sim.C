/*
Load-balancing test program:
  Orion Sky Lawlor, 10/19/1999

  Added more complex comm patterns
  Robert Brunner, 11/3/1999

*/

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include "charm++.h"
#include "LBDatabase.h"
#include "Topo.h"
#include "CentralLB.h"
#include "RandCentLB.h"
#include "RecBisectBfLB.h"
#include "RefineLB.h"
#include "GreedyCommLB.h"
#ifdef USE_METIS
#include "MetisLB.h"
#endif
#include "GreedyLB.h"
#include "NeighborLB.h"
#include "WSLB.h"
#include "GreedyRefLB.h"
#include "RandRefLB.h"

#include "sim.decl.h"

CkChareID mid;//Main ID
CkGroupID topoid;

int n_loadbalance;

#define alpha 35e-6
#define beeta 8e-9

#define N_LOADBALANCE 500 /*Times around ring until we load balance*/

int cycle_count,element_count,step_count,print_count,processor_count;
int min_us,max_us;

class main : public CBase_main {
public:
  int nDone;

  main(CkArgMsg* m);
/*
  void maindone(void);
*/
private:
  void arg_error(char* argv0);
};

static const struct {
  const char *name;//Name of strategy (on command line)
  const char *description;//Text description of strategy
} StratTable[]={
  {"none",
   "none - The null load balancer, collect data, but do nothing"},
  {"random",
   "random - Assign objects to processors randomly"},
  {"greedy",
   "greedy - Use the greedy algorithm to place heaviest object on the "
   "least-loaded processor until done"},
/*
#ifdef USE_METIS
  {"metis",
   "metis - Use Metis(tm) to partition object graph"},
#endif
*/
  {"refine",
   "refine - Move a very few objects away from most heavily-loaded processor"},
  {"greedyref",
   "greedyref - Apply greedy, then refine"},
  {"randref",
   "randref - Apply random, then refine"},
  {"comm",
   "comm - Greedy with communication"},
  {"recbf",
   "recbf - Recursive partitioning with Breadth first enumeration, with 2 nuclei"},

  {NULL,NULL}
};

int stratNo = 0;

main::main(CkArgMsg *m) 
{
  char *strategy;//String name for strategy routine
  char *topology;//String name for communication topology
  nDone=0;

  int cur_arg = 1;

  if (m->argc > cur_arg)
    processor_count=atoi(m->argv[cur_arg++]);

  if (m->argc > cur_arg)
    element_count=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    step_count=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);
  
  if (m->argc > cur_arg)
    print_count=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);
  
  if (m->argc > cur_arg)
    n_loadbalance=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    min_us=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    max_us=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    strategy=m->argv[cur_arg++];
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    topology=m->argv[cur_arg++];
  else arg_error(m->argv[0]);

  //Look up the user's strategy in table
  stratNo=0;
  while (StratTable[stratNo].name!=NULL) {
    if (0==strcasecmp(strategy,StratTable[stratNo].name)) {
      break;
    }
    stratNo++;
  }

  CkPrintf("%d processors\n",CkNumPes());
  CkPrintf("%d elements\n",element_count);
  CkPrintf("Print every %d steps\n",print_count);
  CkPrintf("Sync every %d steps\n",n_loadbalance);
  CkPrintf("First node busywaits %d usec; last node busywaits %d\n",
	   min_us,max_us);

  mid = thishandle;

  if (StratTable[stratNo].name==NULL)
    //The user's strategy name wasn't in the table-- bad!
    CkAbort("ERROR! Strategy not found!  \n");

  CentralLB * lb_ptr;

  switch(stratNo){
  case 0: lb_ptr = new CentralLB;
	  break;	 
  case 1: lb_ptr = new RandCentLB;
	       break;
  case 2: lb_ptr = new HeapCentLB;
	       break;
  case 3: lb_ptr = new RefineLB;
	       break;
  case 4: lb_ptr = new GreedyRefLB;
	       break;
  case 5: lb_ptr = new RandRefLB;
	       break;
  case 6: lb_ptr = new CommLB;
	       break;
  case 7: lb_ptr = new RecBisectBfLB;
	       break;
  default: CkPrintf("Error: Strategy not found\n");
	       break;
  }

  Topo* TopoMap;
  TopoMap = Topo::Create(element_count,topology,min_us,max_us);

  if (TopoMap == NULL)
    CkAbort("ERROR! Topology not found!  \n");
  
  CentralLB::LDStats *stats = new  CentralLB::LDStats[processor_count];
  int pe,obj,comm,comm_temp;

  for(pe = 0; pe < processor_count; pe++){
    stats[pe].total_walltime = 0.0;
    stats[pe].total_cputime = 0.0;
    stats[pe].idletime = 0.0;
    stats[pe].bg_walltime = 0.0;
    stats[pe].bg_cputime = 0.0;
    stats[pe].n_objs = 0;
    stats[pe].n_comm = 0;
    stats[pe].objData = new LDObjData[element_count/processor_count + 1];
    stats[pe].commData = NULL;
  }
  LDObjData* temp_obj;
  double total_load = 0.0;
  for(obj = 0; obj < element_count; obj++){
    temp_obj = &stats[obj%processor_count].objData[obj/processor_count];
    
    temp_obj->id.id[0] = obj;
    temp_obj->id.id[1] = 0;
    temp_obj->id.id[2] = 0;
    temp_obj->id.id[3] = 0;
    
    temp_obj->omID.id = 0;

    temp_obj->omHandle.id = temp_obj->omID;
    temp_obj->handle.omhandle = temp_obj->omHandle;
    temp_obj->handle.id = temp_obj->id;

    temp_obj->cpuTime = temp_obj->wallTime = (TopoMap->Work(obj) * n_loadbalance)*1e-6;
    total_load += temp_obj->wallTime;
    stats[obj%processor_count].n_objs++;
    stats[obj%processor_count].total_walltime = stats[obj%processor_count].total_cputime = 
      stats[obj%processor_count].total_walltime + temp_obj->wallTime;
  }

  for(pe = 0; pe < processor_count; pe++){
    comm = 0;
    for(obj = 0; obj < stats[pe].n_objs; obj++)
      comm += TopoMap->SendCount(stats[pe].objData[obj].id.id[0]);
    stats[pe].commData = new LDCommData[comm];
    stats[pe].n_comm = comm;    
    comm = 0;
    comm_temp = 0;

    for(obj = 0; obj < stats[pe].n_objs; obj++){
      comm_temp = TopoMap->SendCount(stats[pe].objData[obj].id.id[0]);
      Topo::MsgInfo *who_arr = new Topo::MsgInfo[comm_temp];
      TopoMap->SendTo(stats[pe].objData[obj].id.id[0],who_arr);
      for(int itr = 0; itr < comm_temp; itr++,comm++){
	stats[pe].commData[comm].from_proc = 0;
	stats[pe].commData[comm].to_proc = 0;
	stats[pe].commData[comm].src_proc = 0;
	stats[pe].commData[comm].dest_proc = 0;
	stats[pe].commData[comm].senderOM.id = 0;
	stats[pe].commData[comm].sender = stats[pe].objData[obj].id;
	stats[pe].commData[comm].receiverOM.id = 0;
	stats[pe].commData[comm].receiver = stats[pe].commData[comm].sender;
	stats[pe].commData[comm].receiver.id[0] = who_arr[itr].obj;
	stats[pe].commData[comm].messages = n_loadbalance;
	stats[pe].commData[comm].bytes = n_loadbalance * who_arr[itr].bytes;
      }
    }
  }
  CLBMigrateMsg* msg = lb_ptr->callStrategy(stats,processor_count);
  
  int spe=0,dpe=0,new_obj;
  LDObjData new_objData;
  CkPrintf("New length %d\n",msg->n_moves);

  for(int move =0; move < msg->n_moves; move++){
    spe = msg->moves[move].from_pe;
    dpe = msg->moves[move].to_pe;
    
    LDObjData* temp_obj_arr = new LDObjData[stats[spe].n_objs - 1];

    for(obj =0,new_obj=0; obj < stats[spe].n_objs; obj++)
      if(stats[spe].objData[obj].id.id[0] != msg->moves[move].obj.id.id[0])
	temp_obj_arr[new_obj++] = stats[spe].objData[obj];
      else
	new_objData = stats[spe].objData[obj];

    delete stats[spe].objData; 
    stats[spe].objData = temp_obj_arr;
    stats[spe].n_objs--;

    temp_obj_arr = new LDObjData[stats[dpe].n_objs + 1];
    for(obj = 0; obj < stats[dpe].n_objs; obj++)
      temp_obj_arr[obj] = stats[dpe].objData[obj];
    delete stats[dpe].objData; 
    temp_obj_arr[obj] = new_objData;
    stats[dpe].objData = temp_obj_arr;    
    stats[dpe].n_objs++;
  }

/*
  for(int move =0; move < msg->n_moves; move++){
    spe = msg->moves[move].from_pe;
    dpe = msg->moves[move].to_pe;
    for(obj =0,new_obj=0; obj < stats[spe].n_objs; obj++)
      if(stats[spe].objData[obj].id.id[0] == msg->moves[move].obj.id.id[0]){
	stats[spe].total_walltime -= stats[spe].objData[obj].wallTime;
	stats[dpe].total_walltime += stats[spe].objData[obj].wallTime;
	break;
      }
  }
*/
  double load=0.0,max_load=0.0;
  int max_pe=0;

  for(pe = 0; pe < processor_count; pe++){
    load = 0.0;
    for(obj = 0; obj < stats[pe].n_objs; obj++)
      load += stats[pe].objData[obj].wallTime;
    stats[pe].total_cputime = stats[pe].total_walltime = load;
/*    if(load > max_load){
      max_pe = pe;
      max_load = load;
    } */
//    CkPrintf("load on %d = %5.3lf\n",pe,load);
  }

/*
  for(pe = 0; pe < processor_count; pe++){
    load = stats[pe].total_walltime;
    if(load > max_load){
      max_pe = pe;
      max_load = load;
    }
//    CkPrintf("load on %d = %5.3lf\n",pe,stats[pe].total_walltime);
  }
*/

  for(pe = 0; pe < processor_count; pe++)
    for(obj = 0; obj < stats[pe].n_objs; obj++){
      int send_count = TopoMap->SendCount(stats[pe].objData[obj].id.id[0]);
      int recv_count = TopoMap->RecvCount(stats[pe].objData[obj].id.id[0]);
      Topo::MsgInfo *send_arr = new Topo::MsgInfo[send_count];
      TopoMap->SendTo(stats[pe].objData[obj].id.id[0],send_arr);
      Topo::MsgInfo *recv_arr = new Topo::MsgInfo[recv_count];
      TopoMap->RecvFrom(stats[pe].objData[obj].id.id[0],recv_arr);
      int cobj =0;
      int cdata =0;
      for(int itr = 0; itr < send_count; itr++){
	cobj = send_arr[itr].obj;
	cdata = send_arr[itr].bytes;
	int found = 0;
	for(int tempObj = 0; tempObj < stats[pe].n_objs; tempObj++)
	  if(stats[pe].objData[tempObj].id.id[0] == cobj){
	    found = 1;
	    break;
	  }
	if(!found)
	  stats[pe].total_walltime += (alpha/2)*n_loadbalance + beeta*cdata*n_loadbalance; 
      }

      for(int itr = 0; itr < recv_count; itr++){
	cobj = recv_arr[itr].obj;
	cdata = recv_arr[itr].bytes;
	int found = 0;
	for(int tempObj = 0; tempObj < stats[pe].n_objs; tempObj++)
	  if(stats[pe].objData[tempObj].id.id[0] == cobj){
	    found = 1;
	    break;
	  }
	if(!found)
	  stats[pe].total_walltime += (alpha/2)*n_loadbalance + beeta*cdata*n_loadbalance; 
    }
  } 

  for(pe = 0; pe < processor_count; pe++){
    load = 0.0;
    load = stats[pe].total_walltime;
    if(load > max_load){
      max_pe = pe;
      max_load = load;
    }
  }

  CkPrintf("\nmaximum load = %lf\n",max_load);
  CkPrintf("speedup = %5.3lf\n",total_load/max_load);
  CkExit();
}

void main::arg_error(char* argv0)
{
  CkPrintf("Usage: %s \n"
    "<processors>"    
    "<elements> <steps> <print-freq> <lb-freq> <min-dur us> <max-dur us>\n"
    "<strategy> <topology>\n"
    "<strategy> is the load-balancing strategy:\n",argv0);
  int stratNo=0;
  while (StratTable[stratNo].name!=NULL) {
    CkPrintf("  %s\n",StratTable[stratNo].description);
    stratNo++;
  }

  int topoNo = 0;
  CkPrintf("<topology> is the object connection topology:\n");
  while (TopoTable[topoNo].name) {
    CkPrintf("  %s\n",TopoTable[topoNo].desc);
    topoNo++;
  }

  CkPrintf("\n"
	   " The program creates a ring of element_count array elements,\n"
	   "which all compute and send to their neighbor cycle_count.\n"
	   "Computation proceeds across the entire ring simultaniously.\n");
  CkExit();
}

#include "sim.def.h"

