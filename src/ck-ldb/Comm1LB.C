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
#include <stdio.h>

#if CMK_LBDB_ON

#include "cklists.h"

#include "Comm1LB.h"

#define alpha 35e-6
#define beeta 8.5e-9

#define LOWER_FACTOR 0.33
#define UPPER_FACTOR 0.67
#define MAX_WEIGHT 5.0

void CreateComm1LB()
{
  loadbalancer = CProxy_Comm1LB::ckNew();
}

static void lbinit(void) {
//        LBSetDefaultCreate(CreateComm1LB);        
  LBRegisterBalancer("Comm1LB", CreateComm1LB, "another variation of CommLB");
}

#include "Comm1LB.def.h"

Comm1LB::Comm1LB()
{
  if (CkMyPe() == 0)
    CkPrintf("[%d] Comm1LB created\n",CkMyPe());
  lbname = "Comm1LB";
}

CmiBool Comm1LB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

int Comm1LB::search(const LDObjKey &objKey) {
  const LDObjid &oid = objKey.objID();
  const LDOMid &mid = objKey.omID();
  int id, hash;
  
  hash = (oid.id[0] | oid.id[1]) % nobj;

  for(id=0;id<nobj;id++){
    if((translate[htable[(id+hash)%nobj]].objID() == oid)
       &&(translate[htable[(id+hash)%nobj]].omID().id == mid.id))
      return htable[(id + hash)%nobj];
  }
  //  CkPrintf("not found \n");
  return -1;
}

void Comm1LB::alloc(int pe , int id, double load, int nmsg, int nbyte){
  alloc_array[npe][id].load = 1.0;
  alloc_array[pe][id].load = load;
  alloc_array[pe][id].nmsg = nmsg;
  alloc_array[pe][id].nbyte = nbyte;
  alloc_array[pe][nobj].load += load;
  alloc_array[pe][nobj].nmsg += nmsg;
  alloc_array[pe][nobj].nbyte += nbyte;
}

double Comm1LB::compute_cost(int id, int pe, int n_alloc, int &com_msg, int &com_data){
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
  com_cost = weight * (alpha*(com_msg + alloc_array[pe][nobj].nmsg) + beeta*(com_data + alloc_array[pe][nobj].nbyte));
//  CkPrintf("%d, %d \n",com_data,com_msg);
  total_cost = alloc_array[pe][nobj].load + com_cost;
  return total_cost;
}

void Comm1LB::add_graph(int x, int y, int data, int nmsg){
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
  
void Comm1LB::make_hash(){
  int i, hash;
  LDObjid oid;
  
  htable = new int[nobj];
  for(i=0;i<nobj;i++)
    htable[i] = -1;
  
  for(i=0;i<nobj;i++){
    oid = translate[i].objID();
    hash = ((oid.id[0])|(oid.id[1])) % nobj;
    while(htable[hash] != -1)
      hash = (hash+1)%nobj;
    
    htable[hash] = i;
  }

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

LBMigrateMsg* Comm1LB::Strategy(CentralLB::LDStats* stats, int count)
{
  int pe,obj,com;
  double load_pe=0.0,mean_load =0.0;
  ObjectRecord *x;

  //  CkPrintf("[%d] Comm1LB strategy\n",CkMyPe());

  CkVec<MigrateInfo*> migrateInfo;

  alloc_array = new alloc_struct *[count+1];

  nobj = stats->n_objs;
  //  CkPrintf("OBJ: Before \n");

  ObjectHeap maxh(nobj+1);
  for(obj=0; obj < nobj; obj++) {
      x = new ObjectRecord;
      x->id = obj;
      x->pos = obj;
      x->load = stats->objData[obj].wallTime;
      x->pe = stats->from_proc[obj];
      maxh.insert(x);
  }
  for(pe=0; pe < count; pe++) {
     mean_load += stats->procs[pe].total_walltime;
  }
  mean_load /= count;
/*
  for(pe=0; pe < count; pe++) {
    load_pe = 0.0;
    for(obj=0; obj < stats[pe].n_objs; obj++) {
      load_pe += stats->objData[obj].data.wallTime;
      nobj++;
      x = new ObjectRecord;
      x->id = nobj -1;
      x->pos = obj;
      x->load = stats->objData[obj].data.wallTime;
      x->pe = pe;
      maxh.insert(x);
    }
    mean_load += load_pe/count;
//    CkPrintf("LOAD on %d = %5.3lf\n",pe,load_pe);
  }
*/

  npe = count;
  translate = new LDObjKey[nobj];
  int objno=0;

  for(obj=0; obj < stats->n_objs; obj++){ 
      LDObjData &oData = stats->objData[obj];
      translate[objno].omID() = oData.omID();
      translate[objno].objID() = oData.objID();
      objno++;
  }

  make_hash();

  object_graph = new graph[nobj];
  
  for(pe=0;pe <= count;pe++)
    alloc_array[pe] = new alloc_struct[nobj +1];

  init(alloc_array,object_graph,npe,nobj);

  int xcoord=0,ycoord=0;

  for(com =0; com< stats->n_comm;com++) {
      LDCommData &commData = stats->commData[com];
      if((!commData.from_proc())&&(commData.recv_type()==LD_OBJ_MSG)){
	xcoord = search(commData.sender); 
	ycoord = search(commData.receiver.get_destObj());
	if((xcoord == -1)||(ycoord == -1))
	  if (lb_ignoreBgLoad) continue;
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
    MigrateInfo* migrateMe = new MigrateInfo;
    migrateMe->obj = stats->objData[mpos].handle;
    migrateMe->from_pe = spe;
    migrateMe->to_pe = pe;
    migrateInfo.insertAtEnd(migrateMe);
  }

  int out_msg,out_byte,min_msg,min_byte;

  for(id = 1;id<nobj;id++){
    x  = maxh.deleteMax();   

    maxid = x->id;
    spe = x->pe;
    mpos = x->pos;

    for(pe =0; pe < count; pe++)
      if((alloc_array[pe][nobj].load <= mean_load)||(id >= UPPER_FACTOR*nobj))
	break;

    temp_cost = compute_cost(maxid,pe,id,out_msg,out_byte);
    min_cost = temp_cost;
    minpe = pe;
    min_msg = out_msg;
    min_byte = out_byte;
    pe++;
    for(; pe < count;pe++){
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

    alloc(minpe,maxid,x->load,min_msg,min_byte);

    if(minpe != spe){
      //      CkPrintf("**Moving from %d to %d\n",spe,minpe);
      MigrateInfo *migrateMe = new MigrateInfo;
      migrateMe->obj = stats->objData[mpos].handle;
      migrateMe->from_pe = spe;
      migrateMe->to_pe = minpe;
      migrateInfo.insertAtEnd(migrateMe);
    }
  }

  int migrate_count = migrateInfo.length();
  LBMigrateMsg* msg = new(&migrate_count,1) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(int i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  return msg;
}


#endif

/*@}*/


