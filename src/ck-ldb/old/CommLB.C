/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <charm++.h>
#include <stdio.h>

#if CMK_LBDB_ON

#include "CkLists.h"

#include "CommLB.h"
#include "CommLB.def.h"

#define alpha 35e-6
#define beeta 8.5e-9

void CreateCommLB()
{
  loadbalancer = CProxy_CommLB::ckNew();
}

CommLB::CommLB()
{
  if (CkMyPe() == 0)
    CkPrintf("[%d] CommLB created\n",CkMyPe());
}

CmiBool CommLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

int CommLB::search(LDObjid oid, LDOMid mid){
  int id,hash;
  
  hash = (oid.id[0] | oid.id[1]) % nobj;

  for(id=0;id<nobj;id++){
    if((translate[htable[(id+hash)%nobj]].oid.id[0] == oid.id[0])&&(translate[htable[(id+hash)%nobj]].oid.id[1] == oid.id[1])&&(translate[htable[(id+hash)%nobj]].oid.id[2] == oid.id[2])&&(translate[htable[(id+hash)%nobj]].oid.id[3] == oid.id[3])&&(translate[htable[(id+hash)%nobj]].mid.id == mid.id))
      return htable[(id + hash)%nobj];
  }
  //  CkPrintf("not found \n");
  return -1;
}

void CommLB::alloc(int pe,int id,double load){
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
    if(alloc_array[npe][ptr->id] == 0.0)
      continue;
    if(alloc_array[pe][ptr->id] > 0.0)
      continue;
    com_data += ptr->data;
    com_msg += ptr->nmsg;
  }
  
  total_time = alpha*com_msg + beeta*com_data;
  return total_time;
}

void CommLB::add_graph(int x, int y, int data, int nmsg){
  graph * ptr, *temp;

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
  
void CommLB::make_hash(){
  int i, hash;
  LDObjid oid;
  
  htable = new int[nobj];
  for(i=0;i<nobj;i++)
    htable[i] = -1;
  
  for(i=0;i<nobj;i++){
    oid = translate[i].oid;
    hash = ((oid.id[0])|(oid.id[1])) % nobj;
    while(htable[hash] != -1)
      hash = (hash+1)%nobj;
    
    htable[hash] = i;
  }

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

CLBMigrateMsg* CommLB::Strategy(CentralLB::LDStats* stats, int count)
{
  int pe,obj,com;
  double load_pe=0.0;
  ObjectRecord *x;

  //  CkPrintf("[%d] CommLB strategy\n",CkMyPe());

  CkVector migrateInfo;

  alloc_array = new double *[count+1];

  nobj =0;
  for(pe=0; pe < count; pe++) 
    for(obj=0; obj < stats[pe].n_objs; obj++) 
      nobj++;
  //  CkPrintf("OBJ: Before \n");

  ObjectHeap maxh(nobj+1);
  nobj =0;
  for(pe=0; pe < count; pe++) {
    load_pe = 0.0;
    for(obj=0; obj < stats[pe].n_objs; obj++) {
      load_pe += stats[pe].objData[obj].wallTime;
      nobj++;
      x = new ObjectRecord;
      x->id = nobj -1;
      x->pos = obj;
      x->load = stats[pe].objData[obj].wallTime;
      x->pe = pe;
      maxh.insert(x);
    }
//    CkPrintf("LOAD on %d = %5.3lf\n",pe,load_pe);
  }

  npe = count;
  translate = new obj_id[nobj];
  int objno=0;

  for(pe=0; pe < count; pe++) 
    for(obj=0; obj < stats[pe].n_objs; obj++){ 
      translate[objno].mid.id = stats[pe].objData[obj].omID.id;
      translate[objno].oid.id[0] = stats[pe].objData[obj].id.id[0];
      translate[objno].oid.id[1] = stats[pe].objData[obj].id.id[1];
      translate[objno].oid.id[2] = stats[pe].objData[obj].id.id[2];
      translate[objno].oid.id[3] = stats[pe].objData[obj].id.id[3];
      objno++;
    }

  make_hash();

  object_graph = new graph[nobj];
  
  for(pe=0;pe <= count;pe++)
    alloc_array[pe] = new double[nobj +1];

  init(alloc_array,object_graph,npe,nobj);

  int xcoord=0,ycoord=0;

  for(pe=0; pe < count; pe++) 
    for(com =0; com< stats[pe].n_comm;com++)
      if((!stats[pe].commData[com].from_proc)&&(!stats[pe].commData[com].to_proc)){
	xcoord = search(stats[pe].commData[com].sender,stats[pe].commData[com].senderOM); 
	ycoord = search(stats[pe].commData[com].receiver,stats[pe].commData[com].receiverOM);
	if((xcoord == -1)||(ycoord == -1))
	  CkPrintf("Error in search\n");
	add_graph(xcoord,ycoord,stats[pe].commData[com].bytes, stats[pe].commData[com].messages);	
      }
  
  unsigned int id,maxid,spe=0,minpe=0,mpos;
  double temp,total_time,min_temp;

  pe = 0;
  x  = maxh.deleteMax();
  maxid = x->id;
  spe = x->pe;
  mpos = x->pos;
  
  alloc(pe,maxid,stats[spe].objData[mpos].wallTime);
  if(pe != spe){
    //      CkPrintf("**Moving from %d to %d\n",spe,pe);
    MigrateInfo* migrateMe = new MigrateInfo;
    migrateMe->obj = stats[spe].objData[mpos].handle;
    migrateMe->from_pe = spe;
    migrateMe->to_pe = pe;
    migrateInfo.push_back((void *)migrateMe);
  }


  for(id = 1;id<nobj;id++){
    x  = maxh.deleteMax();   

    maxid = x->id;
    spe = x->pe;
    mpos = x->pos;
    temp = compute_com(maxid,0);
    min_temp = temp;
    total_time = temp + alloc_array[0][nobj];
    minpe = 0;

    for(pe =1; pe < count;pe++){
      temp = compute_com(maxid,pe);
      if(total_time > (temp + alloc_array[pe][nobj])){
	minpe = pe;
	total_time = temp + alloc_array[pe][nobj];
	min_temp = temp;
      }
    }

    alloc(minpe,maxid,x->load + min_temp);

    if(minpe != spe){
      //      CkPrintf("**Moving from %d to %d\n",spe,minpe);
      MigrateInfo *migrateMe = new MigrateInfo;
      migrateMe->obj = stats[spe].objData[mpos].handle;
      migrateMe->from_pe = spe;
      migrateMe->to_pe = minpe;
      migrateInfo.push_back((void *)migrateMe);
    }
  }

  int migrate_count = migrateInfo.size();
  CLBMigrateMsg* msg = new(&migrate_count,1) CLBMigrateMsg;
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



