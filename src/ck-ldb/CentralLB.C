/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <charm++.h>
#include <LBDatabase.h>
#include "CentralLB.h"
#include "CentralLB.def.h"

CkGroupID loadbalancer;
char ** avail_vector_address;
int * lb_ptr;
int load_balancer_created;

#if CMK_LBDB_ON

void CreateCentralLB()
{
  loadbalancer = CProxy_CentralLB::ckNew();
}

void set_avail_vector(char * bitmap){
    int count;
    int assigned = 0;
    for(count = 0; count < CkNumPes(); count++){
	(*avail_vector_address)[count] = bitmap[count];
	if((bitmap[count] == 1) && !assigned){
	    *lb_ptr = count;
	    assigned = 1;
	}
    }   
}

void CentralLB::staticMigrated(void* data, LDObjHandle h)
{
  CentralLB *me = (CentralLB*)(data);

  me->Migrated(h);
}

void CentralLB::staticAtSync(void* data)
{
  CentralLB *me = (CentralLB*)(data);

  me->AtSync();
}

CentralLB::CentralLB()
{
  mystep = 0;
  //  CkPrintf("Construct in %d\n",CkMyPe());
  theLbdb = CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),(void*)(this));
  theLbdb->
    NotifyMigrated((LDMigratedFn)(staticMigrated),(void*)(this));

  stats_msg_count = 0;
  statsMsgsList = new CLBStatsMsg*[CkNumPes()];
  for(int i=0; i < CkNumPes(); i++)
    statsMsgsList[i] = 0;

  statsDataList = new LDStats[CkNumPes()];
  myspeed = theLbdb->ProcessorSpeed();
  theLbdb->CollectStatsOn();
  migrates_completed = 0;
  migrates_expected = -1;

  cur_ld_balancer = 0;
  new_ld_balancer = 0;
  int num_proc = CkNumPes();
  avail_vector = new char[num_proc];
  for(int proc = 0; proc < num_proc; proc++)
      avail_vector[proc] = 1;
  avail_vector_address = &(avail_vector);
  lb_ptr = &new_ld_balancer;

  load_balancer_created = 1;
}

CentralLB::~CentralLB()
{
  CkPrintf("Going away\n");
}

void CentralLB::AtSync()
{
  //  CkPrintf("[%d] CentralLB At Sync step %d!!!!\n",CkMyPe(),mystep);

  if (!QueryBalanceNow(step())) {
    MigrationDone();
    return;
  }

  if (CkMyPe() == cur_ld_balancer) {
    start_lb_time = CmiWallTimer();
    CkPrintf("Load balancing step %d starting at %f in %d\n",
    	     step(),start_lb_time, cur_ld_balancer);
  }
  // Send stats
  int sizes[2];
  const int osz = sizes[0] = theLbdb->GetObjDataSz();
  const int csz = sizes[1] = theLbdb->GetCommDataSz();
  
  CLBStatsMsg* msg = new(sizes,2) CLBStatsMsg;
  msg->from_pe = CkMyPe();
  // msg->serial = rand();
  msg->serial = CrnRand();

  theLbdb->TotalTime(&msg->total_walltime,&msg->total_cputime);
  theLbdb->IdleTime(&msg->idletime);
  theLbdb->BackgroundLoad(&msg->bg_walltime,&msg->bg_cputime);
  msg->pe_speed = myspeed;
  //  CkPrintf(
  //    "Processors %d Total time (wall,cpu) = %f %f Idle = %f Bg = %f %f\n",
  //    CkMyPe(),msg->total_walltime,msg->total_cputime,
  //    msg->idletime,msg->bg_walltime,msg->bg_cputime);

  msg->n_objs = osz;
  theLbdb->GetObjData(msg->objData);
  msg->n_comm = csz;
  theLbdb->GetCommData(msg->commData);
  theLbdb->ClearLoads();
  CkPrintf("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
  	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm);

  // Scheduler PART.

  if(CkMyPe() == 0){
      int num_proc = CkNumPes();
      for(int proc = 0; proc < num_proc; proc++){
	  msg->avail_vector[proc] = avail_vector[proc];
      } 
      msg->next_lb = new_ld_balancer;
  }

  CProxy_CentralLB(thisgroup).ReceiveStats(msg,cur_ld_balancer);
}

void CentralLB::Migrated(LDObjHandle h)
{
  migrates_completed++;
  //  CkPrintf("[%d] An object migrated! %d %d\n",
  //  	   CkMyPe(),migrates_completed,migrates_expected);
  if (migrates_completed == migrates_expected) {
    MigrationDone();
  }
}

void CentralLB::ReceiveStats(CLBStatsMsg *m)
{
  int proc;
  const int pe = m->from_pe;
  CkPrintf("Stats msg received, %d %d %d %d %p\n",
  	   pe,stats_msg_count,m->n_objs,m->serial,m);

  if((pe == 0) && (CkMyPe() != 0)){
      new_ld_balancer = m->next_lb;
      int num_proc = CkNumPes();
      for(int proc = 0; proc < num_proc; proc++)
	  avail_vector[proc] = m->avail_vector[proc]; 
  }

  if (statsMsgsList[pe] != 0) {
    CkPrintf("*** Unexpected CLBStatsMsg in ReceiveStats from PE %d ***\n",
	     pe);
  } else {
    statsMsgsList[pe] = m;
    statsDataList[pe].total_walltime = m->total_walltime;
    statsDataList[pe].total_cputime = m->total_cputime;
    statsDataList[pe].idletime = m->idletime;
    statsDataList[pe].bg_walltime = m->bg_walltime;
    statsDataList[pe].bg_cputime = m->bg_cputime;
    statsDataList[pe].pe_speed = m->pe_speed;
    statsDataList[pe].utilization = 1.0;
    statsDataList[pe].available = CmiTrue;

    statsDataList[pe].n_objs = m->n_objs;
    statsDataList[pe].objData = m->objData;
    statsDataList[pe].n_comm = m->n_comm;
    statsDataList[pe].commData = m->commData;
    stats_msg_count++;
  }

  const int clients = CkNumPes();
  if (stats_msg_count == clients) {
    double strat_start_time = CmiWallTimer();

    CkPrintf("Before setting bitmap\n");
    for(proc = 0; proc < clients; proc++)
      statsDataList[proc].available = avail_vector[proc];
    
    CkPrintf("Before Calling Strategy\n");

    CLBMigrateMsg* migrateMsg = Strategy(statsDataList,clients);

    CkPrintf("returned successfully\n");
    int num_proc = CkNumPes();

    for(proc = 0; proc < num_proc; proc++)
	migrateMsg->avail_vector[proc] = avail_vector[proc];
    migrateMsg->next_lb = new_ld_balancer;

    CkPrintf("calling recv migration\n");
    CProxy_CentralLB(thisgroup).ReceiveMigration(migrateMsg);

    // Zero out data structures for next cycle
    for(int i=0; i < clients; i++) {
      delete statsMsgsList[i];
      statsMsgsList[i]=0;
    }
    stats_msg_count=0;
    double strat_end_time = CmiWallTimer();
    //    CkPrintf("Strat elapsed time %f\n",strat_end_time-strat_start_time);
  }
  
}

void CentralLB::ReceiveMigration(CLBMigrateMsg *m)
{
  CkPrintf("[%d] in ReceiveMigration %d moves\n",CkMyPe(),m->n_moves);
  migrates_expected = 0;
  for(int i=0; i < m->n_moves; i++) {
    MigrateInfo& move = m->moves[i];
    const int me = CkMyPe();
    if (move.from_pe == me && move.to_pe != me) {
	//      CkPrintf("[%d] migrating object to %d\n",move.from_pe,move.to_pe);
      theLbdb->Migrate(move.obj,move.to_pe);
    } else if (move.from_pe != me && move.to_pe == me) {
	//  CkPrintf("[%d] expecting object from %d\n",move.to_pe,move.from_pe);
      migrates_expected++;
    }
  }
  
  cur_ld_balancer = m->next_lb;
  if((CkMyPe() == cur_ld_balancer) && (cur_ld_balancer != 0)){
      int num_proc = CkNumPes();
      for(int proc = 0; proc < num_proc; proc++)
	  avail_vector[proc] = m->avail_vector[proc];
  }

  if (migrates_expected == 0 || migrates_completed == migrates_expected)
    MigrationDone();
  delete m;
}


void CentralLB::MigrationDone()
{
  if (CkMyPe() == cur_ld_balancer) {
    double end_lb_time = CmiWallTimer();
    CkPrintf("Load balancing step %d finished at %f duration %f\n",
	     step(),end_lb_time,end_lb_time - start_lb_time);
  }
  migrates_completed = 0;
  migrates_expected = -1;
  // Increment to next step
  mystep++;
  CProxy_CentralLB(thisgroup).ResumeClients(CkMyPe());
}

void CentralLB::ResumeClients()
{
  CkPrintf("Resuming clients on PE %d\n",CkMyPe());
  theLbdb->ResumeClients();
}

CLBMigrateMsg* CentralLB::Strategy(LDStats* stats,int count)
{
  for(int j=0; j < count; j++) {
    int i;
    LDObjData *odata = stats[j].objData;
    const int osz = stats[j].n_objs;

    CkPrintf(
      "Proc %d Sp %d Total time (wall,cpu) = %f %f Idle = %f Bg = %f %f\n",
      j,stats[j].pe_speed,stats[j].total_walltime,stats[j].total_cputime,
      stats[j].idletime,stats[j].bg_walltime,stats[j].bg_cputime);
    CkPrintf("------------- Object Data: PE %d: %d objects -------------\n",
	     j,osz);
    for(i=0; i < osz; i++) {
      CkPrintf("Object %d\n",i);
      CkPrintf("     id = %d\n",odata[i].id.id[0]);
      CkPrintf("  OM id = %d\n",odata[i].omID.id);
      CkPrintf("   Mig. = %d\n",odata[i].migratable);
      CkPrintf("    CPU = %f\n",odata[i].cpuTime);
      CkPrintf("   Wall = %f\n",odata[i].wallTime);
    }

    LDCommData *cdata = stats[j].commData;
    const int csz = stats[j].n_comm;

    CkPrintf("------------- Comm Data: PE %d: %d records -------------\n",
	     j,csz);
    for(i=0; i < csz; i++) {
      CkPrintf("Link %d\n",i);
      
      if (cdata[i].from_proc)
	CkPrintf("    sender PE = %d\n",cdata[i].src_proc);
      else
	CkPrintf("    sender id = %d:%d\n",
		 cdata[i].senderOM.id,cdata[i].sender.id[0]);

      if (cdata[i].to_proc)
	CkPrintf("  receiver PE = %d\n",cdata[i].dest_proc);
      else	
	CkPrintf("  receiver id = %d:%d\n",
		 cdata[i].receiverOM.id,cdata[i].receiver.id[0]);
      
      CkPrintf("     messages = %d\n",cdata[i].messages);
      CkPrintf("        bytes = %d\n",cdata[i].bytes);
    }
  }

  int sizes=0;
  CLBMigrateMsg* msg = new(&sizes,1) CLBMigrateMsg;
  msg->n_moves = 0;

  return msg;
}

void* CLBStatsMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = size + array[0] * sizeof(LDObjData) 
    + array[1] * sizeof(LDCommData) + CkNumPes() * sizeof(char);

  CLBStatsMsg* ret =
    (CLBStatsMsg*)(CkAllocMsg(msgnum,totalsize,priobits));

  ret->objData = (LDObjData*)(((char*)(ret) + size));
  ret->commData = (LDCommData*)(ret->objData + array[0]);

  ret->avail_vector = reinterpret_cast<char *>(ret->commData + array[1]);

  return static_cast<void*>(ret);
}

void* CLBStatsMsg::pack(CLBStatsMsg* m)
{
  m->objData = 
    (LDObjData*)((char*)(m->objData) - (char*)(&m->objData));
  m->commData = 
    reinterpret_cast<LDCommData*>(reinterpret_cast<char*>(m->commData)
      - reinterpret_cast<char*>(&m->commData));

  m->avail_vector =reinterpret_cast<char*>(m->avail_vector
      - reinterpret_cast<char*>(&m->avail_vector));

  return static_cast<void*>(m);
}

CLBStatsMsg* CLBStatsMsg::unpack(void *m)
{
  CLBStatsMsg* ret_val = (CLBStatsMsg*)(m);

  ret_val->objData = 
    (LDObjData*)((char*)(&ret_val->objData) + (size_t)(ret_val->objData));
  ret_val->commData = 
    reinterpret_cast<LDCommData*>(reinterpret_cast<char*>(&ret_val->commData)
	     + reinterpret_cast<size_t>(ret_val->commData));

  ret_val->avail_vector =
    reinterpret_cast<char*>(reinterpret_cast<char*>(&ret_val->avail_vector)
			    +reinterpret_cast<size_t>(ret_val->avail_vector));

  return ret_val;
}

void* CLBMigrateMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = size + array[0] * sizeof(CentralLB::MigrateInfo) 
    + CkNumPes() * sizeof(char);

  CLBMigrateMsg* ret =
    (CLBMigrateMsg*)(CkAllocMsg(msgnum,totalsize,priobits));

  ret->moves = (CentralLB::MigrateInfo*) ((char*)(ret)+ size);

  ret->avail_vector = reinterpret_cast<char *>(ret->moves + array[0]);
  return static_cast<void*>(ret);
}

void* CLBMigrateMsg::pack(CLBMigrateMsg* m)
{
  m->moves = reinterpret_cast<CentralLB::MigrateInfo*>
    (reinterpret_cast<char*>(m->moves) - reinterpret_cast<char*>(&m->moves));

  m->avail_vector =reinterpret_cast<char*>(m->avail_vector
      - reinterpret_cast<char*>(&m->avail_vector));

  return (void*)(m);
}

CLBMigrateMsg* CLBMigrateMsg::unpack(void *m)
{
  CLBMigrateMsg* ret_val = (CLBMigrateMsg*)(m);

  ret_val->moves = reinterpret_cast<CentralLB::MigrateInfo*>
    (reinterpret_cast<char*>(&ret_val->moves) 
     + reinterpret_cast<size_t>(ret_val->moves));

  ret_val->avail_vector =
    reinterpret_cast<char*>(reinterpret_cast<char*>(&ret_val->avail_vector)
			    +reinterpret_cast<size_t>(ret_val->avail_vector));

  return ret_val;
}

#endif
