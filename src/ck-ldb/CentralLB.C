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
#include "CentralLB.h"
#include "CentralLB.def.h"

#define  DEBUGF(x)    // CmiPrintf x;

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
  DEBUGF(("[%d] CentralLB At Sync step %d!!!!\n",CkMyPe(),mystep));

  if (!QueryBalanceNow(step())) {
    MigrationDone();
    return;
  }
  thisProxy [CkMyPe()].ProcessAtSync();
}

void CentralLB::ProcessAtSync()
{
  if (CkMyPe() == cur_ld_balancer) {
    start_lb_time = CmiWallTimer();
    CmiPrintf("Load balancing step %d starting at %f in %d\n",
    step(),start_lb_time, cur_ld_balancer);
  }
  // Send stats
  const int osz = theLbdb->GetObjDataSz();
  const int csz = theLbdb->GetCommDataSz();
  
  int npes = CkNumPes();
  CLBStatsMsg* msg = new(osz, csz, npes, 0) CLBStatsMsg;
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
  DEBUGF(("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
  	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm));

  // Scheduler PART.

  if(CkMyPe() == 0){
      int num_proc = CkNumPes();
      for(int proc = 0; proc < num_proc; proc++){
	  msg->avail_vector[proc] = avail_vector[proc];
      } 
      msg->next_lb = new_ld_balancer;
  }

  thisProxy [cur_ld_balancer].ReceiveStats(msg);
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
//  CkPrintf("Stats msg received, %d %d %d %d %p\n",
//  	   pe,stats_msg_count,m->n_objs,m->serial,m);

  if((pe == 0) && (CkMyPe() != 0)){
      new_ld_balancer = m->next_lb;
      int num_proc = CkNumPes();
      for(int proc = 0; proc < num_proc; proc++)
	  avail_vector[proc] = m->avail_vector[proc]; 
  }

  DEBUGF(("ReceiveStats from %d step: %d\n", pe, mystep));
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

//    CkPrintf("Before setting bitmap\n");
    for(proc = 0; proc < clients; proc++)
      statsDataList[proc].available = (CmiBool)avail_vector[proc];
    
//    CkPrintf("Before Calling Strategy\n");

    LBMigrateMsg* migrateMsg = Strategy(statsDataList,clients);

//    CkPrintf("returned successfully\n");
    int num_proc = CkNumPes();

    for(proc = 0; proc < num_proc; proc++)
	migrateMsg->avail_vector[proc] = avail_vector[proc];
    migrateMsg->next_lb = new_ld_balancer;

//    CkPrintf("calling recv migration\n");
    thisProxy.ReceiveMigration(migrateMsg);

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

void CentralLB::ReceiveMigration(LBMigrateMsg *m)
{
  DEBUGF(("[%d] in ReceiveMigration %d moves\n",CkMyPe(),m->n_moves));
  migrates_expected = 0;
  for(int i=0; i < m->n_moves; i++) {
    MigrateInfo& move = m->moves[i];
    const int me = CkMyPe();
    if (move.from_pe == me && move.to_pe != me) {
      DEBUGF(("[%d] migrating object to %d\n",move.from_pe,move.to_pe));
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
    // CkPrintf("Load balancing step %d finished at %f duration %f\n",
	     // step(),end_lb_time,end_lb_time - start_lb_time);
  }
  migrates_completed = 0;
  migrates_expected = -1;
  // Increment to next step
  mystep++;
  thisProxy [CkMyPe()].ResumeClients();
}

void CentralLB::ResumeClients()
{
  DEBUGF(("Resuming clients on PE %d\n",CkMyPe()));
  theLbdb->ResumeClients();
}

LBMigrateMsg* CentralLB::Strategy(LDStats* stats,int count)
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
      CkPrintf("     id = %d\n",odata[i].id().id[0]);
      CkPrintf("  OM id = %d\n",odata[i].omID().id);
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
      
      if (cdata[i].from_proc())
	CkPrintf("    sender PE = %d\n",cdata[i].src_proc);
      else
	CkPrintf("    sender id = %d:%d\n",
		 cdata[i].senderOM.id,cdata[i].sender.id[0]);

      if (cdata[i].to_proc())
	CkPrintf("  receiver PE = %d\n",cdata[i].dest_proc);
      else	
	CkPrintf("  receiver id = %d:%d\n",
		 cdata[i].receiverOM.id,cdata[i].receiver.id[0]);
      
      CkPrintf("     messages = %d\n",cdata[i].messages);
      CkPrintf("        bytes = %d\n",cdata[i].bytes);
    }
  }

  int sizes=0;
  LBMigrateMsg* msg = new(&sizes,1) LBMigrateMsg;
  msg->n_moves = 0;

  return msg;
}

#endif

/*@}*/
