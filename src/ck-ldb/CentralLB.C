/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <unistd.h>
#include <fcntl.h>

#include <charm++.h>
#include <LBDatabase.h>
#include "CentralLB.h"
#include "CentralLB.def.h"

CkGroupID loadbalancer;

#if CMK_LBDB_ON

void CreateCentralLB()
{
  loadbalancer = CProxy_CentralLB::ckNew();
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

  if (CkMyPe() == 0) {
    start_lb_time = CmiWallTimer();
    CkPrintf("Load balancing step %d starting at %f\n",
	     step(),start_lb_time);
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
  //  CkPrintf("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
  //	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm);
  CProxy_CentralLB(thisgroup).ReceiveStats(msg,0);
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
  const int pe = m->from_pe;
  //  CkPrintf("Stats msg received, %d %d %d %d %p\n",
  //	   pe,stats_msg_count,m->n_objs,m->serial,m);
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
    
    CLBMigrateMsg* migrateMsg = Strategy(statsDataList,clients);
    CProxy_CentralLB(thisgroup).ReceiveMigration(migrateMsg);

    // Zero out data structures for next cycle
    for(int i=0; i < clients; i++) {
      delete statsMsgsList[i];
      statsMsgsList[i]=0;
    }
    stats_msg_count=0;
    double strat_end_time = CmiWallTimer();
    CkPrintf("Strat elapsed time %f\n",strat_end_time-strat_start_time);
  }
  
}

void CentralLB::ReceiveMigration(CLBMigrateMsg *m)
{
  //  CkPrintf("[%d] in ReceiveMigration %d moves\n",CkMyPe(),m->n_moves);
  migrates_expected = 0;
  for(int i=0; i < m->n_moves; i++) {
    MigrateInfo& move = m->moves[i];
    const int me = CkMyPe();
    if (move.from_pe == me && move.to_pe != me) {
      //      CkPrintf("[%d] migrating object to %d\n",move.from_pe,move.to_pe);
      theLbdb->Migrate(move.obj,move.to_pe);
    } else if (move.from_pe != me && move.to_pe == me) {
      //      CkPrintf("[%d] expecting object from %d\n",move.to_pe,move.from_pe);
      migrates_expected++;
    }
  }
  if (migrates_expected == 0 || migrates_completed == migrates_expected)
    MigrationDone();
  delete m;
}


void CentralLB::MigrationDone()
{
  if (CkMyPe() == 0) {
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
  //  CkPrintf("Resuming clients on PE %d\n",CkMyPe());
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

void CentralLB::dumpLDStats(LDStats* statsList, char *file)
{
  int fd = open(file, O_WRONLY|O_CREAT|O_TRUNC, 0644);
  if (fd == -1){
     perror("dumpLDStats");
     return;
  }
  for (int i=0; i<CkNumPes(); i++) 
  {
     write(fd, &statsList[i], sizeof(LDStats));
     write(fd, statsList[i].objData, sizeof(LDObjData)*statsList[i].n_objs);
     write(fd, statsList[i].commData, sizeof(LDCommData)*statsList[i].n_comm);
  }
  close(fd);
}

CentralLB::LDStats* CentralLB::loadLDStats(char *file, int pe)
{
  int fd = open(file, O_RDONLY);
  if (fd == -1){
     perror("loadLDStats");
     return NULL;
  }
  LDStats* statsList = new LDStats[pe];
  for (int i=0; i<pe; i++) 
  {
     if (read(fd, &statsList[i], sizeof(LDStats)) <= 0) { perror("loadLDStats"); return NULL; }
     statsList[i].objData = new LDObjData[statsList[i].n_objs];
     if (statsList[i].n_objs && read(fd, statsList[i].objData, sizeof(LDObjData)*statsList[i].n_objs) <= 0) { perror("loadLDStats"); return NULL; }
     statsList[i].commData = new LDCommData[statsList[i].n_comm];
     if (statsList[i].n_comm && read(fd, statsList[i].commData, sizeof(LDCommData)*statsList[i].n_comm) <= 0) { perror("loadLDStats"); return NULL; }
  }
  close(fd);
  return statsList;
}

void* CLBStatsMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = size + array[0] * sizeof(LDObjData) 
    + array[1] * sizeof(LDCommData);

  CLBStatsMsg* ret =
    (CLBStatsMsg*)(CkAllocMsg(msgnum,totalsize,priobits));

  ret->objData = (LDObjData*)(((char*)(ret) + size));
  ret->commData = (LDCommData*)(ret->objData + array[0]);

  return (void*)(ret);
}

void* CLBStatsMsg::pack(CLBStatsMsg* m)
{
  m->objData = 
    (LDObjData*)((char*)(m->objData) - (char*)(&m->objData));
  m->commData = 
    (LDCommData*)((char*)(m->commData) - (char*)(&m->commData));
  return (void*)(m);
}

CLBStatsMsg* CLBStatsMsg::unpack(void *m)
{
  CLBStatsMsg* ret_val = (CLBStatsMsg*)(m);

  ret_val->objData = 
    (LDObjData*)((char*)(&ret_val->objData) + (size_t)(ret_val->objData));
  ret_val->commData = 
    (LDCommData*)((char*)(&ret_val->commData) + (size_t)(ret_val->commData));
  return ret_val;
}

void* CLBMigrateMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = size + array[0] * sizeof(CentralLB::MigrateInfo);

  CLBMigrateMsg* ret =
    (CLBMigrateMsg*)(CkAllocMsg(msgnum,totalsize,priobits));

  ret->moves = (CentralLB::MigrateInfo*) ((char*)(ret)+ size);

  return (void*)(ret);
}

void* CLBMigrateMsg::pack(CLBMigrateMsg* m)
{
  m->moves = (CentralLB::MigrateInfo*)
    ((char*)(m->moves) - (char*)(&m->moves));

  return (void*)(m);
}

CLBMigrateMsg* CLBMigrateMsg::unpack(void *m)
{
  CLBMigrateMsg* ret_val = (CLBMigrateMsg*)(m);

  ret_val->moves = (CentralLB::MigrateInfo*)
    ((char*)(&ret_val->moves) + (size_t)(ret_val->moves));

  return ret_val;
}

#endif
