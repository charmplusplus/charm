/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef  WIN32
#include <unistd.h>
#endif
#include <charm++.h>
#include <LBDatabase.h>
#include "NborBaseLB.h"
#include "NborBaseLB.def.h"

CkGroupID nborBaselb;

#if CMK_LBDB_ON

void CreateNborBaseLB()
{
  nborBaselb = CProxy_NborBaseLB::ckNew();
}

void NborBaseLB::staticMigrated(void* data, LDObjHandle h)
{
  NborBaseLB *me = (NborBaseLB*)(data);

  me->Migrated(h);
}

void NborBaseLB::staticAtSync(void* data)
{
  NborBaseLB *me = (NborBaseLB*)(data);

  me->AtSync();
}

NborBaseLB::NborBaseLB() :thisproxy(thisgroup)
{
  mystep = 0;
  theLbdb = CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
			    (void*)(this));
  theLbdb->
    NotifyMigrated((LDMigratedFn)(staticMigrated),
		   (void*)(this));


  // I had to move neighbor initialization outside the constructor
  // in order to get the virtual functions of any derived classes
  // so I'll just set them to illegal values here.
  neighbor_pes = 0;
  stats_msg_count = 0;
  statsMsgsList = 0;
  statsDataList = 0;
  migrates_completed = 0;
  migrates_expected = -1;
  mig_msgs_received = 0;
  mig_msgs = 0;

  myStats.pe_speed = theLbdb->ProcessorSpeed();
//  char hostname[80];
//  gethostname(hostname,79);
//  CkPrintf("[%d] host %s speed %d\n",CkMyPe(),hostname,myStats.pe_speed);
  myStats.from_pe = CkMyPe();
  myStats.n_objs = 0;
  myStats.objData = NULL;
  myStats.n_comm = 0;
  myStats.commData = NULL;
  receive_stats_ready = 0;

  theLbdb->CollectStatsOn();
}

NborBaseLB::~NborBaseLB()
{
  CkPrintf("Going away\n");
}

void NborBaseLB::FindNeighbors()
{
  if (neighbor_pes == 0) { // Neighbors never initialized, so init them
                           // and other things that depend on the number
                           // of neighbors
    statsMsgsList = new NLBStatsMsg*[max_neighbors()];
    for(int i=0; i < max_neighbors(); i++)
      statsMsgsList[i] = 0;
    statsDataList = new LDStats[max_neighbors()];

    neighbor_pes = new int[max_neighbors()];
    neighbors(neighbor_pes);
    mig_msgs_expected = num_neighbors();
    mig_msgs = new NLBMigrateMsg*[num_neighbors()];
  }

}

void NborBaseLB::AtSync()
{
  //  CkPrintf("[%d] NborBaseLB At Sync step %d!!!!\n",CkMyPe(),mystep);

  if (neighbor_pes == 0) FindNeighbors();

  if (!QueryBalanceNow(step()) || num_neighbors() == 0) {
    MigrationDone();
    return;
  }

  if (CkMyPe() == 0) {
    start_lb_time = CmiWallTimer();
    CkPrintf("Load balancing step %d starting at %f\n",
	     step(),start_lb_time);
  }

  NLBStatsMsg* msg = AssembleStats();

  int i;
  for(i=0; i < num_neighbors()-1; i++) {
    NLBStatsMsg* m2 = (NLBStatsMsg*) CkCopyMsg((void**)&msg);
    thisproxy [neighbor_pes[i]].ReceiveStats(m2);
  }
  if (0 < num_neighbors()) {
    thisproxy [neighbor_pes[i]].ReceiveStats(msg);
  } else delete msg;

  // Tell our own node that we are ready
  ReceiveStats((NLBStatsMsg*)0);
}

NLBStatsMsg* NborBaseLB::AssembleStats()
{
  // Get stats
  theLbdb->TotalTime(&myStats.total_walltime,&myStats.total_cputime);
  theLbdb->IdleTime(&myStats.idletime);
  theLbdb->BackgroundLoad(&myStats.bg_walltime,&myStats.bg_cputime);

  myStats.move = QueryMigrateStep(step());

  myStats.n_objs = theLbdb->GetObjDataSz();
  if (myStats.objData) delete [] myStats.objData;
  myStats.objData = new LDObjData[myStats.n_objs];
  theLbdb->GetObjData(myStats.objData);

  myStats.n_comm = theLbdb->GetCommDataSz();
  if (myStats.commData) delete [] myStats.commData;
  myStats.commData = new LDCommData[myStats.n_comm];
  theLbdb->GetCommData(myStats.commData);

  myStats.obj_walltime = myStats.obj_cputime = 0;
  for(int i=0; i < myStats.n_objs; i++) {
    myStats.obj_walltime += myStats.objData[i].wallTime;
    myStats.obj_cputime += myStats.objData[i].cpuTime;
  }    

  const int osz = theLbdb->GetObjDataSz();
  const int csz = theLbdb->GetCommDataSz();
  NLBStatsMsg* msg = new(osz, csz, 0)  NLBStatsMsg;

  msg->from_pe = CkMyPe();
  // msg->serial = rand();
  msg->serial = CrnRand();
  msg->pe_speed = myStats.pe_speed;
  msg->total_walltime = myStats.total_walltime;
  msg->total_cputime = myStats.total_cputime;
  msg->idletime = myStats.idletime;
  msg->bg_walltime = myStats.bg_walltime;
  msg->bg_cputime = myStats.bg_cputime;
  msg->obj_walltime = myStats.obj_walltime;
  msg->obj_cputime = myStats.obj_cputime;
  msg->n_objs = osz;
  theLbdb->GetObjData(msg->objData);
  msg->n_comm = csz;
  theLbdb->GetCommData(msg->commData);

  //  CkPrintf(
  //    "Proc %d speed=%d Total(wall,cpu)=%f %f Idle=%f Bg=%f %f Obj=%f %f\n",
  //    CkMyPe(),msg->proc_speed,msg->total_walltime,msg->total_cputime,
  //    msg->idletime,msg->bg_walltime,msg->bg_cputime,
  //    msg->obj_walltime,msg->obj_cputime);

  //  CkPrintf("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
  //	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm);
  return msg;
}

void NborBaseLB::Migrated(LDObjHandle h)
{
  migrates_completed++;
  //  CkPrintf("[%d] An object migrated! %d %d\n",
  //  	   CkMyPe(),migrates_completed,migrates_expected);
  if (migrates_completed == migrates_expected) {
    MigrationDone();
  }
}

void NborBaseLB::ReceiveStats(NLBStatsMsg *m)
{
  if (neighbor_pes == 0) FindNeighbors();

  if (m == 0) { // This is from our own node
    receive_stats_ready = 1;
  } else {
    const int pe = m->from_pe;
    //  CkPrintf("Stats msg received, %d %d %d %d %p\n",
    //  	   pe,stats_msg_count,m->n_objs,m->serial,m);
    int peslot = NeighborIndex(pe);

    if (peslot == -1 || statsMsgsList[peslot] != 0) {
      CkPrintf("*** Unexpected NLBStatsMsg in ReceiveStats from PE %d ***\n",
	       pe);
    } else {
      statsMsgsList[peslot] = m;
      statsDataList[peslot].from_pe = m->from_pe;
      statsDataList[peslot].total_walltime = m->total_walltime;
      statsDataList[peslot].total_cputime = m->total_cputime;
      statsDataList[peslot].idletime = m->idletime;
      statsDataList[peslot].bg_walltime = m->bg_walltime;
      statsDataList[peslot].bg_cputime = m->bg_cputime;
      statsDataList[peslot].pe_speed = m->pe_speed;
      statsDataList[peslot].obj_walltime = m->obj_walltime;
      statsDataList[peslot].obj_cputime = m->obj_cputime;

      statsDataList[peslot].n_objs = m->n_objs;
      statsDataList[peslot].objData = m->objData;
      statsDataList[peslot].n_comm = m->n_comm;
      statsDataList[peslot].commData = m->commData;
      stats_msg_count++;
    }
  }

  const int clients = num_neighbors();
  if (stats_msg_count == clients && receive_stats_ready) {
    double strat_start_time = CmiWallTimer();
    receive_stats_ready = 0;
    NLBMigrateMsg* migrateMsg = Strategy(statsDataList,clients);

    int i;

    // Migrate messages from me to elsewhere
    for(i=0; i < migrateMsg->n_moves; i++) {
      MigrateInfo& move = migrateMsg->moves[i];
      const int me = CkMyPe();
      if (move.from_pe == me && move.to_pe != me) {
	theLbdb->Migrate(move.obj,move.to_pe);
      } else if (move.from_pe != me) {
	CkPrintf("[%d] error, strategy wants to move from %d to  %d\n",
		 me,move.from_pe,move.to_pe);
      }
    }
    
    // Now, send migrate messages to neighbors
    for(i=1; i < num_neighbors(); i++) {
      NLBMigrateMsg* m2 = (NLBMigrateMsg*) CkCopyMsg((void**)&migrateMsg);
      thisproxy [neighbor_pes[i]].ReceiveMigration(m2);
    }
    if (0 < num_neighbors())
      thisproxy [neighbor_pes[0]].ReceiveMigration(migrateMsg);
    else delete migrateMsg;
    
    // Zero out data structures for next cycle
    for(i=0; i < clients; i++) {
      delete statsMsgsList[i];
      statsMsgsList[i]=0;
    }
    stats_msg_count=0;

    theLbdb->ClearLoads();
    if (CkMyPe() == 0) {
      double strat_end_time = CmiWallTimer();
      CkPrintf("Strat elapsed time %f\n",strat_end_time-strat_start_time);
    }
  }
  
}

void NborBaseLB::ReceiveMigration(NLBMigrateMsg *msg)
{
  if (neighbor_pes == 0) FindNeighbors();

  if (mig_msgs_received == 0) migrates_expected = 0;

  mig_msgs[mig_msgs_received] = msg;
  mig_msgs_received++;
  //  CkPrintf("[%d] Received migration msg %d of %d\n",
  //	   CkMyPe(),mig_msgs_received,mig_msgs_expected);

  if (mig_msgs_received > mig_msgs_expected) {
    CkPrintf("[%d] NeighborLB Error! Too many migration messages received\n",
	     CkMyPe());
  }

  if (mig_msgs_received != mig_msgs_expected) {
    return;
  }

  //  CkPrintf("[%d] in ReceiveMigration %d moves\n",CkMyPe(),msg->n_moves);
  for(int neigh=0; neigh < mig_msgs_received;neigh++) {
    NLBMigrateMsg* m = mig_msgs[neigh];
    for(int i=0; i < m->n_moves; i++) {
      MigrateInfo& move = m->moves[i];
      const int me = CkMyPe();
      if (move.from_pe != me && move.to_pe == me) {
	migrates_expected++;
      }
    }
    delete m;
    mig_msgs[neigh]=0;
  }
  //  CkPrintf("[%d] in ReceiveMigration %d expected\n",
  //	   CkMyPe(),migrates_expected);
  mig_msgs_received = 0;
  if (migrates_expected == 0 || migrates_expected == migrates_completed)
    MigrationDone();
}


void NborBaseLB::MigrationDone()
{
  if (CkMyPe() == 0 && migrates_completed) {
    double end_lb_time = CmiWallTimer();
    CkPrintf("Load balancing step %d finished at %f duration %f\n",
	     step(),end_lb_time,end_lb_time - start_lb_time);
  }
  migrates_completed = 0;
  migrates_expected = -1;
  // Increment to next step
  mystep++;
  thisproxy [CkMyPe()].ResumeClients();
}

void NborBaseLB::ResumeClients()
{
  theLbdb->ResumeClients();
}

NLBMigrateMsg* NborBaseLB::Strategy(LDStats* stats,int count)
{
  for(int j=0; j < count; j++) {
    CkPrintf(
    "[%d] Proc %d Speed %d Total(wall,cpu)=%f %f Idle=%f Bg=%f %f obj=%f %f\n",
    CkMyPe(),stats[j].from_pe,stats[j].pe_speed,
    stats[j].total_walltime,stats[j].total_cputime,
    stats[j].idletime,stats[j].bg_walltime,stats[j].bg_cputime,
    stats[j].obj_walltime,stats[j].obj_cputime);
  }

  delete [] myStats.objData;
  myStats.n_objs = 0;
  delete [] myStats.commData;
  myStats.n_comm = 0;

  int sizes=0;
  NLBMigrateMsg* msg = new(&sizes,1) NLBMigrateMsg;
  msg->n_moves = 0;

  return msg;
}

int NborBaseLB::NeighborIndex(int pe)
{
    int peslot = -1;
    for(int i=0; i < num_neighbors(); i++) {
      if (pe == neighbor_pes[i]) {
	peslot = i;
	break;
      }
    }
    return peslot;
}

void* NLBMigrateMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = size + array[0] * sizeof(NborBaseLB::MigrateInfo);

  NLBMigrateMsg* ret =
    (NLBMigrateMsg*)(CkAllocMsg(msgnum,totalsize,priobits));

  ret->moves = (NborBaseLB::MigrateInfo*) ((char*)(ret)+ size);

  return (void*)(ret);
}

void* NLBMigrateMsg::pack(NLBMigrateMsg* m)
{
  m->moves = (NborBaseLB::MigrateInfo*)
    ((char*)(m->moves) - (char*)(&m->moves));

  return (void*)(m);
}

NLBMigrateMsg* NLBMigrateMsg::unpack(void *m)
{
  NLBMigrateMsg* ret_val = (NLBMigrateMsg*)(m);

  ret_val->moves = (NborBaseLB::MigrateInfo*)
    ((char*)(&ret_val->moves) + (size_t)(ret_val->moves));

  return ret_val;
}

#endif
