#include <charm++.h>
#include <LBDatabase.h>
#include "NeighborLB.h"
#include "NeighborLB.def.h"

CkGroupID loadbalancer;

#if CMK_LBDB_ON

void CreateNeighborLB()
{
  loadbalancer = CProxy_NeighborLB::ckNew();
}

void NeighborLB::staticMigrated(void* data, LDObjHandle h)
{
  NeighborLB *me = static_cast<NeighborLB*>(data);

  me->Migrated(h);
}

void NeighborLB::staticAtSync(void* data)
{
  NeighborLB *me = static_cast<NeighborLB*>(data);

  me->AtSync();
}

NeighborLB::NeighborLB()
{
  mystep = 0;
  theLbdb = CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->
    AddLocalBarrierReceiver(reinterpret_cast<LDBarrierFn>(staticAtSync),
			    static_cast<void*>(this));
  theLbdb->
    NotifyMigrated(reinterpret_cast<LDMigratedFn>(staticMigrated),
		   static_cast<void*>(this));

  stats_msg_count = 0;
  statsMsgsList = new NLBStatsMsg*[num_neighbors()];
  for(int i=0; i < CkNumPes(); i++)
    statsMsgsList[i] = 0;
  statsDataList = new LDStats[num_neighbors()];

  neighbor_pes = new int[num_neighbors()];
  neighbors(neighbor_pes);

  migrates_completed = 0;
  migrates_expected = -1;
  mig_msgs_received = 0;
  mig_msgs_expected = num_neighbors();
  mig_msgs = new NLBMigrateMsg*[num_neighbors()];

  proc_speed = theLbdb->ProcessorSpeed();
  obj_data_sz = 0;
  comm_data_sz = 0;
  receive_stats_ready = 0;

  theLbdb->CollectStatsOn();
}

NeighborLB::~NeighborLB()
{
  CkPrintf("Going away\n");
}

void NeighborLB::AtSync()
{
  CkPrintf("[%d] NeighborLB At Sync step %d!!!!\n",CkMyPe(),mystep);

  if (!QueryBalanceNow(step()) || num_neighbors() == 0) {
    MigrationDone();
    return;
  }

  NLBStatsMsg* msg = AssembleStats();

  int i;
  for(i=1; i < num_neighbors(); i++) {
    NLBStatsMsg* m2 = (NLBStatsMsg*) CkCopyMsg((void**)&msg);
    CProxy_NeighborLB(thisgroup).ReceiveStats(m2,neighbor_pes[i]);
  }
  if (0 < num_neighbors()) {
    CProxy_NeighborLB(thisgroup).ReceiveStats(msg,neighbor_pes[0]);
  } else delete msg;

  //  delete msg;
  // Tell our own node that we are ready
  ReceiveStats((NLBStatsMsg*)0);
  CkPrintf("[%d] done with AtSync\n",CkMyPe());
}

NLBStatsMsg* NeighborLB::AssembleStats()
{
  // Send stats
  obj_data_sz = theLbdb->GetObjDataSz();
  myObjData = new LDObjData[obj_data_sz];
  theLbdb->GetObjData(myObjData);

  comm_data_sz = theLbdb->GetCommDataSz();
  myCommData = new LDCommData[comm_data_sz];
  theLbdb->GetCommData(myCommData);
  
  NLBStatsMsg* msg = new NLBStatsMsg;

  msg->from_pe = CkMyPe();
  msg->serial = rand();

  theLbdb->TotalTime(&msg->total_walltime,&msg->total_cputime);
  theLbdb->IdleTime(&msg->idletime);
  theLbdb->BackgroundLoad(&msg->bg_walltime,&msg->bg_cputime);
  msg->proc_speed = proc_speed;

  msg->obj_walltime = msg->obj_cputime = 0;

  for(int i=0; i < obj_data_sz; i++) {
    msg->obj_walltime += myObjData[i].wallTime;
    msg->obj_cputime += myObjData[i].cpuTime;
  }    

  CkPrintf(
    "Proc %d speed=%d Total(wall,cpu)=%f %f Idle=%f Bg=%f %f Obj=%f %f\n",
    CkMyPe(),msg->proc_speed,msg->total_walltime,msg->total_cputime,
    msg->idletime,msg->bg_walltime,msg->bg_cputime,
    msg->obj_walltime,msg->obj_cputime);

  //  CkPrintf("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
  //	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm);
  return msg;
}

void NeighborLB::Migrated(LDObjHandle h)
{
  migrates_completed++;
  //  CkPrintf("[%d] An object migrated! %d %d\n",
  //  	   CkMyPe(),migrates_completed,migrates_expected);
  if (migrates_completed == migrates_expected) {
    MigrationDone();
  }
}

void NeighborLB::ReceiveStats(NLBStatsMsg *m)
{
  if (m == 0) { // This is from our own node
    receive_stats_ready = 1;
  } else {
    const int pe = m->from_pe;
    //  CkPrintf("Stats msg received, %d %d %d %d %p\n",
    //  	   pe,stats_msg_count,m->n_objs,m->serial,m);
    int peslot = -1;
    for(int i=0; i < num_neighbors(); i++) {
      if (pe == neighbor_pes[i]) {
	peslot = i;
	break;
      }
    }
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
      statsDataList[peslot].proc_speed = m->proc_speed;
      statsDataList[peslot].obj_walltime = m->obj_walltime;
      statsDataList[peslot].obj_cputime = m->obj_cputime;
      stats_msg_count++;
    }
  }

  const int clients = num_neighbors();
  if (stats_msg_count == clients && receive_stats_ready) {
    receive_stats_ready = 0;
    NLBMigrateMsg* migrateMsg = Strategy(statsDataList,clients);

    // Migrate messages from me to elsewhere
    for(int i=0; i < migrateMsg->n_moves; i++) {
      MigrateInfo& move = migrateMsg->moves[i];
      const int me = CkMyPe();
      if (move.from_pe == me && move.to_pe != me) {
	CkPrintf("[%d] migrating object to %d\n",move.from_pe,move.to_pe);
	theLbdb->Migrate(move.obj,move.to_pe);
      } else if (move.from_pe != me) {
	CkPrintf("[%d] error, strategy wants to move from %d to  %d\n",
		 me,move.from_pe,move.to_pe);
      }
    }
    
    // Now, send migrate messages to neighbors
    int i;
    for(i=1; i < num_neighbors(); i++) {
      NLBMigrateMsg* m2 = (NLBMigrateMsg*) CkCopyMsg((void**)&migrateMsg);
      CProxy_NeighborLB(thisgroup).ReceiveMigration(m2,neighbor_pes[i]);
    }
    if (0 < num_neighbors())
      CProxy_NeighborLB(thisgroup).ReceiveMigration(migrateMsg,
						    neighbor_pes[0]);
    else delete migrateMsg;
    
    // Zero out data structures for next cycle
    for(int i=0; i < clients; i++) {
      delete statsMsgsList[i];
      statsMsgsList[i]=0;
    }
    stats_msg_count=0;
  }
  
  theLbdb->ClearLoads();
}

void NeighborLB::ReceiveMigration(NLBMigrateMsg *msg)
{
  mig_msgs[mig_msgs_received] = msg;
  mig_msgs_received++;
  //  CkPrintf("[%d] Received migration msg %d of %d\n",
  //	   CkMyPe(),mig_msgs_received,mig_msgs_expected);

  if (mig_msgs_received < mig_msgs_expected)
    return;
  else if (mig_msgs_received < mig_msgs_expected) {
    CkPrintf("[%d] NeighborLB Error! Too many migration messages received\n",
	     CkMyPe());
    return;
  }

  //  CkPrintf("[%d] in ReceiveMigration %d moves\n",CkMyPe(),msg->n_moves);
  migrates_expected = 0;
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


void NeighborLB::MigrationDone()
{
  migrates_completed = 0;
  migrates_expected = -1;
  // Increment to next step
  mystep++;
  //  CkPrintf("[%d] Resuming clients\n",CkMyPe());
  CProxy_NeighborLB(thisgroup).ResumeClients(CkMyPe());
}

void NeighborLB::ResumeClients()
{
  theLbdb->ResumeClients();
}

NLBMigrateMsg* NeighborLB::Strategy(LDStats* stats,int count)
{
  for(int j=0; j < count; j++) {
    CkPrintf(
    "[%d] Proc %d Speed %d Total(wall,cpu)=%f %f Idle=%f Bg=%f %f obj=%f %f\n",
    CkMyPe(),stats[j].from_pe,stats[j].proc_speed,
    stats[j].total_walltime,stats[j].total_cputime,
    stats[j].idletime,stats[j].bg_walltime,stats[j].bg_cputime,
    stats[j].obj_walltime,stats[j].obj_cputime);
  }

  delete [] myObjData;
  obj_data_sz = 0;
  delete [] myCommData;
  comm_data_sz = 0;

  int sizes=0;
  NLBMigrateMsg* msg = new(&sizes,1) NLBMigrateMsg;
  msg->n_moves = 0;

  return msg;
}

// void* NLBStatsMsg::alloc(int msgnum, size_t size, int* array, int priobits)
// {
//   int totalsize = size + array[0] * sizeof(LDObjData) 
//     + array[1] * sizeof(LDCommData);

//   NLBStatsMsg* ret =
//     static_cast<NLBStatsMsg*>(CkAllocMsg(msgnum,totalsize,priobits));

//   ret->objData = reinterpret_cast<LDObjData*>((reinterpret_cast<char*>(ret) 
// 					       + size));
//   ret->commData = reinterpret_cast<LDCommData*>(ret->objData + array[0]);

//   return static_cast<void*>(ret);
// }

// void* NLBStatsMsg::pack(NLBStatsMsg* m)
// {
//   m->objData = 
//     reinterpret_cast<LDObjData*>(reinterpret_cast<char*>(m->objData)
//       - reinterpret_cast<char*>(&m->objData));
//   m->commData = 
//     reinterpret_cast<LDCommData*>(reinterpret_cast<char*>(m->commData)
//       - reinterpret_cast<char*>(&m->commData));
//   return static_cast<void*>(m);
// }

// NLBStatsMsg* NLBStatsMsg::unpack(void *m)
// {
//   NLBStatsMsg* ret_val = static_cast<NLBStatsMsg*>(m);

//   ret_val->objData = 
//     reinterpret_cast<LDObjData*>(reinterpret_cast<char*>(&ret_val->objData)
//       + reinterpret_cast<size_t>(ret_val->objData));
//   ret_val->commData = 
//     reinterpret_cast<LDCommData*>(reinterpret_cast<char*>(&ret_val->commData)
//       + reinterpret_cast<size_t>(ret_val->commData));
//   return ret_val;
// }

void* NLBMigrateMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = size + array[0] * sizeof(NeighborLB::MigrateInfo);

  NLBMigrateMsg* ret =
    static_cast<NLBMigrateMsg*>(CkAllocMsg(msgnum,totalsize,priobits));

  ret->moves = reinterpret_cast<NeighborLB::MigrateInfo*>
    (reinterpret_cast<char*>(ret)+ size);

  return static_cast<void*>(ret);
}

void* NLBMigrateMsg::pack(NLBMigrateMsg* m)
{
  m->moves = reinterpret_cast<NeighborLB::MigrateInfo*>
    (reinterpret_cast<char*>(m->moves) - reinterpret_cast<char*>(&m->moves));

  return static_cast<void*>(m);
}

NLBMigrateMsg* NLBMigrateMsg::unpack(void *m)
{
  NLBMigrateMsg* ret_val = static_cast<NLBMigrateMsg*>(m);

  ret_val->moves = reinterpret_cast<NeighborLB::MigrateInfo*>
    (reinterpret_cast<char*>(&ret_val->moves) 
     + reinterpret_cast<size_t>(ret_val->moves));

  return ret_val;
}

#endif
