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

  // Send stats
  int sizes[2];
  const int osz = sizes[0] = theLbdb->GetObjDataSz();
  const int csz = sizes[1] = theLbdb->GetCommDataSz();
  
  NLBStatsMsg* msg = new(sizes,2) NLBStatsMsg;
  msg->from_pe = CkMyPe();
  msg->serial = rand();

  theLbdb->TotalTime(&msg->total_walltime,&msg->total_cputime);
  theLbdb->IdleTime(&msg->idletime);
  theLbdb->BackgroundLoad(&msg->bg_walltime,&msg->bg_cputime);
  CkPrintf(
	   "Proc %d Total(wall,cpu)=%f %f Idle=%f Bg=%f %f\n",
	   CkMyPe(),msg->total_walltime,msg->total_cputime,
	   msg->idletime,msg->bg_walltime,msg->bg_cputime);

  msg->n_objs = osz;
  theLbdb->GetObjData(msg->objData);
  msg->n_comm = csz;
  theLbdb->GetCommData(msg->commData);
  //  CkPrintf("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
  //	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm);

  int i;
  for(i=1; i < num_neighbors(); i++) {
    NLBStatsMsg* m2 = (NLBStatsMsg*) CkCopyMsg((void**)&msg);
    CProxy_NeighborLB(thisgroup).ReceiveStats(m2,neighbor_pes[i]);
  }
  if (0 < num_neighbors()) {
    CProxy_NeighborLB(thisgroup).ReceiveStats(msg,neighbor_pes[0]);
  }
  else delete msg;
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
    statsDataList[peslot].total_walltime = m->total_walltime;
    statsDataList[peslot].total_cputime = m->total_cputime;
    statsDataList[peslot].idletime = m->idletime;
    statsDataList[peslot].bg_walltime = m->bg_walltime;
    statsDataList[peslot].bg_cputime = m->bg_cputime;
    statsDataList[peslot].n_objs = m->n_objs;
    statsDataList[peslot].n_objs = m->n_objs;
    statsDataList[peslot].objData = m->objData;
    statsDataList[peslot].n_comm = m->n_comm;
    statsDataList[peslot].commData = m->commData;
    stats_msg_count++;
  }

  const int clients = num_neighbors();
  if (stats_msg_count == clients) {
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
  if (migrates_expected == 0)
    MigrationDone();
}


void NeighborLB::MigrationDone()
{
  migrates_completed = 0;
  migrates_expected = -1;
  // Increment to next step
  mystep++;
  //  CkPrintf("[%d] Resuming clients\n",CkMyPe());
  theLbdb->ResumeClients();
}

NLBMigrateMsg* NeighborLB::Strategy(LDStats* stats,int count)
{
  for(int j=0; j < count; j++) {
    LDObjData *odata = stats[j].objData;
    const int osz = stats[j].n_objs;

    CkPrintf(
      "[%d] Proc %d Total(wall,cpu)=%f %f Idle=%f Bg=%f %f\n",
      CkMyPe(),j,stats[j].total_walltime,stats[j].total_cputime,
      stats[j].idletime,stats[j].bg_walltime,stats[j].bg_cputime);
    CkPrintf(
      "[%d] ------------- Object Data: PE %d: %d objects -------------\n",
      CkMyPe(),j,osz);

    int i;
    for(i=0; i < osz; i++) {
      CkPrintf(
        "[%d] Object %4d id = %4d OM id = %4d CPU = %8.4f Wall = %8.3f\n",
	CkMyPe(),i,odata[i].id.id[0],odata[i].omID.id,
	odata[i].cpuTime,odata[i].wallTime);
    }

    LDCommData *cdata = stats[j].commData;
    const int csz = stats[j].n_comm;

    CkPrintf("[%d]------------- Comm Data: PE %d: %d records -------------\n",
	     CkMyPe(),j,csz);
    for(i=0; i < csz; i++) {
      CkPrintf("Link %d ",i);
      
      if (cdata[i].from_proc)
	CkPrintf(" sender PE = %d ",cdata[i].src_proc);
      else
	CkPrintf(" sender id = %d:%d ",
		 cdata[i].senderOM.id,cdata[i].sender.id[0]);

      if (cdata[i].to_proc)
	CkPrintf(" receiver PE = %d ",cdata[i].dest_proc);
      else	
	CkPrintf(" receiver id = %d:%d ",
		 cdata[i].receiverOM.id,cdata[i].receiver.id[0]);
      
      CkPrintf(" messages = %d ",cdata[i].messages);
      CkPrintf(" bytes = %d\n",cdata[i].bytes);
    }
  }

  int sizes=0;
  NLBMigrateMsg* msg = new(&sizes,1) NLBMigrateMsg;
  msg->n_moves = 0;

  return msg;
}

void* NLBStatsMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = size + array[0] * sizeof(LDObjData) 
    + array[1] * sizeof(LDCommData);

  NLBStatsMsg* ret =
    static_cast<NLBStatsMsg*>(CkAllocMsg(msgnum,totalsize,priobits));

  ret->objData = reinterpret_cast<LDObjData*>((reinterpret_cast<char*>(ret) 
					       + size));
  ret->commData = reinterpret_cast<LDCommData*>(ret->objData + array[0]);

  return static_cast<void*>(ret);
}

void* NLBStatsMsg::pack(NLBStatsMsg* m)
{
  m->objData = 
    reinterpret_cast<LDObjData*>(reinterpret_cast<char*>(m->objData)
      - reinterpret_cast<char*>(&m->objData));
  m->commData = 
    reinterpret_cast<LDCommData*>(reinterpret_cast<char*>(m->commData)
      - reinterpret_cast<char*>(&m->commData));
  return static_cast<void*>(m);
}

NLBStatsMsg* NLBStatsMsg::unpack(void *m)
{
  NLBStatsMsg* ret_val = static_cast<NLBStatsMsg*>(m);

  ret_val->objData = 
    reinterpret_cast<LDObjData*>(reinterpret_cast<char*>(&ret_val->objData)
      + reinterpret_cast<size_t>(ret_val->objData));
  ret_val->commData = 
    reinterpret_cast<LDCommData*>(reinterpret_cast<char*>(&ret_val->commData)
      + reinterpret_cast<size_t>(ret_val->commData));
  return ret_val;
}

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
