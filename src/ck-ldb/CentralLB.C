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
  CentralLB *me = static_cast<CentralLB*>(data);

  me->Migrated(h);
}

void CentralLB::staticAtSync(void* data)
{
  CentralLB *me = static_cast<CentralLB*>(data);

  me->AtSync();
}

CentralLB::CentralLB()
{
  step = 0;
  theLbdb = CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->
    AddLocalBarrierReceiver(reinterpret_cast<LDBarrierFn>(staticAtSync),
			    static_cast<void*>(this));
  theLbdb->
    NotifyMigrated(reinterpret_cast<LDMigratedFn>(staticMigrated),
		   static_cast<void*>(this));

  stats_msg_count = 0;
  statsMsgsList = new CLBStatsMsg*[CkNumPes()];
  for(int i=0; i < CkNumPes(); i++)
    statsMsgsList[i] = 0;

  statsDataList = new LDStats[CkNumPes()];
  theLbdb->CollectStatsOn();
}

CentralLB::~CentralLB()
{
  CkPrintf("Going away\n");
}

void CentralLB::AtSync()
{
  CkPrintf("[%d] CentralLB At Sync step %d!!!!\n",CkMyPe(),step);

  if (!QueryBalanceNow(step)) {
    MigrationDone();
    return;
  }

  // Send stats
  int sizes[2];
  const int osz = sizes[0] = theLbdb->GetObjDataSz();
  const int csz = sizes[1] = theLbdb->GetCommDataSz();
  
  CLBStatsMsg* msg = new(sizes,2) CLBStatsMsg;
  msg->from_pe = CkMyPe();
  msg->serial = rand();

  msg->n_objs = osz;
  theLbdb->GetObjData(msg->objData);
  msg->n_comm = csz;
  theLbdb->GetCommData(msg->commData);
  theLbdb->ClearLoads();
  CkPrintf("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm);
  CProxy_CentralLB(thisgroup).ReceiveStats(msg,0);
}

void CentralLB::Migrated(LDObjHandle h)
{
  migrates_completed++;
  CkPrintf("[%d] An object migrated! %d %d\n",
	   CkMyPe(),migrates_completed,migrates_expected);
  if (migrates_completed == migrates_expected) {
    MigrationDone();
  }
}

void CentralLB::ReceiveStats(CLBStatsMsg *m)
{
  const int pe = m->from_pe;
  CkPrintf("Stats msg received, %d %d %d %d %p\n",
	   pe,stats_msg_count,m->n_objs,m->serial,m);
  if (statsMsgsList[pe] != 0) {
    CkPrintf("*** Unexpected CLBStatsMsg in ReceiveStats from PE %d ***\n",
	     pe);
  } else {
    statsMsgsList[pe] = m;
    statsDataList[pe].n_objs = m->n_objs;
    statsDataList[pe].objData = m->objData;
    statsDataList[pe].n_comm = m->n_comm;
    statsDataList[pe].commData = m->commData;
    stats_msg_count++;
  }

  const int clients = CkNumPes();
  if (stats_msg_count == clients) {

    CLBMigrateMsg* migrateMsg = Strategy(statsDataList,clients);
    CProxy_CentralLB(thisgroup).ReceiveMigration(migrateMsg);

    // Zero out data structures for next cycle
    for(int i=0; i < clients; i++) {
      delete statsMsgsList[i];
      statsMsgsList[i]=0;
    }
    stats_msg_count=0;
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
      CkPrintf("[%d] migrating object to %d\n",move.from_pe,move.to_pe);
      theLbdb->Migrate(move.obj,move.to_pe);
    } else if (move.from_pe != me && move.to_pe == me) {
      migrates_expected++;
    }
  }
  if (migrates_expected == 0)
    MigrationDone();
  delete m;
}


void CentralLB::MigrationDone()
{
  migrates_completed = 0;
  migrates_expected = -1;
  // Increment to next step
  step++;
  theLbdb->ResumeClients();
}

CLBMigrateMsg* CentralLB::Strategy(LDStats* stats,int count)
{
  for(int j=0; j < count; j++) {
    int i;
    LDObjData *odata = stats[j].objData;
    const int osz = stats[j].n_objs;

    CkPrintf("------------- Object Data: PE %d: %d objects -------------\n",
	     j,osz);
    for(i=0; i < osz; i++) {
      CkPrintf("Object %d\n",i);
      CkPrintf("     id = %d\n",odata[i].id.id[0]);
      CkPrintf("  OM id = %d\n",odata[i].omID.id);
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
    + array[1] * sizeof(LDCommData);

  CLBStatsMsg* ret =
    static_cast<CLBStatsMsg*>(CkAllocMsg(msgnum,totalsize,priobits));

  ret->objData = reinterpret_cast<LDObjData*>((reinterpret_cast<char*>(ret) 
					       + size));
  ret->commData = reinterpret_cast<LDCommData*>(ret->objData + array[0]);

  return static_cast<void*>(ret);
}

void* CLBStatsMsg::pack(CLBStatsMsg* m)
{
  m->objData = 
    reinterpret_cast<LDObjData*>(reinterpret_cast<char*>(m->objData)
      - reinterpret_cast<char*>(&m->objData));
  m->commData = 
    reinterpret_cast<LDCommData*>(reinterpret_cast<char*>(m->commData)
      - reinterpret_cast<char*>(&m->commData));
  return static_cast<void*>(m);
}

CLBStatsMsg* CLBStatsMsg::unpack(void *m)
{
  CLBStatsMsg* ret_val = static_cast<CLBStatsMsg*>(m);

  ret_val->objData = 
    reinterpret_cast<LDObjData*>(reinterpret_cast<char*>(&ret_val->objData)
      + reinterpret_cast<size_t>(ret_val->objData));
  ret_val->commData = 
    reinterpret_cast<LDCommData*>(reinterpret_cast<char*>(&ret_val->commData)
      + reinterpret_cast<size_t>(ret_val->commData));
  return ret_val;
}

void* CLBMigrateMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = size + array[0] * sizeof(CentralLB::MigrateInfo);

  CLBMigrateMsg* ret =
    static_cast<CLBMigrateMsg*>(CkAllocMsg(msgnum,totalsize,priobits));

  ret->moves = reinterpret_cast<CentralLB::MigrateInfo*>
    (reinterpret_cast<char*>(ret)+ size);

  return static_cast<void*>(ret);
}

void* CLBMigrateMsg::pack(CLBMigrateMsg* m)
{
  m->moves = reinterpret_cast<CentralLB::MigrateInfo*>
    (reinterpret_cast<char*>(m->moves) - reinterpret_cast<char*>(&m->moves));

  return static_cast<void*>(m);
}

CLBMigrateMsg* CLBMigrateMsg::unpack(void *m)
{
  CLBMigrateMsg* ret_val = static_cast<CLBMigrateMsg*>(m);

  ret_val->moves = reinterpret_cast<CentralLB::MigrateInfo*>
    (reinterpret_cast<char*>(&ret_val->moves) 
     + reinterpret_cast<size_t>(ret_val->moves));

  return ret_val;
}

#endif
