#ifndef  WIN32
#include <unistd.h>
#endif
#include <charm++.h>
#include <LBDatabase.h>
#include <CkLists.h>
#include "heap.h"
#include "WSLB.h"
#include "WSLB.def.h"

CkGroupID wslb;

#if CMK_LBDB_ON

// Temporary vacating flags
// Set PROC to -1 to disable

#define VACATE_PROC -1
//#define VACATE_PROC (CkNumPes()/2)
#define VACATE_AFTER 30
#define UNVACATE_AFTER 15

void CreateWSLB()
{
  wslb = CProxy_WSLB::ckNew();
}

void WSLB::staticMigrated(void* data, LDObjHandle h)
{
  WSLB *me = static_cast<WSLB*>(data);

  me->Migrated(h);
}

void WSLB::staticAtSync(void* data)
{
  WSLB *me = static_cast<WSLB*>(data);

  me->AtSync();
}

WSLB::WSLB()
{
  mystep = 0;
  theLbdb = CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->
    AddLocalBarrierReceiver(reinterpret_cast<LDBarrierFn>(staticAtSync),
			    static_cast<void*>(this));
  theLbdb->
    NotifyMigrated(reinterpret_cast<LDMigratedFn>(staticMigrated),
		   static_cast<void*>(this));


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

  myStats.proc_speed = theLbdb->ProcessorSpeed();
//  char hostname[80];
//  gethostname(hostname,79);
//  CkPrintf("[%d] host %s speed %d\n",CkMyPe(),hostname,myStats.proc_speed);
  myStats.obj_data_sz = 0;
  myStats.comm_data_sz = 0;
  receive_stats_ready = 0;

  vacate = CmiFalse;
  usage = 1.0;
  usage_int_err = 0.;

  theLbdb->CollectStatsOn();
}

WSLB::~WSLB()
{
  CkPrintf("Going away\n");
}

void WSLB::FindNeighbors()
{
  if (neighbor_pes == 0) { // Neighbors never initialized, so init them
                           // and other things that depend on the number
                           // of neighbors
    statsMsgsList = new WSLBStatsMsg*[num_neighbors()];
    for(int i=0; i < num_neighbors(); i++)
      statsMsgsList[i] = 0;
    statsDataList = new LDStats[num_neighbors()];

    neighbor_pes = new int[num_neighbors()];
    neighbors(neighbor_pes);
    mig_msgs_expected = num_neighbors();
    mig_msgs = new WSLBMigrateMsg*[num_neighbors()];
  }

}

void WSLB::AtSync()
{
  //  CkPrintf("[%d] WSLB At Sync step %d!!!!\n",CkMyPe(),mystep);

  if (CkMyPe() == 0) {
    start_lb_time = CmiWallTimer();
    CkPrintf("Load balancing step %d starting at %f\n",
	     step(),start_lb_time);
  }

  if (neighbor_pes == 0) FindNeighbors();

  if (!QueryBalanceNow(step()) || num_neighbors() == 0) {
    MigrationDone();
    return;
  }

  WSLBStatsMsg* msg = AssembleStats();

  int i;
  for(i=1; i < num_neighbors(); i++) {
    WSLBStatsMsg* m2 = (WSLBStatsMsg*) CkCopyMsg((void**)&msg);
    CProxy_WSLB(thisgroup).ReceiveStats(m2,neighbor_pes[i]);
  }
  if (0 < num_neighbors()) {
    CProxy_WSLB(thisgroup).ReceiveStats(msg,neighbor_pes[0]);
  } else delete msg;

  // Tell our own node that we are ready
  ReceiveStats((WSLBStatsMsg*)0);
}

WSLBStatsMsg* WSLB::AssembleStats()
{
  // Get stats
  theLbdb->TotalTime(&myStats.total_walltime,&myStats.total_cputime);
  theLbdb->IdleTime(&myStats.idletime);
  theLbdb->BackgroundLoad(&myStats.bg_walltime,&myStats.bg_cputime);
  myStats.obj_data_sz = theLbdb->GetObjDataSz();
  myStats.objData = new LDObjData[myStats.obj_data_sz];
  theLbdb->GetObjData(myStats.objData);

  myStats.comm_data_sz = theLbdb->GetCommDataSz();
  myStats.commData = new LDCommData[myStats.comm_data_sz];
  theLbdb->GetCommData(myStats.commData);

  myStats.obj_walltime = myStats.obj_cputime = 0;
  for(int i=0; i < myStats.obj_data_sz; i++) {
    myStats.obj_walltime += myStats.objData[i].wallTime;
    myStats.obj_cputime += myStats.objData[i].cpuTime;
  }    

  WSLBStatsMsg* msg = new WSLBStatsMsg;

  // Calculate usage percentage
  double myload = myStats.total_walltime - myStats.idletime;
  double myusage;
//   for(i=0; i < myStats.obj_data_sz; i++) {
//     myobjcpu += myStats.objData[i].cpuTime;
//     myobjwall += myStats.objData[i].wallTime;
//   }
//   if (myobjwall > 0)
//     myusage = myobjcpu / myobjwall;
//   else

  if (myload > 0)
    myusage = myStats.total_cputime / myload;
  else myusage = 1.0;
  // Apply proportional-integral control on usage changes
  const double usage_err = myusage - usage;
  usage_int_err += usage_err;
  usage += usage_err * 0.1 + usage_int_err * 0.01;
  //  CkPrintf("[%d] Usage err = %f %f\n",CkMyPe(),usage_err,usage_int_err);
 
  // Allow usage to decrease quickly, but increase slowly
  //   if (myusage > usage)
  //     usage += (myusage-usage) * 0.1;
  //   else usage = myusage;
 

  //  CkPrintf("PE %d myload = %f myusage = %f usage = %f\n",
  //	   CkMyPe(),myload,myusage,usage);

  msg->from_pe = CkMyPe();
  // msg->serial = rand();
  msg->serial = CrnRand();
  msg->proc_speed = myStats.proc_speed;
  msg->total_walltime = myStats.total_walltime;
  msg->total_cputime = myStats.total_cputime;
  msg->idletime = myStats.idletime;
  msg->bg_walltime = myStats.bg_walltime;
  msg->bg_cputime = myStats.bg_cputime;
  msg->obj_walltime = myStats.obj_walltime;
  msg->obj_cputime = myStats.obj_cputime;
  msg->vacate_me = vacate;
  msg->usage = usage;

  //  CkPrintf(
  //    "Proc %d speed=%d Total(wall,cpu)=%f %f Idle=%f Bg=%f %f Obj=%f %f\n",
  //    CkMyPe(),msg->proc_speed,msg->total_walltime,msg->total_cputime,
  //    msg->idletime,msg->bg_walltime,msg->bg_cputime,
  //    msg->obj_walltime,msg->obj_cputime);

  //  CkPrintf("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
  //	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm);
  return msg;
}

void WSLB::Migrated(LDObjHandle h)
{
  migrates_completed++;
  //  CkPrintf("[%d] An object migrated! %d %d\n",
  //  	   CkMyPe(),migrates_completed,migrates_expected);
  if (migrates_completed == migrates_expected) {
    MigrationDone();
  }
}

void WSLB::ReceiveStats(WSLBStatsMsg *m)
{
  if (neighbor_pes == 0) FindNeighbors();

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
      CkPrintf("*** Unexpected WSLBStatsMsg in ReceiveStats from PE %d ***\n",
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
      statsDataList[peslot].vacate_me = m->vacate_me;
      statsDataList[peslot].usage = m->usage;
      stats_msg_count++;
    }
  }

  const int clients = num_neighbors();
  if (stats_msg_count == clients && receive_stats_ready) {
    double strat_start_time = CmiWallTimer();
    receive_stats_ready = 0;
    WSLBMigrateMsg* migrateMsg = Strategy(statsDataList,clients);

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
      WSLBMigrateMsg* m2 = (WSLBMigrateMsg*) CkCopyMsg((void**)&migrateMsg);
      CProxy_WSLB(thisgroup).ReceiveMigration(m2,neighbor_pes[i]);
    }
    if (0 < num_neighbors())
      CProxy_WSLB(thisgroup).ReceiveMigration(migrateMsg,
						    neighbor_pes[0]);
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

void WSLB::ReceiveMigration(WSLBMigrateMsg *msg)
{
  if (neighbor_pes == 0) FindNeighbors();

  if (mig_msgs_received == 0) migrates_expected = 0;

  mig_msgs[mig_msgs_received] = msg;
  mig_msgs_received++;
  //  CkPrintf("[%d] Received migration msg %d of %d\n",
  //	   CkMyPe(),mig_msgs_received,mig_msgs_expected);

  if (mig_msgs_received > mig_msgs_expected) {
    CkPrintf("[%d] WSLB Error! Too many migration messages received\n",
	     CkMyPe());
  }

  if (mig_msgs_received != mig_msgs_expected) {
    return;
  }

  //  CkPrintf("[%d] in ReceiveMigration %d moves\n",CkMyPe(),msg->n_moves);
  for(int neigh=0; neigh < mig_msgs_received;neigh++) {
    WSLBMigrateMsg* m = mig_msgs[neigh];
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


void WSLB::MigrationDone()
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
  CProxy_WSLB(thisgroup).ResumeClients(CkMyPe());
}

void WSLB::ResumeClients()
{
  theLbdb->ResumeClients();
}

CmiBool WSLB::QueryBalanceNow(int step)
{
  double now = CmiWallTimer();

  if (step==0)
    first_step_time = now;
  else if (CkMyPe() == VACATE_PROC && now > VACATE_AFTER
	   && now < (VACATE_AFTER+UNVACATE_AFTER)) {
    if (vacate == CmiFalse) 
      CkPrintf("PE %d vacating at %f\n",CkMyPe(),now);
    vacate = CmiTrue;
  } else {
    if (vacate == CmiTrue)
      CkPrintf("PE %d unvacating at %f\n",CkMyPe(),now);
    vacate = CmiFalse;
  }

  return CmiTrue;
}

WSLBMigrateMsg* WSLB::Strategy(WSLB::LDStats* stats, int count)
{
  //  CkPrintf("[%d] Strategy starting\n",CkMyPe());
  // Compute the average load to see if we are overloaded relative
  // to our neighbors
  const double load_factor = 1.05;
  double objload;

  double myobjcpu=0;
  double myobjwall=0;

  double myload = myStats.total_walltime - myStats.idletime;
  double avgload = myload;
  int unvacated_neighbors = 0;
  int i;
  for(i=0; i < count; i++) {
    // If the neighbor is vacating, skip him
    if (stats[i].vacate_me)
      continue;

    // Scale times we need appropriately for relative proc speeds
    double hisload = stats[i].total_walltime - stats[i].idletime;
    const double hisusage = stats[i].usage;

    const double scale =  (myStats.proc_speed * usage) 
      / (stats[i].proc_speed * hisusage);

    hisload *= scale;
    stats[i].total_walltime *= scale;
    stats[i].idletime *= scale;

    //    CkPrintf("PE %d %d hisload = %f hisusage = %f\n",
    //	     CkMyPe(),i,hisload,hisusage);
    avgload += hisload;
    unvacated_neighbors++;
  }
  if (vacate && unvacated_neighbors == 0)
    CkPrintf("[%d] ALL NEIGHBORS WANT TO VACATE!!!\n",CkMyPe());

  avgload /= (unvacated_neighbors+1);

  CkVector migrateInfo;

  // If we want to vacate, we always dump our load, otherwise
  // only if we are overloaded

  if (vacate || myload > avgload) {
    //    CkPrintf("[%d] OVERLOAD My load is %f, average load is %f\n",
    //	     CkMyPe(),myload,avgload);

    // First, build heaps of other processors and my objects
    // Then assign objects to other processors until either
    //   - The smallest remaining object would put me below average, or
    //   - I only have 1 object left, or
    //   - The smallest remaining object would put someone else 
    //     above average

    // Build heaps
    minHeap procs(count);
    for(i=0; i < count; i++) {
      // If all my neighbors vacate, I won't have anyone to give work 
      // to
      if (!stats[i].vacate_me) {
	InfoRecord* item = new InfoRecord;
	item->load = stats[i].total_walltime - stats[i].idletime;
	item->Id =  stats[i].from_pe;
	procs.insert(item);
      }
    }
      
    maxHeap objs(myStats.obj_data_sz);
    for(i=0; i < myStats.obj_data_sz; i++) {
      InfoRecord* item = new InfoRecord;
      item->load = myStats.objData[i].wallTime;
      item->Id = i;
      objs.insert(item);
    }

    int objs_here = myStats.obj_data_sz;
    do {
      //      if (objs_here <= 1) break;  // For now, always leave 1 object

      InfoRecord* p;
      InfoRecord* obj;

      // Get the lightest-loaded processor
      p = procs.deleteMin();
      if (p == 0) {
	//	CkPrintf("[%d] No destination PE found!\n",CkMyPe());
	break;
      }

      // Get the biggest object
      CmiBool objfound = CmiFalse;
      do {
	obj = objs.deleteMax();
	if (obj == 0) break;

	objload = load_factor * obj->load;

	double new_p_load = p->load + objload;
	double my_new_load = myload - objload;

	// If we're vacating, the biggest object is always good.
	// Otherwise, only take it if it doesn't produce overload
	if (vacate || new_p_load < my_new_load) {
	  objfound = CmiTrue;
	} else {
	  // This object is too big, so throw it away
//	  CkPrintf("[%d] Can't move object w/ load %f to proc %d load %f %f\n",
//		   CkMyPe(),obj->load,p->Id,p->load,avgload);
	  delete obj;
	}
      } while (!objfound);

      if (!objfound) {
	//	CkPrintf("[%d] No suitable object found!\n",CkMyPe());
	break;
      }

      const int me = CkMyPe();
      // Apparently we can give this object to this processor
      CkPrintf("[%d] Obj %d of %d migrating from %d to %d\n",
	       CkMyPe(),obj->Id,myStats.obj_data_sz,me,p->Id);

      MigrateInfo* migrateMe = new MigrateInfo;
      migrateMe->obj = myStats.objData[obj->Id].handle;
      migrateMe->from_pe = me;
      migrateMe->to_pe = p->Id;
      migrateInfo.push_back((void*)migrateMe);

      objs_here--;
      
      // We may want to assign more to this processor, so lets
      // update it and put it back in the heap
      p->load += objload;
      myload -= objload;
      procs.insert(p);
      
      // This object is assigned, so we delete it from the heap
      delete obj;

    } while(vacate || myload > avgload);

    // Now empty out the heaps
    while (InfoRecord* p=procs.deleteMin())
      delete p;
    while (InfoRecord* obj=objs.deleteMax())
      delete obj;
  }  

  // Now build the message to actually perform the migrations
  int migrate_count=migrateInfo.size();
  //  if (migrate_count) {
  //    CkPrintf("PE %d: Sent away %d of %d objects\n",
  //	     CkMyPe(),migrate_count,myStats.obj_data_sz);
  //  }
  WSLBMigrateMsg* msg = new(&migrate_count,1) WSLBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  return msg;
};

void* WSLBMigrateMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = size + array[0] * sizeof(WSLB::MigrateInfo);

  WSLBMigrateMsg* ret =
    static_cast<WSLBMigrateMsg*>(CkAllocMsg(msgnum,totalsize,priobits));

  ret->moves = reinterpret_cast<WSLB::MigrateInfo*>
    (reinterpret_cast<char*>(ret)+ size);

  return static_cast<void*>(ret);
}

void* WSLBMigrateMsg::pack(WSLBMigrateMsg* m)
{
  m->moves = reinterpret_cast<WSLB::MigrateInfo*>
    (reinterpret_cast<char*>(m->moves) - reinterpret_cast<char*>(&m->moves));

  return static_cast<void*>(m);
}

WSLBMigrateMsg* WSLBMigrateMsg::unpack(void *m)
{
  WSLBMigrateMsg* ret_val = static_cast<WSLBMigrateMsg*>(m);

  ret_val->moves = reinterpret_cast<WSLB::MigrateInfo*>
    (reinterpret_cast<char*>(&ret_val->moves) 
     + reinterpret_cast<size_t>(ret_val->moves));

  return ret_val;
}

#endif
