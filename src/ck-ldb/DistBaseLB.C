/**
 * Author: gplkrsh2@illinois.edu (Harshitha Menon)
 * Base class for distributed load balancer.
*/

#include "BaseLB.h"
#include "DistBaseLB.h"
#include "DistBaseLB.def.h"

#if CMK_CUDA
#include "hapi.h"
#include "gpumanager.h"
CsvExtern(GPUManager, gpu_manager);
#endif

#define  DEBUGF(x)      // CmiPrintf x;

#if CMK_GLOBAL_LOCATION_UPDATE
extern void UpdateLocation(MigrateInfo& migData);
#endif

void DistBaseLB::barrierDone() {
#if CMK_LBDB_ON
  if (lb_started) {
    return;
  }
  lb_started = true;

  start_lb_time = 0;

  if (CkNumPes() == 1) {
    MigrationDone(0);
    return;
  }

  start_lb_time = CkWallTimer();
  if (CkMyPe() == 0) {
    if (_lb_args.debug()) {
      CkPrintf("[%s] Load balancing step %d starting at %f\n",
          lbName(), step(),start_lb_time);
    }
  }

#if CMK_CUDA
  // Turn the CUPTI kernel timeline into a per-object GPU load before
  // AssembleStats copies object data out of the LB database, as
  // CentralLB::InvokeLB does; without it LDObjData::gpuTime is zero for every
  // distributed strategy, and a device-bound application's load is invisible
  // because the host only enqueues kernels.
  //
  // Built by the LAST PE of the process to get here, not the first: this
  // barrier fires when this PE's own objects are at AtSync, but the records
  // are shared by the whole process, and a drain by the first arrival drops
  // every kernel the other PEs' objects were still finishing. The earlier
  // arrivals continue from gpuLoadsReady.
  if (!hapiCuptiArrive((uint64_t)step(), CkNodeSize(CkMyNode()))) return;
  hapiPrepareCuptiLoads();
  const int first = CkNodeFirst(CkMyNode());
  for (int r = 0; r < CkNodeSize(CkMyNode()); r++)
    thisProxy[first + r].gpuLoadsReady();
#else
  gpuLoadsReady();
#endif
#endif
}

// The round's GPU loads exist (or there are none to build): copy this PE's
// share out, assemble its stats and start the strategy.
void DistBaseLB::gpuLoadsReady() {
#if CMK_LBDB_ON
#if CMK_CUDA
  lbmgr->SetObjGPULoad(CsvAccess(gpu_manager).cupti_obj_norm_load_);
#endif
  AssembleStats();
  thisProxy[CkMyPe()].LoadBalance();
#endif
}

void DistBaseLB::InvokeLB() {
  // Ensure that the strategy starts only after the barrier
  CkCallback cb (CkReductionTarget(DistBaseLB, barrierDone), thisProxy);
  contribute(cb);
}

DistBaseLB::DistBaseLB(const CkLBOptions &opt): CBase_DistBaseLB(opt) {
#if CMK_LBDB_ON
  lbname = (char *)"DistBaseLB";
  thisProxy = CProxy_DistBaseLB(thisgroup);
  startLbFnHdl = lbmgr->AddStartLBFn(this, &DistBaseLB::barrierDone);

  if (opt.getSeqNo() > 0)
    turnOff();

  migrates_completed = 0;
  migrates_expected = 0;
  lb_started = false;
  mig_msgs = NULL;

  myStats.pe_speed = lbmgr->ProcessorSpeed();
  myStats.from_pe = CkMyPe();

  if (_lb_args.statsOn()) {
    lbmgr->CollectStatsOn();
  }
#endif
}

DistBaseLB::~DistBaseLB() {
#if CMK_LBDB_ON
  lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  if (lbmgr) {
    lbmgr->RemoveStartLBFn(startLbFnHdl);
  }
  if (mig_msgs) {
    delete [] mig_msgs;
  }
#endif
}

// Assemble the stats for the local PE. The stats are collected by the
// LBManager so assemble all the stats.
void DistBaseLB::AssembleStats() {
#if CMK_LBDB_ON
#if CMK_LB_CPUTIMER
  lbmgr->TotalTime(&myStats.total_walltime,&myStats.total_cputime);
  lbmgr->BackgroundLoad(&myStats.bg_walltime,&myStats.bg_cputime);
#else
  lbmgr->TotalTime(&myStats.total_walltime,&myStats.total_walltime);
  lbmgr->BackgroundLoad(&myStats.bg_walltime,&myStats.bg_walltime);
#endif
  lbmgr->IdleTime(&myStats.idletime);

  myStats.move = true;

#if CMK_CUDA
  hapiLBDeviceMemory(&myStats.gpu_mem_remaining, &myStats.pool_buff_mem_remaining,
                     &myStats.gpu_pool_arena_bytes, &myStats.gpu_ipc_slots);
  myStats.gpu_pool_capacity_bytes = hapiLBDevicePoolCapacity();
#endif

  myStats.objData.clear();
  myStats.objData.resize(lbmgr->GetObjDataSz());
  lbmgr->GetObjData(myStats.objData.data());

  myStats.commData.clear();
  myStats.commData.resize(lbmgr->GetCommDataSz());
  lbmgr->GetCommData(myStats.commData.data());

  myStats.obj_walltime = 0;
#if CMK_LB_CPUTIMER
  myStats.obj_cputime = 0;
#endif
  const int n_objs = myStats.objData.size();
  for(int i = 0; i < n_objs; i++) {
    myStats.obj_walltime += myStats.objData[i].wallTime;
#if CMK_LB_CPUTIMER
    myStats.obj_cputime += myStats.objData[i].cpuTime;
#endif
  }    
#endif
}

void DistBaseLB::LoadBalance() {
#if CMK_LBDB_ON
  strat_start_time = CkWallTimer();

  if (CkMyPe() == 0 &&  _lb_args.debug()) {
    CkPrintf("DistLB> %s: step %d starting at %f Memory: %f MB\n",
        lbname, step(), strat_start_time, CmiMemoryUsage()/(1024.0*1024.0));
  }

  migrates_expected = 0;
  migrates_completed = 0;
  Strategy(&myStats);
#endif  
}

void DistBaseLB::Migrated(int waitBarrier) {
  migrates_completed++;
  if (migrates_completed == migrates_expected && lb_started) {
    MigrationDone(1);
  }
}

/*
* Migrates the objs from my PE according to the new mapping specified in the
* migrateMsg
*/
void DistBaseLB::ProcessMigrationDecision(LBMigrateMsg *migrateMsg) {
#if CMK_LBDB_ON
  strat_end_time = CkWallTimer() - strat_start_time;
  const int me = CkMyPe();

  // Migrate messages from me to elsewhere
  for(int i=0; i < migrateMsg->n_moves; i++) {
    MigrateInfo& move = migrateMsg->moves[i];
    if (move.from_pe == me) {
      if (move.to_pe == me) {
        CkAbort("[%i] Error, attempting to migrate object myself to myself\n",
            CkMyPe());
      }
      lbmgr->Migrate(move.obj,move.to_pe);
    } else if (move.from_pe != me) {
      CkPrintf("[%d] Error, strategy wants to move from %d to  %d\n",
          me,move.from_pe,move.to_pe);
      CkAbort("Trying to move objs not on my PE\n");
    }
  }

#if CMK_GLOBAL_LOCATION_UPDATE
  BroadcastLocationUpdate(migrateMsg);
#endif

  if (CkMyPe() == 0) {
    double strat_end_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("%s> Strategy took %fs memory usage: %f MB.\n", lbName(),
          strat_end_time - strat_start_time, CmiMemoryUsage()/(1024.0*1024.0));
  }

  // The message is this function's to free: its callers build it, hand it
  // over and forget it, and BroadcastLocationUpdate above copies rather than
  // keeping it. Freed before MigrationDone, which can run the whole tail of
  // the step.
  delete migrateMsg;

  // If all the expected objs have migrated in, then migration is done
  if (migrates_expected == migrates_completed && lb_started) {
    MigrationDone(1);
  }
#endif
}

#if CMK_GLOBAL_LOCATION_UPDATE
// Each PE only knows the moves it is the source of -- there is no global move
// list to hand a bystander PE the way CentralLB::ReceiveMigration does. So the
// source broadcasts its own moves, and every PE that is neither the source nor
// the destination of a given move (the migration mechanics already update
// those two directly) refreshes its cache for it.
void DistBaseLB::BroadcastLocationUpdate(LBMigrateMsg* migrateMsg) {
  if (migrateMsg->n_moves == 0) return;
  void* copy = CkCopyMsg((void**)&migrateMsg);
  thisProxy.ReceiveLocationUpdate((LBMigrateMsg*)copy);
}

// Same purpose as BroadcastLocationUpdate, for a strategy that migrates objects
// one at a time with lbmgr->Migrate() instead of handing down a move list.
// Those migrations are invisible to the list-based broadcast above, so without
// this every PE keeps a stale cached location for the moved object.
void DistBaseLB::BroadcastSingleLocationUpdate(const LDObjHandle& h, int to_pe) {
  LBMigrateMsg* msg = new(1, CkNumPes(), CkNumPes(), 0) LBMigrateMsg;
  msg->n_moves = 1;
  msg->moves[0].index = 0;
  msg->moves[0].obj = h;
  msg->moves[0].from_pe = CkMyPe();
  msg->moves[0].to_pe = to_pe;
  msg->moves[0].async_arrival = 0;
  msg->moves[0].kind = LB_DECISION_MOVE;
  msg->moves[0].rank = 0;
  thisProxy.ReceiveLocationUpdate(msg);
}
#endif

void DistBaseLB::ReceiveLocationUpdate(LBMigrateMsg* msg) {
#if CMK_GLOBAL_LOCATION_UPDATE
  const int me = CkMyPe();
  for (int i = 0; i < msg->n_moves; i++) {
    MigrateInfo& move = msg->moves[i];
    if (move.from_pe != me && move.to_pe != me) {
      UpdateLocation(move);
    }
  }
#endif
  delete msg;
}

void DistBaseLB::MigrationDone(int balancing) {
#if CMK_LBDB_ON
  // Reset the lb_started flag to indicate that the lb is done
  lb_started = false;
  // Increment to next step
  lbmgr->incStep();
  lbmgr->ClearLoads();

  // if sync resume invoke a barrier
  if (balancing && _lb_args.syncResume()) {
    contribute(CkCallback(CkReductionTarget(DistBaseLB, ResumeClients),
                thisProxy));
  }
  else 
    thisProxy [CkMyPe()].ResumeClients(balancing);
#endif
}

void DistBaseLB::ResumeClients() {
  ResumeClients(1);
}

void DistBaseLB::ResumeClients(int balancing) {
#if CMK_LBDB_ON
  DEBUGF(("[%d] ResumeClients. \n", CkMyPe()));

  if (CkMyPe() == 0 && balancing) {
    double end_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("%s> step %d finished at %f duration %f memory usage: %f\n",
          lbName(), step() - 1, end_lb_time, end_lb_time - strat_start_time,
          CmiMemoryUsage() / (1024.0 * 1024.0));
  }

  lbmgr->ResumeClients();
#endif
}

void DistBaseLB::Strategy(const LDStats* const stats) {
  int sizes=0;
  LBMigrateMsg* msg = new(sizes, CkNumPes(), CkNumPes(), 0) LBMigrateMsg;
  msg->n_moves = 0;
}
