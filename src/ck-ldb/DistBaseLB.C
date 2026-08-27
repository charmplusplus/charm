/**
 * Author: gplkrsh2@illinois.edu (Harshitha Menon)
 * Base class for distributed load balancer.
*/

#include "BaseLB.h"
#include "ckrdmadevice.h"
#include "DistBaseLB.h"
#include "DistBaseLB.def.h"

#if CMK_CUDA
#include <cupti.h>
#include "gpumanager.h"
#include "hapi.h"
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
  // AssembleStats copies object data out of the LB database. Without this
  // LDObjData::gpuTime stays zero for every distributed strategy, so
  // DiffusionLB's across-node dimension (+LBDiffusionGpuDim, which diffuses on
  // gpuTime) sees no load at all and diffuses nothing -- the device work is
  // invisible because the host only enqueues kernels. CentralLB::CallLB does
  // exactly this for the centralized strategies; the distributed path was
  // simply never given the same treatment.
  // Whichever PE thread gets here first does the work; the rest block inside
  // until it is done, so nobody reads cupti_obj_norm_load_ while it is being
  // rebuilt. This used to be "rank 0 does it between two CmiNodeBarrier calls",
  // but both barriers were behind #if CMK_SMP -- which is 0 in the multicore
  // build even though a process really does run many PE threads -- so the
  // barriers compiled away and the other ranks raced the rebuild.
  hapiPrepareCuptiLoads();
  // Every PE picks up the normalized loads for its own objects.
  lbmgr->SetObjGPULoad(CsvAccess(gpu_manager).cupti_obj_norm_load_);
  if (_lb_args.gpuScaling())
    lbmgr->SetObjGPUCosts(CsvAccess(gpu_manager).cupti_obj_epoch_costs_);
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

  myStats.objData.clear();
  myStats.objData.resize(lbmgr->GetObjDataSz());
  lbmgr->GetObjData(myStats.objData.data());

  myStats.commData.clear();
  myStats.commData.resize(lbmgr->GetCommDataSz());
  lbmgr->GetCommData(myStats.commData.data());

  // CHARM_LB_LOADDUMP: is there anything here to balance? Prints this PE's
  // totals and spread in both dimensions, so a strategy that migrates a lot
  // and gains nothing can be told apart from one that never saw any load.
  if (getenv("CHARM_LB_LOADDUMP")) {
    double sw = 0, sg = 0, mw = 0, mg = 0;
    for (const auto& o : myStats.objData) {
      sw += o.wallTime;
      if (o.wallTime > mw) mw = o.wallTime;
#if CMK_CUDA
      sg += o.gpuTime;
      if (o.gpuTime > mg) mg = o.gpuTime;
#endif
    }
    CmiPrintf("[LBLOAD pe=%d] objs=%zu wall_sum=%.6f wall_max=%.6f "
              "gpu_sum=%.6f gpu_max=%.6f busyIpcSlots=%d\n",
              CkMyPe(), myStats.objData.size(), sw, mw, sg, mg,
              CkRdmaDeviceBusyIpcSlots());
    fflush(stdout);
  }

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
// one at a time with lbmgr->Migrate() instead of handing down a move list (see
// DiffusionLB's per-object sends). Those migrations are invisible to the
// list-based broadcast above, so without this every PE keeps a stale cached
// location for the moved object -- which for a GPU-direct send means the sender
// picks its transfer mode for the wrong process (see CkRdmaDeviceOnSender).
void DistBaseLB::BroadcastSingleLocationUpdate(const LDObjHandle& h, int to_pe) {
  const int sizes = 1;
  LBMigrateMsg* msg = new(sizes, CkNumPes(), CkNumPes(), 0) LBMigrateMsg;
  msg->n_moves = 1;
  msg->moves[0].index = 0;
  msg->moves[0].obj = h;
  msg->moves[0].from_pe = CkMyPe();
  msg->moves[0].to_pe = to_pe;
  msg->moves[0].async_arrival = 0;
  thisProxy.ReceiveLocationUpdate(msg);
}

void DistBaseLB::ReceiveLocationUpdate(LBMigrateMsg* msg) {
  const int me = CkMyPe();
  for (int i = 0; i < msg->n_moves; i++) {
    MigrateInfo& move = msg->moves[i];
    if (move.from_pe != me && move.to_pe != me) {
      UpdateLocation(move);
    }
  }
  delete msg;
}
#endif

void DistBaseLB::MigrationDone(int balancing) {
#if CMK_LBDB_ON
  // Reset the lb_started flag to indicate that the lb is done
  lb_started = false;
  // Increment to next step
  lbmgr->incStep();
  lbmgr->ClearLoads();
#if CMK_CUDA
  // Drop the kernel records this round's loads were derived from, so the next
  // round measures the next interval rather than everything since startup
  // (mirrors CentralLB::ProcessMigrationDecision).
  if (CmiMyRank() == 0)
    hapiClearCuptiData();
#endif

  // Settle before resuming.
  //
  // A PE that resumes its own objects as soon as its own migrations are done
  // lets them start sending to objects whose new home this PE has not learned
  // yet -- the location updates for another PE's moves are still in flight.
  // For an ordinary message that is harmless: it goes to the stale PE and is
  // forwarded. For a device zerocopy send it is not. The sender picks its
  // transfer mode from that stale location, and picking MEMCPY (same process)
  // means staging nothing at all, because the receiver is expected to read the
  // source pointer directly. When the message is then forwarded into a
  // different process, that pointer is in an address space the receiver cannot
  // read and there is no staged copy to fall back on -- see the abort in
  // CkRdmaDeviceIssueRgets, which is where this used to surface, as an
  // unregistered rdmaGet or an illegal access on an unrelated stream.
  //
  // So on a CUDA build the barrier is not optional -- +LBSyncResume stops being
  // something the user has to know to pass. Elsewhere the existing behaviour is
  // unchanged.
  //
  // The switch itself is +LBSyncResume, which on a CUDA build LBManager now
  // defaults to on. It has to be that one switch rather than a barrier forced
  // from here: DiffusionLB reads the same flag to decide which of its own two
  // paths performs the migrations, so turning the barrier on behind its back
  // leaves the two halves waiting on each other.
  if (balancing && _lb_args.syncResume()) {
    contribute(CkCallback(CkReductionTarget(DistBaseLB, ResumeClients),
                thisProxy));
  }
  else 
    thisProxy [CkMyPe()].ResumeClients(balancing);
#endif
}

// Quiescence would be the stronger guarantee for the barrier above -- it would
// prove the location updates have been *processed* everywhere, not just that the
// moves finished -- but it cannot be used from inside the LB step: DiffusionLB
// drives its own phases off CkStartQD, so an extra quiescence point in the
// middle of its sequence re-fires whichever of its phase callbacks is still
// armed, and the step never completes. The reduction is what is available, and
// it closes the window that matters: every PE has finished migrating and has
// broadcast its location updates before any PE resumes. A residual stale-location
// send remains possible in principle, and is caught precisely by the abort in
// CkRdmaDeviceIssueRgets rather than corrupting memory.

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
