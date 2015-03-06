/**
 * Author: gplkrsh2@illinois.edu (Harshitha Menon)
 * Base class for distributed load balancer.
*/

#include "BaseLB.h"
#include "DistBaseLB.h"
#include "LBDBManager.h"
#include "DistBaseLB.def.h"

#define  DEBUGF(x)      // CmiPrintf x;

void DistBaseLB::staticMigrated(void* data, LDObjHandle h, int waitBarrier) {
  DistBaseLB *me = (DistBaseLB*)(data);
  me->Migrated(h, waitBarrier);
}

void DistBaseLB::staticAtSync(void* data) {
  DistBaseLB *me = (DistBaseLB*)(data);
  me->ProcessAtSync();
}

void DistBaseLB::staticStartLB(void* data) {
  DistBaseLB *me = (DistBaseLB*)(data);
  me->barrierDone();
}

void DistBaseLB::barrierDone() {
  thisProxy.AtSync();
}

void DistBaseLB::ProcessAtSync() {
  // Ensuring that the strategy starts only after the barrier
  CkCallback cb (CkReductionTarget(DistBaseLB, barrierDone), 0, thisProxy);
  contribute(cb);
}

DistBaseLB::DistBaseLB(const CkLBOptions &opt): CBase_DistBaseLB(opt) {
#if CMK_LBDB_ON
  lbname = (char *)"DistBaseLB";
  thisProxy = CProxy_DistBaseLB(thisgroup);
  receiver = theLbdb->AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
      (void*)(this));
  notifier = theLbdb->getLBDB()->NotifyMigrated((LDMigratedFn)(staticMigrated),
      (void*)(this));
  startLbFnHdl = theLbdb->getLBDB()->AddStartLBFn((LDStartLBFn)(staticStartLB),
      (void*)(this));
  theLbdb->AddStartLBFn((LDStartLBFn)(staticStartLB),(void*)this);

  migrates_completed = 0;
  migrates_expected = 0;
  lb_started = false;
  mig_msgs = NULL;

  myStats.pe_speed = theLbdb->ProcessorSpeed();
  myStats.from_pe = CkMyPe();
  myStats.n_objs = 0;
  myStats.objData = NULL;
  myStats.n_comm = 0;
  myStats.commData = NULL;

  if (_lb_args.statsOn()) {
    theLbdb->CollectStatsOn();
  }
#endif
}

DistBaseLB::~DistBaseLB() {
#if CMK_LBDB_ON
  theLbdb = CProxy_LBDatabase(_lbdb).ckLocalBranch();
  if (theLbdb) {
    theLbdb->getLBDB()->RemoveNotifyMigrated(notifier);
    theLbdb-> RemoveStartLBFn((LDStartLBFn)(staticStartLB));
  }
  if (mig_msgs) {
    delete [] mig_msgs;
  }
#endif
}


void DistBaseLB::AtSync() {
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

  AssembleStats();
  thisProxy[CkMyPe()].LoadBalance();
#endif
}

// Assemble the stats for the local PE. The stats are collected by the
// LBDatabase so assembl all the stats.
void DistBaseLB::AssembleStats() {
#if CMK_LBDB_ON
#if CMK_LB_CPUTIMER
  theLbdb->TotalTime(&myStats.total_walltime,&myStats.total_cputime);
  theLbdb->BackgroundLoad(&myStats.bg_walltime,&myStats.bg_cputime);
#else
  theLbdb->TotalTime(&myStats.total_walltime,&myStats.total_walltime);
  theLbdb->BackgroundLoad(&myStats.bg_walltime,&myStats.bg_walltime);
#endif
  theLbdb->IdleTime(&myStats.idletime);

  myStats.move = true; 

  myStats.n_objs = theLbdb->GetObjDataSz();
  if (myStats.objData) delete [] myStats.objData;
  myStats.objData = new LDObjData[myStats.n_objs];
  theLbdb->GetObjData(myStats.objData);

  myStats.n_comm = theLbdb->GetCommDataSz();
  if (myStats.commData) delete [] myStats.commData;
  myStats.commData = new LDCommData[myStats.n_comm];
  theLbdb->GetCommData(myStats.commData);

  myStats.obj_walltime = 0;
#if CMK_LB_CPUTIMER
  myStats.obj_cputime = 0;
#endif
  for(int i=0; i < myStats.n_objs; i++) {
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

void DistBaseLB::Migrated(LDObjHandle h, int waitBarrier) {
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
    if (move.from_pe == me && move.to_pe != me) {
      theLbdb->Migrate(move.obj,move.to_pe);
    } else if (move.from_pe != me) {
      CkPrintf("[%d] Error, strategy wants to move from %d to  %d\n",
          me,move.from_pe,move.to_pe);
      CkAbort("Trying to move objs not on my PE\n");
    }
  }

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

void DistBaseLB::MigrationDone(int balancing) {
#if CMK_LBDB_ON
  // Reset the lb_started flag to indicate that the lb is done
  lb_started = false;
  // Increment to next step
  theLbdb->incStep();
  theLbdb->ClearLoads();

  // if sync resume invoke a barrier
  if (balancing && _lb_args.syncResume()) {
    CkCallback cb(CkIndex_DistBaseLB::ResumeClients((CkReductionMsg*)NULL), 
        thisProxy);
    contribute(0, NULL, CkReduction::sum_int, cb);
  }
  else 
    thisProxy [CkMyPe()].ResumeClients(balancing);
#endif
}

void DistBaseLB::ResumeClients(CkReductionMsg *msg) {
  ResumeClients(1);
  delete msg;
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

  theLbdb->ResumeClients();
#endif
}

void DistBaseLB::Strategy(const LDStats* const stats) {
  int sizes=0;
  LBMigrateMsg* msg = new(sizes, CkNumPes(), CkNumPes(), 0) LBMigrateMsg;
  msg->n_moves = 0;
}
