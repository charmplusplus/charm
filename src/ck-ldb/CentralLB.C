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
#include "envelope.h"
#include "CentralLB.h"
#include "LBDBManager.h"
#include "LBSimulation.h"

#define  DEBUGF(x)      // CmiPrintf x;

CkGroupID loadbalancer;
int * lb_ptr;
int load_balancer_created;

CreateLBFunc_Def(CentralLB, "CentralLB base class");

void getLoadInfo(BaseLB::LDStats* stats, int count, 
		             LBInfo &info, int considerComm);

static void getPredictedLoadWithMsg(BaseLB::LDStats* stats, int count, 
		             LBMigrateMsg *, LBInfo &info, int considerComm);

/*
void CreateCentralLB()
{
  CProxy_CentralLB::ckNew(0);
}
*/

void CentralLB::staticStartLB(void* data)
{
  CentralLB *me = (CentralLB*)(data);
  me->StartLB();
}

void CentralLB::staticMigrated(void* data, LDObjHandle h, int waitBarrier)
{
  CentralLB *me = (CentralLB*)(data);
  me->Migrated(h, waitBarrier);
}

void CentralLB::staticAtSync(void* data)
{
  CentralLB *me = (CentralLB*)(data);
  me->AtSync();
}

void CentralLB::initLB(const CkLBOptions &opt)
{
#if CMK_LBDB_ON
  lbname = "CentralLB";
  thisProxy = CProxy_CentralLB(thisgroup);
  //  CkPrintf("Construct in %d\n",CkMyPe());

  // create and turn on by default
  receiver = theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),(void*)(this));
  notifier = theLbdb->getLBDB()->
    NotifyMigrated((LDMigratedFn)(staticMigrated),(void*)(this));
  startLbFnHdl = theLbdb->getLBDB()->
    AddStartLBFn((LDStartLBFn)(staticStartLB),(void*)(this));

  // CkPrintf("[%d] CentralLB seq %d\n",CkMyPe(), seq);
  if (opt.getSeqNo() > 0) turnOff();

  stats_msg_count = 0;
  statsMsgsList = new CLBStatsMsg*[CkNumPes()];
  for(int i=0; i < CkNumPes(); i++)
    statsMsgsList[i] = 0;

  statsData = new LDStats;

  // for future predictor
  if (_lb_predict) predicted_model = new FutureModel(_lb_predict_window);
  else predicted_model=0;
  // register user interface callbacks
  theLbdb->getLBDB()->SetupPredictor((LDPredictModelFn)(staticPredictorOn),(LDPredictWindowFn)(staticPredictorOnWin),(LDPredictFn)(staticPredictorOff),(LDPredictModelFn)(staticChangePredictor),(void*)(this));

  myspeed = theLbdb->ProcessorSpeed();

  migrates_completed = 0;
  future_migrates_completed = 0;
  migrates_expected = -1;
  future_migrates_expected = -1;
  cur_ld_balancer = 0;
  lbdone = 0;
  int num_proc = CkNumPes();

  if (_lb_args.statsOn()) theLbdb->CollectStatsOn();

  load_balancer_created = 1;
#endif
}

CentralLB::~CentralLB()
{
#if CMK_LBDB_ON
  delete [] statsMsgsList;
  delete statsData;
  theLbdb = CProxy_LBDatabase(_lbdb).ckLocalBranch();
  if (theLbdb) {
    theLbdb->getLBDB()->
      RemoveNotifyMigrated(notifier);
    theLbdb->
      RemoveStartLBFn((LDStartLBFn)(staticStartLB));
  }
#endif
}

void CentralLB::turnOn() 
{
#if CMK_LBDB_ON
  theLbdb->getLBDB()->
    TurnOnBarrierReceiver(receiver);
  theLbdb->getLBDB()->
    TurnOnNotifyMigrated(notifier);
  theLbdb->getLBDB()->
    TurnOnStartLBFn(startLbFnHdl);
#endif
}

void CentralLB::turnOff() 
{
#if CMK_LBDB_ON
  theLbdb->getLBDB()->
    TurnOffBarrierReceiver(receiver);
  theLbdb->getLBDB()->
    TurnOffNotifyMigrated(notifier);
  theLbdb->getLBDB()->
    TurnOffStartLBFn(startLbFnHdl);
#endif
}

void CentralLB::AtSync()
{
#if CMK_LBDB_ON
  DEBUGF(("[%d] CentralLB AtSync step %d!!!!!\n",CkMyPe(),step()));

  // if num of processor is only 1, nothing should happen
  if (!QueryBalanceNow(step()) || CkNumPes() == 1) {
    MigrationDone(0);
    return;
  }
  thisProxy [CkMyPe()].ProcessAtSync();
#endif
}

#include "ComlibStrategy.h"

void CentralLB::ProcessAtSync()
{
#if CMK_LBDB_ON

  if (CkMyPe() == cur_ld_balancer) {
    start_lb_time = CkWallTimer();
  }
  // build and send stats
  const int osz = theLbdb->GetObjDataSz();
  const int csz = theLbdb->GetCommDataSz();

  int npes = CkNumPes();
  CLBStatsMsg* msg = new CLBStatsMsg(osz, csz);
  msg->from_pe = CkMyPe();
  //msg->serial = CrnRand();

/*
  theLbdb->TotalTime(&msg->total_walltime,&msg->total_cputime);
  theLbdb->IdleTime(&msg->idletime);
  theLbdb->BackgroundLoad(&msg->bg_walltime,&msg->bg_cputime);
*/
  theLbdb->GetTime(&msg->total_walltime,&msg->total_cputime,
		   &msg->idletime, &msg->bg_walltime,&msg->bg_cputime);
  msg->pe_speed = myspeed;
//  CkPrintf(
//    "Processors %d Total time (wall,cpu) = %f %f Idle = %f Bg = %f %f\n",
//    CkMyPe(),msg->total_walltime,msg->total_cputime,
//    msg->idletime,msg->bg_walltime,msg->bg_cputime);

  msg->n_objs = osz;
  theLbdb->GetObjData(msg->objData);
  msg->n_comm = csz;
  theLbdb->GetCommData(msg->commData);
//  theLbdb->ClearLoads();
  DEBUGF(("PE %d sending stats to ReceiveStats %d objs, %d comm\n",
           CkMyPe(),msg->n_objs,msg->n_comm));

// Scheduler PART.

  if(CkMyPe() == cur_ld_balancer) {
    msg->avail_vector = new char[CkNumPes()];
    LBDatabaseObj()->get_avail_vector(msg->avail_vector);
    msg->next_lb = LBDatabaseObj()->new_lbbalancer();
  }

#ifdef __BLUEGENE__
  BgStartStreaming();
#endif

  thisProxy[cur_ld_balancer].ReceiveStats(msg);

#ifdef __BLUEGENE__
  BgEndStreaming();
#endif

  {
  // enfore the barrier to wait until centralLB says no
  LDOMHandle h;
  h.id.id.idx = 0;
  theLbdb->getLBDB()->RegisteringObjects(h);
  }
#endif
}

void CentralLB::Migrated(LDObjHandle h, int waitBarrier)
{
#if CMK_LBDB_ON
  if (waitBarrier) {
    migrates_completed++;
    //  CkPrintf("[%d] An object migrated! %d %d\n",
    //  	   CkMyPe(),migrates_completed,migrates_expected);
    if (migrates_completed == migrates_expected) {
      MigrationDone(1);
    }
  }
  else {
    future_migrates_completed ++;
    DEBUGF(("[%d] An object migrated with no barrier! %d expected: %d\n",
    	   CkMyPe(),future_migrates_completed,future_migrates_expected));
    if (future_migrates_completed == future_migrates_expected)  {
	CheckMigrationComplete();
    }
  }
#endif
}

void CentralLB::MissMigrate(int waitForBarrier)
{
  LDObjHandle h;
  Migrated(h, waitForBarrier);
}

// build data from buffered msg
void CentralLB::buildStats()
{
    statsData->count = stats_msg_count;
    // allocate space
    statsData->objData.resize(statsData->n_objs);
    statsData->from_proc.resize(statsData->n_objs);
    statsData->to_proc.resize(statsData->n_objs);
    statsData->commData.resize(statsData->n_comm);

    int nobj = 0;
    int ncom = 0;
    int nmigobj = 0;
    // copy all data from individule message to this big structure
    for (int pe=0; pe<stats_msg_count; pe++) {
       int i;
       CLBStatsMsg *msg = statsMsgsList[pe];
       for (i=0; i<msg->n_objs; i++) {
         statsData->from_proc[nobj] = statsData->to_proc[nobj] = pe;
	 statsData->objData[nobj] = msg->objData[i];
         if (msg->objData[i].migratable) nmigobj++;
	 nobj++;
       }
       for (i=0; i<msg->n_comm; i++) {
	 statsData->commData[ncom] = msg->commData[i];
	 ncom++;
       }
       // free the memory
       delete msg;
       statsMsgsList[pe]=0;
    }
    statsData->n_migrateobjs = nmigobj;
    if (_lb_args.debug() > 1) {
      CmiPrintf("[%d] n_obj:%d migratable:%d ncom:%d\n", CkMyPe(), nobj, nmigobj, ncom);
    }
}

void CentralLB::ReceiveStats(CkMarshalledCLBStatsMessage &msg)
{
#if CMK_LBDB_ON
  CLBStatsMsg *m = (CLBStatsMsg *)msg.getMessage();
  const int pe = m->from_pe;
//  CkPrintf("Stats msg received, %d %d %d %p\n",
//  	   pe,stats_msg_count,m->n_objs,m);

  // update proc avail bit vector
//  if (pe == cur_ld_balancer && m->avail_vector) {
  if (m->avail_vector) {
      LBDatabaseObj()->set_avail_vector(m->avail_vector,  m->next_lb);
  }

  if (statsMsgsList[pe] != 0) {
    CkPrintf("*** Unexpected CLBStatsMsg in ReceiveStats from PE %d ***\n",
	     pe);
  } else {
    statsMsgsList[pe] = m;
    // store per processor data right away
    struct ProcStats &procStat = statsData->procs[pe];
    procStat.pe = pe;
    procStat.total_walltime = m->total_walltime;
    procStat.total_cputime = m->total_cputime;
    procStat.idletime = m->idletime;
    procStat.bg_walltime = m->bg_walltime;
    procStat.bg_cputime = m->bg_cputime;
    procStat.pe_speed = m->pe_speed;
    //procStat.utilization = 1.0;
    procStat.available = CmiTrue;
    procStat.n_objs = m->n_objs;

    statsData->n_objs += m->n_objs;
    statsData->n_comm += m->n_comm;
    stats_msg_count++;
  }

  DEBUGF(("[0] ReceiveStats from %d step: %d count: %d\n", pe, step(), stats_msg_count));
  const int clients = CkNumPes();

  if (stats_msg_count == clients) {
    thisProxy[CkMyPe()].LoadBalance();
  }
#endif
}

void CentralLB::LoadBalance()
{
#if CMK_LBDB_ON
  int proc;
  if (_lb_args.debug()) 
      CmiPrintf("[%s] Load balancing step %d starting at %f in PE%d\n",
                 lbName(), step(),start_lb_time, cur_ld_balancer);
//    double strat_start_time = CkWallTimer();

  // build data
  buildStats();

  // if we are in simulation mode read data
  if (LBSimulation::doSimulation) simulationRead();

  char *availVector = LBDatabaseObj()->availVector();
  const int clients = CkNumPes();
  for(proc = 0; proc < clients; proc++)
      statsData->procs[proc].available = (CmiBool)availVector[proc];

  preprocess(statsData, clients);

//    CkPrintf("Before Calling Strategy\n");

  double strat_start_time = CkWallTimer();
  LBMigrateMsg* migrateMsg = Strategy(statsData, clients);
  if (_lb_args.debug()) {
    CkPrintf("Strategy took %f seconds.\n", CkWallTimer()-strat_start_time);
    double lbdbMemsize = LBDatabase::Object()->useMem()/1000;
    CkPrintf("[%s] memUsage: LBManager:%dKB CentralLB:%dKB\n", 
  	      lbName(), (int)lbdbMemsize, (int)(useMem()/1000));
  }

//    CkPrintf("returned successfully\n");

  LBDatabaseObj()->get_avail_vector(migrateMsg->avail_vector);
  migrateMsg->next_lb = LBDatabaseObj()->new_lbbalancer();

  // if this is the step at which we need to dump the database
  simulationWrite();

//  calculate predicted load
//  very time consuming though, so only happen when debugging is on
  if (_lb_args.debug()>1) {
      LBInfo info(migrateMsg->expectedLoad, clients);
      getPredictedLoadWithMsg(statsData, clients, migrateMsg, info, 1);
  }

  //  CkPrintf("calling recv migration\n");
  thisProxy.ReceiveMigration(migrateMsg);

  // Zero out data structures for next cycle
  // CkPrintf("zeroing out data\n");
  statsData->clear();
  stats_msg_count=0;
#endif
}

//    double strat_end_time = CkWallTimer();
//    CkPrintf("Strat elapsed time %f\n",strat_end_time-strat_start_time);
// test if sender and receiver in a commData is nonmigratable.
static int isMigratable(LDObjData **objData, int *len, int count, const LDCommData &commData)
{
#if CMK_LBDB_ON
  for (int pe=0 ; pe<count; pe++)
  {
    for (int i=0; i<len[pe]; i++)
      if (LDObjIDEqual(objData[pe][i].objID(), commData.sender.objID()) ||
          LDObjIDEqual(objData[pe][i].objID(), commData.receiver.get_destObj().objID())) 
      return 0;
  }
#endif
  return 1;
}

// rebuild LDStats and remove all non-migratble objects and related things
void CentralLB::removeNonMigratable(LDStats* stats, int count)
{
  int i;

  CkVec<LDObjData> nonmig;
  CkVec<int> new_from_proc, new_to_proc;
  nonmig.resize(stats->n_migrateobjs);
  new_from_proc.resize(stats->n_migrateobjs);
  new_to_proc.resize(stats->n_migrateobjs);
  int n_objs = 0;
  for (i=0; i<stats->n_objs; i++) 
  {
    LDObjData &odata = stats->objData[i];
    if (odata.migratable) {
      nonmig[n_objs] = odata;
      new_from_proc[n_objs] = stats->from_proc[i];
      new_to_proc[n_objs] = stats->to_proc[i];
      n_objs ++;
    }
    else {
      stats->procs[stats->from_proc[i]].bg_walltime += odata.wallTime;
      stats->procs[stats->from_proc[i]].bg_cputime += odata.cpuTime;
    }
  }
  CmiAssert(stats->n_migrateobjs == n_objs);

  stats->makeCommHash();
  
  CkVec<LDCommData> newCommData;
  newCommData.resize(stats->n_comm);
  int n_comm = 0;
  for (i=0; i<stats->n_comm; i++) 
  {
    LDCommData& cdata = stats->commData[i];
    if (!cdata.from_proc()) 
    {
      int idx = stats->getSendHash(cdata);
      CmiAssert(idx != -1);
      if (!stats->objData[idx].migratable) continue;
    }
    switch (cdata.receiver.get_type()) {
    case LD_PROC_MSG:
      break;
    case LD_OBJ_MSG:  {
      int idx = stats->getRecvHash(cdata);
      CmiAssert(idx != -1);
      if (!stats->objData[idx].migratable) continue;
      break;
      }
    case LD_OBJLIST_MSG:    // object message FIXME add multicast
      break;
    }
    newCommData[n_comm] = cdata;
    n_comm ++;
  }

  if (n_objs != stats->n_objs) CmiPrintf("Removed %d nonmigratable %d comms - n_objs:%d migratable:%d\n", stats->n_objs-n_objs, stats->n_objs, stats->n_migrateobjs, stats->n_comm-n_comm);

  // swap to new data
  stats->objData = nonmig;
  stats->from_proc = new_from_proc;
  stats->to_proc = new_to_proc;
  stats->n_objs = n_objs;

  stats->commData = newCommData;
  stats->n_comm = n_comm;

  stats->deleteCommHash();
  stats->makeCommHash();

}


void CentralLB::ReceiveMigration(LBMigrateMsg *m)
{
#if CMK_LBDB_ON
  int i;
  for (i=0; i<CkNumPes(); i++) theLbdb->lastLBInfo.expectedLoad[i] = m->expectedLoad[i];
  
  migrates_expected = 0;
  future_migrates_expected = 0;
  for(i=0; i < m->n_moves; i++) {
    MigrateInfo& move = m->moves[i];
    const int me = CkMyPe();
    if (move.from_pe == me && move.to_pe != me) {
      DEBUGF(("[%d] migrating object to %d\n",move.from_pe,move.to_pe));
      // migrate object, in case it is already gone, inform toPe
      if (theLbdb->Migrate(move.obj,move.to_pe) == 0) 
         thisProxy[move.to_pe].MissMigrate(!move.async_arrival);
    } else if (move.from_pe != me && move.to_pe == me) {
      // CkPrintf("[%d] expecting object from %d\n",move.to_pe,move.from_pe);
      if (!move.async_arrival) migrates_expected++;
      else future_migrates_expected++;
    }
  }
  DEBUGF(("[%d] in ReceiveMigration %d moves expected: %d future expected: %d\n",CkMyPe(),m->n_moves, migrates_expected, future_migrates_expected));
  // if (_lb_debug) CkPrintf("[%d] expecting %d objects migrating.\n", CkMyPe(), migrates_expected);
#if 0
  if (m->n_moves ==0) {
    theLbdb->SetLBPeriod(theLbdb->GetLBPeriod()*2);
  }
#endif

  cur_ld_balancer = m->next_lb;
  if((CkMyPe() == cur_ld_balancer) && (cur_ld_balancer != 0)){
      LBDatabaseObj()->set_avail_vector(m->avail_vector, -2);
  }

  if (migrates_expected == 0 || migrates_completed == migrates_expected)
    MigrationDone(1);
  delete m;
#endif
}


void CentralLB::MigrationDone(int balancing)
{
#if CMK_LBDB_ON
  migrates_completed = 0;
  migrates_expected = -1;
  // clear load stats
  if (balancing) theLbdb->ClearLoads();
  // Increment to next step
  theLbdb->incStep();

  LoadbalanceDone(balancing);        // callback

  // if sync resume invoke a barrier
  if (balancing && _lb_args.syncResume()) {
    CkCallback cb(CkIndex_CentralLB::ResumeClients((CkReductionMsg*)NULL), 
                  thisProxy);
    contribute(0, NULL, CkReduction::sum_int, cb);
  }
  else 
    thisProxy [CkMyPe()].ResumeClients(balancing);
#endif
}

void CentralLB::ResumeClients(CkReductionMsg *msg)
{
  ResumeClients(1);
  delete msg;
}

void CentralLB::ResumeClients(int balancing)
{

#if CMK_LBDB_ON
  DEBUGF(("[%d] Resuming clients. balancing:%d.\n",CkMyPe(),balancing));
  if (balancing && _lb_args.debug() && CkMyPe() == cur_ld_balancer) {
    double end_lb_time = CkWallTimer();
    CkPrintf("[%s] Load balancing step %d finished at %f (duration %fs)\n",
  	      lbName(), step()-1,end_lb_time, end_lb_time - start_lb_time);
  }

  ComlibNotifyMigrationDone();  

  theLbdb->ResumeClients();
  if (balancing)  {
    CheckMigrationComplete();
    if (future_migrates_expected == 0 || 
            future_migrates_expected == future_migrates_completed) {
      CheckMigrationComplete();
    }
  }
#endif
}

/*
  migration of objects contains two different kinds:
  (1) objects want to make a barrier for migration completion
      (waitForBarrier is true)
      migrationDone() to finish and resumeClients
  (2) objects don't need a barrier
  However, next load balancing can only happen when both migrations complete
*/ 
void CentralLB::CheckMigrationComplete()
{
#if CMK_LBDB_ON
  lbdone ++;
  if (lbdone == 2) {
    lbdone = 0;
    future_migrates_expected = -1;
    future_migrates_completed = 0;
    DEBUGF(("[%d] MigrationComplete\n", CkMyPe()));
    // release local barrier  so that the next load balancer can go
    LDOMHandle h;
    h.id.id.idx = 0;
    theLbdb->getLBDB()->DoneRegisteringObjects(h);
    // switch to the next load balancer in the list
    // subtle: called from Migrated() may result in Migrated() called in next LB
    theLbdb->nextLoadbalancer(seqno);
  }
#endif
}

void CentralLB::preprocess(LDStats* stats,int count)
{
  for (int pe=0; pe<count; pe++)
  {
    struct ProcStats &procStat = statsData->procs[pe];
    if (_lb_args.ignoreBgLoad()) {
      procStat.idletime = 0.0;
      procStat.bg_walltime = 0.0;
      procStat.bg_cputime = 0.0;
    }
  }

  // Call the predictor for the future
  if (_lb_predict) FuturePredictor(statsData);
}

// default load balancing strategy
LBMigrateMsg* CentralLB::Strategy(LDStats* stats,int count)
{
#if CMK_LBDB_ON
  work(stats, count);

  if (_lb_args.debug()>1)  {
    CkPrintf("Obj Map:\n");
    for (int i=0; i<stats->n_objs; i++) CkPrintf("%d ", stats->to_proc[i]);
    CkPrintf("\n");
  }

  return createMigrateMsg(stats, count);
#else
  return NULL;
#endif
}

void CentralLB::work(LDStats* stats,int count)
{
  // does nothing but print the database
  stats->print();
}

// generate migrate message from stats->from_proc and to_proc
LBMigrateMsg * CentralLB::createMigrateMsg(LDStats* stats,int count)
{
  int i;
  CkVec<MigrateInfo*> migrateInfo;
  for (i=0; i<stats->n_objs; i++) {
    LDObjData &objData = stats->objData[i];
    int frompe = stats->from_proc[i];
    int tope = stats->to_proc[i];
    if (frompe != tope) {
      //      CkPrintf("[%d] Obj %d migrating from %d to %d\n",
      //         CkMyPe(),obj,pe,dest);
      MigrateInfo *migrateMe = new MigrateInfo;
      migrateMe->obj = objData.handle;
      migrateMe->from_pe = frompe;
      migrateMe->to_pe = tope;
      migrateMe->async_arrival = objData.asyncArrival;
      migrateInfo.insertAtEnd(migrateMe);
    }
  }

  int migrate_count=migrateInfo.length();
  LBMigrateMsg* msg = new(migrate_count,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  if (_lb_args.debug())
    CkPrintf("%s: %d objects migrating.\n", lbname, migrate_count);
  return msg;
}

void CentralLB::simulationWrite() {
  if(step() == LBSimulation::dumpStep)
  {
    // here we are supposed to dump the database
    int dumpFileSize = strlen(LBSimulation::dumpFile) + 4;
    char *dumpFileName = (char *)malloc(dumpFileSize);
    while (sprintf(dumpFileName, "%s.%d", LBSimulation::dumpFile, LBSimulation::dumpStep) >= dumpFileSize) {
      free(dumpFileName);
      dumpFileSize+=3;
      dumpFileName = (char *)malloc(dumpFileSize);
    }
    writeStatsMsgs(dumpFileName);
    free(dumpFileName);
    CmiPrintf("LBDump: Dumped the load balancing data at step %d.\n",LBSimulation::dumpStep);
    ++LBSimulation::dumpStep;
    --LBSimulation::dumpStepSize;
    if (LBSimulation::dumpStepSize <= 0) { // prevent stupid step sizes
      CmiPrintf("Charm++> Exiting...\n");
      CkExit();
    }
    return;
  }
}

void CentralLB::simulationRead() {
  LBSimulation *simResults = NULL, *realResults;
  LBMigrateMsg *voidMessage = new (0,0,0,0) LBMigrateMsg();
  voidMessage->n_moves=0;
  for ( ;LBSimulation::simStepSize > 0; --LBSimulation::simStepSize, ++LBSimulation::simStep) {
    // here we are supposed to read the data from the dump database
    int simFileSize = strlen(LBSimulation::dumpFile) + 4;
    char *simFileName = (char *)malloc(simFileSize);
    while (sprintf(simFileName, "%s.%d", LBSimulation::dumpFile, LBSimulation::simStep) >= simFileSize) {
      free(simFileName);
      simFileSize+=3;
      simFileName = (char *)malloc(simFileSize);
    }
    readStatsMsgs(simFileName);
    free(simFileName);

    // allocate simResults (only the first step)
    if (simResults == NULL) {
      simResults = new LBSimulation(LBSimulation::simProcs);
      realResults = new LBSimulation(LBSimulation::simProcs);
    }
    else {
      // should be the same number of procs of the original simulation!
      if (!LBSimulation::procsChanged) {
	// it means we have a previous step, so in simResults there is data.
	// we can now print the real effects of the load balancer during the simulation
	// or print the difference between the predicted data and the real one.
	realResults->reset();
	// reset to_proc of statsData to be equal to from_proc
	for (int k=0; k < statsData->n_objs; ++k) statsData->to_proc[k] = statsData->from_proc[k];
	findSimResults(statsData, LBSimulation::simProcs, voidMessage, realResults);
	simResults->PrintDifferences(realResults,statsData);
      }
      simResults->reset();
    }

    // now pass it to the strategy routine
    double startT = CkWallTimer();
    preprocess(statsData, LBSimulation::simProcs);
    CmiPrintf("%s> Strategy starts ... \n", lbname);
    LBMigrateMsg* migrateMsg = Strategy(statsData, LBSimulation::simProcs);
    CmiPrintf("%s> Strategy took %fs memory usage: CentralLB:%dKB. \n", 
               lbname, CkWallTimer()-startT, (int)(useMem()/1000));

    // now calculate the results of the load balancing simulation
    findSimResults(statsData, LBSimulation::simProcs, migrateMsg, simResults);

    // now we have the simulation data, so print it and loop
    CmiPrintf("Charm++> LBSim: Simulation of load balancing step %d done.\n",LBSimulation::simStep);
    simResults->PrintSimulationResults();

    delete migrateMsg;
    CmiPrintf("Charm++> LBSim: Passing to the next step\n");
  }
  // deallocate simResults
  delete simResults;
  CmiPrintf("Charm++> Exiting...\n");
  CkExit();
}

void CentralLB::readStatsMsgs(const char* filename) 
{
#if CMK_LBDB_ON
  int i;
  FILE *f = fopen(filename, "r");
  if (f==NULL) {
    CmiPrintf("Fatal Error> Cannot open LB Dump file %s!\n", filename);
    CmiAbort("");
  }

  // at this stage, we need to rebuild the statsMsgList and
  // statsDataList structures. For that first deallocate the
  // old structures
  if (statsMsgsList) {
    for(i = 0; i < stats_msg_count; i++)
      delete statsMsgsList[i];
    delete[] statsMsgsList;
    statsMsgsList=0;
  }

  PUP::fromDisk pd(f);
  PUP::machineInfo machInfo;

  pd((char *)&machInfo, sizeof(machInfo));	// read machine info
  PUP::xlater p(machInfo, pd);

  if (_lb_args.lbversion() >= 1) {
    p|_lb_args.lbversion();		// write version number
    CkPrintf("LB> File version detected: %d\n", _lb_args.lbversion());
    CmiAssert(_lb_args.lbversion() <= LB_FORMAT_VERSION);
  }
  p|stats_msg_count;

  CmiPrintf("readStatsMsgs for %d pes starts ... \n", stats_msg_count);
  if (LBSimulation::simProcs == 0) LBSimulation::simProcs = stats_msg_count;
  if (LBSimulation::simProcs != stats_msg_count) LBSimulation::procsChanged = true;

  // LBSimulation::simProcs must be set
  statsData->pup(p);

  CmiPrintf("Simulation for %d pes \n", LBSimulation::simProcs);
  CmiPrintf("n_obj: %d n_migratble: %d \n", statsData->n_objs, statsData->n_migrateobjs);

  // file f is closed in the destructor of PUP::fromDisk
  CmiPrintf("ReadStatsMsg from %s completed\n", filename);
#endif
}

void CentralLB::writeStatsMsgs(const char* filename) 
{
#if CMK_LBDB_ON
  FILE *f = fopen(filename, "w");
  if (f==NULL) {
    CmiPrintf("Fatal Error> writeStatsMsgs failed to open the output file %s!\n", filename);
    CmiAbort("");
  }

  const PUP::machineInfo &machInfo = PUP::machineInfo::current();
  PUP::toDisk p(f);
  p((char *)&machInfo, sizeof(machInfo));	// machine info

  p|_lb_args.lbversion();		// write version number
  p|stats_msg_count;
  statsData->pup(p);

  fclose(f);

  CmiPrintf("WriteStatsMsgs to %s succeed!\n", filename);
#endif
}

// calculate the predicted wallclock/cpu load for every processors
// considering communication overhead if considerComm is true
void getPredictedLoadWithMsg(BaseLB::LDStats* stats, int count, 
                      LBMigrateMsg *msg, LBInfo &info, 
		      int considerComm)
{
#if CMK_LBDB_ON
	stats->makeCommHash();

 	// update to_proc according to migration msgs
	for(int i = 0; i < msg->n_moves; i++) {
	  MigrateInfo &mInfo = msg->moves[i];
	  int idx = stats->getHash(mInfo.obj.objID(), mInfo.obj.omID());
	  CmiAssert(idx != -1);
          stats->to_proc[idx] = mInfo.to_pe;
	}

	getLoadInfo(stats, count, info, considerComm);
}


void getLoadInfo(BaseLB::LDStats* stats, int count, 
		             LBInfo &info, int considerComm)
{
	int i, pe;
	double *peLoads = info.peLoads;
	double *objLoads = info.objLoads;
	double *comLoads = info.comLoads;
	double *bgLoads = info.bgLoads;
        double minObjLoad = 1.0e20;  // I suppose no object load is beyond this
	double maxObjLoad = 0.0;
	CmiAssert(peLoads);

	double alpha = _lb_args.alpha();
	double beeta = _lb_args.beeta();

	stats->makeCommHash();

	info.clear();

        // get background load
	if (bgLoads)
    	  for(pe = 0; pe < count; pe++)
    	   bgLoads[pe] = stats->procs[pe].bg_walltime;

	for(pe = 0; pe < count; pe++)
    	  peLoads[pe] = stats->procs[pe].bg_walltime;

    	for(int obj = 0; obj < stats->n_objs; obj++)
    	{
		int pe = stats->to_proc[obj];
		double &oload = stats->objData[obj].wallTime;
		if (oload < minObjLoad) minObjLoad = oload;
		if (oload > maxObjLoad) maxObjLoad = oload;
		peLoads[pe] += oload;
		if (objLoads) objLoads[pe] += oload;
	}

	// handling of the communication overheads. 
	if (considerComm) {
	  int* msgSentCount = new int[count]; // # of messages sent by each PE
	  int* msgRecvCount = new int[count]; // # of messages received by each PE
	  int* byteSentCount = new int[count];// # of bytes sent by each PE
	  int* byteRecvCount = new int[count];// # of bytes reeived by each PE
	  for(i = 0; i < count; i++)
	    msgSentCount[i] = msgRecvCount[i] = byteSentCount[i] = byteRecvCount[i] = 0;

	  int mcast_count = 0;
          for (int cidx=0; cidx < stats->n_comm; cidx++) {
	    LDCommData& cdata = stats->commData[cidx];
	    int senderPE, receiverPE;
	    if (cdata.from_proc())
	      senderPE = cdata.src_proc;
  	    else {
	      int idx = stats->getHash(cdata.sender);
	      if (idx == -1) continue;    // sender has just migrated?
	      senderPE = stats->to_proc[idx];
	      CmiAssert(senderPE != -1);
	    }
	    CmiAssert(senderPE < count && senderPE >= 0);

            // find receiver: point-to-point and multicast two cases
	    int receiver_type = cdata.receiver.get_type();
	    if (receiver_type == LD_PROC_MSG || receiver_type == LD_OBJ_MSG) {
              if (receiver_type == LD_PROC_MSG)
	        receiverPE = cdata.receiver.proc();
              else  {  // LD_OBJ_MSG
	        int idx = stats->getHash(cdata.receiver.get_destObj());
	        if (idx == -1) continue;    // receiver has just been removed?
	        receiverPE = stats->to_proc[idx];
	        CmiAssert(receiverPE != -1);
              }
              CmiAssert(receiverPE < count && receiverPE >= 0);
	      if(senderPE != receiverPE)
	      {
	  	msgSentCount[senderPE] += cdata.messages;
		byteSentCount[senderPE] += cdata.bytes;
		msgRecvCount[receiverPE] += cdata.messages;
		byteRecvCount[receiverPE] += cdata.bytes;
	      }
	    }
            else if (receiver_type == LD_OBJLIST_MSG) {
              int nobjs;
              LDObjKey *objs = cdata.receiver.get_destObjs(nobjs);
	      mcast_count ++;
	      for (i=0; i<nobjs; i++) {
	        int idx = stats->getHash(objs[i]);
		CmiAssert(idx != -1);
	        if (idx == -1) continue;    // receiver has just been removed?
	        receiverPE = stats->to_proc[idx];
		CmiAssert(receiverPE < count && receiverPE >= 0);
	        if(senderPE != receiverPE)
	        {
	  	msgSentCount[senderPE] += cdata.messages;
		byteSentCount[senderPE] += cdata.bytes;
		msgRecvCount[receiverPE] += cdata.messages;
		byteRecvCount[receiverPE] += cdata.bytes;
	        }
              }
	    }
	  }   // end of for
          if (_lb_args.debug())
             CkPrintf("Number of MULTICAST: %d\n", mcast_count);

	  // now for each processor, add to its load the send and receive overheads
	  for(i = 0; i < count; i++)
	  {
		double comload = msgRecvCount[i]  * PER_MESSAGE_RECV_OVERHEAD +
			      msgSentCount[i]  * alpha +
			      byteRecvCount[i] * PER_BYTE_RECV_OVERHEAD +
			      byteSentCount[i] * beeta;
		peLoads[i] += comload;
		if (comLoads) comLoads[i] += comload;
	  }
	  delete [] msgRecvCount;
	  delete [] msgSentCount;
	  delete [] byteRecvCount;
	  delete [] byteSentCount;
	}

 	info.minObjLoad = minObjLoad;
 	info.maxObjLoad = maxObjLoad;
#endif
}

void CentralLB::findSimResults(LDStats* stats, int count, LBMigrateMsg* msg, LBSimulation* simResults)
{
    CkAssert(simResults != NULL && count == simResults->numPes);
    // estimate the new loads of the processors. As a first approximation, this is the
    // sum of the cpu times of the objects on that processor
    double startT = CkWallTimer();
    getPredictedLoadWithMsg(stats, count, msg, simResults->lbinfo, 1);
    CmiPrintf("getPredictedLoad finished in %fs\n", CkWallTimer()-startT);
}

void CentralLB::pup(PUP::er &p) { 
  BaseLB::pup(p); 
  if (p.isUnpacking())  {
    initLB(CkLBOptions(seqno)); 
  }
}

int CentralLB::useMem() { 
  return sizeof(CentralLB) + statsData->useMem() + 
         CkNumPes() * sizeof(CLBStatsMsg *);
}


/**
  CLBStatsMsg is not a real message now.
  CLBStatsMsg is used for all processors to fill in their local load and comm
  statistics and send to processor 0
*/

CLBStatsMsg::CLBStatsMsg(int osz, int csz) {
  objData = new LDObjData[osz];
  commData = new LDCommData[csz];
  avail_vector = NULL;
}

CLBStatsMsg::~CLBStatsMsg() {
  delete [] objData;
  delete [] commData;
  if (avail_vector) delete [] avail_vector;
}

void CLBStatsMsg::pup(PUP::er &p) {
  int i;
  p|from_pe;
  p|pe_speed;
  p|total_walltime; p|total_cputime;
  p|idletime;
  p|bg_walltime;   p|bg_cputime;
  p|n_objs;
  if (p.isUnpacking()) objData = new LDObjData[n_objs];
  for (i=0; i<n_objs; i++) p|objData[i];
  p|n_comm;
  if (p.isUnpacking()) commData = new LDCommData[n_comm];
  for (i=0; i<n_comm; i++) p|commData[i];

  int has_avail_vector;
  if (!p.isUnpacking()) has_avail_vector = (avail_vector != NULL);
  p|has_avail_vector;
  if (p.isUnpacking()) {
    if (has_avail_vector) avail_vector = new char[CkNumPes()];
    else avail_vector = NULL;
  }
  if (has_avail_vector) p(avail_vector, CkNumPes());

  p(next_lb);
}

// CkMarshalledCLBStatsMessage is used in the marshalled parameter in
// the entry function, it is just used to use to pup.
// I don't use CLBStatsMsg directly as marshalled parameter because
// I want the data pointer stored and not to be freed by the Charm++.
CkMarshalledCLBStatsMessage::~CkMarshalledCLBStatsMessage() {
  if (msg) delete msg;
}

void CkMarshalledCLBStatsMessage::pup(PUP::er &p)
{
  if (p.isUnpacking()) msg = new CLBStatsMsg;
  else CmiAssert(msg);
  msg->pup(p);
}

#include "CentralLB.def.h"

/*@}*/
