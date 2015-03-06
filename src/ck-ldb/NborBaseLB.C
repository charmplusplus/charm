/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "BaseLB.h"
#include "NborBaseLB.h"
#include "LBDBManager.h"
#include "NborBaseLB.def.h"

#define  DEBUGF(x)      // CmiPrintf x;

//CreateLBFunc_Def(NborBaseLB);

void NborBaseLB::staticMigrated(void* data, LDObjHandle h, int waitBarrier)
{
  NborBaseLB *me = (NborBaseLB*)(data);

  me->Migrated(h, waitBarrier);
}

void NborBaseLB::staticAtSync(void* data)
{
  NborBaseLB *me = (NborBaseLB*)(data);

  me->AtSync();
}

NborBaseLB::NborBaseLB(const CkLBOptions &opt): CBase_NborBaseLB(opt)
{
#if CMK_LBDB_ON
  lbname = (char *)"NborBaseLB";
  thisProxy = CProxy_NborBaseLB(thisgroup);
  receiver = theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
			    (void*)(this));
  notifier = theLbdb->getLBDB()->
    NotifyMigrated((LDMigratedFn)(staticMigrated), (void*)(this));


  // I had to move neighbor initialization outside the constructor
  // in order to get the virtual functions of any derived classes
  // so I'll just set them to illegal values here.
  LBtopoFn topofn = LBTopoLookup(_lbtopo);
  if (topofn == NULL) {
    if (CkMyPe()==0) CmiPrintf("LB> Fatal error: Unknown topology: %s.\n", _lbtopo);
    CmiAbort("");
  }
  topo = topofn(CkNumPes());

  mig_msgs_expected = 0;
  neighbor_pes = NULL;
  stats_msg_count = 0;
  statsMsgsList = NULL;
  statsDataList = NULL;
  migrates_completed = 0;
  migrates_expected = -1;
  mig_msgs_received = 0;
  mig_msgs = NULL;

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

  if (_lb_args.statsOn()) theLbdb->CollectStatsOn();
#endif
}

NborBaseLB::~NborBaseLB()
{
#if CMK_LBDB_ON
  theLbdb = CProxy_LBDatabase(_lbdb).ckLocalBranch();
  if (theLbdb) {
    theLbdb->getLBDB()->
      RemoveNotifyMigrated(notifier);
    //theLbdb->
    //  RemoveStartLBFn((LDStartLBFn)(staticStartLB));
  }
  if (statsMsgsList) delete [] statsMsgsList;
  if (statsDataList) delete [] statsDataList;
  if (neighbor_pes)  delete [] neighbor_pes;
  if (mig_msgs)      delete [] mig_msgs;
#endif
}

void NborBaseLB::FindNeighbors()
{
  if (neighbor_pes == 0) { // Neighbors never initialized, so init them
                           // and other things that depend on the number
                           // of neighbors
    int maxneighbors = topo->max_neighbors();
    statsMsgsList = new NLBStatsMsg*[maxneighbors];
    for(int i=0; i < maxneighbors; i++)
      statsMsgsList[i] = 0;
    statsDataList = new LDStats[maxneighbors];

    neighbor_pes = new int[maxneighbors];
    topo->neighbors(CkMyPe(), neighbor_pes, mig_msgs_expected);
    mig_msgs = new LBMigrateMsg*[mig_msgs_expected];
  }

}

void NborBaseLB::AtSync()
{
#if CMK_LBDB_ON
  //  CkPrintf("[%d] NborBaseLB At Sync step %d!!!!\n",CkMyPe(),mystep);

  if (neighbor_pes == 0) FindNeighbors();
  start_lb_time = 0;

  if (!QueryBalanceNow(step()) || mig_msgs_expected == 0) {
    MigrationDone(0);
    return;
  }

  if (CkMyPe() == 0) {
    start_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("[%s] Load balancing step %d starting at %f\n",
	       lbName(), step(),start_lb_time);
  }

  NLBStatsMsg* msg = AssembleStats();

  if (mig_msgs_expected > 0) {
    CkMarshalledNLBStatsMessage marshmsg(msg);
    thisProxy.ReceiveStats(marshmsg, mig_msgs_expected, neighbor_pes);
  }

  // Tell our own node that we are ready
  CkMarshalledNLBStatsMessage mmsg(NULL);
  thisProxy[CkMyPe()].ReceiveStats(mmsg);
#endif
}

NLBStatsMsg* NborBaseLB::AssembleStats()
{
#if CMK_LBDB_ON
  // Get my own stats, this is used in NeighborLB for example
#if CMK_LB_CPUTIMER
  theLbdb->TotalTime(&myStats.total_walltime,&myStats.total_cputime);
  theLbdb->BackgroundLoad(&myStats.bg_walltime,&myStats.bg_cputime);
#else
  theLbdb->TotalTime(&myStats.total_walltime,&myStats.total_walltime);
  theLbdb->BackgroundLoad(&myStats.bg_walltime,&myStats.bg_walltime);
#endif
  theLbdb->IdleTime(&myStats.idletime);

  myStats.move = QueryMigrateStep(step());

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

  const int osz = theLbdb->GetObjDataSz();
  const int csz = theLbdb->GetCommDataSz();
  NLBStatsMsg* msg = new NLBStatsMsg(osz, csz);

  msg->from_pe = CkMyPe();
  // msg->serial = rand();
  msg->serial = CrnRand();
  msg->pe_speed = myStats.pe_speed;
  msg->total_walltime = myStats.total_walltime;
  msg->idletime = myStats.idletime;
  msg->bg_walltime = myStats.bg_walltime;
  msg->obj_walltime = myStats.obj_walltime;
#if CMK_LB_CPUTIMER
  msg->total_cputime = myStats.total_cputime;
  msg->bg_cputime = myStats.bg_cputime;
  msg->obj_cputime = myStats.obj_cputime;
#endif
  msg->n_objs = osz;
  theLbdb->GetObjData(msg->objData);
  msg->n_comm = csz;
  theLbdb->GetCommData(msg->commData);

  // cleanup 
  delete [] myStats.objData;
  myStats.objData = NULL;
  myStats.n_objs = 0;
  delete [] myStats.commData;
  myStats.commData = NULL;
  myStats.n_comm = 0;

  //  CkPrintf(
  //    "Proc %d speed=%d Total(wall,cpu)=%f %f Idle=%f Bg=%f %f Obj=%f %f\n",
  //    CkMyPe(),msg->proc_speed,msg->total_walltime,msg->total_cputime,
  //    msg->idletime,msg->bg_walltime,msg->bg_cputime,
  //    msg->obj_walltime,msg->obj_cputime);

  //  CkPrintf("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
  //	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm);
  return msg;
#else
  return NULL;
#endif
}

void NborBaseLB::Migrated(LDObjHandle h, int waitBarrier)
{
  migrates_completed++;
  //  CkPrintf("[%d] An object migrated! %d %d\n",
  //  	   CkMyPe(),migrates_completed,migrates_expected);
  if (migrates_completed == migrates_expected) {
    MigrationDone(1);
  }
}

void NborBaseLB::ReceiveStats(CkMarshalledNLBStatsMessage &data)
{
#if CMK_LBDB_ON
  NLBStatsMsg *m = data.getMessage();
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
      statsDataList[peslot].idletime = m->idletime;
      statsDataList[peslot].bg_walltime = m->bg_walltime;
      statsDataList[peslot].pe_speed = m->pe_speed;
      statsDataList[peslot].obj_walltime = m->obj_walltime;
#if CMK_LB_CPUTIMER
      statsDataList[peslot].total_cputime = m->total_cputime;
      statsDataList[peslot].bg_cputime = m->bg_cputime;
      statsDataList[peslot].obj_cputime = m->obj_cputime;
#endif

      statsDataList[peslot].n_objs = m->n_objs;
      statsDataList[peslot].objData = m->objData;
      statsDataList[peslot].n_comm = m->n_comm;
      statsDataList[peslot].commData = m->commData;

      if (_lb_args.ignoreBgLoad()) statsDataList[peslot].clearBgLoad();

      stats_msg_count++;
    }
  }

  const int clients = mig_msgs_expected;
  if (stats_msg_count == clients && receive_stats_ready) {
    double strat_start_time = CkWallTimer();
    receive_stats_ready = 0;
    LBMigrateMsg* migrateMsg = Strategy(statsDataList,clients);

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
    if (clients > 0)
      thisProxy.ReceiveMigration(migrateMsg, clients, neighbor_pes);
    
    // Zero out data structures for next cycle
    for(i=0; i < clients; i++) {
      delete statsMsgsList[i];
      statsMsgsList[i]=NULL;
    }
    stats_msg_count=0;

    //theLbdb->ClearLoads();
    if (CkMyPe() == 0) {
      double strat_end_time = CkWallTimer();
      if (_lb_args.debug())
        CkPrintf("[%d] %s Strat elapsed time %f\n",CkMyPe(),lbName(),strat_end_time-strat_start_time);
    }
  }
#endif  
}

void NborBaseLB::ReceiveMigration(LBMigrateMsg *msg)
{
#if CMK_LBDB_ON
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
    LBMigrateMsg* m = mig_msgs[neigh];
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
    MigrationDone(1);
#endif
}


void NborBaseLB::MigrationDone(int balancing)
{
#if CMK_LBDB_ON
  migrates_completed = 0;
  migrates_expected = -1;
  // Increment to next step
  theLbdb->incStep();

  theLbdb->ClearLoads();

  // if sync resume invoke a barrier
  if (balancing && _lb_args.syncResume()) {
    CkCallback cb(CkIndex_NborBaseLB::ResumeClients((CkReductionMsg*)NULL), 
                  thisProxy);
    contribute(0, NULL, CkReduction::sum_int, cb);
  }
  else 
    thisProxy [CkMyPe()].ResumeClients(balancing);
  //thisProxy [CkMyPe()].ResumeClients();
#endif
}

void NborBaseLB::ResumeClients(CkReductionMsg *msg)
{
  ResumeClients(1);
  delete msg;
}

void NborBaseLB::ResumeClients(int balancing)
{
#if CMK_LBDB_ON
  DEBUGF(("[%d] ResumeClients. \n", CkMyPe()));

  if (CkMyPe() == 0 && balancing) {
    double end_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("[%s] Load balancing step %d finished at %f duration %f\n",
	        lbName(), step()-1,end_lb_time,end_lb_time - start_lb_time);
  }

  theLbdb->ResumeClients();
#endif
}

LBMigrateMsg* NborBaseLB::Strategy(LDStats* stats, int n_nbrs)
{
  for(int j=0; j < n_nbrs; j++) {
    CkPrintf("[%d] Proc %d Speed %d WALL: Total %f Idle %f Bg %f obj %f",
    CkMyPe(), stats[j].from_pe, stats[j].pe_speed, stats[j].total_walltime,
    stats[j].idletime, stats[j].bg_walltime, stats[j].obj_walltime);
#if CMK_LB_CPUTIMER
    CkPrintf(" CPU: Total %f Bg %f obj %f", stats[j].total_cputime,
    stats[j].bg_cputime, stats[j].obj_cputime);
#endif
    CkPrintf("\n");
  }

  int sizes=0;
  LBMigrateMsg* msg = new(sizes,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
  msg->n_moves = 0;

  return msg;
}

int NborBaseLB::NeighborIndex(int pe)
{
    int peslot = -1;
    for(int i=0; i < mig_msgs_expected; i++) {
      if (pe == neighbor_pes[i]) {
	peslot = i;
	break;
      }
    }
    return peslot;
}

NLBStatsMsg::NLBStatsMsg(int osz, int csz) {
  objData = new LDObjData[osz];
  commData = new LDCommData[csz];
}

NLBStatsMsg::NLBStatsMsg(NLBStatsMsg *src) 
{
  int size;
  {
    PUP::sizer  p;
    src->pup(p);
    size = p.size();
  }
  char *buf = new char[size];
  {
    PUP::toMem  p(buf);
    src->pup(p);
  }
  {
    PUP::fromMem  p(buf);
    pup(p);
  }
  delete [] buf;
}

NLBStatsMsg::~NLBStatsMsg() {
  delete [] objData;
  delete [] commData;
}

void NLBStatsMsg::pup(PUP::er &p) {
  int i;
  p|from_pe;
  p|serial;
  p|pe_speed;
  p|total_walltime;
  p|idletime;
  p|bg_walltime;
#if CMK_LB_CPUTIMER
  p|total_cputime;
  p|bg_cputime;
#endif
  p|n_objs;
  if (p.isUnpacking()) objData = new LDObjData[n_objs];
  for (i=0; i<n_objs; i++) p|objData[i];
  p|n_comm;
  if (p.isUnpacking()) commData = new LDCommData[n_comm];
  for (i=0; i<n_comm; i++) p|commData[i];
}

// CkMarshalledCLBStatsMessage is used in the marshalled parameter in
// the entry function, it is just used to use to pup.
// I don't use CLBStatsMsg directly as marshalled parameter because
// I want the data pointer stored and not to be freed by the Charm++.
CkMarshalledNLBStatsMessage::~CkMarshalledNLBStatsMessage() {
  if (msg) delete msg;
}

void CkMarshalledNLBStatsMessage::pup(PUP::er &p)
{
  int isnull;
  if (p.isPacking()) isnull = (msg==NULL?1:0);
  p|isnull;
  if (p.isUnpacking()) {
    if (!isnull) msg = new NLBStatsMsg;
    else msg = NULL;
  }
  if (msg) msg->pup(p);
#if 0
  if (p.isUnpacking()) msg = new NLBStatsMsg;
  else CmiAssert(msg);
  msg->pup(p);
#endif
}



/*@{*/
