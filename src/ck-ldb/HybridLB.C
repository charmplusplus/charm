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

#include "charm++.h"
#include "BaseLB.h"
#include "HybridLB.h"
#include "LBDBManager.h"
#include "HybridLB.def.h"

#define  DEBUGF(x)      // CmiPrintf x;

CreateLBFunc_Def(HybridLB, "Hybrid load balancer");

void HybridLB::staticMigrated(void* data, LDObjHandle h, int waitBarrier)
{
  HybridLB *me = (HybridLB*)(data);

  me->Migrated(h, waitBarrier);
}

void HybridLB::staticAtSync(void* data)
{
  HybridLB *me = (HybridLB*)(data);

  me->AtSync();
}

HybridLB::HybridLB(const CkLBOptions &opt): BaseLB(opt)
{
#if CMK_LBDB_ON
  lbname = (char *)"HybridLB";
  mystep = 0;
  thisProxy = CProxy_HybridLB(thisgroup);
  receiver = theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
			    (void*)(this));
  notifier = theLbdb->getLBDB()->
    NotifyMigrated((LDMigratedFn)(staticMigrated), (void*)(this));


  // I had to move neighbor initialization outside the constructor
  // in order to get the virtual functions of any derived classes
  // so I'll just set them to illegal values here.
  LBtopoFn topofn = LBTopoLookup("4_arytree");
  if (topofn == NULL) {
    if (CkMyPe()==0) CmiPrintf("LB> Fatal error: Unknown topology: %s.\n", _lbtopo);
    CmiAbort("");
  }
  topo = topofn(CkNumPes());

  parent = -1;
  foundNeighbors = 0;
  statsMsgsList = NULL;
  stats_msg_count = 0;
  loadbalancing = 0;
  statsData = new LDStats;

  theLbdb->CollectStatsOn();

  
#if 0
  mig_msgs_expected = 0;
  neighbor_pes = NULL;
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

  theLbdb->CollectStatsOn();
#endif
#endif
}

HybridLB::~HybridLB()
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
#if 0
  if (statsDataList) delete [] statsDataList;
  if (neighbor_pes)  delete [] neighbor_pes;
  if (mig_msgs)      delete [] mig_msgs;
#endif
#endif
}

void HybridLB::FindNeighbors()
{
  if (foundNeighbors == 0) { // Neighbors never initialized, so init them
                           // and other things that depend on the number
                           // of neighbors
    int maxneighbors = topo->max_neighbors();
    statsMsgsList = new CLBStatsMsg*[maxneighbors];
    for(int i=0; i < maxneighbors; i++)
      statsMsgsList[i] = 0;
/*
    statsDataList = new LDStats[maxneighbors];
*/

    int *neighbor_pes = new int[maxneighbors];
    topo->neighbors(CkMyPe(), neighbor_pes, mig_msgs_expected);
//    mig_msgs = new LBMigrateMsg*[mig_msgs_expected];
    int idx = 0;
    if (CkMyPe() != 0) {
      parent = neighbor_pes[idx++];
    }
    else
      parent = -1;
    for (; idx<mig_msgs_expected; idx++)
      children.push_back(neighbor_pes[idx]);
    delete [] neighbor_pes;
    foundNeighbors = 1;
  }

}

// root
void HybridLB::AtSync()
{
#if CMK_LBDB_ON
  //  CkPrintf("[%d] HybridLB At Sync step %d!!!!\n",CkMyPe(),mystep);

  FindNeighbors();

  start_lb_time = 0;

  if (!QueryBalanceNow(step()) || mig_msgs_expected == 0) {
    MigrationDone();
    return;
  }

  if (CkMyPe() == 0) {
    start_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("Load balancing step %d starting at %f\n",
	       step(),start_lb_time);
  }

  // assemble LB database
  CLBStatsMsg* msg = AssembleStats();

  CkMarshalledCLBStatsMessage marshmsg(msg);
  if (children.size() == 0) {
    // we are leaves, send to parent if parent exists
    thisProxy[parent].ReceiveStats(marshmsg);
  }
  else {
    // send to myself and wait for everyone else
    thisProxy[CkMyPe()].ReceiveStats(marshmsg);
  }
#endif
}

CLBStatsMsg* HybridLB::AssembleStats()
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
  msg->serial = CrnRand();

  // Get stats
  theLbdb->GetTime(&msg->total_walltime,&msg->total_cputime,
                   &msg->idletime, &msg->bg_walltime,&msg->bg_cputime);
//  msg->pe_speed = myspeed;
  msg->pe_speed = 1;

  msg->n_objs = osz;
  theLbdb->GetObjData(msg->objData);
  msg->n_comm = csz;
  theLbdb->GetCommData(msg->commData);

  return msg;
#else
  return NULL;
#endif
}

void HybridLB::ReceiveStats(CkMarshalledCLBStatsMessage &data)
{
#if CMK_LBDB_ON
  int i;
  FindNeighbors();

  // store the message
  CLBStatsMsg *m = data.getMessage();
  depositLBStatsMessage(m);
  stats_msg_count++;

  if (stats_msg_count == children.size()+1)  // plus myself
  {
    stats_msg_count = 0;
    if (parent != -1) {
      // combine and shrink message
      buildStats();

      // build a new message based on our LDStats
      CLBStatsMsg* cmsg = buildCombinedLBStatsMessage();

      // send to parent
      CkMarshalledCLBStatsMessage marshmsg(cmsg);
      thisProxy[parent].ReceiveStats(marshmsg);
    }
    else {
      // root of all processors, calls top-level strategy
      // call strategy as if this is centralized lb
      thisProxy[CkMyPe()].Loadbalancing();
    }
  }

#if 0
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

    theLbdb->ClearLoads();
    if (CkMyPe() == 0) {
      double strat_end_time = CkWallTimer();
      if (_lb_args.debug())
        CkPrintf("Strat elapsed time %f\n",strat_end_time-strat_start_time);
    }
  }
#endif
#endif  
}

void HybridLB::depositLBStatsMessage(CLBStatsMsg *m)
{
  int pe = m->from_pe;
  if (statsMsgsList[stats_msg_count] != 0) {
    CkPrintf("*** Unexpected NLBStatsMsg in ReceiveStats from PE %d ***\n", pe);
    CkAbort("HybridLB> Abort!");
  }
  statsMsgsList[stats_msg_count] = m;
      // store per processor data right away
  struct ProcStats &procStat = statsData->procs[pe];
  procStat.pe = pe;
  procStat.total_walltime = m->total_walltime;
  procStat.total_cputime = m->total_cputime;
  procStat.idletime = m->idletime;
  procStat.bg_walltime = m->bg_walltime;
  procStat.bg_cputime = m->bg_cputime;
  procStat.pe_speed = m->pe_speed;
  procStat.utilization = 1.0;
  procStat.available = CmiTrue;
  procStat.n_objs = m->n_objs;

  statsData->n_objs += m->n_objs;
  statsData->n_comm += m->n_comm;
}

void HybridLB::buildStats()
{
#if CMK_LBDB_ON
  // combine into one message as if this is one processor, 
  // communication become local communication
  // build LDStats
  // statsMsgsList
  statsData->count = stats_msg_count+1;
  statsData->objData = new LDObjData[statsData->n_objs];
  statsData->from_proc = new int[statsData->n_objs];
  statsData->to_proc = new int[statsData->n_objs];
  statsData->commData = new LDCommData[statsData->n_comm];
  int nobj = statsData->n_objs;
  int nmigobj = 0;
  int ncom = 0;
  for (int n=0; n<stats_msg_count+1; n++) {
     int i;
     CLBStatsMsg *msg = statsMsgsList[n];
     int pe = msg->from_pe;
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
     statsMsgsList[n]=0;
  }
  statsData->n_migrateobjs = nmigobj;
  if (_lb_args.debug()) {
      CmiPrintf("n_obj:%d migratable:%d ncom:%d\n", nobj, nmigobj, ncom);
  }
#endif
}

CLBStatsMsg * HybridLB::buildCombinedLBStatsMessage()
{
#if CMK_LBDB_ON
  int i;
  // build a message based on our LDStats
  const int osz = statsData->n_objs;
  const int csz = statsData->n_comm;

  CLBStatsMsg* cmsg = new CLBStatsMsg(osz, csz);
  cmsg->from_pe = CkMyPe();
  cmsg->serial = CrnRand();

  // Get stats
  // sum of all childrens
  cmsg->pe_speed = 0;
  cmsg->total_walltime = 0;
  cmsg->total_cputime = 0;
  for (int pe=0; pe<statsData->count; pe++) {
        struct ProcStats &procStat = statsData->procs[pe];
        cmsg->pe_speed += procStat.pe_speed;
        cmsg->total_walltime += procStat.total_walltime;
        cmsg->total_cputime += procStat.total_cputime;
  }
  cmsg->idletime = 0.0;
  cmsg->bg_walltime = 0.0;
  cmsg->bg_cputime = 0.0;

  // copy and shrink data
  cmsg->n_objs = osz;
  for (i=0; i<osz; i++)  {
        cmsg->objData[i] = statsData->objData[i];
  }
  int mype = CkMyPe();
  cmsg->n_comm = csz;
  for (i=0; i<csz; i++) {
     LDCommData &commData = statsData->commData[i];
     cmsg->commData[i] = commData;
     if (commData.from_proc()) cmsg->commData[i].src_proc = mype;
     if (commData.receiver.get_type() == LD_PROC_MSG) cmsg->commData[i].receiver.setProc(mype);
  }
  return cmsg;
#else
  return NULL;
#endif
}

void HybridLB::Loadbalancing()
{
  LBMigrateMsg* migrateMsg = Strategy(statsData, mig_msgs_expected+1);

  loadbalancing = 1;
  // send to children
  thisProxy.ReceiveMigration(migrateMsg, children.size(), children.getVec());

  // zero out stats

  theLbdb->ClearLoads();
  if (CkMyPe() == 0) {
    // FIXME
    double strat_end_time = CkWallTimer();
    if (_lb_args.debug())
        CkPrintf("Strat elapsed time %f\n",strat_end_time-start_lb_time);
  }
}

// migrate object in group
void HybridLB::ReceiveMigration(LBMigrateMsg *msg)
{
#if CMK_LBDB_ON
  FindNeighbors();

  // TODO: need to modify my LDStats to reflect migration
  // extend migration message to have obj load

  // do migration
  for(int i=0; i < msg->n_moves; i++) {
    MigrateInfo& move = msg->moves[i];
    const int me = CkMyPe();
    if (move.from_pe == me && move.to_pe != me) {
      DEBUGF(("[%d] migrating object to %d\n",move.from_pe,move.to_pe));
      // migrate object, in case it is already gone, inform toPe
      theLbdb->Migrate(move.obj,move.to_pe);
    } else if (move.from_pe != me && move.to_pe == me) {
      // CkPrintf("[%d] expecting object from %d\n",move.to_pe,move.from_pe);
      migrates_expected++;
    }
  }

  // TODO: broadcast LBMigrateMsg to children for migration
  thisProxy.ReceiveMigration(msg, children.size(), children.getVec());

  if (migrates_expected == 0 || migrates_expected == migrates_completed)
    MigrationDone();
#endif
}

void HybridLB::Migrated(LDObjHandle h, int waitBarrier)
{
  migrates_completed++;
  //  CkPrintf("[%d] An object migrated! %d %d\n",
  //  	   CkMyPe(),migrates_completed,migrates_expected);
  if (migrates_completed == migrates_expected) {
    MigrationDone();
  }
}


void HybridLB::MigrationDone()
{
#if CMK_LBDB_ON
  // TODO:  notify parent that we are done

  if (CkMyPe() == 0 && start_lb_time != 0.0) {
    double end_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("Load balancing step %d finished at %f duration %f\n",
	        step(),end_lb_time,end_lb_time - start_lb_time);
  }
  migrates_completed = 0;
  migrates_expected = -1;
  // Increment to next step
//  mystep++;
//  thisProxy [CkMyPe()].ResumeClients();

  thisProxy[parent].NotifyMigrationDone();
#endif
}

void HybridLB::NotifyMigrationDone()
{
#if CMK_LBDB_ON
  int reported = 0;
  // count if all children done, notify parent
  reported ++;
  if (reported == children.size()+1) {
    // if I am the one who sent the broadcast migration message, multicast to children to do load balancing
    if (loadbalancing == 0)
      thisProxy[parent].NotifyMigrationDone();
    else {
      thisProxy.Loadbalancing(children.size(), children.getVec());
      // I at this level am done
      // or doing a global barrier
      if (_lb_args.syncResume()) {
        CkCallback cb(CkIndex_HybridLB::ResumeClients((CkReductionMsg*)NULL),
                  thisProxy);
        contribute(0, NULL, CkReduction::sum_int, cb);
      }
      else
        thisProxy[CkMyPe()].ResumeClients();
    }
  }
#endif
}

void HybridLB::ResumeClients(CkReductionMsg *msg)
{
  ResumeClients();
  delete msg;
}

void HybridLB::ResumeClients()
{
#if CMK_LBDB_ON
  theLbdb->ResumeClients();
#endif
}

LBMigrateMsg* HybridLB::Strategy(LDStats* stats,int count)
{
#if 0
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
  LBMigrateMsg* msg = new(sizes,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
  msg->n_moves = 0;

  return msg;
#endif
}

int HybridLB::NeighborIndex(int pe)
{
    int peslot = -1;
    for(int i=0; i < children.size(); i++) {
      if (pe == children[i]) {
	peslot = i;
	break;
      }
    }
    return peslot;
}

/*@{*/
