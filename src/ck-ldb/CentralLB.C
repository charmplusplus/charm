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
#include "CentralLB.def.h"
#include "LBDBManager.h"

#define  DEBUGF(x)    // CmiPrintf x;

CkGroupID loadbalancer;
int * lb_ptr;
int load_balancer_created;

#if CMK_LBDB_ON

static void getPredictedLoad(CentralLB::LDStats* stats, int count, 
		             LBMigrateMsg *, double *peLoads, 
			     double &, double &, int considerComm);

void CreateCentralLB()
{
  loadbalancer = CProxy_CentralLB::ckNew();
}

void CentralLB::staticStartLB(void* data)
{
  CentralLB *me = (CentralLB*)(data);
  me->StartLB();
}

void CentralLB::staticMigrated(void* data, LDObjHandle h)
{
  CentralLB *me = (CentralLB*)(data);
  me->Migrated(h);
}

void CentralLB::staticAtSync(void* data)
{
  CentralLB *me = (CentralLB*)(data);
  me->AtSync();
}

CentralLB::CentralLB()
{
  lbname = "CentralLB";
  mystep = 0;
  //  CkPrintf("Construct in %d\n",CkMyPe());
  theLbdb = CProxy_LBDatabase(lbdb).ckLocalBranch();

  // create and turn on by default
  receiver = theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),(void*)(this));
  notifier = theLbdb->getLBDB()->
    NotifyMigrated((LDMigratedFn)(staticMigrated),(void*)(this));
  startLbFnHdl = theLbdb->getLBDB()->
    AddStartLBFn((LDStartLBFn)(staticStartLB),(void*)(this));

//  turnOff();

  stats_msg_count = 0;
  statsMsgsList = new CLBStatsMsg*[CkNumPes()];
  for(int i=0; i < CkNumPes(); i++)
    statsMsgsList[i] = 0;

  statsData = new LDStats;

  myspeed = theLbdb->ProcessorSpeed();

  migrates_completed = 0;
  migrates_expected = -1;
  cur_ld_balancer = 0;
  int num_proc = CkNumPes();

  theLbdb->CollectStatsOn();

  load_balancer_created = 1;
}

CentralLB::~CentralLB()
{
  CkPrintf("Going away\n");
  delete [] statsMsgsList;
  theLbdb->getLBDB()->
    RemoveNotifyMigrated(notifier);
  theLbdb->
    RemoveStartLBFn((LDStartLBFn)(staticStartLB));
}

void CentralLB::turnOn() 
{
  theLbdb->getLBDB()->
    TurnOnBarrierReceiver(receiver);
  theLbdb->getLBDB()->
    TurnOnNotifyMigrated(notifier);
  theLbdb->getLBDB()->
    TurnOnStartLBFn(startLbFnHdl);
}

void CentralLB::turnOff() 
{
  theLbdb->getLBDB()->
    TurnOffBarrierReceiver(receiver);
  theLbdb->getLBDB()->
    TurnOffNotifyMigrated(notifier);
  theLbdb->getLBDB()->
    TurnOffStartLBFn(startLbFnHdl);
}

void CentralLB::AtSync()
{
  DEBUGF(("[%d] CentralLB At Sync step %d!!!!\n",CkMyPe(),mystep));

  // if num of processor is only 1, nothing should happen
  if (!QueryBalanceNow(step()) || CkNumPes() == 1) {
    MigrationDone(0);
    return;
  }
  thisProxy [CkMyPe()].ProcessAtSync();
}

void CentralLB::ProcessAtSync()
{
  if (CkMyPe() == cur_ld_balancer) {
    start_lb_time = CmiWallTimer();
  }
  // build and send stats
  const int osz = theLbdb->GetObjDataSz();
  const int csz = theLbdb->GetCommDataSz();

  int npes = CkNumPes();
  CLBStatsMsg* msg = new CLBStatsMsg(osz, csz);
  msg->from_pe = CkMyPe();
  // msg->serial = rand();
  msg->serial = CrnRand();

  theLbdb->TotalTime(&msg->total_walltime,&msg->total_cputime);
  theLbdb->IdleTime(&msg->idletime);
  theLbdb->BackgroundLoad(&msg->bg_walltime,&msg->bg_cputime);
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
  DEBUGF(("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
           CkMyPe(),msg->serial,msg->n_objs,msg->n_comm));

// Scheduler PART.

  if(CkMyPe() == cur_ld_balancer) {
    LBDatabaseObj()->get_avail_vector(msg->avail_vector);
    msg->next_lb = LBDatabaseObj()->new_lbbalancer();
  }

  thisProxy[cur_ld_balancer].ReceiveStats(msg);
}

void CentralLB::Migrated(LDObjHandle h)
{
  migrates_completed++;
  //  CkPrintf("[%d] An object migrated! %d %d\n",
  //  	   CkMyPe(),migrates_completed,migrates_expected);
  if (migrates_completed == migrates_expected) {
    MigrationDone(1);
  }
}

// build data from buffered msg
void CentralLB::buildStats()
{
    statsData->count = stats_msg_count;
    statsData->objData = new LDObjData[statsData->n_objs];
    statsData->from_proc = new int[statsData->n_objs];
    statsData->to_proc = new int[statsData->n_objs];
    statsData->commData = new LDCommData[statsData->n_comm];
    int nobj = 0;
    int ncom = 0;
    int nmigobj = 0;
    // copy all data in individule message to this big structure
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
    if (lb_debug) {
      CmiPrintf("n_obj:%d migratable:%d ncom:%d\n", nobj, nmigobj, ncom);
    }
}

void CentralLB::ReceiveStats(CkMarshalledCLBStatsMessage &msg)
{
  CLBStatsMsg *m = (CLBStatsMsg *)msg.getMessage();
  int proc;
  const int pe = m->from_pe;
//  CkPrintf("Stats msg received, %d %d %d %d %p\n",
//  	   pe,stats_msg_count,m->n_objs,m->serial,m);

  if((pe == 0) && (CkMyPe() != 0)){
      LBDatabaseObj()->set_avail_vector(m->avail_vector,  m->next_lb);
  }

  DEBUGF(("ReceiveStats from %d step: %d\n", pe, mystep));
  if (statsMsgsList[pe] != 0) {
    CkPrintf("*** Unexpected CLBStatsMsg in ReceiveStats from PE %d ***\n",
	     pe);
  } else {
    statsMsgsList[pe] = m;
    // store per processor data right away
    struct ProcStats &procStat = statsData->procs[pe];
    procStat.total_walltime = m->total_walltime;
    procStat.total_cputime = m->total_cputime;
    if (lb_ignoreBgLoad) {
      procStat.idletime = 0.0;
      procStat.bg_walltime = 0.0;
      procStat.bg_cputime = 0.0;
    }
    else {
      procStat.idletime = m->idletime;
      procStat.bg_walltime = m->bg_walltime;
      procStat.bg_cputime = m->bg_cputime;
    }
    procStat.pe_speed = m->pe_speed;
    procStat.utilization = 1.0;
    procStat.available = CmiTrue;
    procStat.n_objs = m->n_objs;

    statsData->n_objs += m->n_objs;
    statsData->n_comm += m->n_comm;
    stats_msg_count++;
  }

  const int clients = CkNumPes();
  if (stats_msg_count == clients) {
    if (lb_debug) 
      CmiPrintf("[%s] Load balancing step %d starting at %f in PE%d\n",
                 lbName(), step(),start_lb_time, cur_ld_balancer);
//    double strat_start_time = CmiWallTimer();

    // build data
    buildStats();

    // if this is the step at which we need to dump the database
    simulation();

    char *availVector = LBDatabaseObj()->availVector();
    for(proc = 0; proc < clients; proc++)
      statsData->procs[proc].available = (CmiBool)availVector[proc];

//    CkPrintf("Before Calling Strategy\n");
    LBMigrateMsg* migrateMsg = Strategy(statsData, clients);

//    CkPrintf("returned successfully\n");
    LBDatabaseObj()->get_avail_vector(migrateMsg->avail_vector);
    migrateMsg->next_lb = LBDatabaseObj()->new_lbbalancer();

//  calculate predicted load
//  very time consuming though, so only happen when debugging is on
    if (lb_debug) {
      double minObjLoad, maxObjLoad;
      getPredictedLoad(statsData, clients, migrateMsg, migrateMsg->expectedLoad, minObjLoad, maxObjLoad, 1);
    }

//  CkPrintf("calling recv migration\n");
    thisProxy.ReceiveMigration(migrateMsg);

    // Zero out data structures for next cycle
    // CkPrintf("zeroing out data\n");
    statsData->clear();
    stats_msg_count=0;

//    double strat_end_time = CmiWallTimer();
//    CkPrintf("Strat elapsed time %f\n",strat_end_time-strat_start_time);
  }

}

// test if sender and receiver in a commData is nonmigratable.
static int isMigratable(LDObjData **objData, int *len, int count, const LDCommData &commData)
{
  for (int pe=0 ; pe<count; pe++)
  {
    for (int i=0; i<len[pe]; i++)
      if (LDObjIDEqual(objData[pe][i].objID(), commData.sender.objID()) ||
          LDObjIDEqual(objData[pe][i].objID(), commData.receiver.get_destObj().objID())) 
      return 0;
  }
  return 1;
}

#if 0
// remove in the LDStats those objects that are non migratable
void CentralLB::removeNonMigratable(LDStats* stats, int count)
{
  int pe;
  LDObjData **nonmig = new LDObjData*[count];
  int   *lens = new int[count];
  int n_objs = stats->n_objs;
    LDObjStats *objStat = stats.objData[n]; 
    int l=-1, h=n_objs;
    while (l<h) {
      while (objStat[l+1].data.migratable && l<h) l++;
      while (h>0 && !objStat[h-1].data.migratable && l<h) h--;
      if (h-l>2) {
        LDObjStats tmp = objStat[l+1];
        objStat[l+1] = objStat[h-1];
        objStat[h-1] = tmp;
      }
    }
    stats->n_objs = h;
    if (n_objs != h) CmiPrintf("Removed %d nonmigratable on pe:%d n_objs:%d migratable:%d\n", n_objs-h, pe, n_objs, h);
    nonmig[pe] = objData+stats[pe].n_objs;
    lens[pe] = n_objs-stats[pe].n_objs;

    // now turn nonmigratable to bg load
    for (int j=stats[pe].n_objs; j<n_objs; j++) {
      stats[pe].bg_walltime += objData[j].wallTime;
      stats[pe].bg_cputime += objData[j].cpuTime;
    }

  // modify comm data
  for (pe=0; pe<count; pe++) {
    int n_comm = stats[pe].n_comm;
    if (n_comm == 0) continue;
    LDCommData *commData = stats[pe].commData;
    int l=-1, h=n_comm;
    while (l<h) {
      while (isMigratable(nonmig, lens, count, commData[l+1]) && l<h) l++;
      while (!isMigratable(nonmig, lens, count, commData[h-1]) && l<h) h--;
      if (h-l>2) {
        LDCommData tmp = commData[l+1];
        commData[l+1] = commData[h-1];
        commData[h-1] = tmp;
      }
      else 
        break;
    }
    if (l==-1 && h==-1) h=0;
    stats[pe].n_comm = h;
    if (n_comm != h) CmiPrintf("Removed %d nonmigratable on pe:%d n_comm:%d migratable:%d\n", n_comm-h, pe, n_comm, h);
  }
  delete [] nonmig;
  delete [] lens;
}
#endif

void CentralLB::ReceiveMigration(LBMigrateMsg *m)
{
  int i;
  for (i=0; i<CkNumPes(); i++) theLbdb->lastLBInfo.expectedLoad[i] = m->expectedLoad[i];
  
  DEBUGF(("[%d] in ReceiveMigration %d moves\n",CkMyPe(),m->n_moves));
  migrates_expected = 0;
  for(i=0; i < m->n_moves; i++) {
    MigrateInfo& move = m->moves[i];
    const int me = CkMyPe();
    if (move.from_pe == me && move.to_pe != me) {
      DEBUGF(("[%d] migrating object to %d\n",move.from_pe,move.to_pe));
      theLbdb->Migrate(move.obj,move.to_pe);
    } else if (move.from_pe != me && move.to_pe == me) {
	//  CkPrintf("[%d] expecting object from %d\n",move.to_pe,move.from_pe);
      migrates_expected++;
    }
  }
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
}


void CentralLB::MigrationDone(int balancing)
{
  if (balancing && lb_debug && CkMyPe() == cur_ld_balancer) {
    double end_lb_time = CmiWallTimer();
      CkPrintf("[%s] Load balancing step %d finished at %f\n",
  	        lbName(), step(),end_lb_time);
      double lbdbMemsize = LBDatabase::Object()->useMem()/1000;
      CkPrintf("[%s] duration %fs memUsage: LBManager:%dKB CentralLB:%dKB\n", 
  	        lbName(), end_lb_time - start_lb_time,
	        (int)lbdbMemsize, (int)(useMem()/1000));
  }
  migrates_completed = 0;
  migrates_expected = -1;
  // clear load stats
  if (balancing) theLbdb->ClearLoads();
  // Increment to next step
  mystep++;
  thisProxy [CkMyPe()].ResumeClients();
}

void CentralLB::ResumeClients()
{
  DEBUGF(("Resuming clients on PE %d\n",CkMyPe()));
  theLbdb->ResumeClients();
}

// default load balancing strategy
LBMigrateMsg* CentralLB::Strategy(LDStats* stats,int count)
{
  work(stats, count);
  return createMigrateMsg(stats, count);
}

void CentralLB::work(LDStats* stats,int count)
{
  int i;
  for(int pe=0; pe < count; pe++) {
    struct ProcStats &proc = stats->procs[pe];

    CkPrintf(
      "Proc %d Sp %d Total time (wall,cpu) = %f %f Idle = %f Bg = %f %f\n",
      pe,proc.pe_speed,proc.total_walltime,proc.total_cputime,
      proc.idletime,proc.bg_walltime,proc.bg_cputime);
  }

  int osz = stats->n_objs;
    CkPrintf("------------- Object Data: %d objects -------------\n",
	     stats->n_objs);
    for(i=0; i < osz; i++) {
      LDObjData &odata = stats->objData[i];
      CkPrintf("Object %d\n",i);
      CkPrintf("     id = %d\n",odata.objID().id[0]);
      CkPrintf("  OM id = %d\n",odata.omID().id);
      CkPrintf("   Mig. = %d\n",odata.migratable);
      CkPrintf("    CPU = %f\n",odata.cpuTime);
      CkPrintf("   Wall = %f\n",odata.wallTime);
    }

    const int csz = stats->n_comm;

    CkPrintf("------------- Comm Data: %d records -------------\n",
	     csz);
    LDCommData *cdata = stats->commData;
    for(i=0; i < csz; i++) {
      CkPrintf("Link %d\n",i);

      if (cdata[i].from_proc())
	CkPrintf("    sender PE = %d\n",cdata[i].src_proc);
      else
	CkPrintf("    sender id = %d:%d\n",
		 cdata[i].sender.omID().id,cdata[i].sender.objID().id[0]);

      if (cdata[i].recv_type() == LD_PROC_MSG)
	CkPrintf("  receiver PE = %d\n",cdata[i].receiver.proc());
      else	
	CkPrintf("  receiver id = %d:%d\n",
		 cdata[i].receiver.get_destObj().omID().id,cdata[i].receiver.get_destObj().objID().id[0]);
      
      CkPrintf("     messages = %d\n",cdata[i].messages);
      CkPrintf("        bytes = %d\n",cdata[i].bytes);
    }
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
      migrateInfo.insertAtEnd(migrateMe);
    }
  }

  int migrate_count=migrateInfo.length();
  LBMigrateMsg* msg = new(&migrate_count,1) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  if (lb_debug)
    CkPrintf("%s: %d objects migrating.\n", lbname, migrate_count);
  return msg;
}

void CentralLB::simulation() {
  if(step() == LBSimulation::dumpStep)
  {
    // here we are supposed to dump the database
    writeStatsMsgs(LBSimulation::dumpFile);
    CmiPrintf("LBDump: Dumped the load balancing data.\n");
    CmiPrintf("Charm++> Exiting...\n");
    CkExit();
    return;
  }
  else if(LBSimulation::doSimulation)
  {
    // here we are supposed to read the data from the dump database
    readStatsMsgs(LBSimulation::dumpFile);

    LBSimulation simResults(LBSimulation::simProcs);

    // now pass it to the strategy routine
    double startT = CmiWallTimer();
    CmiPrintf("%s> Strategy starts ... \n", lbname);
    LBMigrateMsg* migrateMsg = Strategy(statsData, LBSimulation::simProcs);
    CmiPrintf("%s> Strategy took %fs. \n", lbname, CmiWallTimer()-startT);

    // now calculate the results of the load balancing simulation
    findSimResults(statsData, LBSimulation::simProcs, migrateMsg, &simResults);

    // now we have the simulation data, so print it and exit
    CmiPrintf("Charm++> LBSim: Simulation of one load balancing step done.\n");
    simResults.PrintSimulationResults();

    delete migrateMsg;
    CmiPrintf("Charm++> Exiting...\n");
    CkExit();
  }
}

void CentralLB::readStatsMsgs(const char* filename) {

  int i;
  FILE *f = fopen(filename, "r");
  if (f==NULL) CmiAbort("Fatal Error> Cannot open LB Dump file!\n");

  // at this stage, we need to rebuild the statsMsgList and
  // statsDataList structures. For that first deallocate the
  // old structures
  for(i = 0; i < stats_msg_count; i++)
  	delete statsMsgsList[i];
  delete[] statsMsgsList;

  PUP::fromDisk p(f);
  p|stats_msg_count;

  CmiPrintf("readStatsMsgs for %d pes starts ... \n", stats_msg_count);
  if (LBSimulation::simProcs == 0) LBSimulation::simProcs = stats_msg_count;

  // LBSimulation::simProcs must be set
  statsData->pup(p);

  CmiPrintf("Simulation for %d pes \n", LBSimulation::simProcs);

  // file f is closed in the destructor of PUP::fromDisk
  CmiPrintf("ReadStatsMsg from %s completed\n", filename);
}

void CentralLB::writeStatsMsgs(const char* filename) {

  FILE *f = fopen(filename, "w");
  if (f == NULL) 
    CmiAbort("writeStatsMsgs failed to open the output file!\n");

  PUP::toDisk p(f);

  p|stats_msg_count;
  statsData->pup(p);

  fclose(f);

  CmiPrintf("WriteStatsMsgs to %s succeed!\n", filename);
}

// calculate the predicted wallclock/cpu load for every processors
// considering communication overhead if considerComm is true
static void getPredictedLoad(CentralLB::LDStats* stats, int count, 
                             LBMigrateMsg *msg, double *peLoads, 
                             double &minObjLoad, double &maxObjLoad,
			     int considerComm)
{
        int i, pe;

        minObjLoad = 1.0e20;	// I suppose no object load is beyond this
	maxObjLoad = 0.0;

	stats->makeCommHash();

 	// update to_proc according to migration msgs
	for(i = 0; i < msg->n_moves; i++) {
	  MigrateInfo &mInfo = msg->moves[i];
	  int idx = stats->getHash(mInfo.obj.objID(), mInfo.obj.omID());
	  CmiAssert(idx != -1);
          stats->to_proc[idx] = mInfo.to_pe;
	}

	for(pe = 0; pe < count; pe++)
    	  peLoads[pe] = stats->procs[pe].bg_walltime;

    	for(int obj = 0; obj < stats->n_objs; obj++)
    	{
		int pe = stats->to_proc[obj];
		double &oload = stats->objData[obj].wallTime;
		if (oload < minObjLoad) minObjLoad = oload;
		if (oload > maxObjLoad) maxObjLoad = oload;
		peLoads[pe] += oload;
	}

	// handling of the communication overheads. 
	if (considerComm) {
	  int* msgSentCount = new int[count]; // # of messages sent by each PE
	  int* msgRecvCount = new int[count]; // # of messages received by each PE
	  int* byteSentCount = new int[count];// # of bytes sent by each PE
	  int* byteRecvCount = new int[count];// # of bytes reeived by each PE
	  for(i = 0; i < count; i++)
	    msgSentCount[i] = msgRecvCount[i] = byteSentCount[i] = byteRecvCount[i] = 0;

          for (int cidx=0; cidx < stats->n_comm; cidx++) {
	    LDCommData& cdata = stats->commData[cidx];
	    int senderPE, receiverPE;
	    if (cdata.from_proc())
	      senderPE = cdata.src_proc;
  	    else {
	      int idx = stats->getHash(cdata.sender);
	      CmiAssert(idx != -1);
	      senderPE = stats->to_proc[idx];
	      CmiAssert(senderPE != -1);
	    }
	    if (cdata.receiver.get_type() == LD_PROC_MSG)
	      receiverPE = cdata.receiver.proc();
	    else {
	      int idx = stats->getHash(cdata.receiver.get_destObj());
	      CmiAssert(idx != -1);
	      receiverPE = stats->to_proc[idx];
	      CmiAssert(receiverPE != -1);
	    }
	    if(senderPE != receiverPE)
	    {
	  	msgSentCount[senderPE] += cdata.messages;
		byteSentCount[senderPE] += cdata.bytes;

		msgRecvCount[receiverPE] += cdata.messages;
		byteRecvCount[receiverPE] += cdata.bytes;
	    }
	  }

	  // now for each processor, add to its load the send and receive overheads
	  for(i = 0; i < count; i++)
	  {
		peLoads[i] += msgRecvCount[i]  * PER_MESSAGE_RECV_OVERHEAD +
			      msgSentCount[i]  * PER_MESSAGE_SEND_OVERHEAD +
			      byteRecvCount[i] * PER_BYTE_RECV_OVERHEAD +
			      byteSentCount[i] * PER_BYTE_SEND_OVERHEAD;
	  }
	  delete [] msgRecvCount;
	  delete [] msgSentCount;
	  delete [] byteRecvCount;
	  delete [] byteSentCount;
	}
}

void CentralLB::findSimResults(LDStats* stats, int count, LBMigrateMsg* msg, LBSimulation* simResults)
{
    CkAssert(simResults != NULL && count == simResults->numPes);
    // estimate the new loads of the processors. As a first approximation, this is the
    // get background load
    for(int pe = 0; pe < count; pe++)
    	  simResults->bgLoads[pe] = stats->procs[pe].bg_walltime;
    // sum of the cpu times of the objects on that processor
    double startT = CmiWallTimer();
    getPredictedLoad(stats, count, msg, simResults->peLoads, 
		     simResults->minObjLoad, simResults->maxObjLoad,1);
    CmiPrintf("getPredictedLoad finished in %fs\n", CmiWallTimer()-startT);
}

int CentralLB::useMem() { 
  return CkNumPes() * (sizeof(CentralLB::LDStats)+sizeof(CLBStatsMsg *)) +
                        sizeof(CentralLB);
}

static inline int i_abs(int c) { return c>0?c:-c; }

inline static int ObjKey(const LDObjid &oid, const int hashSize) {
  // make sure all positive
  return ((i_abs(oid.id[0])<<16)
	 |(i_abs(oid.id[1])<<8)
	 |i_abs(oid.id[2])) % hashSize;
}

void CentralLB::LDStats::makeCommHash() {
  // hash table is already build
  if (objHash) return;
   
  int i;
  hashSize = n_objs*2;
  objHash = new int[hashSize];
  for(i=0;i<hashSize;i++)
        objHash[i] = -1;
   
  for(i=0;i<n_objs;i++){
        const LDObjid &oid = objData[i].objID();
        int hash = ObjKey(oid, hashSize);
	CmiAssert(hash != -1);
        while(objHash[hash] != -1)
            hash = (hash+1)%hashSize;
        objHash[hash] = i;
  }
}

void CentralLB::LDStats::deleteCommHash() {
  if (objHash) delete [] objHash;
  objHash = NULL;
}

int CentralLB::LDStats::getHash(const LDObjid &oid, const LDOMid &mid)
{
    CmiAssert(hashSize > 0);
    int hash = ObjKey(oid, hashSize);

    for(int id=0;id<hashSize;id++){
        int index = (id+hash)%hashSize;
	if (index == -1 || objHash[index] == -1) return -1;
        if (LDObjIDEqual(objData[objHash[index]].objID(), oid) &&
            LDOMidEqual(objData[objHash[index]].omID(), mid))
            return objHash[index];
    }
    //  CkPrintf("not found \n");
    return -1;
}

int CentralLB::LDStats::getHash(const LDObjKey &objKey)
{
  const LDObjid &oid = objKey.objID();
  const LDOMid  &mid = objKey.omID();
  return getHash(oid, mid);
}

void CentralLB::LDStats::pup(PUP::er &p)
{
  int i;
  p(count);  
  p(n_objs);
  p(n_comm);
  if (p.isUnpacking()) {
    // user can specify simulated processors other than the real # of procs.
    int maxpe = count>LBSimulation::simProcs?count:LBSimulation::simProcs;
    procs = new ProcStats[maxpe];
    objData = new LDObjData[n_objs];
    commData = new LDCommData[n_comm];
    from_proc = new int[n_objs];
    to_proc = new int[n_objs];
    objHash = NULL;
  }
  // ignore the background load when unpacking
  if (p.isUnpacking()) {
    ProcStats dummy;
    for (i=0; i<count; i++) p|dummy; 
  }
  else
    for (i=0; i<count; i++) p|procs[i];
  for (i=0; i<n_objs; i++) p|objData[i]; 
  p(from_proc, n_objs);
  p(to_proc, n_objs);
  for (i=0; i<n_comm; i++) p|commData[i];
  if (p.isUnpacking())
    count = LBSimulation::simProcs;
}

int CentralLB::LDStats::useMem() { 
  // calculate the memory usage of this LB (superclass).
  return sizeof(CentralLB) + sizeof(LDStats) + sizeof(ProcStats)*count + 
	 (sizeof(LDObjData) + 2*sizeof(int)) * n_objs +
 	 sizeof(LDCommData) * n_comm;
}

/**
  CLBStatsMsg is not a real message now.
  CLBStatsMsg is used for all processors to fill in their local load and comm
  statistics and send to processor 0
*/

CLBStatsMsg::CLBStatsMsg(int osz, int csz) {
  objData = new LDObjData[osz];
  commData = new LDCommData[csz];
  avail_vector = new char[CkNumPes()];
}

CLBStatsMsg::~CLBStatsMsg() {
  delete [] objData;
  delete [] commData;
  delete [] avail_vector;
}

void CLBStatsMsg::pup(PUP::er &p) {
  int i;
  p|from_pe;
  p|serial;
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
  if (p.isUnpacking()) avail_vector = new char[CkNumPes()];
  p(avail_vector, CkNumPes());
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

#endif

/*@}*/
