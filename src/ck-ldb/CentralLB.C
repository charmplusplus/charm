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
#include "CentralLB.h"
#include "CentralLB.def.h"

#define  DEBUGF(x)    // CmiPrintf x;

CkGroupID loadbalancer;
char ** avail_vector_address;
int * lb_ptr;
int load_balancer_created;

#if CMK_LBDB_ON

static void getPredictedLoad(CentralLB::LDStats* stats, int count, 
                             LBMigrateMsg* msg, double *peLoads);
static int FindPEAfterMigration(LDObjid& id, CentralLB::LDStats* stats, int count,
							 LBMigrateMsg* msg, int bCheckStats);

void CreateCentralLB()
{
  loadbalancer = CProxy_CentralLB::ckNew();
}

void set_avail_vector(char * bitmap){
    int count;
    int assigned = 0;
    for(count = 0; count < CkNumPes(); count++){
	(*avail_vector_address)[count] = bitmap[count];
	if((bitmap[count] == 1) && !assigned){
	    *lb_ptr = count;
	    assigned = 1;
	}
    }
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
  theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),(void*)(this));
  theLbdb->
    NotifyMigrated((LDMigratedFn)(staticMigrated),(void*)(this));
  theLbdb->
    AddStartLBFn((LDStartLBFn)(staticStartLB),(void*)(this));

  stats_msg_count = 0;
  statsMsgsList = new CLBStatsMsg*[CkNumPes()];
  for(int i=0; i < CkNumPes(); i++)
    statsMsgsList[i] = 0;

  statsDataList = new LDStats[CkNumPes()];
  myspeed = theLbdb->ProcessorSpeed();
  theLbdb->CollectStatsOn();
  migrates_completed = 0;
  migrates_expected = -1;

  cur_ld_balancer = 0;
  new_ld_balancer = 0;
  int num_proc = CkNumPes();
  avail_vector = new char[num_proc];
  for(int proc = 0; proc < num_proc; proc++)
      avail_vector[proc] = 1;
  avail_vector_address = &(avail_vector);
  lb_ptr = &new_ld_balancer;

  load_balancer_created = 1;
}

CentralLB::~CentralLB()
{
  CkPrintf("Going away\n");
  theLbdb->
    RemoveStartLBFn((LDStartLBFn)(staticStartLB));
}

void CentralLB::AtSync()
{
  DEBUGF(("[%d] CentralLB At Sync step %d!!!!\n",CkMyPe(),mystep));

  if (!QueryBalanceNow(step())) {
    MigrationDone(0);
    return;
  }
  thisProxy [CkMyPe()].ProcessAtSync();
}

void CentralLB::ProcessAtSync()
{
  if (CkMyPe() == cur_ld_balancer) {
    start_lb_time = CmiWallTimer();
    CmiPrintf("[%s] Load balancing step %d starting at %f in PE%d\n",
    lbName(), step(),start_lb_time, cur_ld_balancer);
  }
  // Send stats
  const int osz = theLbdb->GetObjDataSz();
  const int csz = theLbdb->GetCommDataSz();

  int npes = CkNumPes();
  CLBStatsMsg* msg = new(osz, csz, npes, 0) CLBStatsMsg;
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
  theLbdb->ClearLoads();
  DEBUGF(("PE %d sending %d to ReceiveStats %d objs, %d comm\n",
  	   CkMyPe(),msg->serial,msg->n_objs,msg->n_comm));

  // Scheduler PART.

  if(CkMyPe() == 0){
      int num_proc = CkNumPes();
      for(int proc = 0; proc < num_proc; proc++){
	  msg->avail_vector[proc] = avail_vector[proc];
      }
      msg->next_lb = new_ld_balancer;
  }

  thisProxy [cur_ld_balancer].ReceiveStats(msg);
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

void CentralLB::ReceiveStats(CLBStatsMsg *m)
{
  int proc;
  const int pe = m->from_pe;
//  CkPrintf("Stats msg received, %d %d %d %d %p\n",
//  	   pe,stats_msg_count,m->n_objs,m->serial,m);

  if((pe == 0) && (CkMyPe() != 0)){
      new_ld_balancer = m->next_lb;
      int num_proc = CkNumPes();
      for(int proc = 0; proc < num_proc; proc++)
	  avail_vector[proc] = m->avail_vector[proc];
  }

  DEBUGF(("ReceiveStats from %d step: %d\n", pe, mystep));
  if (statsMsgsList[pe] != 0) {
    CkPrintf("*** Unexpected CLBStatsMsg in ReceiveStats from PE %d ***\n",
	     pe);
  } else {
    statsMsgsList[pe] = m;
    statsDataList[pe].total_walltime = m->total_walltime;
    statsDataList[pe].total_cputime = m->total_cputime;
    statsDataList[pe].idletime = m->idletime;
    statsDataList[pe].bg_walltime = m->bg_walltime;
    statsDataList[pe].bg_cputime = m->bg_cputime;
    statsDataList[pe].pe_speed = m->pe_speed;
    statsDataList[pe].utilization = 1.0;
    statsDataList[pe].available = CmiTrue;

    statsDataList[pe].n_objs = m->n_objs;
    statsDataList[pe].objData = m->objData;
    statsDataList[pe].n_comm = m->n_comm;
    statsDataList[pe].commData = m->commData;
    stats_msg_count++;
  }

  const int clients = CkNumPes();
  if (stats_msg_count == clients) {
//    double strat_start_time = CmiWallTimer();

//    CkPrintf("Before setting bitmap\n");
    for(proc = 0; proc < clients; proc++)
      statsDataList[proc].available = (CmiBool)avail_vector[proc];

    // if this is the step at which we need to dump the database
    simulation();

//    CkPrintf("Before Calling Strategy\n");

    LBMigrateMsg* migrateMsg = Strategy(statsDataList,clients);

//    CkPrintf("returned successfully\n");
    int num_proc = CkNumPes();

    for(proc = 0; proc < num_proc; proc++)
	migrateMsg->avail_vector[proc] = avail_vector[proc];
    migrateMsg->next_lb = new_ld_balancer;

    getPredictedLoad(statsDataList, clients, migrateMsg, migrateMsg->expectedLoad);

//  CkPrintf("calling recv migration\n");
    thisProxy.ReceiveMigration(migrateMsg);

#if 0
    {
      char fname[1024];
      static int phase = 0;
      if (QueryDumpData()) {
        sprintf(fname, "load%d", phase);
	writeStatsMsgs(fname);
      }
      phase ++;
    }
#endif

    // Zero out data structures for next cycle
    // CkPrintf("zeroing out data\n");
    for(int i=0; i < clients; i++) {
      delete statsMsgsList[i];
      statsMsgsList[i]=0;
    }
    stats_msg_count=0;
    double strat_end_time = CmiWallTimer();
    //     CkPrintf("Strat elapsed time %f\n",strat_end_time-strat_start_time);
  }

}

// test if sender and receiver in a commData is nonmigratable.
static int isMigratable(LDObjData **objData, int *len, int count, const LDCommData &commData)
{
  for (int pe=0 ; pe<count; pe++)
  {
    for (int i=0; i<len[pe]; i++)
      if (LDObjIDEqual(objData[pe][i].id(), commData.sender) ||
          LDObjIDEqual(objData[pe][i].id(), commData.receiver)) return 0;
  }
  return 1;
}

// remove in the LDStats those objects that are non migratable
void CentralLB::RemoveNonMigratable(LDStats* stats, int count)
{
  int pe;
  LDObjData **nonmig = new LDObjData*[count];
  int   *lens = new int[count];
  for (pe=0; pe<count; pe++) {
    int n_objs = stats[pe].n_objs;
    LDObjData *objData = stats[pe].objData; 
    int l=-1, h=n_objs;
    while (l<h) {
      while (objData[l+1].migratable && l<h) l++;
      while (h>0 && !objData[h-1].migratable && l<h) h--;
      if (h-l>2) {
        LDObjData tmp = objData[l+1];
        objData[l+1] = objData[h-1];
        objData[h-1] = tmp;
      }
      else 
        break;
    }
    stats[pe].n_objs = h;
    if (n_objs != h) CmiPrintf("Removed %d nonmigratable on pe:%d n_objs:%d migratable:%d\n", n_objs-h, pe, n_objs, h);
    nonmig[pe] = objData+stats[pe].n_objs;
    lens[pe] = n_objs-stats[pe].n_objs;

    // now turn nonmigratable to bg load
    for (int j=stats[pe].n_objs; j<n_objs; j++) {
      stats[pe].bg_walltime += objData[j].wallTime;
      stats[pe].bg_cputime += objData[j].cpuTime;
    }
  }

  // modify comm data
  for (pe=0; pe<count; pe++) {
    int n_comm = stats[pe].n_comm;
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
    stats[pe].n_comm = h;
    if (n_comm != h) CmiPrintf("Removed %d nonmigratable on pe:%d n_comm:%d migratable:%d\n", n_comm-h, pe, n_comm, h);
  }
  delete [] nonmig;
  delete [] lens;
}

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
      int num_proc = CkNumPes();
      for(int proc = 0; proc < num_proc; proc++)
	  avail_vector[proc] = m->avail_vector[proc];
  }

  if (migrates_expected == 0 || migrates_completed == migrates_expected)
    MigrationDone(1);
  delete m;
}


void CentralLB::MigrationDone(int balancing)
{
  if (balancing && CkMyPe() == cur_ld_balancer) {
    double end_lb_time = CmiWallTimer();
    CkPrintf("[%s] Load balancing step %d finished at %f\n",
	     lbName(), step(),end_lb_time);
    CkPrintf("[%s] duration %f memUsage:%dKB\n", lbName(),
	     end_lb_time - start_lb_time,
	     useMem() + LBDatabase::Object()->useMem()/1000);
  }
  migrates_completed = 0;
  migrates_expected = -1;
  // Increment to next step
  mystep++;
  thisProxy [CkMyPe()].ResumeClients();
}

void CentralLB::ResumeClients()
{
  DEBUGF(("Resuming clients on PE %d\n",CkMyPe()));
  theLbdb->ResumeClients();
}

LBMigrateMsg* CentralLB::Strategy(LDStats* stats,int count)
{
  for(int j=0; j < count; j++) {
    int i;
    LDObjData *odata = stats[j].objData;
    const int osz = stats[j].n_objs;

    CkPrintf(
      "Proc %d Sp %d Total time (wall,cpu) = %f %f Idle = %f Bg = %f %f\n",
      j,stats[j].pe_speed,stats[j].total_walltime,stats[j].total_cputime,
      stats[j].idletime,stats[j].bg_walltime,stats[j].bg_cputime);
    CkPrintf("------------- Object Data: PE %d: %d objects -------------\n",
	     j,osz);
    for(i=0; i < osz; i++) {
      CkPrintf("Object %d\n",i);
      CkPrintf("     id = %d\n",odata[i].id().id[0]);
      CkPrintf("  OM id = %d\n",odata[i].omID().id);
      CkPrintf("   Mig. = %d\n",odata[i].migratable);
      CkPrintf("    CPU = %f\n",odata[i].cpuTime);
      CkPrintf("   Wall = %f\n",odata[i].wallTime);
    }

    LDCommData *cdata = stats[j].commData;
    const int csz = stats[j].n_comm;

    CkPrintf("------------- Comm Data: PE %d: %d records -------------\n",
	     j,csz);
    for(i=0; i < csz; i++) {
      CkPrintf("Link %d\n",i);

      if (cdata[i].from_proc())
	CkPrintf("    sender PE = %d\n",cdata[i].src_proc);
      else
	CkPrintf("    sender id = %d:%d\n",
		 cdata[i].senderOM.id,cdata[i].sender.id[0]);

      if (cdata[i].to_proc())
	CkPrintf("  receiver PE = %d\n",cdata[i].dest_proc);
      else	
	CkPrintf("  receiver id = %d:%d\n",
		 cdata[i].receiverOM.id,cdata[i].receiver.id[0]);
      
      CkPrintf("     messages = %d\n",cdata[i].messages);
      CkPrintf("        bytes = %d\n",cdata[i].bytes);
    }
  }

  int sizes=0;
  LBMigrateMsg* msg = new(&sizes,1) LBMigrateMsg;
  msg->n_moves = 0;

  return msg;
}

void CentralLB::simulation() {
  if(step() == CkpvAccess(dumpStep))
  {
    // here we are supposed to dump the database
    writeStatsMsgs(CkpvAccess(dumpFile));
    CkExit();
    return;
  }
  else if(CkpvAccess(doSimulation))
  {
    // here we are supposed to read the data from the dump database
    readStatsMsgs(CkpvAccess(dumpFile));

    CLBSimResults simResults(stats_msg_count);

    // now pass it to the strategy routine
    LBMigrateMsg* migrateMsg = Strategy(statsDataList, stats_msg_count);

    // now calculate the results of the load balancing simulation
    FindSimResults(statsDataList, stats_msg_count, migrateMsg, &simResults);

    // now we have the simulation data, so print it and exit
    CmiPrintf("LBSim: Simulation of one load balancing step done.\n");
		simResults.PrintSimulationResults();

    delete migrateMsg;
    CkExit();
  }
}

void CentralLB::readStatsMsgs(const char* filename) {

  int i;
  FILE *f = fopen(filename, "r");

  // at this stage, we need to rebuild the statsMsgList and
  // statsDataList structures. For that first deallocate the
  // old structures
  for(int i = 0; i < stats_msg_count; i++)
  	delete statsMsgsList[i];
  delete[] statsMsgsList;

  PUP::fromDisk p(f);
  p|stats_msg_count;

  // now rebuild new structures
  statsMsgsList = new CLBStatsMsg*[stats_msg_count];
  statsDataList = new LDStats[stats_msg_count];

  for (i = 0; i < stats_msg_count; i++) {
  	statsMsgsList[i] = new CLBStatsMsg;
    CkPupMessage(p, (void **)&statsMsgsList[i], 1);

	CLBStatsMsg* m = statsMsgsList[i];

    statsDataList[i].total_walltime = m->total_walltime;
    statsDataList[i].total_cputime = m->total_cputime;
    statsDataList[i].idletime = m->idletime;
    statsDataList[i].bg_walltime = m->bg_walltime;
    statsDataList[i].bg_cputime = m->bg_cputime;
    statsDataList[i].pe_speed = m->pe_speed;
    statsDataList[i].utilization = 1.0;
    statsDataList[i].available = CmiTrue;

    statsDataList[i].n_objs = m->n_objs;
    statsDataList[i].objData = m->objData;
    statsDataList[i].n_comm = m->n_comm;
    statsDataList[i].commData = m->commData;
  }

  // file f is closed in the destructor of PUP::fromDisk
  CmiPrintf("readStatsMsg from %s\n", filename);
}

void CentralLB::writeStatsMsgs(const char* filename) {

  int i;
  FILE *f = fopen(filename, "w");

  PUP::toDisk p(f);
  p|stats_msg_count;

  for (i = 0; i < stats_msg_count; i++) {
    CkPupMessage(p, (void **)&statsMsgsList[i], 1);
  }
  CmiPrintf("writeStatsMsgs to %s\n", filename);
  CmiPrintf("LBDump: Dumped the load balancing data.\n");
}

static void getPredictedLoad(CentralLB::LDStats* stats, int count, LBMigrateMsg* msg, double *peLoads)
{
	int* msgSentCount = new int[count]; // # of messages sent by each PE
	int* msgRecvCount = new int[count]; // # of messages received by each PE
	int* byteSentCount = new int[count];// # of bytes sent by each PE
	int* byteRecvCount = new int[count];// # of bytes reeived by each PE

	for(int i = 0; i < count; i++)
		msgSentCount[i] = msgRecvCount[i] = byteSentCount[i] = byteRecvCount[i] = 0;

	for(int pe = 0; pe < count; pe++)
  	{
    	peLoads[pe] = stats[pe].bg_walltime;

    	for(int obj = 0; obj < stats[pe].n_objs; obj++)
    	{
			peLoads[pe] += stats[pe].objData[obj].wallTime;
    	}
	}

	// now for each migration, substract the load of the migrating object from the source pe
	// and add it to the destination pe
	for(int mig = 0; mig < msg->n_moves; mig++)
	{
		int from = msg->moves[mig].from_pe;
		int to = msg->moves[mig].to_pe;
		double wallTime;
		int oidx, cidx;
		
		// find the cpu time for the object that is migrating
		for(oidx = 0; oidx < stats[from].n_objs; oidx++)
			if(stats[from].objData[oidx].handle.id == msg->moves[mig].obj.id)
			{
				wallTime = stats[from].objData[oidx].wallTime;
				break;
			}
		CkAssert(oidx != stats[from].n_objs);
		peLoads[from] -= wallTime;
		peLoads[to] += wallTime;
	}

	// handling of the communication overheads. Here, for each "link" in the communication statistics,
	// find the sender and receiver PE and if they are not the same, add the costs, else don't add
	for(int pe = 0; pe < count; pe++)
	{
		// add the communication loads
		LDCommData* cdata = stats[pe].commData;
		const int csz = stats[pe].n_comm;

		for(int cidx = 0; cidx < csz; cidx++)
		{
			// find the sender and receiver PE for this "link"
			int senderPE, receiverPE;

			if(cdata[cidx].from_proc())
				senderPE = cdata[cidx].src_proc;
			else
			{
				// for sender, check just the migration messages
				senderPE = FindPEAfterMigration(cdata[cidx].sender, stats, count, msg, 0);
				if(senderPE == -1)
					senderPE = pe;
			}

			if(cdata[cidx].to_proc())
				receiverPE = cdata[cidx].dest_proc;
			else
				receiverPE = FindPEAfterMigration(cdata[cidx].receiver, stats, count, msg, 1);

			if(senderPE != receiverPE)
			{
				msgSentCount[senderPE] += cdata[cidx].messages;
				byteSentCount[senderPE] += cdata[cidx].bytes;

				msgRecvCount[receiverPE] += cdata[cidx].messages;
				byteRecvCount[receiverPE] += cdata[cidx].bytes;
			}
		}
	}

	// now for each processor, add to its load the send and receive overheads
	for(int i = 0; i < count; i++)
	{
		peLoads[i] += msgRecvCount[i]  * PER_MESSAGE_RECV_OVERHEAD +
					  msgSentCount[i]  * PER_MESSAGE_SEND_OVERHEAD +
					  byteRecvCount[i] * PER_BYTE_RECV_OVERHEAD +
					  byteSentCount[i] * PER_BYTE_SEND_OVERHEAD;
	}

	delete msgRecvCount;
	delete msgSentCount;
	delete byteRecvCount;
	delete byteSentCount;
}

void CentralLB::FindSimResults(LDStats* stats, int count, LBMigrateMsg* msg, CLBSimResults* simResults)
{
	CkAssert(simResults != NULL && count == simResults->numPes);
	// estimate the new loads of the processors. As a first approximation, this is the
	// sum of the cpu times of the objects on that processor
    getPredictedLoad(stats, count, msg, simResults->peLoads);
}

// find the PE of an object after migration. The bCheckStats flag indicates whether the stats is to
// be checked or not.
static int FindPEAfterMigration(LDObjid& id, CentralLB::LDStats* stats, int count, LBMigrateMsg* msg, int bCheckStats)
{
	// first check in the migration messages
	for(int i = 0; i < msg->n_moves; i++)
		if(msg->moves[i].obj.id == id)
			return msg->moves[i].to_pe;

	if(!bCheckStats)
		return -1;

	// not a migrating object, so find in the stats if requires
	for(int pe = 0; pe < count; pe++)
	{
		CmiBool found = CmiFalse;
		for(int obj = 0; obj < stats[pe].n_objs; obj++)
			if(stats[pe].objData[obj].handle.id == id)
				{ found = CmiTrue; break;}
		if(found)
			return pe;
	}
	CkAssert(0);
	return -1;
}

int CentralLB::useMem() { 
  return CkNumPes() * (sizeof(CentralLB::LDStats)+sizeof(CLBStatsMsg)) +
                        sizeof(CentralLB);
}

#endif

/*@}*/
