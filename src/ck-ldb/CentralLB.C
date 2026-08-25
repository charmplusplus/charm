
/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <algorithm>
#include <charm++.h>
#include "ck.h"
#include "envelope.h"
#include "CentralLB.h"
#include "telemetry.h"
#include "LBSimulation.h"
#if CMK_CUDA || CMK_HIP
#if CMK_CUDA
#include <cupti.h>
#endif
#include "gpumanager.h"
// extern void hapiProcessCuptiBuffers();
// extern void hapiClearCuptiData();
CsvExtern(GPUManager, gpu_manager);
CkpvExtern(int, _lb_obj_index);
#include "hapi.h"
#endif

#define  DEBUGF(x)       // CmiPrintf x;
#define  DEBUG(x)        // x;

#if CMK_MEM_CHECKPOINT
   /* can not handle reduction in inmem FT */
#define USE_REDUCTION         0
#define USE_LDB_SPANNING_TREE 0
#else
#define USE_REDUCTION         1
#define USE_LDB_SPANNING_TREE 1
#endif


#if CMK_GRID_QUEUE_AVAILABLE
CpvExtern(void *, CkGridObject);
#endif

#if CMK_GLOBAL_LOCATION_UPDATE      
extern void UpdateLocation(MigrateInfo& migData); 
#endif

#if CMK_SHRINK_EXPAND
extern "C" void charmrun_realloc(char *s);
extern char willContinue;
extern realloc_state pending_realloc_state;
extern char * se_avail_vector;
extern std::vector<char> se_avail_snapshot;
extern int mynewpe;
extern char *_shrinkexpand_basedir;
extern int numProcessAfterRestart;
#endif
CkGroupID loadbalancer;
int * lb_ptr;
extern bool load_balancer_created;

static void lbinit()
{
  LBRegisterBalancer<CentralLB>("CentralLB", "CentralLB base class", false);
}

static int broadcastThreshold = 32;

static void getPredictedLoadWithMsg(BaseLB::LDStats* stats, int count, 
		             LBMigrateMsg *, LBInfo &info, int considerComm);

void CentralLB::initLB(const CkLBOptions &opt)
{
#if CMK_LBDB_ON
  lbname = "CentralLB";
  thisProxy = CProxy_CentralLB(thisgroup);
  //  CkPrintf("Construct in %d\n",CkMyPe());
  loadbalancer = thisgroup;
  // create and turn on by default
  startLbFnHdl = lbmgr->
    AddStartLBFn(this, &CentralLB::StartLB);

  // CkPrintf("[%d] CentralLB initLB \n",CkMyPe());
  if (opt.getSeqNo() > 0 || (_lb_args.metaLbOn() && _lb_args.metaLbModelDir() != nullptr))
    turnOff();

  #if (CMK_CUDA || CMK_HIP) && CMK_LB_USER_DATA
  CkpvAccess(_lb_obj_index) = LBRegisterObjUserData(sizeof(size_t));//gpu allocation size
  #endif

  stats_msg_count = 0;
  statsMsgsList = NULL;
  statsData = NULL;

  storedMigrateMsg = NULL;
  reduction_started = false;

  // for future predictor
  if (_lb_predict) predicted_model = new FutureModel(_lb_predict_window);
  else predicted_model=0;
  // register user interface callbacks
  lbmgr->SetupPredictor(this, &CentralLB::predictorOn, &CentralLB::predictorOn, &CentralLB::predictorOff, &CentralLB::changePredictor);

  myspeed = lbmgr->ProcessorSpeed();

  migrates_completed = 0;
  future_migrates_completed = 0;
  migrates_expected = -1;
  future_migrates_expected = -1;
  cur_ld_balancer = _lb_args.central_pe();      // 0 default
  lbdone = 0;
  count_msgs=0;
  statsMsg = NULL;
  use_thread = false;

  if (_lb_args.statsOn()) lbmgr->CollectStatsOn();

  load_balancer_created = true;
#endif
#ifdef TEMP_LDB
	logicalCoresPerNode=physicalCoresPerNode=4;
	logicalCoresPerChip=4;
	numSockets=1;
#endif

}

CentralLB::~CentralLB()
{
#if CMK_LBDB_ON
  delete [] statsMsgsList;
  delete statsData;
  lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  if (lbmgr) {
    lbmgr->RemoveStartLBFn(startLbFnHdl);
  }
#endif
}

#if CMK_LBDB_ON && CMK_SHRINK_EXPAND
// On survivor restart, statsMsgsList was allocated against the pre-rescale
// CkNumPes() and won't be re-sized: ReceiveStats only allocates when the
// pointer is NULL, so an expanded cluster's stats from the new PE land at an
// out-of-bounds index, and a shrunk cluster keeps a stale slot. statsData's
// procs vector has the same staleness. Drop both so the next LB step rebuilds
// them with the post-rescale CkNumPes().
//
// Also reset the migration counters. MigrationDoneImpl normally zeros these,
// but the rescale path forks off at CheckForRealloc before MigrationDoneImpl
// runs. On the next LB step, ProcessReceiveMigration sets
// migrates_expected=N for the incoming migrations, but the stale
// migrates_completed (carried over from the rescale-triggering LB) is already
// > N, so the equality check that triggers MigrationDone never matches and
// the LB step hangs after migration. Same for lbdone, which CheckMigrationComplete
// only resets after lbdone reaches 2; if the rescale interrupts at lbdone==1,
// the next LB step starts at lbdone=1 and only needs ONE
// CheckMigrationComplete to flip — which then prematurely calls MigrationDoneImpl.
void CentralLB::flushStates()
{
  BaseLB::flushStates();
  if (statsMsgsList) {
    for (int i = 0; i < stats_msg_count; i++) delete statsMsgsList[i];
    delete[] statsMsgsList;
    statsMsgsList = NULL;
  }
  delete statsData;
  statsData = NULL;
  stats_msg_count = 0;
  reduction_started = false;
  migrates_completed = 0;
  migrates_expected = -1;
  future_migrates_completed = 0;
  future_migrates_expected = -1;
  lbdone = 0;
}
#endif

void CentralLB::SetPESpeed(int speed) 
{
  myspeed = speed;
}

int CentralLB::GetPESpeed() 
{
  return myspeed;
}

void CentralLB::CallLB()
{
  #if CMK_LBDB_ON
  DEBUGF(("[%d] CentralLB AtSync step %d!!!!!\n",CkMyPe(),step()));
#if CMK_MEM_CHECKPOINT
  CkSetInLdb();
#endif

  // if num of processor is only 1, nothing should happen
  if (!QueryBalanceNow(step()) || CkNumPes() == 1) {
    MigrationDone(0);
    return;
  }
  
#if CMK_CUDA || CMK_HIP
#if CMK_SMP
  CmiNodeBarrier();  // ensure rank 0 finishes buffer processing before other ranks read the map
#endif
if (CmiMyRank() == 0)
{
#if CMK_CUDA
  double start = CkWallTimer();
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);//sync flush cupti records which are finished, does not wait for partial records
  hapiProcessCuptiBuffers();
#endif
}
#if CMK_SMP
  CmiNodeBarrier();  // ensure rank 0 finishes buffer processing before other ranks read the map
#endif
  // Every PE matches its own objects against the shared per-process CUPTI map
  lbmgr->SetObjGPULoad(CsvAccess(gpu_manager).cupti_obj_gpu_times_);
#endif

  {
    thisProxy [CkMyPe()].ProcessAtSync();
  }
#endif
}

void CentralLB::InvokeLB()
{
  lbmgr->lb_in_progress = true;
#if CMK_SHRINK_EXPAND
  contribute(CkCallback(CkReductionTarget(CentralLB, CheckForLB), thisProxy[0]));
#else
  CallLB();
#endif
}

#if CMK_SHRINK_EXPAND
/* Opt-in trace of the load balancing round's completion chain, for diagnosing
   a round that starts and never releases its clients. */
static bool CkLBTraceOn() {
  static int on = -1;
  if (on < 0) on = (getenv("CHARM_LB_TRACE") != NULL) ? 1 : 0;
  return on != 0;
}
#define LB_TRACE(...) do { if (CkLBTraceOn()) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while (0)
#else
#define LB_TRACE(...) do { } while (0)
#endif

void CentralLB::ProcessAtSync()
{
#if CMK_LBDB_ON
  LB_TRACE("[%d] LB ProcessAtSync (reduction_started=%d)\n", CkMyPe(),
           (int)reduction_started);
  if (reduction_started) return;              // reducton in progress

  if (CkMyPe() == cur_ld_balancer) {
    start_lb_time = CkWallTimer();
  }


  // build message
  BuildStatsMsg();

#if USE_REDUCTION
    // reduction to get total number of objects and comm
    // so that processor 0 can pre-allocate load balancing database
  int counts[2];
  counts[0] = lbmgr->GetObjDataSz();
  counts[1] = lbmgr->GetCommDataSz();

  CkCallback cb;
  if (concurrent)
    cb = CkCallback(CkReductionTarget(CentralLB, ReceiveCounts), thisProxy); // every PE receives counts
  else
    cb = CkCallback(CkReductionTarget(CentralLB, ReceiveCounts), thisProxy[0]);
  contribute(2*sizeof(int), counts, CkReduction::sum_int, cb);
  reduction_started = true;
#else
  SendStats();
#endif
#endif
}

#if defined(TEMP_LDB)
static int  cpufreq_sysfs_write (
                     const char *setting,int proc
                     )
{
char path[100];
snprintf(path,sizeof(path),"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",proc);
                FILE *fd = fopen (path, "w");

                if (!fd) {
                        printf("PROC#%d ooooooo666 FILE OPEN ERROR file=%s\n",CkMyPe(),path);
                        return -1;
                }
//                else CkPrintf("PROC#%d opened freq file=%s\n",proc,path);

        fseek ( fd , 0 , SEEK_SET );
        int numw=fprintf (fd, setting);
        if (numw <= 0) {

                fclose (fd);
                printf("FILE WRITING ERROR\n");
                return 0;
        }
//        else CkPrintf("Freq for Proc#%d set to %s numw=%d\n",proc,setting,numw);
        fclose(fd);
        return 1;
}


static int cpufreq_sysfs_read (int proc)
{
        FILE *fd;
        char path[100];
        int i=proc;
        snprintf(path,sizeof(path),"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",i);

        fd = fopen (path, "r");

        if (!fd) {
                printf("33 FILE OPEN ERROR file=%s\n",path);
                return 0;
        }
        char val[10];
        fgets(val,10,fd);
        int ff=atoi(val);
        fclose (fd);

        return ff;
}

float CentralLB::getTemp(int cpu)
{
        char val[10];
        FILE *f;
                char path[100];
                snprintf(path,sizeof(path),"/sys/devices/platform/coretemp.%d/temp1_input",cpu);
                f=fopen(path,"r");
                if (!f) {
                        printf("777 FILE OPEN ERROR file=%s\n",path);
                        exit(0);
                }

        if(f==NULL) {printf("ddddddddddddddddddddddddddd\n");exit(0);}
        fgets(val,10,f);
        fclose(f);
        return atof(val)/1000;
}
#endif


// called only on 0 (or every PE if concurrent=true)
void CentralLB::ReceiveCounts(int *counts, int n)
{
  if (!concurrent) CmiAssert(CkMyPe() == 0);
  if (statsData == NULL) statsData = new LDStats;

    // check that only 2 counts are sent
  CmiAssert(n == 2);
  int n_objs = counts[0];
  int n_comm = counts[1];

    // resize database
  statsData->objData.reserve(n_objs);
  statsData->from_proc.reserve(n_objs);
  statsData->to_proc.reserve(n_objs);
  statsData->commData.reserve(n_comm);

  DEBUGF(("[%d] ReceiveCounts: n_objs:%d n_comm:%d\n",CkMyPe(), n_objs, n_comm));
	
  if (concurrent) {
    CkCallback cb = CkCallback(CkReductionTarget(CentralLB, SendStats), thisProxy);
    contribute(cb);
  }
  else thisProxy.SendStats(); // broadcast call to let everybody start to send stats
}

void CentralLB::BuildStatsMsg()
{
#if CMK_LBDB_ON
  // build and send stats
  const int osz = lbmgr->GetObjDataSz();
  const int csz = lbmgr->GetCommDataSz();

  CLBStatsMsg* msg = new CLBStatsMsg(osz, csz);
  _MEMCHECK(msg);
  msg->from_pe = CkMyPe();
  //msg->serial = CrnRand();

#if CMK_LB_CPUTIMER
  lbmgr->GetTime(&msg->total_walltime,&msg->total_cputime,
		   &msg->idletime, &msg->bg_walltime,&msg->bg_cputime);
#else
  lbmgr->GetTime(&msg->total_walltime,&msg->total_walltime,
		   &msg->idletime, &msg->bg_walltime,&msg->bg_walltime);
#endif
#if defined(TEMP_LDB)
	float mytemp=getTemp(CkMyPe()%physicalCoresPerNode);
	int freq=cpufreq_sysfs_read (CkMyPe()%logicalCoresPerNode);
	msg->pe_temp=mytemp;
	msg->pe_speed=freq;
#else
  msg->pe_speed = myspeed;
#endif

#if CMK_CUDA || CMK_HIP
  // printf("CMK_CUDA setting device is %ld\n", hapiMyDevice());
  msg->gpu_device_id = hapiMyDevice();
  size_t freeMem, totalMem;
  hapiMemGetInfo(&freeMem, &totalMem);
  msg->gpu_mem_remaining = freeMem;
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  if(csv_gpu_manager.use_shm) {
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];
    msg->pool_buff_mem_remaining = dm->get_lb_buffer_free_size();
    // printf("PE %d: GPU %ld free mem: %ld, pool buffer free mem: %ld\n", CkMyPe(), msg->gpu_device_id, msg->gpu_mem_remaining, msg->pool_buff_mem_remaining);
  } else 
  {
    msg->pool_buff_mem_remaining = 0;//// should not run
  }
  // printf("msg->gpu_device_id is %ld\n", msg->gpu_device_id);
#endif

  DEBUGF(("Processor %d Total time (wall,cpu) = %f Idle = %f Bg = %f\n", CkMyPe(),msg->total_walltime,msg->idletime,msg->bg_walltime));

  msg->objData.resize(osz);
  lbmgr->GetObjData(msg->objData.data());
  msg->commData.resize(csz);
  lbmgr->GetCommData(msg->commData.data());
//  lbmgr->ClearLoads();
  DEBUGF(("PE %d BuildStatsMsg %d objs, %d comm\n",CkMyPe(),msg->objData.size(),msg->commData.size()));

  if(CkMyPe() == cur_ld_balancer) {
    int count_avail = 0;
    lbmgr->get_avail_vector(msg->avail_vector);
    msg->next_lb = LBManagerObj()->new_lbbalancer();
  }

  CmiAssert(statsMsg == NULL);
  statsMsg = msg;
#endif
}


// called on every processor
void CentralLB::SendStats()
{
#if CMK_LBDB_ON
  CmiAssert(statsMsg != NULL);
  reduction_started = false;

#if USE_LDB_SPANNING_TREE
  if(CkNumPes()>1024)
  {
    if (CkMyPe() == cur_ld_balancer)
      thisProxy[CkMyPe()].ReceiveStats(statsMsg);
    else
      thisProxy[CkMyPe()].ReceiveStatsViaTree(statsMsg);
  }
  else
#endif
  {
    DEBUGF(("[%d] calling ReceiveStats on step %d \n",CmiMyPe(),step()));
    thisProxy[cur_ld_balancer].ReceiveStats(statsMsg);
  }

  statsMsg = NULL;


  {
  // enfore the barrier to wait until centralLB says no
  LDOMHandle h;
  h.id.id.idx = 0;
  lbmgr->RegisteringObjects(h);
  }
#endif
}


void CentralLB::Migrated(int waitBarrier)
{

#if CMK_LBDB_ON
  if (waitBarrier) {
	    migrates_completed++;
      DEBUGF(("[%d] An object migrated! %d %d\n",CkMyPe(),migrates_completed,migrates_expected));
    if (migrates_completed == migrates_expected) {
      MigrationDone(1);
    }
  }
  else {
    future_migrates_completed ++;
    DEBUGF(("[%d] An object migrated with no barrier! %d expected: %d\n",CkMyPe(),future_migrates_completed,future_migrates_expected));
    if (future_migrates_completed == future_migrates_expected)  {
	CheckMigrationComplete();
    }
  }
#endif
}

void CentralLB::MissMigrate(int waitForBarrier)
{
  Migrated(waitForBarrier);
}

// build a complete data from bufferred messages
// not used when USE_REDUCTION = 1
void CentralLB::buildStats()
{
    // copy all data in individual messages to this big structure
    // Space has already been reserved in ReceiveStats
    for (int pe=0; pe<CkNumPes(); pe++) {
       CLBStatsMsg *msg = statsMsgsList[pe];
       if(msg == NULL) continue;
       statsData->objData.insert(statsData->objData.end(), msg->objData.begin(), msg->objData.end());
       statsData->from_proc.insert(statsData->from_proc.end(), msg->objData.size(), pe);
       statsData->to_proc.insert(statsData->to_proc.end(), msg->objData.size(), pe);
       statsData->commData.insert(statsData->commData.end(), msg->commData.begin(), msg->commData.end());
       // free the memory
       delete msg;
       statsMsgsList[pe]=0;
    }
    statsData->n_migrateobjs =
        std::count_if(statsData->objData.begin(), statsData->objData.end(),
                      [](const LDObjData& data) { return data.migratable; });
}

// deposit one processor data at a time, note database is pre-allocated
// to have enough space
// used when USE_REDUCTION = 1
void CentralLB::depositData(CLBStatsMsg *m)
{
  if (m == NULL) return;

  const int n_objs = m->objData.size();
  const int n_comm = m->commData.size();

  const int pe = m->from_pe;
  struct ProcStats &procStat = statsData->procs[pe];
#if defined(TEMP_LDB)
	procStat.pe_temp=m->pe_temp;
	procStat.pe_speed=m->pe_speed;
#endif

  procStat.pe = pe;
  procStat.total_walltime = m->total_walltime;
  procStat.idletime = m->idletime;
  procStat.bg_walltime = m->bg_walltime;
#if CMK_LB_CPUTIMER
  procStat.total_cputime = m->total_cputime;
  procStat.bg_cputime = m->bg_cputime;
#endif
  procStat.pe_speed = m->pe_speed;
#if CMK_CUDA || CMK_HIP
  procStat.gpu_device_id = m->gpu_device_id;
  procStat.gpu_mem_remaining = m->gpu_mem_remaining;
  procStat.pool_buff_mem_remaining = m->pool_buff_mem_remaining;
#endif

  //procStat.utilization = 1.0;
  procStat.available = true;
  procStat.n_objs = n_objs;

  CmiAssert(statsData->objData.size() + n_objs <= statsData->objData.capacity());
  statsData->objData.insert(statsData->objData.end(), m->objData.begin(), m->objData.end());
  statsData->from_proc.insert(statsData->from_proc.end(), n_objs, pe);
  statsData->to_proc.insert(statsData->to_proc.end(), n_objs, pe);

  CmiAssert(statsData->commData.size() + n_comm <= statsData->commData.capacity());
  statsData->commData.insert(statsData->commData.end(), m->commData.begin(), m->commData.end());

  statsData->n_migrateobjs +=
      std::count_if(m->objData.begin(), m->objData.end(),
                    [](const LDObjData& data) { return data.migratable; });
  delete m;
}

void CentralLB::ReceiveStatsFromRoot(CkMarshalledCLBStatsMessage &&msg) {
#if CMK_LBDB_ON
  if (CkMyPe() == cur_ld_balancer) return;
  else ReceiveStats(std::move(msg));
#endif
}

void CentralLB::ReceiveStats(CkMarshalledCLBStatsMessage &&msg)
{
#if CMK_LBDB_ON
  if (concurrent && (CkMyPe() == cur_ld_balancer)) {
    thisProxy.ReceiveStatsFromRoot(msg);  // broadcast stats to all other PEs
  }

  if (statsMsgsList == NULL) {
    statsMsgsList = new CLBStatsMsg*[CkNumPes()];
    CmiAssert(statsMsgsList != NULL);
    for(int i=0; i < CkNumPes(); i++)
      statsMsgsList[i] = 0;
  }
  if (statsData == NULL) statsData = new LDStats;

    //  loop through all CLBStatsMsg in the incoming msg
  int count = msg.getCount();
  int n_objs = 0, n_comm = 0;
  for (int num = 0; num < count; num++) 
  {
    CLBStatsMsg *m = msg.getMessage(num);
    CmiAssert(m!=NULL);
    const int pe = m->from_pe;
    const int msg_n_objs = m->objData.size();
    DEBUGF(("Stats msg received, %d %d %d %p step %d\n", pe,stats_msg_count,msg_n_objs,m,step()));
	
    if (!m->avail_vector.empty()) {
      LBManagerObj()->set_avail_vector(m->avail_vector, m->next_lb);
    }

    if (statsMsgsList[pe] != 0) {
      CkPrintf("*** Unexpected CLBStatsMsg in ReceiveStats from PE %d ***\n",
	     pe);
    } else {
      statsMsgsList[pe] = m;
#if USE_REDUCTION
      depositData(m);
#else
      // store per processor data right away
      struct ProcStats &procStat = statsData->procs[pe];
      procStat.pe = pe;
      procStat.total_walltime = m->total_walltime;
      procStat.idletime = m->idletime;
      procStat.bg_walltime = m->bg_walltime;
#if CMK_LB_CPUTIMER
      procStat.total_cputime = m->total_cputime;
      procStat.bg_cputime = m->bg_cputime;
#endif
      procStat.pe_speed = m->pe_speed;
#if CMK_CUDA || CMK_HIP
      procStat.gpu_device_id = m->gpu_device_id;
      procStat.gpu_mem_remaining = m->gpu_mem_remaining;
      procStat.pool_buff_mem_remaining = m->pool_buff_mem_remaining;
#endif
      //procStat.utilization = 1.0;
      procStat.available = true;
      procStat.n_objs = msg_n_objs;

      n_objs += msg_n_objs;
      n_comm += m->commData.size();
#if defined(TEMP_LDB)
			procStat.pe_temp=m->pe_temp;
			procStat.pe_speed=m->pe_speed;
#endif
#endif

      stats_msg_count++;
    }
  }    // end of for

#if ! USE_REDUCTION
  statsData->objData.reserve(n_objs);
  statsData->from_proc.reserve(n_objs);
  statsData->to_proc.reserve(n_objs);
  statsData->commData.reserve(n_comm);
#endif

  const int clients = CkNumPes();

  DEBUGF(("THIS POINT count = %d, clients = %d\n",stats_msg_count,clients));

  if (stats_msg_count == clients) {
	DEBUGF(("[%d] All stats messages received \n",CmiMyPe()));
    statsData->procs.resize(stats_msg_count);
    if (use_thread)
        thisProxy[CkMyPe()].t_LoadBalance();
    else
        thisProxy[CkMyPe()].LoadBalance();
  }
#endif
}

/** added by Abhinav for receiving msgs via spanning tree */
void CentralLB::ReceiveStatsViaTree(CkMarshalledCLBStatsMessage &&msg)
{
#if CMK_LBDB_ON
	CmiAssert(CkMyPe() != 0);
	bufMsg.add(std::move(msg));         // buffer messages
	count_msgs++;
	//CkPrintf("here %d\n", CkMyPe());
	if (count_msgs == st.numChildren+1) {
		if(st.parent == 0)
		{
			thisProxy[0].ReceiveStats(bufMsg);
			//CkPrintf("from %d\n", CkMyPe());
		}
		else
			thisProxy[st.parent].ReceiveStatsViaTree(bufMsg);
		count_msgs = 0;
                bufMsg.free();
	} 
#endif
}

#if CMK_REPLAYSYSTEM
static LDHandle *loadBalancer_pointers;
#endif

void CentralLB::LoadBalance()
{
#if CMK_LBDB_ON
  int proc;
  const int clients = CkNumPes();

#if ! USE_REDUCTION
  // build data
  buildStats();
#else
  for (proc = 0; proc < clients; proc++) statsMsgsList[proc] = NULL;
#endif

  lbmgr->ResetAdaptive();
  if (!_lb_args.samePeSpeed()) statsData->normalize_speed();

  if (_lb_args.debug() && (CkMyPe() == cur_ld_balancer))
      CmiPrintf("\nCharmLB> %s: PE [%d] step %d starting at %f Memory: %f MB\n",
		  lbname, cur_ld_balancer, step(), start_lb_time,
		  CmiMemoryUsage()/(1024.0*1024.0));

  // if we are in simulation mode read data
  if (LBSimulation::doSimulation) simulationRead();

  const char *availVector = lbmgr->availVector();
  for(proc = 0; proc < clients; proc++)
      statsData->procs[proc].available = (bool)availVector[proc];


  removeCommDataOfDeletedObjs(statsData);
  preprocess(statsData);

  // Snapshot for the elastic scheduler. Everything it needs is already in
  // statsData, so this costs a pass over the stats and no communication.
  CkTelemetryRecord(statsData);

//    CkPrintf("Before Calling Strategy\n");

  if (_lb_args.printSummary()) {
      LBInfo info(clients);
        // not take comm data
      info.getInfo(statsData, clients, 0);
      LBRealType mLoad, mCpuLoad, totalLoad;
      info.getSummary(mLoad, mCpuLoad, totalLoad);
      int nmsgs, nbytes;
      statsData->computeNonlocalComm(nmsgs, nbytes);
      CkPrintf("[%d] Load Summary (before LB): max (with bg load): %f max (obj only): %f average: %f at step %d nonlocal: %d msgs %.2fKB.\n", CkMyPe(), mLoad, mCpuLoad, totalLoad/clients, step(), nmsgs, 1.0*nbytes/1024);
//      if (_lb_args.debug() > 1) {
//        for (int i=0; i<statsData->n_objs; i++)
//          CmiPrintf("[%d] %.10f %.10f\n", i, statsData->objData[i].minWall, statsData->objData[i].maxWall);
//      }
  }
  
  applyLoadSnapshot(statsData);

  storedMigrateMsg = Strategy(statsData);

  if (!concurrent) ApplyDecision(); // immediately apply the migration decision
#endif
}

void CentralLB::ApplyDecision() {
#if CMK_LBDB_ON
  const int clients = CkNumPes();

  LBMigrateMsg *migrateMsg;
  if (concurrent) {
    migrateMsg = createMigrateMsg(statsData);
    if (_lb_args.debug()) printStrategyStats(migrateMsg);
  } else {
    migrateMsg = storedMigrateMsg;
    storedMigrateMsg = NULL;
  }

#if CMK_SHRINK_EXPAND
  // Nothing may be left on a processor the balancer was told is unavailable.
  // On a rescale that PE is about to exit, and survivors keep their elements
  // in memory across the longjmp while a departing PE simply goes away: there
  // is no restore path for anything still on it, so it is lost silently. A
  // strategy is supposed to place nothing there, but one can leave an object
  // where it already was, and a request that races the step can arrive with
  // the placement already made. Redirect anything landing on an unavailable
  // PE and rebuild the migrate message.
  {
    // Availability comes from the stats every PE holds, not from the
    // PE-0-only bitmap state: ApplyDecision runs on whichever PE produced the
    // winning solution, which is usually not PE 0, and there the bitmap state
    // reads as "no rescale pending" and the backstop would quietly do nothing.
    auto doomed = [&](int pe) {
      return pe >= 0 && pe < (int)statsData->procs.size() &&
             !statsData->procs[pe].available;
    };
    std::vector<int> survivors;
    for (int p = 0; p < CkNumPes(); p++)
      if (!doomed(p)) survivors.push_back(p);
    int redirected = 0, unmovable = 0;
    if (!survivors.empty()) {
      size_t rr = 0;
      for (size_t i = 0; i < statsData->objData.size(); i++) {
        if (doomed(statsData->to_proc[i])) {
          if (statsData->objData[i].migratable) {
            statsData->to_proc[i] = survivors[rr++ % survivors.size()];
            redirected++;
          } else {
            unmovable++;
          }
        }
      }
    }
    if (redirected || unmovable) {
      CkPrintf("[%d] CharmLB> redirected %d object(s) off unavailable PEs "
               "(%d unmigratable left behind)\n",
               CkMyPe(), redirected, unmovable);
      delete migrateMsg;
      migrateMsg = createMigrateMsg(statsData);
    }
  }
#endif

#if CMK_REPLAYSYSTEM
  CpdHandleLBMessage(&migrateMsg);
#endif
  
  LBManagerObj()->get_avail_vector(migrateMsg->avail_vector);
  migrateMsg->next_lb = LBManagerObj()->new_lbbalancer();

  // if this is the step at which we need to dump the database
  simulationWrite();

//  calculate predicted load
//  very time consuming though, so only happen when debugging is on
  if (_lb_args.printSummary()) {
      LBInfo info(clients);
        // not take comm data
      getPredictedLoadWithMsg(statsData, clients, migrateMsg, info, 0);
      LBRealType mLoad, mCpuLoad, totalLoad;
      info.getSummary(mLoad, mCpuLoad, totalLoad);
      int nmsgs, nbytes;
      statsData->computeNonlocalComm(nmsgs, nbytes);
      CkPrintf("[%d] Load Summary (after LB): max (with bg load): %f max (obj only): %f average: %f at step %d nonlocal: %d msgs %.2fKB useMem: %.2fKB.\n", CkMyPe(), mLoad, mCpuLoad, totalLoad/clients, step(), nmsgs, 1.0*nbytes/1024, (1.0*useMem())/1024);
      for (int i=0; i<clients; i++)
        migrateMsg->expectedLoad[i] = info.peLoads[i];
  }

  DEBUGF(("[%d]calling recv migration\n",CkMyPe()));

#if CMK_SCATTER_LB_RESULTS
  InitiateScatter(migrateMsg);
#else
  if (1) {
      // broadcast
    thisProxy.ReceiveMigration(migrateMsg);
  }
  else {
    // split the migration for each processor
    for (int p=0; p<CkNumPes(); p++) {
      LBMigrateMsg *m = extractMigrateMsg(migrateMsg, p);
      thisProxy[p].ReceiveMigration(m);
    }
    delete migrateMsg;
  }
#endif
  // Zero out data structures for next cycle
  // CkPrintf("zeroing out data\n");
  statsData->clear();
  stats_msg_count=0;
#endif
}

void CentralLB::t_LoadBalance()
{
    LoadBalance();
}

void CentralLB::InitiateScatter(LBMigrateMsg *msg) {

  if (CkNumPes() <= broadcastThreshold) {
    thisProxy.ReceiveMigration(msg);
    return;
  }

  int middlePe = CkNumPes() / 2;

  // allocate maximum possible size to avoid later copies
  // the messages will be resized before sending
  LBScatterMsg *leftMsg = new (middlePe, msg->n_moves)
    LBScatterMsg(0, middlePe - 1);
  LBScatterMsg *rightMsg = new (CkNumPes() - middlePe, msg->n_moves)
    LBScatterMsg(middlePe, CkNumPes() - 1);

  int *migrateTally = new int[CkNumPes()];
  memset(migrateTally, 0, CkNumPes() * sizeof(int));

  for (int i = 0; i < msg->n_moves; i++) {
    MigrateInfo* item = (MigrateInfo*) &msg->moves[i];
    migrateTally[item->to_pe]++;
    if (item->from_pe < middlePe) {
      leftMsg->moves[leftMsg->numMigrates++] = *item;
    }
    else {
      rightMsg->moves[rightMsg->numMigrates++] = *item;
    }
  }

  memcpy(leftMsg->numMigratesPerPe, migrateTally, middlePe * sizeof(int));
  memcpy(rightMsg->numMigratesPerPe, &migrateTally[middlePe], (CkNumPes() - middlePe) * sizeof(int));

  delete [] migrateTally;

  // shrink the size of the messages
  envelope *env = UsrToEnv(rightMsg);
  env->shrinkUsersize((msg->n_moves - rightMsg->numMigrates) * sizeof(MigrateDecision));

  // left message is not getting sent yet, but better resize it now
  // before we lose track of its original size
  env = UsrToEnv(leftMsg);
  env->shrinkUsersize((msg->n_moves - leftMsg->numMigrates) * sizeof(MigrateDecision));

  // send out results for right half of PEs first
  // to overlap communication with computation
  thisProxy[middlePe].ScatterMigrationResults(rightMsg);

  delete msg;
  ScatterMigrationResults(leftMsg);
}

void CentralLB::ScatterMigrationResults(LBScatterMsg *msg) {

  int finished = false;
  do {
    CkAssert(msg->firstPeInSpan == CkMyPe());
    int numPesInSpan = msg->lastPeInSpan - msg->firstPeInSpan + 1 ;

    if (numPesInSpan <= broadcastThreshold) {
      for (int i = msg->firstPeInSpan; i < msg->lastPeInSpan; i++) {
        // TODO: multicast without allocating new message each time
        LBScatterMsg *msgCopy = new (numPesInSpan, msg->numMigrates)
          LBScatterMsg(msg->firstPeInSpan, msg->lastPeInSpan);
        msgCopy->numMigrates = msg->numMigrates;
        memcpy(msgCopy->numMigratesPerPe, msg->numMigratesPerPe,
               numPesInSpan * sizeof(int));
        memcpy(msgCopy->moves, msg->moves,
               msg->numMigrates * sizeof(MigrateDecision));
        thisProxy[i].ReceiveMigration(msgCopy);
      }
      // use original message for last send
      thisProxy[msg->lastPeInSpan].ReceiveMigration(msg);
      finished = true;
    }
    else {
      int middlePe = (msg->firstPeInSpan + msg->lastPeInSpan + 1) / 2;
      // reuse received message, taking care not to overwrite needed data
      LBScatterMsg *leftMsg = msg;
      int numMigrates = leftMsg->numMigrates;
      int numPesInRightSpan = leftMsg->lastPeInSpan - middlePe + 1;
      LBScatterMsg *rightMsg =
        new (numPesInRightSpan, leftMsg->numMigrates)
        LBScatterMsg(middlePe, leftMsg->lastPeInSpan);
      leftMsg->numMigrates = 0;
      leftMsg->lastPeInSpan = middlePe - 1;
      for (int i = 0; i < numMigrates; i++) {
        if (leftMsg->moves[i].fromPe < middlePe) {
          leftMsg->moves[leftMsg->numMigrates++] = leftMsg->moves[i];
        }
        else {
          rightMsg->moves[rightMsg->numMigrates++] = leftMsg->moves[i];
        }
      }

      memcpy(rightMsg->numMigratesPerPe,
             &leftMsg->numMigratesPerPe[middlePe - leftMsg->firstPeInSpan],
             (numPesInRightSpan) * sizeof(int));

      // shrink the size of the messages
      envelope *env = UsrToEnv(rightMsg);
      env->shrinkUsersize((numMigrates - rightMsg->numMigrates)
                          * sizeof(MigrateDecision));

      // left message is not getting sent yet, but better resize it now
      // before we lose track of its original size
      env = UsrToEnv(leftMsg);
      env->shrinkUsersize((numMigrates - leftMsg->numMigrates)
                          * sizeof(MigrateDecision));

      thisProxy[middlePe].ScatterMigrationResults(rightMsg);
    }

  } while (!finished);

}

// test if sender and receiver in a commData is nonmigratable.
static bool isMigratable(LDObjData **objData, int *len, int count, const LDCommData &commData)
{
#if CMK_LBDB_ON
  for (int pe=0 ; pe<count; pe++)
  {
    for (int i=0; i<len[pe]; i++)
      if (objData[pe][i].objID() == commData.sender.objID() ||
          objData[pe][i].objID() == commData.receiver.get_destObj().objID())
      return false;
  }
#endif
  return true;
}

// rebuild LDStats and remove all non-migratble objects and related things
void CentralLB::removeNonMigratable(LDStats* stats, int count)
{
  int i;

  // check if we have non-migratable objects
  int have = 0;
  for (const auto& odata : stats->objData)
  {
    if (!odata.migratable) {
      have = 1; break;
    }
  }
  if (have == 0) return;

  std::vector<LDObjData> mig;
  std::vector<int> new_from_proc, new_to_proc;
  mig.reserve(stats->n_migrateobjs);
  new_from_proc.reserve(stats->n_migrateobjs);
  new_to_proc.reserve(stats->n_migrateobjs);
  for (i=0; i<stats->objData.size(); i++)
  {
    LDObjData &odata = stats->objData[i];
    if (odata.migratable) {
      mig.push_back(odata);
      new_from_proc.push_back(stats->from_proc[i]);
      new_to_proc.push_back(stats->to_proc[i]);
    }
    else {
      stats->procs[stats->from_proc[i]].bg_walltime += odata.wallTime;
#if CMK_LB_CPUTIMER
      stats->procs[stats->from_proc[i]].bg_cputime += odata.cpuTime;
#endif
    }
  }
  CmiAssert(stats->n_migrateobjs == mig.size());

  stats->makeCommHash();
  
  std::vector<LDCommData> newCommData;
  newCommData.reserve(stats->commData.size());
  for (auto& cdata : stats->commData)
  {
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
      if (stats->complete_flag)
        CmiAssert(idx != -1);
      else if (idx == -1) continue;          // receiver not in this group
      if (!stats->objData[idx].migratable) continue;
      break;
      }
    case LD_OBJLIST_MSG:    // object message FIXME add multicast
      break;
    }
    newCommData.push_back(cdata);
  }

  if (mig.size() != stats->objData.size())
    CmiPrintf("Removed %zu nonmigratable objs (& %zu associated comms) from total n_objs:%zu (%d migratable objs left)\n",
              stats->objData.size() - stats->n_migrateobjs,
              stats->commData.size() - newCommData.size(), stats->objData.size(),
              stats->n_migrateobjs);

  // swap to new data
  stats->objData = mig;
  stats->from_proc = new_from_proc;
  stats->to_proc = new_to_proc;

  stats->commData = newCommData;

  stats->deleteCommHash();
  stats->makeCommHash();

}



void CentralLB::ReceiveMigration(LBScatterMsg *m) {
  if (concurrent) {
    if (CkMyPe() == 0) lbmgr->SetStrategyCost(CkWallTimer() - strat_start_time);
    // Zero out data structures for next cycle
    statsData->clear();
    stats_msg_count=0;
  }
  storedMigrateMsg = NULL;
  storedScatterMsg = m;
#if CMK_MEM_CHECKPOINT
  CkResetInLdb();
#endif
  contribute(CkCallback(CkReductionTarget(CentralLB, ProcessMigrationDecision),
              thisProxy));

}

void CentralLB::ReceiveMigration(LBMigrateMsg *m)
{
  if (concurrent) {
    if (CkMyPe() == 0) lbmgr->SetStrategyCost(CkWallTimer() - strat_start_time);
    // Zero out data structures for next cycle
    statsData->clear();
    stats_msg_count=0;
  }
  storedMigrateMsg = m;
#if CMK_MEM_CHECKPOINT
  CkResetInLdb();
#endif
  contribute(CkCallback(CkReductionTarget(CentralLB, ProcessReceiveMigration),
              thisProxy));
}

void CentralLB::ProcessMigrationDecision() {
#if CMK_LBDB_ON
  LBScatterMsg *m = storedScatterMsg;
  CkAssert(m != NULL);

  migrates_expected = m->numMigratesPerPe[CkMyPe() - m->firstPeInSpan];
  future_migrates_expected = 0;

  for(int i = 0; i < m->numMigrates; i++) {
    MigrateDecision& move = m->moves[i];
    const int me = CkMyPe();
    if (move.fromPe == me) {
      if (move.toPe == me) {
        CkAbort("[%d] Error, attempting to migrate from myself to myself\n",
            CkMyPe());
      }
      DEBUGF(("[%d] migrating object to %d\n", move.fromPe, move.toPe));
      // migrate object, in case it is already gone, inform toPe
      LDObjHandle objInfo = lbmgr->GetObjHandle(move.dbIndex);

      if (lbmgr->Migrate(objInfo,move.toPe) == 0) {
        CkAbort("Error: Async arrival not supported in scattering mode\n");
      }
    }
  }

  if (migrates_expected == 0 || migrates_completed == migrates_expected) {
    MigrationDone(1);
  }

  delete m;
#endif
}

void CentralLB::ProcessReceiveMigration()
{
  // CmiPrintf("[%d] ProcessReceiveMigration\n", CkMyPe());
#if CMK_LBDB_ON
	int i;
        LBMigrateMsg *m = storedMigrateMsg;
        CmiAssert(m!=NULL);

  if (_lb_args.debug() > 1) 
    if (CkMyPe()%1024==0) CmiPrintf("[%d] Starting ReceiveMigration step %d at %f\n",CkMyPe(),step(), CmiWallTimer());

  for (i=0; i<CkNumPes(); i++) lbmgr->lastLBInfo.expectedLoad[i] = m->expectedLoad[i];
  CmiAssert(migrates_expected <= 0 || migrates_completed == migrates_expected);
  migrates_expected = 0;
  future_migrates_expected = 0;
  // CmiPrintf("[%d] ProcessReceiveMigration: n_moves=%d\n", CkMyPe(), m->n_moves);
  for(i=0; i < m->n_moves; i++) {
    MigrateInfo& move = m->moves[i];
    const int me = CkMyPe();
    if (move.from_pe == me && move.to_pe != me) {
#if CMK_DRONE_MODE
      int to_pe_rank0 = CMK_RANK_0(move.to_pe);
      if(move.from_pe == to_pe_rank0) continue;
      move.to_pe = to_pe_rank0;
#endif

      DEBUGF(("[%d] migrating object to %d\n",move.from_pe,move.to_pe));
      // migrate object, in case it is already gone, inform toPe
      if (lbmgr->Migrate(move.obj,move.to_pe) == 0)
         thisProxy[move.to_pe].MissMigrate(!move.async_arrival);
    } else if (move.from_pe != me && move.to_pe == me) {
#if CMK_DRONE_MODE
      int to_pe_rank0 = CMK_RANK_0(move.to_pe);
      if(me != to_pe_rank0) continue;
#endif
       DEBUGF(("[%d] expecting object from %d\n",move.to_pe,move.from_pe));
      if (!move.async_arrival) migrates_expected++;
      else future_migrates_expected++;
    }
    else {
      #if CMK_GLOBAL_LOCATION_UPDATE
      // CmiPrintf("[%d] Updating location for obj id=%llu from %d to %d\n", CkMyPe(), move.obj.id, move.from_pe, move.to_pe);
        UpdateLocation(move);
      #endif
    }
  }

  DEBUGF(("[%d] in ReceiveMigration %d moves expected: %d future expected: %d\n",CkMyPe(),m->n_moves, migrates_expected, future_migrates_expected));
  // if (_lb_debug) CkPrintf("[%d] expecting %d objects migrating.\n", CkMyPe(), migrates_expected);



#if 0
  if (m->n_moves ==0) {
    lbmgr->SetLBPeriod(lbmgr->GetLBPeriod()*2);
  }
#endif
  cur_ld_balancer = m->next_lb;
  if((CkMyPe() == cur_ld_balancer) && (cur_ld_balancer != 0)){
      LBManagerObj()->set_avail_vector(m->avail_vector, -2);
  }

  if (migrates_expected == 0 || migrates_completed == migrates_expected)
    MigrationDone(1);
  delete m;
#endif
}

void CentralLB::CheckForLB() {
  //sleep(5);
#if CMK_SHRINK_EXPAND
  // Concurrent-round guard. CentralLB has one statsData and one
  // stats_msg_count: two interleaved rounds corrupt both (observed as the
  // ProcessReceiveMigration SEGV when two rounds overlapped). The rescale
  // round itself never passes through here in barrier-less mode (it starts
  // via StartLB -> ProcessAtSync), so under the flag ANY arrival while a
  // rescale is pending or in flight is a racing regular step -- drop it: a
  // periodic step re-arms via setTimer, and chares held at a racing AtSync
  // barrier are released by the post-restore ResumeClientsIfHeld. In
  // boundary mode the *_MSG_RECEIVED round IS the rescale vehicle and must
  // flow; only rounds racing the post-decision drain (*_IN_PROGRESS, the
  // early-release window) are dropped.
  {
    extern bool CkRescaleBarrierlessEnabled();
    const bool racing = CkRescaleBarrierlessEnabled()
        ? (pending_realloc_state != NO_REALLOC)
        : (pending_realloc_state &
           (SHRINK_IN_PROGRESS | EXPAND_IN_PROGRESS)) != 0;
    if (racing)
    {
      CkPrintf("CharmLB> Deferring a regular LB step: a rescale is in "
               "flight.\n");
      return;
    }
  }
  if (pending_realloc_state == EXPAND_MSG_RECEIVED)
    CheckForRealloc();
  //else if (pending_realloc_state == NO_REALLOC)
  //  thisProxy.ResumeClients(0);
  else
    thisProxy.CallLB();
#else
  // if we are not in shrink/expand mode, just call LB
  thisProxy.CallLB();
#endif
  //else
  //  thisProxy.ResumeClients(0);
}

// We assume that bit vector would have been aptly set async by either scheduler or charmrun.
void CentralLB::CheckForRealloc(){
#if CMK_SHRINK_EXPAND
  LB_TRACE("[%d] LB CheckForRealloc (pending_realloc_state=%d)\n", CkMyPe(),
           (int)pending_realloc_state);
  if(pending_realloc_state != NO_REALLOC) {
    pending_realloc_state = (pending_realloc_state == SHRINK_MSG_RECEIVED) ? SHRINK_IN_PROGRESS : EXPAND_IN_PROGRESS; //in progress
    CkPrintf("Load balancer invoking charmrun to handle reallocation on pe %d\n", CkMyPe());
    double end_lb_time = CkWallTimer();
    CkPrintf("CharmLB> %s: PE [%d] step %d finished at %f duration %f s\n\n",
        lbname, cur_ld_balancer, step()-1, end_lb_time,	end_lb_time-start_lb_time);
    // On a barrier-less round the application is still running: the doomed
    // PEs hold in-flight state their evacuated elements left behind (queued
    // ghosts to forward, reduction partials to flush upward), and cutting now
    // loses it. Let it land first -- everything keeps running through the
    // grace, and the cut fires afterwards. A boundary round needs no grace:
    // the application is quiescent at the cut by construction.
    extern bool _rescaleBarrierlessRound;
    extern int _rescaleGraceMs;
    // No expand-populate hook needed here: the rescale checkpoint path sets
    // _rescaleResumeCb = LBManager::StartLB() for expands, which runs the
    // populating round after the restore. (A second kick from here raced
    // that round's migrations -- stats built mid-flight, SEGV applying the
    // decisions.)
    if (_rescaleBarrierlessRound && _rescaleGraceMs > 0)
    {
      _rescaleBarrierlessRound = false;
      CkPrintf("CharmLB> Barrier-less rescale: quiet-probe drain (ceiling %d ms) "
               "before the cut.\n", _rescaleGraceMs);
      StartRescaleQuietWatch();
      return;
    }
    // Boundary-mode early release (shrink only): the LB decision is made and
    // the evacuation migrations are complete, so nothing the application does
    // from here on changes the rescale. Resume every chare now -- survivors
    // keep iterating through the drain and cut; only real data dependencies
    // (a ghost owed by a just-migrated neighbor) pace anyone. Expand keeps
    // the hold: its populate round runs post-restore and resumes clients at
    // its own end. The reduction handshake in RescaleEarlyResume guarantees
    // every PE resumed before the quiet watch starts.
    extern bool _rescaleHoldBoundary;
    if (!_rescaleHoldBoundary && pending_realloc_state == SHRINK_IN_PROGRESS &&
        _rescaleGraceMs > 0)
    {
      CkPrintf("CharmLB> Boundary rescale: early release -- resuming clients "
               "before the drain and cut.\n");
      thisProxy.RescaleEarlyResume();
      return;
    }
    // do checkpoint
    CkCallback cb(CkIndex_CentralLB::RescaleCutArmed(), thisProxy[0]);
    CkArmRescaleCut(_shrinkexpand_basedir, cb, se_avail_snapshot);
  } else {
    thisProxy.MigrationDoneImpl(1);
  }
#endif
}

#if CMK_SHRINK_EXPAND
// The flat grace timer guessed how long the doomed PEs needed to finish
// forwarding what their evacuated elements left behind. This measures it
// instead: traffic toward doomed PE d has drained exactly when the summed
// per-destination send counters for d equal d's own arrival counter, stable
// across two consecutive probes (the stability round absorbs samples taken
// while a message was between a sender's bump and the receiver's). The
// application keeps running throughout; only the cut is moved earlier. The
// grace value remains as a ceiling in case the counters never settle.
void CentralLB::RescaleEarlyResume()
{
  // LBManager's resume (not CentralLB's): lb_in_progress must stay true
  // until the post-restore reset, so a racing CCS request keeps buffering.
  lbmgr->ResumeClients();
  CkCallback cb(CkReductionTarget(CentralLB, RescaleEarlyResumeDone), thisProxy[0]);
  contribute(cb);
}

void CentralLB::RescaleEarlyResumeDone()
{
  if (_lb_args.debug())
    CkPrintf("CharmLB> Boundary rescale: clients resumed on every PE; "
             "starting the drain.\n");
  StartRescaleQuietWatch();
}

void CentralLB::StartRescaleQuietWatch()
{
  quietDoomed.clear();
  const int lim = std::min((int)se_avail_snapshot.size(), CkNumPes());
  for (int i = 0; i < lim; i++)
    if (!se_avail_snapshot[i]) quietDoomed.push_back(i);
  quietPrev.clear();
  quietMatchedPrev = false;
  quietProbes = 0;
  quietT0 = CkWallTimer();
  // Point-to-point on purpose: a broadcast reaches some PEs via relays, and a
  // relay bumps its sent-counter AFTER that PE has already sampled for this
  // round -- a persistent one-message skew that never settles. P2p sends all
  // happen here on PE 0 before PE 0's own sample (its self-probe is delivered
  // through the scheduler), so every probe message is counted on both sides
  // within the same round.
  for (int p = 0; p < CkNumPes(); p++)
    thisProxy[p].RescaleQuietProbe(quietDoomed);
}

void CentralLB::RescaleQuietProbe(std::vector<int> doomed)
{
  std::vector<long> rep(2 * doomed.size() + 1, 0);
  for (size_t k = 0; k < doomed.size(); k++)
  {
    rep[2 * k] = CmiRescaleAmSentTo(doomed[k]);
    if (CkMyPe() == doomed[k]) rep[2 * k + 1] = CmiRescaleAmRecvP2p();
  }
  // Direct point-to-point report, NOT a contribute(): the group reduction
  // tree routes kid contributions through interior PEs -- a doomed interior
  // PE among them -- generating exactly the p2p traffic toward the doomed PE
  // that the probe is trying to see drain, and each reduction's
  // ReductionStarting chatter re-arms it every round (observed: interior-
  // doom runs always hit the ceiling). Direct reports touch only
  // survivor->PE0 paths plus PE0's counted probe sends.
  thisProxy[0].RescaleQuietReport(rep.data(), (int)rep.size());
}

void CentralLB::RescaleQuietReport(long* data, int n)
{
  extern int _rescaleGraceMs;
  // Accumulate this round's per-PE reports; evaluate once every PE reported.
  if ((int)quietPrev.size() != n) quietPrev.assign(n, 0);
  for (int i = 0; i < n; i++) quietPrev[i] += data[i];
  if (++quietReports < CkNumPes()) return;
  std::vector<long> sums;
  sums.swap(quietPrev);
  quietReports = 0;
  data = sums.data();
  quietProbes++;
  bool matched = true;
  for (int k = 0; k < n / 2; k++)
    if (data[2 * k] != data[2 * k + 1]) { matched = false; break; }
  if (!matched && _lb_args.debug() > 1)
    for (int k = 0; k < n / 2; k++)
      CkPrintf("CharmLB> quiet probe %d: doomed[%d] sent=%ld recv=%ld\n",
               quietProbes, k, data[2 * k], data[2 * k + 1]);
  const double elapsedMs = (CkWallTimer() - quietT0) * 1000.0;

  // One matched round suffices: quiet is a trigger, not a safety condition --
  // the cut's global flush drains whatever is in flight regardless.
  bool fire = false;
  if (matched)
  {
    CkPrintf("CharmLB> Rescale drain: doomed PEs quiet after %.2f ms "
             "(%d probes); cutting now.\n", elapsedMs, quietProbes);
    fire = true;
  }
  else if (elapsedMs > (double)_rescaleGraceMs)
  {
    CkPrintf("CharmLB> Warning: doomed PEs not provably quiet after %.2f ms "
             "(ceiling %d ms); cutting anyway.\n", elapsedMs, _rescaleGraceMs);
    fire = true;
  }
  if (fire)
  {
    CkCallback cb(CkIndex_CentralLB::RescaleCutArmed(), thisProxy[0]);
    CkArmRescaleCut(_shrinkexpand_basedir, cb, se_avail_snapshot);
    return;
  }
  // Not settled: wait out the in-flight runtime chatter before sampling
  // again. The Ccd timer's real granularity is ~10 ms, which is fine here --
  // an unmatched first round means a message was mid-flight, and the next
  // sample only needs to land after it. Back-to-back rounds never converge
  // (see above).
  CcdCallFnAfterOnPE(
      [](void* arg, double)
      {
        CentralLB* lb = (CentralLB*)arg;
        for (int p = 0; p < CkNumPes(); p++)
          lb->thisProxy[p].RescaleQuietProbe(lb->quietDoomed);
      },
      (void*)this, 1, CkMyPe());
}
#endif

void CentralLB::RescaleCutArmed(){
#if CMK_SHRINK_EXPAND
    // Stamp the start of the post-checkpoint rescale orchestration so the
    // PE-0 print at the end of CkRestartMain can report total / restore /
    // overhead. Uses gettimeofday wall-clock (defined in ckcheckpoint.C)
    // because CmiWallTimer's epoch resets across the rescale longjmp.
    extern double rescale_overhead_start_timer;
    extern double rescale_wall_now();
    if (CkMyPe() == 0) rescale_overhead_start_timer = rescale_wall_now();
    std::vector<char> avail = se_avail_snapshot;
    //free(se_avail_vector);
    thisProxy.WillIbekilled(avail, numProcessAfterRestart);
#endif
}

void CentralLB::WillIbekilled(std::vector<char> avail, int newnumProcessAfterRestart){
#if CMK_SHRINK_EXPAND
 numProcessAfterRestart = newnumProcessAfterRestart;
 mynewpe =  GetNewPeNumber(avail);
 //CkPrintf("[%d] -> new pe %d\n", CkMyPe(), mynewpe);
 willContinue = avail[CkMyPe()];
 //CkPrintf("PE%i> Sending start cleanup reduction\n", CkMyPe());
 CkCallback cb(CkIndex_CentralLB::StartCleanup(), thisProxy[0]);
 contribute(cb);
#endif
}

void CentralLB::StartCleanup(){
#if CMK_SHRINK_EXPAND
	CkCleanup();
#endif
}

void CentralLB::MigrationDone(int balancing)
{
#if CMK_SHRINK_EXPAND
    LB_TRACE("[%d] LB MigrationDone -> contribute to CheckForRealloc\n",
             CkMyPe());
   // barrier to check for reallocation
    CkCallback cb(CkIndex_CentralLB::CheckForRealloc(), thisProxy[0]);
    contribute(cb);
	return;
#else
    MigrationDoneImpl(balancing);
#endif
}

void CentralLB::MigrationDoneImpl (int balancing)
{
#if CMK_LBDB_ON
  LB_TRACE("[%d] LB MigrationDoneImpl\n", CkMyPe());
  migrates_completed = 0;
  migrates_expected = -1;
  // clear load stats
  if (balancing) lbmgr->ClearLoads();
#if CMK_CUDA || CMK_HIP
  if (CmiMyRank() == 0)
    hapiClearCuptiData();
#endif
  // Increment to next step
  lbmgr->incStep();
	DEBUGF(("[%d] Incrementing Step %d \n",CkMyPe(),step()));
  // if sync resume, invoke a barrier


  LBManager::Object()->MigrationDone();    // call registered callbacks

  LoadbalanceDone(balancing);        // callback
  // if sync resume invoke a barrier
  if (balancing && _lb_args.syncResume()) {
    contribute(CkCallback(CkReductionTarget(CentralLB, ResumeClients),
                thisProxy));
  }
  else{
    {
	thisProxy [CkMyPe()].ResumeClients(balancing);
    }	
  }	
#if CMK_GRID_QUEUE_AVAILABLE
  CmiGridQueueDeregisterAll ();
  CpvAccess(CkGridObject) = NULL;
#endif  // if CMK_GRID_QUEUE_AVAILABLE
#endif  // if CMK_LBDB_ON
}

void CentralLB::ResumeClients()
{
  ResumeClients(1);
}

void CentralLB::ResumeClients(int balancing)
{
  LB_TRACE("[%d] LB ResumeClients(%d)\n", CkMyPe(), balancing);
#if CMK_LBDB_ON
  //CkPrintf("[%d] Resuming clients. balancing:%d.\n",CkMyPe(),balancing);
  lbmgr->ResumeClients();
  if (balancing)  {

    CheckMigrationComplete();
    if (future_migrates_expected == 0 || 
            future_migrates_expected == future_migrates_completed) {
      CheckMigrationComplete();
    }
  }
  lbmgr->lb_in_progress = false;

  if (CkMyPe() == 0)
    lbmgr->callRealloc();
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
    double end_lb_time = CkWallTimer();
    if (_lb_args.debug() && CkMyPe()==0) {
      CkPrintf("CharmLB> %s: PE [%d] step %d finished at %f duration %f s\n\n",
                lbname, CkMyPe(), step()-1, end_lb_time,
		end_lb_time-start_lb_time);
    }

    lbmgr->SetMigrationCost(end_lb_time - start_lb_time);

    lbdone = 0;
    future_migrates_expected = -1;
    future_migrates_completed = 0;


    DEBUGF(("[%d] Migration Complete\n", CkMyPe()));
    // release local barrier  so that the next load balancer can go
    LDOMHandle h;
    h.id.id.idx = 0;
    lbmgr->DoneRegisteringObjects(h);
    // switch to the next load balancer in the list
    // subtle: called from Migrated() may result in Migrated() called in next LB
    if (!(_lb_args.metaLbOn() && _lb_args.metaLbModelDir() != nullptr))
      lbmgr->nextLoadbalancer(seqno);
  }
#endif
}

// Remove edges from commData in LDStats which contains deleted elements
void CentralLB::removeCommDataOfDeletedObjs(LDStats* stats) {
  stats->makeCommHash();

  int n_comm = 0;
  for (auto& cdata : stats->commData) {
    switch (cdata.receiver.get_type()) {
      case LD_PROC_MSG:
        break;
      case LD_OBJ_MSG:  {
        if (!cdata.from_proc()) {
          int sidx = stats->getSendHash(cdata);
          int ridx = stats->getRecvHash(cdata);
          if (sidx == -1 || ridx == -1) continue;
        }
        break;
      }
      case LD_OBJLIST_MSG:  {
        int sidx = stats->getSendHash(cdata);
        if (sidx == -1) continue;
        int nobjs;
        LDObjKey *objs = cdata.receiver.get_destObjs(nobjs);
        for (int id=0; id<nobjs; id++) {
          int idx = stats->getHash(objs[id]);
          if (idx == -1)
          {
            objs[id] = objs[nobjs-1];
            id--;
            nobjs--;
          }
        }
        if(nobjs == 0) continue;
        cdata.receiver.dest.destObjs.len = nobjs;
        break;
      }
    }

    stats->commData[n_comm] = cdata;
    n_comm++;
  }

  stats->commData.resize(n_comm);
}

/** Balance on the last well-measured window, not on a short one.
 *
 * Per-object loads are accumulators since the previous round's ClearLoads,
 * so a round that fires soon after another one -- the populating round after
 * an expand, ~100 ms behind the restore; a rescale round landing just after a
 * regular step -- sees a window of a few iterations in which each chare's
 * partial current iteration is a large fraction of its total, and chares at
 * different points of their loop read as differently loaded. A rescale
 * balances the new world on that noise.
 *
 * So PE 0 keeps the per-object loads of the last window it judged well
 * measured, and a round whose own window is much shorter reads those instead
 * for every object it still knows. When this window is comparable or longer
 * it becomes the new snapshot, so a permanently shorter cadence is adopted
 * rather than pinned to history. Objects without a snapshot -- created or
 * migrated between rescales, whose LB id then differs -- keep their live
 * value. For an AtSync application this is exactly "the load at the last
 * AtSync"; for one with no boundaries it is the last long window.
 */
void CentralLB::applyLoadSnapshot(LDStats* stats)
{
  const double now = CmiWallTimer();
  const double window = (lastRoundTime > 0.0) ? (now - lastRoundTime) : 0.0;
  lastRoundTime = now;
  const bool haveSnapshot = !loadSnapshot.empty() && snapshotWindow > 0.0;
  const bool shortWindow = haveSnapshot && window < 0.5 * snapshotWindow;

  if (shortWindow)
  {
    int replaced = 0;
    for (size_t i = 0; i < stats->objData.size(); i++)
    {
      LDObjData& o = stats->objData[i];
      auto it = loadSnapshot.find(o.objID());
      if (it == loadSnapshot.end()) continue;
      o.wallTime = it->second.wall;
#if CMK_LB_CPUTIMER
      o.cpuTime = it->second.cpu;
#endif
#if CMK_CUDA || CMK_HIP
      o.gpuTime = it->second.gpu;
#endif
      replaced++;
    }
    if (_lb_args.debug())
      CkPrintf("CharmLB> step %d: window %.3fs is short; using snapshot loads (%.3fs "
               "window) for %d of %zu objects\n",
               step(), window, snapshotWindow, replaced, stats->objData.size());
    return;
  }

  // This window is as good as the last one, or there is none: record it.
  if (window > 0.0 || !haveSnapshot)
  {
    loadSnapshot.clear();
    for (size_t i = 0; i < stats->objData.size(); i++)
    {
      const LDObjData& o = stats->objData[i];
      LoadSnapshot snap;
      snap.wall = o.wallTime;
#if CMK_LB_CPUTIMER
      snap.cpu = o.cpuTime;
#else
      snap.cpu = 0.0;
#endif
#if CMK_CUDA || CMK_HIP
      snap.gpu = o.gpuTime;
#else
      snap.gpu = 0.0;
#endif
      loadSnapshot[o.objID()] = snap;
    }
    snapshotWindow = window;
  }
}

void CentralLB::preprocess(LDStats* stats)
{
  if (_lb_args.ignoreBgLoad())
    stats->clearBgLoad();

  // Call the predictor for the future
  if (_lb_predict) FuturePredictor(statsData);
}

void CentralLB::printStrategyStats(LBMigrateMsg *msg) {
#if CMK_LBDB_ON
  envelope *env = UsrToEnv(msg);

  double strat_end_time = CkWallTimer();
  double lbdbMemsize = LBManager::Object()->useMem()/1000;
  CkPrintf("CharmLB> %s: PE [%d] Memory: LBManager: %d KB CentralLB: %d KB\n",
        lbname, CkMyPe(), (int)lbdbMemsize, (int)(useMem()/1000));
  CkPrintf("CharmLB> %s: PE [%d] #Objects migrating: %d, LBMigrateMsg size: %.2f MB\n", lbname, CkMyPe(), msg->n_moves, env->getTotalsize()/1024.0/1024.0);
  CkPrintf("CharmLB> %s: PE [%d] strategy finished at %f duration %f s\n",
      lbname, CkMyPe(), strat_end_time, strat_end_time-strat_start_time);
#endif
}

// default load balancing strategy
LBMigrateMsg* CentralLB::Strategy(LDStats* stats)
{
#if CMK_LBDB_ON
  strat_start_time = CkWallTimer();
  if (_lb_args.debug() && (CkMyPe() == cur_ld_balancer))
    CkPrintf("CharmLB> %s: PE [%d] strategy starting at %f\n", lbname, cur_ld_balancer, strat_start_time);

  work(stats);


  if ((_lb_args.debug()>2) && (CkMyPe() == cur_ld_balancer))  {
    CkPrintf("CharmLB> Obj Map:\n");
    for (const auto& val : stats->to_proc) CkPrintf("%d ", val);
    CkPrintf("\n");
  }

  if (concurrent) return NULL;  // migrate msg will only be created on PE with best solution

  LBMigrateMsg *msg = createMigrateMsg(stats);

	/* Extra feature for MetaBalancer
  if (_lb_args.metaLbOn()) {
    int clients = CkNumPes();
    LBInfo info(clients);
    getPredictedLoadWithMsg(stats, clients, msg, info, 0);
    LBRealType mLoad, mCpuLoad, totalLoad, totalLoadWComm;
    info.getSummary(mLoad, mCpuLoad, totalLoad);
    lbmgr->UpdateDataAfterLB(mLoad, mCpuLoad, totalLoad/clients);
  }
	*/

  double strat_end_time = CkWallTimer();
  lbmgr->SetStrategyCost(strat_end_time - strat_start_time);

  if (_lb_args.debug() && (CkMyPe() == cur_ld_balancer)) {
    printStrategyStats(msg);
  }
  return msg;
#else
  return NULL;
#endif
}
/*
void CentralLB::changeFreq(int r)
{
	CkAbort("ERROR: changeFreq in CentralLB should never be called!\n");
}
*/
void CentralLB::changeFreq(int nFreq)
{
#ifdef TEMP_LDB
        //CkPrintf("PROC#%d in changeFreq numProcs=%d\n",CkMyPe(),nFreq);
//  for(int i=0;i<numProcs;i++)
  {
//        if(procFreq[i]!=procFreqNew[i])
        {
              char newfreq[10];
              snprintf(newfreq,sizeof(newfreq),"%d",nFreq);
              cpufreq_sysfs_write(newfreq,CkMyPe()%physicalCoresPerNode);//i%physicalCoresPerNode);
//            CkPrintf("PROC#%d freq changing from %d to %d temp=%f\n",i,procFreq[i],procFreqNew[i],procTemp[i]);
        }
  }
#else
	CmiAbort("You should never call CentralLB::changeFreq without using the flag TEMP_LDB\n");
#endif

}

void CentralLB::work(LDStats* stats)
{
  // does nothing but print the database
  stats->print();
}

// generate migrate message from stats->from_proc and to_proc
LBMigrateMsg * CentralLB::createMigrateMsg(LDStats* stats)
{
  int i;
  std::vector<MigrateInfo*> migrateInfo;
  for (i=0; i<stats->objData.size(); i++) {
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
      migrateInfo.push_back(migrateMe);
    }
  }

  int migrate_count=migrateInfo.size();
  LBMigrateMsg* msg = new(migrate_count,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  return msg;
}

LBMigrateMsg * CentralLB::extractMigrateMsg(LBMigrateMsg *m, int p)
{
  int nmoves = 0;
  int nunavail = 0;
  int i;
  for (i=0; i<m->n_moves; i++) {
    MigrateInfo* item = (MigrateInfo*) &m->moves[i];
    if (item->from_pe == p || item->to_pe == p) nmoves++;
  }
  for (i=0; i<CkNumPes();i++) {
    if (!m->avail_vector[i]) nunavail++;
  }
  LBMigrateMsg* msg;
  if (nunavail) msg = new(nmoves,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
  else msg = new(nmoves,0,0,0) LBMigrateMsg;
  msg->n_moves = nmoves;
  msg->level = m->level;
  msg->next_lb = m->next_lb;
  for (i=0,nmoves=0; i<m->n_moves; i++) {
    MigrateInfo* item = (MigrateInfo*) &m->moves[i];
    if (item->from_pe == p || item->to_pe == p) {
      msg->moves[nmoves] = *item;
      nmoves++;
    }
  }
  // copy processor data
  if (nunavail)
  for (i=0; i<CkNumPes();i++) {
    msg->avail_vector[i] = m->avail_vector[i];
    msg->expectedLoad[i] = m->expectedLoad[i];
  }
  return msg;
}

void CentralLB::simulationWrite() {
  if(step() == LBSimulation::dumpStep)
  {
    // here we are supposed to dump the database
    int dumpFileSize = strlen(LBSimulation::dumpFile) + 4;
    char *dumpFileName = (char *)malloc(dumpFileSize);
    while (snprintf(dumpFileName, dumpFileSize, "%s.%d", LBSimulation::dumpFile, LBSimulation::dumpStep) >= dumpFileSize) {
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
  if (concurrent) CkAbort("Error: LB simulation not supported in concurrent mode");
  LBSimulation *simResults = NULL, *realResults;
  LBMigrateMsg *voidMessage = new (0,0,0,0) LBMigrateMsg();
  voidMessage->n_moves=0;
  for ( ;LBSimulation::simStepSize > 0; --LBSimulation::simStepSize, ++LBSimulation::simStep) {
    // here we are supposed to read the data from the dump database
    int simFileSize = strlen(LBSimulation::dumpFile) + 4;
    char *simFileName = (char *)malloc(simFileSize);
    while (snprintf(simFileName, simFileSize, "%s.%d", LBSimulation::dumpFile, LBSimulation::simStep) >= simFileSize) {
      free(simFileName);
      simFileSize+=3;
      simFileName = (char *)malloc(simFileSize);
    }
    readStatsMsgs(simFileName);

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
        statsData->to_proc = statsData->from_proc;
	findSimResults(statsData, LBSimulation::simProcs, voidMessage, realResults);
	simResults->PrintDifferences(realResults,statsData);
      }
      simResults->reset();
    }

    // now pass it to the strategy routine
    double startT = CkWallTimer();
    preprocess(statsData);
    CmiPrintf("%s> Strategy starts ... \n", lbname);
    LBMigrateMsg* migrateMsg = Strategy(statsData);
    CmiPrintf("%s> Strategy took %fs memory usage: CentralLB: %d KB.\n",
               lbname, CkWallTimer()-startT, (int)(useMem()/1000));

    // now calculate the results of the load balancing simulation
    findSimResults(statsData, LBSimulation::simProcs, migrateMsg, simResults);

    // now we have the simulation data, so print it and loop
    CmiPrintf("Charm++> LBSim: Simulation of load balancing step %d done.\n",LBSimulation::simStep);
    // **CWL** Officially recording my disdain here for using ints for bool
    if (LBSimulation::showDecisionsOnly) {
      simResults->PrintDecisions(migrateMsg, simFileName, 
				 LBSimulation::simProcs);
    } else {
      simResults->PrintSimulationResults();
    }

    free(simFileName);
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
    CkAbort("Fatal Error> Cannot open LB Dump file %s!\n", filename);
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

  if (_lb_args.lbversion() > 1) {
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
  CmiPrintf("n_obj: %zu n_migratable: %d \n", statsData->objData.size(), statsData->n_migrateobjs);

  // file f is closed in the destructor of PUP::fromDisk
  CmiPrintf("ReadStatsMsg from %s completed\n", filename);
#endif
}

void CentralLB::writeStatsMsgs(const char* filename) 
{
#if CMK_LBDB_ON
  FILE *f = fopen(filename, "w");
  if (f==NULL) {
    CkAbort("Fatal Error> writeStatsMsgs failed to open the output file %s!\n", filename);
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

	info.getInfo(stats, count, considerComm);
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
  if (p.isUnpacking())  {
    initLB(CkLBOptions(seqno)); 
  }
  p|reduction_started;
  int has_statsMsg=0;
  if (p.isPacking()) has_statsMsg = (statsMsg!=NULL);
  p|has_statsMsg;
  if (has_statsMsg) {
    if (p.isUnpacking())
      statsMsg = new CLBStatsMsg;
    statsMsg->pup(p);
  }
  p | use_thread;
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
  objData.resize(osz);
  commData.resize(csz);
}

CLBStatsMsg::~CLBStatsMsg() {
}

void CLBStatsMsg::pup(PUP::er &p) {
  p|from_pe;
  p|pe_speed;
#if CMK_CUDA || CMK_HIP
  p|gpu_device_id;
  p|gpu_mem_remaining;
  p|pool_buff_mem_remaining;
#endif
  p|total_walltime;
  p|idletime;
#if defined(TEMP_LDB)
	p|pe_temp;
#endif

  p|bg_walltime;
#if CMK_LB_CPUTIMER
  p|total_cputime;
  p|bg_cputime;
#endif
  p|objData;
  p|commData;

  p|avail_vector;

  p(next_lb);
}

// CkMarshalledCLBStatsMessage is used in the marshalled parameter in
// the entry function, it is just used to use to pup.
// I don't use CLBStatsMsg directly as marshalled parameter because
// I want the data pointer stored and not to be freed by the Charm++.
void CkMarshalledCLBStatsMessage::free() { 
  int count = msgs.size();
  for  (int i=0; i<count; i++) {
    delete msgs[i];
    msgs[i] = NULL;
  }
  msgs.clear();
}

void CkMarshalledCLBStatsMessage::add(CkMarshalledCLBStatsMessage &&m)
{
  int count = m.getCount();
  for (int i=0; i<count; i++) add(m.getMessage(i));
}

void CkMarshalledCLBStatsMessage::pup(PUP::er &p)
{
  int count = msgs.size();
  p|count;
  for (int i=0; i<count; i++) {
    CLBStatsMsg *msg;
    if (p.isUnpacking()) msg = new CLBStatsMsg;
    else { 
      msg = msgs[i]; CmiAssert(msg!=NULL);
    }
    msg->pup(p);
    if (p.isUnpacking()) add(msg);
  }
}

SpanningTree::SpanningTree()
{
	double sq = sqrt(CkNumPes()*4.0-3.0) - 1; // 1 + arity + arity*arity = CkNumPes()
	arity = (int)ceil(sq/2);
	calcParent(CkMyPe());
	calcNumChildren(CkMyPe());
}

void SpanningTree::calcParent(int n)
{
	parent=-1;
	if(n != 0  && arity > 0)
		parent = (n-1)/arity;
}

void SpanningTree::calcNumChildren(int n)
{
	numChildren = 0;
	if (arity == 0) return;
	int fullNode=(CkNumPes()-1-arity)/arity;
	if(n <= fullNode)
		numChildren = arity;
	if(n == fullNode+1)
		numChildren = CkNumPes()-1-(fullNode+1)*arity;
	if(n > fullNode+1)
		numChildren = 0;
}

#include "CentralLB.def.h"
 
/*@}*/
