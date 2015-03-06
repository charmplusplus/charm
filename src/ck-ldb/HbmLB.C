/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <math.h>
#include "HbmLB.h"
#include "LBDBManager.h"
#include "GreedyLB.h"
#include "GreedyCommLB.h"
#include "RefineCommLB.h"
#include "RefineLB.h"

#define  DEBUGF(x)     //  CmiPrintf x;

CreateLBFunc_Def(HbmLB, "HybridBase load balancer")

void HbmLB::staticMigrated(void* data, LDObjHandle h, int waitBarrier)
{
  HbmLB *me = (HbmLB*)(data);

  me->Migrated(h, waitBarrier);
}

void HbmLB::staticAtSync(void* data)
{
  HbmLB *me = (HbmLB*)(data);

  me->AtSync();
}

HbmLB::HbmLB(const CkLBOptions &opt): CBase_HbmLB(opt)
{
#if CMK_LBDB_ON
  lbname = (char *)"HbmLB";
  thisProxy = CProxy_HbmLB(thisgroup);
  receiver = theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
			    (void*)(this));
  notifier = theLbdb->getLBDB()->
    NotifyMigrated((LDMigratedFn)(staticMigrated), (void*)(this));

  // defines topology
  tree = new HypercubeTree;

  currentLevel = 0;
  foundNeighbors = 0;

  maxLoad = 0.0;
  vector_n_moves = 0;
  maxLoad = 0.0;
  maxCpuLoad = 0.0;
  totalLoad = 0.0;
  maxCommCount = 0;
  maxCommBytes = 0.0;

  if (_lb_args.statsOn()) theLbdb->CollectStatsOn();
#endif
}

HbmLB::~HbmLB()
{
#if CMK_LBDB_ON
  theLbdb = CProxy_LBDatabase(_lbdb).ckLocalBranch();
  if (theLbdb) {
    theLbdb->getLBDB()->
      RemoveNotifyMigrated(notifier);
    //theLbdb->
    //  RemoveStartLBFn((LDStartLBFn)(staticStartLB));
  }
  delete tree;
#endif
}

// get tree information
void HbmLB::FindNeighbors()
{
  if (foundNeighbors == 0) { // Neighbors never initialized, so init them
                           // and other things that depend on the number
                           // of neighbors

    int nlevels = tree->numLevels();
    int mype = CkMyPe();
    for (int level=0; level<nlevels; level++) 
    {
      LevelData *data = new LevelData;
      data->parent = tree->parent(mype, level);
      if (tree->isroot(mype, level)) {
        data->nChildren = tree->numChildren(mype, level);
        data->children = new int[data->nChildren];
        tree->getChildren(mype, level, data->children, data->nChildren);
        data->statsData = new LDStats(data->nChildren+1);
        //  a fake processor
        ProcStats &procStat = data->statsData->procs[data->nChildren];
        procStat.available = false;
      }
      levelData.push_back(data);
      DEBUGF(("[%d] level: %d nchildren:%d - %d %d\n", CkMyPe(), level, data->nChildren, data->nChildren>0?data->children[0]:-1, data->nChildren>1?data->children[1]:-1));
    }
    
    foundNeighbors = 1;
  }   // end if
}

void HbmLB::AtSync()
{
#if CMK_LBDB_ON
  //  CkPrintf("[%d] HbmLB At Sync step %d!!!!\n",CkMyPe(),mystep);

  FindNeighbors();

  // if num of processor is only 1, nothing should happen
  if (!QueryBalanceNow(step()) || CkNumPes() == 1) {
    MigrationDone(0);
    return;
  }

  thisProxy[CkMyPe()].ProcessAtSync();
#endif
}

void HbmLB::ProcessAtSync()
{
#if CMK_LBDB_ON
  int i;
  start_lb_time = 0;

  if (CkMyPe() == 0) {
    start_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("[%s] Load balancing step %d starting at %f\n",
	       lbName(), step(), CkWallTimer());
  }

  // build LDStats
  LBRealType total_walltime, total_cputime, idletime, bg_walltime, bg_cputime;
  theLbdb->TotalTime(&total_walltime,&total_cputime);
  theLbdb->IdleTime(&idletime);
  theLbdb->BackgroundLoad(&bg_walltime,&bg_cputime);

  myStats.n_objs = theLbdb->GetObjDataSz();
  myStats.objData.resize(myStats.n_objs);
  myStats.from_proc.resize(myStats.n_objs);
  myStats.to_proc.resize(myStats.n_objs);
  theLbdb->GetObjData(myStats.objData.getVec());
  for (i=0; i<myStats.n_objs; i++)
    myStats.from_proc[i] = myStats.to_proc[i] = 0;    // only one PE

  myStats.n_comm = theLbdb->GetCommDataSz();
  myStats.commData.resize(myStats.n_comm);
  theLbdb->GetCommData(myStats.commData.getVec());

  myStats.complete_flag = 0;

  // send to parent
  DEBUGF(("[%d] Send stats to parent %d\n", CkMyPe(), levelData[0]->parent));
  double tload = 0.0;
  for (i=0; i<myStats.n_objs; i++) tload += myStats.objData[i].wallTime;
  thisProxy[levelData[0]->parent].ReceiveStats(tload, CkMyPe(), 0);
#endif
}

void HbmLB::ReceiveStats(double t, int frompe, int fromlevel)
{
#if CMK_LBDB_ON
  FindNeighbors();

  int atlevel = fromlevel + 1;
  CmiAssert(tree->isroot(CkMyPe(), atlevel));

  DEBUGF(("[%d] ReceiveStats from PE %d from level: %d\n", CkMyPe(), frompe, fromlevel));
  int neighborIdx = NeighborIndex(frompe, atlevel);
  CmiAssert(neighborIdx==0 || neighborIdx==1);
  LevelData *lData = levelData[atlevel];
  lData->statsList[neighborIdx] = t;

  int &stats_msg_count = levelData[atlevel]->stats_msg_count;
  stats_msg_count ++;

  DEBUGF(("[%d] ReceiveStats at level: %d %d/%d\n", CkMyPe(), atlevel, stats_msg_count, levelData[atlevel]->nChildren));
  if (stats_msg_count == levelData[atlevel]->nChildren)  
  {
    stats_msg_count = 0;
    int parent = levelData[atlevel]->parent;

    // load balancing
    thisProxy[CkMyPe()].Loadbalancing(atlevel);
  }

#endif  
}


inline double myabs(double x) { return x>0.0?x:-x; }
inline double mymax(double x, double y) { return x>y?x:y; }

//  LDStats data sent to parent contains real PE
//  LDStats in parent should contain relative PE
void HbmLB::Loadbalancing(int atlevel)
{

  CmiAssert(atlevel >= 1);

  LevelData *lData = levelData[atlevel];
  LDStats *statsData = lData->statsData;
  CmiAssert(statsData);

  // at this time, all objects processor location is relative, and 
  // all incoming objects from outside group belongs to the fake root proc.

  // clear background load if needed
  if (_lb_args.ignoreBgLoad()) statsData->clearBgLoad();

  currentLevel = atlevel;

  double start_lb_time(CkWallTimer());

  double lload = lData->statsList[0];
  double rload = lData->statsList[1];

  double diff = myabs(lload-rload);
  double maxl = mymax(lload, rload);
  double avg =  (lload+rload)/2.0;
CkPrintf("[%d] lload: %f rload: %f atlevel: %d\n", CkMyPe(), lload, rload, atlevel);
  if (diff/avg > 0.02) {
    // we need to perform load balancing
    int numpes = (int)pow(2.0, atlevel);
    double delta = myabs(lload-rload) / numpes;

    int overloaded = lData->children[0];
    if (lload < rload) {
      overloaded = lData->children[1];
    }
    DEBUGF(("[%d] branch %d is overloaded by %f... \n", CkMyPe(), overloaded, delta));
    thisProxy[overloaded].ReceiveMigrationDelta(delta, atlevel, atlevel);
  }
  else {
    LoadbalancingDone(atlevel);
  }
}

// when receiving all response from underloaded pes
void HbmLB::LoadbalancingDone(int atlevel)
{
  LevelData *lData = levelData[atlevel];
  DEBUGF(("[%d] LoadbalancingDone at level: %d\n", CkMyPe(), atlevel));
  if (lData->parent != -1) {
    // send sum up
    double lload = lData->statsList[0];
    double rload = lData->statsList[1];
    double totalLoad = lload + rload;
    thisProxy[lData->parent].ReceiveStats(totalLoad, CkMyPe(), atlevel);
  }
  else {
    // done now, broadcast via tree to resume all
//    thisProxy.ReceiveResumeClients(1, tree->numLevels()-1, lData->nChildren, lData->children);
    thisProxy.ReceiveResumeClients(1, tree->numLevels()-1);
  }
}

void HbmLB::ReceiveResumeClients(int balancing, int fromlevel){
#if 0
  int atlevel = fromlevel-1;
  LevelData *lData = levelData[atlevel];
  if (atlevel != 0) 
    thisProxy.ReceiveResumeClients(balancing, atlevel, lData->nChildren, lData->children);
  else
    ResumeClients(balancing);
#else
  ResumeClients(balancing);    // it is always syncResume
/*
  if (balancing && _lb_args.syncResume()) {
    // max load of all
    CkCallback cb(CkIndex_HbmLB::ResumeClients((CkReductionMsg*)NULL),
                  thisProxy);
    contribute(sizeof(double), &maxLoad, CkReduction::max_double, cb);
  }
  else
    thisProxy[CkMyPe()].ResumeClients(balancing);
  }
*/
#endif
}

// pick objects to migrate "t" amount of work
void HbmLB::ReceiveMigrationDelta(double t, int lblevel, int fromlevel)
{
#if CMK_LBDB_ON
  int i;
  int atlevel = fromlevel-1;
  LevelData *lData = levelData[atlevel];
  if (atlevel != 0) {
    thisProxy.ReceiveMigrationDelta(t, lblevel, atlevel, lData->nChildren, lData->children);
    return;
  }

  // I am leave, find objects to migrate

  CkVec<int> migs;
  CkVec<LDObjData> &objData = myStats.objData;
  for (i=0; i<myStats.n_objs; i++) {
    LDObjData &oData = objData[i];
    if (oData.wallTime < t) {
      migs.push_back(i);
      t -= oData.wallTime;
      if (t == 0.0) break;
    }
  }

  int nmigs = migs.size();
  // send a message to 
  int matchPE = CkMyPe() ^ (1<<(lblevel-1));
  
  DEBUGF(("[%d] migrating %d objs to %d at lblevel %d! \n", CkMyPe(),nmigs,matchPE,lblevel));
  thisProxy[matchPE].ReceiveMigrationCount(nmigs, lblevel);

  // migrate objects
  for (i=0; i<nmigs; i++) {
    int idx = migs[i]-i;
    LDObjData &oData = objData[idx];
    CkVec<LDCommData> comms;
    collectCommData(idx, comms);
    thisProxy[matchPE].ObjMigrated(oData, comms.getVec(), comms.size());
    theLbdb->Migrate(oData.handle, matchPE);
    // TODO modify LDStats
    DEBUGF(("myStats.removeObject: %d, %d, %d\n", migs[i], i, objData.size()));
    myStats.removeObject(idx);
  }
#endif
}

// find sender comms
void HbmLB::collectCommData(int objIdx, CkVec<LDCommData> &comms)
{
#if CMK_LBDB_ON
  LevelData *lData = levelData[0];

  LDObjData &objData = myStats.objData[objIdx];

  for (int com=0; com<myStats.n_comm; com++) {
    LDCommData &cdata = myStats.commData[com];
    if (cdata.from_proc()) continue;
    if (cdata.sender.objID() == objData.objID() && cdata.sender.omID() == objData.omID())
      comms.push_back(cdata);
  }
#endif
}


// an object arrives with only objdata
void HbmLB::ObjMigrated(LDObjData data, LDCommData *cdata, int n)
{
  LevelData *lData = levelData[0];
  // change LDStats
  CkVec<LDObjData> &oData = myStats.objData;

    // need to update LDObjHandle later
  lData->obj_completed++;
  data.handle.handle = -100;
  oData.push_back(data);
  myStats.n_objs++;
  if (data.migratable) myStats.n_migrateobjs++;
  myStats.from_proc.push_back(-1);    // not important
  myStats.to_proc.push_back(0);

    // copy into comm data
  if (n) {
    CkVec<LDCommData> &cData = myStats.commData;
    for (int i=0; i<n; i++) 
        cData.push_back(cdata[i]);
    myStats.n_comm += n;
    myStats.deleteCommHash();
  }

  if (lData->migrationDone()) {
    // migration done finally
    MigrationDone(1);
  }
}

void HbmLB::ReceiveMigrationCount(int count, int lblevel)
{
  lbLevel = lblevel;

  LevelData *lData = levelData[0];
  lData->migrates_expected =  count;
  if (lData->migrationDone()) {
    // migration done finally
    MigrationDone(1);
  }
}

void HbmLB::Migrated(LDObjHandle h, int waitBarrier)
{
  LevelData *lData = levelData[0];

  lData->migrates_completed++;
  newObjs.push_back(h);
  DEBUGF(("[%d] An object migrated! %d %d\n", CkMyPe(),lData->migrates_completed,lData->migrates_expected));
  if (lData->migrationDone()) {
    // migration done finally
    MigrationDone(1);
  }
}

void HbmLB::NotifyObjectMigrationDone(int fromlevel, int lblevel)
{

  int atlevel = fromlevel + 1;
  LevelData *lData = levelData[atlevel];

  lData->mig_reported ++;
  DEBUGF(("[%d] HbmLB::NotifyObjectMigrationDone at level: %d lblevel: %d reported: %d!\n", CkMyPe(), atlevel, lblevel, lData->mig_reported));
  if (atlevel < lblevel) {
    if (lData->mig_reported == lData->nChildren) {
      lData->mig_reported = 0;
      thisProxy[lData->parent].NotifyObjectMigrationDone(atlevel, lbLevel);
    }
  }
  else {
    if (lData->mig_reported == lData->nChildren/2) {   // half tree
      lData->mig_reported = 0;
      // load balancing done at this level
      LoadbalancingDone(atlevel);
    }
  }
}

// migration done at current lbLevel
void HbmLB::MigrationDone(int balancing)
{
#if CMK_LBDB_ON
  int i, j;
  LevelData *lData = levelData[0];

  DEBUGF(("[%d] HbmLB::MigrationDone lbLevel:%d numLevels:%d!\n", CkMyPe(), lbLevel, tree->numLevels()));

  CmiAssert(newObjs.size() == lData->migrates_expected);

#if 0
  if (lbLevel == tree->numLevels()-1) {
    theLbdb->incStep();
    // reset 
    lData->clear();
  }
  else {
    lData->migrates_expected = -1;
    lData->migrates_completed = 0;
    lData->obj_completed = 0;
  }
#else
  lData->migrates_expected = -1;
  lData->migrates_completed = 0;
  lData->obj_completed = 0;
#endif

  CkVec<LDObjData> &oData = myStats.objData;

  // update
  int count=0;
  for (i=0; i<oData.size(); i++)
    if (oData[i].handle.handle == -100) count++;
  CmiAssert(count == newObjs.size());

  for (i=0; i<oData.size(); i++) {
    if (oData[i].handle.handle == -100) {
      LDObjHandle &handle = oData[i].handle;
      for (j=0; j<newObjs.size(); j++) {
        if (handle.omID() == newObjs[j].omID() && 
                  handle.objID() == newObjs[j].objID()) {
          handle = newObjs[j];
          break;
        }
      }
      CmiAssert(j<newObjs.size());
    }
  }
  newObjs.free();

  thisProxy[lData->parent].NotifyObjectMigrationDone(0, lbLevel);
#endif
}

void HbmLB::ResumeClients(CkReductionMsg *msg)
{
  if (CkMyPe() == 0 && _lb_args.printSummary()) {
    double mload = *(double *)msg->getData();
    CkPrintf("[%d] MAX Load: %f at step %d.\n", CkMyPe(), mload, step()-1);
  }
  ResumeClients(1);
  delete msg;
}

void HbmLB::ResumeClients(int balancing)
{
#if CMK_LBDB_ON
  DEBUGF(("[%d] ResumeClients. \n", CkMyPe()));

  theLbdb->incStep();
  // reset 
  LevelData *lData = levelData[0];
  lData->clear();

  if (CkMyPe() == 0 && balancing) {
    double end_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("[%s] Load balancing step %d finished at %f duration %f\n",
	        lbName(), step()-1,end_lb_time,end_lb_time - start_lb_time);
  }
  if (balancing && _lb_args.printSummary()) {
      int count = 1;
      LBInfo info(count);
      LDStats *stats = &myStats;
      info.getInfo(stats, count, 0);	// no comm cost
      LBRealType mLoad, mCpuLoad, totalLoad;
      info.getSummary(mLoad, mCpuLoad, totalLoad);
      int nmsgs, nbytes;
      stats->computeNonlocalComm(nmsgs, nbytes);
      CkPrintf("[%d] Load with %d objs: max (with comm): %f max (obj only): %f total: %f on %d processors at step %d useMem: %fKB nonlocal: %d %.2fKB.\n", CkMyPe(), stats->n_objs, mLoad, mCpuLoad, totalLoad, count, step()-1, (1.0*useMem())/1024, nmsgs, nbytes/1024.0);
      thisProxy[0].reportLBQulity(mLoad, mCpuLoad, totalLoad, nmsgs, 1.0*nbytes/1024.0);
  }

  // zero out stats
  theLbdb->ClearLoads();

  theLbdb->ResumeClients();
#endif
}

// only called on PE 0
void HbmLB::reportLBQulity(double mload, double mCpuLoad, double totalload, int nmsgs, double bytes)
{
  static int pecount=0;
  CmiAssert(CkMyPe() == 0);
  if (mload > maxLoad) maxLoad = mload;
  if (mCpuLoad > maxCpuLoad) maxCpuLoad = mCpuLoad;
  totalLoad += totalload;
  maxCommCount += nmsgs;
  maxCommBytes += bytes;   // KB
  pecount++;
  if (pecount == CkNumPes()) {
    CkPrintf("[%d] Load Summary: max (with comm): %f max (obj only): %f total: %f at step %d nonlocal: %d msgs, %.2fKB reported from %d PEs.\n", CkMyPe(), maxLoad, maxCpuLoad, totalLoad, step(), maxCommCount, maxCommBytes, pecount);
    maxLoad = 0.0;
    maxCpuLoad = 0.0;
    totalLoad = 0.0;
    maxCommCount = 0;
    maxCommBytes = 0.0;
    pecount = 0;
  }
}

void HbmLB::work(LDStats* stats)
{
#if CMK_LBDB_ON
  CkPrintf("[%d] HbmLB::work called!\n", CkMyPe());
#endif
}
  
int HbmLB::NeighborIndex(int pe, int atlevel)
{
    int peslot = -1;
    for(int i=0; i < levelData[atlevel]->nChildren; i++) {
      if (pe == levelData[atlevel]->children[i]) {
	peslot = i;
	break;
      }
    }
    return peslot;
}

int HbmLB::useMem()
{
  int i;
  int memused = 0;
  for (i=0; i<levelData.size(); i++)
    if (levelData[i]) memused+=levelData[i]->useMem();
  return memused;
}

#include "HbmLB.def.h"

/*@{*/


