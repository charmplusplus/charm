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
#include "HybridBaseLB.h"
#include "LBDBManager.h"
#include "GreedyLB.h"
#include "GreedyCommLB.h"
#include "RefineCommLB.h"
#include "RefineLB.h"

#define  DEBUGF(x)      // CmiPrintf x;

CreateLBFunc_Def(HybridBaseLB, "HybridBase load balancer");

void HybridBaseLB::staticMigrated(void* data, LDObjHandle h, int waitBarrier)
{
  HybridBaseLB *me = (HybridBaseLB*)(data);

  me->Migrated(h, waitBarrier);
}

void HybridBaseLB::staticAtSync(void* data)
{
  HybridBaseLB *me = (HybridBaseLB*)(data);

  me->AtSync();
}

HybridBaseLB::HybridBaseLB(const CkLBOptions &opt): BaseLB(opt)
{
#if CMK_LBDB_ON
  lbname = (char *)"HybridBaseLB";
  thisProxy = CProxy_HybridBaseLB(thisgroup);
  receiver = theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
			    (void*)(this));
  notifier = theLbdb->getLBDB()->
    NotifyMigrated((LDMigratedFn)(staticMigrated), (void*)(this));

  // defines topology
  tree = new ThreeLevelTree;

  // decide which load balancer to call
//  greedy = (CentralLB *)AllocateGreedyLB();
//  refine = (CentralLB *)AllocateRefineLB();

  currentLevel = 0;
  foundNeighbors = 0;
  future_migrates_expected = -1;

  if (_lb_args.statsOn()) theLbdb->CollectStatsOn();
#endif
}

HybridBaseLB::~HybridBaseLB()
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
void HybridBaseLB::FindNeighbors()
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
        data->statsMsgsList = new CLBStatsMsg*[data->nChildren];
        for(int i=0; i < data->nChildren; i++)
           data->statsMsgsList[i] = NULL;
        data->statsData = new LDStats(data->nChildren+1);
        //  a fake processor
        ProcStats &procStat = data->statsData->procs[data->nChildren];
        procStat.available = CmiFalse;
      }
      levelData.push_back(data);
      DEBUGF(("[%d] level: %d nchildren:%d - %d %d\n", CkMyPe(), level, data->nChildren, data->nChildren>0?data->children[0]:-1, data->nChildren>1?data->children[1]:-1));
    }
    
    foundNeighbors = 1;
  }   // end if
}

void HybridBaseLB::AtSync()
{
#if CMK_LBDB_ON
  //  CkPrintf("[%d] HybridBaseLB At Sync step %d!!!!\n",CkMyPe(),mystep);

  FindNeighbors();

  // if num of processor is only 1, nothing should happen
  if (!QueryBalanceNow(step()) || CkNumPes() == 1) {
    MigrationDone(0);
    return;
  }

  thisProxy[CkMyPe()].ProcessAtSync();
#endif
}

void HybridBaseLB::ProcessAtSync()
{
#if CMK_LBDB_ON
  start_lb_time = 0;

  if (CkMyPe() == 0) {
    start_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("[%s] Load balancing step %d starting at %f\n",
	       lbName(), step(), CkWallTimer());
  }

  // assemble LB database
  CLBStatsMsg* msg = AssembleStats();

  CkMarshalledCLBStatsMessage marshmsg(msg);
  // send to parent
  thisProxy[levelData[0]->parent].ReceiveStats(marshmsg, 0);

  DEBUGF(("[%d] Send stats to myself\n", CkMyPe()));
#endif
}

CLBStatsMsg* HybridBaseLB::AssembleStats()
{
#if CMK_LBDB_ON
  // build and send stats
  const int osz = theLbdb->GetObjDataSz();
  const int csz = theLbdb->GetCommDataSz();

  CLBStatsMsg* msg = new CLBStatsMsg(osz, csz);
  msg->from_pe = CkMyPe();

  // Get stats
  theLbdb->GetTime(&msg->total_walltime,&msg->total_cputime,
                   &msg->idletime, &msg->bg_walltime,&msg->bg_cputime);
//  msg->pe_speed = myspeed;
  // number of pes
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

void HybridBaseLB::ReceiveStats(CkMarshalledCLBStatsMessage &data, int fromlevel)
{
#if CMK_LBDB_ON
  FindNeighbors();

  // store the message
  CLBStatsMsg *m = data.getMessage();
  int atlevel = fromlevel + 1;
  CmiAssert(tree->isroot(CkMyPe(), atlevel));

  depositLBStatsMessage(m, atlevel);

  int &stats_msg_count = levelData[atlevel]->stats_msg_count;
  stats_msg_count ++;

  DEBUGF(("[%d] ReceiveStats at level: %d %d/%d\n", CkMyPe(), atlevel, stats_msg_count, levelData[atlevel]->nChildren));
  if (stats_msg_count == levelData[atlevel]->nChildren)  
  {
    // build LDStats
    buildStats(atlevel);
    stats_msg_count = 0;
    int parent = levelData[atlevel]->parent;
    if (parent != -1) {
      // combine and shrink message
      // build a new message based on our LDStats
      CLBStatsMsg* cmsg = buildCombinedLBStatsMessage(atlevel);

      // send to parent
      CkMarshalledCLBStatsMessage marshmsg(cmsg);
      thisProxy[parent].ReceiveStats(marshmsg, atlevel);
    }
    else {
      // root of all processors, calls top-level strategy (refine)
      thisProxy[CkMyPe()].Loadbalancing(atlevel);
    }
  }

#endif  
}

// store stats message in a level data
void HybridBaseLB::depositLBStatsMessage(CLBStatsMsg *m, int atlevel)
{
  int pe = m->from_pe;
  int neighborIdx = NeighborIndex(pe, atlevel);

  CLBStatsMsg **statsMsgsList = levelData[atlevel]->statsMsgsList;
  LDStats *statsData = levelData[atlevel]->statsData;
  CmiAssert(statsMsgsList && statsData);

  if (statsMsgsList[neighborIdx] != 0) {
    CkPrintf("*** Unexpected CLBStatsMsg in ReceiveStats from PE %d-%d ***\n", pe,neighborIdx);
    CkAbort("HybridBaseLB> Abort!");
  }

  // replace real pe to relative pe number in preparation for calling Strategy()
  for (int i=0; i<m->n_comm; i++) {
     LDCommData &commData = m->commData[i];
     // modify processor to be this local pe
     if (commData.from_proc()) m->commData[i].src_proc = neighborIdx;
     if (commData.receiver.get_type() == LD_PROC_MSG) m->commData[i].receiver.setProc(neighborIdx);
  }

  statsMsgsList[neighborIdx] = m;
      // store per processor data right away
  struct ProcStats &procStat = statsData->procs[neighborIdx];
  procStat.pe = pe;	// real PE
  procStat.total_walltime = m->total_walltime;
  procStat.total_cputime = m->total_cputime;
  procStat.idletime = m->idletime;
  procStat.bg_walltime = m->bg_walltime;
  procStat.bg_cputime = m->bg_cputime;
  procStat.pe_speed = m->pe_speed;		// important
  procStat.available = CmiTrue;
  procStat.n_objs = m->n_objs;

  statsData->n_objs += m->n_objs;
  statsData->n_comm += m->n_comm;
}

void HybridBaseLB::buildStats(int atlevel)
{
#if CMK_LBDB_ON
  // build LDStats
  LevelData *lData = levelData[atlevel];
  LDStats *statsData = lData->statsData;
  CLBStatsMsg **statsMsgsList = lData->statsMsgsList;
  int stats_msg_count = lData->stats_msg_count;

  // statsMsgsList
  DEBUGF(("[%d] buildStats for %d nobj:%d\n", CkMyPe(), stats_msg_count, statsData->n_objs));
  statsData->count = stats_msg_count;
  statsData->objData.resize(statsData->n_objs);
  statsData->commData.resize(statsData->n_comm);
  statsData->from_proc.resize(statsData->n_objs);
  statsData->to_proc.resize(statsData->n_objs);
  int nobj = 0;
  int nmigobj = 0;
  int ncom = 0;
  for (int n=0; n<stats_msg_count; n++) {
     int i;
     CLBStatsMsg *msg = statsMsgsList[n];
     int pe = msg->from_pe;
     for (i=0; i<msg->n_objs; i++) {
         // need to map index to relative index
         statsData->from_proc[nobj] = statsData->to_proc[nobj] = NeighborIndex(pe, atlevel);
         statsData->objData[nobj] = msg->objData[i];
         if (msg->objData[i].migratable) nmigobj++;
         nobj++;
     }
     for (i=0; i<msg->n_comm; i++) {
         statsData->commData[ncom] = msg->commData[i];
         ncom++;
     }
     // free the message
     delete msg;
     statsMsgsList[n]=0;
  }
  if (_lb_args.debug()>1) {
      CmiPrintf("[%d] n_obj:%d migratable:%d ncom:%d\n", CkMyPe(), nobj, nmigobj, ncom);
  }
  CmiAssert(statsData->n_objs == nobj);
  CmiAssert(statsData->n_comm == ncom);
  statsData->n_migrateobjs = nmigobj;
#endif
}

// build a message based on our LDStats for sending to parent
CLBStatsMsg * HybridBaseLB::buildCombinedLBStatsMessage(int atlevel)
{
#if CMK_LBDB_ON
  int i;
  LDStats *statsData = levelData[atlevel]->statsData;
  const int osz = statsData->n_objs;
  const int csz = statsData->n_comm;

  CLBStatsMsg* cmsg = new CLBStatsMsg(osz, csz);
  int mype = CkMyPe();
  cmsg->from_pe = mype;	// real PE

  // Get stats
  // sum of all childrens
  cmsg->pe_speed = 0;
  cmsg->total_walltime = 0.0;
  cmsg->total_cputime = 0.0;
  for (int pe=0; pe<statsData->count; pe++) {
        struct ProcStats &procStat = statsData->procs[pe];
        cmsg->pe_speed += procStat.pe_speed;		// important
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
  cmsg->n_comm = csz;
  for (i=0; i<csz; i++) {
     LDCommData &commData = statsData->commData[i];
     cmsg->commData[i] = commData;
     // modify processor to be this real pe
     if (commData.from_proc()) cmsg->commData[i].src_proc = mype;
     if (commData.receiver.get_type() == LD_PROC_MSG) cmsg->commData[i].receiver.setProc(mype);
  }
  return cmsg;
#else
  return NULL;
#endif
}

//  LDStats data sent to parent contains real PE
//  LDStats in parent should contain relative PE
void HybridBaseLB::Loadbalancing(int atlevel)
{
  int i;

  // cleanup all objects that has gone
  //cleanupDatabase();

  // if we are leaf, we are done
  CmiAssert(atlevel >= 1);

  LevelData *lData = levelData[atlevel];
  LDStats *statsData = lData->statsData;
  CmiAssert(statsData);

  // at this time, all objects processor location is relative, and 
  // all incoming objects from outside group belongs to the fake root proc.

  DEBUGF(("[%d] Calling Strategy ... \n", CkMyPe()));
  double start_lb_time;
  if (atlevel == tree->numLevels()-1) {
    start_lb_time = CkWallTimer();
  }

  // clear background load if needed
  if (_lb_args.ignoreBgLoad()) statsData->clearBgLoad();

  currentLevel = atlevel;
  int nclients = lData->nChildren;
  LBMigrateMsg* migrateMsg = Strategy(statsData, nclients);

  if (atlevel == tree->numLevels()-1) {
    // FIXME
    double strat_end_time = CkWallTimer();
    if (_lb_args.debug()>1)
        CkPrintf("[%d] Level %d Strat elapsed time %f\n", CkMyPe(), atlevel, strat_end_time-start_lb_time);
  }

  // send to children 
  thisProxy.ReceiveMigration(migrateMsg, nclients, lData->children);

  // inform new objects that are from outside group
  int c = 0;
  for (i=0; i<statsData->n_objs; i++) {
    if (statsData->from_proc[i] == nclients)  {
      CmiAssert(statsData->to_proc[i] < nclients);
      int tope = lData->children[statsData->to_proc[i]];
      // TODO: comm data
      thisProxy[tope].ObjMigrated(statsData->objData[i], atlevel-1);
      c++;
    }
  }
}

LBMigrateMsg* HybridBaseLB::Strategy(LDStats* stats,int count)
{
#if CMK_LBDB_ON
  work(stats, count);

  if (_lb_args.debug()>2)  {
    CkPrintf("Obj Map:\n");
    for (int i=0; i<stats->n_objs; i++) CkPrintf("%d ", stats->to_proc[i]);
    CkPrintf("\n");
  }

  return createMigrateMsg(stats, count);
#else
  return NULL;
#endif
}

// migrate only object LDStat in group
void HybridBaseLB::ReceiveMigration(LBMigrateMsg *msg)
{
#if CMK_LBDB_ON
  FindNeighbors();

  int atlevel = msg->level - 1;

  DEBUGF(("[%d] ReceiveMigration\n", CkMyPe()));

  LevelData *lData = levelData[atlevel];

  // only non NULL when level > 0
  LDStats *statsData = lData->statsData;

  // do LDStats migration
  const int me = CkMyPe();
  lData->migrates_expected = 0;
  for(int i=0; i < msg->n_moves; i++) {
    MigrateInfo& move = msg->moves[i];
    // incoming
    if (move.from_pe != me && move.to_pe == me) {
      // I can not be the parent node
      DEBUGF(("[%d] expecting LDStats object from %d\n",me,move.from_pe));
      // will receive a ObjData message
      lData->migrates_expected ++;
    }
    else if (move.from_pe == me) {   // outgoing
      if (statsData) {		// this is inner node
        // send objdata
        int wasonpe = -1;
//        LDObjKey key;
        int obj;
        int found = 0;
        for (obj = 0; obj<statsData->n_objs; obj++) {
          if (move.obj.omID() == statsData->objData[obj].handle.omID() && 
            move.obj.objID() == statsData->objData[obj].handle.objID())
          {
            DEBUGF(("[%d] level: %d sending objData %d to %d. \n", CkMyPe(), atlevel, obj, move.to_pe));
	    found = 1;
            thisProxy[move.to_pe].ObjMigrated(statsData->objData[obj], atlevel);
            lData->outObjs.push_back(MigrationRecord(move.obj, lData->children[statsData->from_proc[obj]], -1));
            statsData->removeObject(obj);
            break;
          }
        }
        CmiAssert(found == 1);
      }
      else {		// this is leave node
        if (move.to_pe == -1) {
          lData->outObjs.push_back(MigrationRecord(move.obj, CkMyPe(), -1));
        }
        else {
          // migrate the object
          theLbdb->Migrate(move.obj,move.to_pe);
        }
      }
    }   // end if
  }

  if (lData->migrationDone())
    StatsDone(atlevel);
#endif
}

void HybridBaseLB::Migrated(LDObjHandle h, int waitBarrier)
{
  LevelData *lData = levelData[0];

  lData->migrates_completed++;
  DEBUGF(("[%d] An object migrated! %d %d\n", CkMyPe(),lData->migrates_completed,lData->migrates_expected));
  if (lData->migrationDone()) {
    if (!lData->resumeAfterMigration) {
      StatsDone(0);
    }
    else {
      // migration done finally
      MigrationDone(1);
    }
  }
}

// an object arrives with only objdata
void HybridBaseLB::ObjMigrated(LDObjData data, int atlevel)
{
  LevelData *lData = levelData[atlevel];
  LDStats *statsData = lData->statsData;

  if (statsData != NULL) {
    CkVec<LDObjData> &oData = statsData->objData;

    // copy into LDStats
    oData.push_back(data);
    statsData->n_objs++;
    if (data.migratable) statsData->n_migrateobjs++;
    // an incoming object to the root
    // pretend this object belongs to it
    statsData->from_proc.push_back(lData->nChildren);
    statsData->to_proc.push_back(lData->nChildren);
  }
  else { 	// leaf node, from which proc is unknown at this time
    LDObjKey key;
    key.omID() = data.omID();
    key.objID() = data.objID();
    newObjs.push_back(Location(key, -1));
  }

  lData->obj_completed++;
  if (lData->migrationDone()) {
    StatsDone(atlevel);
  }
}


void HybridBaseLB::StatsDone(int atlevel)
{
  int i;

  LevelData *lData = levelData[atlevel];
  lData->obj_expected = -1;
  lData->migrates_expected = -1;
  lData->obj_completed = 0;
  lData->migrates_completed = 0;

  CmiAssert(lData->parent!=-1);

  thisProxy[lData->parent].NotifyObjectMigrationDone(atlevel);
}

void HybridBaseLB::NotifyObjectMigrationDone(int fromlevel)
{
  int i;
  int atlevel = fromlevel + 1;
  LevelData *lData = levelData[atlevel];

  lData->mig_reported ++;
  if (lData->mig_reported == lData->nChildren) {
    lData->mig_reported = 0;
    // start load balancing at this level
    if (atlevel > 1) {
      // I am done at the level, propagate load balancing to next level
      thisProxy.Loadbalancing(atlevel-1, lData->nChildren, lData->children);
    }
    else {  // atlevel = 1
      thisProxy.StartCollectInfo(lData->nChildren, lData->children);
    }
  }
}

void HybridBaseLB::StartCollectInfo()
{
  int i;
  LevelData *lData = levelData[0];
      // we are leaf, start a tree reduction to find from/to proc pairs
      // set this counter
  lData->resumeAfterMigration =  1;

      // Locations
  int migs = lData->outObjs.size() + newObjs.size();
  Location *locs = new Location[migs];
  int count=0;
  int me = CkMyPe();
  for (i=0; i<newObjs.size(); i++) {
        locs[count] = newObjs[i];
        locs[count].loc = me;
        count++;
  }
  for (i=0; i<lData->outObjs.size(); i++) {
        LDObjKey key;
        key.omID() = lData->outObjs[i].handle.omID();
        key.objID() = lData->outObjs[i].handle.objID();
        locs[count].key = key;
        locs[count].loc = -1;		// unknown
        count++;
  }
    // assuming leaf must have a parent
  DEBUGF(("[%d] level 0 has %d unmatched (out)%d+(new)%d. \n", CkMyPe(), migs, lData->outObjs.size(), newObjs.size()));
  thisProxy[lData->parent].CollectInfo(locs, migs, 0);
}

void HybridBaseLB::CollectInfo(Location *loc, int n, int fromlevel)
{
   int atlevel = fromlevel + 1;
   LevelData *lData = levelData[atlevel];
   lData->info_recved++;

   CkVec<Location> &matchedObjs = lData->matchedObjs;
   CkVec<Location> &unmatchedObjs = lData->unmatchedObjs;

   // sort into mactched and unmatched list
   for (int i=0; i<n; i++) {
     // search and see if we have answer, put to matched
     // store in unknown
     int found = 0;
     for (int obj=0; obj<unmatchedObjs.size(); obj++) {
       if (loc[i].key == unmatchedObjs[obj].key) {
         // answer must exist
         CmiAssert(unmatchedObjs[obj].loc != -1 || loc[i].loc != -1);
         if (unmatchedObjs[obj].loc == -1) unmatchedObjs[obj].loc = loc[i].loc;
         matchedObjs.push_back(unmatchedObjs[obj]);
         unmatchedObjs.remove(obj);
         found = 1;
         break;
       }
     }
     if (!found) unmatchedObjs.push_back(loc[i]);
   }

  DEBUGF(("[%d] level %d has %d unmatched and %d matched. \n", CkMyPe(), atlevel, unmatchedObjs.size(), matchedObjs.size()));

   if (lData->info_recved == lData->nChildren) {
     lData->info_recved = 0;
     if (lData->parent != -1) {
       // send only unmatched ones up the tree
       thisProxy[lData->parent].CollectInfo(unmatchedObjs.getVec(), unmatchedObjs.size(), atlevel);
     }
     else { // root
       // we should have all answers now
       CmiAssert(unmatchedObjs.size() == 0);
       // start send match list down
       thisProxy.PropagateInfo(matchedObjs.getVec(), matchedObjs.size(), atlevel, lData->nChildren, lData->children);
       lData->statsData->clear();
     }
   }
}

void HybridBaseLB::PropagateInfo(Location *loc, int n, int fromlevel)
{
#if CMK_LBDB_ON
  int i, obj;
  int atlevel = fromlevel - 1;
  LevelData *lData = levelData[atlevel];
  CkVec<Location> &matchedObjs = lData->matchedObjs;
  CkVec<Location> &unmatchedObjs = lData->unmatchedObjs;

  if (atlevel > 0) {
    // search in unmatched
    for (i=0; i<n; i++) {
     // search and see if we have answer, put to matched
     // store in unknown
       for (obj=0; obj<unmatchedObjs.size(); obj++) {
         if (loc[i].key == unmatchedObjs[obj].key) {
         // answer must exist now
           CmiAssert(unmatchedObjs[obj].loc != -1 || loc[i].loc != -1);
           if (unmatchedObjs[obj].loc == -1) unmatchedObjs[obj].loc = loc[i].loc;
           matchedObjs.push_back(unmatchedObjs[obj]);
           unmatchedObjs.remove(obj);
           break;
         }
       }
    }
    CmiAssert(unmatchedObjs.size() == 0);
    DEBUGF(("[%d] level %d PropagateInfo had %d matchedObjs. \n", CkMyPe(), atlevel, matchedObjs.size()));

  // send down
    thisProxy.PropagateInfo(matchedObjs.getVec(), matchedObjs.size(), atlevel, lData->nChildren, lData->children);

    lData->statsData->clear();
    matchedObjs.free();
  }
  else {  // leaf node
    // now start to migrate
    CkVec<MigrationRecord> & outObjs = lData->outObjs;
    int migs = outObjs.size() + newObjs.size();
    for (i=0; i<outObjs.size(); i++) {
      if (outObjs[i].toPe == -1) {
        for (obj=0; obj<n; obj++) {
          if (loc[obj].key.omID() == outObjs[i].handle.omID() &&
              loc[obj].key.objID() == outObjs[i].handle.objID()) {
            outObjs[i].toPe = loc[obj].loc;
            break;
          }
        }
        CmiAssert(obj < n);
      }
      CmiAssert(outObjs[i].toPe != -1);
        // migrate now!
      theLbdb->Migrate(outObjs[i].handle,outObjs[i].toPe);
    }   // end for out
    // incoming
    lData->migrates_expected = 0;
    future_migrates_expected = 0;
    for (i=0; i<newObjs.size(); i++) {
      if (newObjs[i].loc == -1) {
        for (obj=0; obj<n; obj++) {
          if (loc[obj].key == newObjs[i].key) {
            newObjs[i].loc = loc[obj].loc;
            break;
          }
        }
        CmiAssert(obj < n);
      }
      CmiAssert(newObjs[i].loc != -1);
      lData->migrates_expected++;
    }   // end of for
    DEBUGF(("[%d] expecting %d\n", CkMyPe(), lData->migrates_expected));
    if (lData->migrationDone()) {
      MigrationDone(1);
    }
  }
#endif
}

void HybridBaseLB::MigrationDone(int balancing)
{
#if CMK_LBDB_ON
  LevelData *lData = levelData[0];

  DEBUGF(("[%d] HybridBaseLB::MigrationDone!\n", CkMyPe()));

  theLbdb->incStep();

  // reset 
  for (int i=0; i<tree->numLevels(); i++) 
    levelData[i]->clear();
  newObjs.free();

  DEBUGF(("[%d] calling ResumeClients.\n", CkMyPe()));
  if (balancing && _lb_args.syncResume()) {
    CkCallback cb(CkIndex_HybridBaseLB::ResumeClients((CkReductionMsg*)NULL),
                  thisProxy);
    contribute(0, NULL, CkReduction::sum_int, cb);
  }
  else
    thisProxy[CkMyPe()].ResumeClients(balancing);
#endif
}

void HybridBaseLB::ResumeClients(CkReductionMsg *msg)
{
  ResumeClients(1);
  delete msg;
}

void HybridBaseLB::ResumeClients(int balancing)
{
#if CMK_LBDB_ON
  DEBUGF(("[%d] ResumeClients. \n", CkMyPe()));

  if (CkMyPe() == 0 && balancing) {
    double end_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("[%s] Load balancing step %d finished at %f duration %f\n",
	        lbName(), step()-1,end_lb_time,end_lb_time - start_lb_time);
  }

  // zero out stats
  theLbdb->ClearLoads();

  theLbdb->ResumeClients();
#endif
}

void HybridBaseLB::work(LDStats* stats,int count)
{
#if CMK_LBDB_ON
  CkPrintf("[%d] HybridBaseLB::work called!\n", CkMyPe());
#endif
}
  
LBMigrateMsg * HybridBaseLB::createMigrateMsg(LDStats* stats,int count)
{
#if CMK_LBDB_ON
  int i;

  LevelData *lData = levelData[currentLevel];

  CkVec<MigrateInfo*> migrateInfo;

  // stats contains all objects that belong to this group
  // outObjs contains objects that are migrated out
  for (i=0; i<stats->n_objs; i++) {
    LDObjData &objData = stats->objData[i];
    int frompe = stats->from_proc[i];
    int tope = stats->to_proc[i];
    CmiAssert(tope != -1);
    if (frompe != tope) {
      //      CkPrintf("[%d] Obj %d migrating from %d to %d\n",
      //         CkMyPe(),obj,pe,dest);
#if 0
      // delay until a summary is printed
      if (frompe == lData->nChildren)  {
        frompe = -1;
        CmiAssert(tope != -1 && tope != lData->nChildren);
      }
      else
        frompe = lData->children[frompe];
      if (tope != -1) {
        CmiAssert(tope < lData->nChildren);
        tope = lData->children[tope];
      }
#endif
      MigrateInfo *migrateMe = new MigrateInfo;
      migrateMe->obj = objData.handle;
      migrateMe->from_pe = frompe;
      migrateMe->to_pe = tope;
      migrateMe->async_arrival = objData.asyncArrival;
      migrateInfo.insertAtEnd(migrateMe);
    }
    else 
      CmiAssert(frompe != lData->nChildren);
  }

  // merge outgoing objs
  CkVec<MigrationRecord> &outObjs = lData->outObjs;
  for (i=0; i<outObjs.size(); i++) {
    MigrateInfo *migrateMe = new MigrateInfo;
    migrateMe->obj = outObjs[i].handle;
    migrateMe->from_pe = outObjs[i].fromPe;
    migrateMe->to_pe = -1;
//    migrateMe->async_arrival = objData.asyncArrival;
    migrateInfo.insertAtEnd(migrateMe);
  }

  // construct migration message
  int migrate_count=migrateInfo.length();
  DEBUGF(("[%d] level: %d has %d migrations. \n", CkMyPe(), currentLevel, migrate_count));
  LBMigrateMsg * msg = new(migrate_count,count,count,0) LBMigrateMsg;
  msg->level = currentLevel;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
    DEBUGF(("[%d] obj (%d %d %d %d) migrate from %d to %d\n", CkMyPe(), item->obj.objID().id[0], item->obj.objID().id[1], item->obj.objID().id[2], item->obj.objID().id[3], item->from_pe, item->to_pe));
  }

  if (_lb_args.printSummary() && currentLevel == 1) {
      LBInfo info(msg->expectedLoad, count);
      info.getInfo(stats, count, 0);	// no comm cost
      double maxLoad, totalLoad;
      info.getSummary(maxLoad, totalLoad);
      CkPrintf("[%d] Load Summary: max: %f total: %f on %d processors.\n", CkMyPe(), maxLoad, totalLoad, count);
  }

  // translate relative pe number to its real number
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* move = &msg->moves[i];
    if (move->to_pe != -1) {
      if (move->from_pe == lData->nChildren)  {
          // an object from outside group
        move->from_pe = -1;
        CmiAssert(move->to_pe != -1 && move->to_pe != lData->nChildren);
      }
      else
        move->from_pe = lData->children[move->from_pe];
      CmiAssert(move->to_pe < lData->nChildren);
      move->to_pe = lData->children[move->to_pe];
    }
  }

  return msg;
#else
  return NULL;
#endif
}

int HybridBaseLB::NeighborIndex(int pe, int atlevel)
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

#include "HybridBaseLB.def.h"

/*@{*/


