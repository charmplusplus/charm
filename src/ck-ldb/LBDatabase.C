#include "LBDatabase.h"
#include "cksyncbarrier.h"

#include "ck.h"
#if CMK_GPU
void hapiCuptiObjectDeparting(const LDObjHandle&, int dest);
void hapiCuptiTakeDepartedLoads(std::vector<std::pair<LDObjKey, double>>&);
void hapiCuptiObjectArrived(const LDObjHandle&);
void hapiCuptiDropDepartedLoads();
#endif

LBDatabase::LBDatabase() {
  omCount = omsRegistering = 0;
  obj_walltime = 0;
  obj_gputime = 0;
  statsAreOn = false;
  objsEmptyHead = -1;
  commTable = new LBCommTable;
  syncBarrier = CkSyncBarrier::object();
}

void LBDatabase::CollectStatsOn(void){
  if (!StatsOn()) {
    LDObjHandle runObj;
    if (CkActiveLdHandle(runObj)) LbObj(runObj)->StartTimer();
    TurnStatsOn();
  }
}

LDOMHandle LBDatabase::RegisterOM(LDOMid userID, void* userPtr, LDCallbacks cb) {
  LDOMHandle newHandle;
  newHandle.id = userID;

  LBOM* om = new LBOM(this, userID, userPtr, cb);
  if (om != nullptr) {
    newHandle.handle = oms.size();
    oms.push_back(om);
  } else newHandle.handle = -1;
  om->DepositHandle(newHandle);
  omCount++;
  return newHandle;
}

void LBDatabase::UnregisterOM(LDOMHandle omh) {
  delete oms[omh.handle];
  oms[omh.handle] = nullptr;
  omCount--;
}

void LBDatabase::RegisteringObjects(LDOMHandle omh) {
  // for an unregistered anonymous OM to join and control the barrier
  if (omh.id.id.idx == 0) {
    if (omsRegistering == 0)
      syncBarrier->turnOff();
    omsRegistering++;
  }
  else {
    LBOM* om = oms[omh.handle];
    if (!om->RegisteringObjs()) {
      if (omsRegistering == 0)
        syncBarrier->turnOff();
      omsRegistering++;
      om->SetRegisteringObjs(true);
    }
  }
}

void LBDatabase::DoneRegisteringObjects(LDOMHandle omh)
{
  // for an unregistered anonymous OM to join and control the barrier
  if (omh.id.id.idx == 0) {
    omsRegistering--;
    if (omsRegistering == 0)
      syncBarrier->turnOn();
  }
  else {
    LBOM* om = oms[omh.handle];
    if (om->RegisteringObjs()) {
      omsRegistering--;
      om->SetRegisteringObjs(false);
      if (omsRegistering == 0)
        // This call to turnOn must come after the decrement of omsRegistering and the
        // call to SetRegisteringObjs(false) because turnOn() can start off a chain that
        // calls RegisteringObjects(omh), so this ensures that the variables are in the correct
        // state if flow reaches there.
        syncBarrier->turnOn();
    }
  }
}


LDObjHandle LBDatabase::RegisterObj(LDOMHandle omh, CmiUInt8 id,
                                    void* userPtr, int migratable) {
#if CMK_LBDB_ON
  LDObjHandle newhandle;

  newhandle.omhandle = omh;
  newhandle.id = id;
#if CMK_GPU
  hapiCuptiObjectArrived(newhandle);   // back in this process: its kernels here are a live object's again
#endif

  {
    // objsEmptyHead maintains a linked list of empty positions within the objs array
    // If objsEmptyHead == LDObjEntry::DEFAULT_NEXT, there are no vacant positions, so add to the back
    // If objsEmptyHead != LDObjEntry::DEFAULT_NEXT, we place the new object at index objsEmptyHead and advance
    // objsEmptyHead to the next empty position.
    if (objsEmptyHead == LBObjEntry::DEFAULT_NEXT) {
      newhandle.handle = objs.size();
      LBObj *obj = new LBObj(newhandle, userPtr, migratable);
      objs.emplace_back(obj);
    } else {
      newhandle.handle = objsEmptyHead;
      LBObj *obj = new LBObj(newhandle, userPtr, migratable);
      objs[newhandle.handle].obj = obj;

      objsEmptyHead = objs[newhandle.handle].nextEmpty;
      objs[newhandle.handle].nextEmpty = LBObjEntry::DEFAULT_NEXT;
    }
  }

  return newhandle;
#endif
}

void LBDatabase::UnregisterObj(LDObjHandle h)
{
  delete objs[h.handle].obj;

  objs[h.handle].obj = nullptr;
  // Maintain the linked list of empty positions by adding the newly removed
  // index as the new objsEmptyHead
  objs[h.handle].nextEmpty = objsEmptyHead;
  objsEmptyHead = h.handle;
}

void LBDatabase::Send(const LDOMHandle &destOM, const CmiUInt8 &destID, unsigned int bytes, int destObjProc, int force)
{
#if CMK_LBDB_ON
  if (force || (StatsOn() && _lb_args.traceComm())) {
    LBCommData* item_ptr;

    auto *activeRec = CkActiveLocRec();
    if (activeRec) {
      const LDObjHandle &runObj = activeRec->getLdHandle();

      // Async objects keep sending after their compute measurement closes.
      // Use that same per-object window for traffic, including force=1 array
      // sends: force bypasses the PE-wide switch, not the object's epoch.
      if (LbObj(runObj)->hasJoinedStep()) return;

      // Don't record self-messages from an object to an object
      if (runObj.omhandle.id == destOM.id
          && runObj.id == destID )
        return;

      // In the future, we'll have to eliminate processor to same
      // processor messages as well

      LBCommData item(runObj, destOM.id, destID, destObjProc);
      item_ptr = commTable->HashInsertUnique(item);
    } else {
      LBCommData item(CkMyPe(), destOM.id, destID, destObjProc);
      item_ptr = commTable->HashInsertUnique(item);
    }
    item_ptr->addMessage(bytes);
  }
#endif
}

void LBDatabase::MulticastSend(const LDOMHandle &destOM, CmiUInt8 *destIDs, int nDests, unsigned int bytes, int nMsgs)
{
#if CMK_LBDB_ON
  if (StatsOn() && _lb_args.traceComm()) {
    LBCommData* item_ptr;

    auto *activeRec = CkActiveLocRec();
    if (activeRec) {
      const LDObjHandle &runObj = activeRec->getLdHandle();

      if (LbObj(runObj)->hasJoinedStep()) return;

      LBCommData item(runObj, destOM.id, destIDs, nDests);
      item_ptr = commTable->HashInsertUnique(item);
      item_ptr->addMessage(bytes, nMsgs);
    }
  }
#endif
}

// DistBaseLB: device loads forwarded from the process an object left (its pre-move kernels), added to the live
// object here. One pass over the objects; returns how many found their object.
int LBDatabase::AddDepartedGPULoads(const std::vector<std::pair<LDObjKey, double>>& loads)
{
  int applied = 0;
#if CMK_GPU
  if (loads.empty()) return 0;
  std::unordered_map<LDObjKey, double, LDObjKeyHash> byKey;
  for (const auto& kv : loads) byKey[kv.first] += kv.second;
  for (int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (obj == nullptr || obj->hasGPUDeclared()) continue;
    const LDObjHandle& h = obj->GetLDObjHandle();
    LDObjKey key; key.omID() = h.omID(); key.objID() = h.objID();
    auto it = byKey.find(key);
    if (it == byKey.end()) continue;
    LBRealType g = 0; obj->getGPUTime(&g);
    obj->setGPUTiming(g + it->second);
    applied++;
  }
#endif
  return applied;
}

void LBDatabase::TakeDepartedGPULoads()
{
#if CMK_GPU
  hapiCuptiTakeDepartedLoads(departed_pending_);
#endif
}

int LBDatabase::GetObjDataSz()
{
  int nitems = (int)departed_pending_.size();
  int i;
  if (_lb_args.migObjOnly()) {
  for(i = 0; i < objs.size(); i++)
    if (objs[i].obj && (objs[i].obj)->data.migratable)
      nitems++;
  } else {
  for(i = 0; i < objs.size(); i++)
    if (objs[i].obj)
      nitems++;
  }
  return nitems;
}

// The interval a strategy holds these loads against (LBLoadDim.h, the
// Diffusion cost model) is the wall time since ClearLoads. Under +LBAsync an
// object is measured over less of it: from its reopen once every migration
// has landed until its own join -- at leanmd's lag of 16 steps in a 20-step
// period, 20% of the interval; at sph2d's 800 in 2000, 60%, and 20% of the
// first Diffusion interval after Metis -- while it keeps computing through the
// lag. Held against the interval unscaled, such a load read the device as
// half busy when it was 85% busy, the criticality verdict flipped run to run,
// and the cost model undercharged every load gain by the same factor. The
// window itself stays as it is -- an object's rate depends on the device it
// runs on, so the transition must not be measured -- and the loads leave here
// scaled to the span the object was active (LBObj::activeSpan), traffic
// (GetCommData) by the same factor per sender, so load and traffic keep a
// common horizon. Under the barrier the span is the window itself and nothing
// changes. A load the application declared per interval (EstObjGPULoad) is
// already a full-interval figure and is left alone.
int ckLbParksSinceLast();   // cklocation.C: AtSyncWait parks on this PE since the last call
void LBDatabase::GetObjData(LDObjData *dp)
{
  LBRealType period = 0, cpu = 0;
  TotalTime(&period, &cpu);
  const bool migOnly = _lb_args.migObjOnly();
  int scaled = 0, counted = 0;
  double maxScale = 1.0, sumScale = 0.0;
  for (int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (!obj || (migOnly && !obj->data.migratable)) continue;
    LDObjData d = obj->ObjData();
    d.windowTime = obj->windowTime();
    const double f = obj->sampleScale();
    if (f != 1.0) {
      d.wallTime *= f;
#if CMK_LB_CPUTIMER
      d.cpuTime *= f;
#endif
#if CMK_GPU
      d.driverTime *= f;
      // Not gpuTime: an object's kernels are filed to the round between its joins whenever they were launched
      // (hapi_impl.cpp kCuptiDeferFlag), so its device load already spans the same steps for every object.
#endif
    }
    if (obj->data.migratable) {
      counted++;
      sumScale += f;
      if (f > maxScale) maxScale = f;
      if (f != 1.0) scaled++;
    }
    *dp++ = d;
  }
  // Objects that left this process after launching kernels here: their loads travel as departed entries
  // (handle index -1, not migratable); CentralLB::mergeDepartedLoads adds each to the object's live entry.
  for (const auto& kv : departed_pending_) {
    LDObjData d{};
    d.handle.omhandle.id = kv.first.omID();
    d.handle.id = kv.first.objID();
    d.handle.handle = -1;
    d.migratable = false;
    d.departed = true;
#if CMK_GPU
    d.gpuTime = kv.second;
#endif
    *dp++ = d;
  }
  if (_lb_args.debug() > 1 && !departed_pending_.empty()) CmiPrintf("[DEPART pe=%d] %zu departed entries appended to this PE's stats (version %d)\n", CkMyPe(), departed_pending_.size(), _lb_version);
  departed_pending_.clear();
  // Once per interval on PE 0 (the stats are read more than once per step);
  // the interval clock tells the reads of one step apart from the next.
  static thread_local double lastReported = -1.0;
  const bool newInterval = (lastReported < 0.0 || period < lastReported || period - lastReported > 0.5);
  if (_lb_args.debug() > 0 && scaled > 0 && CkMyPe() == 0 && newInterval) {
    CmiPrintf("[%d] LBDB: %d of %d migratable objects were measured over less of the "
              "%.3f s interval than they were active; loads and traffic scaled by "
              "%.2fx on average, %.2fx at most\n",
              CkMyPe(), scaled, counted, (double)period, sumScale / counted, maxScale);
  }
  // Per PE, once per interval: how much of the interval this PE's objects' windows covered. A PE whose
  // objects reopened late (parked at AtSyncWait until the step finished) samples a sliver and its loads,
  // capped at kMaxSampleScale, read low: one process of eight attributed 2-10% of its kernels against the
  // others' 18-25% (Frontier 2026-10-23, dbg3_async), and the diffusion filled the hole.
  if (_lb_args.debug() > 1 && counted > 0 && newInterval) {
    double sumWin = 0.0, minWin = 1e30, sumSpan = 0.0;
    int zeroWin = 0, capped = 0;
    for (int i = 0; i < objs.size(); i++) {
      LBObj* obj = objs[i].obj;
      if (!obj || !obj->data.migratable) continue;
      const double w = obj->windowTime(), sp = obj->activeSpan();
      sumWin += w; sumSpan += sp;
      if (w < minWin) minWin = w;
      if (w <= 0.0) zeroWin++;
      if (obj->sampleScale() >= LBObj::kMaxSampleScale) capped++;
    }
    CmiPrintf("[%d] LBDB-span: objs=%d window mean %.3f min %.3f (zero %d) span mean %.3f of %.3f s; "
              "scale mean %.2f max %.2f capped %d; parks %d\n",
              CkMyPe(), counted, sumWin / counted, minWin, zeroWin, sumSpan / counted, (double)period,
              sumScale / counted, maxScale, capped, ckLbParksSinceLast());
  }
  if (newInterval) lastReported = period;
}

// Traffic recorded while the sender's window was open, scaled to the interval
// by the sender's factor (see GetObjData). A record whose sender is gone, or
// whose handle now names another object, is handed out as recorded.
void LBDatabase::GetCommData(LDCommData *data)
{
  if (!commTable) return;
  commTable->GetCommData(data, [this](const LDObjHandle& h) -> double {
    if (h.handle < 0 || (size_t)h.handle >= objs.size()) return 1.0;
    LBObj* obj = objs[h.handle].obj;
    if (obj == NULL || obj->data.handle.id != h.id) return 1.0;
    return obj->sampleScale();
  });
}

void LBDatabase::BackgroundLoad(LBRealType* walltime, LBRealType* cputime)
{
  LBRealType total_walltime;
  LBRealType total_cputime;
  TotalTime(&total_walltime, &total_cputime);

  LBRealType idletime;
  IdleTime(&idletime);

  *walltime = total_walltime - idletime - obj_walltime;
  if (*walltime < 0) *walltime = 0.;
#if CMK_LB_CPUTIMER
  *cputime = total_cputime - obj_cputime;
#else
  *cputime = *walltime;
#endif
}

void LBDatabase::GetTime(LBRealType *total_walltime, LBRealType *total_cputime,
                         LBRealType *idletime, LBRealType *bg_walltime,
                         LBRealType *bg_cputime)
{
  TotalTime(total_walltime,total_cputime);

  IdleTime(idletime);

  *bg_walltime = *total_walltime - *idletime - obj_walltime;
  if (*bg_walltime < 0) *bg_walltime = 0.;
#if CMK_LB_CPUTIMER
  *bg_cputime = *total_cputime - obj_cputime;
#else
  *bg_cputime = *bg_walltime;
#endif
  //CkPrintf("HERE [%d] total: %f %f obj: %f %f idle: %f bg: %f\n", CkMyPe(), *total_walltime, *total_cputime, obj_walltime, obj_cputime, *idletime, *bg_walltime);
}

void LBDatabase::GetGPUBGTime(LBRealType *bg_gputime)
{
  // TODO: implement this properly
  *bg_gputime = 0;
}

void LBDatabase::ClearLoads(void)
{
  int i;
  const double now = CkWallTimer();
  for (i = 0; i < objs.size(); i++) {
    LBObj *obj = objs[i].obj;
    if (obj)
    {
      obj->clearWindow(now);
      if (obj->data.wallTime > 0.0) {
        obj->lastWallTime = obj->data.wallTime;
#if CMK_LB_CPUTIMER
        obj->lastCpuTime = obj->data.cpuTime;
#endif
      }
#if CMK_GPU
      // Unconditionally, unlike lastWallTime: an interval that measured no
      // GPU load reads back as 0, which is what it was.
      obj->lastGPUTime = obj->data.gpuTime;
      obj->data.gpuTime = 0.0;
      obj->setGPUDeclared(false);
#endif
      obj->data.wallTime = 0.0;
#if CMK_LB_CPUTIMER
      obj->data.cpuTime = 0.0;
#endif
    }
  }
  delete commTable;
  commTable = new LBCommTable;
  departed_pending_.clear();   // a round that did not take them (DistBaseLB) must never report stale departed entries
#if CMK_GPU
  hapiCuptiDropDepartedLoads();   // nor may the process's unclaimed ones reach a later CentralLB round
#endif
  machineUtil.Clear();
  obj_walltime = 0;
  obj_gputime = 0;
#if CMK_LB_CPUTIMER
  obj_cputime = 0;
#endif
}

int LBDatabase::Migrate(LDObjHandle h, int dest)
{
  if (h.handle >= objs.size()) {
    CmiAbort("[%d] LBDB::Migrate: Handle %d out of range 0-%zu\n",CkMyPe(),h.handle,objs.size());
  }
  else if (!(objs[h.handle].obj)) {
    // The object died between the stats collection the strategy planned from
    // and this decision arriving. That is legal, and the caller's contract is
    // to tell the destination to stop expecting it (CentralLB::MissMigrate).
    // Aborting here made that path unreachable.
    if (getenv("CHARM_DEBUG_MIGRATE") != nullptr) {
      CkPrintf("[LBMIG %d] handle %d no longer registered, reporting a missed migrate\n",
               CkMyPe(), h.handle);
      fflush(stdout);
    }
    return 0;
  }

  LBOM *const om = oms[(objs[h.handle].obj)->parentOM().handle];
#if CMK_GPU
  // Leaving this process: the kernels it launched here that are still in CUPTI's buffers are filed at the next
  // close and follow it as a departed entry (CentralLB::mergeDepartedLoads). Not UnregisterObj: an arrival
  // re-registers and would have been taken for a departure (job 23162022, 16:49).
  // Only when it leaves the PROCESS: a move to another PE of this process keeps its kernels in this process's
  // map, where SetObjGPULoad credits them to the live object. Marking those too (17:12, lmdfx7) split their whole
  // load out as departed entries that no DistBaseLB round takes back: DiffusionLB saw 24% of the device load
  // missing and moved 1180 objects.
  if (CkNodeOf(dest) != CkMyNode()) hapiCuptiObjectDeparting(h, dest);
#endif
  om->Migrate(h, dest);
  return 1;
}

void LBDatabase::MetaLBResumeWaitingChares(int lb_ideal_period) {
#if CMK_LBDB_ON
  for (int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (obj) {
      LBOM *om = oms[obj->parentOM().handle];
      LDObjHandle h = obj->GetLDObjHandle();
      om->MetaLBResumeWaitingChares(h, lb_ideal_period);
    }
  }
#endif
}

void LBDatabase::MetaLBCallLBOnChares() {
#if CMK_LBDB_ON
  for (int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (obj) {
      LBOM *om = oms[obj->parentOM().handle];
      LDObjHandle h = obj->GetLDObjHandle();
      om->MetaLBCallLBOnChares(h);
    }
  }
#endif
}

int LBDatabase::useMem() {
  int size = sizeof(LBDatabase);
  size += oms.size() * sizeof(LBOM);
  size += GetObjDataSz() * sizeof(LBObj);
  size += commTable->useMem();
  return size;
}

void LBDatabase::EstObjLoad(const LDObjHandle &_h, double cputime)
{
#if CMK_LBDB_ON
  LBObj *const obj = LbObj(_h);

  CmiAssert(obj != NULL);
  obj->setTiming(cputime);
#endif
}

void LBDatabase::SetObjJoinedStep(const LDObjHandle &_h, bool joined)
{
#if CMK_LBDB_ON
  LBObj *const obj = LbObj(_h);
  if (obj != NULL) obj->setJoinedStep(joined);
#endif
}

void LBDatabase::EstObjGPULoad(const LDObjHandle &_h, double gputime)
{
#if CMK_GPU
#if CMK_LBDB_ON
  LBObj *const obj = LbObj(_h);

  CmiAssert(obj != NULL);
  obj->data.gpuTime = gputime;
  obj->setGPUDeclared(true);
#endif
#else
    CmiAbort("LBDatabase::EstObjGPULoad called but CMK_GPU is not set");
#endif
}
