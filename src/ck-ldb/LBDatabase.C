#include "LBDatabase.h"
#include "cksyncbarrier.h"

#include "ck.h"

LBDatabase::LBDatabase() {
  omCount = omsRegistering = 0;
  obj_walltime = 0;
  statsAreOn = false;
  objsEmptyHead = -1;
  commTable = new LBCommTable;
  syncBarrier = CkSyncBarrier::object();
}

void LBDatabase::CollectStatsOn(void){
  if (!StatsOn()) {
    auto *activeRec = CkActiveLocRec();
    if (activeRec) {
      const LDObjHandle &runObj = activeRec->getLdHandle();
      LbObj(runObj)->StartTimer();
    }
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

      // An object keeps sending after its measurement window closes at its
      // join (+LBAsync). Its traffic uses the same window as its load,
      // including force=1 array sends: force bypasses the PE-wide switch, not
      // the object's window.
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
  if (nDests > 0 && StatsOn() && _lb_args.traceComm()) {
    LBCommData* item_ptr;

    auto *activeRec = CkActiveLocRec();
    if (activeRec) {
      const LDObjHandle &runObj = activeRec->getLdHandle();

      if (LbObj(runObj)->hasJoinedStep()) return;

      LBCommData item(runObj, destOM.id, destIDs, nDests);
      item_ptr = commTable->HashInsertUnique(item);
    } else {
      LBCommData item(CkMyPe(), destOM.id, destIDs, nDests);
      item_ptr = commTable->HashInsertUnique(item);
    }
    item_ptr->addMessage(bytes, nMsgs);
  }
#endif
}

int LBDatabase::GetObjDataSz()
{
  int nitems = 0;
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

// A strategy holds these loads against the interval, the wall time since
// ClearLoads. Under +LBAsync an object is measured over less of it: from its
// reopen, once the step's migrations are over, until its own join, while it
// keeps computing through the step. Held against the interval unscaled, such a
// load reads the object as idle for the part of the interval its window was
// closed. The window stays as it is -- an object's rate depends on where it
// runs, so the move must not be measured -- and the host load leaves here
// scaled to the span the object was active (LBObj::activeSpan), its traffic
// (GetCommData) by the same factor per sender, so load and traffic keep a
// common horizon. Under the barrier the span is the window itself and nothing
// changes. The device load is not scaled: it is the kernels an object launched
// while its window was open, the same number of steps for every object, and a
// per-object wall-clock factor would put a different error on each.
void LBDatabase::GetObjData(LDObjData *dp)
{
  const bool migOnly = _lb_args.migObjOnly();
  int scaled = 0, counted = 0, lateDeclared = 0;
  double maxScale = 1.0, sumScale = 0.0;
  for (int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (!obj || (migOnly && !obj->data.migratable)) continue;
    if (obj->declaredWhileJoined) lateDeclared++;
    LDObjData d = obj->ObjData();
    d.windowTime = obj->windowTime();
    const double f = obj->sampleScale();
    if (f != 1.0) {
      d.wallTime *= f;
#if CMK_LB_CPUTIMER
      d.cpuTime *= f;
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
  if (_lb_args.debug() > 0) {
    // Once per interval (the stats are read more than once per step); the
    // interval clock tells the reads of one step from those of the next.
    LBRealType period = 0, cpu = 0;
    TotalTime(&period, &cpu);
    static thread_local double lastReported = -1.0;
    const bool newInterval =
        (lastReported < 0.0 || period < lastReported || period - lastReported > 0.5);
    if (newInterval && scaled > 0 && CkMyPe() == 0)
      CmiPrintf("[%d] LBDB: %d of %d migratable objects were measured over less of the "
                "%.3f s interval than they were active; loads and traffic scaled by "
                "%.2fx on average, %.2fx at most\n",
                CkMyPe(), scaled, counted, (double)period, sumScale / counted, maxScale);
    if (newInterval && lateDeclared > 0)
      CmiPrintf("[%d] LBDB: %d of %d objects declared a device load after joining the "
                "step; this round reads their declaration from before the join\n",
                CkMyPe(), lateDeclared, (int)objs.size());
    if (newInterval) lastReported = period;
  }
}

// Traffic recorded while the sender's window was open, scaled by the sender's
// factor (see GetObjData). A record whose sender is gone, or whose handle now
// names another object, is handed out as recorded.
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
      obj->data.wallTime = 0.0;
#if CMK_LB_CPUTIMER
      obj->data.cpuTime = 0.0;
#endif
#if CMK_CUDA
      obj->data.gpuTime = 0.0;
#endif
    }
  }
  delete commTable;
  commTable = new LBCommTable;
  machineUtil.Clear();
  obj_walltime = 0;
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
    // The object died between the statistics the strategy planned from and
    // this decision. That is legal -- more so under +LBAsync, where elements
    // keep running through the step -- and the caller tells the destination to
    // stop expecting it (CentralLB::MissMigrate).
    return 0;
  }

  LBOM *const om = oms[(objs[h.handle].obj)->parentOM().handle];
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
#if CMK_CUDA && CMK_LBDB_ON
  LBObj *const obj = LbObj(_h);

  CmiAssert(obj != NULL);
  // The window is closed from the join to the resume, and a declaration made
  // in it is about work the round must not see, as a launch is: under
  // +LBAsync the elements keep stepping while the round's stats are built, and
  // a later step's declaration would replace the one the round stands for.
  if (obj->hasJoinedStep()) {
    obj->declaredWhileJoined = true;
    return;
  }
  obj->setGPUTiming(gputime);
#else
  CmiAbort("LBDatabase::EstObjGPULoad called but CMK_CUDA is not set");
#endif
}
