/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>

#if CMK_LBDB_ON

#include "LBDBManager.h"

struct MigrateCB;

/*************************************************************
 * Set up the builtin barrier-- the load balancer needs somebody
 * to call AtSync on each PE in case there are no atSync array 
 * elements.  The builtin-atSync caller (batsyncer) does this.
 */

//Called periodically-- starts next load balancing cycle
void LBDB::batsyncer::gotoSync(void *bs)
{
  LBDB::batsyncer *s=(LBDB::batsyncer *)bs;
  s->gotoSyncCalled = true;
  s->db->AtLocalBarrier(s->BH);
}
//Called at end of each load balancing cycle
void LBDB::batsyncer::resumeFromSync(void *bs)
{
  LBDB::batsyncer *s=(LBDB::batsyncer *)bs;
//  CmiPrintf("[%d] LBDB::batsyncer::resumeFromSync with %gs\n", CkMyPe(), s->period);

#if 0
  double curT = CmiWallTimer();
  if (s->nextT<curT)  s->period *= 2;
  s->nextT = curT + s->period;
#endif

  if (s->gotoSyncCalled) {
    CcdCallFnAfterOnPE((CcdVoidFn)gotoSync, (void *)s, 1000*s->period, CkMyPe());
    s->gotoSyncCalled = false;
  }
}

// initPeriod in seconds
void LBDB::batsyncer::init(LBDB *_db,double initPeriod)
{
  db=_db;
  period=initPeriod;
  nextT = CmiWallTimer() + period;
  BH = db->AddLocalBarrierClient((LDResumeFn)resumeFromSync,(void*)(this));
  gotoSyncCalled = true;
  //This just does a CcdCallFnAfter
  resumeFromSync((void *)this);
}


/*************************************************************
 * LBDB Code
 *************************************************************/

LBDB::LBDB(): useBarrier(true)
{
    statsAreOn = false;
    omCount = oms_registering = 0;
    obj_running = false;
    commTable = new LBCommTable;
    obj_walltime = 0;
#if CMK_LB_CPUTIMER
    obj_cputime = 0;
#endif
    startLBFn_count = 0;
    predictCBFn = NULL;
    batsync.init(this, _lb_args.lbperiod());	    // original 1.0 second
    objsEmptyHead = -1;
}

LDOMHandle LBDB::AddOM(LDOMid _userID, void* _userData, 
		       LDCallbacks _callbacks)
{
  LDOMHandle newhandle;

  newhandle.ldb.handle = (void*)(this);
//  newhandle.user_ptr = _userData;
  newhandle.id = _userID;

  LBOM* om = new LBOM(this,_userID,_userData,_callbacks);
  if (om != NULL) {
    newhandle.handle = oms.length();
    oms.insertAtEnd(om);
  } else newhandle.handle = -1;
  om->DepositHandle(newhandle);
  omCount++;
  return newhandle;
}

void LBDB::RemoveOM(LDOMHandle om)
{
  delete oms[om.handle];
  oms[om.handle] = NULL;
  omCount--;
}


#if CMK_BIGSIM_CHARM
#define LBOBJ_OOC_IDX 0x1
#endif

LDObjHandle LBDB::AddObj(LDOMHandle _omh, CmiUInt8 _id,
			 void *_userData, bool _migratable)
{
  LDObjHandle newhandle;

  newhandle.omhandle = _omh;
//  newhandle.user_ptr = _userData;
  newhandle.id = _id;

#if CMK_BIGSIM_CHARM
  if(_BgOutOfCoreFlag==2){ //taking object into memory
    //first find the first (LBOBJ_OOC_IDX) in objs and insert the object at that position
    int newpos = -1;
    for(int i=0; i<objs.size(); i++){
	if(objs[i].obj==(LBObj *)LBOBJ_OOC_IDX){
	    newpos = i;
	    break;
	}
    }
    if(newpos==-1) newpos = objs.size();
    newhandle.handle = newpos;
    LBObj *obj = new LBObj(newhandle, _userData, _migratable);
    if (newpos == -1) {
      objs.emplace_back(obj);
    } else {
      objs[newpos].obj = obj;
    }
    //objCount is not increased since it's the original object which is pupped
    //through out-of-core emulation. 
    //objCount++;
  }else
#endif
  {
    //BIGSIM_OOC DEBUGGING
    //CkPrintf("Proc[%d]: In AddObj for real migration\n", CkMyPe());

    // objsEmptyHead maintains a linked list of empty positions within the objs array.
    // If objsEmptyHead is -1, then there are no vacant positions, so we just add to the back.
    // If objsEmptyHead is > -1, then we place the new object at index objsEmptyHead and advance
    // objsEmptyHead to the next empty position.
    if (objsEmptyHead == -1) {
      newhandle.handle = objs.size();
      LBObj *obj = new LBObj(newhandle, _userData, _migratable);
      objs.emplace_back(obj);
    } else {
      newhandle.handle = objsEmptyHead;
      LBObj *obj = new LBObj(newhandle, _userData, _migratable);
      objs[objsEmptyHead].obj = obj;

      objsEmptyHead = objs[objsEmptyHead].next;
    }
  }
  //BIGSIM_OOC DEBUGGING
  //CkPrintf("LBDBManager.C: New handle: %d, LBObj=%p\n", newhandle.handle, objs[newhandle.handle]);

  return newhandle;
}

void LBDB::UnregisterObj(LDObjHandle _h)
{
//  (objs[_h.handle])->registered=false;
// free the memory, it is a memory leak.
// CmiPrintf("[%d] UnregisterObj: %d\n", CkMyPe(), _h.handle);
  delete objs[_h.handle].obj;

#if CMK_BIGSIM_CHARM
  //hack for BigSim out-of-core emulation.
  //we want the chare array object to keep at the same
  //position even going through the pupping routine.
  if(_BgOutOfCoreFlag==1){ //in taking object out of memory
    objs[_h.handle].obj = (LBObj *)(LBOBJ_OOC_IDX);
  }else
#endif
  {
    objs[_h.handle].obj = NULL;
    // Maintain the linked list of empty positions by adding the newly removed
    // index as the new objsEmptyHead
    objs[_h.handle].next = objsEmptyHead;
    objsEmptyHead = _h.handle;
  }
}

void LBDB::RegisteringObjects(LDOMHandle _h)
{
  // for an unregistered anonymous OM to join and control the barrier
  if (_h.id.id.idx == 0) {
    if (oms_registering == 0)
      localBarrier.TurnOff();
    oms_registering++;
  }
  else {
  LBOM* om = oms[_h.handle];
  if (!om->RegisteringObjs()) {
    if (oms_registering == 0)
      localBarrier.TurnOff();
    oms_registering++;
    om->SetRegisteringObjs(true);
  }
  }
}

void LBDB::DoneRegisteringObjects(LDOMHandle _h)
{
  // for an unregistered anonymous OM to join and control the barrier
  if (_h.id.id.idx == 0) {
    oms_registering--;
    if (oms_registering == 0 && useBarrier)
      localBarrier.TurnOn();
  }
  else {
  LBOM* om = oms[_h.handle];
  if (om->RegisteringObjs()) {
    oms_registering--;
    if (oms_registering == 0 && useBarrier)
      localBarrier.TurnOn();
    om->SetRegisteringObjs(false);
  }
  }
}


void LBDB::Send(const LDOMHandle &destOM, const CmiUInt8 &destid, unsigned int bytes, int destObjProc)
{
  LBCommData* item_ptr;

  if (obj_running) {
    const LDObjHandle &runObj = RunningObj();

    // Don't record self-messages from an object to an object
    if ( LDOMidEqual(runObj.omhandle.id,destOM.id)
	 && runObj.id == destid )
      return;

    // In the future, we'll have to eliminate processor to same 
    // processor messages as well

    LBCommData item(runObj,destOM.id,destid, destObjProc);
    item_ptr = commTable->HashInsertUnique(item);
  } else {
    LBCommData item(CkMyPe(),destOM.id,destid, destObjProc);
    item_ptr = commTable->HashInsertUnique(item);
  }  
  item_ptr->addMessage(bytes);
}

void LBDB::MulticastSend(const LDOMHandle &destOM, CmiUInt8 *destids, int ndests, unsigned int bytes, int nMsgs)
{
  LBCommData* item_ptr;
  //CmiAssert(obj_running);
  if (obj_running) {
    const LDObjHandle &runObj = RunningObj();

    LBCommData item(runObj,destOM.id,destids, ndests);
    item_ptr = commTable->HashInsertUnique(item);
    item_ptr->addMessage(bytes, nMsgs);
  }
}

void LBDB::ClearLoads(void)
{
  int i;
  for(i=0; i < objs.size(); i++) {
    LBObj *obj = objs[i].obj;
    if (obj)
    {
      if (obj->data.wallTime>.0) {
        obj->lastWallTime = obj->data.wallTime;
#if CMK_LB_CPUTIMER
        obj->lastCpuTime = obj->data.cpuTime;
#endif
      }
      obj->data.wallTime = 0.;
#if CMK_LB_CPUTIMER
      obj->data.cpuTime = 0.;
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

int LBDB::ObjDataCount()
{
  int nitems=0;
  int i;
  if (_lb_args.migObjOnly()) {
  for(i=0; i < objs.size(); i++)
    if (objs[i].obj && (objs[i].obj)->data.migratable)
      nitems++;
  }
  else {
  for(i=0; i < objs.size(); i++)
    if (objs[i].obj)
      nitems++;
  }
  return nitems;
}

void LBDB::GetObjData(LDObjData *dp)
{
  if (_lb_args.migObjOnly()) {
  for(int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (obj && obj->data.migratable)
      *dp++ = obj->ObjData();
  }
  }
  else {
  for(int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (obj)
      *dp++ = obj->ObjData();
  }
  }
}

int LBDB::Migrate(LDObjHandle h, int dest)
{
    //BIGSIM_OOC DEBUGGING
    //CmiPrintf("[%d] LBDB::Migrate: incoming handle %d with handle range 0-%d\n", CkMyPe(), h.handle, objCount);

  if (h.handle >= objs.size()) {
    CkAbort("[%d] LBDB::Migrate: Handle %d out of range 0-%zu\n",CkMyPe(),h.handle,objs.size());
  }
  else if (!(objs[h.handle].obj)) {
    CkAbort("[%d] LBDB::Migrate: Handle %d no longer registered, range 0-%zu\n", CkMyPe(),h.handle,objs.size());
  }

  LBOM *const om = oms[(objs[h.handle].obj)->parentOM().handle];
  om->Migrate(h, dest);
  return 1;
}

void LBDB::MetaLBResumeWaitingChares(int lb_ideal_period) {
  for (int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (obj) {
      LBOM *om = oms[obj->parentOM().handle];
      LDObjHandle h = obj->GetLDObjHandle();
      om->MetaLBResumeWaitingChares(h, lb_ideal_period);
    }
  }
}

void LBDB::MetaLBCallLBOnChares() {
  for (int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (obj) {
      LBOM *om = oms[obj->parentOM().handle];
      LDObjHandle h = obj->GetLDObjHandle();
      om->MetaLBCallLBOnChares(h);
    }
  }
}

void LBDB::Migrated(LDObjHandle h, int waitBarrier)
{
  // Object migrated, inform load balancers

  // subtle: callback may change (on) when switching LBs
  // call in reverse order
  //for(int i=0; i < migrateCBList.length(); i++) {
  for(int i=migrateCBList.length()-1; i>=0; i--) {
    MigrateCB* cb = (MigrateCB*)migrateCBList[i];
    if (cb && cb->on) (cb->fn)(cb->data,h,waitBarrier);
  }
  
}


int LBDB::NotifyMigrated(LDMigratedFn fn, void* data)
{
  // Save migration function
  MigrateCB* callbk = new MigrateCB;

  callbk->fn = fn;
  callbk->data = data;
  callbk->on = 1;
  migrateCBList.insertAtEnd(callbk);
  return migrateCBList.size()-1;
}

void LBDB::RemoveNotifyMigrated(int handle)
{
  MigrateCB* callbk = migrateCBList[handle];
  migrateCBList[handle] = NULL;
  delete callbk;
}

int LBDB::AddStartLBFn(LDStartLBFn fn, void* data)
{
  // Save startLB function
  StartLBCB* callbk = new StartLBCB;

  callbk->fn = fn;
  callbk->data = data;
  callbk->on = 1;
  startLBFnList.push_back(callbk);
  startLBFn_count++;
  return startLBFnList.size()-1;
}

void LBDB::RemoveStartLBFn(LDStartLBFn fn)
{
  for (int i=0; i<startLBFnList.length(); i++) {
    StartLBCB* callbk = startLBFnList[i];
    if (callbk && callbk->fn == fn) {
      delete callbk;
      startLBFnList[i] = 0; 
      startLBFn_count --;
      break;
    }
  }
}

void LBDB::StartLB() 
{
  if (startLBFn_count == 0) {
    CmiAbort("StartLB is not supported in this LB");
  }
  for (int i=0; i<startLBFnList.length(); i++) {
    StartLBCB *startLBFn = startLBFnList[i];
    if (startLBFn && startLBFn->on) startLBFn->fn(startLBFn->data);
  }
}

int LBDB::AddMigrationDoneFn(LDMigrationDoneFn fn, void* data) {
  // Save migrationDone callback function
  MigrationDoneCB* callbk = new MigrationDoneCB;

  callbk->fn = fn;
  callbk->data = data;
  migrationDoneCBList.push_back(callbk);
  return migrationDoneCBList.size()-1;
}

void LBDB::RemoveMigrationDoneFn(LDMigrationDoneFn fn) {
  for (int i=0; i<migrationDoneCBList.length(); i++) {
    MigrationDoneCB* callbk = migrationDoneCBList[i];
    if (callbk && callbk->fn == fn) {
      delete callbk;
      migrationDoneCBList[i] = 0; 
      break;
    }
  }
}

void LBDB::MigrationDone() {
  for (int i=0; i<migrationDoneCBList.length(); i++) {
    MigrationDoneCB *callbk = migrationDoneCBList[i];
    if (callbk) callbk->fn(callbk->data);
  }
}

void LBDB::SetupPredictor(LDPredictModelFn on, LDPredictWindowFn onWin, LDPredictFn off, LDPredictModelFn change, void* data)
{
  if (predictCBFn==NULL) predictCBFn = new PredictCB;
  predictCBFn->on = on;
  predictCBFn->onWin = onWin;
  predictCBFn->off = off;
  predictCBFn->change = change;
  predictCBFn->data = data;
}

void LBDB::BackgroundLoad(LBRealType* bg_walltime, LBRealType* bg_cputime)
{
  LBRealType total_walltime;
  LBRealType total_cputime;
  TotalTime(&total_walltime, &total_cputime);

  LBRealType idletime;
  IdleTime(&idletime);

  *bg_walltime = total_walltime - idletime - obj_walltime;
  if (*bg_walltime < 0) *bg_walltime = 0.;
#if CMK_LB_CPUTIMER
  *bg_cputime = total_cputime - obj_cputime;
#else
  *bg_cputime = *bg_walltime;
#endif
}

void LBDB::GetTime(LBRealType *total_walltime,LBRealType *total_cputime,
                   LBRealType *idletime, LBRealType *bg_walltime, LBRealType *bg_cputime)
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

void LBDB::DumpDatabase()
{
#ifdef DEBUG  
  CmiPrintf("Database contains %d object managers\n",omCount);
  CmiPrintf("Database contains %d objects\n",objs.size());
#endif
}

int LBDB::useMem() {
  int size = sizeof(LBDB);
  size += oms.length() * sizeof(LBOM);
  size += ObjDataCount() * sizeof(LBObj);
  size += migrateCBList.length() * sizeof(MigrateCBList);
  size += startLBFnList.length() * sizeof(StartLBCB);
  size += commTable->useMem();
  return size;
}

class client {
  friend class LocalBarrier;
  void* data;
  LDResumeFn fn;
  int refcount;
};
class receiver {
  friend class LocalBarrier;
  void* data;
  LDBarrierFn fn;
  int on;
};

LDBarrierClient LocalBarrier::AddClient(LDResumeFn fn, void* data)
{
  client* new_client = new client;
  new_client->fn = fn;
  new_client->data = data;
  new_client->refcount = cur_refcount;

  CmiPrintf("[%d][%d][%d] LocalBarrier::AddClient and Data is %p\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), data);

#if CMK_BIGSIM_CHARM
  if(_BgOutOfCoreFlag!=2){
    //during out-of-core emualtion for BigSim, if taking procs from disk to mem,
    //client_count should not be increased
    client_count++;
  }
#else  
  client_count++;
#endif

  return LDBarrierClient(clients.insert(clients.end(), new_client));
}

void LocalBarrier::RemoveClient(LDBarrierClient c)
{
  delete *(c.i);
  clients.erase(c.i);

#if CMK_BIGSIM_CHARM
  //during out-of-core emulation for BigSim, if taking procs from mem to disk,
  //client_count should not be increased
  if(_BgOutOfCoreFlag!=1)
  {
    client_count--;
  }
#else
  client_count--;
#endif
}

LDBarrierReceiver LocalBarrier::AddReceiver(LDBarrierFn fn, void* data)
{
  receiver* new_receiver = new receiver;
  new_receiver->fn = fn;
  new_receiver->data = data;
  new_receiver->on = 1;

  return LDBarrierReceiver(receivers.insert(receivers.end(), new_receiver));
}

void LocalBarrier::RemoveReceiver(LDBarrierReceiver c)
{
  delete *(c.i);
  receivers.erase(c.i);
}

void LocalBarrier::TurnOnReceiver(LDBarrierReceiver c)
{
  (*c.i)->on = 1;
}

void LocalBarrier::TurnOffReceiver(LDBarrierReceiver c)
{
  (*c.i)->on = 0;
}

void LocalBarrier::AtBarrier(LDBarrierClient h)
{
  (*h.i)->refcount++;
  at_count++;
  CheckBarrier();
}

void LocalBarrier::DecreaseBarrier(LDBarrierClient h, int c)
{
  at_count-=c;
}

void LocalBarrier::CheckBarrier()
{
  if (!on) return;

  // If there are no clients, resume as soon as we're turned on
  if (client_count == 0) {
    cur_refcount++;
    CallReceivers();
  }

  // If there have been enough AtBarrier calls, check to see if all clients have
  // made it to the barrier. It's possible to have gotten multiple AtSync calls
  // from a single client, which is why this check is necessary.
  if (at_count >= client_count) {
    bool at_barrier = true;

    for (auto& c : clients) {
      if (c->refcount < cur_refcount) {
        at_barrier = false;
        break;
      }
    }

    if (at_barrier) {
      at_count -= client_count;
      cur_refcount++;
      CallReceivers();
    }
  }
}

void LocalBarrier::CallReceivers(void)
{
  bool called_receiver=false;

  for (std::list<receiver *>::iterator i = receivers.begin();
       i != receivers.end(); ++i) {
    receiver *recv = *i;
    if (recv->on) {
      recv->fn(recv->data);
      called_receiver = true;
    }
  }

  if (!called_receiver)
    ResumeClients();
}

void LocalBarrier::ResumeClients(void)
{
  for (std::list<client *>::iterator i = clients.begin(); i != clients.end(); ++i)
    (*i)->fn((*i)->data);
}

#endif

/*@}*/
