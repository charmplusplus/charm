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

  CcdCallFnAfterOnPE((CcdVoidFn)gotoSync, (void *)s, 1000*s->period, CkMyPe());
}

// initPeriod in seconds
void LBDB::batsyncer::init(LBDB *_db,double initPeriod)
{
  db=_db;
  period=initPeriod;
  nextT = CmiWallTimer() + period;
  BH = db->AddLocalBarrierClient((LDResumeFn)resumeFromSync,(void*)(this));
  //This just does a CcdCallFnAfter
  resumeFromSync((void *)this);
}


/*************************************************************
 * LBDB Code
 *************************************************************/

LBDB::LBDB(): useBarrier(CmiTrue)
{
    statsAreOn = CmiFalse;
    omCount = objCount = oms_registering = 0;
    obj_running = CmiFalse;
    commTable = new LBCommTable;
    obj_walltime = 0;
#if CMK_LB_CPUTIMER
    obj_cputime = 0;
#endif
    startLBFn_count = 0;
    predictCBFn = NULL;
    batsync.init(this, _lb_args.lbperiod());	    // original 1.0 second
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

#if CMK_BIGSIM_CHARM
#define LBOBJ_OOC_IDX 0x1
#endif

LDObjHandle LBDB::AddObj(LDOMHandle _omh, LDObjid _id,
			 void *_userData, CmiBool _migratable)
{
  LDObjHandle newhandle;

  newhandle.omhandle = _omh;
//  newhandle.user_ptr = _userData;
  newhandle.id = _id;
  
#if 1
#if CMK_BIGSIM_CHARM
  if(_BgOutOfCoreFlag==2){ //taking object into memory
    //first find the first (LBOBJ_OOC_IDX) in objs and insert the object at that position
    int newpos = -1;
    for(int i=0; i<objs.length(); i++){
	if(objs[i]==(LBObj *)LBOBJ_OOC_IDX){
	    newpos = i;
	    break;
	}
    }
    if(newpos==-1) newpos = objs.length();
    newhandle.handle = newpos;
    LBObj *obj = new LBObj(this, newhandle, _userData, _migratable);
    objs.insert(newpos, obj);
    //objCount is not increased since it's the original object which is pupped
    //through out-of-core emulation. 
    //objCount++;
  }else
#endif
  {
    //BIGSIM_OOC DEBUGGING
    //CkPrintf("Proc[%d]: In AddObj for real migration\n", CkMyPe());
    newhandle.handle = objs.length();
    LBObj *obj = new LBObj(this, newhandle, _userData, _migratable);
    objs.insertAtEnd(obj);
    objCount++;
  }
  //BIGSIM_OOC DEBUGGING
  //CkPrintf("LBDBManager.C: New handle: %d, LBObj=%p\n", newhandle.handle, objs[newhandle.handle]);

#else
  LBObj *obj = new LBObj(this,_omh,_id,_userData,_migratable);
  if (obj != NULL) {
    newhandle.handle = objs.length();
    objs.insertAtEnd(obj);
  } else {
    newhandle.handle = -1;
  }
  obj->DepositHandle(newhandle);
#endif
  return newhandle;
}

void LBDB::UnregisterObj(LDObjHandle _h)
{
//  (objs[_h.handle])->registered=CmiFalse;
// free the memory, it is a memory leak.
// CmiPrintf("[%d] UnregisterObj: %d\n", CkMyPe(), _h.handle);
  delete objs[_h.handle];

#if CMK_BIGSIM_CHARM
  //hack for BigSim out-of-core emulation.
  //we want the chare array object to keep at the same
  //position even going through the pupping routine.
  if(_BgOutOfCoreFlag==1){ //in taking object out of memory
    objs[_h.handle] = (LBObj *)(LBOBJ_OOC_IDX);
  }else
#endif
  {
    objs[_h.handle] = NULL;
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
    om->SetRegisteringObjs(CmiTrue);
  }
  }
}

void LBDB::DoneRegisteringObjects(LDOMHandle _h)
{
  // for an unregistered anonymous OM to join and control the barrier
  if (_h.id.id.idx == 0) {
    oms_registering--;
    if (oms_registering == 0)
      localBarrier.TurnOn();
  }
  else {
  LBOM* om = oms[_h.handle];
  if (om->RegisteringObjs()) {
    oms_registering--;
    if (oms_registering == 0)
      localBarrier.TurnOn();
    om->SetRegisteringObjs(CmiFalse);
  }
  }
}


void LBDB::Send(const LDOMHandle &destOM, const LDObjid &destid, unsigned int bytes, int destObjProc)
{
  LBCommData* item_ptr;

  if (obj_running) {
    const LDObjHandle &runObj = RunningObj();

    // Don't record self-messages from an object to an object
    if ( LDOMidEqual(runObj.omhandle.id,destOM.id)
	 && LDObjIDEqual(runObj.id,destid) )
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

void LBDB::MulticastSend(const LDOMHandle &destOM, LDObjid *destids, int ndests, unsigned int bytes, int nMsgs)
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
  for(i=0; i < objCount; i++) {
    LBObj *obj = objs[i]; 
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
  for(i=0; i < objCount; i++)
    if (objs[i] && (objs[i])->data.migratable)
      nitems++;
  }
  else {
  for(i=0; i < objCount; i++)
    if (objs[i])
      nitems++;
  }
  return nitems;
}

void LBDB::GetObjData(LDObjData *dp)
{
  if (_lb_args.migObjOnly()) {
  for(int i = 0; i < objs.length(); i++) {
    LBObj* obj = objs[i];
    if ( obj && obj->data.migratable)
      *dp++ = obj->ObjData();
  }
  }
  else {
  for(int i = 0; i < objs.length(); i++) {
    LBObj* obj = objs[i];
    if (obj)
      *dp++ = obj->ObjData();
  }
  }
}

int LBDB::Migrate(LDObjHandle h, int dest)
{
    //BIGSIM_OOC DEBUGGING
    //CmiPrintf("[%d] LBDB::Migrate: incoming handle %d with handle range 0-%d\n", CkMyPe(), h.handle, objCount);

  if (h.handle > objCount)
    CmiPrintf("[%d] LBDB::Migrate: Handle %d out of range 0-%d\n",CkMyPe(),h.handle,objCount);
  else if (!objs[h.handle]) {
    CmiPrintf("[%d] LBDB::Migrate: Handle %d no longer registered, range 0-%d\n", CkMyPe(),h.handle,objCount);
    return 0;
  }

  if ((h.handle < objCount) && objs[h.handle]) {
    LBOM *const om = oms[(objs[h.handle])->parentOM().handle];
    om->Migrate(h, dest);
  }
  return 1;
}

void LBDB::MetaLBResumeWaitingChares(int lb_ideal_period) {
  for (int i = 0; i < objs.length(); i++) {
    LBObj* obj = objs[i];
    if (obj) {
      LBOM *om = oms[obj->parentOM().handle];
      LDObjHandle h = obj->GetLDObjHandle();
      om->MetaLBResumeWaitingChares(h, lb_ideal_period);
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
  CmiPrintf("Database contains %d objects\n",objCount);
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

LDBarrierClient LocalBarrier::AddClient(LDResumeFn fn, void* data)
{
  client* new_client = new client;
  new_client->fn = fn;
  new_client->data = data;
  new_client->refcount = cur_refcount;

  LDBarrierClient ret_val;
#if CMK_BIGSIM_CHARM
  ret_val.serial = first_free_client_slot;
  clients.insert(ret_val.serial, new_client);

  //looking for the next first free client slot
  int nextfree=-1;
  for(int i=first_free_client_slot+1; i<clients.size(); i++)
    if(clients[i]==NULL) { nextfree = i; break; }
  if(nextfree==-1) nextfree = clients.size();
  first_free_client_slot = nextfree;

  if(_BgOutOfCoreFlag!=2){
    //during out-of-core emualtion for BigSim, if taking procs from disk to mem,
    //client_count should not be increased
    client_count++;
  }

#else  
  //ret_val.serial = max_client;
  ret_val.serial = clients.size();
  clients.insertAtEnd(new_client);
  //max_client++;
  client_count++;
#endif

  return ret_val;
}

void LocalBarrier::RemoveClient(LDBarrierClient c)
{
  const int cnum = c.serial;
#if CMK_BIGSIM_CHARM
  if (cnum < clients.size() && clients[cnum] != 0) {
    delete (clients[cnum]);
    clients[cnum] = 0;

    if(cnum<=first_free_client_slot) first_free_client_slot = cnum;

    if(_BgOutOfCoreFlag!=1){
	//during out-of-core emulation for BigSim, if taking procs from mem to disk,
	//client_count should not be increased
	client_count--;
    }
  }
#else
  //if (cnum < max_client && clients[cnum] != 0) {
  if (cnum < clients.size() && clients[cnum] != 0) {
    delete (clients[cnum]);
    clients[cnum] = 0;
    client_count--;
  }
#endif
}

LDBarrierReceiver LocalBarrier::AddReceiver(LDBarrierFn fn, void* data)
{
  receiver* new_receiver = new receiver;
  new_receiver->fn = fn;
  new_receiver->data = data;
  new_receiver->on = 1;

  LDBarrierReceiver ret_val;
//  ret_val.serial = max_receiver;
  ret_val.serial = receivers.size();
  receivers.insertAtEnd(new_receiver);
//  max_receiver++;

  return ret_val;
}

void LocalBarrier::RemoveReceiver(LDBarrierReceiver c)
{
  const int cnum = c.serial;
  //if (cnum < max_receiver && receivers[cnum] != 0) {
  if (cnum < receivers.size() && receivers[cnum] != 0) {
    delete (receivers[cnum]);
    receivers[cnum] = 0;
  }
}

void LocalBarrier::TurnOnReceiver(LDBarrierReceiver c)
{
  const int cnum = c.serial;
  //if (cnum < max_receiver && receivers[cnum] != 0) {
  if (cnum < receivers.size() && receivers[cnum] != 0) {
    receivers[cnum]->on = 1;
  }
}

void LocalBarrier::TurnOffReceiver(LDBarrierReceiver c)
{
  const int cnum = c.serial;
  //if (cnum < max_receiver && receivers[cnum] != 0) {
  if (cnum < receivers.size() && receivers[cnum] != 0) {
    receivers[cnum]->on = 0;
  }
}

void LocalBarrier::AtBarrier(LDBarrierClient h)
{
  (clients[h.serial])->refcount++;
  at_count++;
  CheckBarrier();
}

void LocalBarrier::CheckBarrier()
{
  if (!on) return;

  // If there are no clients, resume as soon as we're turned on

  if (client_count == 0) {
    cur_refcount++;
    CallReceivers();
  }
  if (at_count >= client_count) {
    CmiBool at_barrier = CmiFalse;

//    for(int i=0; i < max_client; i++)
    for(int i=0; i < clients.size(); i++)
      if (clients[i] != 0 && ((client*)clients[i])->refcount >= cur_refcount)
	at_barrier = CmiTrue;
		
    if (at_barrier) {
      at_count -= client_count;
      cur_refcount++;
      CallReceivers();
    }
  }
}

void LocalBarrier::CallReceivers(void)
{
  CmiBool called_receiver=CmiFalse;

//  for(int i=0; i < max_receiver; i++)
//   for (int i=max_receiver-1; i>=0; i--) {
   for (int i=receivers.size()-1; i>=0; i--) {
      receiver *recv = receivers[i];
      if (recv != 0 && recv->on) {
        recv->fn(recv->data);
        called_receiver = CmiTrue;
      }
  }

  if (!called_receiver)
    ResumeClients();
  
}

void LocalBarrier::ResumeClients(void)
{
//  for(int i=0; i < max_client; i++)
  for(int i=0; i < clients.size(); i++)
    if (clients[i] != 0) {
      ((client*)clients[i])->fn(((client*)clients[i])->data);
    }	
}

#endif

/*@}*/
