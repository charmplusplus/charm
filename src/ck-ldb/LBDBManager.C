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

#if CMK_LBDB_ON

#include <iostream.h>
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
  CcdCallFnAfterOnPE((CcdVoidFn)gotoSync,(void *)s,(int)(1000*s->period), CkMyPe());
}

void LBDB::batsyncer::init(LBDB *_db,double initPeriod)
{
  db=_db;
  period=initPeriod;
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
    obj_walltime = obj_cputime = 0;
    batsync.init(this,1.0);
    startLBFn_count = 0;
}

LDOMHandle LBDB::AddOM(LDOMid _userID, void* _userData, 
		       LDCallbacks _callbacks)
{
  LDOMHandle newhandle;

  newhandle.ldb.handle = (void*)(this);
  newhandle.user_ptr = _userData;
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

LDObjHandle LBDB::AddObj(LDOMHandle _omh, LDObjid _id,
			 void *_userData, CmiBool _migratable)
{
  LDObjHandle newhandle;

  newhandle.omhandle = _omh;
  newhandle.user_ptr = _userData;
  newhandle.id = _id;
  
#if 1
  newhandle.handle = objs.length();
  LBObj *obj = new LBObj(this, newhandle, _migratable);
  objs.insertAtEnd(obj);
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
  objCount++;
  return newhandle;
}

void LBDB::UnregisterObj(LDObjHandle _h)
{
  (objs[_h.handle])->registered=CmiFalse;
}

void LBDB::RegisteringObjects(LDOMHandle _h)
{
  LBOM* om = oms[_h.handle];
  if (!om->RegisteringObjs()) {
    if (oms_registering == 0)
      localBarrier.TurnOff();
    oms_registering++;
    om->SetRegisteringObjs(CmiTrue);
  }
}

void LBDB::DoneRegisteringObjects(LDOMHandle _h)
{
  LBOM* om = oms[_h.handle];
  if (om->RegisteringObjs()) {
    oms_registering--;
    if (oms_registering == 0)
      localBarrier.TurnOn();
    om->SetRegisteringObjs(CmiFalse);
  }
}


void LBDB::Send(const LDOMHandle &destOM, const LDObjid &destid, unsigned int bytes)
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

    LBCommData item(runObj,destOM.id,destid);
    item_ptr = commTable->HashInsertUnique(item);
  } else {
    LBCommData item(CkMyPe(),destOM.id,destid);
    item_ptr = commTable->HashInsertUnique(item);
  }  
  item_ptr->addMessage(bytes);
}

void LBDB::ClearLoads(void)
{
  int i;
  for(i=0; i < objCount; i++) {
    LBObj *obj = objs[i];
    if (obj->registered)
    {
      if (obj->data.cpuTime>.0) {
        obj->lastCpuTime = obj->data.cpuTime;
        obj->lastWallTime = obj->data.wallTime;
      }
      obj->data.wallTime = 
	obj->data.cpuTime = 0.;
    }
  }
  delete commTable;
  commTable = new LBCommTable;
  machineUtil.Clear();
  obj_walltime = obj_cputime = 0;
}

int LBDB::ObjDataCount()
{
  int nitems=0;
  int i;
  for(i=0; i < objCount; i++)
    if ((objs[i])->registered)
      nitems++;
  return nitems;
}

void LBDB::GetObjData(LDObjData *dp)
{
  for(int i = 0; i < objs.length(); i++) {
    LBObj* obj = objs[i];
    if ( obj->registered )
      *dp++ = obj->ObjData();
  }
}

void LBDB::Migrate(LDObjHandle h, int dest)
{
  if (h.handle > objCount)
    CmiPrintf("[%d] Handle %d out of range 0-%d\n",CkMyPe(),h.handle,objCount);
  else if (!(objs[h.handle])->registered)
    CmiPrintf("[%d] Handle %d no longer registered, range 0-%d\n",
	    CkMyPe(),h.handle,objCount);

  if ((h.handle < objCount) && ((objs[h.handle])->registered)) {
    LBOM *const om = oms[(objs[h.handle])->parentOM().handle];
    om->Migrate(h, dest);
  }
  return;
}

void LBDB::Migrated(LDObjHandle h)
{
  // Object migrated, inform load balancers

  for(int i=0; i < migrateCBList.length(); i++) {
    MigrateCB* cb = (MigrateCB*)migrateCBList[i];
    (cb->fn)(cb->data,h);
  }
  
}

void LBDB::NotifyMigrated(LDMigratedFn fn, void* data)
{
  // Save migration function
  MigrateCB* callbk = new MigrateCB;

  callbk->fn = fn;
  callbk->data = data;
  migrateCBList.insertAtEnd(callbk);
}

void LBDB::AddStartLBFn(LDStartLBFn fn, void* data)
{
  // Save startLB function
  StartLBCB* callbk = new StartLBCB;

  callbk->fn = fn;
  callbk->data = data;
  startLBFnList.push_back(callbk);
  startLBFn_count++;
}

void LBDB::RemoveStartLBFn(LDStartLBFn fn)
{
  for (int i=0; i<startLBFnList.length(); i++) {
    if (startLBFnList[i]->fn == fn) {
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
    if (startLBFn) startLBFn->fn(startLBFn->data);
  }
}

void LBDB::BackgroundLoad(double* walltime, double* cputime)
{
  double totalwall;
  double totalcpu;
  TotalTime(&totalwall,&totalcpu);

  double idle;
  IdleTime(&idle);
  
  *walltime = totalwall - idle - obj_walltime;
  *cputime = totalcpu - obj_cputime;
}

void LBDB::DumpDatabase()
{
#ifdef DEBUG  
  CmiPrintf("Database contains %d object managers\n",omCount);
  CmiPrintf("Database contains %d objects\n",objCount);
#endif
}

LDBarrierClient LocalBarrier::AddClient(LDResumeFn fn, void* data)
{
  client* new_client = new client;
  new_client->fn = fn;
  new_client->data = data;
  new_client->refcount = cur_refcount;

  LDBarrierClient ret_val;
  ret_val.serial = max_client;
  clients.insertAtEnd(new_client);
  max_client++;

  client_count++;

  return ret_val;
}

void LocalBarrier::RemoveClient(LDBarrierClient c)
{
  const int cnum = c.serial;
  if (cnum < max_client && clients[cnum] != 0) {
    delete (clients[cnum]);
    clients[cnum] = 0;
    client_count--;
  }
}

LDBarrierReceiver LocalBarrier::AddReceiver(LDBarrierFn fn, void* data)
{
  receiver* new_receiver = new receiver;
  new_receiver->fn = fn;
  new_receiver->data = data;

  LDBarrierReceiver ret_val;
  ret_val.serial = max_receiver;
  receivers.insertAtEnd(new_receiver);
  max_receiver++;

  return ret_val;
}

void LocalBarrier::RemoveReceiver(LDBarrierReceiver c)
{
  const int cnum = c.serial;
  if (cnum < max_receiver && receivers[cnum] != 0) {
    delete (receivers[cnum]);
    receivers[cnum] = 0;
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

    for(int i=0; i < max_client; i++)
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
    for (int i=max_receiver-1; i>=0; i--)
    if (receivers[i] != 0) {
      ((receiver*)receivers[i])->fn(((receiver*)receivers[i])->data);
      called_receiver = CmiTrue;
    }

  if (!called_receiver)
    ResumeClients();
  
}

void LocalBarrier::ResumeClients(void)
{
  for(int i=0; i < max_client; i++)
    if (clients[i] != 0) 
      ((client*)clients[i])->fn(((client*)clients[i])->data);
}

#endif

/*@}*/
