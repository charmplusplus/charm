/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <converse.h>

#if CMK_LBDB_ON

#include <iostream.h>
#include "LBDBManager.h"

struct MigrateCB;

/*************************************************************
 * LBDB Code
 *************************************************************/

LDOMHandle LBDB::AddOM(LDOMid _userID, void* _userData, 
		       LDCallbacks _callbacks)
{
  LDOMHandle newhandle;

  newhandle.ldb.handle = static_cast<void *>(this);
  newhandle.user_ptr = _userData;
  newhandle.id = _userID;

  LBOM* om = new LBOM(this,_userID,_userData,_callbacks);
  if (om != NULL) {
    newhandle.handle = oms.size();
    oms.push_back(om);
  } else newhandle.handle = -1;
  om->DepositHandle(newhandle);
  omCount++;
  return newhandle;
}

LDObjHandle LBDB::AddObj(LDOMHandle _h, LDObjid _id,
			 void *_userData, CmiBool _migratable)
{
  LDObjHandle newhandle;

  newhandle.omhandle = _h;
  newhandle.user_ptr = _userData;
  newhandle.id = _id;
  
  LBObj *obj = new LBObj(this,_h,_id,_userData,_migratable);
  if (obj != NULL) {
    newhandle.handle = objs.size();
    objs.push_back(obj);
  } else {
    newhandle.handle = -1;
  }
  obj->DepositHandle(newhandle);
  objCount++;
  return newhandle;
}

void LBDB::UnregisterObj(LDObjHandle _h)
{
  ((LBObj*)objs[_h.handle])->registered=CmiFalse;
}

void LBDB::RegisteringObjects(LDOMHandle _h)
{
  LBOM* om = (LBOM*)oms[_h.handle];
  if (!om->RegisteringObjs()) {
    if (oms_registering == 0)
      localBarrier.TurnOff();
    oms_registering++;
    om->SetRegisteringObjs(CmiTrue);
  }
}

void LBDB::DoneRegisteringObjects(LDOMHandle _h)
{
  LBOM* om = (LBOM*)oms[_h.handle];
  if (om->RegisteringObjs()) {
    oms_registering--;
    if (oms_registering == 0)
      localBarrier.TurnOn();
    om->SetRegisteringObjs(CmiFalse);
  }
}


void LBDB::Send(LDOMHandle destOM, LDObjid destid, unsigned int bytes)
{
  LBCommData* item_ptr;

  if (obj_running) {

    // Don't record self-messages from an object to an object
    if ( LDOMidEqual(runningObj.omhandle.id,destOM.id)
	 && LDObjIDEqual(runningObj.id,destid) )
      return;

    // In the future, we'll have to eliminate processor to same 
    // processor messages as well

    LBCommData item(runningObj,destOM.id,destid);
    item_ptr = commTable->HashInsertUnique(item);
  } else {
    LBCommData item(CmiMyPe(),destOM.id,destid);
    item_ptr = commTable->HashInsertUnique(item);
  }  
  item_ptr->addMessage(bytes);
}

void LBDB::ClearLoads(void)
{
  int i;
  for(i=0; i < objCount; i++)
    if (((LBObj*)objs[i])->registered)
    {
      ((LBObj*)objs[i])->data.wallTime = 
	((LBObj*)objs[i])->data.cpuTime = 0.;
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
    if (((LBObj*)objs[i])->registered)
      nitems++;
  return nitems;
}

void LBDB::GetObjData(LDObjData *dp)
{
  for(int i = 0; i < objs.size(); i++) {
    LBObj* obj = (LBObj*) objs[i];
    if ( obj->registered )
      *dp++ = obj->ObjData();
  }
}

void LBDB::Migrate(LDObjHandle h, int dest)
{
  if (h.handle > objCount)
    CmiPrintf("[%d] Handle %d out of range 0-%d\n",CmiMyPe(),h.handle,objCount);
  else if (!((LBObj*)objs[h.handle])->registered)
    CmiPrintf("[%d] Handle %d no longer registered, range 0-%d\n",
	    CmiMyPe(),h.handle,objCount);

  if ((h.handle < objCount) && (((LBObj*)objs[h.handle])->registered)) {
    LBOM *const om = (LBOM*)oms[((LBObj*)objs[h.handle])->parentOM.handle];
    om->Migrate(h, dest);
  }
  return;
}

void LBDB::Migrated(LDObjHandle h)
{
  // Object migrated, inform load balancers

  for(int i=0; i < migrateCBList.size(); i++) {
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
  migrateCBList.push_back((void*)callbk);
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
  clients.push_back((void*)new_client);
  max_client++;

  client_count++;

  return ret_val;
}

void LocalBarrier::RemoveClient(LDBarrierClient c)
{
  const int cnum = c.serial;
  if (cnum < max_client && clients[cnum] != 0) {
    delete ((client*)clients[cnum]);
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
  receivers.push_back(new_receiver);
  max_receiver++;

  return ret_val;
}

void LocalBarrier::RemoveReceiver(LDBarrierReceiver c)
{
  const int cnum = c.serial;
  if (cnum < max_receiver && receivers[cnum] != 0) {
    delete ((receiver*)receivers[cnum]);
    receivers[cnum] = 0;
  }
}

void LocalBarrier::AtBarrier(LDBarrierClient h)
{
  ((client*)clients[h.serial])->refcount++;
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
