#include <converse.h>

#if CMK_LBDB_ON

#include <iostream.h>
#include "LBDBManager.h"

struct MigrateCB;

#if CMK_STL_USE_DOT_H
template class vector<LBOM*>;
template class vector<LBObj*>;
template class vector<LBDB::MigrateCB*>;
template class vector<LocalBarrier::client*>;
template class vector<LocalBarrier::receiver*>;
#else
template class std::vector<LBOM*>;
template class std::vector<LBObj*>;
template class std::vector<LBDB::MigrateCB*>;
template class std::vector<LocalBarrier::client*>;
template class std::vector<LocalBarrier::receiver*>;
#endif

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
  objs[_h.handle]->registered=CmiFalse;
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


void LBDB::Send(LDOMHandle destOM, LDObjid destid, unsigned int bytes)
{
  LBCommData* item_ptr;

  if (obj_running) {
    LBCommData item(runningObj,destOM.id,destid);
    item_ptr = commTable->HashInsertUnique(item);

//     CmiPrintf("[%d] Sending %d from object manager %d, object {%d,%d,%d,%d}\n"
//  	      "     to object manager %d, object {%d,%d,%d,%d}\n",
//   	      CmiMyPe(),bytes,
//   	      runningObj.omhandle.id.id,
//   	      runningObj.id.id[0],runningObj.id.id[1],
//   	      runningObj.id.id[2],runningObj.id.id[3],
//   	      destOM.id.id,
//   	      destid.id[0],destid.id[1],
//   	      destid.id[2],destid.id[3]
//   	      );
  } else {
    LBCommData item(CmiMyPe(),destOM.id,destid);
    item_ptr = commTable->HashInsertUnique(item);

//     CmiPrintf("[%d] Sending %d from processor %d\n"
//   	      "     to object manager %d, object {%d,%d,%d,%d}\n",
//  	      CmiMyPe(),bytes,
// 	      CmiMyPe(),
//    	      destOM.id.id,
//    	      destid.id[0],destid.id[1],
//    	      destid.id[2],destid.id[3]
//    	      );
  }  
  item_ptr->addMessage(bytes);
}

void LBDB::ClearLoads(void)
{
  int i;
  for(i=0; i < objCount; i++)
    if (objs[i]->registered)
    {
      objs[i]->data.wallTime = 
	objs[i]->data.cpuTime = 0.;
    }
  delete commTable;
  commTable = new LBCommTable;
}

int LBDB::ObjDataCount()
{
  int nitems=0;
  int i;
  for(i=0; i < objCount; i++)
    if (objs[i]->registered)
      nitems++;
  return nitems;
}

void LBDB::GetObjData(LDObjData *dp)
{
  for(ObjList::iterator ol = objs.begin(); ol != objs.end(); ol++)
    if ((*ol)->registered)
      *dp++ = (*ol)->ObjData();
}

void LBDB::Migrate(LDObjHandle h, int dest)
{
  if (h.handle > objCount)
    CmiPrintf("[%d] Handle %d out of range 0-%d\n",CmiMyPe(),h.handle,objCount);
  else if (!objs[h.handle]->registered)
    CmiPrintf("[%d] Handle %d no longer registered, range 0-%d\n",
	    CmiMyPe(),h.handle,objCount);

  if ((h.handle < objCount) && (objs[h.handle]->registered)) {
    LBOM *const om = oms[objs[h.handle]->parentOM.handle];
    om->Migrate(h, dest);
  }
  return;
}

void LBDB::Migrated(LDObjHandle h)
{
  // Object migrated, inform load balancers

  for(int i=0; i < migrateCBList.size(); i++)
    (migrateCBList[i]->fn)(migrateCBList[i]->data,h);
  
}

void LBDB::NotifyMigrated(LDMigratedFn fn, void* data)
{
  // Save migration function
  MigrateCB* callbk = new MigrateCB;

  callbk->fn = fn;
  callbk->data = data;
  migrateCBList.push_back(callbk);
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
  clients.push_back(new_client);
  max_client++;

  client_count++;

  return ret_val;
}

void LocalBarrier::RemoveClient(LDBarrierClient c)
{
  const int cnum = c.serial;
  if (cnum < max_client && clients[cnum] != 0) {
    delete clients[cnum];
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
    delete receivers[cnum];
    receivers[cnum] = 0;
  }
}

void LocalBarrier::AtBarrier(LDBarrierClient h)
{
  clients[h.serial]->refcount++;
  at_count++;
  CheckBarrier();
}

void LocalBarrier::CheckBarrier()
{
  if (!on) return;

  if (at_count >= client_count) {
    CmiBool at_barrier = CmiFalse;

    for(int i=0; i < max_client; i++)
      if (clients[i] != 0 && clients[i]->refcount >= cur_refcount)
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

  for(int i=0; i < max_receiver; i++)
    if (receivers[i] != 0) {
      receivers[i]->fn(receivers[i]->data);
      called_receiver = CmiTrue;
    }

  if (!called_receiver)
    ResumeClients();
  
}

void LocalBarrier::ResumeClients(void)
{
  for(int i=0; i < max_client; i++)
    if (clients[i] != 0) 
      clients[i]->fn(clients[i]->data);
}

#endif
