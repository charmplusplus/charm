#include <converse.h>

#if CMK_LBDB_ON

#include "lbdb.h"
#include "LBObj.h"
#include "LBOM.h"
#include "LBDBManager.h"
  
extern "C" LDHandle LDCreate(void)
{
  LDHandle h;
  h.handle = static_cast<void*>(new LBDB);
  return h;
}

extern "C" LDOMHandle LDRegisterOM(LDHandle _db, LDOMid _userID,
				   void *_userptr, LDCallbacks _callbacks)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);
  return db->AddOM(_userID, _userptr, _callbacks);
}

extern "C" void LDRegisteringObjects(LDOMHandle _h)
{
  LBDB *const db = static_cast<LBDB*>(_h.ldb.handle);
  db->RegisteringObjects(_h);
}

extern "C" void LDDoneRegisteringObjects(LDOMHandle _h)
{
  LBDB *const db = static_cast<LBDB*>(_h.ldb.handle);
  db->DoneRegisteringObjects(_h);
}

extern "C" LDObjHandle LDRegisterObj(LDOMHandle _h, LDObjid _id, 
				       void *_userData, int _migratable)
{
  LBDB *const db = static_cast<LBDB*>(_h.ldb.handle);
  return db->AddObj(_h, _id, _userData, static_cast<CmiBool>(_migratable));
}

extern "C" void LDUnregisterObj(LDObjHandle _h)
{
  LBDB *const db = static_cast<LBDB*>(_h.omhandle.ldb.handle);
  db->UnregisterObj(_h);
  return;
}

extern "C" void LDObjTime(LDObjHandle _h,
			    double walltime, double cputime)
{
  LBDB *const db = static_cast<LBDB*>(_h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(_h);
  obj->IncrementTime(walltime,cputime);
}
  
extern "C" void LDDumpDatabase(LDHandle _db)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);
  db->DumpDatabase();
}

void LDNotifyMigrated(LDHandle _db, LDMigratedFn fn, void* data)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);
  db->NotifyMigrated(fn,data);
}

extern "C" void LDCollectStatsOn(LDHandle _db)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);
  db->TurnStatsOn();
}

extern "C" void LDCollectStatsOff(LDHandle _db)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);
  db->TurnStatsOff();
}

extern "C" void LDObjectStart(LDObjHandle _h)
{
  LBDB *const db = static_cast<LBDB*>(_h.omhandle.ldb.handle);
  if (db->StatsOn()) {
    db->RunningObj(_h);
    LBObj *const obj = db->LbObj(_h);
    obj->StartTimer();
  }
}

extern "C" void LDObjectStop(LDObjHandle _h)
{
  LBDB *const db = static_cast<LBDB*>(_h.omhandle.ldb.handle);
  if (db->StatsOn()) {
    LBObj *const obj = db->LbObj(_h);
    double walltime, cputime;
    obj->StopTimer(&walltime,&cputime);
    obj->IncrementTime(walltime,cputime);
    db->NoRunningObj();
  }
}

extern "C" void LDSend(LDOMHandle destOM, LDObjid destid, unsigned int bytes)
{
  LBDB *const db = static_cast<LBDB*>(destOM.ldb.handle);
  if (db->StatsOn())
    db->Send(destOM,destid,bytes);
}

extern "C" void LDClearLoads(LDHandle _db)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  db->ClearLoads();
}

extern "C" int LDGetObjDataSz(LDHandle _db)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  return db->ObjDataCount();
}

extern "C" void LDGetObjData(LDHandle _db, LDObjData *data)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  db->GetObjData(data);
}

extern "C" int LDGetCommDataSz(LDHandle _db)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  return db->CommDataCount();
}

extern "C" void LDGetCommData(LDHandle _db, LDCommData *data)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  db->GetCommData(data);
  return;
}

extern "C" void LDMigrate(LDObjHandle _h, int dest)
{
  LBDB *const db = static_cast<LBDB*>(_h.omhandle.ldb.handle);

  db->Migrate(_h,dest);
}

extern "C" void LDMigrated(LDObjHandle _h)
{
  LBDB *const db = static_cast<LBDB*>(_h.omhandle.ldb.handle);

  db->Migrated(_h);
}

extern "C" LDBarrierClient 
LDAddLocalBarrierClient(LDHandle _db, LDResumeFn fn, void* data)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  return db->AddLocalBarrierClient(fn,data);
}

extern "C" void LDRemoveLocalBarrierClient(LDHandle _db, LDBarrierClient h)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  db->RemoveLocalBarrierClient(h);
}

extern "C" LDBarrierReceiver 
LDAddLocalBarrierReceiver(LDHandle _db,LDBarrierFn fn, void* data)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  return db->AddLocalBarrierReceiver(fn,data);
}

extern "C" void 
LDRemoveLocalBarrierReceiver(LDHandle _db,LDBarrierReceiver h)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  db->RemoveLocalBarrierReceiver(h);
}

extern "C" void LDAtLocalBarrier(LDHandle _db, LDBarrierClient h)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  db->AtLocalBarrier(h);
}

extern "C" void LDResumeClients(LDHandle _db)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);

  db->ResumeClients();
}

#endif // CMK_LBDB_ON
