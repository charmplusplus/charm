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

extern "C" int LDRunningObject(LDHandle _h, LDObjHandle* _o)
{
  LBDB *const db = static_cast<LBDB*>(_h.handle);

  if (db->ObjIsRunning()) {
    *_o = db->RunningObj();
    return 1;
  } else return 0;
}

extern "C" void LDObjectStart(LDObjHandle _h)
{
  LBDB *const db = static_cast<LBDB*>(_h.omhandle.ldb.handle);

  if (db->ObjIsRunning()) LDObjectStop(db->RunningObj());

  db->SetRunningObj(_h);

  if (db->StatsOn()) {
    LBObj *const obj = db->LbObj(_h);
    obj->StartTimer();
  }
}

extern "C" void LDObjectStop(LDObjHandle _h)
{
  LBDB *const db = static_cast<LBDB*>(_h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(_h);

  if (db->StatsOn()) {
    double walltime, cputime;
    obj->StopTimer(&walltime,&cputime);
    obj->IncrementTime(walltime,cputime);
  }
  db->NoRunningObj();
}

extern "C" void LDSend(LDOMHandle destOM, LDObjid destid, unsigned int bytes)
{
  LBDB *const db = static_cast<LBDB*>(destOM.ldb.handle);
  if (db->StatsOn())
    db->Send(destOM,destid,bytes);
}

extern "C" void LDBackgroundLoad(LDHandle _db,
				 double* walltime, double* cputime)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);
  db->BackgroundLoad(walltime,cputime);

  return;
}

extern "C" void LDIdleTime(LDHandle _db,double* walltime)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);
  db->IdleTime(walltime);

  return;
}

extern "C" void LDTotalTime(LDHandle _db,double* walltime, double* cputime)
{
  LBDB *const db = static_cast<LBDB*>(_db.handle);
  db->TotalTime(walltime,cputime);

  return;
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

static int work(int iter_block) {
  int result=0;
  int i;
  for(i=0; i < iter_block; i++) {
    double b=0.1+0.1*result;
    result=(int)(sqrt(log(atan(b))));
  }
  return result;
}

extern "C" int LDProcessorSpeed()
{
  double wps = 0;
  // First, count how many iterations for 1 second.
  // Since we are doing lots of function calls, this will be rough
  const double end_time = CmiCpuTimer()+1;
  wps = 0;
  while(CmiCpuTimer() < end_time) {
    work(100);
    wps+=100;
  }

  // Now we have a rough idea of how many iterations there are per
  // second, so just perform a few cycles of correction by
  // running for what we think is 1 second.  Then correct
  // the number of iterations per second to make it closer
  // to the correct value
  
  for(int i=0; i < 2; i++) {
    const double start_time = CmiCpuTimer();
    work(wps);
    const double end_time = CmiCpuTimer();
    const double correction = 1. / (end_time-start_time);
    wps *= correction;
  }
  
  // If necessary, do a check now
  //    const double start_time3 = CmiWallTimer();
  //    work(msec * 1e-3 * wps);
  //    const double end_time3 = CmiWallTimer();
  //    CkPrintf("[%d] Work block size is %d %d %f\n",
  //	     thisIndex,wps,msec,1.e3*(end_time3-start_time3));
  return wps;
}

#endif // CMK_LBDB_ON
