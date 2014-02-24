/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <converse.h>

#include <math.h>

#include "lbdb.h"
#include "LBObj.h"
#include "LBOM.h"
#include "LBDatabase.h"
#include "LBDBManager.h"
  
#if CMK_LBDB_ON

extern "C" LDHandle LDCreate(void)
{
  LDHandle h;
  h.handle = (void*)(new LBDB);
  return h;
}

extern "C" LDOMHandle LDRegisterOM(LDHandle _db, LDOMid _userID,
				   void *_userptr, LDCallbacks _callbacks)
{
  LBDB *const db = (LBDB*)(_db.handle);
  return db->AddOM(_userID, _userptr, _callbacks);
}

extern "C" void LDUnregisterOM(LDHandle _db, LDOMHandle om)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->RemoveOM(om);
}

extern "C" void LDOMMetaLBResumeWaitingChares(LDHandle _db, int lb_ideal_period) {
  LBDB *const db = (LBDB*)(_db.handle);
  db->MetaLBResumeWaitingChares(lb_ideal_period);
}

extern "C" void LDOMMetaLBCallLBOnChares(LDHandle _db) {
  LBDB *const db = (LBDB*)(_db.handle);
  db->MetaLBCallLBOnChares();
}

extern "C" void * LDOMUserData(LDOMHandle &_h)
{
  LBDB *const db = (LBDB*)(_h.ldb.handle);
  return db->LbOM(_h)->getUserData();
}

extern "C" void LDRegisteringObjects(LDOMHandle _h)
{
  LBDB *const db = (LBDB*)(_h.ldb.handle);
  db->RegisteringObjects(_h);
}

extern "C" void LDDoneRegisteringObjects(LDOMHandle _h)
{
  LBDB *const db = (LBDB*)(_h.ldb.handle);
  db->DoneRegisteringObjects(_h);
}

extern "C" LDObjHandle LDRegisterObj(LDOMHandle _h, LDObjid _id, 
				       void *_userData, int _migratable)
{
  LBDB *const db = (LBDB*)(_h.ldb.handle);
  return db->AddObj(_h, _id, _userData, (bool)(_migratable));
}

extern "C" void LDUnregisterObj(LDObjHandle _h)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);
  db->UnregisterObj(_h);
  return;
}

const LDObjHandle &LDGetObjHandle(LDHandle h, int oh)
{
  LBDB *const db = (LBDB*)(h.handle);
  LBObj *const obj = db->LbObjIdx(oh);
  return obj->GetLDObjHandle();
}

extern "C" void LDObjTime(LDObjHandle &_h,
			    LBRealType walltime, LBRealType cputime)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(_h);
  obj->IncrementTime(walltime,cputime);
  db->MeasuredObjTime(walltime,cputime);
}
  
extern "C" void LDGetObjLoad(LDObjHandle &_h, LBRealType *wallT, LBRealType *cpuT)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(_h);
  obj->getTime(wallT, cpuT);
}

extern "C" void LDQueryKnownObjLoad(LDObjHandle &_h, LBRealType *wallT, LBRealType *cpuT)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(_h);
  obj->lastKnownLoad(wallT, cpuT);
}

extern "C" void * LDObjUserData(LDObjHandle &_h)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(_h);
  return obj->getLocalUserData();
}

#if CMK_LB_USER_DATA
extern "C" void * LDDBObjUserData(LDObjHandle &_h, int idx)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(_h);
  return obj->getDBUserData(idx);
}
#endif

extern "C" void LDDumpDatabase(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->DumpDatabase();
}

extern "C" void LDNotifyMigrated(LDHandle _db, LDMigratedFn fn, void* data)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->NotifyMigrated(fn,data);
}

extern "C" void LDAddStartLBFn(LDHandle _db, LDStartLBFn fn, void* data)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->AddStartLBFn(fn,data);
}

extern "C" void LDRemoveStartLBFn(LDHandle _db, LDStartLBFn fn)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->RemoveStartLBFn(fn);
}

extern "C" void LDStartLB(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->StartLB();
}

extern "C" void LDTurnManualLBOn(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->TurnManualLBOn();
}

extern "C" void LDTurnManualLBOff(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->TurnManualLBOff();
}

extern "C" int LDAddMigrationDoneFn(LDHandle _db, LDMigrationDoneFn fn,  void* data) 
{
  LBDB *const db = (LBDB*)(_db.handle);
  return db->AddMigrationDoneFn(fn,data);
}

extern "C" void  LDRemoveMigrationDoneFn(LDHandle _db, LDMigrationDoneFn fn)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->RemoveMigrationDoneFn(fn);
}

extern "C" void LDMigrationDone(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->MigrationDone();
}

extern "C" void LDTurnPredictorOn(LDHandle _db, void *model)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->TurnPredictorOn(model);
}

extern "C" void LDTurnPredictorOnWin(LDHandle _db, void *model, int wind)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->TurnPredictorOn(model, wind);
}

extern "C" void LDTurnPredictorOff(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->TurnPredictorOff();
}

/* the parameter model is really of class LBPredictorFunction in file LBDatabase.h */
extern "C" void LDChangePredictor(LDHandle _db, void *model)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->ChangePredictor(model);
}

extern "C" void LDCollectStatsOn(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);

  if (!db->StatsOn()) {
    if (db->ObjIsRunning()) {
       // stats on in the middle of an entry, start timer
      const LDObjHandle &oh = db->RunningObj();
      LBObj *obj = db->LbObj(oh);
      obj->StartTimer();
    }
    db->TurnStatsOn();
  }
}

extern "C" void LDCollectStatsOff(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->TurnStatsOff();
}

extern "C" int CLDCollectingStats(LDHandle _db)
{
//  LBDB *const db = (LBDB*)(_db.handle);
//
//  return db->StatsOn();
  return LDCollectingStats(_db);
}

extern "C" int CLDRunningObject(LDHandle _h, LDObjHandle* _o)
{
  return LDRunningObject(_h, _o);
}

extern "C" void LDObjectStart(const LDObjHandle &_h)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);

  if (db->ObjIsRunning()) LDObjectStop(db->RunningObj());

  db->SetRunningObj(_h);

  if (db->StatsOn()) {
    LBObj *const obj = db->LbObj(_h);
    obj->StartTimer();
  }
}

extern "C" void LDObjectStop(const LDObjHandle &_h)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(_h);

  if (db->StatsOn()) {
    LBRealType walltime, cputime;
    obj->StopTimer(&walltime,&cputime);
    obj->IncrementTime(walltime,cputime);
    db->MeasuredObjTime(walltime,cputime);
  }
  db->NoRunningObj();
}

extern "C" void LDSend(const LDOMHandle &destOM, const LDObjid &destid, unsigned int bytes, int destObjProc, int force)
{
  LBDB *const db = (LBDB*)(destOM.ldb.handle);
  if (force || db->StatsOn() && _lb_args.traceComm())
    db->Send(destOM,destid,bytes, destObjProc);
}

extern "C" void LDMulticastSend(const LDOMHandle &destOM, LDObjid *destids, int ndests, unsigned int bytes, int nMsgs)
{
  LBDB *const db = (LBDB*)(destOM.ldb.handle);
  if (db->StatsOn() && _lb_args.traceComm())
    db->MulticastSend(destOM,destids,ndests,bytes,nMsgs);
}

extern "C" void LDBackgroundLoad(LDHandle _db,
				 LBRealType* walltime, LBRealType* cputime)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->BackgroundLoad(walltime,cputime);

  return;
}

extern "C" void LDIdleTime(LDHandle _db,LBRealType* walltime)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->IdleTime(walltime);

  return;
}

extern "C" void LDTotalTime(LDHandle _db,LBRealType* walltime, LBRealType* cputime)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->TotalTime(walltime,cputime);

  return;
}

extern "C" void LDGetTime(LDHandle _db, LBRealType *total_walltime,
                   LBRealType *total_cputime,
                   LBRealType *idletime, LBRealType *bg_walltime, LBRealType *bg_cputime)
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->GetTime(total_walltime, total_cputime, idletime, bg_walltime, bg_cputime);
}

extern "C" void LDNonMigratable(const LDObjHandle &h)
{
  LBDB *const db = (LBDB*)(h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(h);

  obj->SetMigratable(false);
}

extern "C" void LDMigratable(const LDObjHandle &h)
{
  LBDB *const db = (LBDB*)(h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(h);

  obj->SetMigratable(true);
}

extern "C" void LDSetPupSize(const LDObjHandle &h, size_t obj_pup_size)
{
  LBDB *const db = (LBDB*)(h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(h);

  obj->setPupSize(obj_pup_size);
}

extern "C" void LDAsyncMigrate(const LDObjHandle &h, bool async)
{
  LBDB *const db = (LBDB*)(h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(h);

  obj->UseAsyncMigrate(async);
}

extern "C" void LDClearLoads(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);

  db->ClearLoads();
}

extern "C" int LDGetObjDataSz(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);

  return db->ObjDataCount();
}

extern "C" void LDGetObjData(LDHandle _db, LDObjData *data)
{
  LBDB *const db = (LBDB*)(_db.handle);

  db->GetObjData(data);
}

extern "C" int LDGetCommDataSz(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);

  return db->CommDataCount();
}

extern "C" void LDGetCommData(LDHandle _db, LDCommData *data)
{
  LBDB *const db = (LBDB*)(_db.handle);

  db->GetCommData(data);
  return;
}

extern "C" int LDMigrate(LDObjHandle _h, int dest)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);

  return db->Migrate(_h,dest);
}

extern "C" void LDMigrated(LDObjHandle _h, int waitBarrier)
{
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);

  db->Migrated(_h, waitBarrier);
}

LDBarrierClient LDAddLocalBarrierClient(LDHandle _db, LDResumeFn fn, void* data)
{
  LBDB *const db = (LBDB*)(_db.handle);

  return db->AddLocalBarrierClient(fn,data);
}

void LDRemoveLocalBarrierClient(LDHandle _db, LDBarrierClient h)
{
  LBDB *const db = (LBDB*)(_db.handle);

  db->RemoveLocalBarrierClient(h);
}

LDBarrierReceiver LDAddLocalBarrierReceiver(LDHandle _db,LDBarrierFn fn, void* data)
{
  LBDB *const db = (LBDB*)(_db.handle);

  return db->AddLocalBarrierReceiver(fn,data);
}

void LDRemoveLocalBarrierReceiver(LDHandle _db,LDBarrierReceiver h)
{
  LBDB *const db = (LBDB*)(_db.handle);

  db->RemoveLocalBarrierReceiver(h);
}

extern "C" void LDAtLocalBarrier(LDHandle _db, LDBarrierClient h)
{
  LBDB *const db = (LBDB*)(_db.handle);
	
  db->AtLocalBarrier(h);
}

extern "C" void LDDecreaseLocalBarrier(LDHandle _db, LDBarrierClient h, int c)
{
  LBDB *const db = (LBDB*)(_db.handle);
	
  db->DecreaseLocalBarrier(h, c);
}

extern "C" void LDLocalBarrierOn(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);

  db->LocalBarrierOn();
}

extern "C" void LDLocalBarrierOff(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);

  db->LocalBarrierOff();
}


extern "C" void LDResumeClients(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);

  db->ResumeClients();
}

static void work(int iter_block, int* result) {
  int i;
  *result = 1;
  for(i=0; i < iter_block; i++) {
    double b=0.1 + 0.1 * *result;
    *result=(int)(sqrt(1+cos(b * 1.57)));
  }
}

extern "C" int LDProcessorSpeed()
{
  // for SMP version, if one processor have done this testing,
  // we can skip the other processors by remember the number here
  static int thisProcessorSpeed = -1;

  if (_lb_args.samePeSpeed() || CkNumPes() == 1)  // I think it is safe to assume that we can
    return 1;            // skip this if we are only using 1 PE
  
  if (thisProcessorSpeed != -1) return thisProcessorSpeed;

  //if (CkMyPe()==0) CkPrintf("Measuring processor speeds...");

  static int result=0;  // I don't care what this is, its just for
			// timing, so this is thread safe.
  int wps = 0;
  const double elapse = 0.4;
  // First, count how many iterations for .2 second.
  // Since we are doing lots of function calls, this will be rough
  const double end_time = CmiCpuTimer()+elapse;
  wps = 0;
  while(CmiCpuTimer() < end_time) {
    work(1000,&result);
    wps+=1000;
  }

  // Now we have a rough idea of how many iterations there are per
  // second, so just perform a few cycles of correction by
  // running for what we think is 1 second.  Then correct
  // the number of iterations per second to make it closer
  // to the correct value
  
  for(int i=0; i < 2; i++) {
    const double start_time = CmiCpuTimer();
    work(wps,&result);
    const double end_time = CmiCpuTimer();
    const double correction = elapse / (end_time-start_time);
    wps = (int)((double)wps * correction + 0.5);
  }
  
  // If necessary, do a check now
  //    const double start_time3 = CmiWallTimer();
  //    work(msec * 1e-3 * wps);
  //    const double end_time3 = CmiWallTimer();
  //    CkPrintf("[%d] Work block size is %d %d %f\n",
  //	     thisIndex,wps,msec,1.e3*(end_time3-start_time3));
  thisProcessorSpeed = wps;

  //if (CkMyPe()==0) CkPrintf(" Done.\n");

  return wps;
}

extern "C" void LDSetLBPeriod(LDHandle _db, double s)   // s is in seconds
{
  LBDB *const db = (LBDB*)(_db.handle);
  db->SetPeriod(s);
}

extern "C" double LDGetLBPeriod(LDHandle _db)   // s is in seconds
{
  LBDB *const db = (LBDB*)(_db.handle);
  return db->GetPeriod();
}

/*
// to be implemented
extern "C" void LDEstObjLoad(LDObjHandle h, double load)
{
}
*/

// to be implemented
extern "C" void LDQueryEstLoad(LDHandle bdb)
{
}

extern "C" int LDMemusage(LDHandle _db) 
{
  LBDB *const db = (LBDB*)(_db.handle);
  return db->useMem();
}

#else
extern "C" int LDProcessorSpeed() { return 1; }
#endif // CMK_LBDB_ON

bool LDOMidEqual(const LDOMid &i1, const LDOMid &i2)
{
 return i1.id == i2.id?true:false;
}

bool LDObjIDEqual(const LDObjid &i1, const LDObjid &i2)
{
  return (i1.id[0] == i2.id[0] 
	 && i1.id[1] == i2.id[1] && i1.id[2] == i2.id[2] 
	 && i1.id[3] == i2.id[3]);
}

/*@}*/
