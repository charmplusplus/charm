#ifndef LBDATABASE_H
#define LBDATABASE_H

#include "lbdb.h"
#include "LBDatabase.decl.h"

class LBDBInit : public Chare {
  public:
    LBDBInit(CkArgMsg*);
};

class LBDatabase : public Group {
public:
  LBDatabase(void) {
    myLDHandle = LDCreate();  
  };

  /*
   * Calls from object managers to load database
   */
  LDOMHandle RegisterOM(LDOMid userID, void *userptr, LDCallbacks cb) {
    return LDRegisterOM(myLDHandle,userID, userptr, cb);
  };

  void RegisteringObjects(LDOMHandle _om) {
    LDRegisteringObjects(_om);
  };

  void DoneRegisteringObjects(LDOMHandle _om) {
    LDDoneRegisteringObjects(_om);
  };

  LDObjHandle RegisterObj(LDOMHandle h, LDObjid id,
			  void *userptr,int migratable) {
    return LDRegisterObj(h,id,userptr,migratable);
  };

  void UnregisterObj(LDObjHandle h) { LDUnregisterObj(h); };

  void ObjTime(LDObjHandle h, double walltime, double cputime) {
    LDObjTime(h,walltime,cputime);
  };

  int RunningObject(LDObjHandle* _o) { 
    return LDRunningObject(myLDHandle,_o);
  };
  void ObjectStart(LDObjHandle _h) { LDObjectStart(_h); };
  void ObjectStop(LDObjHandle _h) { LDObjectStop(_h); };
  void Send(LDOMHandle _om, LDObjid _id, unsigned int _b) {
    LDSend(_om, _id, _b);
  };

  void EstObjLoad(LDObjHandle h, double load) { LDEstObjLoad(h,load); };
  void NonMigratable(LDObjHandle h) { LDNonMigratable(h); };
  void Migratable(LDObjHandle h) { LDMigratable(h); };
  void DumpDatabase(void) { LDDumpDatabase(myLDHandle); };

  /*
   * Calls from load balancer to load database
   */  
  void NotifyMigrated(LDMigratedFn fn, void *data) 
  {
    LDNotifyMigrated(myLDHandle,fn,data);
  };
 
  void CollectStatsOn(void) { LDCollectStatsOn(myLDHandle); };
  void CollectStatsOff(void) { LDCollectStatsOff(myLDHandle); };
  void QueryEstLoad(void) { LDQueryEstLoad(myLDHandle); };

  int GetObjDataSz(void) { return LDGetObjDataSz(myLDHandle); };
  void GetObjData(LDObjData *data) { LDGetObjData(myLDHandle,data); };
  int GetCommDataSz(void) { return LDGetCommDataSz(myLDHandle); };
  void GetCommData(LDCommData *data) { LDGetCommData(myLDHandle,data); };

  void BackgroundLoad(double *walltime, double *cputime) {
    LDBackgroundLoad(myLDHandle,walltime,cputime);
  }

  void IdleTime(double *walltime) {
    LDIdleTime(myLDHandle,walltime);
  };

  void TotalTime(double *walltime, double *cputime) {
    LDTotalTime(myLDHandle,walltime,cputime);
  }

  void ClearLoads(void) { LDClearLoads(myLDHandle); };
  void Migrate(LDObjHandle h, int dest) { LDMigrate(h,dest); };

  void Migrated(LDObjHandle h) { LDMigrated(h); };

  LDBarrierClient AddLocalBarrierClient(LDResumeFn fn, void* data) {
    return LDAddLocalBarrierClient(myLDHandle,fn,data);
  };

  void RemoveLocalBarrierClient(LDBarrierClient h) {
    LDRemoveLocalBarrierClient(myLDHandle, h);
  };

  LDBarrierReceiver AddLocalBarrierReceiver(LDBarrierFn fn, void *data) {
    return LDAddLocalBarrierReceiver(myLDHandle,fn,data);
  };

  void RemoveLocalBarrierReceiver(LDBarrierReceiver h) {
    LDRemoveLocalBarrierReceiver(myLDHandle,h);
  };

  void AtLocalBarrier(LDBarrierClient h) { LDAtLocalBarrier(myLDHandle,h); };
  void ResumeClients() { LDResumeClients(myLDHandle); }

  int ProcessorSpeed() { return LDProcessorSpeed(); };
private:
  LDHandle myLDHandle;

};

#endif /* LDATABASE_H */
