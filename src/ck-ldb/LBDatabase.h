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

#ifndef LBDATABASE_H
#define LBDATABASE_H

#include "lbdb.h"
#include "LBDatabase.decl.h"

extern CkGroupID lbdb;

class LBDB;

CkpvExtern(int, numLoadBalancers);
CkpvExtern(int, hasNullLB);
CkpvExtern(int, lbdatabaseInited);
CkpvExtern(int, dumpStep);
CkpvExtern(char*, dumpFile);
CkpvExtern(int, doSimulation);

typedef void (*LBDefaultCreateFn)(void);
void LBSetDefaultCreate(LBDefaultCreateFn f);

void LBRegisterBalancer(const char *, LBDefaultCreateFn, const char *);

void _LBDBInit();

class LBDBInit : public Chare {
  public:
    LBDBInit(CkArgMsg*);
    LBDBInit(CkMigrateMessage *m) {}
};


class LBDatabase : public Group {
public:
  static int manualOn;
public:
  LBDatabase(void) {
    myLDHandle = LDCreate();  
    CkpvAccess(lbdatabaseInited) = 1;
#if CMK_LBDB_ON
    if (manualOn) TurnManualLBOn();
#endif
  };
  LBDatabase(CkMigrateMessage *m) { myLDHandle = LDCreate(); }
  inline static LBDatabase * Object() { return CkpvAccess(lbdatabaseInited)?(LBDatabase *)CkLocalBranch(lbdb):NULL; }
#if CMK_LBDB_ON
  inline LBDB *getLBDB() {return (LBDB*)(myLDHandle.handle);}
#endif

  /*
   * Calls from object managers to load database
   */
  inline LDOMHandle RegisterOM(LDOMid userID, void *userptr, LDCallbacks cb) {
    return LDRegisterOM(myLDHandle,userID, userptr, cb);
  };

  inline void RegisteringObjects(LDOMHandle _om) {
    LDRegisteringObjects(_om);
  };

  inline void DoneRegisteringObjects(LDOMHandle _om) {
    LDDoneRegisteringObjects(_om);
  };

  inline LDObjHandle RegisterObj(LDOMHandle h, LDObjid id,
			  void *userptr,int migratable) {
    return LDRegisterObj(h,id,userptr,migratable);
  };

  inline void UnregisterObj(LDObjHandle h) { LDUnregisterObj(h); };

  inline void ObjTime(LDObjHandle h, double walltime, double cputime) {
    LDObjTime(h,walltime,cputime);
  };

  inline int RunningObject(LDObjHandle* _o) { 
    return LDRunningObject(myLDHandle,_o);
  };
  inline void ObjectStart(const LDObjHandle &_h) { LDObjectStart(_h); };
  inline void ObjectStop(const LDObjHandle &_h) { LDObjectStop(_h); };
  inline void Send(const LDOMHandle &_om, const LDObjid _id, unsigned int _b) {
    LDSend(_om, _id, _b);
  };

  inline void EstObjLoad(LDObjHandle h, double load) { LDEstObjLoad(h,load); };
  inline void NonMigratable(LDObjHandle h) { LDNonMigratable(h); };
  inline void Migratable(LDObjHandle h) { LDMigratable(h); };
  inline void DumpDatabase(void) { LDDumpDatabase(myLDHandle); };

  /*
   * Calls from load balancer to load database
   */  
  inline void NotifyMigrated(LDMigratedFn fn, void *data) 
  {
    LDNotifyMigrated(myLDHandle,fn,data);
  };
 
  inline void AddStartLBFn(LDStartLBFn fn, void *data) 
  {
    LDAddStartLBFn(myLDHandle,fn,data);
  };

  inline void RemoveStartLBFn(LDStartLBFn fn) 
  {
    LDRemoveStartLBFn(myLDHandle,fn);
  };

  inline void StartLB() { LDStartLB(myLDHandle); }
  inline void TurnManualLBOn() { LDTurnManualLBOn(myLDHandle); }
  inline void TurnManualLBOff() { LDTurnManualLBOff(myLDHandle); }
 
  inline void CollectStatsOn(void) { LDCollectStatsOn(myLDHandle); };
  inline void CollectStatsOff(void) { LDCollectStatsOff(myLDHandle); };
  inline void QueryEstLoad(void) { LDQueryEstLoad(myLDHandle); };

  inline int GetObjDataSz(void) { return LDGetObjDataSz(myLDHandle); };
  inline void GetObjData(LDObjData *data) { LDGetObjData(myLDHandle,data); };
  inline int GetCommDataSz(void) { return LDGetCommDataSz(myLDHandle); };
  inline void GetCommData(LDCommData *data) { LDGetCommData(myLDHandle,data); };

  inline void BackgroundLoad(double *walltime, double *cputime) {
    LDBackgroundLoad(myLDHandle,walltime,cputime);
  }

  inline void IdleTime(double *walltime) {
    LDIdleTime(myLDHandle,walltime);
  };

  inline void TotalTime(double *walltime, double *cputime) {
    LDTotalTime(myLDHandle,walltime,cputime);
  }

  inline void ClearLoads(void) { LDClearLoads(myLDHandle); };
  inline void Migrate(LDObjHandle h, int dest) { LDMigrate(h,dest); };

  inline void Migrated(LDObjHandle h) { LDMigrated(h); };

  inline LDBarrierClient AddLocalBarrierClient(LDResumeFn fn, void* data) {
    return LDAddLocalBarrierClient(myLDHandle,fn,data);
  };

  inline void RemoveLocalBarrierClient(LDBarrierClient h) {
    LDRemoveLocalBarrierClient(myLDHandle, h);
  };

  inline LDBarrierReceiver AddLocalBarrierReceiver(LDBarrierFn fn, void *data) {
    return LDAddLocalBarrierReceiver(myLDHandle,fn,data);
  };

  inline void RemoveLocalBarrierReceiver(LDBarrierReceiver h) {
    LDRemoveLocalBarrierReceiver(myLDHandle,h);
  };

  inline void AtLocalBarrier(LDBarrierClient h) { LDAtLocalBarrier(myLDHandle,h); }
  inline void LocalBarrierOn(void) { LDLocalBarrierOn(myLDHandle); };
  inline void LocalBarrierOff(void) { LDLocalBarrierOn(myLDHandle); };
  inline void ResumeClients() { LDResumeClients(myLDHandle); }

  inline int ProcessorSpeed() { return LDProcessorSpeed(); };
  inline void SetLBPeriod(double s) { LDSetLBPeriod(myLDHandle, s);}
  inline double GetLBPeriod() { return LDGetLBPeriod(myLDHandle);}
private:
  LDHandle myLDHandle;

};

void TurnManualLBOn();
void TurnManualLBOff();

#endif /* LDATABASE_H */

/*@}*/
