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

#ifndef LBDB_H
#define LBDB_H

#include "converse.h"
#include "lbdb.h"
#include "cklists.h"

#include "LBObj.h"
#include "LBOM.h"
#include "LBComm.h"
#include "LBMachineUtil.h"

class LocalBarrier {
friend class LBDB;
public:
  LocalBarrier() { cur_refcount = 1; client_count = 0; max_client = 0;
                   max_receiver= 0; at_count = 0; on = CmiFalse; };
  ~LocalBarrier() { };

  LDBarrierClient AddClient(LDResumeFn fn, void* data);
  void RemoveClient(LDBarrierClient h);
  LDBarrierReceiver AddReceiver(LDBarrierFn fn, void* data);
  void RemoveReceiver(LDBarrierReceiver h);
  void AtBarrier(LDBarrierClient h);
  void TurnOn() { on = CmiTrue; CheckBarrier(); };
  void TurnOff() { on = CmiFalse; };

private:
  void CallReceivers(void);
  void CheckBarrier();
  void ResumeClients(void);

  struct client {
    void* data;
    LDResumeFn fn;
    int refcount;
  };
   struct receiver {
    void* data;
    LDBarrierFn fn;
  };

  CkVec<client*> clients;
  CkVec<receiver*> receivers;

  int cur_refcount;
  int max_client;
  int client_count;
  int max_receiver;
  int at_count;
  CmiBool on;
};

class LBDB {
public:
  LBDB();
  ~LBDB() { }

  void SetPeriod(double secs) {batsync.setPeriod(secs);}

  void insert(LBOM *om);

  LDOMHandle AddOM(LDOMid _userID, void* _userData, 
		   LDCallbacks _callbacks);
  LDObjHandle AddObj(LDOMHandle _h, LDObjid _id, void *_userData,
		     CmiBool _migratable);
  void UnregisterObj(LDObjHandle _h);

  void RegisteringObjects(LDOMHandle _h);
  void DoneRegisteringObjects(LDOMHandle _h);

  LBOM *LbOM(LDOMHandle h) { return oms[h.handle]; };
  LBObj *LbObj(LDObjHandle h) { return objs[h.handle]; };
  void DumpDatabase(void);
  void TurnStatsOn(void) { statsAreOn = CmiTrue; machineUtil.StatsOn(); };
  void TurnStatsOff(void) { statsAreOn = CmiFalse; machineUtil.StatsOff(); };
  CmiBool StatsOn(void) { return statsAreOn; };
  void Send(LDOMHandle destOM, LDObjid destid, unsigned int bytes);
  int ObjDataCount();
  void GetObjData(LDObjData *data);
  int CommDataCount() { 
    if (commTable)
      return commTable->CommCount();
    else return 0;
  }
  void GetCommData(LDCommData *data) { 
    if (commTable) commTable->GetCommData(data);
  };

  void Migrate(LDObjHandle h, int dest);
  void Migrated(LDObjHandle h);
  void NotifyMigrated(LDMigratedFn fn, void* data);
  void IdleTime(double* walltime) { 
    machineUtil.IdleTime(walltime); 
  };
  void TotalTime(double* walltime, double* cputime) {
    machineUtil.TotalTime(walltime,cputime);
  };
  void BackgroundLoad(double* walltime, double* cputime);
  void ClearLoads(void);
  void SetRunningObj(LDObjHandle _h) {
    runningObj = _h; obj_running = CmiTrue;
  };
  void NoRunningObj() { obj_running = CmiFalse; };
  CmiBool ObjIsRunning() { return obj_running; };
  LDObjHandle RunningObj() { return runningObj; };
  
  LDBarrierClient AddLocalBarrierClient(LDResumeFn fn, void* data) { 
    return localBarrier.AddClient(fn,data);
  };
  void RemoveLocalBarrierClient(LDBarrierClient h) {
    localBarrier.RemoveClient(h);
  };
  LDBarrierReceiver AddLocalBarrierReceiver(LDBarrierFn fn, void* data) {
    return localBarrier.AddReceiver(fn,data);
  };
  void RemoveLocalBarrierReceiver(LDBarrierReceiver h) {
    localBarrier.RemoveReceiver(h);
  };
  void AtLocalBarrier(LDBarrierClient h) {
    localBarrier.AtBarrier(h);
  };
  void ResumeClients() {
    localBarrier.ResumeClients();
  };
  void MeasuredObjTime(double wtime, double ctime) {
    if (statsAreOn) {
      obj_walltime += wtime;
      obj_cputime += ctime;
    }
  };

  //This class controls the builtin-atsync frequency
  class batsyncer {
  private:
    LBDB *db; //Enclosing LBDB object
    double period;//Time (seconds) between builtin-atsyncs  
    LDBarrierClient BH;//Handle for the builtin-atsync barrier 
    static void gotoSync(void *bs);
    static void resumeFromSync(void *bs);
  public:
    void init(LBDB *_db,double initPeriod);
    void setPeriod(double p) {period=p;}
  };

private:
  struct MigrateCB {
    LDMigratedFn fn;
    void* data;
  };

  typedef CkVec<LBOM*> OMList;
  typedef CkVec<LBObj*> ObjList;
  typedef CkVec<MigrateCB*> MigrateCBList;

  LBCommTable* commTable;
  OMList oms;
  int omCount;
  int oms_registering;
  ObjList objs;
  int objCount;
  CmiBool statsAreOn;
  MigrateCBList migrateCBList;
  CmiBool obj_running;
  LDObjHandle runningObj;

  batsyncer batsync;

  LocalBarrier localBarrier;
  LBMachineUtil machineUtil;
  double obj_walltime;
  double obj_cputime;
};

#endif

/*@}*/
