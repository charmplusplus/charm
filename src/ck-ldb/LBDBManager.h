#ifndef LBDB_H
#define LBDB_H

#if CMK_STL_USE_DOT_H
#include <vector.h>
#else  // CMK_STL_NO_DOT_H
#include <vector>
#endif

#include "converse.h"
#include "lbdb.h"

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

#if CMK_STL_USE_DOT_H
  vector<client*> clients;
  vector<receiver*> receivers;
#else
  std::vector<client*> clients;
  std::vector<receiver*> receivers;
#endif

  int cur_refcount;
  int max_client;
  int client_count;
  int max_receiver;
  int at_count;
  CmiBool on;
};

class LBDB {
public:
  LBDB() {
    statsAreOn = CmiFalse;
    omCount = objCount = oms_registering = 0;
    obj_running = CmiFalse;
    commTable = new LBCommTable;
    obj_walltime = obj_cputime = 0;
  }

  ~LBDB() { }

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

private:
  struct MigrateCB {
    LDMigratedFn fn;
    void* data;
  };

#if CMK_STL_USE_DOT_H
  typedef vector<LBOM*> OMList;
  typedef vector<LBObj*> ObjList;
  typedef vector<MigrateCB*> MigrateCBList;
#else
  typedef std::vector<LBOM*> OMList;
  typedef std::vector<LBObj*> ObjList;
  typedef std::vector<MigrateCB*> MigrateCBList;
#endif
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

  LocalBarrier localBarrier;
  LBMachineUtil machineUtil;
  double obj_walltime;
  double obj_cputime;
};

#endif

