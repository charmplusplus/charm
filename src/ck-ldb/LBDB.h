#ifndef LBDB_H
#define LBDB_H

#if CMK_STL_USE_DOT_H
#include <vector.h>
#else  // CMK_STL_NO_DOT_H
#include <vector>
#endif

#include "lbdb.h"

#include "LBObj.h"
#include "LBOM.h"

class LocalBarrier {
friend class LBDB;
public:
  LocalBarrier() { cur_refcount = 1; client_count = 0; max_client = 0;
                   max_receiver= 0; at_count = 0; on = False; };
  ~LocalBarrier() { };

  LDBarrierClient AddClient(LDResumeFn fn, void* data);
  void RemoveClient(LDBarrierClient h);
  LDBarrierReceiver AddReceiver(LDBarrierFn fn, void* data);
  void RemoveReceiver(LDBarrierReceiver h);
  void AtBarrier(LDBarrierClient h);
  void TurnOn() { on = True; CheckBarrier(); };
  void TurnOff() { on = False; };

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
  Bool on;
};

class LBDB {
public:
  LBDB() {
    statsAreOn = False;
    omCount = objCount = oms_registering = 0;
  }

  ~LBDB() { }

  void insert(LBOM *om);

  LDOMHandle AddOM(LDOMid _userID, void* _userData, 
		   LDCallbacks _callbacks);
  LDObjHandle AddObj(LDOMHandle _h, LDObjid _id, void *_userData,
		     Bool _migratable);
  void UnregisterObj(LDObjHandle _h);

  void RegisteringObjects(LDOMHandle _h);
  void DoneRegisteringObjects(LDOMHandle _h);

  LBOM *LbOM(LDOMHandle h) { return oms[h.handle]; };
  LBObj *LbObj(LDObjHandle h) { return objs[h.handle]; };
  void DumpDatabase(void);
  void TurnStatsOn(void) { statsAreOn = True; };
  void TurnStatsOff(void) { statsAreOn = False; };
  Bool StatsOn(void) { return statsAreOn; };
  int ObjDataCount();
  void GetObjData(LDObjData *data);
  LDObjData *FetchData(int *nitems);
  void Migrate(LDObjHandle h, int dest);
  void Migrated(LDObjHandle h);
  void NotifyMigrated(LDMigratedFn fn, void* data);
  void ClearLoads(void);
  
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

  OMList oms;
  int omCount;
  int oms_registering;
  ObjList objs;
  int objCount;
  Bool statsAreOn;
  MigrateCBList migrateCBList;

  LocalBarrier localBarrier;
};

#endif

