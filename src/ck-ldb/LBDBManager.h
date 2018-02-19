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

class client;
class receiver;

class LocalBarrier {
friend class LBDB;
public:
  LocalBarrier() { cur_refcount = 1; client_count = 0;
                   max_receiver= 0; at_count = 0; on = false; 
	#if CMK_BIGSIM_CHARM
	first_free_client_slot = 0;
	#endif
    };
  ~LocalBarrier() { };

  LDBarrierClient AddClient(LDResumeFn fn, void* data);
  void RemoveClient(LDBarrierClient h);
  LDBarrierReceiver AddReceiver(LDBarrierFn fn, void* data);
  void RemoveReceiver(LDBarrierReceiver h);
  void TurnOnReceiver(LDBarrierReceiver h);
  void TurnOffReceiver(LDBarrierReceiver h);
  void AtBarrier(LDBarrierClient h);
  void DecreaseBarrier(LDBarrierClient h, int c);
  void TurnOn() { on = true; CheckBarrier(); };
  void TurnOff() { on = false; };

private:
  void CallReceivers(void);
  void CheckBarrier();
  void ResumeClients(void);

  std::list<client*> clients;
  std::list<receiver*> receivers;

  int cur_refcount;
  int client_count;
  int max_receiver;
  int at_count;
  bool on;

  #if CMK_BIGSIM_CHARM
  int first_free_client_slot;
  #endif
};

class LBDB {
public:
  LBDB();
  ~LBDB() { }

  void SetPeriod(double secs) {batsync.setPeriod(secs);}
  double GetPeriod() {return batsync.getPeriod();}

  void insert(LBOM *om);

  LDOMHandle AddOM(LDOMid _userID, void* _userData, LDCallbacks _callbacks);
  void RemoveOM(LDOMHandle om);

  LDObjHandle AddObj(LDOMHandle _h, CmiUInt8 _id, void *_userData,
		     bool _migratable);
  void UnregisterObj(LDObjHandle _h);

  void RegisteringObjects(LDOMHandle _h);
  void DoneRegisteringObjects(LDOMHandle _h);

  inline void LocalBarrierOn() 
       { localBarrier.TurnOn();}
  inline void LocalBarrierOff() 
       { localBarrier.TurnOff();}

  inline LBOM *LbOM(LDOMHandle h) 
       { return oms[h.handle]; };
  inline LBObj *LbObj(const LDObjHandle &h) const 
       { return objs[h.handle].obj; };
  inline LBObj *LbObjIdx(int h) const 
       { return objs[h].obj; };
  void DumpDatabase(void);

  inline void TurnStatsOn(void) 
       {statsAreOn = true; machineUtil.StatsOn();}
  inline void TurnStatsOff(void) 
       {statsAreOn = false;machineUtil.StatsOff();}
  inline bool StatsOn(void) const 
       { return statsAreOn; };

  void SetupPredictor(LDPredictModelFn on, LDPredictWindowFn onWin, LDPredictFn off, LDPredictModelFn change, void* data);
  inline void TurnPredictorOn(void *model) {
    if (predictCBFn!=NULL) predictCBFn->on(predictCBFn->data, model);
    else CmiPrintf("Predictor not supported in this load balancer\n");
  }
  inline void TurnPredictorOn(void *model, int wind) {
    if (predictCBFn!=NULL) predictCBFn->onWin(predictCBFn->data, model, wind);
    else CmiPrintf("Predictor not supported in this load balancer\n");
  }
  inline void TurnPredictorOff(void) {
    if (predictCBFn!=NULL) predictCBFn->off(predictCBFn->data);
    else CmiPrintf("Predictor not supported in this load balancer\n");
  }
  /* the parameter model is really of class LBPredictorFunction in file LBDatabase.h */
  inline void ChangePredictor(void *model) {
    if (predictCBFn!=NULL) predictCBFn->change(predictCBFn->data, model);
    else CmiPrintf("Predictor not supported in this load balancer");
  }

  void Send(const LDOMHandle &destOM, const CmiUInt8 &destid, unsigned int bytes, int destObjProc);
  void MulticastSend(const LDOMHandle &destOM, CmiUInt8 *destids, int ndests, unsigned int bytes, int nMsgs);
  int ObjDataCount();
  void GetObjData(LDObjData *data);
  inline int CommDataCount() { 
    if (commTable)
      return commTable->CommCount();
    else return 0;
  }
  inline void GetCommData(LDCommData *data) 
       { if (commTable) commTable->GetCommData(data); };

  inline void GetCommInfo(int& bytes, int& msgs, int& withinbytes, int& acrossbytes, int& num_nghbors, int& hops, int& hopbytes) {
    if (commTable)
      commTable->GetCommInfo(bytes, msgs, withinbytes, acrossbytes, num_nghbors, hops, hopbytes);
  };

  void MetaLBResumeWaitingChares(int lb_ideal_period);
  void MetaLBCallLBOnChares();
  int  Migrate(LDObjHandle h, int dest);
  void Migrated(LDObjHandle h, int waitBarrier=1);
  int  NotifyMigrated(LDMigratedFn fn, void* data);
  void TurnOnNotifyMigrated(int handle)
       { migrateCBList[handle]->on = 1; }
  void TurnOffNotifyMigrated(int handle)
       { migrateCBList[handle]->on = 0; }
  void RemoveNotifyMigrated(int handle);

  inline void TurnManualLBOn() 
       { useBarrier = false; }
  inline void TurnManualLBOff() 
       { useBarrier = true; }

  int AddStartLBFn(LDStartLBFn fn, void* data);
  void TurnOnStartLBFn(int handle)
       { startLBFnList[handle]->on = 1; }
  void TurnOffStartLBFn(int handle)
       { startLBFnList[handle]->on = 0; }
  void RemoveStartLBFn(LDStartLBFn fn);
  void StartLB();

  int AddMigrationDoneFn(LDMigrationDoneFn fn, void* data);
  void RemoveMigrationDoneFn(LDMigrationDoneFn fn);
  void MigrationDone();

  inline void IdleTime(LBRealType* walltime) 
       { machineUtil.IdleTime(walltime); };
  inline void TotalTime(LBRealType* walltime, LBRealType* cputime) 
       { machineUtil.TotalTime(walltime,cputime); };
  void BackgroundLoad(LBRealType* walltime, LBRealType* cputime);
  void GetTime(LBRealType *total_walltime,LBRealType *total_cputime,
                   LBRealType *idletime, LBRealType *bg_walltime, LBRealType *bg_cputime);
  void ClearLoads(void);

  /**
    runningObj records the obj handler index so that load balancer
    knows if an event(e.g. Send) is in an entry function or not.
    An index is enough here because LDObjHandle can be retrieved from 
    objs array. Copying LDObjHandle is expensive.
  */
  inline void SetRunningObj(const LDObjHandle &_h) 
       { runningObj = _h.handle; obj_running = true; };
  inline const LDObjHandle &RunningObj() const 
       { return objs[runningObj].obj->GetLDObjHandle(); };
  inline void NoRunningObj() 
       { obj_running = false; };
  inline bool ObjIsRunning() const 
       { return obj_running; };
  
  inline LDBarrierClient AddLocalBarrierClient(LDResumeFn fn, void* data) 
       { return localBarrier.AddClient(fn,data); };
  inline void RemoveLocalBarrierClient(LDBarrierClient h) 
       { localBarrier.RemoveClient(h); };
  inline LDBarrierReceiver AddLocalBarrierReceiver(LDBarrierFn fn, void* data) 
       { return localBarrier.AddReceiver(fn,data); };
  inline void RemoveLocalBarrierReceiver(LDBarrierReceiver h) 
       { localBarrier.RemoveReceiver(h); };
  inline void TurnOnBarrierReceiver(LDBarrierReceiver h) 
       { localBarrier.TurnOnReceiver(h); };
  inline void TurnOffBarrierReceiver(LDBarrierReceiver h) 
       { localBarrier.TurnOffReceiver(h); };
  inline void AtLocalBarrier(LDBarrierClient h) 
       { if (useBarrier) localBarrier.AtBarrier(h); };
  inline void DecreaseLocalBarrier(LDBarrierClient h, int c) 
       { if (useBarrier) localBarrier.DecreaseBarrier(h, c); };
  inline void ResumeClients() 
       { localBarrier.ResumeClients(); };
  inline void MeasuredObjTime(double wtime, double ctime) {
    (void)ctime;
    if (statsAreOn) {
      obj_walltime += wtime;
#if CMK_LB_CPUTIMER
      obj_cputime += ctime;
#endif
    }
  };

  //This class controls the builtin-atsync frequency
  class batsyncer {
  private:
    LBDB *db; //Enclosing LBDB object
    double period;//Time (seconds) between builtin-atsyncs  
    double nextT;
    LDBarrierClient BH;//Handle for the builtin-atsync barrier 
    bool gotoSyncCalled;
    static void gotoSync(void *bs);
    static void resumeFromSync(void *bs);
  public:
    void init(LBDB *_db,double initPeriod);
    void setPeriod(double p) {period=p;}
    double getPeriod() {return period;}
  };

private:
  struct MigrateCB {
    LDMigratedFn fn;
    void* data;
    int on;
  };

  struct StartLBCB {
    LDStartLBFn fn;
    void* data;
    int on;
  };

  struct MigrationDoneCB {
    LDMigrationDoneFn fn;
    void* data;
  };

  struct PredictCB {
    LDPredictModelFn on;
    LDPredictWindowFn onWin;
    LDPredictFn off;
    LDPredictModelFn change;
    void* data;
  };

  struct LBObjEntry {
    LBObj* obj;
    LDObjIndex next;

    LBObjEntry(LBObj* obj, LDObjIndex next = -1) : obj(obj), next(next) {}
  };

  typedef CkVec<LBOM*> OMList;
  typedef std::vector<LBObjEntry> ObjList;
  typedef CkVec<MigrateCB*> MigrateCBList;
  typedef CkVec<StartLBCB*> StartLBCBList;
  typedef CkVec<MigrationDoneCB*> MigrationDoneCBList;

  LBCommTable* commTable;
  OMList oms;
  int omCount;
  int oms_registering;

  LDObjIndex objsEmptyHead;
  ObjList objs;

  bool statsAreOn;
  MigrateCBList migrateCBList;

  MigrationDoneCBList migrationDoneCBList;

  PredictCB* predictCBFn;

  bool obj_running;
  int runningObj;		// index of the runningObj in ObjList

  batsyncer batsync;

  LocalBarrier localBarrier;    // local barrier to trigger LB automatically
  bool useBarrier;           // use barrier or not

  LBMachineUtil machineUtil;
  double obj_walltime;
#if CMK_LB_CPUTIMER
  double obj_cputime;
#endif

  StartLBCBList  startLBFnList;
  int            startLBFn_count;
public:
  int useMem();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    int validObjHandle(LDObjHandle h ){
            if(h.handle >= objs.size())
                return 0;
            if(objs[h.handle].obj == NULL)
                return 0;

            return 1;
    }
#endif


  const ObjList& getObjs() {return objs;}



};

#endif

/*@}*/
