/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef LBDATABASE_H
#define LBDATABASE_H

#include "lbdb.h"

#include "LBObj.h"
#include "LBOM.h"
#include "LBComm.h"
#include "LBMachineUtil.h"

#define LB_FORMAT_VERSION     3

class MetaBalancer;
extern int _lb_version;

class client;
class receiver;

class LocalBarrier {
friend class LBDatabase;
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

// command line options
class CkLBArgs
{
private:
  double _autoLbPeriod;		// in seconds
  double _lb_alpha;		// per message send overhead
  double _lb_beta;		// per byte send overhead
  int _lb_debug;		// 1 or greater
  int _lb_printsumamry;		// print summary
  int _lb_loop;                 // use multiple load balancers in loop
  int _lb_ignoreBgLoad;
  int _lb_migObjOnly;		// only consider migratable objs
  int _lb_syncResume;
  int _lb_samePeSpeed;		// ignore cpu speed
  int _lb_testPeSpeed;		// test cpu speed
  int _lb_useCpuTime;           // use cpu instead of wallclock time
  int _lb_statson;		// stats collection
  int _lb_traceComm;		// stats collection for comm
  int _lb_central_pe;           // processor number for centralized startegy
  int _lb_percentMovesAllowed; //Specifies restriction on num of chares to be moved(as a percentage of total number of chares). Used by RefineKLB
  int _lb_teamSize;		// specifies the team size for TeamLB
  int _lb_maxDistPhases;  // Specifies the max number of LB phases in DistributedLB
  double _lb_targetRatio; // Specifies the target load ratio for LBs that aim for a particular load ratio
  int _lb_metaLbOn;
  char* _lb_metaLbModelDir;

 public:
  CkLBArgs() {
#if CMK_BIGSIM_CHARM
    _autoLbPeriod = 0.02;       // bigsim needs it to be faster (lb may hang)
#else
    _autoLbPeriod = 0.5;	// 0.5 second default
#endif
    _lb_debug = _lb_ignoreBgLoad = _lb_syncResume = _lb_useCpuTime = 0;
    _lb_printsumamry = _lb_migObjOnly = 0;
    _lb_statson = _lb_traceComm = 1;
    _lb_percentMovesAllowed=100;
    _lb_loop = 0;
    _lb_central_pe = 0;
    _lb_teamSize = 1;
    _lb_maxDistPhases = 10;
    _lb_targetRatio = 1.05;
    _lb_metaLbOn = 0;
    _lb_metaLbModelDir = nullptr;
  }
  inline double & lbperiod() { return _autoLbPeriod; }
  inline int & debug() { return _lb_debug; }
  inline int & teamSize() {return _lb_teamSize; }
  inline int & printSummary() { return _lb_printsumamry; }
  inline int & lbversion() { return _lb_version; }
  inline int & loop() { return _lb_loop; }
  inline int & ignoreBgLoad() { return _lb_ignoreBgLoad; }
  inline int & migObjOnly() { return _lb_migObjOnly; }
  inline int & syncResume() { return _lb_syncResume; }
  inline int & samePeSpeed() { return _lb_samePeSpeed; }
  inline int & testPeSpeed() { return _lb_testPeSpeed; }
  inline int & useCpuTime() { return _lb_useCpuTime; }
  inline int & statsOn() { return _lb_statson; }
  inline int & traceComm() { return _lb_traceComm; }
  inline int & central_pe() { return _lb_central_pe; }
  inline double & alpha() { return _lb_alpha; }
  inline double & beta() { return _lb_beta; }
  inline int & percentMovesAllowed() { return _lb_percentMovesAllowed;}
  inline int & maxDistPhases() { return _lb_maxDistPhases; }
  inline double & targetRatio() { return _lb_targetRatio; }
  inline int & metaLbOn() {return _lb_metaLbOn;}
  inline char*& metaLbModelDir() { return _lb_metaLbModelDir; }
};

extern CkLBArgs _lb_args;

extern int _lb_predict;
extern int _lb_predict_delay;
extern int _lb_predict_window;
extern bool _lb_psizer_on;
#ifndef PREDICT_DEBUG
#define PREDICT_DEBUG  0   // 0 = No debug, 1 = Debug info on
#endif
#define PredictorPrintf  if (PREDICT_DEBUG) CmiPrintf

// used in constructor of all load balancers
class CkLBOptions
{
private:
  int seqno;		// for centralized lb, the seqno
public:
  CkLBOptions(): seqno(-1) {}
  CkLBOptions(int s): seqno(s) {}
  int getSeqNo() const { return seqno; }
};
PUPbytes(CkLBOptions)
                                                                                
#include "LBDatabase.decl.h"

extern CkGroupID _lbdb;

CkpvExtern(int, numLoadBalancers);
CkpvExtern(bool, hasNullLB);
CkpvExtern(bool, lbdatabaseInited);

// LB options, mostly controled by user parameter
extern char * _lbtopo;

typedef void (*LBCreateFn)();
typedef BaseLB * (*LBAllocFn)();
void LBDefaultCreate(LBCreateFn f);

void LBRegisterBalancer(const char *, LBCreateFn, LBAllocFn, const char *, int shown=1);

void _LBDBInit();

// main chare
class LBDBInit : public Chare {
  public:
    LBDBInit(CkArgMsg*);
    LBDBInit(CkMigrateMessage *m):Chare(m) {}
};

// class which implement a virtual function for the FuturePredictor
class LBPredictorFunction {
public:
  virtual ~LBPredictorFunction() {}
  int num_params;

  virtual void initialize_params(double *x) {double normall=1.0/pow((double)2,(double)31); for (int i=0; i<num_params; ++i) x[i]=rand()*normall;}

  virtual double predict(double x, double *params) =0;
  virtual void print(double *params) {PredictorPrintf("LB: unknown model\n");};
  virtual void function(double x, double *param, double &y, double *dyda) =0;
};

// a default implementation for a FuturePredictor function
class DefaultFunction : public LBPredictorFunction {
 public:
  // constructor
  DefaultFunction() {num_params=6;};

  // compute the prediction function for the variable x with parameters param
  double predict(double x, double *param) {return (param[0] + param[1]*x + param[2]*x*x + param[3]*sin(param[4]*(x+param[5])));}

  void print(double *param) {PredictorPrintf("LB: %f + %fx + %fx^2 + %fsin%f(x+%f)\n",param[0],param[1],param[2],param[3],param[4],param[5]);}

  // compute the prediction function and its derivatives respect to the parameters
  void function(double x, double *param, double &y, double *dyda) {
    double tmp;

    y = predict(x, param);

    dyda[0] = 1;
    dyda[1] = x;
    dyda[2] = x*x;
    tmp = param[4] * (x+param[5]);
    dyda[3] = sin(tmp);
    dyda[4] = param[3] * (x+param[5]) * cos(tmp);
    dyda[5] = param[3] * param[4] *cos(tmp);
  }
};


class LBDatabase : public CBase_LBDatabase {
public:
  class batsyncer {
  private:
    LBDatabase *db; //Enclosing LBDB object
    double period; //Time (seconds) between builtin-atsyncs
    double nextT;
    LDBarrierClient BH; //Handle for the builtin-atsync barrier
    bool gotoSyncCalled;
    static void gotoSync(void *bs);
    static void resumeFromSync(void *bs);
  public:
    void init(LBDatabase *_db, double initPeriod);
    void setPeriod(double p) { period = p; }
    double getPeriod() { return period; }
  };

private:
  struct LBObjEntry {
    LBObj* obj;
    LDObjIndex next;

    LBObjEntry(LBObj* obj, LDObjIndex next = -1) : obj(obj), next(next) {}
  };

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

  LBCommTable* commTable;
  std::vector<LBOM*> oms;
  int omCount;
  int omsRegistering;

  LDObjIndex objsEmptyHead;
  std::vector<LBObjEntry> objs;

  bool obj_running;
  LDObjIndex runningObj; // index of the runningObj in objs

  double obj_walltime;
#if CMK_LB_CPUTIMER
  double obj_cputime;
#endif

  bool statsAreOn;

  std::vector<MigrateCB*> migrateCBList;
  std::vector<StartLBCB*> startLBFnList;
  std::vector<MigrationDoneCB*> migrationDoneCBList;

  LBMachineUtil machineUtil;

  LocalBarrier localBarrier;    // local barrier to trigger LB automatically
  bool useBarrier;           // use barrier or not

  PredictCB* predictCBFn;

  batsyncer batsync;

  int startLBFn_count;

public:
  LBDatabase(void)  { init(); }
  LBDatabase(CkMigrateMessage *m) : CBase_LBDatabase(m)  { init(); }
  ~LBDatabase()  { if (avail_vector) delete [] avail_vector; }
  
private:
  void init();
public:
  inline static LBDatabase * Object() { return CkpvAccess(lbdatabaseInited)?(LBDatabase *)CkLocalBranch(_lbdb):NULL; }
  inline LBOM* LbOM(LDOMHandle h) {
    return oms[h.handle];
  }
  inline LBObj *LbObj(const LDObjHandle &h) const {
    return objs[h.handle].obj;
  }
  inline LBObj *LbObjIdx(int h) const {
    return objs[h].obj;
  }

  inline void TurnStatsOn(void)
       {statsAreOn = true; machineUtil.StatsOn();}
  inline void TurnStatsOff(void)
       {statsAreOn = false; machineUtil.StatsOff();}
  inline bool StatsOn(void) const
       { return statsAreOn; };


  /**
     runningObj records the obj handler index so that load balancer
     knows if an event(e.g. Send) is in an entry function or not.
     An index is enough here because LDObjHandle can be retrieved from
     objs array. Copying LDObjHandle is expensive.
  */
  inline void SetRunningObj(const LDObjHandle &_h) {
    runningObj = _h.handle;
    obj_running = true;
  }
  inline const LDObjHandle &RunningObj() const {
    return objs[runningObj].obj->GetLDObjHandle();
  }
  inline void NoRunningObj() {
    obj_running = false;
  }
  inline bool ObjIsRunning() const {
    return obj_running;
  }

  inline void MeasuredObjTime(double wtime, double ctime) {
    if (statsAreOn) {
      obj_walltime += wtime;
#if CMK_LB_CPUTIMER
      obj_cputime += ctime;
#endif
    }
  }

  static void initnodeFn(void);

  void pup(PUP::er& p);

  /*
   * Calls from object managers to load database
   */
  LDOMHandle RegisterOM(LDOMid userID, void *userptr, LDCallbacks cb);
  void UnregisterOM(LDOMHandle omh);

  void RegisteringObjects(LDOMHandle omh);
  void DoneRegisteringObjects(LDOMHandle omh);

  void ResetAdaptive();

  LDObjHandle RegisterObj(LDOMHandle omh, CmiUInt8 id, void* userPtr,
                          int migratable);
  void UnregisterObj(LDObjHandle h);

  inline void ObjTime(LDObjHandle h, double walltime, double cputime) {
    LbObj(h)->IncrementTime(walltime, cputime);
    MeasuredObjTime(walltime, cputime);
  };

  inline void GetObjLoad(LDObjHandle &h, LBRealType &walltime, LBRealType &cputime) {
    LbObj(h)->getTime(&walltime, &cputime);
  };

  inline void* GetObjUserData(LDObjHandle &h) {
    return LbObj(h)->getLocalUserData();
  }

#if CMK_LB_USER_DATA
  inline void* GetDBObjUserData(LDObjHandle &h, int idx)
  {
    return LbObj(h)->getDBUserData(idx);
  }
#endif

  inline void QueryKnownObjLoad(LDObjHandle &h, LBRealType &walltime, LBRealType &cputime) {
    LbObj(h)->lastKnownLoad(&walltime, &cputime);
  };

  inline int RunningObject(LDObjHandle* _o) const { 
#if CMK_LBDB_ON
      if (ObjIsRunning()) {
        *_o = RunningObj();
        return 1;
      } 
#endif
      return 0;
  };
  inline const LDObjHandle *RunningObject() const { 
#if CMK_LBDB_ON
      if (ObjIsRunning()) {
        return &(RunningObj());
      } 
#endif
      return NULL;
  };
  inline const LDObjHandle &GetObjHandle(int idx) {
    return LbObjIdx(idx)->GetLDObjHandle();
  }
  inline void ObjectStart(const LDObjHandle &h) {
    if (ObjIsRunning()) {
      ObjectStop(*RunningObject());
    }

    SetRunningObj(h);

    if (StatsOn()) {
      LbObj(h)->StartTimer();
    }
  };
  inline void ObjectStop(const LDObjHandle &h) {
    LBObj* const obj = LbObj(h);

    if (StatsOn()) {
      LBRealType walltime, cputime;
      obj->StopTimer(&walltime, &cputime);
      obj->IncrementTime(walltime, cputime);
      MeasuredObjTime(walltime, cputime);
    }

    NoRunningObj();
  };
  void Send(const LDOMHandle &destOM, const CmiUInt8 &destID, unsigned int bytes, int destObjProc, int force = 0);
  void MulticastSend(const LDOMHandle &_om, CmiUInt8 *_ids, int _n, unsigned int _b, int _nMsgs=1);

  void EstObjLoad(const LDObjHandle &h, double cpuload);
  inline void NonMigratable(LDObjHandle h) { LbObj(h)->SetMigratable(false); };
  inline void Migratable(LDObjHandle h) { LbObj(h)->SetMigratable(true); };
  inline void setPupSize(LDObjHandle h, size_t pup_size) { LbObj(h)->setPupSize(pup_size);};
  inline void UseAsyncMigrate(LDObjHandle h, bool flag) { LbObj(h)->UseAsyncMigrate(flag); };
  void DumpDatabase(void);

  /*
   * Calls from load balancer to load database
   */  
  int NotifyMigrated(LDMigratedFn fn, void* data);
  inline void TurnOnNotifyMigrated(int handle)
       { migrateCBList[handle]->on = 1; }
  inline void TurnOffNotifyMigrated(int handle)
       { migrateCBList[handle]->on = 0; }
  void RemoveNotifyMigrated(int handle);


  int AddStartLBFn(LDStartLBFn fn, void *data);
  void TurnOnStartLBFn(int handle)
       { startLBFnList[handle]->on = 1; }
  void TurnOffStartLBFn(int handle)
       { startLBFnList[handle]->on = 0; }
  void RemoveStartLBFn(LDStartLBFn fn);

  void StartLB();

  int AddMigrationDoneFn(LDMigrationDoneFn fn, void *data);
  void RemoveMigrationDoneFn(LDMigrationDoneFn fn);
  void MigrationDone();

public:
  inline void TurnManualLBOn() {
    useBarrier = false;
    LocalBarrierOff();
  }
  inline void TurnManualLBOff() {
    useBarrier = true;
    if (omsRegistering == 0) {
      LocalBarrierOn();
    }
  }
 
  inline void PredictorOn(LBPredictorFunction *model) {
    if (predictCBFn!=NULL) predictCBFn->on(predictCBFn->data, model);
    else CmiPrintf("Predictor not supported in this load balancer\n");
  }
  inline void PredictorOn(LBPredictorFunction *model, int wind) {
    if (predictCBFn!=NULL) predictCBFn->onWin(predictCBFn->data, model, wind);
    else CmiPrintf("Predictor not supported in this load balancer\n");
  }
  inline void PredictorOff() {
    if (predictCBFn!=NULL) predictCBFn->off(predictCBFn->data);
    else CmiPrintf("Predictor not supported in this load balancer\n");
  }
  inline void ChangePredictor(LBPredictorFunction *model) {
    if (predictCBFn!=NULL) predictCBFn->change(predictCBFn->data, model);
    else CmiPrintf("Predictor not supported in this load balancer");
  }

  void SetupPredictor(LDPredictModelFn on, LDPredictWindowFn onWin, LDPredictFn off, LDPredictModelFn change, void* data);

  inline void CollectStatsOn(void) {
    if (!StatsOn()) {
      if (ObjIsRunning()) {
        LbObj(*RunningObject())->StartTimer();
      }
      TurnStatsOn();
    }
  };
  inline void CollectStatsOff(void) { TurnStatsOff(); };
  inline int  CollectingStats(void) {
  #if CMK_LBDB_ON
    return StatsOn();
  #else
    return 0;
  #endif
  };
  inline int CollectingCommStats(void) { return CollectingStats() && _lb_args.traceComm(); };

  int GetObjDataSz(void);
  void GetObjData(LDObjData *data);
  inline int GetCommDataSz(void) {
    if (commTable)
      return commTable->CommCount();
    else return 0;
  }

  inline void GetCommData(LDCommData *data) {
    if (commTable) commTable->GetCommData(data);
  }

  inline void GetCommInfo(int& bytes, int& msgs, int& withinbytes, int& outsidebytes, int& num_nghbors, int& hops, int& hopbytes) {
    if (commTable)
      commTable->GetCommInfo(bytes, msgs, withinbytes, outsidebytes, num_nghbors, hops, hopbytes);
  }

  void BackgroundLoad(LBRealType *walltime, LBRealType *cputime);

  inline void IdleTime(LBRealType *walltime) {
    machineUtil.IdleTime(walltime);
  }
  inline void TotalTime(LBRealType *walltime, LBRealType *cputime) {
    machineUtil.TotalTime(walltime, cputime);
  }

  void GetTime(LBRealType *total_walltime, LBRealType *total_cputime,
               LBRealType *idletime, LBRealType *bg_walltime,
               LBRealType *bg_cputime);

  void ClearLoads(void);

  int Migrate(LDObjHandle h, int dest);
  void Migrated(LDObjHandle h, int waitBarrier=1);

  inline LDBarrierClient AddLocalBarrierClient(LDResumeFn fn, void* data) {
    return localBarrier.AddClient(fn,data);
  }
  inline void RemoveLocalBarrierClient(LDBarrierClient h) {
    localBarrier.RemoveClient(h);
  }
  inline LDBarrierReceiver AddLocalBarrierReceiver(LDBarrierFn fn, void* data) {
    return localBarrier.AddReceiver(fn,data);
  }
  inline void RemoveLocalBarrierReceiver(LDBarrierReceiver h) {
    localBarrier.RemoveReceiver(h);
  }
  inline void AtLocalBarrier(LDBarrierClient h) {
    localBarrier.AtBarrier(h);
  }
  inline void DecreaseLocalBarrier(LDBarrierClient h, int c) {
    localBarrier.DecreaseBarrier(h, c);
  }
  inline void TurnOnBarrierReceiver(LDBarrierReceiver h) {
    localBarrier.TurnOnReceiver(h);
  }
  inline void TurnOffBarrierReceiver(LDBarrierReceiver h) {
    localBarrier.TurnOffReceiver(h);
  }

  inline void LocalBarrierOn(void) { localBarrier.TurnOn(); };
  inline void LocalBarrierOff(void) { localBarrier.TurnOff(); };
  void ResumeClients();
  static int ProcessorSpeed();
  inline void SetLBPeriod(double s) { batsync.setPeriod(s);} // s in seconds
  inline double GetLBPeriod() { return batsync.getPeriod();}

  void MetaLBResumeWaitingChares(int lb_period);
  void MetaLBCallLBOnChares();

  void SetMigrationCost(double cost);
  void SetStrategyCost(double cost);
  void UpdateDataAfterLB(double mLoad, double mCpuLoad, double avgLoad);

private:
  int mystep;
  static char *avail_vector;	// processor bit vector
  static bool avail_vector_set;
  int new_ld_balancer;		// for Node 0
  CkVec<BaseLB *>   loadbalancers;
  int nloadbalancers;
  MetaBalancer* metabalancer;

public:
  BaseLB** getLoadBalancers() {return loadbalancers.getVec();}
  int getNLoadBalancers() {return nloadbalancers;}

public:
  static bool manualOn;
public:
  char *availVector() { return avail_vector; }
  void get_avail_vector(char * bitmap);
  void set_avail_vector(char * bitmap, int new_ld=-1);
  int & new_lbbalancer() { return new_ld_balancer; }

  struct LastLBInfo {
    LBRealType *expectedLoad;
    LastLBInfo();
  };
  LastLBInfo lastLBInfo;
  inline LBRealType myExpectedLoad() { return lastLBInfo.expectedLoad[CkMyPe()]; }
  inline LBRealType* expectedLoad() { return lastLBInfo.expectedLoad; }
  int useMem();

  int getLoadbalancerTicket();
  void addLoadbalancer(BaseLB *lb, int seq);
  void nextLoadbalancer(int seq);
  void switchLoadbalancer(int switchFrom, int switchTo);
  const char *loadbalancer(int seq);

  inline int step() { return mystep; }
  inline void incStep() { mystep++; }

  const std::vector<LBObjEntry>& getObjs() {return objs;}
};

void TurnManualLBOn();
void TurnManualLBOff();

void LBTurnPredictorOn(LBPredictorFunction *model);
void LBTurnPredictorOn(LBPredictorFunction *model, int wind);
void LBTurnPredictorOff();
void LBChangePredictor(LBPredictorFunction *model);

void LBSetPeriod(double second);

#if CMK_LB_USER_DATA
int LBRegisterObjUserData(int size);
#endif

extern "C" void LBTurnInstrumentOn();
extern "C" void LBTurnInstrumentOff();
extern "C" void LBTurnCommOn();
extern "C" void LBTurnCommOff();
void LBClearLoads();

inline LBDatabase* LBDatabaseObj() { return LBDatabase::Object(); }

inline void CkStartLB() { LBDatabase::Object()->StartLB(); }

inline void get_avail_vector(char * bitmap) {
  LBDatabaseObj()->get_avail_vector(bitmap);
}

inline void set_avail_vector(char * bitmap) {
  LBDatabaseObj()->set_avail_vector(bitmap);
}

//  a helper class to suspend/resume load instrumentation when calling into
//  runtime apis

class SystemLoad
{
  const LDObjHandle *objHandle;
  LBDatabase *lbdb;
public:
  SystemLoad() {
    lbdb = LBDatabaseObj();
    objHandle = lbdb->RunningObject();
    if (objHandle != NULL) {
      lbdb->ObjectStop(*objHandle);
    }
  }
  ~SystemLoad() {
    if (objHandle) lbdb->ObjectStart(*objHandle);
  }
};

#define CK_RUNTIME_API          SystemLoad load_entry;

#endif /* LDATABASE_H */

/*@}*/
