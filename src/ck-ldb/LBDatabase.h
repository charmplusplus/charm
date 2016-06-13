/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef LBDATABASE_H
#define LBDATABASE_H

#include "lbdb.h"
#include "LBDBManager.h"
#include "lbdb++.h"

#define LB_FORMAT_VERSION     3

class MetaBalancer;
extern int _lb_version;

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
  int _lb_metaLbOn;
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
    _lb_metaLbOn = 0;
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
  inline int & metaLbOn() {return _lb_metaLbOn;}
};

extern CkLBArgs _lb_args;

extern int _lb_predict;
extern int _lb_predict_delay;
extern int _lb_predict_window;
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

class LBDB;

CkpvExtern(int, numLoadBalancers);
CkpvExtern(int, hasNullLB);
CkpvExtern(int, lbdatabaseInited);

// LB options, mostly controled by user parameter
extern "C" char * _lbtopo;

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


class LBDatabase : public IrrGroup {
public:
  LBDatabase(void)  { init(); }
  LBDatabase(CkMigrateMessage *m)  { (void)m; init(); }
  ~LBDatabase()  { if (avail_vector) delete [] avail_vector; }
  
private:
  void init();
public:
  inline static LBDatabase * Object() { return CkpvAccess(lbdatabaseInited)?(LBDatabase *)CkLocalBranch(_lbdb):NULL; }
#if CMK_LBDB_ON
  inline LBDB *getLBDB() {return (LBDB*)(myLDHandle.handle);}
#endif

  static void initnodeFn(void);

  void pup(PUP::er& p);

  /*
   * Calls from object managers to load database
   */
  inline LDOMHandle RegisterOM(LDOMid userID, void *userptr, LDCallbacks cb) {
    return LDRegisterOM(myLDHandle,userID, userptr, cb);
  };

  inline void UnregisterOM(LDOMHandle omHandle) {
    return LDUnregisterOM(myLDHandle, omHandle);
  };

  inline void RegisteringObjects(LDOMHandle _om) {
    LDRegisteringObjects(_om);
  };

  inline void DoneRegisteringObjects(LDOMHandle _om) {
    LDDoneRegisteringObjects(_om);
  };

  void ResetAdaptive();

  inline LDObjHandle RegisterObj(LDOMHandle h, LDObjid id,
			  void *userptr,int migratable) {
    return LDRegisterObj(h,id,userptr,migratable);
  };

  inline void UnregisterObj(LDObjHandle h) { LDUnregisterObj(h); };

  inline void ObjTime(LDObjHandle h, double walltime, double cputime) {
    LDObjTime(h,walltime,cputime);
  };

  inline void GetObjLoad(LDObjHandle &h, LBRealType &walltime, LBRealType &cputime) {
    LDGetObjLoad(h,&walltime,&cputime);
  };

#if CMK_LB_USER_DATA
  inline void *GetDBObjUserData(LDObjHandle &h, int idx)
  {
    return LDDBObjUserData(h, idx);
  }
#endif

  inline void QueryKnownObjLoad(LDObjHandle &h, LBRealType &walltime, LBRealType &cputime) {
    LDQueryKnownObjLoad(h,&walltime,&cputime);
  };

  inline int RunningObject(LDObjHandle* _o) const { 
#if CMK_LBDB_ON
      LBDB *const db = (LBDB*)(myLDHandle.handle);
      if (db->ObjIsRunning()) {
        *_o = db->RunningObj();
        return 1;
      } 
#endif
      return 0;
      //return LDRunningObject(myLDHandle,_o);
  };
  inline const LDObjHandle *RunningObject() const { 
#if CMK_LBDB_ON
      LBDB *const db = (LBDB*)(myLDHandle.handle);
      if (db->ObjIsRunning()) {
        return &db->RunningObj();
      } 
#endif
      return NULL;
  };
  inline const LDObjHandle &GetObjHandle(int idx) { return LDGetObjHandle(myLDHandle, idx);}
  inline void ObjectStart(const LDObjHandle &_h) { LDObjectStart(_h); };
  inline void ObjectStop(const LDObjHandle &_h) { LDObjectStop(_h); };
  inline void Send(const LDOMHandle &_om, const LDObjid _id, unsigned int _b, int _p, int force = 0) {
    LDSend(_om, _id, _b, _p, force);
  };
  inline void MulticastSend(const LDOMHandle &_om, LDObjid *_ids, int _n, unsigned int _b, int _nMsgs=1) {
    LDMulticastSend(_om, _ids, _n, _b, _nMsgs);
  };

  void EstObjLoad(const LDObjHandle &h, double cpuload);
  inline void NonMigratable(LDObjHandle h) { LDNonMigratable(h); };
  inline void Migratable(LDObjHandle h) { LDMigratable(h); };
  inline void setPupSize(LDObjHandle h, size_t pup_size) {LDSetPupSize(h, pup_size);};
  inline void UseAsyncMigrate(LDObjHandle h, bool flag) { LDAsyncMigrate(h, flag); };
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

  inline void AddMigrationDoneFn(LDMigrationDoneFn fn, void *data) 
  {
    LDAddMigrationDoneFn(myLDHandle,fn,data);
  };

  inline void RemoveMigrationDoneFn(LDMigrationDoneFn fn) 
  {
    LDRemoveMigrationDoneFn(myLDHandle,fn);
  };

  inline void MigrationDone() { LDMigrationDone(myLDHandle); }

public:
  inline void TurnManualLBOn() { LDTurnManualLBOn(myLDHandle); }
  inline void TurnManualLBOff() { LDTurnManualLBOff(myLDHandle); }
 
  inline void PredictorOn(LBPredictorFunction *model) { LDTurnPredictorOn(myLDHandle,model); }
  inline void PredictorOn(LBPredictorFunction *model,int wind) { LDTurnPredictorOnWin(myLDHandle,model,wind); }
  inline void PredictorOff() { LDTurnPredictorOff(myLDHandle); }
  inline void ChangePredictor(LBPredictorFunction *model) { LDTurnPredictorOn(myLDHandle,model); }

  inline void CollectStatsOn(void) { LDCollectStatsOn(myLDHandle); };
  inline void CollectStatsOff(void) { LDCollectStatsOff(myLDHandle); };
  inline int  CollectingStats(void) { return LDCollectingStats(myLDHandle); };
  inline int  CollectingCommStats(void) { return LDCollectingStats(myLDHandle) && _lb_args.traceComm(); };
  inline void QueryEstLoad(void) { LDQueryEstLoad(myLDHandle); };

  inline int GetObjDataSz(void) { return LDGetObjDataSz(myLDHandle); };
  inline void GetObjData(LDObjData *data) { LDGetObjData(myLDHandle,data); };
  inline int GetCommDataSz(void) { return LDGetCommDataSz(myLDHandle); };
  inline void GetCommData(LDCommData *data) { LDGetCommData(myLDHandle,data); };

  inline void BackgroundLoad(LBRealType *walltime, LBRealType *cputime) {
    LDBackgroundLoad(myLDHandle,walltime,cputime);
  }

  inline void IdleTime(LBRealType *walltime) {
    LDIdleTime(myLDHandle,walltime);
  };

  inline void TotalTime(LBRealType *walltime, LBRealType *cputime) {
    LDTotalTime(myLDHandle,walltime,cputime);
  }

  inline void GetTime(LBRealType *total_walltime,LBRealType *total_cputime,
                   LBRealType *idletime, LBRealType *bg_walltime, LBRealType *bg_cputime) {
    LDGetTime(myLDHandle, total_walltime, total_cputime, idletime, bg_walltime, bg_cputime);
  }

  inline void ClearLoads(void) { LDClearLoads(myLDHandle); };
  inline int Migrate(LDObjHandle h, int dest) { return LDMigrate(h,dest); };

  inline void Migrated(LDObjHandle h, int waitBarrier=1) {
    LDMigrated(h, waitBarrier);
  };

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

  inline void AtLocalBarrier(LDBarrierClient h) {
    LDAtLocalBarrier(myLDHandle,h);
  }
  inline void DecreaseLocalBarrier(LDBarrierClient h, int c) {
    LDDecreaseLocalBarrier(myLDHandle,h,c);
  }
  inline void LocalBarrierOn(void) { LDLocalBarrierOn(myLDHandle); };
  inline void LocalBarrierOff(void) { LDLocalBarrierOn(myLDHandle); };
  void ResumeClients();
  inline int ProcessorSpeed() { return LDProcessorSpeed(); };
  inline void SetLBPeriod(double s) { LDSetLBPeriod(myLDHandle, s);}
  inline double GetLBPeriod() { return LDGetLBPeriod(myLDHandle);}

  inline void MetaLBResumeWaitingChares(int lb_period) {
#if CMK_LBDB_ON
    LDOMMetaLBResumeWaitingChares(myLDHandle, lb_period);
#endif
  }

  inline void MetaLBCallLBOnChares() {
#if CMK_LBDB_ON
    LDOMMetaLBCallLBOnChares(myLDHandle);
#endif
  }

  void SetMigrationCost(double cost);
  void SetStrategyCost(double cost);
	void UpdateDataAfterLB(double mLoad, double mCpuLoad, double avgLoad);

private:
  int mystep;
  LDHandle myLDHandle;
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
  static int manualOn;

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
  inline int useMem() { return LDMemusage(myLDHandle); }

  int getLoadbalancerTicket();
  void addLoadbalancer(BaseLB *lb, int seq);
  void nextLoadbalancer(int seq);
  const char *loadbalancer(int seq);

  inline int step() { return mystep; }
  inline void incStep() { mystep++; }
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
