/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef LBMANAGER_H
#define LBMANAGER_H

#include "LBDatabase.h"
#include "json.hpp"
using json = nlohmann::json;

#define LB_FORMAT_VERSION     3


class MetaBalancer;
extern int _lb_version;

class client;
class receiver;


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
  int _lb_teamSize;		// specifies the team size for TeamLB
  int _lb_maxDistPhases;  // Specifies the max number of LB phases in DistributedLB
  double _lb_targetRatio; // Specifies the target load ratio for LBs that aim for a particular load ratio
  int _lb_metaLbOn;
  char* _lb_metaLbModelDir;
  char* _lb_treeLBFile = (char*)"treelb.json";
  std::vector<const char*> _lb_legacyCentralizedStrategies;  // list of centralized strategies specified by command-line (legacy mode)

 public:
  CkLBArgs() {
#if CMK_BIGSIM_CHARM
    _autoLbPeriod = 0.02;       // bigsim needs it to be faster (lb may hang)
#else
    _autoLbPeriod = -1.0;	// 0.5 second default
#endif
    _lb_debug = _lb_ignoreBgLoad = _lb_syncResume = _lb_useCpuTime = 0;
    _lb_printsumamry = _lb_migObjOnly = 0;
    _lb_statson = _lb_traceComm = 1;
    _lb_loop = 0;
    _lb_central_pe = 0;
    _lb_teamSize = 1;
    _lb_maxDistPhases = 10;
    _lb_targetRatio = 1.05;
    _lb_metaLbOn = 0;
    _lb_metaLbModelDir = nullptr;
  }
  inline char*& treeLBFile() { return _lb_treeLBFile; }
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
  inline int & maxDistPhases() { return _lb_maxDistPhases; }
  inline double & targetRatio() { return _lb_targetRatio; }
  inline int & metaLbOn() {return _lb_metaLbOn;}
  inline char*& metaLbModelDir() { return _lb_metaLbModelDir; }
  inline std::vector<const char*>& legacyCentralizedStrategies() { return _lb_legacyCentralizedStrategies; }
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

#include "LBManager.decl.h"

extern CkGroupID _lbmgr;

CkpvExtern(int, numLoadBalancers);
CkpvExtern(bool, lbmanagerInited);

// LB options, mostly controled by user parameter
extern char * _lbtopo;

typedef void (*LBCreateFn)();
typedef BaseLB * (*LBAllocFn)();
void LBDefaultCreate(LBCreateFn f);

void LBRegisterBalancer(const char *, LBCreateFn, LBAllocFn, const char *, int shown=1);

class LocalBarrier {
friend class LBManager;
public:
  LocalBarrier() { cur_refcount = 1; client_count = 0; iter_no = -1; propagated_atsync_step=0;
                   max_receiver= 0; at_count = 0; on = false;
  #if CMK_BIGSIM_CHARM
  first_free_client_slot = 0;
  #endif
    };
  ~LocalBarrier() { };

  void SetMgr(LBManager *mgr){ _mgr = mgr;};
  LDBarrierReceiver AddReceiver(LDBarrierFn fn, void* data);
  void propagate_atsync();
  void RemoveReceiver(LDBarrierReceiver h);
  void TurnOnReceiver(LDBarrierReceiver h);
  void TurnOffReceiver(LDBarrierReceiver h);
  void AtBarrier(Chare* _n_c, int recvd_iter=-1);
  void DecreaseBarrier(int c);
  void TurnOn() { on = true; CheckBarrier(); };
  void TurnOff() { on = false; };

public:
  void CallReceivers(void);
  void CheckBarrier(int recvd_iter=-1);
  void ResumeClients(void);

  std::list<receiver*> receivers;

  LBManager *_mgr;

  int cur_refcount;
  int client_count;
  int max_receiver;
  int at_count;
  bool on;
  int propagated_atsync_step;
  int step;

  int iter_no;

  #if CMK_BIGSIM_CHARM
  int first_free_client_slot;
  #endif
};

void _LBMgrInit();

// main chare
class LBMgrInit : public Chare {
  public:
    LBMgrInit(CkArgMsg*);
    LBMgrInit(CkMigrateMessage *m):Chare(m) {}
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


class LBManager : public CBase_LBManager {
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

  LBDatabase *lbdb_obj;

  LocalBarrier localBarrier;    // local barrier to trigger LB automatically
  bool useBarrier;           // use barrier or not

  PredictCB* predictCBFn;

  int startLBFn_count;

public:
  std::list<Chare*> chares;
  std::list<int> local_pes_to_notify;
  int chare_count;
  bool received_from_left;
  bool received_from_right;
  bool received_from_rank0;
  bool rank0pe;

  bool startedAtSync;

  LBManager(void)  { init(); }
  LBManager(CkMigrateMessage *m) : CBase_LBManager(m)  { init(); }
  ~LBManager()  { if (avail_vector) delete [] avail_vector; }

private:
  void init();
public:
  LBDatabase *getLBDB() {return lbdb_obj;}
  inline static LBManager * Object() { return CkpvAccess(lbmanagerInited)?(LBManager *)CkLocalBranch(_lbmgr):NULL; }


  static void initnodeFn(void);

  void pup(PUP::er& p);

  void configureTreeLB(const char *json_str);
  void configureTreeLB(json &config);

  /*
   * Calls from object managers to load database
   */
//  LDOMHandle RegisterOM(LDOMid userID, void *userptr, LDCallbacks cb);

  static void periodicLB(void*);
  void callAt();
  void setTimer();

  LDOMHandle RegisterOM(LDOMid userID, void *userptr, LDCallbacks cb)
  { return lbdb_obj->RegisterOM(userID, userptr, cb);}
  int Migrate(LDObjHandle h, int dest) { return lbdb_obj->Migrate(h, dest);}
  void UnregisterOM(LDOMHandle omh) { lbdb_obj->UnregisterOM(omh);}
  void RegisteringObjects(LDOMHandle omh) { lbdb_obj->RegisteringObjects(this, omh);}
  void DoneRegisteringObjects(LDOMHandle omh) { lbdb_obj->DoneRegisteringObjects(this, omh);}
  void ObjectStart(const LDObjHandle &h) { lbdb_obj->ObjectStart(h);}
  void ObjectStop(const LDObjHandle &h) { lbdb_obj->ObjectStop(h);}
  void NonMigratable(LDObjHandle h) { lbdb_obj->NonMigratable(h);}
  void Migratable(LDObjHandle h) { lbdb_obj->Migratable(h);}
  void setPupSize(LDObjHandle h, size_t pup_size) { lbdb_obj->setPupSize(h, pup_size);}
  void UseAsyncMigrate(LDObjHandle h, bool flag) { lbdb_obj->UseAsyncMigrate(h, flag); };
  int GetObjDataSz(void) { return lbdb_obj->GetObjDataSz();}
  int GetCommDataSz(void) { return lbdb_obj->GetCommDataSz();}
  void GetObjData(LDObjData *data) { lbdb_obj->GetObjData(data);}
  void GetCommData(LDCommData *data) { lbdb_obj->GetCommData(data);}
  void GetCommInfo(int& bytes, int& msgs, int& withinbytes, int& outsidebytes,
                   int& num_nghbors, int& hops, int& hopbytes) {
    lbdb_obj->GetCommInfo(bytes, msgs, withinbytes, outsidebytes, num_nghbors, hops, hopbytes);}
  void CollectStatsOn(void) { lbdb_obj->CollectStatsOn();}
  void CollectStatsOff(void) { lbdb_obj->CollectStatsOff();}
  int RunningObject(LDObjHandle* _o) const { return lbdb_obj->RunningObject(_o);}
  const LDObjHandle *RunningObject() { return lbdb_obj->RunningObject();}
  void ObjTime(LDObjHandle h, double walltime, double cputime) {
    lbdb_obj->ObjTime(h, walltime, cputime);}
  void GetObjLoad(LDObjHandle &h, LBRealType &walltime, LBRealType &cputime) {
    lbdb_obj->GetObjLoad(h, walltime, cputime);
  };
  void* GetObjUserData(LDObjHandle &h) { return lbdb_obj->GetObjUserData(h);}
  void MetaLBCallLBOnChares() { lbdb_obj->MetaLBCallLBOnChares();}
  void MetaLBResumeWaitingChares(int lb_period) {
    lbdb_obj->MetaLBResumeWaitingChares(lb_period);}
  void ClearLoads(void) { lbdb_obj->ClearLoads();}
  const LDObjHandle &GetObjHandle(int idx) { return lbdb_obj->GetObjHandle(idx);}
  void IdleTime(LBRealType *walltime) { lbdb_obj->IdleTime(walltime);}
  void TotalTime(LBRealType *walltime, LBRealType *cputime) {
    lbdb_obj->TotalTime(walltime, cputime);}

  void GetTime(LBRealType *total_walltime, LBRealType *total_cputime,
               LBRealType *idletime, LBRealType *bg_walltime,
               LBRealType *bg_cputime) { lbdb_obj->GetTime(total_walltime,
               total_cputime, idletime, bg_walltime, bg_cputime);}
  LDObjHandle RegisterObj(LDOMHandle omh, CmiUInt8 id,
                          void* userPtr, int migratable) {
    return lbdb_obj->RegisterObj(omh, id, userPtr, migratable);
  }
  void UnregisterObj(LDObjHandle h) { lbdb_obj->UnregisterObj(h);}
  void EstObjLoad(const LDObjHandle &h, double cpuload) {
    lbdb_obj->EstObjLoad(h, cpuload);}
  void BackgroundLoad(LBRealType *walltime, LBRealType *cputime) {
    lbdb_obj->BackgroundLoad(walltime, cputime);}
  void Send(const LDOMHandle &destOM, const CmiUInt8 &destID, unsigned int bytes, int destObjProc, int force = 0) {
    lbdb_obj->Send(destOM, destID, bytes, destObjProc, force);}
  void MulticastSend(const LDOMHandle &_om, CmiUInt8 *_ids, int _n, unsigned int _b, int _nMsgs=1) {
    lbdb_obj->MulticastSend(_om, _ids, _n, _b, _nMsgs);}
  int useMem(){ return lbdb_obj->useMem(this);}

#if CMK_LB_USER_DATA
  inline void* GetDBObjUserData(LDObjHandle &h, int idx)
  {
    return lbdb_obj->LbObj(h)->getDBUserData(idx);
  }
#endif

  void ResetAdaptive();

  void DumpDatabase(void);

  void reset();
  void recv_lb_start(int lb_step, int phynode, int pe);
  void invoke_lb_start(int pe, int lb_step, int phynode, int mype);

  /*
   * Calls from load balancer to load database
   */

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
  inline void TurnManualLBOn() { useBarrier = false; }
  inline void TurnManualLBOff() { useBarrier = true; }

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

  inline int CollectingCommStats(void) { return lbdb_obj->CollectingStats() && _lb_args.traceComm(); };

  void Migrated(LDObjHandle h, int waitBarrier=1);

  void AddClients(Chare* _new_chare, bool atsync, bool atsync_notify);
  void RemoveClients(Chare* _new_chare);

  inline LDBarrierReceiver AddLocalBarrierReceiver(LDBarrierFn fn, void* data) {
    return localBarrier.AddReceiver(fn,data);
  }
  inline void RemoveLocalBarrierReceiver(LDBarrierReceiver h) {
    localBarrier.RemoveReceiver(h);
  }
  inline void AtLocalBarrier(Chare* _n_c) {
    if (useBarrier) localBarrier.AtBarrier(_n_c);
  }
  inline void DecreaseLocalBarrier(int c) {
    if (useBarrier) localBarrier.DecreaseBarrier(c);
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
  inline void SetLBPeriod(double s) { }
  inline double GetLBPeriod() { return 0;}

  void SetMigrationCost(double cost);
  void SetStrategyCost(double cost);
  void UpdateDataAfterLB(double mLoad, double mCpuLoad, double avgLoad);


private:
  int mystep;
  static char *avail_vector;	// processor bit vector
  static bool avail_vector_set;
  int new_ld_balancer;		// for Node 0
  MetaBalancer* metabalancer;
public:
  int nloadbalancers;
  CkVec<BaseLB *>   loadbalancers;

  std::vector<MigrateCB*> migrateCBList;
  std::vector<StartLBCB*> startLBFnList;
  std::vector<MigrationDoneCB*> migrationDoneCBList;

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

  int getLoadbalancerTicket();
  void addLoadbalancer(BaseLB *lb, int seq);
  void nextLoadbalancer(int seq);
  void switchLoadbalancer(int switchFrom, int switchTo);
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

inline LBManager* LBManagerObj() { return LBManager::Object(); }

inline void CkStartLB() { LBManager::Object()->StartLB(); }

inline void get_avail_vector(char * bitmap) {
  LBManagerObj()->get_avail_vector(bitmap);
}

inline void set_avail_vector(char * bitmap) {
  LBManagerObj()->set_avail_vector(bitmap);
}

//  a helper class to suspend/resume load instrumentation when calling into
//  runtime apis

class SystemLoad
{
  const LDObjHandle *objHandle;
  LBManager *lbmgr;
public:
  SystemLoad() {
    lbmgr = LBManagerObj();
    objHandle = lbmgr->RunningObject();
    if (objHandle != NULL) {
      lbmgr->ObjectStop(*objHandle);
    }
  }
  ~SystemLoad() {
    if (objHandle) lbmgr->ObjectStart(*objHandle);
  }
};

#define CK_RUNTIME_API          SystemLoad load_entry;

#endif /* LDATABASE_H */

/*@}*/
