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

#include <math.h>
#include "lbdb.h"

// command line options
class CkLBArgs
{
private:
  double _autoLbPeriod;		// in seconds
  int _lb_debug;
  int _lb_ignoreBgLoad;
  int _lb_syncResume;
public:
  CkLBArgs() {
    _autoLbPeriod = 1.0;
    _lb_debug = _lb_ignoreBgLoad = _lb_syncResume = 0;
  }
  double & lbperiod() { return _autoLbPeriod; }
  int & debug() { return _lb_debug; }
  int & ignoreBgLoad() { return _lb_ignoreBgLoad; }
  int & syncResume() { return _lb_syncResume; }
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
PUPbytes(CkLBOptions);
                                                                                
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
  int num_params;

  virtual void initialize_params(double *x) {double normall=1.0/pow((double)2,31); for (int i=0; i<num_params; ++i) x[i]=rand()*normall;}

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
  LBDatabase(CkMigrateMessage *m)  { init(); }
  ~LBDatabase()  { delete [] avail_vector; }
  
private:
  void init();
public:
  inline static LBDatabase * Object() { return CkpvAccess(lbdatabaseInited)?(LBDatabase *)CkLocalBranch(_lbdb):NULL; }
#if CMK_LBDB_ON
  inline LBDB *getLBDB() {return (LBDB*)(myLDHandle.handle);}
#endif

  void pup(PUP::er& p);

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
  inline void Send(const LDOMHandle &_om, const LDObjid _id, unsigned int _b, int _p) {
    LDSend(_om, _id, _b, _p);
  };

  inline void EstObjLoad(LDObjHandle h, double load) { LDEstObjLoad(h,load); };
  inline void NonMigratable(LDObjHandle h) { LDNonMigratable(h); };
  inline void Migratable(LDObjHandle h) { LDMigratable(h); };
  inline void UseAsyncMigrate(LDObjHandle h, CmiBool flag) { LDAsyncMigrate(h, flag); };
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

public:
  inline void StartLB() { LDStartLB(myLDHandle); }
  inline void TurnManualLBOn() { LDTurnManualLBOn(myLDHandle); }
  inline void TurnManualLBOff() { LDTurnManualLBOff(myLDHandle); }
 
  inline void PredictorOn(LBPredictorFunction *model) { LDTurnPredictorOn(myLDHandle,model); }
  inline void PredictorOn(LBPredictorFunction *model,int wind) { LDTurnPredictorOnWin(myLDHandle,model,wind); }
  inline void PredictorOff() { LDTurnPredictorOff(myLDHandle); }
  inline void ChangePredictor(LBPredictorFunction *model) { LDTurnPredictorOn(myLDHandle,model); }

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
  inline int Migrate(LDObjHandle h, int dest) { return LDMigrate(h,dest); };

  inline void Migrated(LDObjHandle h, int waitBarrier=1) { LDMigrated(h, waitBarrier); };

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
  int mystep;
  LDHandle myLDHandle;
  char *avail_vector;			// processor bit vector
  int new_ld_balancer;		// for Node 0
  CkVec<BaseLB *>   loadbalancers;
  int nloadbalancers;

public:
  static int manualOn;

public:
  char *availVector() { return avail_vector; }
  void get_avail_vector(char * bitmap);
  void set_avail_vector(char * bitmap, int new_ld=-1);
  int & new_lbbalancer() { return new_ld_balancer; }

  struct LastLBInfo {
    double *expectedLoad;
    LastLBInfo() { expectedLoad=new double[CkNumPes()]; 
		   for (int i=0; i<CkNumPes(); i++) expectedLoad[i]=0.0;}
  };
  LastLBInfo lastLBInfo;
  inline double myExpectedLoad() { return lastLBInfo.expectedLoad[CkMyPe()]; }
  inline double* expectedLoad() { return lastLBInfo.expectedLoad; }
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

void LBTurnInstrumentOn();
void LBTurnInstrumentOff();

inline LBDatabase* LBDatabaseObj() { return LBDatabase::Object(); }

inline void get_avail_vector(char * bitmap) {
  LBDatabaseObj()->get_avail_vector(bitmap);
}

inline void set_avail_vector(char * bitmap) {
  LBDatabaseObj()->set_avail_vector(bitmap);
}

#endif /* LDATABASE_H */

/*@}*/
