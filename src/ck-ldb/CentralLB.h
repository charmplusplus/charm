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

#ifndef CENTRALLB_H
#define CENTRALLB_H

#include <math.h>
#include "BaseLB.h"
#include "CentralLB.decl.h"

extern CkGroupID loadbalancer;

#define PER_MESSAGE_SEND_OVERHEAD   35e-6
#define PER_BYTE_SEND_OVERHEAD      8.5e-9
#define PER_MESSAGE_RECV_OVERHEAD   0.0
#define PER_BYTE_RECV_OVERHEAD      0.0

void CreateCentralLB();

class CLBStatsMsg;
class LBSimulation;

/// for backward compatibility
typedef LBMigrateMsg  CLBMigrateMsg;

class CentralLB : public BaseLB
{
private:
  void initLB(const CkLBOptions &);
public:
  CentralLB(const CkLBOptions & opt):BaseLB(opt) { initLB(opt); } 
  CentralLB(CkMigrateMessage *m):BaseLB(m) {}
  virtual ~CentralLB();

  void pup(PUP::er &p);

  void turnOn();
  void turnOff();
  inline int step() { return theLbdb->step(); }

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier
  void ProcessAtSync(void); // Receive a message from AtSync to avoid
                            // making projections output look funny

  void ReceiveStats(CkMarshalledCLBStatsMessage &msg);	// Receive stats on PE 0
  void ResumeClients(int);                     // Resuming clients needs
	                                        // to be resumed via message
  void ReceiveMigration(LBMigrateMsg *); 	// Receive migration data

  // manual predictor start/stop
  static void staticPredictorOn(void* data, void* model);
  static void staticPredictorOnWin(void* data, void* model, int wind);
  static void staticPredictorOff(void* data);
  static void staticChangePredictor(void* data, void* model);

  // manual start load balancing
  inline void StartLB() { thisProxy.ProcessAtSync(); }
  static void staticStartLB(void* data);

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h);
  void Migrated(LDObjHandle h);

  void MigrationDone(int balancing);  // Call when migration is complete

  struct ProcStats {  // per processor data
    double total_walltime;
    double total_cputime;
    double idletime;
    double bg_walltime;
    double bg_cputime;
    int pe_speed;
    double utilization;
    CmiBool available;
    int   n_objs;
    ProcStats(): total_walltime(0.0), total_cputime(0.0), idletime(0.0),
	   	 bg_walltime(0.0), bg_cputime(0.0), pe_speed(1),
		 utilization(1.0), available(CmiTrue), n_objs(0)  {}
    inline void pup(PUP::er &p) {
      p|total_walltime;  p|total_cputime; p|idletime;
      p|bg_walltime; p|bg_cputime; p|pe_speed;
      p|utilization; p|available; p|n_objs;
    }
  };

  struct LDStats {  // Passed to Strategy
    struct ProcStats  *procs;
    int count;
    
    int   n_objs;
    int   n_migrateobjs;
    LDObjData* objData;
    int   n_comm;
    LDCommData* commData;
    int  *from_proc, *to_proc;

    int *objHash; 
    int  hashSize;

    LDStats(): n_objs(0), n_migrateobjs(0), n_comm(0), 
               objData(NULL), commData(NULL), from_proc(NULL), to_proc(NULL), 
               objHash(NULL) {  procs = new ProcStats[CkNumPes()]; }
    void assign(int oid, int pe) { CmiAssert(procs[pe].available); to_proc[oid] = pe; }
      // build hash table
    void makeCommHash();
    void deleteCommHash();
    int getHash(const LDObjKey &);
    int getHash(const LDObjid &oid, const LDOMid &mid);
    void clear() {
      n_objs = n_comm = 0;
      delete [] objData;
      delete [] commData;
      delete [] from_proc;
      delete [] to_proc;
      deleteCommHash();
    }
    double computeAverageLoad();
    void pup(PUP::er &p);
    int useMem();
  };

  // IMPLEMENTATION FOR FUTURE PREDICTOR
  void FuturePredictor(LDStats* stats);

/*
  // class which implement a virtual function for the FuturePredictor
  class LBPredictorFunction {
  public:
    int num_params;

    void initialize_params(double *x) {double normall=1.0/pow(2,31); x[0]=rand()*normall; x[1]=rand()*normall; x[2]=rand()*normall; x[3]=rand()*normall; x[4]=rand()*normall; x[5]=rand()*normall;}

    virtual double predict(double x, double *params) =0;
    virtual void print(double *params) {CkPrintf("LB: unknown model\n");};
    virtual void function(double x, double *param, double &y, double *dyda) =0;
    virtual LBPredictorFunction* constructor() =0;
  };

  class DefaultFunction : public LBPredictorFunction {
  public:
    // constructor
    DefaultFunction() {num_params=6;};

    DefaultFunction* constructor() {return new DefaultFunction();}

    // compute the prediction function for the variable x with parameters param
    double predict(double x, double *param) {return (param[0] + param[1]*x + param[2]*x*x + param[3]*sin(param[4]*(x+param[5])));}

    void print(double *param) {CkPrintf("LB: %f + %fx + %fx^2 + %fsin%f(x+%f)\n",param[0],param[1],param[2],param[3],param[4],param[5]);}

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
*/

  struct FutureModel {
    int n_stats;    // total number of statistics allocated
    int cur_stats;   // number of statistics currently present
    int start_stats; // next stat to be written
    LDStats *collection;
    int n_objs;     // each object has its own parameters
    LBPredictorFunction *predictor;
    double **parameters;
    bool *model_valid;

    FutureModel(): n_stats(0), cur_stats(0), start_stats(0), collection(NULL),
	 n_objs(0), parameters(NULL) {predictor = new DefaultFunction();}

    FutureModel(int n): n_stats(n), cur_stats(0), start_stats(0), n_objs(0),
	 parameters(NULL) {
      collection = new LDStats[n];
      for (int i=0;i<n;++i) collection[i].objData=NULL;
      predictor = new DefaultFunction();
    }

    FutureModel(int n, LBPredictorFunction *myfunc): n_stats(n), cur_stats(0), start_stats(0), n_objs(0), parameters(NULL) {
      collection = new LDStats[n];
      for (int i=0;i<n;++i) collection[i].objData=NULL;
      predictor = myfunc;
    }

    ~FutureModel() {
      delete[] collection;
      for (int i=0;i<n_objs;++i) delete[] parameters[i];
      delete[] parameters;
      delete predictor;
    }

    void changePredictor(LBPredictorFunction *new_predictor) {
      delete predictor;
      // gain control of the provided predictor;
      predictor = new_predictor;
      for (int i=0;i<n_objs;++i) delete[] parameters[i];
      for (int i=0;i<n_objs;++i) {
	parameters[i] = new double[new_predictor->num_params];
	model_valid = false;
      }
    }
  };

  // create new predictor, if one already existing, delete it first
  // if "pred" == 0 then the default function is used
  void predictorOn(LBPredictorFunction *pred) {
    predictorOn(pred, _lb_predict_window);
  }
  void predictorOn(LBPredictorFunction *pred, int window_size) {
    if (predicted_model) PredictorPrintf("Predictor already allocated");
    else {
      _lb_predict_window = window_size;
      if (pred) predicted_model = new FutureModel(window_size, pred);
      else predicted_model = new FutureModel(window_size);
      _lb_predict = CmiTrue;
    }
    PredictorPrintf("Predictor turned on, window size %d\n",window_size);
  }

  // deallocate the predictor
  void predictorOff() {
    if (predicted_model) delete predicted_model;
    predicted_model = 0;
    _lb_predict = CmiFalse;
    PredictorPrintf("Predictor turned off\n");
  }

  // change the function of the predictor, at runtime
  // it will do nothing if it does not exist
  void changePredictor(LBPredictorFunction *new_predictor) {
    if (predicted_model) {
      predicted_model->changePredictor(new_predictor);
      PredictorPrintf("Predictor model changed\n");
    }
  }
  // END IMPLEMENTATION FOR FUTURE PREDICTOR

  LBMigrateMsg* callStrategy(LDStats* stats,int count){
    return Strategy(stats,count);
  };

  int cur_ld_balancer;

  void readStatsMsgs(const char* filename);
  void writeStatsMsgs(const char* filename);

  virtual LBMigrateMsg* Strategy(LDStats* stats,int count);
  virtual void work(LDStats* stats,int count);
  virtual LBMigrateMsg * createMigrateMsg(LDStats* stats,int count);
protected:
  virtual CmiBool QueryBalanceNow(int) { return CmiTrue; };  
  virtual CmiBool QueryDumpData() { return CmiFalse; };  

  void simulationRead();
  void simulationWrite();
  void findSimResults(LDStats* stats, int count, 
                      LBMigrateMsg* msg, LBSimulation* simResults);
//  void removeNonMigratable(LDStats* statsDataList, int count);

private:  
  CProxy_CentralLB thisProxy;
  int myspeed;
  int stats_msg_count;
  CLBStatsMsg **statsMsgsList;
  LDStats *statsData;
  int migrates_completed;
  int migrates_expected;
  double start_lb_time;

  FutureModel *predicted_model;

  void buildStats();

public:
  int useMem();
};

// CLBStatsMsg is not directly sent in the entry function
// CkMarshalledCLBStatsMessage is used instead to use the pup defined here.
//class CLBStatsMsg: public CMessage_CLBStatsMsg {
class CLBStatsMsg {
public:
  int from_pe;
  int serial;
  int pe_speed;
  double total_walltime;
  double total_cputime;
  double idletime;
  double bg_walltime;
  double bg_cputime;
  int n_objs;
  LDObjData *objData;
  int n_comm;
  LDCommData *commData;

  char * avail_vector;
  int next_lb;
public:
  CLBStatsMsg(int osz, int csz);
  CLBStatsMsg()  {}
  ~CLBStatsMsg();
  void pup(PUP::er &p);
}; 

#endif /* CENTRALLB_H */

/*@}*/


