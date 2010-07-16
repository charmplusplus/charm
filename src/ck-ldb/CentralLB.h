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

void CreateCentralLB();

class CLBStatsMsg;
class LBSimulation;

/// for backward compatibility
typedef LBMigrateMsg  CLBMigrateMsg;

class LBInfo
{
public:
  double *peLoads; 	// total load: object + background
  double *objLoads; 	// total obj load
  double *comLoads; 	// total comm load
  double *bgLoads; 	// background load
  int    numPes;
  int    msgCount;	// total non-local communication
  CmiUInt8  msgBytes;	// total non-local communication
  double minObjLoad, maxObjLoad;
  LBInfo(): peLoads(NULL), objLoads(NULL), comLoads(NULL), 
            bgLoads(NULL), numPes(0), msgCount(0),
            msgBytes(0), minObjLoad(0.0), maxObjLoad(0.0) {}
  LBInfo(double *pl, int count): peLoads(pl), objLoads(NULL), 
            comLoads(NULL), bgLoads(NULL), numPes(count), msgCount(0),
            msgBytes(0), minObjLoad(0.0), maxObjLoad(0.0) {}
  LBInfo(int count);
  ~LBInfo();
  void getInfo(BaseLB::LDStats* stats, int count, int considerComm);
  void clear();
  void print();
  void getSummary(double &maxLoad, double &maxCpuLoad, double &totalLoad);
};

/** added by Abhinav
 * class for computing the parent and children of a processor 
 */
class SpanningTree
{
	public:
		int arity;
		int parent;
		int numChildren;
		SpanningTree();
		void calcParent(int n);
		void calcNumChildren(int n);
};

class CentralLB : public BaseLB
{
private:
  CLBStatsMsg *statsMsg;
  int count_msgs;
  void initLB(const CkLBOptions &);
public:
  CkMarshalledCLBStatsMessage bufMsg;
  SpanningTree st;
  CentralLB(const CkLBOptions & opt):BaseLB(opt) { initLB(opt); 
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        lbDecisionCount= resumeCount=0;
#endif
} 
  CentralLB(CkMigrateMessage *m):BaseLB(m) {}
  virtual ~CentralLB();

  void pup(PUP::er &p);

  void turnOn();
  void turnOff();

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier
  void ProcessAtSync(void); // Receive a message from AtSync to avoid
                            // making projections output look funny

  void SendStats();
  void ReceiveCounts(CkReductionMsg *);
  void ReceiveStats(CkMarshalledCLBStatsMessage &msg);	// Receive stats on PE 0
  void ReceiveStatsViaTree(CkMarshalledCLBStatsMessage &msg); // Receive stats using a tree structure  
  
  void depositData(CLBStatsMsg *m);
  void LoadBalance(void); 
  void ResumeClients(int);                      // Resuming clients needs
	                                        // to be resumed via message
  void ResumeClients(CkReductionMsg *);
  void ReceiveMigration(LBMigrateMsg *); 	// Receive migration data
  void ProcessReceiveMigration(CkReductionMsg  *);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	void ReceiveDummyMigration(int _step);
#endif
  void MissMigrate(int waitForBarrier);

  // manual predictor start/stop
  static void staticPredictorOn(void* data, void* model);
  static void staticPredictorOnWin(void* data, void* model, int wind);
  static void staticPredictorOff(void* data);
  static void staticChangePredictor(void* data, void* model);

  // manual start load balancing
  inline void StartLB() { thisProxy.ProcessAtSync(); }
  static void staticStartLB(void* data);

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h, int waitBarrier=1);
  void Migrated(LDObjHandle h, int waitBarrier=1);

  void MigrationDone(int balancing);  // Call when migration is complete
  void CheckMigrationComplete();      // Call when all migration is complete

  // IMPLEMENTATION FOR FUTURE PREDICTOR
  void FuturePredictor(LDStats* stats);

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
      //for (int i=0;i<n;++i) collection[i].objData=NULL;
      predictor = new DefaultFunction();
    }

    FutureModel(int n, LBPredictorFunction *myfunc): n_stats(n), cur_stats(0), start_stats(0), n_objs(0), parameters(NULL) {
      collection = new LDStats[n];
      //for (int i=0;i<n;++i) collection[i].objData=NULL;
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
      int i;
      // gain control of the provided predictor;
      predictor = new_predictor;
      for (i=0;i<n_objs;++i) delete[] parameters[i];
      for (i=0;i<n_objs;++i) {
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

  void preprocess(LDStats* stats,int count);
  virtual LBMigrateMsg* Strategy(LDStats* stats,int count);
  virtual void work(LDStats* stats,int count);
  virtual LBMigrateMsg * createMigrateMsg(LDStats* stats,int count);
  virtual LBMigrateMsg * extractMigrateMsg(LBMigrateMsg *m, int p);
protected:
  virtual CmiBool QueryBalanceNow(int) { return CmiTrue; };  
  virtual CmiBool QueryDumpData() { return CmiFalse; };  
  virtual void LoadbalanceDone(int balancing) {}

  void simulationRead();
  void simulationWrite();
  void findSimResults(LDStats* stats, int count, 
                      LBMigrateMsg* msg, LBSimulation* simResults);
  void removeNonMigratable(LDStats* statsDataList, int count);

private:  
  CProxy_CentralLB thisProxy;
  int myspeed;
  int stats_msg_count;
  CLBStatsMsg **statsMsgsList;
  LDStats *statsData;
  int migrates_completed;
  int migrates_expected;
  int future_migrates_completed;
  int future_migrates_expected;
  int lbdone;
  double start_lb_time;
  LBMigrateMsg   *storedMigrateMsg;

  FutureModel *predicted_model;

  void BuildStatsMsg();
  void buildStats();

public:
  int useMem();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    int savedBalancing;
    void endMigrationDone(int balancing);
    int lbDecisionCount ,resumeCount;
#endif
};

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_)) 
    void resumeCentralLbAfterChkpt(void *lb);
#endif

// CLBStatsMsg is not directly sent in the entry function
// CkMarshalledCLBStatsMessage is used instead to use the pup defined here.
//class CLBStatsMsg: public CMessage_CLBStatsMsg {
class CLBStatsMsg {
public:
  int from_pe;
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
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	int step;
#endif

public:
  CLBStatsMsg(int osz, int csz);
  CLBStatsMsg(): from_pe(0), pe_speed(0), 
                 total_walltime(0.0), total_cputime(0.0),
                 idletime(0.0), bg_walltime(0.0), bg_cputime(0.0),
                 n_objs(0), objData(NULL), n_comm(0), commData(NULL), 
                 avail_vector(NULL), next_lb(0)  {}
  ~CLBStatsMsg();
  void pup(PUP::er &p);
}; 


// compute load distribution info
void getLoadInfo(BaseLB::LDStats* stats, int count, LBInfo &info, int considerComm);

#endif /* CENTRALLB_H */

/*@}*/


