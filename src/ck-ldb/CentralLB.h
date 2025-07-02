/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef CENTRALLB_H
#define CENTRALLB_H

#include "BaseLB.h"
#include "CentralLB.decl.h"

#include <vector>
#include "pup_stl.h"
#include "manager.h"
extern CkGroupID loadbalancer;

void CreateCentralLB();

class CLBStatsMsg;
class LBSimulation;

/// for backward compatibility
typedef LBMigrateMsg  CLBMigrateMsg;

class LBInfo
{
public:
  LBRealType *peLoads; 	// total load: object + background
  LBRealType *objLoads; 	// total obj load
  LBRealType *comLoads; 	// total comm load
  LBRealType *bgLoads; 	// background load
  int    numPes;
  int    msgCount;	// total non-local communication
  CmiUInt8  msgBytes;	// total non-local communication
  LBRealType minObjLoad, maxObjLoad;
  LBInfo(): peLoads(NULL), objLoads(NULL), comLoads(NULL), 
            bgLoads(NULL), numPes(0), msgCount(0),
            msgBytes(0), minObjLoad(0.0), maxObjLoad(0.0) {}
  LBInfo(LBRealType *pl, int count): peLoads(pl), objLoads(NULL), 
            comLoads(NULL), bgLoads(NULL), numPes(count), msgCount(0),
            msgBytes(0), minObjLoad(0.0), maxObjLoad(0.0) {}
  LBInfo(int count);
  ~LBInfo();
  void getInfo(BaseLB::LDStats* stats, int count, int considerComm);
  void clear();
  void print();
  void getSummary(LBRealType &maxLoad, LBRealType &maxCpuLoad, LBRealType &totalLoad);
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

class CentralLB : public CBase_CentralLB
{
private:
  CLBStatsMsg *statsMsg;
  int count_msgs;
  void initLB(const CkLBOptions &);
public:
  CkMarshalledCLBStatsMessage bufMsg;
  SpanningTree st;
  CentralLB(const CkLBOptions & opt) : CBase_CentralLB(opt), concurrent(false) { initLB(opt);
#if CMK_SHRINK_EXPAND
		manager_init();
#endif
  }
  CentralLB(CkMigrateMessage *m) : CBase_CentralLB(m), concurrent(false) {
#if CMK_SHRINK_EXPAND
		manager_init();
#endif
  }
#if defined(TEMP_LDB) 
	float getTemp(int);
	  FILE* logFD;
	int physicalCoresPerNode;
	int logicalCoresPerNode,numSockets;
	int logicalCoresPerChip;
#endif

  virtual ~CentralLB();

  void pup(PUP::er &p);

  void SetPESpeed(int);
  int GetPESpeed();
  inline void setConcurrent(bool c) { concurrent = c; }

  void InvokeLB(); // Everything is at the PE barrier
  void ProcessAtSync(void); // Receive a message from AtSync to avoid
                            // making projections output look funny
  void SendStats();
  void ReceiveCounts(int *counts, int n);
  void ReceiveStats(CkMarshalledCLBStatsMessage &&msg);	// Receive stats on PE 0
  void ReceiveStatsViaTree(CkMarshalledCLBStatsMessage &&msg); // Receive stats using a tree structure  
  void ReceiveStatsFromRoot(CkMarshalledCLBStatsMessage &&msg);
  
  void depositData(CLBStatsMsg *m);
  void LoadBalance(void); 
  void t_LoadBalance(void); 
  void ApplyDecision(void);
  void ResumeClients(int);                      // Resuming clients needs

  void ResumeClients(); // to be resumed via message
  void InitiateScatter(LBMigrateMsg *msg);
  void ScatterMigrationResults(LBScatterMsg *);
  void ReceiveMigration(LBScatterMsg *);
  void ReceiveMigration(LBMigrateMsg *); 	// Receive migration data
  void ProcessMigrationDecision();
  void ProcessReceiveMigration();
  void MissMigrate(int waitForBarrier);

  //Shrink-Expand related functions
  void CheckForRealloc ();
  void ResumeFromReallocCheckpoint();
  void MigrationDoneImpl (int );
  void WillIbekilled(std::vector<char> avail, int);
  void StartCleanup();

  // manual start load balancing
  inline void StartLB() { thisProxy.ProcessAtSync(); }

  // Migrated-element callback
  void Migrated(int waitBarrier=1);

  void MigrationDone(int balancing);  // Call when migration is complete
  void CheckMigrationComplete();      // Call when all migration is complete

  // IMPLEMENTATION FOR FUTURE PREDICTOR
  void FuturePredictor(LDStats* stats);

  struct FutureModel {
    int cur_stats;   // number of statistics currently present
    int start_stats; // next stat to be written
    std::vector<LDStats> collection;
    LBPredictorFunction *predictor;
    std::vector<std::vector<double>> parameters;
    std::vector<bool> model_valid;

    FutureModel() : FutureModel(0, new DefaultFunction()) {}

    FutureModel(int n) : FutureModel(n, new DefaultFunction()) {}

    FutureModel(int n, LBPredictorFunction* myfunc)
        : cur_stats(0), start_stats(0), collection(n), predictor(myfunc)
    {
    }

    ~FutureModel() {
      delete predictor;
    }

    void changePredictor(LBPredictorFunction* new_predictor)
    {
      // gain control of the provided predictor
      delete predictor;
      predictor = new_predictor;

      for (auto& param_list : parameters)
      {
        param_list.resize(new_predictor->num_params);
      }
      model_valid.assign(model_valid.size(), false);
    }
  };

  // create new predictor, if one already existing, delete it first
  // if "pred" == 0 then the default function is used
  void predictorOn(LBPredictorFunction *pred) {
    predictorOn(pred, _lb_predict_window);
  }
  void predictorOn(LBPredictorFunction *pred, int window_size) {
    if (predicted_model) {
      PredictorPrintf("Predictor already allocated");
    } else {
      _lb_predict_window = window_size;
      if (pred) predicted_model = new FutureModel(window_size, pred);
      else predicted_model = new FutureModel(window_size);
      _lb_predict = true;
    }
    PredictorPrintf("Predictor turned on, window size %d\n",window_size);
  }

  // deallocate the predictor
  void predictorOff() {
    if (predicted_model) delete predicted_model;
    predicted_model = 0;
    _lb_predict = false;
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
    return Strategy(stats);
  };

  int cur_ld_balancer;

  void readStatsMsgs(const char* filename);
  void writeStatsMsgs(const char* filename);

  void removeCommDataOfDeletedObjs(LDStats* stats);
  void preprocess(LDStats* stats);
  virtual LBMigrateMsg* Strategy(LDStats* stats);
  virtual void work(LDStats* stats);
	virtual void changeFreq(int n);
  virtual LBMigrateMsg * createMigrateMsg(LDStats* stats);
  virtual LBMigrateMsg * extractMigrateMsg(LBMigrateMsg *m, int p);

  // Not to be used -- maintained for legacy applications
  virtual LBMigrateMsg* Strategy(LDStats* stats, int nprocs) {
    return Strategy(stats);
  }

protected:
  virtual bool QueryBalanceNow(int) { return true; };  
  virtual bool QueryDumpData() { return false; };  
  virtual void LoadbalanceDone(int balancing) {}

  void simulationRead();
  void simulationWrite();
  void findSimResults(LDStats* stats, int count, 
                      LBMigrateMsg* msg, LBSimulation* simResults);
  void removeNonMigratable(LDStats* statsDataList, int count);
  void loadbalance_with_thread() { use_thread = true; }

  bool concurrent;
private:  
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
  double strat_start_time;
  LBMigrateMsg   *storedMigrateMsg;
  LBScatterMsg   *storedScatterMsg;
  bool  reduction_started;
  bool  use_thread;

  FutureModel *predicted_model;

  void BuildStatsMsg();
  void buildStats();
  void printStrategyStats(LBMigrateMsg *msg);

public:
  int useMem();
};


// CLBStatsMsg is not directly sent in the entry function
// CkMarshalledCLBStatsMessage is used instead to use the pup defined here.
//class CLBStatsMsg: public CMessage_CLBStatsMsg {
class CLBStatsMsg {
public:
#if defined(TEMP_LDB)
	float pe_temp;
#endif

  int from_pe;
  int pe_speed;
  LBRealType total_walltime;
  LBRealType idletime;
  LBRealType bg_walltime;
#if CMK_LB_CPUTIMER
  LBRealType total_cputime;
  LBRealType bg_cputime;
#endif
  std::vector<LDObjData> objData;
  std::vector<LDCommData> commData;

  std::vector<char> avail_vector;
  int next_lb;

public:
  CLBStatsMsg(int osz, int csz);
  CLBStatsMsg(): from_pe(0), pe_speed(0), total_walltime(0.0), idletime(0.0),
		 bg_walltime(0.0),
#if defined(TEMP_LDB)
		pe_temp(1.0),
#endif

#if CMK_LB_CPUTIMER
		 total_cputime(0.0), bg_cputime(0.0),
#endif
		 next_lb(0) {}
  ~CLBStatsMsg();
  void pup(PUP::er &p);
}; 


// compute load distribution info
void getLoadInfo(BaseLB::LDStats* stats, int count, LBInfo &info, int considerComm);

#endif /* CENTRALLB_H */

/*@}*/


