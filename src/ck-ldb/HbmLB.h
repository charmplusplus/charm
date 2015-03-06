/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef HBMLB_H
#define HBMLB_H

#include "CentralLB.h"
#include "HbmLB.decl.h"

#include "topology.h"

void CreateHbmLB();

/// for backward compatibility
typedef LBMigrateMsg NLBMigrateMsg;

// base class
class MyHmbHierarchyTree {
public:
  MyHmbHierarchyTree() {}
  virtual ~MyHmbHierarchyTree() {}
  virtual int numLevels() = 0;
  virtual int parent(int mype, int level) = 0;
  virtual int isroot(int mype, int level) = 0;
  virtual int numChildren(int mype, int level) = 0;
  virtual void getChildren(int mype, int level, int *children, int &count) = 0;
};

// a simple 3 layer tree, fat at level 1
//        1
//     ---+---
//     0  4  8
//  ---+--
//  0 1 2 3
class HypercubeTree: public MyHmbHierarchyTree {
private:
  int toproot;
  int nLevels;
public:
  HypercubeTree() {
    nLevels = 0;
    int npes = CkNumPes();
    while ( npes != (1 << nLevels)) { nLevels++; }
    nLevels++;
    toproot = 0;
  }
  virtual ~HypercubeTree() {}
  virtual int numLevels() { return nLevels; }
  virtual int parent(int mype, int level) {
    if (level == nLevels-1) return -1;
    return (mype & ~(1<<level));
  }
  virtual int isroot(int mype, int level) {
    if (level == 0) return 0;
    return (mype & ((1<<level)-1)) == 0;
  }
  virtual int numChildren(int mype, int level) {
    return 2;
  }
  virtual void getChildren(int mype, int level, int *children, int &count) {
    CmiAssert(isroot(mype, level));
    count = numChildren(mype, level);
    children[0] = mype;
    children[1] = mype | (1<<(level-1));
  }
};

class HbmLB : public CBase_HbmLB
{
public:
  HbmLB(const CkLBOptions &);
  HbmLB(CkMigrateMessage *m):CBase_HbmLB(m) {}
  ~HbmLB();

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier
  void ProcessAtSync(void);

  void ReceiveStats(double t, int frompe, int fromlevel); 
  void ResumeClients(CkReductionMsg *msg);
  void ResumeClients(int balancing);
  void ReceiveMigrationCount(int, int lblevel);       // Receive migration count
  void ReceiveMigrationDelta(double t, int lblevel, int level);   // Receive migration amount

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h, int waitBarrier);
  void Migrated(LDObjHandle h, int waitBarrier);
  void ObjMigrated(LDObjData data, LDCommData *cdata, int n);
  void collectCommData(int objIdx, CkVec<LDCommData> &comms);

  void MigrationDone(int balancing);  // Call when migration is complete
  void NotifyObjectMigrationDone(int level, int lblevel);	
  virtual void Loadbalancing(int level);	// start load balancing
  void LoadbalancingDone(int level);	// start load balancing
  void ReceiveResumeClients(int fromlevel, int balancing);
  void reportLBQulity(double mload, double mCpuLoad, double totalload, int nmsgs, double bytesentry );

  struct MigrationRecord {
    LDObjHandle handle;
    int      fromPe;		// real from pe
    int      toPe;
    MigrationRecord(): fromPe(-1), toPe(-1) {}
    MigrationRecord(LDObjHandle &k, int f, int t): handle(k), fromPe(f), toPe(t) {}
    void pup(PUP::er &p) { p|handle; p|fromPe; p|toPe; }
  };

private:
  CProxy_HbmLB  thisProxy;
  int              foundNeighbors;
  LDStats myStats;

protected:
  virtual bool QueryBalanceNow(int) { return true; };  
  virtual bool QueryMigrateStep(int) { return true; };  
  //virtual LBMigrateMsg* Strategy(LDStats* stats);
  virtual void work(LDStats* stats);

  virtual int     useMem();
  int NeighborIndex(int pe, int atlevel);   // return the neighbor array index

  MyHmbHierarchyTree  *tree;

  class LevelData {
  public:
    int parent;
    int*  children;
    int nChildren;
    double statsList[2];		// bianry tree
    int stats_msg_count;
    LDStats *statsData;
    int obj_expected, obj_completed;
    int migrates_expected, migrates_completed;
    int mig_reported;		// for NotifyObjectMigrationDone
    int info_recved;		// for CollectInfo()
    int vector_expected, vector_completed;
    int resumeAfterMigration;
    CkVec<MigrationRecord> outObjs;
    CkVec<Location> unmatchedObjs;
    CkVec<Location> matchedObjs;	 // don't need to be sent up
  public:
    LevelData(): parent(-1), children(NULL), nChildren(0), stats_msg_count(0),
                 statsData(NULL), obj_expected(-1), obj_completed(0),
		 migrates_expected(-1), migrates_completed(0),
                 mig_reported(0), info_recved(0), 
		 vector_expected(-1), vector_completed(0),
		 resumeAfterMigration(0)
 		 { statsList[0] = statsList[1] = 0.0; }
    ~LevelData() {
      if (children) delete [] children;
      if (statsData) delete statsData;
    }
    int migrationDone() {
//CkPrintf("[%d] checking migrates_expected: %d migrates_completed: %d obj_completed: %d\n", CkMyPe(), migrates_expected, migrates_completed, obj_completed);
      return migrates_expected == 0 || migrates_completed == migrates_expected && obj_completed == migrates_expected;
    }
    int vectorReceived() {
      return vector_expected==0 || vector_expected == vector_completed;
    }
    void clear() {
      obj_expected = -1;
      obj_completed = 0;
      migrates_expected = -1;
      migrates_completed = 0;
      mig_reported = 0;
      info_recved = 0;
      vector_expected = -1;
      vector_completed = 0;
      resumeAfterMigration = 0;
      statsList[0] = statsList[1] = 0.0;
      if (statsData) statsData->clear();
      outObjs.free();
      matchedObjs.free();
      unmatchedObjs.free();
    }
    int useMem() {
      int memused = sizeof(LevelData);
      if (statsData) memused += statsData->useMem();
      memused += outObjs.size() * sizeof(MigrationRecord);
      memused += (unmatchedObjs.size()+matchedObjs.size()) * sizeof(Location);
      return memused;
    }
  };

  CkVec<LevelData *>  levelData;

  int currentLevel;
  int lbLevel;

private:
  void FindNeighbors();

  int migrate_expected;
  LBMigrateMsg** mig_msgs;
  int mig_msgs_received;
  int cur_ld_balancer;
  double start_lb_time;

  double maxLoad;
  double maxCpuLoad;		    // on level = 1
  double maxCommBytes;      // on level = 1
  int    maxCommCount;      // on level = 1
  double totalLoad;
  double maxMem;                    // on level = max - 1

  int vector_n_moves;

  CkVec<LDObjHandle> newObjs;
};

/*
class NLBStatsMsg {
public:
  int from_pe;
  int serial;
  int pe_speed;
  double total_walltime;
  double total_cputime;
  double idletime;
  double bg_walltime;
  double bg_cputime;
  double obj_walltime;   // may not needed
  double obj_cputime;   // may not needed
  int n_objs;
  LDObjData *objData;
  int n_comm;
  LDCommData *commData;
public:
  NLBStatsMsg(int osz, int csz);
  NLBStatsMsg(NLBStatsMsg *s);
  NLBStatsMsg()  {}
  ~NLBStatsMsg();
  void pup(PUP::er &p);
}; 
*/

#endif /* NBORBASELB_H */

/*@}*/
