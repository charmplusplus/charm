/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef HYBRIDBASELB_H
#define HYBRIDBASELB_H

#include <map>

#include "charm++.h"
#include "BaseLB.h"
#include "CentralLB.h"
#include "HybridBaseLB.decl.h"

#include "topology.h"

void CreateHybridBaseLB();

/// for backward compatibility
typedef LBMigrateMsg NLBMigrateMsg;

inline int mymin(int x, int y) { return x<y?x:y; }

// base class
class MyHierarchyTree {
protected:
  int *span;
  int nLevels;
  const char *myname;
public:
  MyHierarchyTree(): span(NULL), myname(NULL) {}
  virtual ~MyHierarchyTree() {}
  const char* name() const { return myname; }
  virtual int numLevels() const { return nLevels; }
  virtual int parent(int mype, int level) = 0;
  virtual int isroot(int mype, int level) = 0;
  virtual int numChildren(int mype, int level) = 0;
  virtual void getChildren(int mype, int level, int *children, int &count) = 0;
  virtual int numNodes(int level) {
    CmiAssert(level>=0 && level<nLevels);
    int count=1;
    for (int i=0; i<level; i++) count *= span[i];
    CmiAssert(CkNumPes()%count ==0);
    return CkNumPes()/count;
  }
};

// a simple 2 layer tree, fat at level 1
//        0
//     ---+---
//     0  1  2
class TwoLevelTree: public MyHierarchyTree {
private:
  int toproot;
public:
  TwoLevelTree() {
    myname = "TwoLevelTree";
    span = new int[1];
    nLevels = 2;
    span[0] = CkNumPes();
    toproot = 0;
  }
  virtual ~TwoLevelTree() { delete [] span; }
  virtual int parent(int mype, int level) {
    if (level == 0) return toproot;
    if (level == 1) return -1;
    CmiAssert(0);
    return -1;
  }
  virtual int isroot(int mype, int level) {
    if (level == 0) return 0;
    if (level == 1 && mype == toproot) return 1;
    return 0;
  }
  virtual int numChildren(int mype, int level) {
    if (level == 0) return 0;
    if (level == 1) return CkNumPes();
    CmiAssert(0);
    return 0;
  }
  virtual void getChildren(int mype, int level, int *children, int &count) {
    CmiAssert(isroot(mype, level));
    count = numChildren(mype, level);
    if (count == 0) { return; }
    if (level == 1) {
      for (int i=0; i<count; i++) 
        children[i] = i;
    }
  }
};

// a simple 3 layer tree, fat at level 1
//        1
//     ---+---
//     0  4  8
//  ---+--
//  0 1 2 3
class ThreeLevelTree: public MyHierarchyTree {
private:
  int toproot;
public:
  ThreeLevelTree(int groupsize=512) {
    myname = "ThreeLevelTree";
    span = new int[2];
    nLevels = 3;
    while (groupsize && CkNumPes() / groupsize < 2) {
      groupsize /= 2;
    }
    while ( CkNumPes() % groupsize ) --groupsize;
    if ( groupsize == 1 ) {
      ++groupsize;
      while ( CkNumPes() % groupsize ) ++groupsize;
    }
    span[0] = groupsize;
    CmiAssert(span[0]>1);
    span[1] = (CkNumPes()+span[0]-1)/span[0];
    if (CmiNumPhysicalNodes() > 1)
      toproot = CmiGetFirstPeOnPhysicalNode(1);
    else
      toproot = 1;
  }
  virtual ~ThreeLevelTree() { delete [] span; }
  virtual int parent(int mype, int level) {
    if (level == 0) return mype/span[0]*span[0];
    if (level == 1) return toproot;
    if (level == 2) return -1;
    CmiAssert(0);
    return -1;
  }
  virtual int isroot(int mype, int level) {
    if (level == 0) return 0;
    if (level == 1 && mype % span[0] == 0) return 1;
    if (level == 2 && mype == toproot) return 1;
    return 0;
  }
  virtual int numChildren(int mype, int level) {
    if (level == 0) return 0;
    if (level == 1) return mymin(CkNumPes(), mype+span[0]) - mype;
    if (level == 2) return span[1];
    CmiAssert(0);
    return 0;
  }
  virtual void getChildren(int mype, int level, int *children, int &count) {
    CmiAssert(isroot(mype, level));
    count = numChildren(mype, level);
    if (count == 0) { return; }
    if (level == 1) {
      for (int i=0; i<count; i++) 
        children[i] = mype + i;
    }
    if (level == 2) {
      for (int i=0; i<count; i++) 
        children[i] = i*span[0];
    }
  }
};

// a simple 4 layer tree, fat at level 1
//        1
//     ---+---
//     0  4  8
//  ---+--
//  0 
// -+-
// 0 1 2 3
class FourLevelTree: public MyHierarchyTree {
private:
  int toproot;
public:
  FourLevelTree() {
    myname = "FourLevelTree";
    span = new int[3];
    nLevels = 4;
#if 1
    span[0] = 64;
    span[1] = 32;
    span[2] = 32;
#else
    span[0] = 4;
    span[1] = 2;
    span[2] = 2;
#endif
    CmiAssert(CkNumPes() == span[0]*span[1]*span[2]);
    toproot = 2;
  }
  virtual ~FourLevelTree() { delete [] span; }
  virtual int parent(int mype, int level) {
    if (level == 0) return mype/span[0]*span[0];
    if (level == 1) return mype/span[0]/span[1]*span[0]*span[1]+1;
    if (level == 2) return toproot;
    if (level == 3) return -1;
    CmiAssert(0);
    return -1;
  }
  virtual int isroot(int mype, int level) {
    if (level == 0) return 0;
    if (level == 1 && (mype % span[0]) == 0) return 1;
    if (level == 2 && ((mype-1)%(span[0]*span[1])) == 0) return 1;
    if (level == 3 && mype == toproot) return 1;
    return 0;
  }
  virtual int numChildren(int mype, int level) {
    if (level == 0) return 0;
    if (level == 1) return span[0];
    if (level == 2) return span[1];
    if (level == 3) return span[2];
    CmiAssert(0);
    return 0;
  }
  virtual void getChildren(int mype, int level, int *children, int &count) {
    CmiAssert(isroot(mype, level));
    count = numChildren(mype, level);
    if (count == 0) { return; }
    if (level == 1) {
      for (int i=0; i<count; i++) 
        children[i] = mype + i;
    }
    if (level == 2) {
      for (int i=0; i<count; i++) 
        children[i] = mype-1+i*span[0];
    }
    if (level == 3) {
      for (int i=0; i<count; i++) 
        children[i] = i*span[0]*span[1]+1;
    }
  }
};

// a tree with same G branching factor at all levels
class KLevelTree: public MyHierarchyTree {
private:
  int toproot;
  int G;
public:
  KLevelTree(int k) {
    myname = "KLevelTree";
    nLevels = k;
    span = new int[nLevels-1];
    int P=CkNumPes();
    G = (int)(exp(log(P*1.0)/(nLevels-1))+0.5);
    if (pow(G*1.0, nLevels-1) != P) {
      CkPrintf("KLevelTree failed: P=%d Level=%d G=%d\n", P, nLevels, G);
      CmiAbort("KLevelTree failed");
    }
    for (int i=0; i<nLevels-1; i++) span[i] = G;
    if (CmiNumPhysicalNodes() > 1)
      toproot = CmiGetFirstPeOnPhysicalNode(1);
    else
      toproot = 1;
    if (CkMyPe()==0) CmiPrintf("KLevelTree: %d (toproot:%d).\n", G, toproot);
  }
  virtual ~KLevelTree() { delete [] span; }
  virtual int parent(int mype, int level) {
    if (level == nLevels-2) return toproot;
    if (level == nLevels-1) return -1;
    int S = 1;
    for (int i=0; i<=level; i++) S*=span[i];
    return mype/S*S+level;
  }
  virtual int isroot(int mype, int level) {
    if (level == 0) return 0;
    if (level == nLevels-1) return mype == toproot;
    int S = 1;
    for (int i=0; i<level; i++) S*=span[i];
    if ((mype - (level-1)) % S == 0) return 1;
    return 0;
  }
  virtual int numChildren(int mype, int level) {
    if (level == 0) return 0;
    return span[level-1];
  }
  virtual void getChildren(int mype, int level, int *children, int &count) {
    CmiAssert(isroot(mype, level));
    count = numChildren(mype, level);
    if (count == 0) { return; }
    int S = 1;
    for (int i=0; i<level-1; i++) S*=span[i];

    if (level == nLevels-1) {
      for (int i=0; i<count; i++)
        children[i] = i*S+(level-2);
    }
    else if (level == 1) {
      for (int i=0; i<count; i++)
        children[i] = mype+i*S;
    }
    else {
      for (int i=0; i<count; i++)
        children[i] = mype-1+i*S;
    }
  }
};

class HybridBaseLB : public CBase_HybridBaseLB
{
public:
  HybridBaseLB(const CkLBOptions &);
  HybridBaseLB(CkMigrateMessage *m): CBase_HybridBaseLB(m) {}
  ~HybridBaseLB();

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier
  void ProcessAtSync(void);

  void ReceiveStats(CkMarshalledCLBStatsMessage &m, int fromlevel); 
  void ResumeClients(CkReductionMsg *msg);
  void ResumeClients(int balancing);
  void ReceiveMigration(LBMigrateMsg *); 	// Receive migration data
  void ReceiveVectorMigration(LBVectorMigrateMsg *); // Receive migration data
  virtual void GetObjsToMigrate(int toPe, double load, LDStats *stats,
      int atlevel, CkVec<LDCommData>& comms, CkVec<LDObjData>& objs);
  void CreateMigrationOutObjs(int atlevel, LDStats* stats, int objidx);
  void TotalObjMigrated(int count, int level);

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h, int waitBarrier);
  void Migrated(LDObjHandle h, int waitBarrier);

  void ObjMigrated(LDObjData data, LDCommData *cdata, int n, int level);
  void ObjsMigrated(CkVec<LDObjData>& data, int m, LDCommData *cdata, int n, int level);
  void VectorDone(int atlevel);
  void MigrationDone(int balancing);  // Call when migration is complete
  void StatsDone(int level);  // Call when LDStats migration is complete
  void NotifyObjectMigrationDone(int level);	
  virtual void Loadbalancing(int level);	// start load balancing
  void StartCollectInfo(DummyMsg *m);
  void CollectInfo(Location *loc, int n, int fromlevel);
  void PropagateInfo(Location *loc, int n, int fromlevel);

  void reportLBQulity(double mload, double mCpuLoad, double totalload, int nmsgs, double bytesentry );
  void reportLBMem(double);

  struct MigrationRecord {
    LDObjHandle handle;
    int      fromPe;		// real from pe
    int      toPe;
    MigrationRecord(): fromPe(-1), toPe(-1) {}
    MigrationRecord(LDObjHandle &k, int f, int t): handle(k), fromPe(f), toPe(t) {}
    void pup(PUP::er &p) { p|handle; p|fromPe; p|toPe; }
  };

private:
  CProxy_HybridBaseLB  thisProxy;
  int              foundNeighbors;
  CmiGroup            group1;              // level 1 multicast group
  int                 group1_created;

protected:
  virtual bool QueryBalanceNow(int) { return true; };  
  virtual bool QueryMigrateStep(int) { return true; };  
  virtual LBMigrateMsg* Strategy(LDStats* stats);
  virtual void work(LDStats* stats);
  virtual LBMigrateMsg * createMigrateMsg(LDStats* stats);
  // helper function
  LBMigrateMsg * createMigrateMsg(CkVec<MigrateInfo *> &migrateInfo, int count);
  virtual LBVectorMigrateMsg* VectorStrategy(LDStats* stats);
  void    printSummary(LDStats *stats, int count);
  void    initTree();
  void collectCommData(int objIdx, CkVec<LDCommData> &comm, int atlevel);

  // Not to be used -- maintained for legacy applications
  virtual LBMigrateMsg* Strategy(LDStats* stats, int nprocs) {
    return Strategy(stats);
  }

  virtual CLBStatsMsg* AssembleStats();
  virtual int     useMem();
  int NeighborIndex(int pe, int atlevel);   // return the neighbor array index

  MyHierarchyTree  *tree;
  int shrinklevel;

  class LevelData {
  public:
    int parent;
    int*  children;
    int nChildren;
    CLBStatsMsg **statsMsgsList;
    int stats_msg_count;
    LDStats *statsData;
    int obj_expected, obj_completed;
    int migrates_expected, migrates_completed;
    int mig_reported;		// for NotifyObjectMigrationDone
    int info_recved;		// for CollectInfo()
    int vector_expected, vector_completed;
    int resumeAfterMigration;
    CkVec<MigrationRecord> outObjs;
    //CkVec<Location> unmatchedObjs;
    std::map< LDObjKey, int >  unmatchedObjs;
    CkVec<Location> matchedObjs;	 // don't need to be sent up
  public:
    LevelData(): parent(-1), children(NULL), nChildren(0), 
                 statsMsgsList(NULL), stats_msg_count(0),
                 statsData(NULL), obj_expected(-1), obj_completed(0),
		 migrates_expected(-1), migrates_completed(0),
                 mig_reported(0), info_recved(0), 
		 vector_expected(-1), vector_completed(0),
		 resumeAfterMigration(0)
 		 {}
    ~LevelData() {
      if (children) delete [] children;
      if (statsMsgsList) delete [] statsMsgsList;
      if (statsData) delete statsData;
    }
    int migrationDone() {
//CkPrintf("[%d] checking migrates_expected: %d migrates_completed: %d obj_completed: %d\n", CkMyPe(), migrates_expected, migrates_completed, obj_completed);
      return migrates_expected == 0 || migrates_completed + obj_completed == migrates_expected;
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
      if (statsData) statsData->clear();
      outObjs.free();
      matchedObjs.free();
      unmatchedObjs.clear();
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

  enum StatsStrategy { FULL, SHRINK, SHRINK_NULL} ;
  StatsStrategy statsStrategy;

private:
  void FindNeighbors();
  void buildStats(int level);
  CLBStatsMsg * buildCombinedLBStatsMessage(int atlevel);
  void depositLBStatsMessage(CLBStatsMsg *msg, int atlevel);

  int future_migrates_expected;
  LBMigrateMsg** mig_msgs;
  int mig_msgs_received;
  int cur_ld_balancer;
  double start_lb_time;

  double maxLoad;		    // on level = 1
  double maxCpuLoad;		    // on level = 1
  double maxCommBytes;      // on level = 1
  int    maxCommCount;      // on level = 1
  double totalLoad;
  double maxMem;                    // on level = max - 1

  CkVec<Location> newObjs;

  int vector_n_moves;
};


#endif /* NBORBASELB_H */

/*@}*/
