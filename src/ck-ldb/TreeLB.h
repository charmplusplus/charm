#ifndef TREELB_H
#define TREELB_H

#include "BaseLB.h"
#include "TreeLB.decl.h"
#include <vector>
#include "json.hpp"
using json = nlohmann::json;


#define DEBUG__TREE_LB_L1 0
#define DEBUG__TREE_LB_L2 0
#define DEBUG__TREE_LB_L3 0


void CreateTreeLB();


/**
  * Base class for messages defined/used by LevelLogic classes
  * IMPORTANT: The derived msg class T must first inherit from this, and then from CMessage_T,
  * i.e. class T : public TreeLBMessage, public CMessage_T { ... }
  */
class TreeLBMessage
{
public:
  uint8_t level;
  // WARNING: don't add any virtual methods here
};


class LevelLogic
{
public:

  virtual ~LevelLogic() {}

  /// return msg with lb stats for this PE. only needed at leaves
  virtual TreeLBMessage *getStats() {
    CkAbort("LevelLogic::getStats not implemented\n");
  }
  // FIXME maybe these should be "=0" methods, but then the subclass would
  // have to implement empty methods if it doesn't need them

  /// deposit stats msg received from a child
  virtual void depositStats(TreeLBMessage *stats) {
    stats_msgs.push_back(stats);
  }

  /// return number of received stats msgs (in this lb step)
  virtual size_t numStatsReceived() {
    return stats_msgs.size();
  }

  /**
    * returns true if we should stop sending stats up and start load balancing
    * at this level; false otherwise
    */
  virtual bool cutoff() {
    return false;
  }

  /**
    * merge stats msgs received from children and return a new msg (to send up
    * the tree). the logic decides how to merge. normally, if the object is
    * just a comm-logic object joining levels, it will only "concatenate" the
    * information. if the object represents the logic for a tree level, it will
    * aggregate the stats in some fashion
    * IMPORTANT: logic must delete old messages when done with them
    */
  virtual TreeLBMessage *mergeStats() {
    CkAbort("LevelLogic::mergeStats not implemented\n");
  }

  /**
    * make some load balancing decision for my subtree and return list of
    * decision msgs. the first msg is for my PE if I'm a member of the next
    * tree level. the rest are for the children in the comm-tree that
    * communicates with the next level.
    * Note that the logic decides the content of decision msgs: they could all
    * be the same message (broadcast), or it could be a scatter, etc.
    */
  virtual void loadBalance(std::vector<TreeLBMessage*> &decisions, IDM &idm) {
    CkAbort("LevelLogic::loadBalance not implemented\n");
  }

  /**
    * only needed at comm-logic objects.
    * splits a decision msg into multiple msgs. each is for a child in the
    * comm-tree that communicates with the next level
    * IMPORTANT: logic must delete decision if no longer needed
    */
  virtual void splitDecision(TreeLBMessage *decision, std::vector<TreeLBMessage*> &decisions) {
    CkAbort("LevelLogic::splitDecision not implemented\n");
  }

  /**
    * process decision msg. must always return two integers representing amount of
    * expected incoming and outgoing load at this level.
    * Note that a positive load exchange always implies exchanging load with
    * other subtrees.
    * On level 0, if outgoing > 0, this method must start migration of any
    * chares specified in the decision msg.
    * On other levels, if outgoing > 0, do nothing. Tokens will be sent on a
    * separate code path (see below)
    */
  virtual void processDecision(TreeLBMessage *decision, int &incoming, int &outgoing) {
    CkAbort("LevelLogic::processDecision not implemented\n");
  }

  virtual bool makesTokens() {
    return false;
  }

  /// return nominal load that is being transferred in the tokens
  virtual int getTokenSets(TreeLBMessage *transferMsg,
                           std::vector<TreeLBMessage*> &token_sets,
                           std::vector<int> &destinations) {
    CkAbort("LevelLogic::getTokenSets not implemented\n");
  }

  /// IMPORTANT: logic must delete msg when done
  virtual int tokensReceived(TreeLBMessage *token_set) {
    CkAbort("LevelLogic::tokensReceived not implemented\n");
  }

  virtual int migrateObjects(const std::vector<std::pair<int,int>> &mig_order) {
    CkAbort("LevelLogic::migrateObjects not implemented\n");
  }

protected:

  std::vector<TreeLBMessage*> stats_msgs;

};


class LBTreeBuilder
{
public:

  /// return number of levels in tree
  virtual uint8_t build(std::vector<LevelLogic*> &logic, std::vector<int> &comm_parent,
                        std::vector<std::vector<int>> &comm_children,
                        std::vector<LevelLogic*> &comm_logic, json &config) = 0;

  virtual ~LBTreeBuilder() {}

};

// TODO I think the staticXXX functions are ugly. All load balancers should probably
// just inherit from a common interface, and the load balancers should just register
// themselves with LBManager instead of registering these functions
class TreeLB : public CBase_TreeLB
{
public:

  TreeLB(const CkLBOptions &opts) : CBase_TreeLB(opts) { init(opts); }
  TreeLB(CkMigrateMessage *m) : CBase_TreeLB(m) {}
  virtual ~TreeLB();

  /// these can be called multiple times to re-configure
  void configure(LBTreeBuilder &builder, json &config);
  void configure(json &config);

  // start load balancing (non-AtSync mode)  NOTE: This seems to do a broadcast
  // (is this the behavior we want?)
  inline void StartLB() { thisProxy.ProcessAtSync(); }
  static void staticStartLB(void *);

  // TODO: I would rename this group of functions (to maybe something like startLBLocal)
  // since they are also used in non-AtSync mode
  static void staticAtSync(void *);
  virtual void InvokeLB();  // Start load balancing at this PE
  void Migrated(int waitBarrier=1);
  void ProcessAtSync();  // Receive a message from AtSync to avoid making projections output look funny
                         // TODO: do we still need this?

  // send stats up using the comm-tree for this level
  void sendStatsUp(CkMessage *stats);

  void sendDecisionDown(CkMessage *decision);

  void recvLoadTokens(CkMessage *tokens);

  void multicastIDM(const IDM &idm, int num_pes, int *_pes);

  // called by LBManager when an actual chare migrates into this PE.
  // only happens in last level of tree
  static void staticObjMovedIn(void *me, LDObjHandle h, bool waitBarrier=true);
  void objMovedIn(bool waitBarrier=true);

  void resumeClients();

  void reportLbTime(double *times, int n);

private:

  void init(const CkLBOptions &opts);

  // receive load stats from lower level
  void receiveStats(TreeLBMessage *stats, int level);

  void loadBalanceSubtree(int level);

  // receive lb decision from parent (decision could be empty -do nothing-)
  // a non-empty decision implies load is moved from one subtree to another subtree
  void receiveDecision(TreeLBMessage *decision, int level);

  void transferLoadToken(TreeLBMessage *transferMsg, int level);

  void migrateObjects(const IDM &idm);

  // load can be actual objects or tokens
  inline bool checkLoadSent(int level) {
    if (load_sent[level] == expected_outgoing[level]) {
      load_sent[level] = expected_outgoing[level] = 0;
      return true;
    } else return false;
  }

  // load can be actual objects or tokens
  inline bool checkLoadReceived(int level) {
    if (load_received[level] == expected_incoming[level]) {
      load_received[level] = expected_incoming[level] = 0;
      return true;
    } else return false;
  }

  inline void checkLoadExchanged(int level) {
    if (checkLoadSent(level) && checkLoadReceived(level))
      loadBalanceSubtree(level);
  }

  void lb_done();

  uint8_t numLevels = 0;  // total number of tree levels (this chare won't necessarily participate in all levels)
  std::vector<LevelLogic*> logic;  // level -> my logic object at this level
  std::vector<int> comm_parent;    // level -> my parent PE in comm-tree connecting level to level+1
  std::vector<std::vector<int>> comm_children;  // level -> my children PEs in comm-tree connecting level to level+1
  std::vector<LevelLogic*> comm_logic;          // level -> comm logic object for this level

  std::vector<int> expected_incoming;
  std::vector<int> expected_outgoing;
  std::vector<int> load_sent;
  std::vector<int> load_received;
  //std::vector<int> notify_after_transfer;
  std::vector<bool> awaitingLB;

  double startTime;
  uint8_t mcast_bfactor = 4;

  int step_report_lb_times = -1;
  std::vector<double> lb_times;  // lb time of this PE for each step
  // a barrier before/after lb helps to obtain consistent load balancing times between PEs
  bool barrier_before_lb = false;
  bool barrier_after_lb = false;
};


#endif  /* TREELB_H */
