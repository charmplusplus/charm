// Author: Juan Galvez <jjgalvez@illinois.edu>

#include "TreeBuilder.h"  // TODO this can be deleted if we change it so that LBManager instantiates the builders
#include "TreeLB.h"
#include "TreeStrategyFactory.h"
#include "spanningTree.h"
#include "ck.h"
#include <fstream>  // TODO delete if json file is read from LBManager
#include <sstream>
#include "json.hpp"

extern int quietModeRequested;
#if CMK_SHRINK_EXPAND
extern "C" void charmrun_realloc(char *s);
extern char willContinue;
extern realloc_state pending_realloc_state;
extern char * se_avail_vector;
extern char *_shrinkexpand_basedir;
extern int numProcessAfterRestart;
extern bool load_balancer_created;
#endif

static void lbinit()
{
  const auto& names = TreeStrategy::LBNames;
  std::ostringstream o;
  for (const auto& name : names)
  {
    o << "\n\t" << name;
  }
  LBRegisterBalancer<TreeLB>(
      "TreeLB", "Pluggable hierarchical LB with available strategies:" + o.str());
#if CMK_SHRINK_EXPAND
  load_balancer_created = true;
#endif
}

void TreeLB::Migrated(int waitBarrier)
{
  objMovedIn(waitBarrier);
}

void TreeLB::StartLB(){
  CkPrintf("TreeLB::StartLB called on PE %d\n", CkMyPe());
  if (logic[1]) {
    CkPrintf("size of stats_msgs = %d\n", logic[1]->stats_msgs.size());
  }

  int rateAware = false;
  LBStatsMsg_1* mm = (LBStatsMsg_1*)logic[1]->stats_msgs[0];
  if ((void*)mm->speeds != (void*)mm->obj_start) rateAware = true;

  if (logic[1]->getNumNewPes() == 0 || !rateAware) {
    CkPrintf("TreeLB::StartLB: no new PEs detected, starting load balancing\n");
    loadBalanceSubtree(numLevels - 1);
  }
  else {
    thisProxy.restartFromSE();
    CkPrintf("TreeLB::StartLB: new PEs detected, setting up speeds first\n");
  }
}

void TreeLB::loadConfigFile(const CkLBOptions& opts)
{
  config.clear();
  std::ifstream ifs(_lb_args.treeLBFile(), std::ifstream::in);
  if (opts.hasLegacyName())
  {
    // support legacy mode, e.g. map "+GreedyLB" to PE_Root tree using Greedy
    // use 2-level tree
    config["tree"] = "PE_Root";
    config["root"]["strategies"] = {opts.getLegacyName()};
    if (CkMyPe() == 0 && !quietModeRequested)
      CkPrintf("[%d] TreeLB in LEGACY MODE support\n", CkMyPe());
  }
  else if (ifs.good())
  {
    config = json::parse(ifs, nullptr, false);
    if (config.is_discarded())
    {
      CkPrintf(
          "Error reading TreeLB configuration file: %s.\n"
          "Ensure your configuration file is valid JSON.\n",
          _lb_args.treeLBFile());
      CkExit(1);
    }
  }
  else
  {
    if (CkMyPe() == 0 && !quietModeRequested)
      CkPrintf(
          "[%d] No TreeLB configuration file found. Choosing a default configuration.\n",
          CkMyPe());
    // try to pick reasonable defaults
    // the problem with using a 2 or 3 level tree in large jobs is that the root's
    // strategy could take a long time to run, and also many strategies (but not all)
    // could move objects across the whole machine (which means potentially slow
    // migrations and could also hurt communication performance depending on the
    // strategy).
    if (CmiNumPhysicalNodes() >= 128)
    {
      // use 4-level tree
      config["tree"] = "PE_Process_ProcessGroup_Root";
      config["root"]["pe"] = 0;
      config["processgroup"]["num_groups"] = CmiNumPhysicalNodes() / 32;
      config["root"]["step_freq"] = 10;
      config["processgroup"]["step_freq"] = 5;
      config["process"]["strategies"] = {"GreedyRefine"};
      config["processgroup"]["strategies"] = {"GreedyRefine"};
      config["processgroup"]["GreedyRefine"]["tolerance"] = 1.03;
    }
    else if (CmiNumNodes() > 1 && CmiNodeSize(0) > 1)
    {
      // use 3-level tree
      config["tree"] = "PE_Process_Root";
      config["root"]["pe"] = 0;
      config["root"]["step_freq"] = 3;
      config["root"]["strategies"] = {"GreedyRefine"};
      config["process"]["strategies"] = {"GreedyRefine"};
    }
    else
    {
      // use 2-level tree
      config["tree"] = "PE_Root";
      config["root"]["pe"] = 0;
      config["root"]["strategies"] = {"GreedyRefine"};
    }
  }
  ifs.close();
}

void TreeLB::init(const CkLBOptions& opts)
{
#if CMK_LBDB_ON

  lbname = "TreeLB";

  if (_lb_args.syncResume()) barrier_after_lb = true;

  // create and turn on by default
  startLbFnHdl = lbmgr->AddStartLBFn(this, &TreeLB::StartLB);

  configure(config);

  // TODO this functionality needs to move to LBManager
  if (_lb_args.statsOn())
    lbmgr->CollectStatsOn();  // collect load and (optionally) comm stats

  if (opts.getSeqNo() > 0)
  {
    turnOff();
  }

#endif
}

void TreeLB::collectSpeeds(int pe_id, float speed) {
  if (_lb_args.debug() > 2) CkPrintf("[PE %d] TreeLB::collectSpeeds from PE %d speed=%f\n", CkMyPe(), pe_id, speed);
  if (logic[1]->collectSpeeds(pe_id, speed))
    loadBalanceSubtree(numLevels - 1);
  else
    CkPrintf("[PE %d] TreeLB::collectSpeeds: still waiting for more speeds\n", CkMyPe());
}

void TreeLB::restartFromSE() {
  // TODO: need to collect and recompute bg load as well for the new pes

  if (CkMyPe() == 0) {
    // if there was just 1 pe initially, the speed isn't set, so recompute it here
    // TODO: ideally this should be rearranged so that the stats msgs are always set up correctly
    LBStatsMsg_1* msg = (LBStatsMsg_1*)logic[1]->stats_msgs[0];
    for (int i = 0; i < msg->nPes; i++) {
        if (msg->pe_ids[i] == 0 && msg->speeds[i] == 1.0  ) {
          msg->speeds[i] = lbmgr->ProcessorSpeed();
        }
      }
  }
  if (thisPeNew) {
    if (CkMyPe() == 0) CkAbort("[PE %d] Should never be new\n", CkMyPe());
    float speed = float(lbmgr->ProcessorSpeed());
    thisProxy[0].collectSpeeds(CkMyPe(), speed);
    thisPeNew = false;
  }
}

void TreeLB::expand_init()
{
  awaitingLB[0] = true;
  awaitingLB[1] = false;

  if (CkMyPe() == 0)
    awaitingLB[1] = true; // root level also needs to do LB

  if (CkNumPes() == 1)
    awaitingLB[0] = awaitingLB[1] = false; // no need for PE level if only 1 PE

  numLevels = 2;
}

TreeLB::~TreeLB()
{
#if CMK_LBDB_ON
  lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  if (lbmgr)
  {
    lbmgr->RemoveStartLBFn(startLbFnHdl);
  }

  for (auto l : logic) delete l;
  for (auto l : comm_logic) delete l;

#endif
}

void TreeLB::configure(LBTreeBuilder& builder, json& config)
{
#if CMK_LBDB_ON
  if (_lb_args.debug() > 0)
  if (numLevels > 0 && CkMyPe() == 0 && !quietModeRequested)
  {
    CkPrintf("[%d] Reconfiguring TreeLB\n", CkMyPe());
  }

  mcast_bfactor = builder.getProperty("mcast_bfactor", mcast_bfactor, config);
  step_report_lb_times =
      builder.getProperty("report_lb_times_at_step", step_report_lb_times, config);

  if (numLevels > 0)
  {
    for (auto l : logic) delete l;
    for (auto l : comm_logic) delete l;
  }
  numLevels = builder.build(logic, comm_parent, comm_children, comm_logic, config);
  CkAssert(numLevels > 0 && logic.size() == numLevels &&
           comm_parent.size() == numLevels && comm_children.size() == numLevels &&
           comm_logic.size() == numLevels);

  expected_incoming.resize(numLevels);
  expected_outgoing.resize(numLevels);
  load_sent.resize(numLevels);
  load_received.resize(numLevels);
  // notify_after_transfer.resize(numLevels);
  awaitingLB.resize(numLevels);

  // reset all values since this may be a re-configuration
  std::fill(expected_incoming.begin(), expected_incoming.end(), 0);
  std::fill(expected_outgoing.begin(), expected_outgoing.end(), 0);
  std::fill(load_sent.begin(), load_sent.end(), 0);
  std::fill(load_received.begin(), load_received.end(), 0);
  // std::fill(notify_after_transfer.begin(), notify_after_transfer.end(), -1);
  std::fill(awaitingLB.begin(), awaitingLB.end(), false);

#endif
}

void TreeLB::configure(json& config)
{
#if CMK_LBDB_ON
  const std::string& tree_type = config["tree"];
  if (tree_type == "PE_Root")
  {
    // builder = new PE_Root_Tree();
    auto tree = PE_Root_Tree(config);
    configure(tree, config);
  }
  else if (tree_type == "PE_Process_Root")
  {
    auto tree = PE_Node_Root_Tree(config);
    configure(tree, config);
  }
  else if (tree_type == "PE_Process_ProcessGroup_Root")
  {
    auto tree = PE_Node_NodeSet_Root_Tree(config);
    configure(tree, config);
  }
  else
  {
    CkAbort("TreeLB: configured tree not recognized\n");
  }
#endif
}

void TreeLB::pup(PUP::er& p)
{
  if (_lb_args.debug() > 2)
    CkPrintf("[%d] TreeLB::pup numLevels=%d\n", CkMyPe(), numLevels);

  p|seqno;
  
  if(p.isUnpacking()){
    loadConfigFile(CkLBOptions(seqno));
    init(CkLBOptions(seqno));
    manager_init();
  }

  assert(numLevels == 2); // rn this only supports the two level tree

  if (logic[1] == nullptr) { // TODO: delete this memory
    logic[1] = new RootLevel(); // this is needed because logic[1] is null on PE1, but PE1 still needs to participate in this... confusing?
  } 

  if (_lb_args.debug() > 2)
    CkPrintf("[%d] TreeLB::pupping logic things\n", CkMyPe());

  int oldPE;
  if (p.isPacking()) oldPE = CkMyPe();
  p|oldPE;
  if (p.isUnpacking()) {
    if (CkMyPe() != oldPE) {
      thisPeNew = true;
    }
  }

  p|*logic[0];
  p|*logic[1];  

  if (p.isUnpacking())
    expand_init();
}

void TreeLB::CallLB()
{
  #if CMK_LBDB_ON
  #if CMK_SHRINK_EXPAND
  
  // if (pending_realloc_state != NO_REALLOC) {
  //   // if (_lb_args.debug() > 0)
  //   //   CkPrintf("TreeLB::CallLB pending_realloc_state=%d (EXPAND_MSG_RECEIVED %d, NO_REALLOC %d)\n", pending_realloc_state, EXPAND_MSG_RECEIVED, NO_REALLOC);
  //   configure(config); // reconfigure tree in case number of PEs changed
  //   CkPrintf("Done reconfiguring tree\n");
  // }


  #endif    

  if (barrier_before_lb)
  {
    contribute(CkCallback(CkReductionTarget(TreeLB, ProcessAtSync), thisProxy));
  }
  else
  {
    thisProxy[CkMyPe()].ProcessAtSync();
  }
  #endif
}

void TreeLB::InvokeLB()
{
#if CMK_LBDB_ON
  // NOTE: I'm assuming new LBManager will know when (and when not to) call AtSync
  lbmgr->lb_in_progress = true;
  CallLB();
#endif
}

void TreeLB::ProcessAtSync()
{
#if CMK_LBDB_ON
  startTime = CkWallTimer();
  if (CkMyPe() == 0 && _lb_args.debug() > 0)
  {
    CkPrintf("--------- Started LB step %d ---------\n", lbmgr->step());
  }
  CkAssert(numLevels > 0 && !awaitingLB[0]);
  TreeLBMessage* stats = logic[0]->getStats();
  stats->level = 0;
  awaitingLB[0] = true;

  sendStatsUp((CkMessage*)stats);
#endif
}

void TreeLB::CheckForLB() {
#if CMK_SHRINK_EXPAND
//   // if (_lb_args.debug() > 0)
//   //   CkPrintf("TreeLB::CheckForLB pending_realloc_state=%d (EXPAND_MSG_RECEIVED %d, NO_REALLOC %d)\n", pending_realloc_state, EXPAND_MSG_RECEIVED, NO_REALLOC);

  if (pending_realloc_state == EXPAND_MSG_RECEIVED)
    checkForRealloc();
  //else if (pending_realloc_state == NO_REALLOC)
  //  thisProxy.resumeClients(0);
  else
    loadBalanceSubtree(numLevels - 1);
    //thisProxy.CallLB();
#else
  //thisProxy.CallLB();
  loadBalanceSubtree(numLevels - 1);
#endif
}

// send stats up using the comm-tree for this level
void TreeLB::sendStatsUp(CkMessage* msg)
{
  TreeLBMessage* stats = (TreeLBMessage*)msg;
  int level = stats->level;
  if (comm_parent.size() <= level || comm_children.size() <= level ||
      comm_logic.size() <= level)
  {
    CkAbort("TreeLB: sendStatsUp invalid level %d, or comm_parent not initialized\n", level);
  }

  int comm_parent_pe = comm_parent[level];

  // fprintf(stderr, "[%d] TreeLB::sendStatsUp - received msg level=%d comm_parent=%d\n",
  // CkMyPe(), level, comm_parent_pe);
  if (comm_parent_pe == -1)
  {
    // I'm the root of this comm-tree (current destination of the stats)
    receiveStats(stats, level);
  }
  else if (comm_children[level].size() == 0)
  {
    // don't have children so don't have to aggregate any msgs
    thisProxy[comm_parent_pe].sendStatsUp((CkMessage*)stats);
  }
  else
  {
    LevelLogic* logic = comm_logic[level];
    logic->depositStats(stats);
    if (logic->numStatsReceived() == comm_children[level].size() + 1)
    {
      TreeLBMessage* newMsg = logic->mergeStats();
      newMsg->level = level;
      thisProxy[comm_parent_pe].sendStatsUp((CkMessage*)newMsg);
    }
  }
}

void TreeLB::receiveStats(TreeLBMessage* stats, int level)
{
  level += 1;
  awaitingLB[level] = true;
  LevelLogic* l = logic[level];
  l->depositStats(stats);
  size_t expected_msgs = comm_children[level - 1].size();
  if (logic[level - 1] != nullptr) expected_msgs += 1;  // expect msg from myself too
  if (l->numStatsReceived() == expected_msgs)
  {
    CkAssert(load_sent[level] == 0 && load_received[level] == 0);
    if (level == numLevels - 1 || l->cutoff())
    {
      // cutoff can be adjusted dynamically, to prevent lb between upper-level domains.
      // can be used, for example, to only do within-node lb on some steps
      TreeLBMessage* newMsg = l->mergeStats();  // this is IN PLACE 
      
      #if CMK_SHRINK_EXPAND
        //contribute(CkCallback(CkReductionTarget(TreeLB, CheckForLB), thisProxy[0]));
        CheckForLB();
      #else
        //CallLB();
        loadBalanceSubtree(level);
      #endif
      //loadBalanceSubtree(level);
    }
    else
    {
      TreeLBMessage* newMsg = l->mergeStats();
      newMsg->level = level;
      sendStatsUp((CkMessage*)newMsg);
    }
  }
}

void TreeLB::loadBalanceSubtree(int level)
{
  if (_lb_args.debug()) CkPrintf("[PE %d] TreeLB::loadBalanceSubtree called for level %d, awaiting %s\n", CkMyPe(), level, awaitingLB[level] ? "true" : "false");
  if (!awaitingLB[level]) return;
  awaitingLB[level] = false;
  if (level == 0) return lb_done();

  // CkPrintf("[%d] TreeLB::loadBalanceSubtree - level=%d\n", CkMyPe(), level);

  /// CkMessage *inter_subtree_migrations = nullptr;
  IDM idm;
  if (_lb_args.debug()) CkPrintf("[PE %d] Calling loadBalance at level %d\n", CkMyPe(), level);
  TreeLBMessage* decision = logic[level]->loadBalance(idm); // this result is the MigMsg
  if (idm.size() > 0)
  {
    // this can happen when final destinations of chares has been decided,
    // and chares from a subtree need to migrate to a PE in a different subtree
    std::vector<int> idm_dests(1 + idm.size());
    int index = 0;
    idm_dests[index++] = CkMyPe();
    for (auto& move : idm)
    {
      CkAssert(move.second.size() > 0);
      idm_dests[index++] = move.first;
    }
    ST_RecursivePartition<std::vector<int>::iterator> tb(false, false);
    int num_subtrees =
        tb.buildSpanningTree(idm_dests.begin(), idm_dests.end(), mcast_bfactor);
    for (int i = 0; i < num_subtrees; i++)
      thisProxy[*tb.begin(i)].multicastIDM(idm, tb.subtreeSize(i), &(*tb.begin(i)));
  }

  // send decision to next level
  decision->level = level - 1;
  sendDecisionDown((CkMessage*)decision);


}

void TreeLB::multicastIDM(const IDM& mig_order, int num_pes, int* _pes)
{
#if DEBUG__TREE_LB_L3
  fprintf(stderr, "[%d] Received IDM\n", CkMyPe());
#endif
  ST_RecursivePartition<int*> tb(false, false);
  if (num_pes > 1)
  {
    int num_subtrees = tb.buildSpanningTree(_pes, _pes + num_pes, mcast_bfactor);
    for (int i = 0; i < num_subtrees; i++)
      thisProxy[*tb.begin(i)].multicastIDM(mig_order, tb.subtreeSize(i), tb.begin(i));
  }
  migrateObjects(mig_order);

  
}

void TreeLB::sendDecisionDown(CkMessage* msg)
{
  TreeLBMessage* decision = (TreeLBMessage*)msg;
  const int level = decision->level;
  std::vector<int>& children = comm_children[level];
  if (children.empty())
  {
    receiveDecision(decision, level);
  }
  else
  {
    // comm logic is free to split (scatter) the message, or send same msg to every child,
    // etc.
    CkAssert(comm_logic[level] != nullptr);
    std::vector<TreeLBMessage*> decisions =
        comm_logic[level]->splitDecision(decision, children);
    CkAssert(decisions.size() == children.size() + 1);
    if (logic[level] != nullptr)
    {
      receiveDecision(decisions[0], level);
    }
    else
    {
      delete decisions[0];
    }
    for (int i = 0; i < children.size(); i++)
    {
      // Necessary because in some cases every message in decisions is actually
      // the same message that we are reusing, so mark as unused
      _SET_USED(UsrToEnv(decisions[i + 1]), 0);
      thisProxy[children[i]].sendDecisionDown((CkMessage*)decisions[i + 1]);
    }
  }
}

void TreeLB::receiveDecision(TreeLBMessage* decision, int level)
{
  // fprintf(stderr, "[%d] TreeLB::receiveDecision, level=%d\n", CkMyPe(), level);
  // incoming and outgoing are integers. logic objects determine and interpret these
  // values
  int& incoming = expected_incoming[level];
  int& outgoing = expected_outgoing[level];
  //CkPrintf("[PE %d] TreeLB::receiveDecision at level %d, incoming=%d outgoing=%d\n", CkMyPe(), level, incoming, outgoing);
  logic[level]->processDecision(decision, incoming, outgoing);
  // fprintf(stderr, "[%d] level=%d incoming=%d outgoing=%d\n", CkMyPe(), level, incoming,
  // outgoing);
  if (incoming == 0 && outgoing == 0)
  {
    // no exchange with other subtrees, can do lb for my subtree now
    loadBalanceSubtree(level);
  }
  else
  {
    // awaiting load from other subtrees
    if (outgoing > 0 && level > 0)
    {
      // need to pass info on actual chares to other subtree(s) (they will need it sooner
      // or later since some of my chares will be moving there, but I'm not migrating
      // chares themselves yet, since their final destination is not yet known) at this
      // level I might not know about individual chares. if that's the case, I have to
      // delegate to lower levels to pass this info to the other subtree
      transferLoadToken(decision, level);
    }

    // if outgoing > 0 and we are in last level (0), concrete objects are moved to
    // concrete PEs. for chares whose destination is known, the logic object would have
    // made call to migrate them inside processDecision (above) NOTE: I might be moving
    // some objects to other subtrees, might need to wait to receive messages from those
    // subtrees telling me to which PEs to migrate them
  }
  // fprintf(stderr, "[%d] TreeLB::receiveDecision, deleting decision msg\n", CkMyPe());
  delete (CkMessage*)decision;  // FIXME this won't call the subclass' delete method
  // a check here is needed because PE might have been sending chares/tokens or
  // receiving them even before receiving the decision msg (with the final tally)
  // from the parent. So, exchange might have already completed
  checkLoadExchanged(level);
}

// order transfer of load token to dest
void TreeLB::transferLoadToken(TreeLBMessage* transferMsg, int level)
{
  // TODO this is a simplified implementation where the first level where tokens are
  // needed needs to be able to generate the tokens (so, token requests cannot be
  // propagated down the tree). There is a more advanced implementation in charm4py
  if (logic[level]->makesTokens())
  {
    // one token set goes to one destination
    std::vector<TreeLBMessage*> token_sets;
    std::vector<int> destinations;
    load_sent[level] += logic[level]->getTokenSets(transferMsg, token_sets, destinations);
    for (size_t i = 0; i < token_sets.size(); i++)
    {
      token_sets[i]->level = level;
      thisProxy[destinations[i]].recvLoadTokens((CkMessage*)token_sets[i]);
    }
  }
  else
  {
    // implemented in python. see code
    CkAbort("TreeLB::transferLoadToken - NOT IMPLEMENTED\n");
  }
}

void TreeLB::recvLoadTokens(CkMessage* tokens)
{
  TreeLBMessage* token_set = (TreeLBMessage*)tokens;
  int level = token_set->level;
#if DEBUG__TREE_LB_L3
  fprintf(stderr, "[%d] Received load token, level=%d\n", CkMyPe(), level);
#endif
  int load = logic[level]->tokensReceived(token_set);
  load_received[level] += load;
  CkPrintf("[PE %d] TreeLB::recvLoadTokens, load_received = %d\n", CkMyPe(), load_received[level]);

  checkLoadExchanged(level);
}

void TreeLB::objMovedIn(bool waitBarrier) // this should be called, but is not
{
  if (!waitBarrier) CkAbort("TreeLB future migrates not supported\n");

  // fprintf(stderr, "[%d] TreeLB::objMovedIn\n", CkMyPe());

  int level = 0;
  CkAssert(numLevels > 0 && awaitingLB[level]);
  load_received[level] += 1;

  checkLoadExchanged(level);
}

void TreeLB::migrateObjects(const IDM& mig_order)
{
  int level = 0;
  int sent = logic[level]->migrateObjects(mig_order.at(CkMyPe()));
  load_sent[level] += sent;
#if DEBUG__TREE_LB_L2
  fprintf(stderr, "[%d] Received IDM order, sent=%d\n", CkMyPe(), sent);
#endif
  checkLoadExchanged(level);
}

void TreeLB::checkForRealloc()
{
#if CMK_SHRINK_EXPAND
if (_lb_args.debug() > 0) {
      CkPrintf(
        "Check for Realloc. Number of stats messages: %d\n",
        logic[1]->stats_msgs.size()
      );
}

  if(pending_realloc_state != NO_REALLOC) {
    pending_realloc_state = (pending_realloc_state == SHRINK_MSG_RECEIVED) ? SHRINK_IN_PROGRESS : EXPAND_IN_PROGRESS; //in progress
    CkPrintf("Load balancer invoking charmrun to handle reallocation on pe %d\n", CkMyPe());
    double end_lb_time = CkWallTimer();
   
    // do checkpoint
    CkCallback cb(CkIndex_TreeLB::resumeFromReallocCheckpoint(), thisProxy[0]);

    // print avail vector
    if (_lb_args.debug() > 0) {
      CkPrintf("Shrink/Expand se_avail_vector on pe %d: ", CkMyPe());
      for(int i=0;i<CkNumPes();i++) CkPrintf("%d ", se_avail_vector[i]);
      CkPrintf("\n");
    }

    //print a couple object loads to sample;
    CkStartRescaleCheckpoint(_shrinkexpand_basedir, cb, 
      std::vector<char>(se_avail_vector, se_avail_vector + CkNumPes()));
  }
  else
  {
    thisProxy.lb_done_impl();
  }
#endif
}

void TreeLB::resumeFromReallocCheckpoint()
{
#if CMK_SHRINK_EXPAND
  std::vector<char> avail(se_avail_vector, se_avail_vector + CkNumPes());
  free(se_avail_vector);
  thisProxy.willIbekilled(avail, numProcessAfterRestart);
#endif
}

void TreeLB::willIbekilled(std::vector<char> avail, int newnumProcessAfterRestart){
#if CMK_SHRINK_EXPAND
  numProcessAfterRestart = newnumProcessAfterRestart;
  CkCallback cb(CkIndex_TreeLB::startCleanup(), thisProxy[0]);
  contribute(cb);
#endif
}

void TreeLB::startCleanup()
{
#if CMK_SHRINK_EXPAND
  CkCleanup();
#endif
}

void TreeLB::lb_done()
{
#if CMK_SHRINK_EXPAND
  // barrier to check for reallocation
  CkCallback cb(CkIndex_TreeLB::checkForRealloc(), thisProxy[0]);
  contribute(cb);
  return;
#else
    lb_done_impl();
#endif
}

void TreeLB::lb_done_impl()
{
  // fprintf(stderr, "[%d] lb_done step %d lb_time=%f\n", CkMyPe(), lbmgr->step(),
  // CkWallTimer() - startTime);

  // TODO LBManager should do all of this, including global syncResume ******
  // Currently, TreeLB does syncResume by setting barrier_after_lb=true


#if CMK_SHRINK_EXPAND
  // Only clear loads if not in the middle of a reallocation (EXPAND/SHRINK)
  if (pending_realloc_state == NO_REALLOC){
    lbmgr->ClearLoads();
  }
#else
  lbmgr->ClearLoads();
#endif

  if (CkMyPe() == 0 && _lb_args.debug() > 0)
  {
    CkPrintf("--------- Finished LB step %d ---------\n", lbmgr->step());
  }

  // Advance to next load balancer
  if (!(_lb_args.metaLbOn() && _lb_args.metaLbModelDir() != nullptr))
    lbmgr->nextLoadbalancer(seqno);

  // Increment to next step
  lbmgr->incStep();

  LBManager::Object()
      ->MigrationDone();  // call registered callbacks. not sure what this is for

  // again, TreeLB shouldn't be doing these things. it should just notify LBManager
  // that it's done. LBManager should take care of the rest.
  // not sure why this has to be called as entry method. but if not, it seems like
  // the last object to migrate in is not resumed
  if (barrier_after_lb)
  {
    contribute(CkCallback(CkReductionTarget(TreeLB, resumeClients), thisProxy));
  }
  else
  {
    thisProxy[CkMyPe()].resumeClients();
  }
}

void TreeLB::resumeClients()
{
  double lb_time = CkWallTimer() - startTime;
  if (CkMyPe() == 0 && _lb_args.debug() > 0)
    CkPrintf("[%d] lb time = %f\n", CkMyPe(), lb_time);
  int step = lbmgr->step() - 1;
  // fprintf(stderr, "[%d] step %d lb_time=%f\n", CkMyPe(), step, lb_time);
  if (step_report_lb_times >= step)
  {
    lb_times.push_back(lb_time);
    if (step == step_report_lb_times)
    {
      contribute(lb_times, CkReduction::sum_double,
                 CkCallback(CkReductionTarget(TreeLB, reportLbTime), thisProxy[0]));
      lb_times.clear();
    }
  }
  lbmgr->ResumeClients();

  lbmgr->lb_in_progress = false;

  if (CkMyPe() == 0)
    lbmgr->callRealloc();
}

void TreeLB::reportLbTime(double* times, int n)
{
  fprintf(stderr, "lb times: ");
  for (int i = 0; i < n; i++)
  {
    double avg_time = times[i] / CkNumPes();
    fprintf(stderr, "%f ", avg_time);
  }
  fprintf(stderr, "\n");
}

#include "TreeLB.def.h"
