#include "TreeLB.h"
#include "spanningTree.h"
#include "tree_builder.h"  // TODO this can be deleted if we change it so that LBManager instantiates the builders
#include <fstream> // TODO delete if json file is read from LBManager

extern int quietModeRequested;

CreateLBFunc_Def(TreeLB, "TreeLB")

void TreeLB::staticStartLB(void* data) {
  ((TreeLB*)data)->StartLB();
}

void TreeLB::staticObjMovedIn(void* data, LDObjHandle h, bool waitBarrier) {
//  ((TreeLB*)data)->objMovedIn(h, waitBarrier);
}

void TreeLB::Migrated(int waitBarrier){
  bool barrier = false;
  if(waitBarrier) barrier = true;
  objMovedIn(barrier);
}


void TreeLB::init(const CkLBOptions &opts) {
#if CMK_LBDB_ON

  lbname = "TreeLB";

  if (_lb_args.syncResume())
    barrier_after_lb = true;

  // create and turn on by default
  startLbFnHdl = lbmgr->AddStartLBFn((LDStartLBFn)(staticStartLB), (void*)(this));

  json config;
  std::ifstream ifs(_lb_args.treeLBFile(), std::ifstream::in);
  if (ifs.good()) {
    try {
      ifs >> config;
    } catch (const std::exception &e) {
      CkPrintf("Error reading TreeLB configuration file: %s\n", e.what());
      CkExit(1);
    }
  } else if (_lb_args.legacyCentralizedStrategies().size() > 0) {
    // support legacy mode, e.g. map "+GreedyLB" to PE_Root tree using Greedy
    // use 2-level tree
    config["tree"] = "PE_Root";
    config["Root"]["strategies"] = { _lb_args.legacyCentralizedStrategies()[0] };
    if (CkMyPe() == 0 && !quietModeRequested)
        CkPrintf("[%d] TreeLB in LEGACY MODE support\n", CkMyPe());
  } else {
    if (CkMyPe() == 0 && !quietModeRequested)
      CkPrintf("[%d] No TreeLB configuration file found. Choosing a default configuration.\n", CkMyPe());
    // try to pick reasonable defaults
    // the problem with using a 2 or 3 level tree in large jobs is that the root's
    // strategy could take a long time to run, and also many strategies (but not all)
    // could move objects across the whole machine (which means potentially slow migrations
    // and could also hurt communication performance depending on the strategy).
    if (CmiNumPhysicalNodes() >= 128) {
      // use 4-level tree
      config["tree"] = "PE_Process_ProcessGroup_Root";
      config["Root"]["pe"] = 0;
      config["ProcessGroup"]["num_groups"] = CmiNumPhysicalNodes() / 32;
      config["Root"]["step_freq"] = 10;
      config["ProcessGroup"]["step_freq"] = 5;
      config["Process"]["strategies"] = {"GreedyRefine"};
      config["ProcessGroup"]["strategies"] = {"GreedyRefine"};
      config["ProcessGroup"]["GreedyRefine"]["tolerance"] = 1.03;
    } else if (CmiNumNodes() > 1 && CmiNodeSize(0) > 1) {
      // use 3-level tree
      config["tree"] = "PE_Process_Root";
      config["Root"]["pe"] = 0;
      config["Root"]["step_freq"] = 3;
      config["Root"]["strategies"] = {"GreedyRefine"};
      config["Process"]["strategies"] = {"GreedyRefine"};
    } else {
      // use 2-level tree
      config["tree"] = "PE_Root";
      config["Root"]["pe"] = 0;
      config["Root"]["strategies"] = {"GreedyRefine"};
    }
  }
  ifs.close();
  configure(config);

  // TODO this functionality needs to move to LBManager
  if (_lb_args.statsOn()) lbmgr->CollectStatsOn();  // collect load and (optionally) comm stats

#endif
}

TreeLB::~TreeLB() {
#if CMK_LBDB_ON
  lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  if (lbmgr) {
    lbmgr->RemoveStartLBFn((LDStartLBFn)(staticStartLB));
  }

  for (auto l : logic) delete l;
  for (auto l : comm_logic) delete l;

#endif
}

void TreeLB::configure(LBTreeBuilder &builder, json &config) {
#if CMK_LBDB_ON

  if (numLevels > 0 && CkMyPe() == 0 && !quietModeRequested) {
    CkPrintf("[%d] Reconfiguring TreeLB\n", CkMyPe());
  }

  auto config_entry = config.find("mcast_bfactor");
  if (config_entry != config.end())
    mcast_bfactor = *config_entry;

  config_entry = config.find("report_lb_times_at_step");
  if (config_entry != config.end())
    step_report_lb_times = *config_entry;

  if (numLevels > 0) {
    for (auto l : logic) delete l;
    for (auto l : comm_logic) delete l;
  }
  numLevels = builder.build(logic, comm_parent, comm_children, comm_logic, config);
  CkAssert(numLevels > 0 && logic.size() == numLevels && comm_parent.size() == numLevels
                         && comm_children.size() == numLevels && comm_logic.size() == numLevels);

  expected_incoming.resize(numLevels);
  expected_outgoing.resize(numLevels);
  load_sent.resize(numLevels);
  load_received.resize(numLevels);
  //notify_after_transfer.resize(numLevels);
  awaitingLB.resize(numLevels);

  // reset all values since this may be a re-configuration
  std::fill(expected_incoming.begin(), expected_incoming.end(), 0);
  std::fill(expected_outgoing.begin(), expected_outgoing.end(), 0);
  std::fill(load_sent.begin(), load_sent.end(), 0);
  std::fill(load_received.begin(), load_received.end(), 0);
  //std::fill(notify_after_transfer.begin(), notify_after_transfer.end(), -1);
  std::fill(awaitingLB.begin(), awaitingLB.end(), false);

#endif
}

void TreeLB::configure(json &config) {
#if CMK_LBDB_ON

  LBTreeBuilder *builder;
  const std::string &tree_type = config["tree"];
  if (tree_type == "PE_Root") {
    builder = new PE_Root_Tree();
  } else if (tree_type == "PE_Process_Root") {
    builder = new PE_Node_Root_Tree();
  } else if (tree_type == "PE_Process_ProcessGroup_Root") {
    builder = new PE_Node_NodeSet_Root_Tree(config["ProcessGroup"]["num_groups"]);
  } else {
    CkAbort("TreeLB: configured tree not recognized\n");
  }

  configure(*builder, config);
  delete builder;

#endif
}

void TreeLB::InvokeLB()
{
#if CMK_LBDB_ON
  // NOTE: I'm assuming new LBManager will know when (and when not to) call AtSync
  if (barrier_before_lb) {
    contribute(CkCallback(CkReductionTarget(TreeLB, ProcessAtSync), thisProxy));
  } else {
    thisProxy[CkMyPe()].ProcessAtSync();
  }
#endif
}

void TreeLB::ProcessAtSync()
{
#if CMK_LBDB_ON
  startTime = CkWallTimer();
  if (CkMyPe() == 0 && _lb_args.debug() > 0) {
    CkPrintf("--------- Started LB step %d ---------\n", lbmgr->step());
  }
  //CmiAssert(CmiNodeAlive(CkMyPe()));   // TODO move this logic to LBManager
  int level = 0;  // load balancing starts at the lowest level
  CkAssert(numLevels > 0 && !awaitingLB[level]);
  TreeLBMessage *stats = logic[level]->getStats();
  stats->level = level;
  awaitingLB[level] = true;
  sendStatsUp((CkMessage*)stats);
#endif
}

// send stats up using the comm-tree for this level
void TreeLB::sendStatsUp(CkMessage *msg) {
  TreeLBMessage *stats = (TreeLBMessage*)msg;
  int level = stats->level;
  int comm_parent_pe = comm_parent[level];
  //fprintf(stderr, "[%d] TreeLB::sendStatsUp - received msg level=%d comm_parent=%d\n", CkMyPe(), level, comm_parent_pe);
  if (comm_parent_pe == -1) {
    // I'm the root of this comm-tree (current destination of the stats)
    receiveStats(stats, level);
  } else if (comm_children[level].size() == 0) {
    // don't have children so don't have to aggregate any msgs
    thisProxy[comm_parent_pe].sendStatsUp((CkMessage*)stats);
  } else {
    LevelLogic *logic = comm_logic[level];
    logic->depositStats(stats);
    if (logic->numStatsReceived() == comm_children[level].size() + 1) {
      TreeLBMessage *newMsg = logic->mergeStats();
      newMsg->level = level;
      thisProxy[comm_parent_pe].sendStatsUp((CkMessage*)newMsg);
    }
  }
}

void TreeLB::receiveStats(TreeLBMessage *stats, int level) {
  level += 1;
  awaitingLB[level] = true;
  LevelLogic *l = logic[level];
  l->depositStats(stats);
  size_t expected_msgs = comm_children[level-1].size();
  if (logic[level-1] != nullptr) expected_msgs += 1;  // expect msg from myself too
  if (l->numStatsReceived() == expected_msgs) {
    CkAssert(load_sent[level] == 0 && load_received[level] == 0);
    if (level == numLevels-1 || l->cutoff()) {
      // cutoff can be adjusted dynamically, to prevent lb between upper-level domains.
      // can be used, for example, to only do within-node lb on some steps
      loadBalanceSubtree(level);
    } else {
      TreeLBMessage *newMsg = l->mergeStats();
      newMsg->level = level;
      sendStatsUp((CkMessage*)newMsg);
    }
  }
}

void TreeLB::loadBalanceSubtree(int level) {
  if (!awaitingLB[level]) return;
  awaitingLB[level] = false;
  if (level == 0) return lb_done();

  //CkPrintf("[%d] TreeLB::loadBalanceSubtree - level=%d\n", CkMyPe(), level);

  std::vector<TreeLBMessage*> decisions;
  ///CkMessage *inter_subtree_migrations = nullptr;
  IDM idm;
  logic[level]->loadBalance(decisions, idm);
  if (idm.numDests() > 0) {
    // this can happen when final destinations of chares has been decided,
    // and chares from a subtree need to migrate to a PE in a different subtree
    std::vector<int> idm_dests;
    idm_dests.reserve(1 + idm.numDests());
    idm_dests.push_back(CkMyPe());
    for (auto &move : idm.data) {
      CkAssert(move.second.size() > 0);
      idm_dests.push_back(move.first);
    }
    ST_RecursivePartition<std::vector<int>::iterator> tb(false, false);
    int num_subtrees = tb.buildSpanningTree(idm_dests.begin(), idm_dests.end(), mcast_bfactor);
    for (int i=0; i < num_subtrees; i++)
      thisProxy[*tb.begin(i)].multicastIDM(idm, tb.subtreeSize(i), &(*tb.begin(i)));
  }

  // send decision to next level
  level -= 1;
  int curr_child = 0;
  std::vector<int> &children = comm_children[level];
  for (int i=0; i < decisions.size(); i++) {
    if ((i == 0) && (logic[level] != nullptr)) {
      CkAssert(decisions.size() == children.size() + 1);
      // first decision msg is for this PE
      receiveDecision(decisions[0], level);
    } else {
      decisions[i]->level = level;
      thisProxy[children[curr_child++]].sendDecisionDown((CkMessage*)decisions[i]);
    }
  }
}

void TreeLB::multicastIDM(const IDM &mig_order, int num_pes, int *_pes) {
#if DEBUG__TREE_LB_L3
  fprintf(stderr, "[%d] Received IDM\n", CkMyPe());
#endif
  ST_RecursivePartition<int*> tb(false, false);
  if (num_pes > 1) {
    int num_subtrees = tb.buildSpanningTree(_pes, _pes + num_pes, mcast_bfactor);
    for (int i=0; i < num_subtrees; i++)
      thisProxy[*tb.begin(i)].multicastIDM(mig_order, tb.subtreeSize(i), tb.begin(i));
  }
  migrateObjects(mig_order);
}

void TreeLB::sendDecisionDown(CkMessage *msg) {
  TreeLBMessage *decision = (TreeLBMessage*)msg;
  int level = decision->level;
  std::vector<int> &children = comm_children[level];
  if (children.size() == 0) {
    receiveDecision(decision, level);
  } else {
    // comm logic is free to split (scatter) the message, or send same msg to every child, etc.
    std::vector<TreeLBMessage*> decisions;
    CkAssert(comm_logic[level] != nullptr);
    comm_logic[level]->splitDecision(decision, decisions);
    CkAssert(decisions.size() == children.size() + 1);
    decisions[0]->level = level;
    receiveDecision(decisions[0], level);
    for (int i=0; i < children.size(); i++) {
      decisions[i+1]->level = level;
      thisProxy[children[i]].sendDecisionDown((CkMessage*)decisions[i+1]);
    }
  }
}

void TreeLB::receiveDecision(TreeLBMessage *decision, int level) {

  //fprintf(stderr, "[%d] TreeLB::receiveDecision, level=%d\n", CkMyPe(), level);

  // incoming and outgoing are integers. logic objects determine and interpret these values
  int &incoming = expected_incoming[level];
  int &outgoing = expected_outgoing[level];
  logic[level]->processDecision(decision, incoming, outgoing);
  //fprintf(stderr, "[%d] level=%d incoming=%d outgoing=%d\n", CkMyPe(), level, incoming, outgoing);
  if (incoming == 0 && outgoing == 0) {
    // no exchange with other subtrees, can do lb for my subtree now
    loadBalanceSubtree(level);
  } else {
    // awaiting load from other subtrees
    if (outgoing > 0 && level > 0) {
      // need to pass info on actual chares to other subtree(s) (they will need it sooner
      // or later since some of my chares will be moving there, but I'm not migrating chares
      // themselves yet, since their final destination is not yet known)
      // at this level I might not know about individual chares. if that's the case,
      // I have to delegate to lower levels to pass this info to the other subtree
      transferLoadToken(decision, level);
    }

    // if outgoing > 0 and we are in last level (0), concrete objects are moved to concrete PEs.
    // for chares whose destination is known, the logic object would have made call to
    // migrate them inside processDecision (above)
    // NOTE: I might be moving some objects to other subtrees, might need to wait to receive
    // messages from those subtrees telling me to which PEs to migrate them
  }
  //fprintf(stderr, "[%d] TreeLB::receiveDecision, deleting decision msg\n", CkMyPe());
  delete (CkMessage*)decision;   // FIXME this won't call the subclass' delete method
  // a check here is needed because PE might have been sending chares/tokens or
  // receiving them even before receiving the decision msg (with the final tally)
  // from the parent. So, exchange might have already completed
  checkLoadExchanged(level);
}

// order transfer of load token to dest
void TreeLB::transferLoadToken(TreeLBMessage *transferMsg, int level) {
  // TODO this is a simplified implementation where the first level where tokens are needed
  // needs to be able to generate the tokens (so, token requests cannot be propagated
  // down the tree). There is a more advanced implementation in charm4py
  if (logic[level]->makesTokens()) {
    // one token set goes to one destination
    std::vector<TreeLBMessage*> token_sets;
    std::vector<int> destinations;
    load_sent[level] += logic[level]->getTokenSets(transferMsg, token_sets, destinations);
    for (size_t i=0; i < token_sets.size(); i++) {
      token_sets[i]->level = level;
      thisProxy[destinations[i]].recvLoadTokens((CkMessage*)token_sets[i]);
    }
  } else {
    // implemented in python. see code
    CkAbort("TreeLB::transferLoadToken - NOT IMPLEMENTED\n");
  }
}

void TreeLB::recvLoadTokens(CkMessage *tokens) {
  TreeLBMessage *token_set = (TreeLBMessage*)tokens;
  int level = token_set->level;
#if DEBUG__TREE_LB_L3
  fprintf(stderr, "[%d] Received load token, level=%d\n", CkMyPe(), level);
#endif
  int load = logic[level]->tokensReceived(token_set);
  load_received[level] += load;
  checkLoadExchanged(level);
}

void TreeLB::objMovedIn( bool waitBarrier) {
  if (!waitBarrier)
    CkAbort("TreeLB future migrates not supported\n");

  //fprintf(stderr, "[%d] TreeLB::objMovedIn\n", CkMyPe());

  int level = 0;
  CkAssert(numLevels > 0 && awaitingLB[level]);
  load_received[level] += 1;
  checkLoadExchanged(level);
}

void TreeLB::migrateObjects(const IDM &mig_order) {
  int level = 0;
  int sent = logic[level]->migrateObjects(mig_order.data.at(CkMyPe()));
  load_sent[level] += sent;
#if DEBUG__TREE_LB_L2
  fprintf(stderr, "[%d] Received IDM order, sent=%d\n", CkMyPe(), sent);
#endif
  checkLoadExchanged(level);
}

void TreeLB::lb_done() {

  //fprintf(stderr, "[%d] lb_done step %d lb_time=%f\n", CkMyPe(), lbmgr->step(), CkWallTimer() - startTime);

  // TODO LBManager should do all of this, including global syncResume ******
  // Currently, TreeLB does syncResume by setting barrier_after_lb=true

  // clear load stats
  lbmgr->ClearLoads();

  if (CkMyPe() == 0 && _lb_args.debug() > 0) {
    CkPrintf("--------- Finished LB step %d ---------\n", lbmgr->step());
  }

  // Increment to next step
  lbmgr->incStep();

  LBManager::Object()->MigrationDone();    // call registered callbacks. not sure what this is for

  // again, TreeLB shouldn't be doing these things. it should just notify LBManager
  // that it's done. LBManager should take care of the rest.
  // not sure why this has to be called as entry method. but if not, it seems like
  // the last object to migrate in is not resumed
  if (barrier_after_lb) {
    contribute(CkCallback(CkReductionTarget(TreeLB, resumeClients), thisProxy));
  } else {
    thisProxy[CkMyPe()].resumeClients();
  }
}

void TreeLB::resumeClients() {
  double lb_time = CkWallTimer() - startTime;
  if (CkMyPe() == 0 && _lb_args.debug() > 0)
    CkPrintf("[%d] lb time = %f\n", CkMyPe(), lb_time);
  int step = lbmgr->step() - 1;
  //fprintf(stderr, "[%d] step %d lb_time=%f\n", CkMyPe(), step, lb_time);
  if (step_report_lb_times >= step) {
    lb_times.push_back(lb_time);
    if (step == step_report_lb_times) {
      contribute(lb_times, CkReduction::sum_double,
                 CkCallback(CkReductionTarget(TreeLB, reportLbTime), thisProxy[0]));
      lb_times.clear();
    }
  }
  lbmgr->ResumeClients();
}

void TreeLB::reportLbTime(double *times, int n) {
  fprintf(stderr, "lb times: ");
  for (int i=0; i < n; i++) {
    double avg_time = times[i] / CkNumPes();
    fprintf(stderr, "%f ", avg_time);
  }
  fprintf(stderr, "\n");
}

#include "TreeLB.def.h"
