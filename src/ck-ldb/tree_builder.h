#ifndef TREE_BUILDERS_H
#define TREE_BUILDERS_H

#include "TreeLB.h"
#include "tree_level.h"
#include "spanningTree.h"
#include <algorithm>

extern int quietModeRequested;


class LBTreeBuilderCommon : public LBTreeBuilder
{

public:

  virtual ~LBTreeBuilderCommon() {}

protected:

  void reset(uint8_t num_levels, std::vector<LevelLogic*> &logic,
             std::vector<int> &comm_parent, std::vector<std::vector<int>> &comm_children,
             std::vector<LevelLogic*> &comm_logic) {

    logic.resize(num_levels);
    comm_parent.resize(num_levels);
    comm_children.resize(num_levels);
    comm_logic.resize(num_levels);

    std::fill(logic.begin(), logic.end(), nullptr);
    std::fill(comm_parent.begin(), comm_parent.end(), -1);
    for (auto &children : comm_children) children.clear();
    std::fill(comm_logic.begin(), comm_logic.end(), nullptr);
  }

};


class PE_Root_Tree : public LBTreeBuilderCommon
{

public:

  virtual ~PE_Root_Tree() {}

  virtual uint8_t build(std::vector<LevelLogic*> &logic, std::vector<int> &comm_parent,
                        std::vector<std::vector<int>> &comm_children,
                        std::vector<LevelLogic*> &comm_logic, json &config) {

    uint8_t L = 2;  // num levels
    reset(L, logic, comm_parent, comm_children, comm_logic);

    int rootPE = 0;
    try {
      rootPE = config["Root"]["pe"];
    } catch (const std::exception &e) {
    }
    CkAssert(rootPE >= 0 && rootPE < CkNumPes());

    LBManager *lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();

    // PE level (level 0)
    logic[0] = new PELevel(lbmgr);

    int parent, num_children;
    int *children;
    getPETopoTreeEdges(CkMyPe(), rootPE, NULL, -1, 4, &parent, &num_children, &children);
    comm_parent[0] = parent;
    if (num_children > 0) {
      comm_children[0].assign(children, children + num_children);
      free(children);
    }
    comm_logic[0] = new MsgAggregator(num_children);

    // root of tree (level 1)
    if (CkMyPe() == rootPE) {
      RootLevel *level = new RootLevel();
      level->configure(false, config["Root"]);
      logic[1] = level;
    }

    if (CkMyPe() == 0 && !quietModeRequested) {
      auto &strategies = config["Root"]["strategies"];
      if (strategies.size() == 1) {
        const std::string &strategy = strategies[0];
        CkPrintf("[%d] TreeLB: Using PE_Root tree with strategy %s\n", CkMyPe(), strategy.c_str());
      } else {
        CkPrintf("[%d] TreeLB: Using PE_Root tree\n", CkMyPe());
      }
    }

    return L;
  }

};


class PE_Node_Root_Tree : public LBTreeBuilderCommon
{

public:

  virtual ~PE_Node_Root_Tree() {}

  virtual uint8_t build(std::vector<LevelLogic*> &logic, std::vector<int> &comm_parent,
                        std::vector<std::vector<int>> &comm_children,
                        std::vector<LevelLogic*> &comm_logic, json &config) {

    uint8_t L = 3;  // num levels
    reset(L, logic, comm_parent, comm_children, comm_logic);

    int rootPE = 0;
    try {
      rootPE = config["Root"]["pe"];
    } catch (const std::exception &e) {}
    CkAssert(rootPE >= 0 && rootPE < CkNumPes());

    int mype = CkMyPe();
    int mynode = CkMyNode();
    int level1root = CkNodeFirst(mynode);
    int level2root = rootPE;
    LBManager *lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();

    // PE level (level 0)
    int lvl = 0;
    logic[lvl] = new PELevel(lbmgr);

    // set up comm-tree between levels 0 and 1
    if (mype == level1root) {
      // pes in my node excluding me
      for (int pe=0; pe < CkNumPes(); pe++) {
        if ((pe != mype) && (CkNodeOf(pe) == mynode))
          comm_children[lvl].push_back(pe);
      }
    } else {
      comm_parent[lvl] = level1root;
    }

    int step_freq_lvl2;
    try {
      step_freq_lvl2 = config["Root"]["step_freq"];
    } catch (const std::exception &e) {
      step_freq_lvl2 = 1;
    }

    // node level (level 1)
    lvl = 1;
    if (mype == level1root) {
      std::vector<int> pes_in_node;
      for (int pe=0; pe < CkNumPes(); pe++) {
        if (CkNodeOf(pe) == mynode) pes_in_node.push_back(pe);
      }
      NodeLevel *level = new NodeLevel(lbmgr, pes_in_node);
      level->configure(false, config["Process"], step_freq_lvl2);
      logic[lvl] = level;

      // set up comm-tree between levels 1 and 2
      std::vector<int> level1_roots;
      level1_roots.push_back(level2root); // need this for getPETopoTreeEdges
      for (int pe=0; pe < CkNumPes(); pe++) {
        if (pe == level2root) continue;
        if (CkNodeFirst(CkNodeOf(pe)) == pe) level1_roots.push_back(pe);
      }
      int parent, num_children;
      int *children;
      getPETopoTreeEdges(mype, level2root, level1_roots.data(), level1_roots.size(), 4, &parent, &num_children, &children);
      comm_parent[lvl] = parent;
      if (num_children > 0) {
        comm_children[lvl].assign(children, children + num_children);
        free(children);
      }
      comm_logic[lvl] = new MsgAggregator(num_children);
    }

    // root of tree (level 2)
    lvl = 2;
    if (mype == level2root) {
      RootLevel *level = new RootLevel();
      level->configure(false, config["Root"]);
      logic[lvl] = level;
    }

    if (CkMyPe() == 0 && !quietModeRequested)
      CkPrintf("[%d] TreeLB: Using PE_Process_Root tree\n", CkMyPe());

    return L;
  }

};

class PE_Node_NodeSet_Root_Tree : public LBTreeBuilderCommon
{
public:

  PE_Node_NodeSet_Root_Tree(int _num_groups) : num_groups(_num_groups) {
    CkAssert(CkNumNodes() >= num_groups);
    // NOTE to simplify things for now, assume equal number of nodes per group
    CkAssert(CkNumNodes() % num_groups == 0);
    NODES_PER_GROUP = CkNumNodes() / num_groups;
  }

  virtual ~PE_Node_NodeSet_Root_Tree() {}

  virtual uint8_t build(std::vector<LevelLogic*> &logic, std::vector<int> &comm_parent,
                        std::vector<std::vector<int>> &comm_children,
                        std::vector<LevelLogic*> &comm_logic, json &config) {

    uint8_t L = 4;  // num levels
    reset(L, logic, comm_parent, comm_children, comm_logic);

    int rootPE = 0;
    try {
      rootPE = config["Root"]["pe"];
    } catch (const std::exception &e) {}
    CkAssert(rootPE >= 0 && rootPE < CkNumPes());

    int mype = CkMyPe();
    int mynode = CkMyNode();
    int mygroup = GroupOf(mype);
    int level1root = CkNodeFirst(mynode);
    int level2root = GroupFirstPe(mygroup);
    int level3root = rootPE;
    LBManager *lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();

    // PE level (level 0)
    int lvl = 0;
    logic[lvl] = new PELevel(lbmgr);

    // set up comm-tree between levels 0 and 1
    if (mype == level1root) {
      // pes in my node excluding me
      for (int pe=0; pe < CkNumPes(); pe++) {
        if ((pe != mype) && (CkNodeOf(pe) == mynode))
          comm_children[lvl].push_back(pe);
      }
    } else {
      comm_parent[lvl] = level1root;
    }

    int step_freq_lvl2, step_freq_lvl3;
    try {
      step_freq_lvl2 = config["ProcessGroup"]["step_freq"];
    } catch (const std::exception &e) {
      step_freq_lvl2 = 1;
    }
    try {
      step_freq_lvl3 = config["Root"]["step_freq"];
      if (step_freq_lvl3 % step_freq_lvl2 != 0)
        CkAbort("step_freq of Root level is not multiple of previous level\n");
    } catch (const std::exception &e) {
      step_freq_lvl3 = step_freq_lvl2;
    }

    // node level (level 1)
    lvl = 1;
    if (mype == level1root) {
      std::vector<int> pes_in_node;
      for (int pe=0; pe < CkNumPes(); pe++) {
        if (CkNodeOf(pe) == mynode) pes_in_node.push_back(pe);
      }
      NodeLevel *level = new NodeLevel(lbmgr, pes_in_node);
      level->configure(false, config["Process"], step_freq_lvl2);
      logic[lvl] = level;

      // set up comm-tree between levels 1 and 2
      std::vector<int> level1_roots;
      for (int pe=0; pe < CkNumPes(); pe++) {
        if ((CkNodeFirst(CkNodeOf(pe)) == pe) && (GroupOf(pe) == mygroup)) level1_roots.push_back(pe);
      }
      int parent, num_children;
      int *children;
      getPETopoTreeEdges(mype, level2root, level1_roots.data(), level1_roots.size(), 4, &parent, &num_children, &children);
      comm_parent[lvl] = parent;
      if (num_children > 0) {
        comm_children[lvl].assign(children, children + num_children);
        free(children);
      }
      comm_logic[lvl] = new MsgAggregator(num_children);
    }

    // nodeset level (level 2)
    lvl = 2;
    if (mype == level2root) {
      std::vector<int> pes_in_group;
      for (int pe=0; pe < CkNumPes(); pe++) {
        if (GroupOf(pe) == mygroup) pes_in_group.push_back(pe);
      }
      NodeSetLevel *level = new NodeSetLevel(lbmgr, pes_in_group);
      level->configure(false, config["ProcessGroup"], step_freq_lvl3);
      logic[lvl] = level;

      /*if (mype == level3root) {
        for (int pe=0; pe < CkNumPes(); pe++) {
          if ((pe != mype) && (GroupFirstPe(GroupOf(pe)) == pe)) comm_children[lvl].push_back(pe);
        }
      } else {
        comm_parent[lvl] = level3root;
      }*/
      if (mype != level3root)  // NEW
        comm_parent[lvl] = level3root;
    }

    // root of tree (level 3)
    lvl = 3;
    if (mype == level3root) {

      for (int pe=0; pe < CkNumPes(); pe++) {  // NEW
        if ((pe != mype) && (GroupFirstPe(GroupOf(pe)) == pe)) comm_children[lvl-1].push_back(pe);
      }

      RootLevel *level = new RootLevel(num_groups);
      level->configure(false, config["Root"]);
      logic[lvl] = level;
    }

    if (CkMyPe() == 0 && !quietModeRequested)
      CkPrintf("[%d] TreeLB: Using PE_Process_ProcessGroup_Root tree with %d groups\n",
               CkMyPe(), int(config["ProcessGroup"]["num_groups"]));

    return L;
  }

private:

  int GroupOf(int pe) { return CkNodeOf(pe) / NODES_PER_GROUP; }

  int GroupFirstPe(int group) {
    return CkNodeFirst(group * NODES_PER_GROUP);
  }

  int num_groups;
  int NODES_PER_GROUP;

};


#endif  /* TREE_BUILDERS_H */
