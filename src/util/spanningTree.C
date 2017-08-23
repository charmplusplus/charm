/**
 * Author: jjgalvez@illinois.edu (Juan Galvez)
 * Uses recursive bisect/trisect functionality similar to what was in
 * src/util/treeStrategy_3dTorus_minHops.h
 */
#include "spanningTree.h"
#include "TopoManager.h"
#include <algorithm>
#include <limits.h>

#include <unordered_map>
typedef std::unordered_map<int,int> intMap;

#include <bitset>
#define DIM_SET_SIZE 32     // bitset size

#define _DEBUG_SPANNING_TREE_ 0

#if _DEBUG_SPANNING_TREE_
#include <sstream>
#endif

template <typename Iterator>
class ST_RecursivePartition<Iterator>::PhyNode {
public:
  PhyNode(int id, int pe, TopoManager *tmgr) : id(id), pe(pe) {
    if (tmgr->haveTopologyInfo()) tmgr->rankToCoordinates(pe, coords);
  }
  inline void addNode(int n) { nodes.push_back(n); }
  inline int size() const { return nodes.size(); }
  inline int getNode(int i) const {
    CkAssert(i >= 0 && i < nodes.size());
    return nodes[i];
  }
  /// distance to other phynode
  inline int distance(const PhyNode &o, TopoManager *tmgr) const {
    return tmgr->getHopsBetweenRanks(pe, o.pe);
  }

#if _DEBUG_SPANNING_TREE_
  void print() {
    CkPrintf("phynode %d, pe=%d, coords=", id, pe);
    for (int i=0; i < coords.size(); i++) CkPrintf("%d ", coords[i]);
    CkPrintf(", nodes: ");
    for (int i=0; i < nodes.size(); i++) CkPrintf("%d ", nodes[i]);
    CkPrintf("\n");
  }
#endif

  int id;
  int pe; /// a pe in physical node (doesn't matter which one it is)
  std::vector<int> nodes;  /// (charm)nodes in this phynode
  std::vector<int> coords; /// coordinates of this phynode
};

template <typename Iterator>
class ST_RecursivePartition<Iterator>::PhyNodeCompare {
public:
  PhyNodeCompare(int dim): dim(dim) {}
  inline bool operator()(const typename ST_RecursivePartition::PhyNode *a,
                         const typename ST_RecursivePartition::PhyNode *b) const {
    if (a->coords[dim] == b->coords[dim]) return (a->id < b->id);
    else return (a->coords[dim] < b->coords[dim]);
  }
private:
  const int dim;
};

// ----------------- ST_RecursivePartition -----------------

template <typename Iterator>
ST_RecursivePartition<Iterator>::ST_RecursivePartition(bool nodeTree, bool preSorted)
  : nodeTree(nodeTree), preSorted(preSorted)
{
  tmgr = TopoManager::getTopoManager();
  if (tmgr->haveTopologyInfo()) {
    for (int i=0; i < tmgr->getNumDims(); i++) {
      if (tmgr->getDimSize(i) > DIM_SET_SIZE)
        CkAbort("ST_RecursivePartition:: Increase bitset size to match size of largest topology dimension");
    }
  }
#if _DEBUG_SPANNING_TREE_
  if (CkMyNode() == 0) {
    CkPrintf("TopoManager reports topoinfo=%d, %d dims, dim sizes: ", tmgr->haveTopologyInfo(), tmgr->getNumDims());
    for (int i=0; i < tmgr->getNumDims(); i++) CkPrintf("%d ", tmgr->getDimSize(i));
    CkPrintf("\n");
  }
#endif
}

template <typename Iterator>
int ST_RecursivePartition<Iterator>::buildSpanningTree(Iterator start, Iterator end,
                                                       unsigned int maxBranches)
{
  children.clear();
  const int numNodes = std::distance(start, end);
  if (numNodes == 0) CkAbort("Error: requested spanning tree but no nodes\n");
  else if (numNodes == 1) return 0;

#if _DEBUG_SPANNING_TREE_
  CkPrintf("[%d] ST_RecursivePartition:: Root is %d, being requested %d children, Num nodes incl root is %d\n",
           CkMyNode(), *start, maxBranches, numNodes);
#endif

  // group nodes into phynodes
  std::vector<typename ST_RecursivePartition<Iterator>::PhyNode> phynodes;
  initPhyNodes(start, end, phynodes);
  std::vector<typename ST_RecursivePartition<Iterator>::PhyNode*> pphynodes(phynodes.size());
  for (int i=0; i < phynodes.size(); i++) pphynodes[i] = &phynodes[i];

  // build the spanning tree of physical nodes
  build(pphynodes, start, maxBranches);

#if _DEBUG_SPANNING_TREE_
  // print this node and children
  for (int i=0; i < children.size()-1; i++) {
    std::ostringstream oss;
    for (Iterator j=children[i]; j != children[i+1]; j++) {
      if (j == children[i]) oss << "[" << CkMyNode() << "] subtree " << *j << ": ";
      else oss << *j << " ";
    }
    CkPrintf("%s\n", oss.str().c_str());
  }
#endif
  return (children.size() - 1);
}

template <typename Iterator>
void ST_RecursivePartition<Iterator>::initPhyNodes(Iterator start, Iterator end,
                                                   std::vector<PhyNode> &phynodes) const
{
#if _DEBUG_SPANNING_TREE_
  int rootPhyNodeId;
  if (nodeTree) rootPhyNodeId = CmiPhysicalNodeID(CmiNodeFirst(*start));
  else rootPhyNodeId = CmiPhysicalNodeID(*start);   // contains pes
  CkPrintf("[%d] Root phynode is %d\n", CkMyNode(), rootPhyNodeId);
#endif

  const int numNodes = std::distance(start, end);
  phynodes.reserve(std::min(CmiNumPhysicalNodes(), numNodes));
  intMap phyNodeMap;
  int last = -1;
  for (Iterator i=start; i != end; i++) {
    int n = *i;
    int pe = n;
    if (nodeTree) pe = CmiNodeFirst(n);
    int phyNodeId = CmiPhysicalNodeID(pe);
#if _DEBUG_SPANNING_TREE_
    if (phyNodeId == rootPhyNodeId) CkPrintf("[%d] Node %d is in root phynode\n", CkMyNode(), n);
#endif
    PhyNode *phyNode; // phynode of node n
    if (preSorted) {
      if (phyNodeId != last) {
        phynodes.push_back(PhyNode(phyNodeId,pe,tmgr));
        last = phyNodeId;
      }
      phyNode = &(phynodes.back());
    } else {
      intMap::iterator it = phyNodeMap.find(phyNodeId);
      if (it == phyNodeMap.end()) {
        phynodes.push_back(PhyNode(phyNodeId,pe,tmgr));
        phyNodeMap[phyNodeId] = int(phynodes.size()-1);
        phyNode = &(phynodes.back());
      } else {
        phyNode = &(phynodes[it->second]);
      }
    }
    phyNode->addNode(n);
  }

#if _DEBUG_SPANNING_TREE_
  CkPrintf("%d physical nodes:\n", int(phynodes.size()));
  for (int i=0; i < phynodes.size(); i++) phynodes[i].print();
#endif
#if XE6_TOPOLOGY
  translateCoordinates(phynodes);
#endif
}

template <typename Iterator>
void ST_RecursivePartition<Iterator>::withinPhyNodeTree(PhyNode &rootPhyNode, int bfactor, Iterator &pos)
{
  if (rootPhyNode.size() == 1) return; // only one element in physical node

  std::vector<int> nodes; // nodes in this phynode (root is ignored)
  std::map<int, std::vector<int>> nodePes; // PEs in each node (used when building PE tree)
  if (nodeTree) nodes.assign(rootPhyNode.nodes.begin()+1, rootPhyNode.nodes.end());
  else {
    // group PEs into nodes
    for (int i=1; i < rootPhyNode.size(); i++) {
      int pe = rootPhyNode.getNode(i);
      nodePes[CkNodeOf(pe)].push_back(pe);
    }
    std::map<int, std::vector<int>>::iterator it;
    for (it = nodePes.begin(); it != nodePes.end(); it++) nodes.push_back(it->first);
  }

  const int numNodes = nodes.size();
  if (!nodeTree && (numNodes == 1)) {
    // make all PEs in node direct children
    std::vector<int> &pes = nodePes.begin()->second;
    for (int i=0; i < pes.size(); i++) {
      children.push_back(pos);
      *pos = pes[i]; pos++;
    }
  } else {
    int numChildren = std::min(bfactor, numNodes);
    int partSize = numNodes / numChildren, parts=0;
    for (std::vector<int>::iterator i=nodes.begin(); parts < numChildren; i += partSize, parts++) {
      children.push_back(pos);
      std::vector<int>::iterator end;
      if (parts == numChildren-1) end = nodes.end();
      else end = i + partSize;
      for (std::vector<int>::iterator j=i; j != end; j++) {
        int n = *j;
        if (!nodeTree) {
          std::vector<int> &pes = nodePes[n];
          for (int k=0; k < pes.size(); k++) { *pos = pes[k]; pos++; }
        } else {
          *pos = n; pos++;
        }
      }
    }
  }
}

template <typename Iterator>
void ST_RecursivePartition<Iterator>::build(std::vector<PhyNode*> &phyNodes,
                                            Iterator start,
                                            unsigned int maxBranches)
{
  typename ST_RecursivePartition<Iterator>::PhyNode *rootPhyNode = phyNodes[0];
  children.reserve(rootPhyNode->size() + maxBranches); // reserve for max number of children

  Iterator pos = start+1;
  withinPhyNodeTree(*rootPhyNode, maxBranches, pos);

  // TODO another option, don't know if better, is if
  // I'm the root node of a phynode (and phynodes.size() > 1), only have one other node
  // in my phynode as direct child, and have that child direct-send to every other
  // node in the phynode. This would be an easy change.

  if (phyNodes.size() == 1) {
    children.push_back(pos);
    return;
  }

  // this will partition the nodes in phyNodes, by reorganizing the list.
  // phyNodeChildren will point to where each partition starts
  std::vector<int> phyNodeChildren;
  phyNodeChildren.reserve(maxBranches+1);
  partition(phyNodes, 1, phyNodes.size(), maxBranches, phyNodeChildren);
  phyNodeChildren.push_back(phyNodes.size());
  if (tmgr->haveTopologyInfo())
    // choose root phynode in each subtree (closest one to top-level root phynode), put at beginning
    chooseSubtreeRoots(phyNodes, phyNodeChildren);

  // store result as subtrees of nodes
  for (int i=0; i < phyNodeChildren.size() - 1; i++) {
    children.push_back(pos);
    for (int j=phyNodeChildren[i]; j < phyNodeChildren[i+1]; j++) {  // for each phynode in subtree
      for (int k=0; k < phyNodes[j]->size(); k++) {    // for each node in phynode
        *pos = phyNodes[j]->getNode(k);
        pos++;
      }
    }
  }
  children.push_back(pos);
}

/**
 * phyNodes is list of phyNodes, grouped by subtrees (rootPhyNode in position 0)
 * phyNodeChildren contains the indices (in phyNodes) of first node of each subtree
 */
template <typename Iterator>
void ST_RecursivePartition<Iterator>::chooseSubtreeRoots(std::vector<PhyNode*> &phyNodes,
                                                         std::vector<int> &children) const
{
  for (int i=0; i < children.size() - 1; i++) { // for each subtree
    int start = children[i];  // subtree start
    int minDistance = INT_MAX;
    int closestIdx = -1;
    for (int j=start; j < children[i+1]; j++) { // for each phynode in subtree
      int d = phyNodes[0]->distance(*phyNodes[j], tmgr);
      if (d < minDistance) {
        minDistance = d;
        closestIdx = j;
      }
    }
#if _DEBUG_SPANNING_TREE_
    if (CkMyNode() == 0) CkPrintf("Subtree %d, closest phynode to root is %d, distance=%d\n", i, phyNodes[closestIdx]->id, minDistance);
#endif
    // make closest one the root
    std::swap(phyNodes[start], phyNodes[closestIdx]);
  }
}

/// recursive partitioning of phynodes into numPartitions
template <typename Iterator>
void ST_RecursivePartition<Iterator>::partition(std::vector<PhyNode*> &nodes,
                                      int start, int end, int numPartitions,
                                      std::vector<int> &children) const
{
#if _DEBUG_SPANNING_TREE_
    CkPrintf("Partitioning into at most %d parts, phynodes [", numPartitions);
    for (int i=start; i < end; i++) CkPrintf("%d ", nodes[i]->id);
    CkPrintf("]\n");
#endif
  int numNodes = end - start;
  if ((numPartitions > 1) && (numNodes > 1)) {
    // further partitioning is needed and there are nodes left to partition
    if (numPartitions % 3 == 0) trisect(nodes, start, end, numPartitions, children);
    else bisect(nodes, start, end, numPartitions, children);
  } else if ((numPartitions >= 1) && (numNodes >= 1)) {
    // just register the remaining node(s) as a sub-tree
    children.push_back(start);
  } else if (numNodes == 0) {
    // there are no nodes left, do nothing
  } else if ((numNodes >= 0) && (numPartitions == 0)) {
    // if there are nodes remaining but no partitions to put them in
    CkAbort("\nThere are nodes left but no remaining partitions to put them in.");
  } else {
    // fall through case. Should never get here unless something is wrong
    CkAbort("\nPartitioning fell through to the default case (which it never should). Check the logic in this routine.");
  }
}

template <typename Iterator>
void ST_RecursivePartition<Iterator>::bisect(std::vector<PhyNode*> &nodes,
                                   int start, int end, int numPartitions,
                                   std::vector<int> &children) const
{
  const int numNodes = end - start;
  int median = start + (numNodes / 2);
  if (tmgr->haveTopologyInfo()) {
    // Find the dimension along which to bisect the bounding box
    int maxSpreadDim = maxSpreadDimension(nodes,start,end);
    // Bisect the vertex list at the median element
    typename std::vector<PhyNode*>::iterator itr = nodes.begin();
    std::nth_element(itr+start, itr+median, itr+end, typename ST_RecursivePartition::PhyNodeCompare(maxSpreadDim));
#if _DEBUG_SPANNING_TREE_
    CkPrintf("Bisecting, maxSpreadDim=%d\n", maxSpreadDim);
#endif
  }
  // Partition the two pieces further
  int numLeft = numPartitions/2;
  partition(nodes, start, median, numLeft, children);
  partition(nodes, median, end, numPartitions - numLeft, children);
}

template <typename Iterator>
void ST_RecursivePartition<Iterator>::trisect(std::vector<PhyNode*> &nodes,
                                   int start, int end, int numPartitions,
                                   std::vector<int> &children) const
{
  const int numNodes = end - start;
  /// Pin the location of the 1/3 and 2/3 elements
  int oneThird = start + (numNodes / 3);
  int twoThird = oneThird + (numNodes / 3);
  if (tmgr->haveTopologyInfo()) {
    int maxSpreadDim = maxSpreadDimension(nodes,start,end);
    typename std::vector<PhyNode*>::iterator itr = nodes.begin();
    std::nth_element(itr+start,    itr+oneThird, itr+end, typename ST_RecursivePartition::PhyNodeCompare(maxSpreadDim));
    std::nth_element(itr+oneThird, itr+twoThird, itr+end, typename ST_RecursivePartition::PhyNodeCompare(maxSpreadDim));
#if _DEBUG_SPANNING_TREE_
    CkPrintf("Trisecting, maxSpreadDim=%d\n", maxSpreadDim);
#endif
  }
  /// Partition the three pieces further
  int numLeft = numPartitions/3;
  partition(nodes, start,    oneThird, numLeft, children);
  partition(nodes, oneThird, twoThird, numLeft, children);
  partition(nodes, twoThird, end,      numLeft, children);
}

template <typename Iterator>
int ST_RecursivePartition<Iterator>::maxSpreadDimension(std::vector<PhyNode*> &nodes,
                                                        int start, int end) const
{
  const int nDims = tmgr->getNumDims();
  if (!tmgr->haveTopologyInfo() || (nDims <= 1)) return 0;

  std::vector<std::bitset<DIM_SET_SIZE> > used(nDims);
  for (int i=start; i < end; i++) {
    PhyNode *n = nodes[i];
    for (int j=0; j < nDims; j++) used[j].set(n->coords[j]);
  }
  int max_spread = -1;
  int max_spread_dim = -1;
  for (int i=0; i < nDims; i++) {
    int c(used[i].count());
    if (c > max_spread) {
      max_spread = c;
      max_spread_dim = i;
    }
  }
  return max_spread_dim;
}

#if XE6_TOPOLOGY

inline static int modulo(int k, int n) { return ((k %= n) < 0) ? k+n : k; }

/**
 * Translate coordinates of phynodes such that the max spread in each dimension
 * is equal (or closer) to the number of coordinates actually used in that dimension.
 * In other words, "moves" all phynodes by some translation offset so that their bounding box
 * includes minimum number of coordinates, which implies that adjacent nodes won't go through torus edges.
 * Works for any number of dimensions (N-d torus)
 */
template <typename Iterator>
void ST_RecursivePartition<Iterator>::translateCoordinates(std::vector<PhyNode> &nodes) const
{
  const int nDims = tmgr->getNumDims();
  std::vector<std::bitset<DIM_SET_SIZE> > usedCoordinates(nDims);
  std::vector<int> max_coord(nDims, -1);
  std::vector<int> min_coord(nDims, INT_MAX);
  std::vector<int> maxSpread(nDims);
  std::vector<int> gapCenter(nDims, -1);
  std::vector<int> dimSizes(nDims);
  for (int i=0; i < nDims; i++) dimSizes[i] = tmgr->getDimSize(i);

  for (int i=0; i < nodes.size(); i++) {
    PhyNode &n = nodes[i];
    for (int j=0; j < nDims; j++) {
      int c = n.coords[j];
      usedCoordinates[j].set(c);
      if (c > max_coord[j]) max_coord[j] = c;
      if (c < min_coord[j]) min_coord[j] = c;
    }
  }
  for (int i=0; i < nDims; i++) {
    maxSpread[i] = max_coord[i] - min_coord[i] + 1; // store max spread of each dimension
    int sum = 0, nbUnusedCoords = 0;
    for (int j=0; j < dimSizes[i]; j++) {
      if (!usedCoordinates[i].test(j)) {  // j coordinate not used by any element
        sum += j;
        nbUnusedCoords++;
      }
    }
    if (nbUnusedCoords > 0) gapCenter[i] = sum / nbUnusedCoords;
  }

#if _DEBUG_SPANNING_TREE_
  if (CkMyNode() == 0) {
    CkPrintf("Used coordinates in each dimension:\n");
    for (int i=0; i < nDims; i++) {
      CkPrintf("%d: ", i);
      for (int j=0; j < dimSizes[i]; j++) if (usedCoordinates[i].test(j)) CkPrintf("%d ", j);
      CkPrintf(", %d\n", int(usedCoordinates[i].count()));
    }
    CkPrintf("Max,min coord in each dimension:\n");
    for (int i=0; i < nDims; i++) CkPrintf("%d: %d %d\n", i, max_coord[i], min_coord[i]);
    CkPrintf("Gap center for each dimension:\n");
    for (int i=0; i < nDims; i++) CkPrintf("%d: %d\n", i, gapCenter[i]);
    CkPrintf("Max spread for each dimension:\n");
    for (int i=0; i < nDims; i++) CkPrintf("%d: %d\n", i, maxSpread[i]);
  }
#endif

  for (int i=0; i < nDims; i++) { // find best translation offset for each dimension
    if (maxSpread[i] == int(usedCoordinates[i].count())) continue; // nothing to correct for this dimension

#if _DEBUG_SPANNING_TREE_
    if (CkMyNode() == 0) CkPrintf("Going to attempt to correct coordinates on dimension %d\n", i);
#endif

    // choose direction of unused coordinates to finish faster
    int direction = 1;  // go "right"
    if (gapCenter[i] < dimSizes[i]/2) direction = -1;   // go "left"
#if _DEBUG_SPANNING_TREE_
    if (CkMyNode() == 0) CkPrintf("Chose direction %d\n", direction);
#endif

    // we're going to attempt to minimize the max spread in dimension i
    int bestMaxSpread = maxSpread[i];
    int bestOffset=0;
    for (int m=0; ; m++) {
      // apply offset of 'm' in 'direction' to all nodes
      int max_coord = -1;
      int min_coord = INT_MAX;
      for (int j=0; j < nodes.size(); j++) {
        int &x = nodes[j].coords[i];
        x += direction;
        if (x >= dimSizes[i]) x = 0;
        else if (x < 0) x = dimSizes[i] - 1;
        if (x > max_coord) max_coord = x;
        if (x < min_coord) min_coord = x;
      }
      // evaluate max spread with new offset
      int maxSpread_new = max_coord - min_coord + 1;
#if _DEBUG_SPANNING_TREE_
      if (CkMyNode() == 0) CkPrintf("%d %d\n", m, maxSpread_new);
#endif
      if (maxSpread_new == int(usedCoordinates[i].count())) {
#if _DEBUG_SPANNING_TREE_
        if (CkMyNode() == 0) CkPrintf("FIXED after %d movements\n", m+1);
#endif
        break;
      } else if (maxSpread_new < bestMaxSpread) {
        bestMaxSpread = maxSpread_new;
        bestOffset = m;
      }
      if (m == dimSizes[i] - 2) {
        // did max number of possible movements (another movement would return us to original
        // coordinates/offset), exit loop
        if (maxSpread_new > bestMaxSpread) {
          for (int j=0; j < nodes.size(); j++) {
            // roll back to bestOffset
            int &x = nodes[j].coords[i];
            x += ((m - bestOffset)*-direction);
            x = modulo(x, dimSizes[i]);
          }
        }
#if _DEBUG_SPANNING_TREE_
        if ((CkMyNode() == 0) && (bestMaxSpread < maxSpread[i])) CkPrintf("Improved to %d max spread\n", bestMaxSpread);
#endif
        break;   // we're done correcting in this dimension
      }
    }
  }
}
#endif

template class ST_RecursivePartition<int*>;
template class ST_RecursivePartition<std::vector<int>::iterator>;

// ------------------------------------------------------------------
typedef std::vector<int>::iterator TreeIterator;

void getNeighborsTopoTree_R(TreeIterator start, TreeIterator end, int myElem, int prevLvlParent,
                            bool nodeTree, unsigned int bfactor, CmiSpanningTreeInfo &t)
{
  ST_RecursivePartition<TreeIterator> tb(nodeTree, prevLvlParent != -1);
  int numSubtrees = tb.buildSpanningTree(start, end, std::min(bfactor, (unsigned int)std::distance(start,end)-1));
  if (myElem == *start) {
    // I am the root of this subtree so we're done (collect my children and return)
    t.parent = prevLvlParent;
    if (numSubtrees > 0) t.children = (int*)malloc(sizeof(int)*numSubtrees);
    t.child_count = numSubtrees;
    for (int i=0; i < numSubtrees; i++) t.children[i] = *tb.begin(i);
    return;
  }
  // find in which subtree myElem is in and recursively continue on only that subtree
  for (int i=0; i < numSubtrees; i++) {
    TreeIterator subtreeStart = tb.begin(i), subtreeEnd = tb.end(i);
    TreeIterator f = std::find(subtreeStart, subtreeEnd, myElem);
    if (f != subtreeEnd) {
      getNeighborsTopoTree_R(subtreeStart, subtreeEnd, myElem, *start, nodeTree, bfactor, t);
      break;
    }
  }
}

/**
 * Obtain parent and children of 'myNode' in tree rooted at 'rootNode', using
 * ST_RecursivePartition algorithm. Spanning tree assumed to cover all nodes.
 */
void getNodeNeighborsTopoTree(int rootNode, int myNode, CmiSpanningTreeInfo &t, unsigned int bfactor)
{
  std::vector<int> nodes;
  nodes.reserve(CkNumNodes());
  nodes.push_back(rootNode);
  for (int i=0; i < CkNumNodes(); i++) {
    if (i == rootNode) continue;
    nodes.push_back(i);
  }
  getNeighborsTopoTree_R(nodes.begin(), nodes.end(), myNode, -1, true, bfactor, t);
}

/// same as above but for processors
void getProcNeighborsTopoTree(int rootPE, int myPE, CmiSpanningTreeInfo &t, unsigned int bfactor)
{
  std::vector<int> pes;
  pes.reserve(CkNumPes());
  pes.push_back(rootPE);
  for (int i=0; i < CkNumPes(); i++) {
    if (i == rootPE) continue;
    pes.push_back(i);
  }
  getNeighborsTopoTree_R(pes.begin(), pes.end(), myPE, -1, false, bfactor, t);
}

typedef std::unordered_map<int,CmiSpanningTreeInfo*> TreeInfoMap;

static TreeInfoMap trees;
CmiNodeLock _treeLock;

CmiSpanningTreeInfo *ST_RecursivePartition_getTreeInfo(int root) {
  if (trees.size() == 0) {
    _treeLock = CmiCreateLock();
#if CMK_ERROR_CHECKING
    if (CkMyRank() != 0) CkAbort("First call to getTreeInfo has to be by rank 0");
#endif
  }
  CmiLock(_treeLock);
  TreeInfoMap::iterator it = trees.find(root);
  if (it != trees.end()) {
    CmiSpanningTreeInfo *t = it->second;
    CmiUnlock(_treeLock);
    return t;
  } else {
    CmiSpanningTreeInfo *t = new CmiSpanningTreeInfo;
    t->children = NULL;
    trees[root] = t;
    getNodeNeighborsTopoTree(root, CkMyNode(), *t, 4);
    CmiUnlock(_treeLock);
    return t;
  }
}

void get_topo_tree_nbs(int root, int *parent, int *child_count, int **children) {
  CmiSpanningTreeInfo *t = ST_RecursivePartition_getTreeInfo(root);
  *parent = t->parent;
  *child_count = t->child_count;
  *children = t->children;
}
