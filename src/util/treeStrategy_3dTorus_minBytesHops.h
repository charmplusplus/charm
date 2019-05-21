#include <algorithm>
#include "charm++.h"

#if __DEBUG_SPANNING_TREE_
#include <sstream>
#include <fstream>
#endif

#ifndef TREE_STRATEGY_3D_TORUS_MIN_BYTES_HOPS
#define TREE_STRATEGY_3D_TORUS_MIN_BYTES_HOPS
namespace topo {

/** A concrete tree builder for use on machines with a 3D Torus topology. Naturally, can also work
 * with 3D meshes (portions of the torus).
 *
 * Reduces the total number of bytes that reach the network (ie reduces inter-node traffic) by
 * building separate sub-trees to span intra-node PEs, and then reduces the total number of hops
 * across the whole tree. Hence, should be more effictive than the strategy that reduces hops alone,
 * but possibly at the expense of perfect balance in the spanning tree.
 *
 * Specialized and implemented only for data type in input container = vtxType / SpanningTreeVertex.
 * @note: If its a container of SpanningTreeVertices, the next gen info is stored in the parent
 * element and a copy of the parent is also returned.
 */
template <typename Iterator,typename ValueType = typename std::iterator_traits<Iterator>::value_type>
class SpanningTreeStrategy_3dTorus_minBytesHops: public SpanningTreeStrategy<Iterator>
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2) = 0;
};



/** Partial specialization when input is a container of SpanningTreeVertices
 */
template <typename Iterator>
class SpanningTreeStrategy_3dTorus_minBytesHops<Iterator,SpanningTreeVertex>: public SpanningTreeStrategy<Iterator>
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2);
};



/** Partial specialization when input is a container of vtxTypes.
 *
 * Simply builds a container of SpanningTreeVertices from the input data and delegates the actual
 * tree building to another specialization.
 */
template <typename Iterator>
class SpanningTreeStrategy_3dTorus_minBytesHops<Iterator,vtxType>: public SpanningTreeStrategy<Iterator>
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        {
            /// Create a container of SpanningTreeVertices from the input container and fill it with vertex IDs
            std::vector<SpanningTreeVertex> tree;
            for (Iterator itr = firstVtx; itr != beyondLastVtx; itr++)
                tree.push_back( SpanningTreeVertex(*itr) );
            /// Instantiate the real builder and let it build the next generation
            SpanningTreeStrategy_3dTorus_minBytesHops< std::vector<SpanningTreeVertex>::iterator > theRealBuilder;
            SpanningTreeVertex *result = theRealBuilder.buildNextGen(tree.begin(),tree.end(),maxBranches);
            /// Copy the reordered vertex indices back into the user's data structure
            int indx=0;
            for (Iterator itr = firstVtx; itr != beyondLastVtx; itr++)
            {
                *itr = tree[indx].id;
                indx++;
            }
            /// Return information about the parent vertex (which contains child indices etc)
            return result;
        }
};

} // end namespace topo


#include "TopoManager.h"

namespace topo {

#if __DEBUG_SPANNING_TREE_ && XE6_TOPOLOGY
void writeAllocTopoManager(const char *fileName) {
  std::ofstream outfile(fileName);
  outfile << "x y z t" << std::endl;
  int coords[4];
  for (int i=0; i < CkNumPes(); i++) {
      TopoManager_getPeCoordinates(i, coords);
      outfile << coords[0] << " " << coords[1] << " " << coords[2] << " " << std::endl;
  }
  outfile.close();
}

template <typename Iterator>
void writeCoordinatesPEList(const char *fileName, const Iterator start, const Iterator end) {
  std::ofstream outfile(fileName);
  outfile << "x y z t" << std::endl;
  for (Iterator itr = start; itr != end; itr++) {
    for (int j=0; j < 3; j++) outfile << (*itr).X[j] << " ";
    outfile << std::endl;
  }
  outfile.close();
}
#endif

template <typename Iterator>
SpanningTreeVertex* SpanningTreeStrategy_3dTorus_minBytesHops<Iterator,SpanningTreeVertex>::buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches)
{
    /// If the parent vertex already has a(n older) list of children, clear it
    (*firstVtx).childIndex.clear();
    (*firstVtx).childIndex.reserve(maxBranches);

    /// Get a handle on TopoManager
    TopoManager *aTopoMgr = TopoManager::getTopoManager();
    /// The number of vertices in the tree that require network traversal
    int numLocalDestinations = -1, numRemoteDestinations = 0;
    Iterator beyondLastLocal = firstVtx;

    /// Find the machine coordinates of each vertex and also collate the local destination vertices
    for (Iterator itr = firstVtx; itr != beyondLastVtx; itr++)
    {
        (*itr).X.reserve(3);
        (*itr).X.assign(3,0);
        int coreNum; ///< dummy var. Get and discard the core number
        ///@todo: If the machine coordinates are already stored in the vertices, do we want to find them again?
        aTopoMgr->rankToCoordinates( (*itr).id, (*itr).X[0], (*itr).X[1], (*itr).X[2], coreNum );
        /// If this is a not a local node (separated by the network from the tree root)
        if (!(*firstVtx).sameCoordinates(*itr))
            numRemoteDestinations++;
        else
        {
            numLocalDestinations++;
            /// Collate it near the top of the container
            if (itr != beyondLastLocal)
                std::iter_swap(beyondLastLocal,itr);
            /// Increment iterator to reflect new range of on-node vertices
            beyondLastLocal++;
        }
    }
    impl::TreeBoundingBoxOn3dTorus<Iterator> treePart;
#if XE6_TOPOLOGY
    if ((aTopoMgr->getDimNX() > MAX_TORUS_DIM_SIZE) ||
        (aTopoMgr->getDimNY() > MAX_TORUS_DIM_SIZE) ||
        (aTopoMgr->getDimNZ() > MAX_TORUS_DIM_SIZE)) {
      CkAbort("Torus dimension size larger than supported limit. Please increase limit");
    }
#if __DEBUG_SPANNING_TREE_
    if (CkMyPe() == 0) {
      writeAllocTopoManager("alloc_tmgr.txt");
      writeCoordinatesPEList("pe_coords.txt", firstVtx, beyondLastVtx);
      CkPrintf("numRemoteDestinations = %d\n", numRemoteDestinations);
      CkPrintf("numLocalDestinations = %d\n", numLocalDestinations);
    }
#endif
    treePart.translateCoordinates(firstVtx, beyondLastVtx, 3);
#if __DEBUG_SPANNING_TREE_
    if (CkMyPe() == 0) {
      writeCoordinatesPEList("pe_coords_translated.txt", firstVtx, beyondLastVtx);
    }
#endif
#endif

    /// The number of branches that can be devoted to local destinations (vertices on the same node)
    int numLocalBranches = 0;
    /// If there are any local vertices at all
    if (numLocalDestinations > 0)
    {
        numLocalBranches = (numRemoteDestinations >= maxBranches)? 1 : (maxBranches - numRemoteDestinations);
        /// Distribute the local destination vertices amongst numLocalBranches branches
        SpanningTreeVertex *localTree = impl::buildNextGen_topoUnaware(firstVtx,beyondLastLocal,numLocalBranches);
        /// Append the local tree info to the child info
        for (int i=0,n=localTree->childIndex.size(); i<n; i++)
            firstVtx->childIndex.push_back( localTree->childIndex[i] );
        /// Intermediate results no longer needed
        delete localTree;
#if __DEBUG_SPANNING_TREE_
        if (CkMyPe() == 0) {
          std::cout << "Local destinations are:" << std::endl;
          for (Iterator itr = firstVtx; itr != beyondLastLocal; itr++) std::cout << (*itr).id << " ";
          std::cout << std::endl;
        }
#endif
    }

    /// Partition the remote-vertex bounding box into the remaining number of branches
    treePart.partition(firstVtx,beyondLastLocal,beyondLastVtx,maxBranches);

    /// Identify the closest member in each remote branch and put it at the corresponding childIndex location
    for (int i=numLocalBranches, numChildren=(*firstVtx).childIndex.size(); i<numChildren; i++)
    {
        Iterator rangeStart = firstVtx;
        std::advance(rangeStart,(*firstVtx).childIndex[i]);
        Iterator rangeEnd   = firstVtx;
        if (i+1 == numChildren)
            rangeEnd = beyondLastVtx;
        else
            std::advance(rangeEnd, (*firstVtx).childIndex[i+1] );
        Iterator closestItr = pickClosest(*firstVtx,rangeStart,rangeEnd);
#if __DEBUG_SPANNING_TREE_
        if (CkMyPe() == 0) {
          CkPrintf("Closest PE in subtree %d is %d with distance %d\n", i, (*closestItr).id, numHops(*firstVtx,*closestItr));
        }
#endif
        std::iter_swap(rangeStart,closestItr);
    }
#if __DEBUG_SPANNING_TREE_
    // dump the tree
    char dbout[1024];
    int offset = 0;
    int coords[4];
    int numChildren = (*firstVtx).childIndex.size();
    TopoManager_getPeCoordinates(getProcID(*firstVtx), coords);
    offset += sprintf(dbout, "TREE: %d (%d,%d,%d) %d ", getProcID(*firstVtx), coords[0], coords[1], coords[2], numChildren);
    for (int i=0; i < numChildren; i++) {
      Iterator rangeStart = firstVtx;
      std::advance(rangeStart,(*firstVtx).childIndex[i]);
      int childPE = getProcID(*rangeStart);
      TopoManager_getPeCoordinates(childPE, coords);
      offset += sprintf(dbout + offset, "%d (%d,%d,%d) ", childPE, coords[0], coords[1], coords[2]);
    }
    sprintf(dbout + offset, "\n");
    CkPrintf(dbout);
#endif
    /// Return a copy of the parent in keeping with the generic interface
    return (new SpanningTreeVertex(*firstVtx) );
}

} // end namespace topo
#endif // TREE_STRATEGY_3D_TORUS_MIN_BYTES_HOPS

