#include <algorithm>
#include "charm++.h"

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

template <typename Iterator>
SpanningTreeVertex* SpanningTreeStrategy_3dTorus_minBytesHops<Iterator,SpanningTreeVertex>::buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches)
{
    /// If the parent vertex already has a(n older) list of children, clear it
    (*firstVtx).childIndex.clear();
    (*firstVtx).childIndex.reserve(maxBranches);

    /// Get a handle on a TopoManager. @note: Avoid this per-call instantiation cost by using an instance manager? Ideally, TopoManager should be a singleton.
    TopoManager aTopoMgr;
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
        aTopoMgr.rankToCoordinates( (*itr).id, (*itr).X[0], (*itr).X[1], (*itr).X[2], coreNum );
        /// If this is a not a local node (separated by the network from the tree root)
        if (numHops(*firstVtx,*itr) > 0)
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
    }

    /// Partition the remote-vertex bounding box into the remaining number of branches
    impl::TreeBoundingBoxOn3dTorus<Iterator> treePart;
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
        std::iter_swap(rangeStart,closestItr);
    }
    /// Return a copy of the parent in keeping with the generic interface
    return (new SpanningTreeVertex(*firstVtx) );
}

} // end namespace topo
#endif // TREE_STRATEGY_3D_TORUS_MIN_BYTES_HOPS

