#include "charm++.h"
#include <algorithm>

#ifndef TREE_STRATEGY_NODE_AWARE_MIN_BYTES
#define TREE_STRATEGY_NODE_AWARE_MIN_BYTES
namespace topo {

/// Implementation artifacts shouldnt pollute topo::
namespace impl {
    template <typename Iterator>
    SpanningTreeVertex* buildNextGen_nodeAware_minBytes(const vtxType parentPE, const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2);
}

/** A concrete tree builder that is aware of cpu topology (ie, node aware) while constructing
 * spanning trees.
 *
 * Minimizes the total bytes on the network by constructing separate sub-tree(s) for same-node PEs.
 * Uses the cpuTopology API defined in charm. Hence, should be node-aware in all environments.
 *
 * Note that this node awareness has nothing to do with charm smp builds. Rather, its about
 * minimizing the number of messages that arrive or leave a single physical machine node for
 * a multicast / reduction.
 */
template <typename Iterator,typename ValueType = typename std::iterator_traits<Iterator>::value_type>
class SpanningTreeStrategy_nodeAware_minBytes: public SpanningTreeStrategy<Iterator>
{
    public:
        inline virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        { return impl::buildNextGen_nodeAware_minBytes(*firstVtx, firstVtx,beyondLastVtx,maxBranches); }
};



/** Partial specialization when input is a container of SpanningTreeVertices.
 *
 * Exactly the same as the default implementation, except that this stores the results in the parent
 * vertex (in the container) too
 */
template <typename Iterator>
class SpanningTreeStrategy_nodeAware_minBytes<Iterator,SpanningTreeVertex>: public SpanningTreeStrategy<Iterator>
{
    public:
        inline virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        {
            /// Clear any existing list of children
            (*firstVtx).childIndex.clear();
            /// Build the next generation
            SpanningTreeVertex *parent = impl::buildNextGen_nodeAware_minBytes((*firstVtx).id,firstVtx,beyondLastVtx,maxBranches);
            /// Copy the results into the parent vertex too
            *firstVtx = *parent;
            /// Return the results
            return parent;
        }
};


namespace impl {
    /**
     * Separates on-node PEs into separate sub-tree(s) and then builds a dumb spanning tree for the off-node PEs
     */
    template <typename Iterator>
    SpanningTreeVertex* buildNextGen_nodeAware_minBytes(const vtxType parentPE, const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches)
    {
        /// ------------- Obtain a list of all PEs on this physical machine node -------------
        CkAssert(parentPE < CkNumPes() );
        int numOnNode, *pesOnNode;
        CmiGetPesOnPhysicalNode(CmiPhysicalNodeID(parentPE),&pesOnNode,&numOnNode);

        /// ------------- Identify the PEs that are on the same node as the tree root ------------- 
        /// The number of local destinations (same node PEs) and the number of branches that will span these destinations
        int numLocalDestinations = 0, numLocalBranches = 0;
        /// The object that will hold the final results
        SpanningTreeVertex *parent = 0;
        /// 
        Iterator itr = firstVtx, lastLocal = firstVtx;

        /// Scan the tree members until we identify all possible same node PEs or run out of tree members
        while ( numLocalDestinations < numOnNode -1 && itr != beyondLastVtx)
        {
            /// Try to find the next same-node PE
            itr = std::find_first_of(++itr,beyondLastVtx,pesOnNode,pesOnNode + numOnNode
#if CMK_FIND_FIRST_OF_PREDICATE
				     , vtxEqual()
#endif
				     );

            /// If such a PE was found...
            if (itr != beyondLastVtx)
            {
                numLocalDestinations++;
                if (itr != ++lastLocal)
                    std::iter_swap(itr,lastLocal);
            }
        }

        /// ------------- Construct a generation in the tree with local sub-tree(s) if necessary ------------- 
        /// If there are any local vertices at all
        if (numLocalDestinations > 0)
        {
            /// Determine how many branches can be used to span the local destinations
            int numRemoteDestinations = std::distance(firstVtx,beyondLastVtx) -1 - numLocalDestinations; ///< @warning: This is O(treeSize) for Iterator != random iterator
            numLocalBranches = (numRemoteDestinations >= maxBranches)? 1 : (maxBranches - numRemoteDestinations);

            /// Distribute the local destination vertices amongst numLocalBranches branches
            Iterator beyondLastLocal = lastLocal;
            parent = buildNextGen_topoUnaware(firstVtx,++beyondLastLocal,numLocalBranches);

            /// Construct a topo-unaware tree for the rest (off-node PEs) of the vertices
            SpanningTreeVertex *remoteSubTrees = buildNextGen_topoUnaware(lastLocal,beyondLastVtx,maxBranches); ///< Abuse the interface by faking lastLocal as the tree root

            /// Append the remote sub-tree info to the result object
            for (int i=0, n=remoteSubTrees->childIndex.size(); i< n; i++)
                parent->childIndex.push_back(remoteSubTrees->childIndex[i] + numLocalDestinations);
            delete remoteSubTrees;
        }
        else
            parent = buildNextGen_topoUnaware(firstVtx,beyondLastVtx,maxBranches);

        /// Return the output structure
        return parent;
    }
} // end namespace impl

} // end namespace topo
#endif // TREE_STRATEGY_NODE_AWARE_MIN_BYTES

