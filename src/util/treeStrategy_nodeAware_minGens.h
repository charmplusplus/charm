#include "charm++.h"
#include <algorithm>

#ifndef TREE_STRATEGY_NODE_AWARE_MIN_GENS
#define TREE_STRATEGY_NODE_AWARE_MIN_GENS
namespace topo {

/// Implementation artifacts shouldnt pollute topo::
namespace impl {
    template <typename Iterator>
    SpanningTreeVertex* buildNextGen_nodeAware_minGens(const vtxType parentPE, const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2);
}

/** A concrete tree builder that is aware of cpu topology (ie, node aware) while constructing
 * spanning trees.
 *
 * First generates a balanced spanning tree and then promotes any PEs on the same node
 * (up to a maximum of maxBranches) as direct children. Uses the cpuTopology API defined in
 * charm. Hence, should be node-aware in all environments.
 *
 * The algorithm prioritizes balance over minimizing inter-node traffic. By ensuring that the 
 * tree is as balanced as it can be, we can reduce the number of generations taken to span all
 * the tree vertices.
 */
template <typename Iterator,typename ValueType = typename std::iterator_traits<Iterator>::value_type>
class SpanningTreeStrategy_nodeAware_minGens: public SpanningTreeStrategy<Iterator>
{
    public:
        inline virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        { return impl::buildNextGen_nodeAware_minGens(*firstVtx, firstVtx,beyondLastVtx,maxBranches); }
};



/** Partial specialization for the scenario of a container of SpanningTreeVertices.
 *
 * Exactly the same as the default implementation, except that this stores the results in the parent
 * vertex (in the container) too
 */
template <typename Iterator>
class SpanningTreeStrategy_nodeAware_minGens<Iterator,SpanningTreeVertex>: public SpanningTreeStrategy<Iterator>
{
    public:
        inline virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        {
            /// Clear any existing list of children
            (*firstVtx).childIndex.clear();
            /// Build the next generation
            SpanningTreeVertex *parent = impl::buildNextGen_nodeAware_minGens((*firstVtx).id,firstVtx,beyondLastVtx,maxBranches);
            /// Copy the results into the parent vertex too
            *firstVtx = *parent;
            /// Return the results
            return parent;
        }
};


namespace impl {

    #if CMK_FIND_FIRST_OF_PREDICATE
    /** Hack for PGI and Fujitsu STLport-using compilers because of their incorrect implementation of an STL routine.
     *
     * std::find_first_of(inpItr,inpItr,fwdItr,fwdItr) should use operator== for comparison.
     * However, PGI's implementation just instantiates a std::equal_to<T> object with
     * T=inpItr::value_type and delegates the work to std::find_first_of(inpItr,inpItr,fwdItr,fwdItr,BinPred).
     *
     * The implementation makes the wrong assumption that inpItr::value_type is the same as
     * fwdItr::value_type and hence simply instantiates std::equal_to with one of their value_types.
     * This breaks when inpItr and fwdItr have different value_types but the programmer has overloaded
     * operator== to allow comparisons between compare inpItr::value_type and fwdItr::value_type.
     *
     * We sidestep the issue by supplying our own binary predicate when we use std::find_first_of when
     * charm is built with a PGI compiler.
     * @note: This hack is not needed from pgCC 10.1 onwards
     */
    class vtxEqual
    {
        public:
            inline bool operator() (const SpanningTreeVertex &vtx, const int num) { return vtx.id == num; }
            inline bool operator() (const int num, const SpanningTreeVertex &vtx) { return vtx.id == num; }
            inline bool operator() (const int a, const int b) { return a == b; }
    };
    #endif

    /**
     * Common implementation for all value_types. The differences (obtaining the parent PE) for
     * different value_types are handled at the call location itself.
     *
     * Simply generates a balanced tree in a topo unaware manner and then promotes any same node PEs as direct children
     */
    template <typename Iterator>
    SpanningTreeVertex* buildNextGen_nodeAware_minGens(const vtxType parentPE, const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches)
    {
        /// Construct the next generation in a blind, aware-of-nothing manner
        SpanningTreeVertex *parent = impl::buildNextGen_topoUnaware(firstVtx,beyondLastVtx,maxBranches);

        /// Obtain a list of all PEs on this physical machine node
        CkAssert(parentPE < CkNumPes() );
        int numOnNode, *pesOnNode;
        CmiGetPesOnPhysicalNode(CmiPhysicalNodeID(parentPE),&pesOnNode,&numOnNode);

        /// Find any (upto maxBranches) tree members that are on the same node and make them the direct children
        if (numOnNode > 0 && parent->childIndex.size() > 0)
        {
            Iterator itr = firstVtx;
            int brNum = 0, numBranches = parent->childIndex.size();

            /// Scan the tree members until we identify same node PEs for all direct children or run out of tree members
            while ( brNum < numBranches && itr != beyondLastVtx)
            {
                std::vector<int>::iterator isChild;
                /// Search the tree members for a PE on this node that is not already a direct child
                do {
		    itr = std::find_first_of(++itr,beyondLastVtx,pesOnNode,pesOnNode + numOnNode
#if CMK_FIND_FIRST_OF_PREDICATE
					     , vtxEqual()
#endif
			);
                    int dist = std::distance(firstVtx,itr);
                    /// Check if this vertex is already a direct child
                    isChild = std::find(parent->childIndex.begin(),parent->childIndex.end(),dist);
                } while (itr != beyondLastVtx && isChild != parent->childIndex.end());

                Iterator childLoc;
                int *pos;
                /// Search the child locations for the first slot that does not have a same-node PE
                do {
                    childLoc = firstVtx;
                    std::advance(childLoc,parent->childIndex[brNum]);
                    /// Check if this child is already on the same node
                    pos = std::find(pesOnNode,pesOnNode+numOnNode,*childLoc);
                } while( pos < (numOnNode+pesOnNode) && ++brNum < numBranches );

                /// If a same-node PE is found and a child slot without a same-node PE exists
                if (itr != beyondLastVtx && brNum < numBranches)
                {
                    std::iter_swap(itr,childLoc);
                    brNum++;
                }
            }
        }

        /// Return the output structure
        return parent;
    }
} // end namespace impl

} // end namespace topo
#endif // TREE_STRATEGY_NODE_AWARE_MIN_GENS

