#include "charm++.h"

#ifndef TREE_STRATEGY_TOPO_UNAWARE
#define TREE_STRATEGY_TOPO_UNAWARE
namespace topo {

/// Implementation artifacts shouldnt pollute topo::
namespace impl {
    template <typename Iterator>
    SpanningTreeVertex* buildNextGen_topoUnaware(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches);
}

/** A concrete tree builder that is NOT topology aware. 
 *
 * Randomly selects child vertices and sub-trees etc. It should hence be, resource-wise, quite 
 * cheap. Use in situations when topology info is not available or a topo-aware tree build is 
 * costly (eg. early generations of really large multicast trees). 
 *
 * Technically, you are supposed to pass in a container of PEs or SpanningTreeVertices. However, 
 * since this class simply chops up the indices and makes no assumptions on the data in the container, 
 * it is possible to abuse this and randomly divvy up any container across numBranches.
 */
template <typename Iterator,typename ValueType = typename std::iterator_traits<Iterator>::value_type>
class SpanningTreeStrategy_topoUnaware: public SpanningTreeStrategy<Iterator>
{
    public:
        inline virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        { return impl::buildNextGen_topoUnaware(firstVtx,beyondLastVtx,maxBranches); }
};



/** Partial specialization when input is a container of SpanningTreeVertices. 
 *
 * Simply stores the results in the parent vertex (in the container) too 
 */
template <typename Iterator>
class SpanningTreeStrategy_topoUnaware<Iterator,SpanningTreeVertex>: public SpanningTreeStrategy<Iterator> 
{
    public:
        inline virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        {
            /// Clear any existing list of children
            (*firstVtx).childIndex.clear();
            /// Build the next generation
            SpanningTreeVertex *parent = impl::buildNextGen_topoUnaware(firstVtx,beyondLastVtx,maxBranches);
            /// Copy the results into the parent vertex too
            *firstVtx = *parent;
            /// Return the results
            return parent;
        }
};


namespace impl {
    template <typename Iterator>
    SpanningTreeVertex* buildNextGen_topoUnaware(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches)
    {
        // if (maxBranches < 1) throw;
        CkAssert(maxBranches >= 1);
        /// Check validity and ranges etc.
        const int numDescendants = std::distance(firstVtx,beyondLastVtx) - 1;
        // if (numDescendants < 0) throw;
        CkAssert(numDescendants >= 0);

        /// Create the output data structure
        SpanningTreeVertex *parent = new SpanningTreeVertex(*firstVtx);

        /// Compute the number of vertices in each branch
        int numInSubTree = numDescendants / maxBranches;
        int remainder    = numDescendants % maxBranches;

        /// Push the appropriate relative distances (from the tree root) into the container of child indices
        for (int i=0, indx=1; (i<maxBranches && indx<=numDescendants); i++, indx += numInSubTree)
        {
            parent->childIndex.push_back(indx);
            /// Distribute any remainder vertices as evenly as possible amongst all the branches 
            if (remainder-- > 0) indx++;
        }

        /// Return the output structure
        return parent;
    }
}

} // end namespace topo
#endif // TREE_STRATEGY_TOPO_UNAWARE

