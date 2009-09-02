#include "charm++.h"

#ifndef TREE_STRATEGY_TOPO_UNAWARE
#define TREE_STRATEGY_TOPO_UNAWARE
namespace topo {

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
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        {
            // if (maxBranches < 1) throw;
            CkAssert(maxBranches >= 1);
            /// Create the output data structure
            SpanningTreeVertex *parent = new SpanningTreeVertex(*firstVtx);
            /// Check validity and ranges etc.
            const int numDescendants = std::distance(firstVtx,beyondLastVtx) - 1;
            // if (numDescendants < 0) throw;
            CkAssert(numDescendants >= 0);
            /// Compute the number of vertices in each branch
            int numInSubTree = numDescendants / maxBranches; 
            int remainder    = numDescendants % maxBranches; 
            /// Push the appropriate indices into the container of child indices
            for (int i=0, indx=1; (i<maxBranches && indx<=numDescendants); i++, indx += numInSubTree)
            {
                parent->childIndex.push_back(indx);
                /// Distribute any remainder vertices as evenly as possible amongst all the branches 
                if (remainder-- > 0) indx++;
            }
            /// Return the output structure
            return parent;
        }
};




/** Partial specialization for the scenario of a container of SpanningTreeVertices. 
 *
 * Exactly the same as the default implementation, except that this stores the results in the parent vertex (in the container) too 
 */
template <typename Iterator>
class SpanningTreeStrategy_topoUnaware<Iterator,SpanningTreeVertex>: public SpanningTreeStrategy<Iterator> 
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        {
            // if (maxBranches < 1) throw;
            CkAssert(maxBranches >= 1);
            /// Check validity and ranges etc.
            const int numDescendants = std::distance(firstVtx,beyondLastVtx) - 1;
            // if (numDescendants < 0) throw;
            CkAssert(numDescendants >= 0);
            /// Clear any existing list of children
            (*firstVtx).childIndex.clear();
            /// Compute the number of vertices in each branch
            int numInSubTree = numDescendants / maxBranches; 
            int remainder    = numDescendants % maxBranches;
            /// Push the appropriate indices into the container of child indices
            for (int i=0, indx=1; (i<maxBranches && indx<=numDescendants); i++, indx += numInSubTree)
            {
                (*firstVtx).childIndex.push_back(indx); 
                /// Distribute any remainder vertices as evenly as possible amongst all the branches 
                if (remainder-- > 0) indx++;
            }
            /// Create the output data structure
            SpanningTreeVertex *parent = new SpanningTreeVertex(*firstVtx);
            /// Return the output structure
            return parent;
        }
};

} // end namespace topo
#endif // TREE_STRATEGY_TOPO_UNAWARE

