#include <algorithm>
#include "charm++.h"

#ifndef TREE_STRATEGY_3D_TORUS
#define TREE_STRATEGY_3D_TORUS
namespace topo {

/** A concrete tree builder for use on machines with a 3D Torus topology. Naturally, can also work with 3D meshes (portions of the torus)
 *
 * Specialized and implemented only for data type in input container = vtxType / SpanningTreeVertex. 
 * @warning: If its a container of SpanningTreeVertices, the generation info is stored in the parent 
 * element and a pointer to the parent is returned. Do not free this returned pointer as it will 
 * mess up the input container.
 */
template <typename Iterator,typename ValueType = typename std::iterator_traits<Iterator>::value_type>
class SpanningTreeStrategy_3dTorus: public SpanningTreeStrategy<Iterator>
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2) = 0;
};



/** Partial specialization for scenario for a container of SpanningTreeVertices 
 */
template <typename Iterator>
class SpanningTreeStrategy_3dTorus<Iterator,SpanningTreeVertex>: public SpanningTreeStrategy<Iterator> 
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2);

    protected:
        /// Partition the range along the longest dimension into numPartitions parts
        void partition      (const Iterator parent, const Iterator start, const Iterator end, const int numPartitions);
        /// Bisect the range along the longest dimension
        void bisect         (const Iterator parent, const Iterator start, const Iterator end, const int numPartitions);
        /// Trisect the range along the longest dimension
        void trisect        (const Iterator parent, const Iterator start, const Iterator end, const int numPartitions);
        /// Pick the vertex closes to the parent in the given range
        Iterator pickClosest(const Iterator parent, const Iterator start, const Iterator end);
        /// Returns the dimension along which the bounding box of a range of vertices has the longest side
        int findMaxSpreadDimension (const Iterator start, const Iterator end);
        /// Return the number of hops (on the machine network) between two vertices in the tree
        int numHops (const Iterator vtx1, const Iterator vtx2);
        /// Configure a lessThan functor to compare vertices
        class lessThan
        {
            public:
                lessThan(const int _dim): dim(_dim) {}
                inline bool operator() (const SpanningTreeVertex &a, const SpanningTreeVertex &b)
                {
                    return (a.X[dim] < b.X[dim]);
                }
            private:
                const int dim;
        };
};



/** Partial specialization for scenario when a container of vtxTypes is input. 
 *
 * Simply builds a container of SpanningTreeVertices from the input data and delegates the actual 
 * tree building to another specialization.
 */
template <typename Iterator>
class SpanningTreeStrategy_3dTorus<Iterator,vtxType>: public SpanningTreeStrategy<Iterator> 
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        {
            /// Create a container of SpanningTreeVertices from the input container and fill it with vertex IDs
            std::vector<SpanningTreeVertex> tree;
            for (Iterator itr = firstVtx; itr != beyondLastVtx; itr++)
                tree.push_back( SpanningTreeVertex(*itr) );
            /// Instantiate the real builder and let it build the next generation
            SpanningTreeStrategy_3dTorus< std::vector<SpanningTreeVertex>::iterator > theRealBuilder;
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



template <typename Iterator>
inline void SpanningTreeStrategy_3dTorus<Iterator,SpanningTreeVertex>::partition(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions)
{
    /// Find the number of vertices in the range
    int numVertices = std::distance(start,end);
    /// If further partitioning is needed and there are vertices left to partition
    if ( (numPartitions > 1) && (numVertices > 1) )
    {
        if (numPartitions % 3 == 0)
            trisect(parent,start,end,numPartitions);
        else
            bisect (parent,start,end,numPartitions);
    }
    /// else, just register the remaining vertex(ices) as a sub-tree
    else if ( (numPartitions >= 1) && (numVertices >= 1) )
    {
        int indx = std::distance(parent,start);
        (*parent).childIndex.push_back(indx);
    }
    /// else if there are no vertices left, do nothing
    else if (numVertices == 0)
    {
    }
    /// else, if there are vertices remaining but no partitions to put them in
    else if ( (numVertices >= 0) && (numPartitions == 0) )
        CkAbort("\nThere are vertices left but no remaining partitions to put them in.");
    /// fall through case. Should never get here unless something is wrong
    else
        CkAbort("\nPartitioning fell through to the default case (which it never should). Check the logic in this routine.");
}



template <typename Iterator>
void SpanningTreeStrategy_3dTorus<Iterator,SpanningTreeVertex>::bisect(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions)
{
    /// Find the number of vertices in the range
    int numVertices = std::distance(start,end);
    /// Find the dimension along which to bisect the bounding box 
    int maxSpreadDim = findMaxSpreadDimension(start,end);
    /// Pin the location of the median element
    Iterator median = start;
    std::advance(median,numVertices/2);
    /// Bisect the vertex list at the median element
    std::nth_element( start, median, end, lessThan(maxSpreadDim) );
    /// Partition the two pieces as necessary
    int numLeft = numPartitions/2;
    partition(parent, start, median, numLeft);
    partition(parent, median, end, numPartitions - numLeft);
}



template <typename Iterator>
void SpanningTreeStrategy_3dTorus<Iterator,SpanningTreeVertex>::trisect(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions)
{
    /// If the number of partitions left really can be trisected, then do it
    if (numPartitions % 3 == 0)
    {
        /// Find the number of vertices in the range
        int numVertices = std::distance(start,end);
        /// Find the dimension along which to bisect the bounding box 
        int maxSpreadDim = findMaxSpreadDimension(start,end);
        /// Pin the location of the 1/3 and 2/3 elements
        Iterator oneThird = start;
        std::advance(oneThird,numVertices/3);
        Iterator twoThird = oneThird;
        std::advance(twoThird,numVertices/3);
        /// Trisect the vertex list at the median element
        std::nth_element( start,    oneThird, end, lessThan(maxSpreadDim) );
        std::nth_element( oneThird, twoThird, end, lessThan(maxSpreadDim) );
        /// Partition the three pieces further
        int numLeft = numPartitions/3;
        partition(parent, start,    oneThird, numLeft);
        partition(parent, oneThird, twoThird, numLeft);
        partition(parent, twoThird, end,      numLeft);
    }
    /// else simply call partition to let it handle things
    else
        partition(parent, start, end, numPartitions);
}



template <typename Iterator>
inline Iterator SpanningTreeStrategy_3dTorus<Iterator,SpanningTreeVertex>::pickClosest(const Iterator parent, const Iterator start, const Iterator end)
{
    Iterator itr     = start;
    Iterator closest = itr++;
    int      minHops = numHops(parent,closest); // aTopoMgr.getHopsBetweenRanks( parentPE, (*closest).id ); 
    /// Loop thro the range and identify the vertex closest to the parent
    for (; itr != end; itr++)
    {
        int hops = numHops(parent,itr); //aTopoMgr.getHopsBetweenRanks( parentPE, (*itr).id );
        if (hops < minHops)
        {
            closest = itr;
            minHops = hops;
        }
    }
    return closest;
}



template <typename Iterator>
int SpanningTreeStrategy_3dTorus<Iterator,SpanningTreeVertex>::findMaxSpreadDimension(const Iterator start, const Iterator end)
{
    int nDims = (*start).X.size();
    std::vector<int> min, max, spread; 
    min.reserve(nDims);
    max.reserve(nDims);
    spread.reserve(nDims);
    /// Find the min and max coordinates along each dimension of the bounding box
    min = max = (*start).X;
    for (Iterator itr = start; itr != end; itr++)
    {
        /// @todo: Assert that the dimensions of the coordinate vectors of the this vertex are the same as the parent's
        for (int i=0; i<nDims; i++)
        {
            if ( (*itr).X[i] < min[i] )
                min[i] = (*itr).X[i];
            if ( (*itr).X[i] > max[i] )
                max[i] = (*itr).X[i];
        }
    }
    /// Identify the dimension of the maximum spread in coordinates
    int maxSpread = abs(max[0] - min[0]);
    int maxSpreadDimension = 0;
    for (int i=1; i<nDims; i++)
    {
        int spread = abs(max[i] - min[i]);
        if (maxSpread < spread )
        {
            maxSpread = spread;
            maxSpreadDimension = i;
        }
    }
    return maxSpreadDimension;
}



template <typename Iterator>
inline int SpanningTreeStrategy_3dTorus<Iterator,SpanningTreeVertex>::numHops(const Iterator vtx1, const Iterator vtx2)
{
    /// @todo: Assert that the dimensions of the coordinate vectors of the two vertices are the same
    int nHops = 0; 
    for (int i=0, nDims=(*vtx1).X.size(); i<nDims; i++)
        nHops += abs( (*vtx1).X[i] - (*vtx2).X[i] );
    return nHops;
}

} // end namespace topo

#include "TopoManager.h"

namespace topo {

template <typename Iterator>
SpanningTreeVertex* SpanningTreeStrategy_3dTorus<Iterator,SpanningTreeVertex>::buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches)
{
    // if (maxBranches < 1) throw;
    CkAssert(maxBranches >= 1);
    /// Check validity and ranges etc.
    const int numDescendants = std::distance(firstVtx,beyondLastVtx) - 1;
    // if (numDescendants < 0) throw;
    CkAssert(numDescendants >= 0);
    /// If the parent vertex already has a(n older) list of children, clear it
    (*firstVtx).childIndex.clear();
    (*firstVtx).childIndex.reserve(maxBranches);
    /// Get a handle on a TopoManager. @note: Avoid this per-call instantiation cost by using an instance manager? Ideally, TopoManager should be a singleton.
    TopoManager aTopoMgr;
    /// Find the machine coordinates of each vertex
    for (Iterator itr = firstVtx; itr != beyondLastVtx; itr++)
    {
        (*itr).X.reserve(3);
        (*itr).X.assign(3,0);
        aTopoMgr.rankToCoordinates( (*itr).id, (*itr).X[0], (*itr).X[1], (*itr).X[2] );
    }
    ///@todo: If the machine coordinates are already stored in the vertices, do we want to find them again?
    /// Partition the vertex bounding box into maxBranches portions
    Iterator firstDescendant = firstVtx;
    partition(firstVtx,++firstDescendant,beyondLastVtx,maxBranches);
    /// Identify the closest member in each portion and put it at the corresponding childIndex location
    for (int i=0, numChildren=(*firstVtx).childIndex.size(); i<numChildren; i++)
    {
        Iterator rangeStart = firstVtx;
        std::advance(rangeStart,(*firstVtx).childIndex[i]);
        Iterator rangeEnd   = firstVtx;
        if (i+1 == numChildren) 
            rangeEnd = beyondLastVtx;
        else
            std::advance(rangeEnd, (*firstVtx).childIndex[i+1] );
        Iterator closestItr = pickClosest(firstVtx,rangeStart,rangeEnd);
        std::iter_swap(rangeStart,closestItr);
    }
    /// Return a copy of the parent in keeping with the generic interface
    return (new SpanningTreeVertex(*firstVtx) );
}

} // end namespace topo
#endif // TREE_STRATEGY_3D_TORUS

