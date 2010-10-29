#include <algorithm>
#include "charm++.h"

#ifndef TREE_STRATEGY_3D_TORUS_MIN_HOPS
#define TREE_STRATEGY_3D_TORUS_MIN_HOPS
namespace topo {

/** A concrete tree builder for use on machines with a 3D Torus topology. Naturally, can also work with
 * 3D meshes (portions of the torus).
 *
 * Reduces hop-bytes by trying to reduce the total number of hops across the tree. Is implicitly
 * node-aware as on-node PEs will have a distance of zero and will end up as direct children in the
 * spanning tree. Does not pay any attention to reducing the number of bytes on the network by
 * minimizing inter-node traffic. For that, refer to SpanningTreeStrategy_3dTorus_minBytesHops.
 *
 * Specialized and implemented only for data type in input container = vtxType / SpanningTreeVertex.
 * @note: If its a container of SpanningTreeVertices, the next gen info is stored in the parent
 * element and a copy of the parent is also returned.
 */
template <typename Iterator,typename ValueType = typename std::iterator_traits<Iterator>::value_type>
class SpanningTreeStrategy_3dTorus_minHops: public SpanningTreeStrategy<Iterator>
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2) = 0;
};



/** Partial specialization for scenario for a container of SpanningTreeVertices
 */
template <typename Iterator>
class SpanningTreeStrategy_3dTorus_minHops<Iterator,SpanningTreeVertex>: public SpanningTreeStrategy<Iterator>
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2);
};



/** Partial specialization for scenario when a container of vtxTypes is input.
 *
 * Simply builds a container of SpanningTreeVertices from the input data and delegates the actual
 * tree building to another specialization.
 */
template <typename Iterator>
class SpanningTreeStrategy_3dTorus_minHops<Iterator,vtxType>: public SpanningTreeStrategy<Iterator>
{
    public:
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2)
        {
            /// Create a container of SpanningTreeVertices from the input container and fill it with vertex IDs
            std::vector<SpanningTreeVertex> tree;
            for (Iterator itr = firstVtx; itr != beyondLastVtx; itr++)
                tree.push_back( SpanningTreeVertex(*itr) );
            /// Instantiate the real builder and let it build the next generation
            SpanningTreeStrategy_3dTorus_minHops< std::vector<SpanningTreeVertex>::iterator > theRealBuilder;
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



namespace impl {
/**
 * Utility class to partition the bounding box of a spanning (sub)tree on a 3D mesh machine and 
 * divide that into the necessary number of branches
 */
template <typename Iterator>
class TreeBoundingBoxOn3dTorus
{
    public:
        /// Partition the range along the longest dimension into numPartitions parts
        void partition(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions);

    protected:
        /// Bisect the range along the longest dimension
        void bisect(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions);
        /// Trisect the range along the longest dimension
        void trisect(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions);
        /// Returns the dimension along which the bounding box of a range of vertices has the longest side
        int findMaxSpreadDimension(const Iterator start, const Iterator end);
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
} // end namespace impl

} // end namespace topo


#include "TopoManager.h"

namespace topo {

template <typename Iterator>
SpanningTreeVertex* SpanningTreeStrategy_3dTorus_minHops<Iterator,SpanningTreeVertex>::buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches)
{
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
        int coreNum; ///< dummy var. Get and discard the core number
        aTopoMgr.rankToCoordinates( (*itr).id, (*itr).X[0], (*itr).X[1], (*itr).X[2], coreNum );
    }
    ///@todo: If the machine coordinates are already stored in the vertices, do we want to find them again?

    /// Partition the vertex bounding box into maxBranches portions
    Iterator firstDescendant = firstVtx;
    impl::TreeBoundingBoxOn3dTorus<Iterator> treePart;
    treePart.partition(firstVtx,++firstDescendant,beyondLastVtx,maxBranches);

    /// Identify the closest member in each subtree and put it at the corresponding childIndex location
    for (int i=0, numChildren=(*firstVtx).childIndex.size(); i<numChildren; i++)
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



namespace impl {

template <typename Iterator>
inline void TreeBoundingBoxOn3dTorus<Iterator>::partition(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions)
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
void TreeBoundingBoxOn3dTorus<Iterator>::bisect(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions)
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
void TreeBoundingBoxOn3dTorus<Iterator>::trisect(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions)
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
int TreeBoundingBoxOn3dTorus<Iterator>::findMaxSpreadDimension(const Iterator start, const Iterator end)
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

} // end namespace impl

} // end namespace topo
#endif // TREE_STRATEGY_3D_TORUS_MIN_HOPS

