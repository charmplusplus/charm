#include <algorithm>
#include "charm++.h"

#ifndef TREE_STRATEGY_3D_TORUS_MIN_HOPS
#define TREE_STRATEGY_3D_TORUS_MIN_HOPS

#if XE6_TOPOLOGY
#include <bitset>
#endif

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
#if XE6_TOPOLOGY
        void translateCoordinates(const Iterator start, const Iterator end, int nDims);
#endif

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
                    //return (a.X[dim] < b.X[dim]);
                    // modified so that it not only orders by coordinate in dim, but also
                    // places PEs with the same coordinates (all dims) consecutively
                    if (a.X[dim] < b.X[dim]) return true;
                    else if (a.X[dim] == b.X[dim]) {
                      switch(dim) {
                        case 0: other[0] = 1; other[1] = 2; break;
                        case 1: other[0] = 0; other[1] = 2; break;
                        case 2: other[0] = 0; other[1] = 1; break;
                        default: CkAbort("NOT SUPPORTED\n");
                      }
                      if (a.X[other[0]] < b.X[other[0]]) return true;
                      else if (a.X[other[0]] == b.X[other[0]]) {
                        return (a.X[other[1]] < b.X[other[1]]);
                      } else {
                        return false;
                      }
                    } else {
                      return false;
                    }
                }
            private:
                const int dim;
                int other[2];
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

    /// Get a handle on TopoManager
    TopoManager *aTopoMgr = TopoManager::getTopoManager();
    /// Find the machine coordinates of each vertex
    for (Iterator itr = firstVtx; itr != beyondLastVtx; itr++)
    {
        (*itr).X.reserve(3);
        (*itr).X.assign(3,0);
        int coreNum; ///< dummy var. Get and discard the core number
        aTopoMgr->rankToCoordinates( (*itr).id, (*itr).X[0], (*itr).X[1], (*itr).X[2], coreNum );
    }
    ///@todo: If the machine coordinates are already stored in the vertices, do we want to find them again?
    impl::TreeBoundingBoxOn3dTorus<Iterator> treePart;
#if XE6_TOPOLOGY
    if ((aTopoMgr->getDimNX() > MAX_TORUS_DIM_SIZE) ||
        (aTopoMgr->getDimNY() > MAX_TORUS_DIM_SIZE) ||
        (aTopoMgr->getDimNZ() > MAX_TORUS_DIM_SIZE)) {
      CkAbort("Torus dimension size larger than supported limit. Please increase limit");
    }
    treePart.translateCoordinates(firstVtx, beyondLastVtx, 3);
#endif

    /// Partition the vertex bounding box into maxBranches portions
    Iterator firstDescendant = firstVtx;
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

#if __DEBUG_SPANNING_TREE_
// make sure that all PEs in the same node (i.e. with same coordinates)
// are placed consecutively in the list
template <typename Iterator>
void checkSection(const Iterator start, const Iterator end, const Iterator p1, const Iterator p2) {
  if ((p1 != start) && ((*p1).X == (*(p1-1)).X)) {
    CkAbort("checkSection: section point p1 incorrect\n");
  }
  if ((p2 != start) && ((*p2).X == (*(p2-1)).X)) {
    CkAbort("checkSection: section point p2 incorrect\n");
  }
  for (Iterator it = start; it != end; it++) {
    int d=1;
    for (Iterator it2 = it+1; it2 != end; it2++, d++) {
      if ((*it).X == (*it2).X) {
        if (d == 1) break;
        else CkAbort("ERROR PEs in same node are NOT together\n");
      }
    }
  }
}
#endif

// TODO current solution to avoid splitting at middle of physical node
// is inefficient (mainly the sort). Ideally, algorithm should bisect nodes
// (not PEs) and then we could use the original more efficient algorithm
template <typename Iterator>
void TreeBoundingBoxOn3dTorus<Iterator>::bisect(const Iterator parent, const Iterator start, const Iterator end, const int numPartitions)
{
    /// Find the number of vertices in the range
    int numVertices = std::distance(start,end);
    /// Find the dimension along which to bisect the bounding box
    int maxSpreadDim = findMaxSpreadDimension(start,end);
    std::sort(start, end, lessThan(maxSpreadDim));
    Iterator median = start;
    std::advance(median,numVertices/2);

    if ((median != start) && ((*median).X == (*(median-1)).X)) {
      // correct so that no splitting occurs in the middle of a physical node
      // by moving median to first PE of the node
#if __DEBUG_SPANNING_TREE_
      CkPrintf("[%d] WARNING: median at middle of physical node\n", CkMyPe());
#endif
      while ((median != start) && ((*(median-1)).X == (*median).X)) median--;
    }
#if __DEBUG_SPANNING_TREE_
    checkSection(start, end, median, median);
#endif
    if (median == start) {
      partition(parent, start, end, 1);
      return;
    }

    /// Partition the two pieces as necessary
    int numLeft = numPartitions/2;
    partition(parent, start, median, numLeft);
    partition(parent, median, end, numPartitions - numLeft);
}

// TODO current solution to avoid splitting at middle of physical node
// is inefficient (mainly the sort) and for trisection possibly not all that
// accurate. Ideally, algorithm should trisect nodes (not PEs) and then we could
// use the original more efficient algorithm
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
        std::sort(start, end, lessThan(maxSpreadDim));
        Iterator oneThird = start;
        std::advance(oneThird,numVertices/3);
        Iterator twoThird = oneThird;
        std::advance(twoThird,numVertices/3);

        // correct so that no splitting occurs in the middle of a physical node
        if ((oneThird != start) && ((*oneThird).X == (*(oneThird-1)).X)) {
#if __DEBUG_SPANNING_TREE_
          CkPrintf("[%d] WARNING: oneThird at middle of physical node\n", CkMyPe());
#endif
          while ((oneThird != start) && ((*(oneThird-1)).X == (*oneThird).X)) oneThird--;
        }
        if ((twoThird != start) && ((*twoThird).X == (*(twoThird-1)).X)) {
#if __DEBUG_SPANNING_TREE_
          CkPrintf("[%d] WARNING: twoThird at middle of physical node\n", CkMyPe());
#endif
          while ((twoThird != start) && ((*(twoThird-1)).X == (*twoThird).X)) twoThird--;
        }
#if __DEBUG_SPANNING_TREE_
        checkSection(start, end, oneThird, twoThird);
#endif
        if (oneThird == twoThird) { // trisection points are in same physical node
          while ((twoThird != end) && ((*twoThird).X == (*oneThird).X)) twoThird++;
          if (twoThird == end) {
            if (oneThird == start) {
              partition(parent, start, end, 1);
              return;
            } else {
              int numLeft = numPartitions/2;
              partition(parent, start, oneThird, numLeft);
              partition(parent, oneThird, end, numPartitions - numLeft);
              return;
            }
          }
        }
        if (oneThird == start) {
          int numLeft = numPartitions/2;
          partition(parent, start, twoThird, numLeft);
          partition(parent, twoThird, end, numPartitions - numLeft);
          return;
        }

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
#if XE6_TOPOLOGY
    std::vector<std::bitset<MAX_TORUS_DIM_SIZE> > usedCoordinates(nDims);
    for (Iterator itr = start; itr != end; itr++) {
      for (int i=0; i < nDims; i++) usedCoordinates[i].set((*itr).X[i]);
    }
    int maxSpreadDimension = 0;
    int maxSpread = -1;
    for (int i=0; i < nDims; i++) {
      int count = usedCoordinates[i].count();
#if __DEBUG_SPANNING_TREE_
      if (CkMyPe() == 0) CkPrintf("Spread on dimension %d is %d\n", i, count);
#endif
      if (count > maxSpread) {
        maxSpread = count;
        maxSpreadDimension = i;
      }
    }
#if __DEBUG_SPANNING_TREE_
    if (CkMyPe() == 0) CkPrintf("Max spread dimension is %d\n", maxSpreadDimension);
#endif
    //return 1;
    return maxSpreadDimension;
#else
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
#endif
}

#if XE6_TOPOLOGY

#include <limits.h>

inline int modulo(int k, int n) {
  return ((k %= n) < 0) ? k+n : k;
}

// Translate coordinates of elements in range [start,end]
// such that the max spread in each dimension is equal (or closer) to the number
// of coordinates actually used in each dimension.
// In other words, "moves" all elements by some translation offset so that their bounding box
// includes minimum number of coordinates, which implies that adjacent nodes won't go through torus edges.
// Note that new coordinates are only used for the purpose of the torus spanning tree algorithm,
// real coordinates of elements are unchanged.
// Works for any number of dimensions (N-d torus)
// TODO Algorithm should ideally call this method with physical nodes, NOT PEs
template <typename Iterator>
void TreeBoundingBoxOn3dTorus<Iterator>::translateCoordinates(const Iterator start, const Iterator end, int nDims)
{
  std::vector<std::bitset<MAX_TORUS_DIM_SIZE> > usedCoordinates(nDims);
  std::vector<int> max_coord(nDims, -1);
  std::vector<int> min_coord(nDims, INT_MAX);
  std::vector<int> maxSpread(nDims);
  std::vector<int> gapCenter(nDims, -1);
  std::vector<int> dimSizes(nDims+1);
  TopoManager_getDims(&(dimSizes.front()));
#if __DEBUG_SPANNING_TREE_
  if (CkMyPe() == 0)
    std::cout << "Dim sizes are: " << dimSizes[0] << " " << dimSizes[1] << " " << dimSizes[2] << std::endl;
#endif

  int numVertices = 0;
  for (Iterator itr = start; itr != end; itr++, numVertices++) {
    for (int i=0; i < nDims; i++) {
      int c = (*itr).X[i];
      usedCoordinates[i].set(c);
      if (c > max_coord[i]) max_coord[i] = c;
      if (c < min_coord[i]) min_coord[i] = c;
    }
  }
  for (int i=0; i < nDims; i++) {
    maxSpread[i] = max_coord[i] - min_coord[i] + 1; // store max spread of each dimension
    int sum = 0, nbUnusedCoords = 0;
    for (int j=0; j < dimSizes[i]; j++) {
      if (!usedCoordinates[i].test(j)) {  // j coordinate not used by any element
#if __DEBUG_SPANNING_TREE_
        if (CkMyPe() == 0)
          std::cout << "Dimension " << i << ": coordinate " << j << " is unused" << std::endl;
#endif
        sum += j;
        nbUnusedCoords++;
      }
    }
    if (nbUnusedCoords > 0) gapCenter[i] = sum / nbUnusedCoords;
  }

#if __DEBUG_SPANNING_TREE_
  if (CkMyPe() == 0) {
    std::cout << "Used coordinates in each dimension:" << std::endl;
    for (int i=0; i < nDims; i++) {
      std::cout << i << ": ";
      for (int j=0; j < dimSizes[i]; j++) if (usedCoordinates[i].test(j)) std::cout << j << " ";
      std::cout << ", " << usedCoordinates[i].count() << std::endl;
    }
    std::cout << "Max,min coord in each dimension:" << std::endl;
    for (int i=0; i < nDims; i++) std::cout << i << ": " << max_coord[i] << " " << min_coord[i] << std::endl;
    std::cout << "Gap center for each dimension:" << std::endl;
    for (int i=0; i < nDims; i++) std::cout << i << ": " << gapCenter[i] << std::endl;
    std::cout << "Max spread for each dimension:" << std::endl;
    for (int i=0; i < nDims; i++) std::cout << i << ": " << maxSpread[i] << std::endl;
  }
#endif

  //std::vector<int> bestCoord(numVertices);
  for (int i=0; i < nDims; i++) { // find best translation offset for each dimension
    if (maxSpread[i] == usedCoordinates[i].count()) continue; // nothing to correct for this dimension

#if __DEBUG_SPANNING_TREE_
    if (CkMyPe() == 0)
      std::cout << "Going to attempt to correct coordinates on dimension " << i << std::endl;
#endif

    // choose direction of unused coordinates to finish faster
    int direction = 1;
    if (gapCenter[i] < dimSizes[i]/2) direction = -1;
#if __DEBUG_SPANNING_TREE_
    if (CkMyPe() == 0) std::cout << "Chose direction " << direction << std::endl;
#endif

    // we're going to attempt to minimize the max spread in dimension i
    int bestMaxSpread = maxSpread[i];
    int bestOffset=0/*, j*/;
    Iterator itr;
    //for (itr=start, j=0; itr != end; itr++, j++) bestCoord[j] = (*itr).X[i];
    for (int m=0; ; m++) {
      // apply offset of 'm' in 'direction' to all elements
      int max_coord = -1;
      int min_coord = INT_MAX;
      for (itr = start; itr != end; itr++) {
        int &x = (*itr).X[i];
        x += direction;
        if (x >= dimSizes[i]) x = 0;
        else if (x < 0) x = dimSizes[i] - 1;
        if (x > max_coord) max_coord = x;
        if (x < min_coord) min_coord = x;
      }
      // evaluate max spread with new offset
      int maxSpread_new = max_coord - min_coord + 1;
#if __DEBUG_SPANNING_TREE_
      if (CkMyPe() == 0)
        std::cout << m << " " << maxSpread_new << std::endl;
#endif
      if (maxSpread_new == usedCoordinates[i].count()) {
#if __DEBUG_SPANNING_TREE_
        if (CkMyPe() == 0)
          std::cout << "FIXED after " << (m+1) << " movements" << std::endl;
#endif
        break;
      } else if (maxSpread_new < bestMaxSpread) {
        bestMaxSpread = maxSpread_new;
        //for (itr=start, j=0; itr != end; itr++, j++) bestCoord[j] = (*itr).X[i];
        bestOffset = m;
      }
      if (m == dimSizes[i] - 2) {
        // did max number of possible movements (another movement would return us to original
        // coordinates/offset), exit loop
        if (maxSpread_new > bestMaxSpread) {
          for (itr=start/*, j=0*/; itr != end; itr++/*, j++*/) {
            //(*itr).X[i] = bestCoord[j];
            // roll back to bestOffset
            int &x = (*itr).X[i];
            x += ((m - bestOffset)*-direction);
            x = modulo(x, dimSizes[i]);
          }
        }
#if __DEBUG_SPANNING_TREE_
        if ((CkMyPe() == 0) && (bestMaxSpread < maxSpread[i]))
          std::cout << "Improved to " << bestMaxSpread << " max spread" << std::endl;
#endif
        break;   // we're done
      }
    }
  }
}
#endif

} // end namespace impl

} // end namespace topo
#endif // TREE_STRATEGY_3D_TORUS_MIN_HOPS

