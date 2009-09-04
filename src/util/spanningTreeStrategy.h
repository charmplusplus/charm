#include <iostream>
#include <vector>
#include <iterator>

#include "conv-config.h"

#ifndef SPANNING_TREE_STRATEGY
#define SPANNING_TREE_STRATEGY

#if ! CMK_HAS_ITERATOR_TRAITS 
namespace std {

template <class Iterator>
  struct iterator_traits {
    typedef typename Iterator::iterator_category iterator_category;
    typedef typename Iterator::value_type        value_type;
    typedef typename Iterator::difference_type   difference_type;
    typedef typename Iterator::pointer           pointer;
    typedef typename Iterator::reference         reference;
  };

  template <class T>
  struct iterator_traits<T*> {
    typedef random_access_iterator_tag iterator_category;
    typedef T                          value_type;
    typedef ptrdiff_t                  difference_type;
    typedef T*                         pointer;
    typedef T&                         reference;
  };

}

#endif

#if ! CMK_HAS_STD_DISTANCE 
namespace std {

template <class Iterator>
int distance(Iterator first, Iterator last)
{
  int n=0;
  while (first!=last)
  {
       ++first;
       ++n;
  }

  return n;
}
}
#endif

//-------------------------------- Declarations of the main entities in this file --------------------------------

/// This namespace contains network topology related classes and utilities
namespace topo {

/// Alias for the actual data type of a vertex id (PE/node number)
typedef int vtxType;

/// Container returned by a SpanningTreeStrategy holding information about a vertex in the tree and its children
class SpanningTreeVertex;

/// Base class for all the spanning tree build strategies
template <typename Iterator,typename ValueType = typename std::iterator_traits<Iterator>::value_type>
class SpanningTreeStrategy;

/// Tiny factory method that returns an appropriate spanning tree builder based on the machine network topology information (if available)
template <typename Iterator>
SpanningTreeStrategy<Iterator>* getSpanningTreeStrategy();

/// Builds one generation of the spanning tree given a container of vertices with the tree root as the first element in the container
template <typename Iterator>
SpanningTreeVertex* buildSpanningTreeGeneration
(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2, SpanningTreeStrategy<Iterator> *bldr = getSpanningTreeStrategy<Iterator>());

/// Builds the complete spanning tree given a container of vertices with the tree root as the first element in the container
template <typename Iterator>
void buildSpanningTree
(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2, SpanningTreeStrategy<Iterator> *bldr = getSpanningTreeStrategy<Iterator>());

} // end namespace topo



//--------------------------------  Class and function definitions --------------------------------
namespace topo {

/** 
 * Contains indices to direct children. childIndex[i]+1 and childIndex[i+1] are the first and 
 * beyondLast indices of the sub-tree members of the child at childIndex[i]. 
 * @note: We're using a (vertex, edge) terminology to talk about spanning trees. Consciously staying 
 * away from using "node" to avoid ambiguity with machine nodes and PEs. This is inspite of the fact 
 * that typically a vertex of a spanning tree is a machine node / PE.
 */
class SpanningTreeVertex
{
    public:
        /// The id (PE) of the vertex 
        vtxType id;
        /// The parent of this vertex. Uncomment if needed
        // vtxType parent;
        /// The machine coordinates of this vertex
        std::vector<int> X; 
        /// Relative distance (in the container) from the position of this vertex to direct children (and their sub-tree members)
        std::vector<int> childIndex;
        /// Constructor
        SpanningTreeVertex(const vtxType _id=-1): id(_id) {}

    /// Stream inserter. Note: not a member function
    friend std::ostream& operator<< (std::ostream &out, const SpanningTreeVertex &obj)
    {
        out<<" "<<obj.id;
        if (obj.X.size()>0) 
        {
            out<<"("<<obj.X[0];
            for (int i=1,cSize=obj.X.size(); i<cSize; i++)
                out<<","<<obj.X[i];
            out<<") ";
        }
        return out;
    }
};




/** The spanning tree build strategy interface. Partitions and reorders a collection of tree members, and returns information about children and their sub-trees.
 *
 * @warning: User code should only specify typename Iterator. ValType is automatically given the 
 * default value based on Iterator and should not be specified by the user. 
 *
 * @todo: If compile-time asserts become available in the charm world, then assert that ValType is 
 * nothing illegal in the appropriate subclasses. Better still, (if possible) let the subclass remain
 * abstract and implement buildNextGen only for the appropriate partial specializations of ValType.
 */
template <typename Iterator,typename ValueType>
class SpanningTreeStrategy
{
    public:
        /// Concrete builders should implement this (preferably only for the appropriate specializations)
        virtual SpanningTreeVertex* buildNextGen(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2) = 0;
};




/** Facade function to hide all the template muck for the mainstream usecases. 
 *
 * Use when you're happy with the default spanning tree strategy. This should hide the very existence
 * of all this template stuff for the default use-cases. 
 *
 * For scenarios, where you want to specify the tree building strategy, you need to supply a strategy
 * object. In this case, instantiate a strategy object manually (the factory method 
 * getSpanningTreeStrategy() will give you what it thinks is best) and plug it into this function 
 * call.
 *
 * Its advisable to use this facade even when explicitly specifying a strategy object, as this leaves
 * room for future additions to the procedure without affecting the user code
 */
template <typename Iterator>
inline SpanningTreeVertex* buildSpanningTreeGeneration(const Iterator firstVtx, 
                                                       const Iterator beyondLastVtx, 
                                                       const int maxBranches, 
                                                       SpanningTreeStrategy<Iterator> *bldr 
                                                      )
{
    return bldr->buildNextGen(firstVtx,beyondLastVtx,maxBranches);
}




/// Nested namespace to prevent the implementation muck from polluting topo::
namespace impl {

/// Tag dispatched function that does the actual work of building the spanning complete tree
template <typename Iterator>
void buildSpanningTree(SpanningTreeVertex* dispatchTag,
                       const Iterator firstVtx,
                       const Iterator beyondLastVtx,
                       const int maxBranches=2,
                       SpanningTreeStrategy<Iterator> *bldr = getSpanningTreeStrategy<Iterator>()
                      )
{
    /// Build a tree only if there are any vertices in the input range
    if (firstVtx < beyondLastVtx)
    {
        /// Build the next generation of the tree rooted at *firstVtx
        buildSpanningTreeGeneration(firstVtx,beyondLastVtx,maxBranches,bldr);
        int numChildren = (*firstVtx).childIndex.size();
        /// For each direct child...
        for (int i=0; i< numChildren; i++)
        {
            /// Identify the range of vertices that are part of this subtree
            Iterator start = firstVtx, end = firstVtx;
            std::advance(start, (*firstVtx).childIndex[i] );
            if (i < numChildren -1)
                std::advance(end, (*firstVtx).childIndex[i+1] );
            else
                end = beyondLastVtx;
            /// Build this subtree
            buildSpanningTree(dispatchTag,start,end,maxBranches,bldr);
        }
    }
}

} // end namespace impl 
  



/**
 * Facade function to build a complete spanning tree given an input container of SpanningTreeVertex-es
 * Uses tag-dispatching to ensure at compile-time that the input container holds only SpanningTreeVertex 
 * and nothing else.
 */
template <typename Iterator>
inline void buildSpanningTree(const Iterator firstVtx,
                              const Iterator beyondLastVtx,
                              const int maxBranches,
                              SpanningTreeStrategy<Iterator> *bldr
                             )
{
    typename std::iterator_traits<Iterator>::value_type *tag = 0;
    impl::buildSpanningTree(tag,firstVtx,beyondLastVtx,maxBranches,bldr);
}

} // end namespace topo



#include "treeStrategy_topoUnaware.h"
#include "treeStrategy_3dTorus.h"
namespace topo {

/**
 * Simply uses machine specific preprocessor macros to return an appropriate strategy.
 * Expects that these macros will be available in the current compilation unit thanks to charm innards.
 * Could potentially also use other heuristics to decide on the best strategy, but that might be possible 
 * only if it has some more input information to use.
 *
 * @note: The list of machine macros here should sync with the ones that TopoManager is aware of.
 */
template <typename Iterator>
inline SpanningTreeStrategy<Iterator>* getSpanningTreeStrategy() 
{
    #if CMK_BLUEGENEL || CMK_BLUEGENEP
        return ( new SpanningTreeStrategy_3dTorus<Iterator>() );
    #elif XT3_TOPOLOGY || XT4_TOPOLOGY || XT5_TOPOLOGY
        return ( new SpanningTreeStrategy_3dTorus<Iterator>() );
    #else
        return ( new SpanningTreeStrategy_topoUnaware<Iterator>() );
    #endif
}

} // end namespace topo

#endif // SPANNING_TREE_STRATEGY

