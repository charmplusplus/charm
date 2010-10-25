#include <iostream>
#include <vector>
#include <iterator>
#include <cstdlib>

#include "conv-config.h"
#include "spanningTreeVertex.h"

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

/**
 * @file: This header includes utility routines and several strategies for constructing spanning trees for msg delivery.
 *
 * Multicasts and reductions can be efficiently implemented by propogating the message(s) along spanning trees
 * that are constructed using as much information about the machine as possible. Libraries that implement any
 * multicast/redn algorithms can use this header to easily generate a spanning tree from a list of target PEs.
 *
 * Client code does not have to worry about machine specifics and any tree generation algorithms. Generated spanning
 * trees will be based on the best available information (node-awareness, network-topo awareness etc.).
 *
 * The focus remains on distributed logic that generates efficient (for practical purposes) spanning trees and not
 * theoretical MSTs (minimum spanning trees) etc. Hence the intended use is to supply a list of PEs as input and
 * obtain info about the immediate next generation of vertices in the tree and the members of each of their
 * respective sub-trees. As an afterthought, the API includes calls that will generate a complete spanning tree
 * from the input too. However, no guarantees are currently made about the efficiency of this use.
 *
 * Usage:
 *       Simple use cases shouldn't have to worry about anything beyond the self-explanatory functions:
 *       buildSpanningTreeGeneration()
 *       buildSpanningTree()
 *
 *       SpanningTreeVertex is a data structure that facilitates input/output, although input can be any container
 *       (C arrays, vectors etc.) of PEs
 *
 *       Another routine to be aware of is a factory method that is used behind the scenes to obtain what appears
 *       to be the best strategy for building trees given the current machine/input etc: getSpanningTreeStrategy().
 *
 *       Users can override this choice by specifying a strategy when calling one of the build routines. A list of
 *       strategies should be apparent from the headers included near the bottom of this file.
 *
 * Note to developers:
 *       This code is probably more grandiose than it needs to be. Basically, there are a bunch of strategy classes
 *       (that inherit from SpanningTreeStrategy) for constructing spanning trees. There is a factory method 
 *       (getSpanningTreeStrategy) that returns the strategy that it thinks is the best choice given the machine, 
 *       input etc. There are some template gimmicks just to allow users to pass input however they want (arrays /
 *        vectors / CkVecs etc. of ints / SpanningTreeVertices)
 *
 *       Steps for adding new algorithms for constructing spanning trees should look something like this.
 *       - Create a new template class (that inherits from the template base class SpanningTreeStrategy)
 *       - Put it in a new file following appropriate naming conventions.
 *       - Add an include statement near the bottom of this file along with the other existing includes
 *       - Override the virtual method buildNextGen() in your new strategy class to implement your algorithm
 *       - Existing strategies provide partial specializations based on the type of input passed in. Input 
 *         is typically a container of int or SpanningTreeVertex. You may have to do something different for
 *         each of these input types. (for eg, for SpanningTreeVertex store the results in the parent vertex too)
 *       - Enclose any implementation details (functions, classes etc.) within another namespace (say, impl)
 *         to avoid polluting this top-level namespace.
 *       - Test your strategy by explicitly calling it from some test code to check its operation.
 *       - Make appropriate changes to getSpanningTreeStrategy() to return your new strategy for appropriate
 *         input on the appropriate machine(s)
 */


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

//@{
/// Builds one generation of the spanning tree given a container of vertices with the tree root as the first element in the container
// Use a default strategy
template <typename Iterator>
SpanningTreeVertex* buildSpanningTreeGeneration
(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2);
// Use the strategy provided by the caller
template <typename Iterator>
SpanningTreeVertex* buildSpanningTreeGeneration
(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches, SpanningTreeStrategy<Iterator> *bldr);
//@}

//@{
/// Builds the complete spanning tree given a container of vertices with the tree root as the first element in the container
// Use a default strategy
template <typename Iterator>
void buildSpanningTree
(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches=2);
// Use the strategy provided by the caller
template <typename Iterator>
void buildSpanningTree
(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches, SpanningTreeStrategy<Iterator> *bldr);
//@}

/// Tiny factory method that returns a tree construction strategy that it thinks is best (based on inputs, the machine's network topology info etc)
template <typename Iterator>
SpanningTreeStrategy<Iterator>* getSpanningTreeStrategy
(const Iterator firstVtx, const Iterator beyondLastVtx, const int maxBranches);

} // end namespace topo



//--------------------------------  Class and function definitions --------------------------------
namespace topo {

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
    CkAssert(NULL != bldr);

    /// Validate input. Invalid inputs are not exceptions. They are just no-ops
    if (maxBranches < 1 || firstVtx == beyondLastVtx)
        return new SpanningTreeVertex();
    else
	/// Delegate the actual work
	return bldr->buildNextGen(firstVtx,beyondLastVtx,maxBranches);
}


// Overload to automatically use the default strategy
template <typename Iterator>
inline SpanningTreeVertex* buildSpanningTreeGeneration(const Iterator firstVtx,
                                                       const Iterator beyondLastVtx,
                                                       const int maxBranches
                                                      )
{
    SpanningTreeStrategy<Iterator> *bldr =
	getSpanningTreeStrategy(firstVtx,beyondLastVtx,maxBranches);
    SpanningTreeVertex *result = buildSpanningTreeGeneration(firstVtx, beyondLastVtx, maxBranches, bldr);
    delete bldr;
    return result;
}



/// Nested namespace to prevent the implementation muck from polluting topo::
namespace impl {

/// Tag dispatched function that does the actual work of building the complete spanning tree
template <typename Iterator>
void buildSpanningTree(SpanningTreeVertex* dispatchTag,
                       const Iterator firstVtx,
                       const Iterator beyondLastVtx,
                       const int maxBranches,
                       SpanningTreeStrategy<Iterator> *bldr
                      )
{
    /// Validate input. Invalid inputs are not exceptions. They are just no-ops
    if (maxBranches < 1 || firstVtx == beyondLastVtx)
        return;

    /// Build the next generation of the tree rooted at *firstVtx
    SpanningTreeVertex *tmp = buildSpanningTreeGeneration(firstVtx,beyondLastVtx,maxBranches,bldr);
    /// Delete the copy of firstVtx that gets returned from the call
    delete tmp;

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
    CkAssert(NULL != bldr);
    /// Create a tag
    typename std::iterator_traits<Iterator>::value_type *tag = 0;
    /// Delegate the work
    impl::buildSpanningTree(tag,firstVtx,beyondLastVtx,maxBranches,bldr);
}


// Overload to automatically use the default strategy
template <typename Iterator>
inline void buildSpanningTree(const Iterator firstVtx,
                              const Iterator beyondLastVtx,
                              const int maxBranches
                             )
{
    SpanningTreeStrategy<Iterator> *bldr =
	getSpanningTreeStrategy(firstVtx,beyondLastVtx,maxBranches);
    buildSpanningTree(firstVtx, beyondLastVtx, maxBranches, bldr);
    delete bldr;
}

} // end namespace topo



/** @note: Concrete strategy implementations depend on the definitions in this file. Hence
 * its necessary that user code include this file and not just any of the individual strategy
 * headers that follow.
 */
#include "treeStrategy_topoUnaware.h"
#include "treeStrategy_nodeAware_minGens.h"
#include "treeStrategy_nodeAware_minBytes.h"
#include "treeStrategy_3dTorus_minHops.h"
#include "treeStrategy_3dTorus_minBytesHops.h"

namespace topo {

/**
 * Simply uses machine specific preprocessor macros to return an appropriate strategy.
 * Expects that these macros will be available in the current compilation unit thanks to charm innards.
 *
 * Reserves the right to use other heuristics (based on the input data) to select a strategy, if such 
 * needs arise in the future.
 *
 * @note: The list of machine macros here should sync with the ones that TopoManager is aware of.
 */
template <typename Iterator>
inline SpanningTreeStrategy<Iterator>* getSpanningTreeStrategy(const Iterator firstVtx,
                                                               const Iterator beyondLastVtx,
                                                               const int maxBranches)
{
    #if CMK_BLUEGENEL || CMK_BLUEGENEP
        return ( new SpanningTreeStrategy_3dTorus_minBytesHops<Iterator>() );
    #elif XT3_TOPOLOGY || XT4_TOPOLOGY || XT5_TOPOLOGY
        return ( new SpanningTreeStrategy_3dTorus_minBytesHops<Iterator>() );
    #else
        /// Nested, utility class to let us to use the parent PE for different Iterator::value_types
        class NumOnNode
        {
            public:
                inline int operator() (const vtxType vtx)             { return CmiNumPesOnPhysicalNode( CmiPhysicalNodeID(vtx) ); }
                inline int operator() (const SpanningTreeVertex &vtx) { return CmiNumPesOnPhysicalNode( CmiPhysicalNodeID(vtx.id) ); }
        } findNumPEsOnNode;

        /// Find the number of PEs on the same node as the parent vertex (max possible on-node destinations)
        int numPEsOnNode = findNumPEsOnNode(*firstVtx);

        /// If minimizing tree generations may be more beneficial than reducing inter-node traffic
        if (numPEsOnNode <= 4 && maxBranches < 4)
            return ( new SpanningTreeStrategy_nodeAware_minGens<Iterator>() );
        else
            return ( new SpanningTreeStrategy_nodeAware_minBytes<Iterator>() );
    #endif
}

} // end namespace topo

#endif // SPANNING_TREE_STRATEGY

