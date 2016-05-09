#ifndef _PARTITIONING_STRATEGIES_H
#define _PARTITIONING_STRATEGIES_H

#include "TopoManager.h"
/** \brief A function to traverse the given processors, and get a hilbert list
 */
extern void getHilbertList(int * procList);

/** \brief A function to traverse the given processors, and get a planar list
 */
extern void getPlanarList(int *procList);

/** \brief A function to traverse the given processors, and get a recursive bisection list
 */
extern void getRecursiveBisectionList(int numparts, 
    TopoManager_getPartitionSize getSize, int *procList);

/** \brief A function to traverse the given processors, and get a blocked list
 */
extern void getBlockedTraversalList(int numparts, 
    TopoManager_getPartitionSize getSize, void *blocked_dims, int *procList);
#endif
