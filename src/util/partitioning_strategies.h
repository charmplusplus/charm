#ifndef _PARTITIONING_STRATEGIES_H
#define _PARTITIONING_STRATEGIES_H

/** \brief A function to traverse the given processors, and get a hilbert list
 */
extern void getHilbertList(int * procList);

/** \brief A function to traverse the given processors, and get a planar list
 */
extern void getPlanarList(int *procList);

/** \brief A function to traverse the given processors, and get a recursive bisection list
 */
extern void getRecursiveBisectionList(int numparts, int *procList);

#endif
