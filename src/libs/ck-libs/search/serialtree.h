/*
 * Parallel state space search library
 *
 * Jonathan A. Booth
 * Sun Mar  2 22:34:13 CST 2003
 */

#ifndef __UIUC_CHARM_SERIALTREE_H
#define __UIUC_CHARM_SERIALTREE_H

#include <charm++.h>
#include "cklibs/problem.h"

// The serial tree performs a serial, depth-first, depth-bounded search.
// The reason it includes depth-bounded is because it shouldn't ever be
// called in a situation where it isn't depth bounded in this library.
// This could be crappy design, if it wasn't ment exclusivly for use
// with the search lib.
class SerialTree {
 public:
  problem *Solution;
  int SolutionHeight;
  CkQ<problem *> Children;
  int Expanded;

 private:
  void ChildSearch(problem *current, int heightLeft);
  void Search(problem *current, int heightLeft);

 public:
  SerialTree(problem *start, int height = 1);
  ~SerialTree();
};

#endif /* __UIUC_CHARM_SERIALTREE_H */
