/*
 * Parallel state space search library
 *
 * Jonathan A. Booth
 * Sun Mar  2 22:39:32 CST 2003
 */

#include "cklibs/serialtree.h"



SerialTree::SerialTree(problem *start, int height)
: Expanded(0), SolutionHeight(0), Solution(NULL) {
  // If this is a solution, then note that and you can cease
  if ( start->solution() ) {
    Solution = start;
    SolutionHeight = height;
    return;
  }

  // Otherwise we need to search the children of this problem.
  ChildSearch(start, height);
}

SerialTree::~SerialTree() {
  // Free any memory that is mine
  problem *tmp;
  while ( (tmp = Children.deq()) != NULL ) {
    delete tmp;
  }
  if ( Solution ) {
    delete Solution;
    Solution = NULL;
  }
}



void SerialTree::ChildSearch(problem *current, int heightLeft) {
  // Expand this node
  CkQ<problem *> kids;
  current->children(&kids);
  Expanded++;

  // For each of those children, search it
  problem *tmp;
  while ( (tmp = kids.deq()) != NULL ) {
    Search(tmp, heightLeft - 1);
  }
}

void SerialTree::Search(problem *current, int heightLeft) {
  // If I'm a solution, chuck myself in the solution queue and quit searching.
  if ( current->solution() ) {
    Solution = current;
    SolutionHeight = heightLeft;
    return;
  }

  // If I've found a solution and it's better than I could be, abort
  if ( heightLeft < SolutionHeight ) {
    delete current;
    return;
  }

  // Else if I'm out of height left to search, stick myself in the
  // children needing to be respawned queue
  if ( heightLeft <= 0 ) {
    Children.enq(current);
    return;
  }

  // Otherwise I'm an intermediate node in the search area. I'm not needed
  // after my children have been searched, so fire off a search on them
  // and then delete me.
  ChildSearch(current, heightLeft);
  delete current;
}
