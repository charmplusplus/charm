/*
 * Parallel state space search library
 *
 * Jonathan A. Booth
 * Sun Mar  2 22:39:32 CST 2003
 */

#ifndef __UIUC_CHARM_PROBLEM_H
#define __UIUC_CHARM_PROBLEM_H

#include <charm++.h>
#include <ckbitvector.h>
#include <cklists.h>
#include <pup.h>

class problem : public PUP::able {
 public:
  // Declared public so the child classes can mess with these without a
  // stupid set of functions to read and write them. You should leave both
  // these fields set at their defaults when you create a child class of
  // problem.
  CkChareID Parent;
  int Root;
  CkBitVector Priority;

 public:
  // I define these
  problem();
  problem(const problem &p);
  problem(CkMigrateMessage *m);

  // User must define the following functions in their child class
  virtual void children(CkQ<problem *> *list) = 0;// Return children of node
  virtual problem * clone() = 0;		  // Clone this node
  virtual int depth();				  // How far are we?
  virtual int depthToSolution();		  // and How far is the answer?
  virtual void print();				  // Print this node
  virtual void pup(PUP::er &p);			  // Pack/unpack this node
  virtual bool solution() = 0;			  // Node is solution?

  // This is required to make pup work with an abstract class
  PUPable_abstract(problem);
};

#endif /* __UIUC_CHARM_PROBLEM_H */
