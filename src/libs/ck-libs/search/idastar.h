/*
 * Parallel state space search library
 *
 * Jonathan A. Booth
 * Mon Mar  3 14:36:49 CST 2003
 */

#ifndef __UIUC_CHARM_ASTAR_H
#define __UIUC_CHARM_ASTAR_H

#include <charm++.h>
#include "cklibs/problem.h"
#include "cklibs/serialtree.h"
#include "cklibs/search.decl.h"

// A function to fake aStar using the idaStar.
void aStar(problem *issue,
	   int maxdepth, int charesize, int serialdist,
	   CkCallback finished);

// A function to make spawning idaStar easier.
void idaStar(problem *issue,
	     int startdepth, int maxdepth, int stride, int window,
	     int charesize, int serialdist,
	     CkCallback finished);

// Chare who does the work
class idaStarWorker : public CBase_idaStarWorker {
 protected:
  CkGroupID Master;
  problem *Issue;
  problem *Solution;
  SerialTree *Solver;
  int Waiting;

 public:
  idaStarWorker(CkGroupID master,
  		problem *issue, int maxdepth,
                int charesize, int serialdist);
  ~idaStarWorker();
  void ChildFinished(int dummy);
};


// Group who coordinates the chares above
class idaStarGroup : public CBase_idaStarGroup {
 protected:
 public:
  unsigned int NodesExpanded, CharesExpanded;
  problem *Issue;
  problem *Solution;
  unsigned int BestSolutionDepth;
  CkCallback Finished;
  int StartDepth, CurrentDepth, MaxDepth, Stride;
  int Running;
  int ChareSize, SerialDist;

 protected:
  void Launch(problem *it, int maxdepth, int charesize, int serialdist);

 public:
  idaStarGroup(problem *issue,
               int startdepth, int maxdepth, int stride, int window,
	       int charesize, int serialdist,
               CkCallback finished);
  ~idaStarGroup();

  // Non-user functions
  void ChildFinished(int dummy);
  void ReductionResults(CkReductionMsg *m);
  void SolutionFound(problem *soln);
  void SpawnIteration();
  void Terminate();

  // Let an idaStarWorker just access my CharesExpanded and NodesExpanded
  // variables to update them.
  friend class idaStarWorker;
};


#endif /* __UIUC_CHARM_ASTAR_H */
