/*
 * Parallel state space search library
 *
 * Jonathan A. Booth
 * Fri Mar 28 19:00:04 CST 2003
 */

#include "cklibs/idastar.h"
#include "cklibs/search.h"

#define min(a,b) (a<b?a:b)
#define max(a,b) (a>b?a:b)



/* ************************************************************************
 * aStar
 *
 * This function fakes having an aStar search using the idaStar. All it has
 * to do is set the start depth = end depth = max depth for the aStar search.
 * ************************************************************************ */
void aStar(
  problem *issue,
  int maxdepth, int charesize, int serialdist,
  CkCallback finished
) {
  idaStar(issue, maxdepth, maxdepth, 1, 1,
	  charesize, serialdist, finished);
}



/* ************************************************************************
 * idaStar
 *
 * This function makes spawning an idaStar a bit prettier than usual. It
 * removes the need to do the ugly CProxy_idaStar::ckNew garbage.
 * ************************************************************************ */
void idaStar(
  problem *issue,
  int startdepth, int maxdepth, int stride, int window,
  int charesize, int serialdist,
  CkCallback finished
) {
  CProxy_idaStarGroup::ckNew(issue, startdepth, maxdepth, stride, window,
		             charesize, serialdist, finished);
}



/* ************************************************************************
 * idaStarWorker
 *
 * A worker class (chare) that actually does the idaStar search. It uses the
 * idaStar class (group) as a coordinating master, that helps it track nodes
 * expanded, and other similar work.
 * ************************************************************************ */
idaStarWorker::idaStarWorker(
  CkGroupID master,
  problem *issue, int maxdepth,
  int charesize, int serialdist
) : Master(master), Issue(issue), Waiting(0), Solver(NULL), Solution(NULL) {
  // If we're out of our depth, abort
  if ( maxdepth < issue->depth() + issue->depthToSolution() ) {
    ChildFinished(0);
    return;
  }

  // Catch a local plug to the master
  CProxy_idaStarGroup gp(master);
  idaStarGroup *local = gp.ckLocalBranch();
  if ( !local ) {
    ckerr << "A request for a local group branch failed in idaStarWorker::idaStarWorker. Aborting computation." << endl;
    CkExit();
  }

  // If a solution has been found at some depth and I'm searching at a longer
  // depth there is no need to search.
  if ( local->BestSolutionDepth < maxdepth ) {
    ChildFinished(0);
    return;
  }

  // Make sure that a better solution hasn't already been found
  if ( local->BestSolutionDepth < issue->depth() + issue->depthToSolution() ) {
    ChildFinished(0);
    return;
  }

  // What is the maximal depth that this node should search?
  // We may not ever search further than depthleft, so we know that
  // it is a minimum bound. We may not ever search further than
  // charesize as well (the depth of tree within this chare)
  //
  // Alternatly if serialdist is larger than the depth to solution
  // then we ought to run it out even if that's larger than charesize,
  // but we're still bound by depthleft.
  int depthsearched = charesize;
  if ( issue->depthToSolution() < serialdist ) {
    depthsearched = serialdist;
  }
  depthsearched = min( depthsearched, local->BestSolutionDepth-issue->depth() );
  Solver = new SerialTree(issue, depthsearched);
  Solution = Solver->Solution;

  // Update the stats on the master
  local->CharesExpanded++;
  local->NodesExpanded += Solver->Expanded;

  // First things first, send solutions to the master.
  //
  // Note that as this is a local move, the master gains control of
  // the memory and is now responsable for freeing it.
  //
  // Further, if we have solutions, then we shouldn't spawn any children
  // so dequeue and delete all children in the queue.
  if ( Solution != NULL ) {
    CkEntryOptions eo;
    eo.setPriority(1<<31);
    gp.SolutionFound(Solution, &eo);
    ChildFinished(0);
    return;
  }

  // Now we need to figure out what the heck we're going to do with the
  // results. We should fire off any children there were, wait for
  // them to come back and when they all do then we can tell the parent
  // we're done.
  //
  // Note that as this is a local move, the master gains control of
  // the memory and is now responsable for freeing it.
  problem *child;
  Waiting = 0;
  while ( ( child = Solver->Children.deq() ) != NULL ) {
    // Spawning a child that is just going to kill itself takes a lot of time
    // so we'd best make sure it won't do so!
    if ( local->BestSolutionDepth > child->depth()+child->depthToSolution() ) {
      child->Parent = thishandle;
      child->Root = 0;
      local->Launch(child, maxdepth, charesize, serialdist);
      Waiting++;
    } else {
      // throw the kid away rather than spawning it.
      delete child;
    }
  }
  if ( Solver ) {
    delete Solver;
    Solver = NULL;
  }

  // If we weren't waiting on anything -- that is, if we're a terminal
  // node, we should exit.
  if ( Waiting == 0 ) { ChildFinished(0); }
}


idaStarWorker::~idaStarWorker() {
  // Clear out the Solver
  if ( Issue ) { delete Issue; }
  if ( Solver ) { delete Solver; }
}


void idaStarWorker::ChildFinished(int dummy) {
  // A child of mine finished.
  Waiting--;

  // If I'm not waiting anymore, notify my parent and exit
  if ( Waiting <= 0 ) {
    CkEntryOptions eo;
    eo.setPriority(1<<31);
    if ( Issue->Root ) {
      CProxy_idaStarGroup parent(Master);
      parent.ChildFinished(0, &eo);
    } else {
      CProxy_idaStarWorker parent(Issue->Parent);
      parent.ChildFinished(0, &eo);
    }
    delete this;
  }
}



/* ************************************************************************
 * idaStar
 *
 * The idaStar group is what controls the idaStar search done by the workers
 * above. It also coordinates tracking of the number of nodes and
 * chares expanded by the workers. It handles delivering the results back
 * to the user as well.
 * ************************************************************************ */
// Create the group and begin the search.
idaStarGroup::idaStarGroup(problem *issue,
		 int startdepth, int maxdepth, int stride, int window,
		 int charesize, int serialdist,
		 CkCallback finished)
: CharesExpanded(0), NodesExpanded(0), Running(0), Issue(issue),
  ChareSize(charesize), SerialDist(serialdist),
  StartDepth(startdepth), CurrentDepth(startdepth),
  MaxDepth(maxdepth), Stride(stride),
  BestSolutionDepth(maxdepth), Solution(NULL), Finished(finished) {
  // Everyone in the group gets the stuff to do, but only the one on
  // processor zero is actually going to run it.
  Issue->Root = 1;
  for ( int i = 0 ; i < window && CurrentDepth <= MaxDepth ; i++ ) {
    SpawnIteration();
  }
}

// Destroy a group member. Make sure to free the memory.
idaStarGroup::~idaStarGroup() {
  // Chuck any solutions we had, because they aren't important if we're
  // getting deleted.
  if ( Solution != NULL ) {
    delete Solution;
  }
  if ( Issue != NULL ) {
    delete Issue;
  }
}



// Actually runs a problem as an idaStarWorker. By default it will queue
// it up on the current PE. If RANDOM_STARTING_PROC is defined it will
// instead start it up on a random processor.
void idaStarGroup::Launch(problem *it, int maxdepth, int charesize, int serialdist) {
  // Fire it off! Construct the priority and let it rip.
  CkEntryOptions eo;
  eo.setPriority(it->Priority);
  int onPE = CkMyPe();
#ifdef RANDOM_STARTING_PROC
  onPE = CK_PE_ANY;
#endif
  CProxy_idaStarWorker::ckNew(thisgroup, it, maxdepth, charesize, serialdist,
  			    (CkChareID *)NULL, onPE, &eo);

  // Since pupping it will create a copy, delete the old copy I held
  delete it;
}



// The search tree finished if this function is getting called. Collect up
// the stats on all the processors and then send a message to the user
// with the solution, the number of chares expanded, and the number of
// nodes expanded.
void idaStarGroup::ChildFinished(int dummy) {
  Running--;

  // While we're within our maximum depth and we haven't found a solution
  // we should spawn off the next iteration. Either of these qualities
  // flop and we shouldn't.
  if ( CurrentDepth <= MaxDepth && Solution == NULL) {
    SpawnIteration();
    return;
  }

  if ( Running == 0 ) {
    CkCallback result(CkIndex_idaStarGroup::ReductionResults(NULL), thishandle);
    unsigned int local[2];
    local[0] = NodesExpanded;
    local[1] = CharesExpanded;
    contribute(2*sizeof(unsigned int), local, CkReduction::sum_int, result);
  }
}

// This function collects the results from a reduction of all the
// nodes expanded and chares expanded in the computation. Once
// it has that data it can call the finished callback the user
// provided to me with the best solution found, the number of
// chares expanded and the number of nodes expanded.
void idaStarGroup::ReductionResults(CkReductionMsg *m) {
  if ( !m ) {
    CkAbort("idaStar::ReductionResults, got passed a null reduction message!");
  }

  unsigned int *values = (unsigned int *)m->getData();
  searchResults *sr = new searchResults;
  sr->GroupID = thisgroup;
  sr->NodesExpanded = values[0];
  sr->CharesExpanded = values[1];
  Finished.send(sr);
}

// The idaStarWorker search tree found a solution. This method gets called
// via a group broadcast so all nodes in the group know the current best
// solution.
void idaStarGroup::SolutionFound(problem *soln) {
  // Is this solution we found better than our current best solution?
  if ( soln->depth() < BestSolutionDepth ) {
    // Yes! Update the depth, delete the old copy of the solution we
    // held, and set this as the new solution.
    BestSolutionDepth = soln->depth();
    if ( Solution ) {
      delete Solution;
    }
    Solution = soln;
  } else {
    // No! Do no pass go, do not remain in memory.
    delete soln;
  }
}

// Spawn off another iteration of yourself.
void idaStarGroup::SpawnIteration() {
  if ( CkMyPe() == 0 ) {
    problem *child = Issue->clone();

    // Compute the unique priority lead to give to this iteration.
    child->Priority.Resize(0);
    int k = CkBitVector::ilog2((CurrentDepth-StartDepth)/Stride+2)-1, i;

    // Set the first k bits to 1.
    for ( i = 0 ; i < k ; i++ ) {
      child->Priority.Set(i+1);
    }

    // Set the next bit to '0'
    child->Priority.Clear(0);

    // Now add on an expression of where we are, presuming of course
    // we aren't working with a length of 0.
    if ( k != 0 ) {
      CkBitVector choices(((CurrentDepth-StartDepth)/Stride)-(1<<k)+1, 1<<k);
      child->Priority.Concat(choices);
    }

ckerr << "Spawn Iteration " << child->Priority << endl;

    // And fire it off
    Launch(child, CurrentDepth, ChareSize, SerialDist);
  }
  Running++;
  CurrentDepth += Stride;
}

// This causes the group to delete itself. This is somewhat of a futile
// gesture as though the group can delete itself charm won't entirely
// free up the space it used in static array structures.
void idaStarGroup::Terminate() {
  delete this;
}
