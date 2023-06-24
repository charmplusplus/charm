/*
 * Parallel state space search library
 *
 * Jonathan A. Booth
 * Sun Mar  2 22:39:32 CST 2003
 */

#include "cklibs/problem.h"


problem::problem()
: PUP::able(),
  Root(1),
  Priority()
{
}

problem::problem(const problem &p)
: PUP::able(p),
  Root(p.Root),
  Priority(p.Priority)
{
}

problem::problem(CkMigrateMessage *m)
: PUP::able(m)
{
}



int problem::depth() {
  return 0;
}

int problem::depthToSolution() {
  return (1<<(sizeof(int)*8-2))-1;
}

void problem::print() {
}

void problem::pup(PUP::er &p) {
  p|Parent;
  p|Root;
  p|Priority;
}
