/*
 * Parallel state space search library
 *
 * Jonathan A. Booth
 * Sun Mar  2 22:39:32 CST 2003
 */

#include "cklibs/problem.h"


problem::problem()
: Priority(),
  Root(1),
  PUP::able()
{
}

problem::problem(const problem &p)
: Priority(p.Priority),
  Root(p.Root),
  PUP::able(p)
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
