#ifndef _SCHED_H_
#define _SCHED_H_

#include "sched.decl.h"

class main : public Chare {
  int maxObjects;
  long int n;
  int connectivity;
  public:
  main(CkArgMsg *m);
  main(CkMigrateMessage *) {}
};

#endif
