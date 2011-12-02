#ifndef GROUPRING_H
#define GROUPRING_H

#include "megatest.h"
#include "nodering.decl.h"

#define NITER 100

extern readonly<CkGroupID> nodering_thisgroup;

class nodering_message : public CMessage_nodering_message {
 public:
  int seqnum;
  int iter;
  nodering_message(int s, int i) : seqnum(s), iter(i) {}
};

class nodering_main : public CBase_nodering_main {
  public:
    nodering_main(CkArgMsg *m) {
      delete m;
      nodering_thisgroup = CProxy_nodering_group::ckNew();
    }
    nodering_main(CkMigrateMessage *m) {}
};

class nodering_group : public CBase_nodering_group {
  public:
    nodering_group(void) {}
    nodering_group(CkMigrateMessage *m) {}
    void start(nodering_message *msg);
};

#endif 
