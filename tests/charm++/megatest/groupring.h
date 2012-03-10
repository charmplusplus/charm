#ifndef GROUPRING_H
#define GROUPRING_H

#include "megatest.h"
#include "groupring.decl.h"

#define NITER 100

class groupring_message : public CMessage_groupring_message {
 public:
  int seqnum;
  int iter;
  groupring_message(int s, int i) : seqnum(s), iter(i) {}
  groupring_message(CkMigrateMessage *m) {}
};

class groupring_main : public CBase_groupring_main {
  public:
    groupring_main(CkArgMsg *m) {
      delete m;
    }
    groupring_main(CkMigrateMessage *m) {}
};

class groupring_group : public CBase_groupring_group {
  public:
    groupring_group(void) {}
    groupring_group(CkMigrateMessage *m) {}
    void start(groupring_message *msg);
};

#endif 
