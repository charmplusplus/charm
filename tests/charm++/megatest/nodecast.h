#ifndef _GROUPCAST_H
#define _GROUPCAST_H

#include "megatest.h"
#include "nodecast.decl.h"

class nodecast_SimpleMsg : public CMessage_nodecast_SimpleMsg {
public:
  CkGroupID gid;
  CkChareID mainhandle;
};

class nodecast_BCMsg : public CMessage_nodecast_BCMsg {
private:
  char checkString[11];
public:
  nodecast_BCMsg(void) { sprintf(checkString, "Broadcast!");} 
  int check(void) { 
    if (strcmp("Broadcast!", checkString) == 0)
      return 1;
    else
      return 0;
  }
};

class nodecast_main : public CBase_nodecast_main {
private:
  CkGroupID gid;
  int count;
public:
  nodecast_main(void);
  nodecast_main(CkMigrateMessage *m) {}
  void reply(nodecast_SimpleMsg *em);
};

class nodecast_group : public CBase_nodecast_group {
private:
  CkChareID myMain;
public:
  nodecast_group(nodecast_SimpleMsg *sm);
  nodecast_group(CkMigrateMessage *m) {}
  void doBroadcast(nodecast_BCMsg *bcm);
};

#endif
