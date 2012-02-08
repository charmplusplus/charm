#ifndef _GROUPCAST_H
#define _GROUPCAST_H

#include "megatest.h"
#include "groupcast.decl.h"

class groupcast_SimpleMsg : public CMessage_groupcast_SimpleMsg {
public:
  CkGroupID gid;
  CkChareID mainhandle;
};

class groupcast_BCMsg : public CMessage_groupcast_BCMsg {
private:
  double dval;
  char checkString[11];
public:
  groupcast_BCMsg(void) { dval = 3.145; sprintf(checkString, "Broadcast!");} 
  groupcast_BCMsg(CkMigrateMessage *m) {}
  int check(void) { 
    if (strcmp("Broadcast!", checkString) == 0 && dval == 3.145)
      return 1;
    else
      return 0;
  }
};

class groupcast_main : public CBase_groupcast_main {
private:
  CkGroupID gid;
  int count;
public:
  groupcast_main(void);
  groupcast_main(CkMigrateMessage *m) {}
  void groupReady(void);
};

class groupcast_group : public CBase_groupcast_group {
private:
  CkChareID myMain;
public:
  groupcast_group(groupcast_SimpleMsg *sm);
  groupcast_group(CkMigrateMessage *m) {}
  void doBroadcast(groupcast_BCMsg *bcm);
};

#endif
