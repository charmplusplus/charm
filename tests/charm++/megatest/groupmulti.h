#ifndef _GROUPCAST_H
#define _GROUPCAST_H

#include "megatest.h"
#include "groupmulti.decl.h"

class groupmulti_SimpleMsg : public CMessage_groupmulti_SimpleMsg {
public:
  CkGroupID gid;
  CkChareID mainhandle;
};

class groupmulti_BCMsg : public CMessage_groupmulti_BCMsg {
private:
  double dval;
  char checkString[11];
public:
  groupmulti_BCMsg(void) { dval = 3.145; sprintf(checkString, "Multicast!");} 
  groupmulti_BCMsg(CkMigrateMessage *m) {}
  int check(void) { 
    if (strcmp("Multicast!", checkString) == 0 && dval == 3.145)
      return 1;
    else
      return 0;
  }
};

class groupmulti_main : public CBase_groupmulti_main {
private:
  CkGroupID gid;
  int count;
public:
  groupmulti_main(void);
  groupmulti_main(CkMigrateMessage *m) {}
  void groupReady(void);
};

class groupmulti_group : public CBase_groupmulti_group {
private:
  CkChareID myMain;
public:
  groupmulti_group(groupmulti_SimpleMsg *sm);
  groupmulti_group(CkMigrateMessage *m) {}
  void doBroadcast(groupmulti_BCMsg *bcm);
};

#endif
