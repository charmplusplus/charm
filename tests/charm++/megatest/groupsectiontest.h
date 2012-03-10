#ifndef _GROUPSECTIONTEST_H
#define _GROUPSECTIONTEST_H

#include "groupsectiontest.decl.h"
#include "megatest.h"

#define NITER 4

/* this tests that section multicast works for groups */

/* low half sends to high half and visa versa */

class groupsectiontest_msg : public CMessage_groupsectiontest_msg, CkMcastBaseMsg
{
  public:
  int iterationCount;
  bool side;
};

class groupsectiontest : public CBase_groupsectiontest
{
  private:
    bool low;
    int boundary;
    int floor;
    int ceiling;
    int sectionSize;
    int msgCount;
    int iteration;
    CProxySection_groupsectiontest groupSectionProxy;
    CProxySection_groupsectiontest groupAllProxy;
  public:
    groupsectiontest(void);
    groupsectiontest(CkMigrateMessage *m) {}
    void init();
    void recv(groupsectiontest_msg *);
    void nextIteration(groupsectiontest_msg *);
    void doneIteration(CkReductionMsg *);
};

#endif
