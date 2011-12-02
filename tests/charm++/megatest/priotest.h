#ifndef _PRIOTEST_H
#define _PRIOTEST_H

#include "priotest.decl.h"
#include "megatest.h"

#define NMSG 100

class priotest_msg : public CMessage_priotest_msg
{
  public:
    int prio;
    priotest_msg(int p) : prio(p) {}
};

class priotest_chare : public CBase_priotest_chare
{
  private:
    int lastprio;
  public:
    priotest_chare(void);
    priotest_chare(CkMigrateMessage *m) {}
    void recv(priotest_msg *);
};

#endif
