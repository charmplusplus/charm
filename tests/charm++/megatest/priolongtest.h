#ifndef _PRIOLONGTEST_H
#define _PRIOLONGTEST_H

#include "priolongtest.decl.h"
#include "megatest.h"

#define NMSG 100

/* slightly hacked up version of milind's priotest */

class priolongtest_msg : public CMessage_priolongtest_msg
{
  public:
    CmiInt8 prio;
    priolongtest_msg(CmiInt8 p) : prio(p) {}
};

class priolongtest_chare : public CBase_priolongtest_chare
{
  private:
    CmiInt8 lastprio;
  public:
    priolongtest_chare(void);
    priolongtest_chare(CkMigrateMessage *m) {}
    void recv(priolongtest_msg *);
};

#endif
