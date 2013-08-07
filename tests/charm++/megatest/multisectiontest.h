#ifndef _MULTISECTIONTEST_H
#define _MULTISECTIONTEST_H

#include "multisectiontest.decl.h"
#include "megatest.h"

#define NITER 4

/* this tests that cross section multicast works for 
   group 
   array1d
 */
/* we initialize them by telling everyone the ckIDs of their partners
   in a post startup initialization message sent to sections we built
   in main.  

   So we test cross section logic from main, and then every iteration
   the low half of all groups sends to a high half section of all
   arrays whle high half of groups send to low half of all arrays. The
   arrays reverse the process onto the groups.  The groups contribute
   to a reduction to the master when they've received messages from
   all their array partners.

   This bootstrapping scheme also avoids using globals.

   For simplicity everyone reduces to the 0th element of the master group
*/

class multisectiontest_msg : public CMessage_multisectiontest_msg, CkMcastBaseMsg
{
  public:
  int iterationCount;
  bool side;
};

class multisectionAID_msg : public CMessage_multisectionAID_msg, CkMcastBaseMsg
{
  public:
  CkArrayID *IDs;
  int numIDs;

};

class multisectionGID_msg : public CMessage_multisectionGID_msg, CkMcastBaseMsg
{
  public:
  CkGroupID *IDs;
  int numIDs;

};


class multisection_proxymsg : public CMessage_multisectionGID_msg, CkMcastBaseMsg
{
 public:
  CProxySection_multisectiontest_array1d aproxy;
  CProxySection_multisectiontest_grp gproxy;
};


/* this could be a chare, but singleton chares are unusual and
   would make the code look weirder than necessary */

class multisectiontest_master : public CBase_multisectiontest_master
{
  private:
    int msgCount;
    int iteration;
    int numgroups;
    int startCount;
    CkGroupID *groupIDs;
  public:
    multisectiontest_master(int groups);
    multisectiontest_master(CkMigrateMessage *m) {}
    void recvID(multisectionGID_msg *msg);
    void doneIteration(CkReductionMsg *);
    void doneSetup(CkReductionMsg *);
    void finishSetup();
};

class multisectiontest_grp : public CBase_multisectiontest_grp
{
  private:
    bool low;
    int boundary;
    int floor;
    int ceiling;
    int sectionSize;
    int aboundary;
    int afloor;
    int aceiling;
    int asectionSize;
    int msgCount;
    int iteration;
    CkGroupID masterGroup;
    CProxySection_multisectiontest_array1d multisectionProxy;
  public:
    multisectiontest_grp(int, CkGroupID master);
    multisectiontest_grp(CkMigrateMessage *m) {}
    void recvSProxy(   multisection_proxymsg *m);
    void recvID(multisectionAID_msg *msg);
    void recv(multisectiontest_msg *);
    void nextIteration(multisectiontest_msg *);
};

class multisectiontest_array1d : public CBase_multisectiontest_array1d
{
  private:
  bool gotIds;
    bool low;
    int boundary;
    int floor;
    int ceiling;
    int sectionSize;
    int aboundary;
    int afloor;
    int aceiling;
    int asectionSize;
    int msgCount;
    int iteration;
    CkGroupID masterGroup;
    CProxySection_multisectiontest_grp multisectionProxy;
  public:
    multisectiontest_array1d(CkGroupID master);
    multisectiontest_array1d(CkMigrateMessage *m) {}
    void recvID(multisectionGID_msg *);
    void recvSProxy(   multisection_proxymsg *m);
    void recv(multisectiontest_msg *);
};

#endif
