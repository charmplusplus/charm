/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _WAITQD_H
#define _WAITQD_H

#include "charm++.h"
#include "waitqd.decl.h"

extern "C" void CkWaitQD(void);

class ckGroupIDMsg : public CMessage_ckGroupIDMsg {
  public:
    CkGroupID gid;
    ckGroupIDMsg(CkGroupID _g) : gid(_g) {}
};

class ckGroupCreateMsg : public CMessage_ckGroupCreateMsg {
  public:
    int cidx;
    int consIdx;
    void *msg;
    ckGroupCreateMsg(int _c, int _cc, void *_m) :
      cidx(_c), consIdx(_cc), msg(_m) {}
};

class waitqd_QDChare : public Chare {
 private:
   int waitStarted;
   void *threadList;
 public:
   waitqd_QDChare(CkArgMsg *ckam);
   waitqd_QDChare(CkMigrateMessage *m) {}
   void waitQD(void);
   void onQD(CkQdMsg *ckqm);
};

class waitGC_chare : public Chare {
  private:
    CthThread thr;
    ckGroupIDMsg *mgid;
    int retEP;
  public:
    waitGC_chare(void) { 
      thr=0; mgid=0;
      retEP = CProxy_waitGC_chare::ckIdx_recvGroupID((ckGroupIDMsg *)0);
    }
    waitGC_chare(CkMigrateMessage *m) {}
    ckGroupIDMsg *createGroup(ckGroupCreateMsg *msg) {
      mgid = 0;
      CkCreateGroup(msg->cidx, msg->consIdx, msg->msg, retEP, &thishandle);
      delete msg;
      thr = CthSelf();
      CthSuspend();
      return mgid;
    }
    ckGroupIDMsg *createNodeGroup(ckGroupCreateMsg *msg) {
      mgid = 0;
      CkCreateNodeGroup(msg->cidx, msg->consIdx, msg->msg, retEP, &thishandle);
      delete msg;
      thr = CthSelf();
      CthSuspend();
      return mgid;
    }
    void recvGroupID(ckGroupIDMsg *msg) {
      mgid = msg;
      CthAwaken(thr);
      thr=0;
    }
};

#endif
