//////////////////////////////////////////////////
//
//  waitqd.h
//
//  
//
//  Author: Michael Lang
//  Created: 7/15/99
//
//////////////////////////////////////////////////

#ifndef _WAITQD_H
#define _WAITQD_H

#include "charm++.h"
#include "waitqd.decl.h"

extern "C" void CkWaitQD(void);

class ckGroupMsg : public CMessage_ckGroupMsg {
  public:
    CkGroupID gid;
};

class waitqd_QDChare : public Chare {
 private:
   int waitStarted;
   void *threadList;
 public:
   waitqd_QDChare(CkArgMsg *ckam);
   void waitQD(void);
   void onQD(CkQdMsg *ckqm);
};

extern CkGroupID waitGC_gchandle;

class waitGC_group : public Group {
  private:
    CthThread thr;
    CkGroupID gid;
    int retEP;
  public:
    waitGC_group(void) { 
      thr=0; 
      retEP = CProxy_waitGC_group::ckIdx_recvGroupID((ckGroupMsg *)0);
    }
    CkGroupID createGroup(int cidx, int consIdx, void *msg) {
      CkCreateGroup(cidx, consIdx, msg, retEP, &thishandle);
      thr = CthSelf();
      CthSuspend();
      return gid;
    }
    CkGroupID createNodeGroup(int cidx, int consIdx, void *msg) {
      CkCreateNodeGroup(cidx, consIdx, msg, retEP, &thishandle);
      thr = CthSelf();
      CthSuspend();
      return gid;
    }
    void recvGroupID(ckGroupMsg *msg) {
      gid = msg->gid;
      CthAwaken(thr);
      thr=0;
    }
};

#endif
