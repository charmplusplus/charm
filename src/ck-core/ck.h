/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CK_H_
#define _CK_H_

#include <string.h>
#include <stdlib.h>

#include "charm++.h"

#include "envelope.h"
#include "qd.h"
#include "register.h"
#include "stats.h"
#include "ckfutures.h"

#ifndef CMK_OPTIMIZE
#define _CHECK_VALID(p, msg) do {if((p)==0){CkAbort(msg);}} while(0)
#else
#define _CHECK_VALID(p, msg) do { } while(0)
#endif

class VidBlock {
    enum VidState {FILLED, UNFILLED};
    VidState state;
    PtrQ *msgQ;
    CkChareID actualID;
    void msgDeliver(envelope *env) {
        env->setSrcPe(CkMyPe());           
        env->setMsgtype(ForChareMsg);
        env->setObjPtr(actualID.objPtr);
        CldEnqueue(actualID.onPE, env, _infoIdx);
        CpvAccess(_qd)->create();      
    }
  public:
    VidBlock() ;
    void send(envelope *env) {
      if(state==UNFILLED) {
        msgQ->enq((void *)env);
      } else {
        msgDeliver(env);
      }
    }
    void fill(int onPE, void *oPtr) {
      state = FILLED;
      actualID.onPE = onPE;
      actualID.objPtr = oPtr;
      envelope *env;
      while(NULL!=(env=(envelope*)msgQ->deq())) {
        msgDeliver(env);
      }
      delete msgQ; msgQ=0;
    }
    void *getLocalChare(void) {
      if (state==FILLED && actualID.onPE==CkMyPe()) 
          return actualID.objPtr;
      return NULL;
    }
};

extern void _processHandler(void *);
extern void _infoFn(void *msg, CldPackFn *pfn, int *len,
                    int *queueing, int *priobits, UInt **prioptr);
extern void _createGroupMember(CkGroupID groupID, int eIdx, void *env);
extern void _createNodeGroupMember(CkGroupID groupID, int eIdx, void *env);
extern void _createGroup(CkGroupID groupID, envelope *env);
extern void _createNodeGroup(CkGroupID groupID, envelope *env);
#endif
