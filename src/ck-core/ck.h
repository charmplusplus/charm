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
#include <math.h>
#include "charm++.h"
#include "envelope.h"
#include "qd.h"
#include "register.h"
#include "stats.h"
#include "ckfutures.h"
#include "charisma.h"

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


// All the state that's useful to have on the receive side in the Charm Core (ck.C)
class CkCoreState {
	GroupTable *groupTable;
	QdState *qd;
public:
	CkCoreState() 
		:groupTable(&CkpvAccess(_groupTable)), 
		 qd(CpvAccess(_qd)) {}
	
	inline GroupTable *getGroupTable() {
#ifdef __BLUEGENE__ 
		return &CkpvAccess(_groupTable);
#else
 		return groupTable;
#endif
	}
	inline IrrGroup *localBranch(CkGroupID gID) {
#ifdef __BLUEGENE__
		return CkpvAccess(_groupTable).find(gID).getObj();
#else
		return groupTable->find(gID).getObj();
#endif
	}
	
	inline QdState *getQD() {return qd;}
	inline void process(int n=1) {qd->process(n);}
};

CkpvExtern(CkCoreState *, _coreState);


extern void _processHandler(void *converseMsg,CkCoreState *ck);
extern void _processBocInitMsg(CkCoreState *ck,envelope *msg);
extern void _processNodeBocInitMsg(CkCoreState *ck,envelope *msg);
extern void _infoFn(void *msg, CldPackFn *pfn, int *len,
                    int *queueing, int *priobits, UInt **prioptr);
extern void _createGroupMember(CkGroupID groupID, int eIdx, void *env);
extern void _createNodeGroupMember(CkGroupID groupID, int eIdx, void *env);
extern void _createGroup(CkGroupID groupID, envelope *env);
extern void _createNodeGroup(CkGroupID groupID, envelope *env);
extern int _getGroupIdx(int,int,int);
#endif
