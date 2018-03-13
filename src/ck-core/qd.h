#ifndef _QD_H
#define _QD_H

#include "ckcallback.h"
#include "envelope.h"
#include "init.h"

class QdMsg;
class QdCallback;

class QdState {
  private:
    int stage; // 0..2
    char cDirty;
    CmiInt8 oProcessed;
    CmiInt8 mCreated, mProcessed;
    CmiInt8 cCreated, cProcessed;
    int nReported;
    PtrQ *callbacks;
    int nChildren;
    int parent;
    std::vector<int> children;
  public:
    CmiInt8 oldCount;

    QdState():stage(0),mCreated(0),mProcessed(0),nReported(0) {
      cCreated = 0; cProcessed = 0; cDirty = 0;
      oProcessed = 0;
      oldCount = -1; // should not match the first time
      callbacks = new PtrQ();
      _MEMCHECK(callbacks);
      nChildren = CmiNumSpanTreeChildren(CmiMyPe());
      parent = CmiSpanTreeParent(CmiMyPe());
      if (nChildren != 0) {
	children.resize(nChildren);
	_MEMCHECK(children.data());
	CmiSpanTreeChildren(CmiMyPe(), children.data());
      }
      /* CmiPrintf("[%d] n:%d parent:%d - %d %d %d %d %d %d.\n", CmiMyPe(), nChildren, parent, nChildren?children[0]:-1, nChildren?children[1]:-1, nChildren?children[2]:-1, nChildren?children[3]:-1, nChildren?children[4]:-1, nChildren?children[5]:-1); */
    }
    void propagate(QdMsg *msg) {
      envelope *env = UsrToEnv((void *)msg);
      CmiSetHandler(env, _qdHandlerIdx);
      for(int i=0; i<nChildren; i++) {
#if CMK_BIGSIM_CHARM
        CmiSyncSendFn(children[i], env->getTotalsize(), (char *)env);
#else
        CmiSyncSend(children[i], env->getTotalsize(), (char *)env);
#endif
      }
    }
    int getParent(void) { return parent; }
    QdCallback *deq(void) { return (QdCallback*) callbacks->deq(); }
    void enq(QdCallback *c) { callbacks->enq((void *) c); }
    void create(int n=1) { 
        mCreated += n; 
#if CMK_IMMEDIATE_MSG
        sendCount(false, n);
#endif
    }
    void sendCount(bool isCreated, int count);     // send msg to rank 0 for counting
    void process(int n=1) { 
         mProcessed += n; 
#if CMK_IMMEDIATE_MSG
        sendCount(true, n);
#endif
    }
    CmiInt8 getCreated(void) { return mCreated; }
    CmiInt8 getProcessed(void) { return mProcessed; }
    CmiInt8 getCCreated(void) { return cCreated; }
    CmiInt8 getCProcessed(void) { return cProcessed; }
    void subtreeCreate(CmiInt8 c) { cCreated += c; }
    void subtreeProcess(CmiInt8 p) { cProcessed += p; }
    int getStage(void) { return stage; }
    void setStage(int p) { stage = p; }
    void reported(void) { nReported++; }
    int allReported(void) {return nReported==(nChildren+1);}
    void reset(void) { nReported=0; cCreated=0; cProcessed=0; cDirty=0; }
    void markProcessed(void) { oProcessed = mProcessed; }
    int isDirty(void) { return ((mProcessed > oProcessed) || cDirty); }
    void subtreeSetDirty(char d) { cDirty = cDirty || d; }
    void flushStates() {
      stage = mCreated = mProcessed = nReported = 0;
      cCreated = 0; cProcessed = 0; cDirty = 0;
      oProcessed = 0;
    }
};

extern void _qdHandler(envelope *);
extern void _qdCommHandler(envelope *);
CpvExtern(QdState*, _qd);

#endif
