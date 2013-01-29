#ifndef _QD_H
#define _QD_H

class QdMsg {
  private:
    int phase; // 0..2
    union {
      struct { /* none */ } p1;
      struct { int created; int processed; } p2;
      struct { /* none */ } p3;
      struct { int dirty; } p4;
    } u;
    CkCallback cb;
  public:
    int getPhase(void) { return phase; }
    void setPhase(int p) { phase = p; }
    CkCallback getCb(void) { CkAssert(phase==0); return cb; }
    void setCb(CkCallback cb_) { CkAssert(phase==0); cb = cb_; }
    int getCreated(void) { CkAssert(phase==1); return u.p2.created; }
    void setCreated(int c) { CkAssert(phase==1); u.p2.created = c; }
    int getProcessed(void) { CkAssert(phase==1); return u.p2.processed; }
    void setProcessed(int p) { CkAssert(phase==1); u.p2.processed = p; }
    int getDirty(void) { CkAssert(phase==2); return u.p4.dirty; }
    void setDirty(int d) { CkAssert(phase==2); u.p4.dirty = d; }
};

class QdCommMsg {
  public:
    int flag;     //  0: create   1: process
    int count;
};

class QdCallback {
  public:
	CkCallback cb;
  public:
    QdCallback(int e, CkChareID c) : cb(e, c) {}
	QdCallback(CkCallback cb_) : cb(cb_) {}
//    void send(void) { CkSendMsg(ep,CkAllocMsg(0,0,0),&cid); }
    void send(void) {
      // pretending pe 0 in blue gene mode, switch back after the call.
#if CMK_CONDS_USE_SPECIAL_CODE
      int old = CmiSwitchToPE(0);
#endif
      cb.send(NULL);
#if CMK_CONDS_USE_SPECIAL_CODE
      CmiSwitchToPE(old);
#endif
    }
};

class QdState {
  private:
    int stage; // 0..2
    int oProcessed;
    int mCreated, mProcessed;
    int cCreated, cProcessed;
    int cDirty;
    int nReported;
    PtrQ *callbacks;
    int nChildren;
    int parent;
    int *children;
  public:
    QdState():stage(0),mCreated(0),mProcessed(0),nReported(0) {
      cCreated = 0; cProcessed = 0; cDirty = 0;
      oProcessed = 0;
      callbacks = new PtrQ();
      _MEMCHECK(callbacks);
      nChildren = CmiNumSpanTreeChildren(CmiMyPe());
      parent = CmiSpanTreeParent(CmiMyPe());
      if (nChildren != 0) {
	children = new int[nChildren];
	_MEMCHECK(children);
	CmiSpanTreeChildren(CmiMyPe(), children);
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
        sendCount(0, n);
#endif
    }
    void sendCount(int flag, int count);     // send msg to rank 0 for counting
    void process(int n=1) { 
         mProcessed += n; 
#if CMK_IMMEDIATE_MSG
        sendCount(1, n);
#endif
    }
    int getCreated(void) { return mCreated; }
    int getProcessed(void) { return mProcessed; }
    int getCCreated(void) { return cCreated; }
    int getCProcessed(void) { return cProcessed; }
    void subtreeCreate(int c) { cCreated += c; }
    void subtreeProcess(int p) { cProcessed += p; }
    int getStage(void) { return stage; }
    void setStage(int p) { stage = p; }
    void reported(void) { nReported++; }
    int allReported(void) {return nReported==(nChildren+1);}
    void reset(void) { nReported=0; cCreated=0; cProcessed=0; cDirty=0; }
    void markProcessed(void) { oProcessed = mProcessed; }
    int isDirty(void) { return ((mProcessed > oProcessed) || cDirty); }
    void subtreeSetDirty(int d) { cDirty = cDirty || d; }
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
