// File: sim.h
// Module for basic simulation object; incorporates all other modules
// Messages: eventMsg, cancelMsg, prioMsg
// Classes: sim
// Last Modified: 06.04.01 by Terry L. Wilmarth

#ifndef SIM_H
#define SIM_H

#include "sim.decl.h"
//#include <cstdarg>
#include <stdarg.h>
//#include <ansidecl.h>
//#define VA_OPEN(AP, VAR)        { va_list AP; va_start(AP, VAR); { struct Qdmy
//#define VA_CLOSE(AP)            } va_end(AP); }

extern CProxy_sim POSE_Objects;  // a global readonly proxy to the array of
                                 // POSE simulation objects

extern CkChareID POSE_Coordinator_ID;

class sim;                       // needed for eventMsg definition below

// Three basic POSE messages: eventMsg, from which all user messages inherit; 
// cancelMsg, for cancelling events; and prioMsg, a null message.
// All three have a priority field which is set to the timestamp; thus all 
// messages with earlier timestamp have higher priority.

// How events get around; all user event types inherit from this
class eventMsg : public CMessage_eventMsg {
public:
  int timestamp;  // the event's timestamp
  eventID evID;   // the event's globally unique ID
  sim *parent;    // used when creating rep object
  strat *str;     // used when creating rep object
  int msgSize;    // used for message recycling
  int fromPE;     // used for comm-based LB
  eventMsg() { }
  void Timestamp(int t) { 
    timestamp = t; evID = GetEventID(); 
    setPriority(t-INT_MAX); 
  }
  eventMsg& operator=(const eventMsg& obj) {  // assignment copies prio too
    timestamp = obj.timestamp;
    evID = obj.evID;
    parent = obj.parent;
    str = obj.str;
    msgSize = obj.msgSize;
    setPriority(timestamp-INT_MAX);
    return *this;
  }
  void *operator new (size_t size) {  // allocates space for priority too
    //EventMsgPool *localPool = (EventMsgPool *)CkLocalBranch(EvmPoolID);
    //if (localPool->CheckPool(MapSizeToIdx(size)))
    //return localPool->GetEventMsg(MapSizeToIdx(size));
    //else {
    void *msg = CkAllocMsg(CMessage_eventMsg::__idx, size, 8*sizeof(int));
    ((eventMsg *)msg)->msgSize = size;
    return msg;
    //}
  }
  void operator delete(void *p) { 
    //EventMsgPool *localPool = (EventMsgPool *)CkLocalBranch(EvmPoolID);
    //if (localPool->CheckPool(MapSizeToIdx(((eventMsg *)p)->msgSize)) 
    //< MAX_POOL_SIZE) {
    //size_t msgSize = ((eventMsg *)p)->msgSize;
    //memset(p, 0, msgSize);
    //((eventMsg *)p)->msgSize = msgSize;
    //localPool->PutEventMsg(MapSizeToIdx(msgSize), p);
    //}
    //else
      CkFreeMsg(p);
  }
  void setPriority(int prio) {  // sets priority field
    *((int*)CkPriorityPtr(this)) = prio;
    CkSetQueueing(this, CK_QUEUEING_IFIFO);
  }
};

// Note the striking similarity to an eventMsg; but nothing derives from it
class cancelMsg : public CMessage_cancelMsg {
public:
  eventID evID;           // only need this to find the event to cancel
  int timestamp;          // but providing this as well makes finding it faster
  void *operator new (size_t size) {  // allocate with priority field
    return CkAllocMsg(CMessage_cancelMsg::__idx, size, 8*sizeof(int));
  } 
  void operator delete(void *p) {  CkFreeMsg(p);  }
  void setPriority(int prio) {  // sets priority field
    *((int*)CkPriorityPtr(this)) = prio;
    CkSetQueueing(this, CK_QUEUEING_IFIFO);
  }
};

// Prioritized null msg; comes in handy for sorting the Step calls
class prioMsg : public CMessage_prioMsg {
public:
  void *operator new (size_t size) {
    return CkAllocMsg(CMessage_eventMsg::__idx, size, 8*sizeof(int));
  }
  void operator delete(void *p) {  CkFreeMsg(p);  }
  void setPriority(int prio) {
    *((int*)CkPriorityPtr(this)) = prio;
    CkSetQueueing(this, CK_QUEUEING_IFIFO);
  }
};

class destMsg : public CMessage_destMsg {
public:
  int destPE;
};


// The simulation object base class; all user simulation objects are 
// translated to this object, which acts as a wrapper around the actual 
// object to control the simulation behavior.
class sim : public ArrayElement1D {
 protected:
  int active; // set if Step message queued; sync strategy
  int myPVTidx,  myLBidx;  // unique global IDs for this object on PVT and LB
 public:
  int DOs, UNDOs;
  int sync;
  int *srVector;    // number of sends/recvs per PE
  eventQueue *eq;   // the object's event queue
  strat *myStrat;   // the object's simulation strategy
  rep *objID;       // the user's simulated object
  rep *recyc[1];
  int recycCount;
  PVT *localPVT;    // the local PVT to report to
#ifdef POSE_STATS_ON
  localStat *localStats; 
#endif
#ifdef LB_ON
  LBgroup *localLBG;
#endif
  CancelList cancels;  // list of incoming cancellations
  sim(void);
  sim(CkMigrateMessage *) {};
  virtual ~sim();
  virtual void pup(PUP::er &p) {  // pup the entire simulation object
    ArrayElement1D::pup(p);       // call parent class pup method

    //if (p.isUnpacking()) CkPrintf("_%d ", CkMyPe());
    /*
    if (p.isUnpacking())
      CkPrintf("[%d] sim %d is unpacking...\n", CkMyPe(), thisIndex);
    else if (p.isPacking())
      CkPrintf("[%d] sim %d is packing...\n", CkMyPe(), thisIndex);
    else
      CkPrintf("[%d] sim %d is sizing...\n", CkMyPe(), thisIndex);
    */
    // pup simple types
    p(active); p(myPVTidx); p(myLBidx); p(sync); p(DOs); p(UNDOs);

    // pup event queue
    if (p.isUnpacking())
      eq = new eventQueue();
    eq->pup(p);

    // pup cancellations
    cancels.pup(p);

    if (p.isUnpacking()) {        // reactivate migrated object
#ifdef POSE_STATS_ON
      localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
      localPVT = (PVT *)CkLocalBranch(ThePVT);
#ifdef LB_ON
      localLBG = TheLBG.ckLocalBranch();
#endif
      recycCount = active = 0;
      myPVTidx = localPVT->objRegister(thisIndex, localPVT->getGVT(), sync, this);
#ifdef LB_ON
      myLBidx = localLBG->objRegister(thisIndex, sync, this);
#endif
    }
    else if (p.isPacking()) {     // deactivate migrating object
      active = -1;
      localPVT->objRemove(myPVTidx);
#ifdef LB_ON
      localLBG->objRemove(myLBidx);
#endif
    }
  }
  void Step();                 // Creates a forward execution step with myStrat
  void Step(prioMsg *m);       // Creates a prioritized step with myStrat
  void Status();               // Reports SafeTimes to PVT
  void Commit();               // Commit events based on new GVT value
  void Cancel(cancelMsg *m);   // Add m to cancellation list
  void ReportLBdata();
  void Migrate(destMsg *m);
  int PVTindex() { return myPVTidx; }
  int IsActive() { return active; }
  void Activate() { active = 1; }
  void Deactivate() { active = 0; }
  virtual void ResolveFn(int fnIdx, void *msg);          // invoke method w/msg
  virtual void ResolveCommitFn(int fnIdx, void *msg);    // invoke commit w/msg
  void registerSent(int timestamp) { // Notify GVT of message send
    localPVT->objUpdate(timestamp, SEND);
  }
  //  void CommitPrintf VPARAMS ((const char *Fmt, ...)) {
  void CommitPrintf (const char *Fmt, ...) {
    char *tmp;
    size_t tmplen=eq->currentPtr->commitBfrLen + strlen(Fmt) + 1 +100;
    va_list ap;
    if (!(tmp = (char *)malloc(tmplen //
			       * sizeof(char)))) {
      CkPrintf("ERROR: sim::CommitPrintf: OUT OF MEMORY!\n");
      CkExit();
    }
    //VA_FIXEDARG (ap, const char *, Fmt);
    va_start(ap,Fmt);
    if ((eq->currentPtr->commitBfr)&&(eq->currentPtr->commitBfrLen))
      {
	strcpy(tmp,eq->currentPtr->commitBfr);
	free(eq->currentPtr->commitBfr);
	vsnprintf(tmp+strlen(tmp),tmplen,Fmt,  ap); 
      }
    else
      {
	vsnprintf(tmp,tmplen, Fmt,ap ); 
      }
    va_end(ap);
    //      tmp=(char *) realloc(tmp,(strlen(tmp)+1)*sizeof(char));
    eq->currentPtr->commitBfrLen = strlen(tmp) + 1;  
    eq->currentPtr->commitBfr = tmp;
  }
  void CommitPrint(char *s) {       // Buffered output function
    char *tmp;
    if (!(tmp = (char *)malloc((eq->currentPtr->commitBfrLen + strlen(s) + 1) 
			       * sizeof(char)))) {
      CkPrintf("ERROR: sim::CommitPrint: OUT OF MEMORY!\n");
      CkExit();
    }
    if (eq->currentPtr->commitBfr)
      sprintf(tmp, "%s%s", eq->currentPtr->commitBfr, s); 
    else
      sprintf(tmp, "%s", s); 
    eq->currentPtr->commitBfrLen = strlen(tmp) + 1;
    free(eq->currentPtr->commitBfr);
    eq->currentPtr->commitBfr = tmp;
  }
  void dump(int pdb_level);         // dump entire sim object
};

#endif









