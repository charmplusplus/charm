/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
/**
\file
\addtogroup CkEnvelope
*/
#ifndef _ENVELOPE_H
#define _ENVELOPE_H

#ifndef CkIntbits
#define CkIntbits (sizeof(int)*8)
#endif

#ifndef CMK_OPTIMIZE
#define _SET_USED(env, x) (env)->setUsed((x))
#define _CHECK_USED(env) do { if(env->isUsed()) \
                           CmiAbort("Message being re-sent. Aborting...\n"); \
                         } while(0)
#else
#define _SET_USED(env, x) do{}while(0)
#define _CHECK_USED(env) do{}while(0)
#endif

#define CkMsgAlignmentMask     (sizeof(double)-1)
#define CkMsgAlignLength(x) (((x)+CkMsgAlignmentMask)&(~(CkMsgAlignmentMask)))
#define CkMsgAlignOffset(x)     (CkMsgAlignLength(x)-(x))
#define CkPriobitsToInts(nBits)    ((nBits+CkIntbits-1)/CkIntbits)
// silly ancient name: for backward compatability only.
#define PW(x) CkPriobitsToInts(x) 



// #define USE_CRITICAL_PATH_HEADER_ARRAY

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
// This critical path detection is still experimental
// Added by Isaac (Dec 2008)

// stores the pointer to the currently executing msg
// used in cklocation.C, ck.C
// TODO: convert to CkPv

extern envelope * currentlyExecutingMsg;
extern bool thisMethodSentAMessage;
extern double timeEntryMethodStarted;

// The methods provided by the control point framework to access 
// the most Critical path seen by this PE. 
// These ought not to be called by the user program.
// (defined in controlPoints.C)
extern void resetPECriticalPath();
extern void printPECriticalPath();
extern void registerTerminalEntryMethod();

// Reset the counts for the currently executing message, 
// and also reset the PE's critical path detection
// To be called by the user program
extern void resetCricitalPathDetection();

// Reset the counts for the currently executing message
extern void resetThisEntryPath();


extern void bracketStartCriticalPathMethod(envelope * env);
extern void bracketEndCriticalPathMethod(envelope * env);


#endif


/** static sizes for arrays in PathHistory objects */
#define numEpIdxs 150
#define numArrayIds 20

/** A class that is used to track the entry points and other information 
    about a critical path as a charm++ program executes.

    This class won't do useful things unless USE_CRITICAL_PATH_HEADER_ARRAY is defined

*/
class PathHistory {
 private:
  int epIdxCount[numEpIdxs];
  int arrayIdxCount[numArrayIds];
  double totalTime;

 public:
  
  const int* getEpIdxCount(){
    return epIdxCount;
  }
  
  const int* getArrayIdxCount(){
    return arrayIdxCount;
  }

  int getEpIdxCount(int i) const {
    return epIdxCount[i];
  }
  
  int getArrayIdxCount(int i) const {
    return arrayIdxCount[i];
  }


  double getTotalTime() const{
    return totalTime;
  }

  
  PathHistory(){
    reset();
  }

  void pup(PUP::er &p) {
    for(int i=0;i<numEpIdxs;i++)
      p|epIdxCount[i];
    for(int i=0;i<numArrayIds;i++)
      p|arrayIdxCount[i];
    p | totalTime;
  } 
  
  double getTime(){
    return totalTime;
  }
  
  void reset(){
    // CkPrintf("reset() currentlyExecutingMsg=%p\n", currentlyExecutingMsg);
    
    totalTime = 0.0;
    
    for(int i=0;i<numEpIdxs;i++){
      epIdxCount[i] = 0;
    }
    for(int i=0;i<numArrayIds;i++){
      arrayIdxCount[i]=0;
    }
  }
  
  int updateMax(const PathHistory & other){
    if(other.totalTime > totalTime){
      //	  CkPrintf("[%d] Found a longer terminal path:\n", CkMyPe());
      //	  other.print();
      
      totalTime = other.totalTime;
      for(int i=0;i<numEpIdxs;i++){
	epIdxCount[i] = other.epIdxCount[i];
      }
      for(int i=0;i<numArrayIds;i++){
	arrayIdxCount[i]=other.arrayIdxCount[i];
      }
      return 1;
    }
    return 0;
  }
  
  
  void print() const {
    CkPrintf("Critical Path Time=%lf : ", (double)totalTime);
    for(int i=0;i<numEpIdxs;i++){
      if(epIdxCount[i]>0){
	CkPrintf("EP %d count=%d : ", i, (int)epIdxCount[i]);
      }
    }
    for(int i=0;i<numArrayIds;i++){
      if(arrayIdxCount[i]>0){
	CkPrintf("Array %d count=%d : ", i, (int)arrayIdxCount[i]);
      }
    }
    CkPrintf("\n");
  }

  /// Write a description of the path into the beginning of the provided buffer. The buffer ought to be large enough.
  void printHTMLToString(char* buf) const {
    buf[0] = '\0';

    sprintf(buf+strlen(buf), "Path Time=%lf<br>", (double)totalTime);
    for(int i=0;i<numEpIdxs;i++){
      if(epIdxCount[i]>0){
	sprintf(buf+strlen(buf),"EP %d count=%d<br>", i, (int)epIdxCount[i]);
      }
    }
    for(int i=0;i<numArrayIds;i++){
      if(arrayIdxCount[i]>0){
	sprintf(buf+strlen(buf), "Array %d count=%d<br>", i, (int)arrayIdxCount[i]);
      }
    }
  }
  
  void incrementTotalTime(double time){
    totalTime += time;
  }
  

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  void createPath(PathHistory *parentPath){
    // Note that we are likely sending a message
    // FIXME: (this should be moved to the actual send point)
    thisMethodSentAMessage = true;
    double timeNow = CmiWallTimer();
    
    if(parentPath != NULL){
      //	  CkPrintf("createPath() totalTime = %lf + %lf\n",(double)currentlyExecutingMsg->pathHistory.totalTime, (double)timeNow-timeEntryMethodStarted);
      totalTime = parentPath->totalTime + (timeNow-timeEntryMethodStarted);
      
      for(int i=0;i<numEpIdxs;i++){
	epIdxCount[i] = parentPath->epIdxCount[i];
      }
      for(int i=0;i<numArrayIds;i++){
	arrayIdxCount[i] = parentPath->arrayIdxCount[i];
      }
      
    }
    else {
      totalTime = 0.0;
      
      for(int i=0;i<numEpIdxs;i++){
	epIdxCount[i] = 0;
      } 
      for(int i=0;i<numArrayIds;i++){
	arrayIdxCount[i]=0;
      }
    }
	
  }
#endif
      
  void incrementEpIdxCount(int ep){
    epIdxCount[ep]++;
  }

  void incrementArrayIdxCount(int arr){
    arrayIdxCount[arr]++;
  }
      
};




/**
 This set of message type (mtype) constants
 defines the basic class of charm++ message.
 
 It is very questionable whether bizarre stuff like
 "ExitMsg", "StatMsg", "ROMsgMsg" should actually
 share the envelope with regular user messages;
 but it doesn't waste any space so it's probably OK.
*/
typedef enum {
  NewChareMsg    =1,
  NewVChareMsg   =2,
  BocInitMsg     =3,
  ForChareMsg    =4,
  ForBocMsg      =5,
  ForVidMsg      =6,
  FillVidMsg     =7,
  RODataMsg      =8,
  ROMsgMsg       =9,
  ExitMsg        =10,
  ReqStatMsg     =11,
  StatMsg        =12,
  NodeBocInitMsg =13,
  ForNodeBocMsg  =14,
  ArrayEltInitMsg =15,
  ForArrayEltMsg  =16
} CkEnvelopeType;

typedef unsigned int   UInt;
typedef unsigned short UShort;
typedef unsigned char  UChar;

#include "charm.h" // for CkGroupID
#ifdef _FAULT_MLOG_
#include "ckobjid.h" //for the ckobjId
#endif

/**
The "envelope" sits at the start of every Charm++
message. It stores information about the handler and
destination of the charm++ message that follows, and 
what to do with it on the receiving side.

A Charm++ message's memory layout has the 
Charm envelope ("envelope" class) first, which includes
a Converse envelope as its **first** field.  After the 
Charm envelope is a variable-length amount of user 
data, and finally the priority data stored as ints.

The converse layers modify the beginning of the envelope
without referencing the "envelope::core" member. The
envelope is treated as a void*, and the first 
CmiReservedHeaderSize bytes are available for the 
converse functions. Therefore, do not put any members
at the beginning of the envelope class.

<pre>
 Envelope pointer        \
 Converse message pointer -> [ [ Converse envelope ]       ]
                             [       Charm envelope        ] 
 User message pointer     -> [ User data ... ]
 Priority pointer         -> [ Priority ints ... ]
</pre>

The "message pointers" passed to and from
users bypass the envelope and point *directly* to the 
user data--the routine "EnvToUsr" below adjusts an 
envelope (or converse message) pointer into this 
direct-to-user pointer.  There is a corresponding
routine "UsrToEnv" which takes the user data pointer
and returns a pointer to the envelope/converse message.

Unfortunately, in the guts of Charm++ it's not always 
clear whether you've been given a converse or user
message pointer, as both tend to be passed as void *.
Confusing the two will invariably result in data 
corruption and bizarre crashes.

FIXME: Make CkMessage inherit from envelope,
which would unify converse, envelope, and 
user message pointers.
*/
class envelope {
  private:
    /// Converse message envelope, Must be first field in this class
    char   core[CmiReservedHeaderSize];
public:
 /**
   This union stores the type-specific message information.
   Keeing this in a union allows the different kinds of messages 
   to have different fields/types, in an alignment-safe way, 
   without wasting any storage.
 */
    union u_type {
      struct s_chare { //NewChareMsg, NewVChareMsg, ForChareMsg, ForVidMsg, FillVidMsg
      	void *ptr;
      	UInt forAnyPe; ///< Used only by newChare
      } chare;
      struct s_group {
	CkGroupID g; ///< GroupID
	CkNodeGroupID rednMgr; ///< Reduction manager for this group (constructor only!)
	int epoch; ///< "epoch" this group was created during (0--mainchare, 1--later)
	UShort arrayEp; ///< Used only for array broadcasts
      } group;
      struct s_array{ ///< For arrays only
	CkArrayIndexStruct index;///< Array element index
	int listenerData[CK_ARRAYLISTENER_MAXLEN]; ///< For creation
	CkGroupID arr; ///< Array manager GID
	UChar hopCount;///< number of times message has been routed
    	UChar ifNotThere; ///< what to do if array element is missing
      } array;
      struct s_roData { ///< RODataMsg
      	UInt count;
      } roData;
      struct s_roMsg { ///< ROMsgMsg
      	UInt roIdx;
      } roMsg;
    };
    struct s_attribs { //Packed bitwise struct
    	UChar msgIdx; ///< Usertype of message (determines pack routine)
	UChar mtype; ///< e.g., ForBocMsg
    	UChar queueing:4; ///< Queueing strategy (FIFO, LIFO, PFIFO, ...)
    	UChar isPacked:1; ///< If true, message must be unpacked before use
    	UChar isUsed:1; ///< Marker bit to prevent message re-send.
    };
#ifdef _FAULT_MLOG_
        CkObjID sender;
        CkObjID recver;
        MCount SN;
        MCount TN;
        MlogEntry *localMlogEntry;
#endif
private:
    u_type type; ///< Depends on message type (attribs.mtype)
    UShort ref; ///< Used by futures
    s_attribs attribs;
    UChar align[CkMsgAlignOffset(CmiReservedHeaderSize+sizeof(u_type)+sizeof(UShort)+sizeof(s_attribs))];
    
    //This struct should now be sizeof(void*) aligned.
    UShort priobits; ///< Number of bits of priority data after user data
    UShort epIdx;  ///< Entry point to call
    UInt   pe;    ///< source processor
    UInt   event; ///< used by projections
    UInt   totalsize; ///< Byte count from envelope start to end of priobits
    
  public:
#ifdef _FAULT_MLOG_
        UInt piggyBcastIdx;
#endif
    void pup(PUP::er &p);
    UInt   getEvent(void) const { return event; }
    void   setEvent(const UInt e) { event = e; }
    UInt   getRef(void) const { return ref; }
    void   setRef(const UShort r) { ref = r; }
    UChar  getQueueing(void) const { return attribs.queueing; }
    void   setQueueing(const UChar q) { attribs.queueing=q; }
    UChar  getMsgtype(void) const { return attribs.mtype; }
    void   setMsgtype(const UChar m) { attribs.mtype = m; }
#ifndef CMK_OPTIMIZE
    UChar  isUsed(void) { return attribs.isUsed; }
    void   setUsed(const UChar u) { attribs.isUsed=u; }
#else /* CMK_OPTIMIZE */
    inline void setUsed(const UChar u) {}
#endif
    UChar  getMsgIdx(void) const { return attribs.msgIdx; }
    void   setMsgIdx(const UChar idx) { attribs.msgIdx = idx; }
    UInt   getTotalsize(void) const { return totalsize; }
    void   setTotalsize(const UInt s) { totalsize = s; }
    UInt   getUsersize(void) const { return totalsize - priobits - sizeof(envelope); }
    UChar  isPacked(void) const { return attribs.isPacked; }
    void   setPacked(const UChar p) { attribs.isPacked = p; }
    UShort getPriobits(void) const { return priobits; }
    void   setPriobits(const UShort p) { priobits = p; }
    UShort getPrioWords(void) const { return CkPriobitsToInts(priobits); }
    UShort getPrioBytes(void) const { return getPrioWords()*sizeof(int); }
    void*  getPrioPtr(void) const { 
      return (void *)((char *)this + totalsize - getPrioBytes());
    }
    static envelope *alloc(const UChar type, const UInt size=0, const UShort prio=0)
    {
      CkAssert(type>=NewChareMsg && type<=ForArrayEltMsg);
      register UInt tsize = sizeof(envelope)+ 
            CkMsgAlignLength(size)+
	    sizeof(int)*CkPriobitsToInts(prio);
      register envelope *env = (envelope *)CmiAlloc(tsize);
      env->setMsgtype(type);
      env->totalsize = tsize;
      env->priobits = prio;
      env->setPacked(0);
      _SET_USED(env, 0);
      //for record-replay
      env->setEvent(0);
#ifdef _FAULT_MLOG_
            env->sender.type = TypeInvalid;
            env->recver.type = TypeInvalid;
            env->SN = 0;
            env->TN = 0;
            env->localMlogEntry = NULL;
#endif
      return env;
    }
    UShort getEpIdx(void) const { return epIdx; }
    void   setEpIdx(const UShort idx) { epIdx = idx; }
    UInt   getSrcPe(void) const { return pe; }
    void   setSrcPe(const UInt s) { pe = s; }
    static void setSrcPe(char *env, const UInt s) { ((envelope*)env)->setSrcPe(s); }

// Readonly-specific fields
    UInt   getCount(void) const { 
      CkAssert(getMsgtype()==RODataMsg); return type.roData.count; 
    }
    void   setCount(const UInt c) { 
      CkAssert(getMsgtype()==RODataMsg); type.roData.count = c; 
    }
    UInt   getRoIdx(void) const { 
      CkAssert(getMsgtype()==ROMsgMsg); return type.roMsg.roIdx; 
    }
    void   setRoIdx(const UInt r) { 
      CkAssert(getMsgtype()==ROMsgMsg); type.roMsg.roIdx = r; 
    }
    
 // Chare-specific fields
    UInt isForAnyPE(void) { 
      CkAssert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg); 
      return type.chare.forAnyPe; 
    }
    void setForAnyPE(UInt f) { 
      CkAssert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg); 
      type.chare.forAnyPe = f; 
    }
    void*  getVidPtr(void) const {
      CkAssert(getMsgtype()==NewVChareMsg || getMsgtype()==ForVidMsg
          || getMsgtype()==FillVidMsg);
      return type.chare.ptr;
    }
    void   setVidPtr(void *p) {
      CkAssert(getMsgtype()==NewVChareMsg || getMsgtype()==ForVidMsg
          || getMsgtype()==FillVidMsg);
      type.chare.ptr = p;
    }
    void*  getObjPtr(void) const { 
      CkAssert(getMsgtype()==ForChareMsg); return type.chare.ptr; 
    }
    void   setObjPtr(void *p) { 
      CkAssert(getMsgtype()==ForChareMsg); type.chare.ptr = p; 
    }

// Group-specific fields
    CkGroupID   getGroupNum(void) const {
      CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==ForBocMsg
          || getMsgtype()==NodeBocInitMsg || getMsgtype()==ForNodeBocMsg);
      return type.group.g;
    }
    void   setGroupNum(const CkGroupID g) {
      CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==ForBocMsg
          || getMsgtype()==NodeBocInitMsg || getMsgtype()==ForNodeBocMsg);
      type.group.g = g;
    }
    void setGroupEpoch(int epoch) { type.group.epoch=epoch; }
    int getGroupEpoch(void) { return type.group.epoch; }
    void setRednMgr(CkNodeGroupID r){ type.group.rednMgr = r; }
    CkNodeGroupID getRednMgr(){ return type.group.rednMgr; }

// Array-specific fields
    CkGroupID &getsetArrayMgr(void) {return type.array.arr;}
    UShort &getsetArrayEp(void) {return epIdx;}
    UShort &getsetArrayBcastEp(void) {return type.group.arrayEp;}
    UInt &getsetArraySrcPe(void) {return pe;}
    UChar &getsetArrayHops(void) {return type.array.hopCount;}
    int getArrayIfNotThere(void) {return type.array.ifNotThere;}
    void setArrayIfNotThere(int nt) {type.array.ifNotThere=nt;}
    int *getsetArrayListenerData(void) {return type.array.listenerData;}
    CkArrayIndexMax &getsetArrayIndex(void) 
    	{return *(CkArrayIndexMax *)&type.array.index;}

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
 public:

    /// The information regarding the entry methods that executed along the path to this one.
    PathHistory pathHistory;

    void resetEpIdxHistory(){
      pathHistory.reset();
    }

    /** Fill in the values for the path When creating a message in an entry method, 
	deriving the values from the entry method's creation message */
    void setEpIdxHistory(){
      if(currentlyExecutingMsg){
	pathHistory.createPath(& currentlyExecutingMsg->pathHistory);
      } else {
	pathHistory.createPath(NULL);
      }
    }

    void printEpIdxHistory(){
      pathHistory.print();
    }
      
    /// Called when beginning to execute an entry method 
    void updateCounts(){
      pathHistory.incrementEpIdxCount(getEpIdx());
      if(attribs.mtype==ForArrayEltMsg){
	CkGroupID &a = type.array.arr;
	pathHistory.incrementArrayIdxCount(a.idx);
      }
    }
#endif

};

inline envelope *UsrToEnv(const void *const msg) {
  return (((envelope *) msg)-1);
}

inline void *EnvToUsr(const envelope *const env) {
  return ((void *)(env+1));
}

inline envelope *_allocEnv(const int msgtype, const int size=0, const int prio=0) {
  return envelope::alloc(msgtype,size,prio);
}

inline void *_allocMsg(const int msgtype, const int size, const int prio=0) {
  return EnvToUsr(envelope::alloc(msgtype,size,prio));
}

extern UChar   _defaultQueueing;

extern void CkPackMessage(envelope **pEnv);
extern void CkUnpackMessage(envelope **pEnv);

class MsgPool: public SafePool<void *> {
private:
    static void *_alloc(void) {
      /* CkAllocSysMsg() called in .def.h is not thread of sigio safe */
      register envelope *env = _allocEnv(ForChareMsg,0,0);
      env->setQueueing(_defaultQueueing);
      env->setMsgIdx(0);
      return EnvToUsr(env);
    }
public:
    MsgPool():SafePool<void*>(_alloc, CkFreeMsg) {}
#ifdef _FAULT_MLOG_
        void *get(void){
            return allocfn();
        }
        void put(void *m){
        }
#endif
};

CkpvExtern(MsgPool*, _msgPool);



#endif
