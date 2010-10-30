/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
/**
 @defgroup CkEnvelope
 \brief  Charm++ message header.
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



//#define USE_CRITICAL_PATH_HEADER_ARRAY

/**
    \addtogroup CriticalPathFramework 
    @{
*/

/** A class that is used to track the entry points and other information 
    about a critical path as a charm++ program executes.

    This class won't do useful things unless USE_CRITICAL_PATH_HEADER_ARRAY is defined
*/
class PathHistoryEnvelope {
 protected:
  // When passing paths forward, store information on PEs, in backward pass, lookup necessary information
  int sender_history_table_idx;
  double totalTime;
 public:
  double getTotalTime() const{ return totalTime; }
  int get_sender_history_table_idx() const{ return sender_history_table_idx; }
  void set_sender_history_table_idx(int i) { sender_history_table_idx = i; }
  PathHistoryEnvelope(){ reset(); }
  double getTime() const{ return totalTime; }
  void setTime(double t){ totalTime = t; }
  void pup(PUP::er &p) {
    p | sender_history_table_idx;
    p | totalTime;
  } 
  void reset();
  void print() const;
  /// Write a description of the path into the beginning of the provided buffer. The buffer ought to be large enough.
  void printHTMLToString(char* buf) const{
    buf[0] = '\0';
    sprintf(buf+strlen(buf), "Path Time=%lf<br> Sender idx=%d", (double)totalTime, (int)sender_history_table_idx);
  }
  /// The number of available EP counts 
  int getNumUsed() const;
  /// Return the count value for the idx'th available EP  
  int getUsedCount(int idx) const;
  /// Return the idx'th available EP 
  int getUsedEp(int idx) const;
  int getEpCount(int ep) const;
  void incrementTotalTime(double time);
  //  void createPath(envelope *originatingMsg);
  void setDebug100();
};
/** @} */



typedef unsigned int   UInt;
typedef unsigned short UShort;
typedef unsigned char  UChar;

#include "charm.h" // for CkGroupID, and CkEnvelopeType
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
#include "ckobjid.h" //for the ckobjId
#endif

/**
@addtogroup CkEnvelope
*/

CkpvExtern(int, envelopeEventID);

/**
@{
The class envelope defines a Charm++ message's header. The first
'CmiReservedHeaderSize' bytes of memory is exclusively reserved for Converse
header, which is defined in convere.h and platform specific config files.

After Charm++ envelope comes the payload, i.e. a variable-length amount of user
data. Following the user data, optionally, a variable number of bits of
priority data can be stored at the end. Function
envelope::alloc(msgtype,size,prio) is always used to allocate the whole
message. Note that this memory layout must be observed.

The following are a few terms that are used often:

<pre>
 Envelope pointer        \
 Converse message pointer -> [ [ Converse envelope ]       ]
                             [       Charm envelope        ] 
 User message pointer     -> [ User data/payload ... ]
 Priority pointer         -> [ Priority ints ... ]
</pre>

The "message pointers" passed to and from users bypass the envelope and point
*directly* to the user data--the routine "EnvToUsr" below adjusts an envelope
(or converse message) pointer into this user message pointer.  There is a
corresponding routine "UsrToEnv" which takes the user data pointer and returns
a pointer to the envelope/converse message.

Unfortunately, in the guts of Charm++ it's not always clear whether you've been
given a converse or user message pointer, as both tend to be passed as void *.
Confusing the two will invariably result in data corruption and bizarre
crashes.

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
   Keeping this in a union allows the different kinds of messages 
   to have different fields/types, in an alignment-safe way, 
   without wasting any storage.
 */
    union u_type {
      struct s_chare {  // NewChareMsg, NewVChareMsg, ForChareMsg, ForVidMsg, FillVidMsg
        void *ptr;      ///< object pointer
        UInt forAnyPe;  ///< Used only by newChare
      } chare;
      struct s_group {         // NodeBocInitMsg, BocInitMsg, ForNodeBocMsg, ForBocMsg
        CkGroupID g;           ///< GroupID
        CkNodeGroupID rednMgr; ///< Reduction manager for this group (constructor only!)
        CkGroupID dep;         ///< create after dep is created (constructor only!)
        int epoch;             ///< "epoch" this group was created during (0--mainchare, 1--later)
        UShort arrayEp;        ///< Used only for array broadcasts
      } group;
      struct s_array{             ///< For arrays only (ArrayEltInitMsg, ForArrayEltMsg)
        CkArrayIndexStruct index; ///< Array element index
        int listenerData[CK_ARRAYLISTENER_MAXLEN]; ///< For creation
        CkGroupID arr;            ///< Array manager GID
        UChar hopCount;           ///< number of times message has been routed
        UChar ifNotThere;         ///< what to do if array element is missing
      } array;
      struct s_roData {    ///< RODataMsg for readonly data type
        UInt count;
      } roData;
      struct s_roMsg {     ///< ROMsgMsg for readonlys defined in ci files
        UInt roIdx;
      } roMsg;
    };
    struct s_attribs {  // Packed bitwise struct
      UChar msgIdx;     ///< Usertype of message (determines pack routine)
      UChar mtype;      ///< e.g., ForBocMsg
      UChar queueing:4; ///< Queueing strategy (FIFO, LIFO, PFIFO, ...)
      UChar isPacked:1; ///< If true, message must be unpacked before use
      UChar isUsed:1;   ///< Marker bit to prevent message re-send.
    };
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CkObjID sender;
    CkObjID recver;
    MCount SN;
    MCount TN;
    MlogEntry *localMlogEntry;
	bool freeMsg;
#endif
private:
    u_type type;           ///< Depends on message type (attribs.mtype)
    UShort ref;            ///< Used by futures
    s_attribs attribs;
    UChar align[CkMsgAlignOffset(CmiReservedHeaderSize+sizeof(u_type)+sizeof(UShort)+sizeof(s_attribs))];    ///< padding to make sure sizeof(double) alignment
    
    //This struct should now be sizeof(void*) aligned.
    UShort priobits;   ///< Number of bits of priority data after user data
    UShort epIdx;      ///< Entry point to call
    UInt   pe;         ///< source processor
    UInt   event;      ///< used by projections
    UInt   totalsize;  ///< Byte count from envelope start to end of priobits
    
  public:
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
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
#ifndef CMK_OPTIMIZE
      memset(env, 0, sizeof(envelope));
#endif
      env->setMsgtype(type);
      env->totalsize = tsize;
      env->priobits = prio;
      env->setPacked(0);
      env->type.group.dep.setZero();
      _SET_USED(env, 0);
      //for record-replay
      env->setEvent(++CkpvAccess(envelopeEventID));
      env->setRef(0);

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
      env->pathHistory.reset();
#endif

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
      env->sender.type = TypeInvalid;
      env->recver.type = TypeInvalid;
      env->SN = 0;
      env->TN = 0;
      env->localMlogEntry = NULL;
#endif

      return env;
    }
    void reset() {
      setEvent(++CkpvAccess(envelopeEventID));
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
          || getMsgtype()==FillVidMsg ||  getMsgtype()==DeleteVidMsg);
      return type.chare.ptr;
    }
    void   setVidPtr(void *p) {
      CkAssert(getMsgtype()==NewVChareMsg || getMsgtype()==ForVidMsg
          || getMsgtype()==FillVidMsg ||  getMsgtype()==DeleteVidMsg);
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
    CkGroupID getGroupDep(){ return type.group.dep; }
    void setGroupDep(const CkGroupID &r){ type.group.dep = r; }

// Array-specific fields
    CkGroupID &getsetArrayMgr(void) {return type.array.arr;}
    int getArrayMgrIdx(void) const {return type.array.arr.idx;}
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
    /** The information regarding the entry methods that executed along the path to this one.
	\addtogroup CriticalPathFramework
    */
    PathHistoryEnvelope pathHistory;
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

inline void _resetEnv(envelope *env) {
  env->reset();
}

/** @} */

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
    static void _reset(void* m) {
      register envelope *env = UsrToEnv(m);
      _resetEnv(env);
    }
public:
    MsgPool():SafePool<void*>(_alloc, CkFreeMsg, _reset) {}
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        void *get(void){
            return allocfn();
        }
        void put(void *m){
        }
#endif
};

CkpvExtern(MsgPool*, _msgPool);

#endif
