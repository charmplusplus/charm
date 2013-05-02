/**
 @defgroup CkEnvelope
 \brief  Charm++ message header.
*/
#ifndef _ENVELOPE_H
#define _ENVELOPE_H

#include <pup.h>
#include <middle.h>
#include <ckarrayindex.h>
#include <cklists.h>
#include <objid.h>

#ifndef CkIntbits
#define CkIntbits (sizeof(int)*8)
#endif

#if CMK_ERROR_CHECKING
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

#if CMK_MESSAGE_LOGGING
#define CK_FREE_MSG_MLOG 	0x1
#define CK_BYPASS_DET_MLOG 	0x2
#define CK_MULTICAST_MSG_MLOG 	0x4
#define CK_REDUCTION_MSG_MLOG 	0x8
#endif

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
  PathHistoryEnvelope(){ 
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
    reset(); 
#endif
  }
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
header, which is defined in converse.h and platform specific config files.

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
 Extra data pointer       -> [ data specific to this message type ]
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

 /**
   These structures store the type-specific message information.
 */
struct s_chare {  // NewChareMsg, NewVChareMsg, ForChareMsg, ForVidMsg, FillVidMsg
        void *ptr;      ///< object pointer
        UInt forAnyPe;  ///< Used only by newChare
        int  bype;      ///< created by this pe
};

struct s_groupinit {         // NodeBocInitMsg, BocInitMsg
        CkGroupID g;           ///< GroupID
        CkNodeGroupID rednMgr; ///< Reduction manager for this group (constructor only!)
        CkGroupID dep;         ///< create after dep is created (constructor only!)
        int epoch;             ///< "epoch" this group was created during (0--mainchare, 1--later)
};

struct s_group {         // ForNodeBocMsg, ForBocMsg
        CkGroupID g;           ///< GroupID
        UShort arrayEp;        ///< Used only for array broadcasts
};

struct s_array{             ///< ForArrayEltMsg
        CkArrayIndexBase index; ///< Array element index
        CkGroupID arr;            ///< Array manager GID
#if CMK_SMP_TRACE_COMMTHREAD
        UInt srcpe; 
#endif
        UChar hopCount;           ///< number of times message has been routed
        UChar ifNotThere;         ///< what to do if array element is missing
};

struct s_objid {
        ck::ObjID id;
#if CMK_SMP_TRACE_COMMTHREAD
        UInt srcpe;
#endif
        UChar hopCount;           ///< number of times message has been routed
        UChar ifNotThere;         ///< what to do if array element is missing
};

struct s_arrayinit{         ///< ArrayEltInitMsg
        CkArrayIndexBase index; ///< Array element index
        CkGroupID arr;            ///< Array manager GID
#if CMK_SMP_TRACE_COMMTHREAD
        UInt srcpe; 
#endif
        UChar hopCount;           ///< number of times message has been routed
        UChar ifNotThere;         ///< what to do if array element is missing
        int listenerData[CK_ARRAYLISTENER_MAXLEN]; ///< For creation
};

struct s_roData {    ///< RODataMsg for readonly data type
        UInt count;
};

struct s_roMsg {     ///< ROMsgMsg for readonlys defined in ci files
        UInt roIdx;
};

inline UShort extraSize(CkEnvelopeType type)
{
  int ret = 0;
  switch (type) {
  case NewChareMsg:
  case NewVChareMsg:
  case ForChareMsg:
  case ForVidMsg:
  case FillVidMsg:
  case DeleteVidMsg:
    ret = sizeof(struct s_chare);
    break;
  case BocInitMsg:
  case NodeBocInitMsg:
    ret = sizeof(struct s_groupinit);
    break;
  case ForBocMsg:
  case ForNodeBocMsg:
    ret = sizeof(struct s_group);
    break;
  case ArrayEltInitMsg:
    ret = sizeof(struct s_arrayinit);
    break;
  case ForArrayEltMsg:
    ret = sizeof(struct s_array);
    break;
  case ForIDedObjMsg:
    ret = sizeof(struct s_objid);
    break;
  case RODataMsg:
    ret = sizeof(struct s_roData);
    break;
  case ROMsgMsg:
    ret = sizeof(struct s_roMsg);
    break;
  case StartExitMsg:
  case ExitMsg:
  case ReqStatMsg:
  case StatMsg:
    break;
  default:
    CmiAbort("piggysize: unknown message type.");
  }
  return ret;
}

extern UInt  envMaxExtraSize;

class envelope {
  private:
    /// Converse message envelope, Must be first field in this class
    char   core[CmiReservedHeaderSize];
public:
    struct s_attribs {  // Packed bitwise struct
      UChar msgIdx;     ///< Usertype of message (determines pack routine)
      UChar mtype;      ///< e.g., ForBocMsg
      UChar queueing:4; ///< Queueing strategy (FIFO, LIFO, PFIFO, ...)
      UChar isPacked:1; ///< If true, message must be unpacked before use
      UChar isUsed:1;   ///< Marker bit to prevent message re-send.
    };
private:
    //u_type type;           ///< Depends on message type (attribs.mtype)
    
    CMK_REFNUM_TYPE ref;            ///< Used by futures
    UShort   extrasize;  ///< Byte count specific for message types
    s_attribs attribs;
    UChar align[CkMsgAlignOffset(CmiReservedHeaderSize+sizeof(CMK_REFNUM_TYPE)+sizeof(UShort)+sizeof(s_attribs))];    ///< padding to make sure sizeof(double) alignment
    
    //This struct should now be sizeof(void*) aligned.
    UShort priobits;   ///< Number of bits of priority data after user data
    UShort epIdx;      ///< Entry point to call
    UInt   pe;         ///< source processor
    UInt   event;      ///< used by projections
    UInt   totalsize;  ///< Byte count from envelope start to end of priobits
    
  public:
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CkObjID sender;
    CkObjID recver;
    MCount SN;
#if defined(_FAULT_CAUSAL_)
    MCount TN;
    MCount mlogPadding;		//HACK: aligns envelope to double size (to make xlc work)
#endif
    int incarnation;
    int flags;
    UInt piggyBcastIdx;
#endif
    void pup(PUP::er &p);
    UInt   getEvent(void) const { return event; }
    void   setEvent(const UInt e) { event = e; }
    CMK_REFNUM_TYPE   getRef(void) const { return ref; }
    void   setRef(const CMK_REFNUM_TYPE r) { ref = r; }
    UChar  getQueueing(void) const { return attribs.queueing; }
    void   setQueueing(const UChar q) { attribs.queueing=q; }
    UChar  getMsgtype(void) const { return attribs.mtype; }
    void   setMsgtype(const UChar m) { if (attribs.mtype!=m) { int old = extrasize; extrasize = extraSize((CkEnvelopeType)m); totalsize += extrasize - old; } attribs.mtype = m; }
#if CMK_ERROR_CHECKING
    UChar  isUsed(void) { return attribs.isUsed; }
    void   setUsed(const UChar u) { attribs.isUsed=u; }
#else /* CMK_ERROR_CHECKING */
    inline void setUsed(const UChar u) {}
#endif
    UChar  getMsgIdx(void) const { return attribs.msgIdx; }
    void   setMsgIdx(const UChar idx) { attribs.msgIdx = idx; }
    UInt   getTotalsize(void) const { return totalsize; }
    void   setTotalsize(const UInt s) { totalsize = s; }
    UInt   getUsersize(void) const { 
      return totalsize - getPrioBytes() - sizeof(envelope) - extrasize; 
    }
    void   setUsersize(const UInt s) {
      CkAssert(s < getUsersize());
      UInt newPrioOffset = sizeof(envelope) + CkMsgAlignLength(s);
      UInt newExtraDataOffset = newPrioOffset + getPrioBytes();
      UInt newTotalsize = newExtraDataOffset + getExtrasize();
      void *newPrioPtr = (void *) ((char *) this + newPrioOffset); 
      void *newExtraPtr = (void *) ((char *) this + newExtraDataOffset);
      // use memmove instead of memcpy in case memory areas overlap
      memmove(newPrioPtr, getPrioPtr(), getPrioBytes()); 
      memmove(newExtraPtr, (void *) extraData(), getExtrasize());
      setTotalsize(newTotalsize); 
    }

    // s specifies number of bytes to remove from user portion of message
    void shrinkUsersize(const UInt s) {
      CkAssert(s < getUsersize());
      setUsersize(getUsersize() - s);
    }

    UShort getExtrasize(void) const { return extrasize; }
    void   setExtrasize(const UShort s) { extrasize = s; }
    UChar  isPacked(void) const { return attribs.isPacked; }
    void   setPacked(const UChar p) { attribs.isPacked = p; }
    UShort getPriobits(void) const { return priobits; }
    void   setPriobits(const UShort p) { priobits = p; }
    UShort getPrioWords(void) const { return CkPriobitsToInts(priobits); }
    UShort getPrioBytes(void) const { return getPrioWords()*sizeof(int); }
    void*  getPrioPtr(void) const { 
      return (void *)((char *)this + totalsize - extrasize - getPrioBytes());
    }
    static envelope *alloc(const UChar type, const UInt size=0, const UShort prio=0)
    {
      CkAssert(type >= NewChareMsg && type < LAST_CK_ENVELOPE_TYPE);

#if CMK_USE_STL_MSGQ
      // Ideally, this should be a static compile-time assert. However we need API changes for that
      CkAssert(sizeof(CMK_MSG_PRIO_TYPE) >= sizeof(int)*CkPriobitsToInts(prio));
#endif

      register UShort extrasize = extraSize((CkEnvelopeType)type);
      register UInt tsize0 = sizeof(envelope)+ 
            CkMsgAlignLength(size)+
	    sizeof(int)*CkPriobitsToInts(prio);
      register UInt tsize = tsize0 + extrasize;
      register envelope *env = (envelope *)CmiAlloc(tsize0);
#if CMK_REPLAYSYSTEM
      //for record-replay
      memset(env, 0, sizeof(envelope));
      env->setEvent(++CkpvAccess(envelopeEventID));
#endif
      env->setMsgtype(type);
      env->totalsize = tsize;
      env->extrasize = extrasize;
      env->priobits = prio;
      env->setPacked(0);
      //env->type.group.dep.setZero();
      ((struct s_groupinit *)env->extraData())->dep.setZero();
      _SET_USED(env, 0);
      env->setRef(0);
      env->setEpIdx(0);

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
      env->pathHistory.reset();
#endif

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
      env->sender.type = TypeInvalid;
      env->recver.type = TypeInvalid;
      env->SN = 0;
#if defined(_FAULT_CAUSAL_)
      env->TN = 0;
#endif
	  env->incarnation = -1;
#endif

      return env;
    }
    void reset() {
#if CMK_REPLAYSYSTEM
      setEvent(++CkpvAccess(envelopeEventID));
#endif
      //type.group.dep.setZero();
      ((struct s_groupinit *)extraData())->dep.setZero();
    }
    UShort getEpIdx(void) const { return epIdx; }
    void   setEpIdx(const UShort idx) { epIdx = idx; }
    UInt   getSrcPe(void) const { return pe; }
    void   setSrcPe(const UInt s) { pe = s; }
    static void setSrcPe(char *env, const UInt s) { ((envelope*)env)->setSrcPe(s); }

// Readonly-specific fields
    inline char * extraData() const { return (char*)this+totalsize-extrasize; }

    UInt   getCount(void) const { 
      CkAssert(getMsgtype()==RODataMsg); return ((struct s_roData *)extraData())->count; 
    }
    void   setCount(const UInt c) { 
      CkAssert(getMsgtype()==RODataMsg); ((struct s_roData *)extraData())->count = c; 
    }
    UInt   getRoIdx(void) const { 
      CkAssert(getMsgtype()==ROMsgMsg); return ((struct s_roMsg*)extraData())->roIdx; 
    }
    void   setRoIdx(const UInt r) { 
      CkAssert(getMsgtype()==ROMsgMsg); ((struct s_roMsg*)extraData())->roIdx = r; 
    }
    
 // Chare-specific fields
    UInt isForAnyPE(void) { 
      CkAssert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg); 
      return ((struct s_chare*)extraData())->forAnyPe; 
    }
    void setForAnyPE(UInt f) { 
      CkAssert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg); 
      ((struct s_chare*)extraData())->forAnyPe = f; 
    }
    void*  getVidPtr(void) const {
      CkAssert(getMsgtype()==NewVChareMsg || getMsgtype()==ForVidMsg
          || getMsgtype()==FillVidMsg ||  getMsgtype()==DeleteVidMsg);
      return ((struct s_chare*)extraData())->ptr;
    }
    void   setVidPtr(void *p) {
      CkAssert(getMsgtype()==NewVChareMsg || getMsgtype()==ForVidMsg
          || getMsgtype()==FillVidMsg ||  getMsgtype()==DeleteVidMsg);
      ((struct s_chare*)extraData())->ptr = p;
    }
    void*  getObjPtr(void) const { 
      CkAssert(getMsgtype()==ForChareMsg); return ((struct s_chare*)extraData())->ptr; 
    }
    void   setObjPtr(void *p) { 
      CkAssert(getMsgtype()==ForChareMsg); ((struct s_chare*)extraData())->ptr = p; 
    }
    UInt getByPe(void) { 
      CkAssert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg); 
      return ((struct s_chare*)extraData())->bype; 
    }
    void setByPe(UInt pe) { 
      CkAssert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg); 
      ((struct s_chare*)extraData())->bype = pe; 
    }

// Group-specific fields
    CkGroupID   getGroupNum(void) const {
      CkAssert(getMsgtype()==ForBocMsg || getMsgtype()==ForNodeBocMsg);
      return ((struct s_group*)extraData())->g;
    }
    void   setGroupNum(const CkGroupID g) {
      CkAssert(getMsgtype()==ForBocMsg || getMsgtype()==ForNodeBocMsg);
      ((struct s_group*)extraData())->g = g;
    }

    CkGroupID getInitGroupNum(void) const {
      CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg);
      return ((struct s_groupinit*)extraData())->g;
    }
    void   setInitGroupNum(const CkGroupID g) {
      CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg);
      ((struct s_groupinit*)extraData())->g = g;
    }
    void setGroupEpoch(int epoch) { CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg); ((struct s_groupinit*)extraData())->epoch=epoch; }
    int getGroupEpoch(void) { CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg); return ((struct s_groupinit*)extraData())->epoch; }
    void setRednMgr(CkNodeGroupID r){CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg);  ((struct s_groupinit*)extraData())->rednMgr = r; }
    CkNodeGroupID getRednMgr(){ CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg); return ((struct s_groupinit*)extraData())->rednMgr; }
    CkGroupID getGroupDep(){ CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg); return ((struct s_groupinit*)extraData())->dep; }
    void setGroupDep(const CkGroupID &r){ CkAssert(getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg ); ((struct s_groupinit*)extraData())->dep = r; }

// Array-specific fields
    CkGroupID getArrayMgr(void) const {
        if (getMsgtype() == ForArrayEltMsg || getMsgtype() == ArrayEltInitMsg)
            return ((struct s_array*)extraData())->arr;
        else if (getMsgtype() == ForIDedObjMsg)
            return ((struct s_objid*)extraData())->id.getCollectionID();
        else
            CkAbort("Cannot return ArrayID from msg for non-array entity");
    }

    void setArrayMgr(const CkGroupID gid) { CkAssert(getMsgtype() == ForArrayEltMsg || getMsgtype() == ArrayEltInitMsg); ((struct s_array*)extraData())->arr = gid; }
    int getArrayMgrIdx(void) const { CkAssert(getMsgtype() == ForArrayEltMsg || getMsgtype() == ArrayEltInitMsg); return ((struct s_array*)extraData())->arr.idx;}
    UShort &getsetArrayEp(void) {return epIdx;}
    UShort &getsetArrayBcastEp(void) {return ((struct s_group*)extraData())->arrayEp;}
    UChar &getsetArrayHops(void) { CkAssert(getMsgtype() == ForArrayEltMsg || getMsgtype() == ArrayEltInitMsg); return ((struct s_array*)extraData())->hopCount;}
    int getArrayIfNotThere(void) { CkAssert(getMsgtype() == ForArrayEltMsg || getMsgtype() == ArrayEltInitMsg); return ((struct s_array*)extraData())->ifNotThere;}
    void setArrayIfNotThere(int nt) { CkAssert(getMsgtype() == ForArrayEltMsg || getMsgtype() == ArrayEltInitMsg); ((struct s_array*)extraData())->ifNotThere=nt;}
    int *getsetArrayListenerData(void) { CkAssert(getMsgtype() == ArrayEltInitMsg); return ((struct s_arrayinit*)extraData())->listenerData;}
#if CMK_SMP_TRACE_COMMTHREAD
    UInt &getsetArraySrcPe(void) {return ((struct s_array*)extraData())->srcpe;}
#else
    UInt &getsetArraySrcPe(void) {return pe;}
#endif
    CkArrayIndex &getsetArrayIndex(void) 
    {
      CkAssert(getMsgtype() == ForArrayEltMsg || getMsgtype() == ArrayEltInitMsg);
      return *(CkArrayIndex *)&((struct s_array*)extraData())->index;
    }

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

inline void setEventID(envelope *env){
  env->setEvent(++CkpvAccess(envelopeEventID));
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
