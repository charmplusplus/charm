/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _ENVELOPE_H
#define _ENVELOPE_H

#define CINTBITS (sizeof(int)*8)

#ifndef CMK_OPTIMIZE
#define _SET_USED(env, x) (env)->setUsed((x))
#define _CHECK_USED(env) do { if(env->isUsed()) \
                           CmiAbort("Message being re-sent. Aborting...\n"); \
                         } while(0)
#else
#define _SET_USED(env, x) do{}while(0)
#define _CHECK_USED(env) do{}while(0)
#endif

typedef unsigned int   UInt;
typedef unsigned short UShort;
typedef unsigned char  UChar;

#define SVM1     (sizeof(double)-1)
#define ALIGN(x) (((x)+SVM1)&(~(SVM1)))
#define A(x)     ALIGN(x)
#define D(x)     (A(x)-(x))
#define PW(x)    ((x+CINTBITS-1)/CINTBITS)

#define NewChareMsg    1
#define NewVChareMsg   2
#define BocInitMsg     3
#define ForChareMsg    4
#define ForBocMsg      5
#define ForVidMsg      6
#define FillVidMsg     7
#define DBocReqMsg     8
#define DBocNumMsg     9
#define RODataMsg      10
#define ROMsgMsg       11
#define ExitMsg        12
#define ReqStatMsg     13
#define StatMsg        14
#define NodeBocInitMsg 15
#define ForNodeBocMsg  16
#define DNodeBocReqMsg 17
#define DNodeBocNumMsg 18

class envelope {
  private:
    char   core[CmiExtHeaderSizeBytes];
 //This union allows the different kinds of messages to have different
 // fields/types in an alignment-safe way without wasting any storage.
public:
    union u_type {
      struct s_chare { //NewChareMsg, NewVChareMsg, ForChareMsg, ForVidMsg, FillVidMsg
      	void *ptr;
      	UInt forAnyPe; //Used by new-only
      } chare;
      struct s_group{ //BocInitMsg, DBocMsg, DNodeBocMsg, ForBocMsg, ForNodeBocMsg
      	union u_gtype{
          struct s_dgroup{
            void *usrMsg; //For DBoc only
      	  } dgroup;
          struct s_array{ //For arrays only
      	    CkArrayIndexStruct index;//Array element index
      	    UInt srcPe;//Original sender
      	    UShort epIdx;//Array element entry point
      	    UChar hopCount;//number of times message has been routed
      	  } array;
        } gtype;//Group subtype
        UChar num; //Group number
      } group;
      struct s_roData { //RODataMsg
      	UInt count;
      } roData;
      struct s_roMsg { //ROMsgMsg
      	UInt roIdx;
      } roMsg;
    };
    struct s_attribs { //Packed bitwise struct
    	UChar msgIdx; //Usertype of message (determines pack routine)
	UChar mtype;
    	UChar queueing:4; //Queueing strategy (FIFO, LIFO, PFIFO, ...)
    	UChar isPacked:1;
    	UChar isUsed:1;
    	UChar ifNotThere:2; //Used by arrays
    };
private:
    u_type type; //Depends on message type (attribs.mtype)
    UShort ref; //Used by futures
    s_attribs attribs;
    UChar align[D(sizeof(u_type)+sizeof(UShort)+sizeof(s_attribs))];
    
    //This struct should now be sizeof(void*) aligned.
    UShort priobits;
    UShort epIdx;  //Entry point to call
    UInt   pe;    // source processor
    UInt   event; // used by projections
    UInt   totalsize; //Byte count from envelope start to end of priobits
    
  public:
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
#endif
    UChar  getMsgIdx(void) const { return attribs.msgIdx; }
    void   setMsgIdx(const UChar idx) { attribs.msgIdx = idx; }
    UInt   getTotalsize(void) const { return totalsize; }
    void   setTotalsize(const UInt s) { totalsize = s; }
    UChar  getIfNotThere(void) const { return attribs.ifNotThere; }
    void   setIfNotThere(const UChar s) { attribs.ifNotThere = s; }
    UChar  isPacked(void) const { return attribs.isPacked; }
    void   setPacked(const UChar p) { attribs.isPacked = p; }
    UShort getPriobits(void) const { return priobits; }
    void   setPriobits(const UShort p) { priobits = p; }
    UShort getPrioWords(void) const { return (priobits+CINTBITS-1)/CINTBITS; }
    UShort getPrioBytes(void) const { return getPrioWords()*sizeof(int); }
    void*  getPrioPtr(void) const { 
      return (void *)((char *)this + totalsize - getPrioBytes());
    }
    UInt   getCount(void) const { assert(getMsgtype()==RODataMsg); return type.roData.count; }
    void   setCount(const UInt c) { assert(getMsgtype()==RODataMsg); type.roData.count = c; }
    UInt   getRoIdx(void) const { assert(getMsgtype()==ROMsgMsg); return type.roMsg.roIdx; }
    void   setRoIdx(const UInt r) { assert(getMsgtype()==ROMsgMsg); type.roMsg.roIdx = r; }
    static envelope *alloc(const UChar type, const UInt size=0, const UShort prio=0)
    {
      assert(type>=NewChareMsg && type<=DNodeBocNumMsg);
      register UInt tsize = sizeof(envelope)+ALIGN(size)+sizeof(int)*PW(prio);
      register envelope *env = (envelope *)CmiAlloc(tsize);
      env->setMsgtype(type);
      env->totalsize = tsize;
      env->priobits = prio;
      env->setPacked(0);
      _SET_USED(env, 0);
      return env;
    }
    UShort getEpIdx(void) const {
      assert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg
          || getMsgtype()==ForChareMsg || getMsgtype()==ForVidMsg
          || getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg
          || getMsgtype()==ForBocMsg || getMsgtype()==ForNodeBocMsg);
      return epIdx;
    }
    void   setEpIdx(const UShort idx) {
      assert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg
          || getMsgtype()==ForChareMsg || getMsgtype()==ForVidMsg
          || getMsgtype()==BocInitMsg || getMsgtype()==NodeBocInitMsg
          || getMsgtype()==ForBocMsg || getMsgtype()==ForNodeBocMsg);
      epIdx = idx;
    }
    UInt isForAnyPE(void) { 
      assert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg); 
      return type.chare.forAnyPe; 
    }
    void setForAnyPE(UInt f) { 
      assert(getMsgtype()==NewChareMsg || getMsgtype()==NewVChareMsg); 
      type.chare.forAnyPe = f; 
    }
    void*  getVidPtr(void) const {
      assert(getMsgtype()==NewVChareMsg || getMsgtype()==ForVidMsg
          || getMsgtype()==FillVidMsg);
      return type.chare.ptr;
    }
    void   setVidPtr(void *p) {
      assert(getMsgtype()==NewVChareMsg || getMsgtype()==ForVidMsg
          || getMsgtype()==FillVidMsg);
      type.chare.ptr = p;
    }
    UInt   getSrcPe(void) const { return pe; }
    void   setSrcPe(const UInt s) { pe = s; }
    void*  getObjPtr(void) const { assert(getMsgtype()==ForChareMsg); return type.chare.ptr; }
    void   setObjPtr(void *p) { assert(getMsgtype()==ForChareMsg); type.chare.ptr = p; }
    UShort getRetEp(void) const {
      assert(getMsgtype()==DBocReqMsg || getMsgtype()==DNodeBocReqMsg
          || getMsgtype()==DBocNumMsg || getMsgtype()==DNodeBocNumMsg); 
      return epIdx; 
    }
    void   setRetEp(const UShort e) {
      assert(getMsgtype()==DBocReqMsg || getMsgtype()==DNodeBocReqMsg
          || getMsgtype()==DBocNumMsg || getMsgtype()==DNodeBocNumMsg); 
      epIdx = e; 
    }
    void*  getUsrMsg(void) const { 
      assert(getMsgtype()==DBocReqMsg || getMsgtype()==DBocNumMsg
          || getMsgtype()==DNodeBocReqMsg || getMsgtype()==DNodeBocNumMsg); 
      return type.group.gtype.dgroup.usrMsg; 
    }
    void   setUsrMsg(void *p) { 
      assert(getMsgtype()==DBocReqMsg || getMsgtype()==DBocNumMsg
          || getMsgtype()==DNodeBocReqMsg || getMsgtype()==DNodeBocNumMsg); 
      type.group.gtype.dgroup.usrMsg = p; 
    }
    UInt   getGroupNum(void) const {
      assert(getMsgtype()==BocInitMsg || getMsgtype()==ForBocMsg
          || getMsgtype()==DBocNumMsg || getMsgtype()==NodeBocInitMsg
          || getMsgtype()==ForNodeBocMsg || getMsgtype()==DNodeBocNumMsg);
      return type.group.num;
    }
    void   setGroupNum(const UInt g) {
      assert(getMsgtype()==BocInitMsg || getMsgtype()==ForBocMsg
          || getMsgtype()==DBocNumMsg || getMsgtype()==NodeBocInitMsg
          || getMsgtype()==ForNodeBocMsg || getMsgtype()==DNodeBocNumMsg);
      type.group.num = g;
    }
    CkArrayIndexMax &array_index(void) {return 
        *(CkArrayIndexMax *)&type.group.gtype.array.index;}
    unsigned short &array_ep(void) {return type.group.gtype.array.epIdx;}
    unsigned char &array_hops(void) {return type.group.gtype.array.hopCount;}
    unsigned int &array_srcPe(void) {return type.group.gtype.array.srcPe;}
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

#define MAXMSGS 32

class MsgPool {
  private:
    int num;
    void *msgs[MAXMSGS];
    static void *_alloc(void) {
      register envelope *env = _allocEnv(ForChareMsg,0,0);
      env->setQueueing(_defaultQueueing);
      env->setMsgIdx(0);
      return EnvToUsr(env);
    }
  public:
    MsgPool();
    void *get(void) {
      return (num ? msgs[--num] : _alloc());
    }
    void put(void *m) {
      if (num==MAXMSGS)
        CkFreeMsg(m);
      else
        msgs[num++] = m;
    }
};

CpvExtern(MsgPool*, _msgPool);
extern void _processBocInitMsg(envelope *);
extern void _processNodeBocInitMsg(envelope *);

#endif
