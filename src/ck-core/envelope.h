#ifndef _ENVELOPE_H
#define _ENVELOPE_H

#define CINTBITS (sizeof(int)*8)

typedef unsigned int   UInt;
typedef unsigned short UShort;
typedef unsigned char  UChar;

#define SVM1     (sizeof(void*)-1)
#define ALIGN(x) (((x)+SVM1)&(~(SVM1)))
#define A(x)     ALIGN(x)
#define D(x)     (A(x)-(x))
#define PW(x)    ((x+CINTBITS-1)/CINTBITS)

#define NewChareMsg  1
#define NewVChareMsg 2
#define BocInitMsg   3
#define ForChareMsg  4
#define ForBocMsg    5
#define ForVidMsg    6
#define FillVidMsg   7
#define DBocReqMsg   8
#define DBocNumMsg   9
#define RODataMsg    10
#define ROMsgMsg     11
#define ExitMsg      12
#define ReqStatMsg   13
#define StatMsg      14

// NewChareMsg : s1=epIdx
// NewVChareMsg: ptr=vidPtr, s1=epIdx
// ForChareMsg : ptr=objPtr, s1=epIdx
// ForVidMsg   : ptr=vidPtr, s1=epIdx
// FillVidMsg  : ptr=vidPtr
// BocInitMsg  : i1=groupnum, s1=epIdx
// DBocReqMsg  : ptr=usrmsg, s1=retEp
// DBocNumMsg  : ptr=usrmsg, i1=groupnum, s1=retEp
// RODataMsg   : i1=count
// ROMsgMsg    : i1=roIdx
// ForBocMsg   : i1=groupnum, s1=epidx

class envelope {
  private:
    char   core[CmiExtHeaderSizeBytes];
    void*  ptr;
    UInt   event; // used by projections
    UInt   pe;    // source processor
    UInt   totalsize;
    UInt   i1;
    UShort s1;
    UShort s2;
    UShort priobits;
    UChar  msgtype;
    UChar  msgIdx;
    UChar  queueing;
    UChar  packed;
    // to make envelope void* aligned
    UChar padding[D(3*sizeof(UShort)+4*sizeof(UChar))];
  public:
    UInt   getEvent(void) const { return event; }
    void   setEvent(const UInt e) { event = e; }
    UInt   getRef(void) const { return s2; }
    void   setRef(const UShort r) { s2 = r; }
    UChar  getQueueing(void) const { return queueing; }
    void   setQueueing(const UChar q) { queueing = q; }
    UChar  getMsgtype(void) const { return msgtype; }
    void   setMsgtype(const UChar m) { msgtype = m; }
    UChar  getMsgIdx(void) const { return msgIdx; }
    void   setMsgIdx(const UChar idx) { msgIdx = idx; }
    UInt   getTotalsize(void) const { return totalsize; }
    void   setTotalsize(const UInt s) { totalsize = s; }
    UChar  isPacked(void) const { return packed; }
    void   setPacked(const UChar p) { packed = p; }
    UShort getPriobits(void) const { return priobits; }
    void   setPriobits(const UShort p) { priobits = p; }
    UShort getPrioWords(void) const { return (priobits+CINTBITS-1)/CINTBITS; }
    UShort getPrioBytes(void) const { return getPrioWords()*sizeof(int); }
    void*  getPrioPtr(void) const { 
      return (void *)((char *)this + totalsize - getPrioBytes());
    }
    UInt   getCount(void) const { assert(msgtype==RODataMsg); return i1; }
    void   setCount(const UInt c) { assert(msgtype==RODataMsg); i1 = c; }
    UInt   getRoIdx(void) const { assert(msgtype==ROMsgMsg); return i1; }
    void   setRoIdx(const UInt r) { assert(msgtype==ROMsgMsg); i1 = r; }
    static envelope *alloc(const UChar type, const UInt size=0, const UShort prio=0)
    {
      assert(type>=NewChareMsg && type<=StatMsg);
      register UInt tsize = sizeof(envelope)+ALIGN(size)+sizeof(int)*PW(prio);
      register envelope *env = (envelope *)CmiAlloc(tsize);
      env->msgtype = type;
      env->totalsize = tsize;
      env->priobits = prio;
      env->packed = 0;
      return env;
    }
    UShort getEpIdx(void) const {
      assert(msgtype==NewChareMsg || msgtype==NewVChareMsg
          || msgtype==ForChareMsg || msgtype==ForVidMsg
          || msgtype==BocInitMsg || msgtype==ForBocMsg);
      return s1;
    }
    void   setEpIdx(const UShort idx) {
      assert(msgtype==NewChareMsg || msgtype==NewVChareMsg
          || msgtype==ForChareMsg || msgtype==ForVidMsg
          || msgtype==BocInitMsg || msgtype==ForBocMsg);
      s1 = idx;
    }
    void*  getVidPtr(void) const {
      assert(msgtype==NewVChareMsg || msgtype==ForVidMsg
          || msgtype==FillVidMsg);
      return ptr;
    }
    void   setVidPtr(void *p) {
      assert(msgtype==NewVChareMsg || msgtype==ForVidMsg
          || msgtype==FillVidMsg);
      ptr = p;
    }
    UInt   getSrcPe(void) const { return pe; }
    void   setSrcPe(const UInt s) { pe = s; }
    void*  getObjPtr(void) const { assert(msgtype==ForChareMsg); return ptr; }
    void   setObjPtr(void *p) { assert(msgtype==ForChareMsg); ptr = p; }
    UShort getRetEp(void) const { assert(msgtype==DBocReqMsg); return s1; }
    void   setRetEp(const UShort e) { assert(msgtype==DBocReqMsg); s1 = e; }
    void*  getUsrMsg(void) const { 
      assert(msgtype==DBocReqMsg || msgtype==DBocNumMsg); 
      return ptr; 
    }
    void   setUsrMsg(void *p) { 
      assert(msgtype==DBocReqMsg || msgtype==DBocNumMsg); 
      ptr = p; 
    }
    UInt   getGroupNum(void) const {
      assert(msgtype==BocInitMsg || msgtype==ForBocMsg
          || msgtype==DBocNumMsg);
      return i1;
    }
    void   setGroupNum(const UInt g) {
      assert(msgtype==BocInitMsg || msgtype==ForBocMsg
          || msgtype==DBocNumMsg);
      i1 = g;
    }
};

static inline envelope *UsrToEnv(const void *const msg) {
  return (((envelope *) msg)-1);
}

static inline void *EnvToUsr(const envelope *const env) {
  return ((void *)(env+1));
}

static inline envelope *_allocEnv(const int msgtype, const int size=0, const int prio=0) {
  return envelope::alloc(msgtype,size,prio);
}

static inline void *_allocMsg(const int msgtype, const int size, const int prio=0) {
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
    MsgPool() { 
      for(int i=0;i<MAXMSGS;i++)
        msgs[i] = _alloc();
      num = MAXMSGS;
    }
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

#endif
