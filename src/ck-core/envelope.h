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
    UInt   getEvent(void) { return event; }
    void   setEvent(UInt e) { event = e; }
    UInt   getRef(void) { return s2; }
    void   setRef(UShort r) { s2 = r; }
    UChar  getQueueing(void) { return queueing; }
    void   setQueueing(UChar q) { queueing = q; }
    UChar  getMsgtype(void) { return msgtype; }
    void   setMsgtype(UChar m) { msgtype = m; }
    UChar  getMsgIdx(void) { return msgIdx; }
    void   setMsgIdx(UChar idx) { msgIdx = idx; }
    UInt   getTotalsize(void) { return totalsize; }
    void   setTotalsize(UInt s) { totalsize = s; }
    UChar  isPacked(void) { return packed; }
    void   setPacked(UChar p) { packed = p; }
    UShort getPriobits(void) { return priobits; }
    void   setPriobits(UShort p) { priobits = p; }
    UShort getPrioWords(void) { return (priobits+CINTBITS-1)/CINTBITS; }
    UShort getPrioBytes(void) { return getPrioWords()*sizeof(int); }
    void*  getPrioPtr(void) { 
      return (void *)((char *)this + totalsize - getPrioBytes());
    }
    UInt   getCount(void) { assert(msgtype==RODataMsg); return i1; }
    void   setCount(UInt c) { assert(msgtype==RODataMsg); i1 = c; }
    UInt   getRoIdx(void) { assert(msgtype==ROMsgMsg); return i1; }
    void   setRoIdx(UInt r) { assert(msgtype==ROMsgMsg); i1 = r; }
    static envelope *alloc(UChar type, UInt size=0, UShort prio=0)
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
    UShort getEpIdx(void) {
      assert(msgtype==NewChareMsg || msgtype==NewVChareMsg
          || msgtype==ForChareMsg || msgtype==ForVidMsg
          || msgtype==BocInitMsg || msgtype==ForBocMsg);
      return s1;
    }
    void   setEpIdx(UShort idx) {
      assert(msgtype==NewChareMsg || msgtype==NewVChareMsg
          || msgtype==ForChareMsg || msgtype==ForVidMsg
          || msgtype==BocInitMsg || msgtype==ForBocMsg);
      s1 = idx;
    }
    void*  getVidPtr(void) {
      assert(msgtype==NewVChareMsg || msgtype==ForVidMsg
          || msgtype==FillVidMsg);
      return ptr;
    }
    void   setVidPtr(void *p) {
      assert(msgtype==NewVChareMsg || msgtype==ForVidMsg
          || msgtype==FillVidMsg);
      ptr = p;
    }
    UInt   getSrcPe(void) { return pe; }
    void   setSrcPe(UInt s) { pe = s; }
    void*  getObjPtr(void) { assert(msgtype==ForChareMsg); return ptr; }
    void   setObjPtr(void *p) { assert(msgtype==ForChareMsg); ptr = p; }
    UShort getRetEp(void) { assert(msgtype==DBocReqMsg); return s1; }
    void   setRetEp(UShort e) { assert(msgtype==DBocReqMsg); s1 = e; }
    void*  getUsrMsg(void) { 
      assert(msgtype==DBocReqMsg || msgtype==DBocNumMsg); 
      return ptr; 
    }
    void   setUsrMsg(void *p) { 
      assert(msgtype==DBocReqMsg || msgtype==DBocNumMsg); 
      ptr = p; 
    }
    UInt   getGroupNum(void) {
      assert(msgtype==BocInitMsg || msgtype==ForBocMsg
          || msgtype==DBocNumMsg);
      return i1;
    }
    void   setGroupNum(UInt g) {
      assert(msgtype==BocInitMsg || msgtype==ForBocMsg
          || msgtype==DBocNumMsg);
      i1 = g;
    }
};

static inline envelope *UsrToEnv(void *msg) {
  return (((envelope *) msg)-1);
}

static inline void *EnvToUsr(envelope *env) {
  return ((void *)(env+1));
}

static inline envelope *_allocEnv(int msgtype, int size=0, int prio=0) {
  return envelope::alloc(msgtype,size,prio);
}

static inline void *_allocMsg(int msgtype, int size, int prio=0) {
  return EnvToUsr(envelope::alloc(msgtype,size,prio));
}

extern UChar   _defaultQueueing;

#define MAXMSGS 32

class MsgPool {
  private:
    int num;
    void *msgs[MAXMSGS];
    void *_alloc(void) {
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
