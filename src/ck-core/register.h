/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _REGISTER_H
#define _REGISTER_H

class EntryInfo {
  public:
    const char *name;
    CkCallFnPtr call;
    int msgIdx;
    int chareIdx;
    bool traceEnabled; /* Charm++ Tracing enabled for this ep (dynamic) */
    bool noKeep; /* Method doesn't keep message passed in to it (static) */
    CkMarshallUnpackFn marshallUnpack;

    EntryInfo(const char *n, CkCallFnPtr c, int m, int ci) : 
      name(n), call(c), msgIdx(m), chareIdx(ci), marshallUnpack(0)
    { traceEnabled=true; noKeep=false; }
};

class MsgInfo {
  public:
    const char *name;
    CkPackFnPtr pack;
    CkUnpackFnPtr unpack;
    CkCoerceFnPtr coerce;
    size_t size;

    MsgInfo(const char *n,CkPackFnPtr p,CkUnpackFnPtr u,CkCoerceFnPtr c,int s):
      name(n), pack(p), unpack(u), coerce(c), size(s)
    {}
};


class ChareInfo {
  public:
    const char *name;
    int size;
    int classIdx;
    int defCtor,migCtor; //Default (no argument) and migration constructor indices
    int numbases;
    int bases[16];
    ChareInfo(const char *n, int s) : name(n), size(s), classIdx(1) {
      defCtor=migCtor=-1;
      numbases = 0;
    }
    void setClassIdx(int idx) { classIdx = idx; }
    int getClassIdx(void) { return classIdx; }
    void setDefaultCtor(int idx) { defCtor = idx; }
    int getDefaultCtor(void) { return defCtor; }
    void setMigCtor(int idx) { migCtor = idx; }
    int getMigCtor(void) { return migCtor; }
    void addBase(int idx) { bases[numbases++] = idx; }
};

class MainInfo {
  public:
    int chareIdx;
    int entryIdx;
    MainInfo(int c, int e) : chareIdx(c), entryIdx(e) {}
};

class ReadonlyInfo {
  public:
    const char *name,*type;
    int size;
    void *ptr;
    CkPupReadonlyFnPtr pup;
    void pupData(PUP::er &p) {
      if (pup!=NULL)
        (pup)((void *)&p);
      else
        p(ptr,size);
    }
    ReadonlyInfo(const char *n,const char *t,
	 int s, void *p,CkPupReadonlyFnPtr pf) 
	: name(n), type(t), size(s), ptr(p), pup(pf) {}
};

class ReadonlyMsgInfo {
  public:
    const char *name, *type;
    void **pMsg;
    ReadonlyMsgInfo(const char *n, const char *t,
	void **p) : name(n), type(t), pMsg(p) {}
};

extern EntryInfo**        _entryTable;
extern MsgInfo**          _msgTable;
extern ChareInfo**        _chareTable;
extern MainInfo**         _mainTable;
extern ReadonlyInfo**     _readonlyTable;
extern ReadonlyMsgInfo**  _readonlyMsgs;

extern int _numEntries;
extern int _numMsgs;
extern int _numChares;
extern int _numMains;
extern int _numReadonlies;
extern int _numReadonlyMsgs;

#define _ENTRY_TABLE_SIZE     1024
#define _MSG_TABLE_SIZE       256
#define _CHARE_TABLE_SIZE     1024
#define _MAIN_TABLE_SIZE      128
#define _READONLY_TABLE_SIZE  128

extern void _registerInit(void);
#endif
