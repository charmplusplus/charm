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

    EntryInfo(const char *n, CkCallFnPtr c, int m, int ci) : 
      name(n), call(c), msgIdx(m), chareIdx(ci)
    {}
};

class MsgInfo {
  public:
    const char *name;
    CkPackFnPtr pack;
    CkUnpackFnPtr unpack;
    CkCoerceFnPtr coerce;
    int size;

    MsgInfo(const char *n,CkPackFnPtr p,CkUnpackFnPtr u,CkCoerceFnPtr c,int s):
      name(n), pack(p), unpack(u), coerce(c), size(s)
    {}
};


class ChareInfo {
  public:
    const char *name;
    int size;
    int classIdx;
    int migCtor;
    int numbases;
    int bases[16];
    ChareInfo(const char *n, int s) : name(n), size(s), classIdx(1) {
      migCtor=-1;
      numbases = 0;
    }
    void setClassIdx(int idx) { classIdx = idx; }
    int getClassIdx(void) { return classIdx; }
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
    int size;
    void *ptr;
    ReadonlyInfo(int s, void *p) : size(s), ptr(p) {}
};

class ReadonlyMsgInfo {
  public:
    void **pMsg;
    ReadonlyMsgInfo(void **p) : pMsg(p) {}
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
