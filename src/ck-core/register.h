#ifndef _REGISTER_H
#define _REGISTER_H

class EntryInfo {
  public:
    char *name;
    CkCallFnPtr call;
    int msgIdx;
    int chareIdx;

    EntryInfo(char *n, CkCallFnPtr c, int m, int ci) : 
      name(n), call(c), msgIdx(m), chareIdx(ci)
    {}
};

class MsgInfo {
  public:
    char *name;
    CkPackFnPtr pack;
    CkUnpackFnPtr unpack;
    CkCoerceFnPtr coerce;
    int size;

    MsgInfo(char *n, CkPackFnPtr p, CkUnpackFnPtr u, CkCoerceFnPtr c, int s) : 
      name(n), size(s), pack(p), unpack(u), coerce(c) 
    {}
};

class ChareInfo {
  public:
    char *name;
    int size;
    ChareInfo(char *n, int s) : name(n), size(s) {}
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
