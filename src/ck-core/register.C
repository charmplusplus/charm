#include "ck.h"

EntryInfo**        _entryTable;
MsgInfo**          _msgTable;
ChareInfo**        _chareTable;
MainInfo**         _mainTable;
ReadonlyInfo**     _readonlyTable;
ReadonlyMsgInfo**  _readonlyMsgs;

int _numEntries;
int _numMsgs;
int _numChares;
int _numMains;
int _numReadonlies;
int _numReadonlyMsgs;

void _registerInit(void)
{
  _numEntries =0;
  _numMsgs = 0;
  _numChares = 0;
  _numMains = 0;
  _numReadonlies = 0;
  _numReadonlyMsgs = 0;
  _entryTable = new EntryInfo*[_ENTRY_TABLE_SIZE];
  _msgTable = new MsgInfo*[_MSG_TABLE_SIZE];
  _chareTable = new ChareInfo*[_CHARE_TABLE_SIZE];
  _mainTable = new MainInfo*[_MAIN_TABLE_SIZE];
  _readonlyTable = new ReadonlyInfo*[_READONLY_TABLE_SIZE];
  _readonlyMsgs = new ReadonlyMsgInfo*[_READONLY_TABLE_SIZE];;
}

extern "C"
int CkRegisterMsg(const char *name, CkPackFnPtr pack, CkUnpackFnPtr unpack, 
                  CkCoerceFnPtr coerce, int size)
{
  _msgTable[_numMsgs] = new MsgInfo(name, pack, unpack, coerce, size);
  return _numMsgs++;
}

extern "C"
int CkRegisterEp(const char *name, CkCallFnPtr call, int msgIdx, int chareIdx)
{
  _entryTable[_numEntries] = new EntryInfo(name, call, msgIdx, chareIdx);
  return _numEntries++;
}

extern "C"
int CkRegisterChare(const char *name, int dataSz)
{
  _chareTable[_numChares] = new ChareInfo(name, dataSz);
  return _numChares++;
}

extern "C"
int CkRegisterMainChare(int chareIdx, int entryIdx)
{
  _mainTable[_numMains] = new MainInfo(chareIdx, entryIdx);
  return _numMains++;
}

extern "C"
void CkRegisterReadonly(int size, void *ptr)
{
  _readonlyTable[_numReadonlies++] = new ReadonlyInfo(size, ptr);
  return;
}

extern "C"
void CkRegisterReadonlyMsg(void **pMsg)
{
  _readonlyMsgs[_numReadonlyMsgs++] = new ReadonlyMsgInfo(pMsg);
}

// temporarily here for satisfying NAMD, it should go to tracing module

extern "C"
int registerEvent(char *name)
{
  return 0;
}
