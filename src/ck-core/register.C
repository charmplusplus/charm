/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

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

static int __registerDone = 0;;

void _registerInit(void)
{
  if(__registerDone)
    return;
  _numEntries =0;
  _numMsgs = 0;
  _numChares = 0;
  _numMains = 0;
  _numReadonlies = 0;
  _numReadonlyMsgs = 0;
  _entryTable = new EntryInfo*[_ENTRY_TABLE_SIZE];
  _MEMCHECK(_entryTable);
  _msgTable = new MsgInfo*[_MSG_TABLE_SIZE];
  _MEMCHECK(_msgTable);
  _chareTable = new ChareInfo*[_CHARE_TABLE_SIZE];
  _MEMCHECK(_chareTable);
  _mainTable = new MainInfo*[_MAIN_TABLE_SIZE];
  _MEMCHECK(_mainTable);
  _readonlyTable = new ReadonlyInfo*[_READONLY_TABLE_SIZE];
  _MEMCHECK(_readonlyTable);
  _readonlyMsgs = new ReadonlyMsgInfo*[_READONLY_TABLE_SIZE];;
  _MEMCHECK(_readonlyMsgs);
  __registerDone = 1;
}

extern "C"
int CkRegisterMsg(const char *name, CkPackFnPtr pack, CkUnpackFnPtr unpack, 
                  CkCoerceFnPtr coerce, size_t size)
{
  if(!__registerDone)
    _registerInit();
  _msgTable[_numMsgs] = new MsgInfo(name, pack, unpack, coerce, size);
  _MEMCHECK(_msgTable[_numMsgs]);
  return _numMsgs++;
}

extern "C"
int CkRegisterEp(const char *name, CkCallFnPtr call, int msgIdx, int chareIdx)
{
  if(!__registerDone)
    _registerInit();
  _entryTable[_numEntries] = new EntryInfo(name, call, msgIdx, chareIdx);
  _MEMCHECK(_entryTable[_numEntries]);
  return _numEntries++;
}

extern "C"
int CkRegisterChare(const char *name, int dataSz)
{
  if(!__registerDone)
    _registerInit();
  _chareTable[_numChares] = new ChareInfo(name, dataSz);
  _MEMCHECK(_chareTable[_numChares]);
  return _numChares++;
}

extern "C"
void CkRegisterDefaultCtor(int chareIdx, int ctorEpIdx)
{
  if(!__registerDone)
    _registerInit();
  _chareTable[chareIdx]->setDefaultCtor(ctorEpIdx);
}
extern "C"
void CkRegisterMigCtor(int chareIdx, int ctorEpIdx)
{
  if(!__registerDone)
    _registerInit();
  _chareTable[chareIdx]->setMigCtor(ctorEpIdx);
}

extern "C"
int CkRegisterMainChare(int chareIdx, int entryIdx)
{
  if(!__registerDone)
    _registerInit();
  _mainTable[_numMains] = new MainInfo(chareIdx, entryIdx);
  _MEMCHECK(_mainTable[_numMains]);
  return _numMains++;
}

extern "C"
void CkRegisterReadonly(const char *name,const char *type,
	int size, void *ptr,CkPupReadonlyFnPtr pup_fn)
{
  if(!__registerDone)
    _registerInit();
  _readonlyTable[_numReadonlies] = 
	new ReadonlyInfo(name,type,size,ptr,pup_fn);
  _MEMCHECK(_readonlyTable[_numReadonlies]);
  _numReadonlies++;
  return;
}

extern "C"
void CkRegisterReadonlyMsg(const char *name,const char *type,void **pMsg)
{
  if(!__registerDone)
    _registerInit();
  _readonlyMsgs[_numReadonlyMsgs] = new ReadonlyMsgInfo(name,type,pMsg);
  _MEMCHECK(_readonlyMsgs[_numReadonlyMsgs]);
  _numReadonlyMsgs++;
  return;
}


extern "C"
void CkRegisterMarshallUnpackFn(int epIndex,CkMarshallUnpackFn m)
{
	_entryTable[epIndex]->marshallUnpack=m;
}

extern "C"
CkMarshallUnpackFn CkLookupMarshallUnpackFn(int epIndex)
{
	return _entryTable[epIndex]->marshallUnpack;
}

extern "C" 
int CkDisableTracing(int epIdx) {
	int oldStatus = _entryTable[epIdx]->traceEnabled;
	_entryTable[epIdx]->traceEnabled=false;
	return oldStatus;
}

extern "C" 
void CkEnableTracing(int epIdx) {
	_entryTable[epIdx]->traceEnabled=true;
}


// temporarily here for satisfying NAMD, it should go to tracing module

extern "C"
int registerEvent(char *name)
{
  if(!__registerDone)
    _registerInit();
  return 0;
}

int _GETIDX(int cidx)
{
  return _chareTable[cidx]->getClassIdx();
}

void _REGISTER_BASE(int didx, int bidx)
{
  if (bidx!=-1)
    _chareTable[didx]->addBase(bidx);
}

//These pup functions are used by the CpdList interface, below
#define PCOM(field) p.comment(#field); p(c->field);
#define PCOMS(field) p.comment(#field); p((char *)c->field,strlen(c->field));
static void pupEntry(PUP::er &p,int i)
{
  EntryInfo *c=_entryTable[i];
  PCOMS(name) PCOM(msgIdx) PCOM(chareIdx)
}
static void pupMsg(PUP::er &p,int i)
{
  MsgInfo *c=_msgTable[i];
  PCOMS(name) PCOM(size)
}
static void pupChare(PUP::er &p,int i)
{
  ChareInfo *c=_chareTable[i];
  PCOMS(name) PCOM(size) PCOM(defCtor) PCOM(migCtor)
  PCOM(numbases)
  p.comment("List of base classes:");
  p(c->bases,c->numbases);
}
static void pupMain(PUP::er &p,int i)
{
  MainInfo *c=_mainTable[i];
  PCOM(chareIdx) PCOM(entryIdx)
}
static void pupReadonly(PUP::er &p,int i)
{
  ReadonlyInfo *c=_readonlyTable[i];
  PCOMS(name) PCOMS(type) PCOM(size) 
  p.comment("value");
  c->pupData(p);
}
static void pupReadonlyMsg(PUP::er &p,int i)
{
  ReadonlyMsgInfo *c=_readonlyMsgs[i];
  PCOMS(name) PCOMS(type)
  p.comment("value");
  CkPupMessage(p,c->pMsg,0);
}

void _REGISTER_DONE(void)
{
  CpdListRegister(new CpdSimpleListAccessor("charm/entries",_numEntries,pupEntry));
  CpdListRegister(new CpdSimpleListAccessor("charm/messages",_numMsgs,pupMsg));
  CpdListRegister(new CpdSimpleListAccessor("charm/chares",_numChares,pupChare));
  CpdListRegister(new CpdSimpleListAccessor("charm/mains",_numMains,pupMain));
  CpdListRegister(new CpdSimpleListAccessor("charm/readonly",_numReadonlies,pupReadonly));
  CpdListRegister(new CpdSimpleListAccessor("charm/readonlyMsg",_numReadonlyMsgs,pupReadonlyMsg));
}

//Print a debugging version of this entry method index:
void CkPrintEntryMethod(int epIdx) {
	if (epIdx<=0 || epIdx>=_numEntries) 
		CkPrintf("INVALID ENTRY METHOD %d!",epIdx);
	else {
		EntryInfo *e=_entryTable[epIdx];
		CkPrintChareName(e->chareIdx);
		CkPrintf("::%s",e->name);
	}
}

//Print a debugging version of this chare index:
void CkPrintChareName(int chareIdx) {
	if (chareIdx<=0 || chareIdx>=_numChares)
		CkPrintf("INVALID CHARE INDEX %d!",chareIdx);
	else {
		ChareInfo *c=_chareTable[chareIdx];
		CkPrintf("%s",c->name);
	}
}




