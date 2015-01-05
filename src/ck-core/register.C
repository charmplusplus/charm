/**
\addtogroup CkRegister

These routines keep track of the various chares
(see ChareInfo and the _chareTable in register.h)
and entry methods (see EntryInfo and the _entryTable)
that exist in the program.

These routines are normally called by a translator-generated
_registerModule routine in the .def file.  Because these routines
fill out global tables, they are normally called exactly
once per node at Charm startup time.
*/
#include "ck.h"
#include "debug-charm.h"

CkRegisteredInfo<EntryInfo> _entryTable;
CkRegisteredInfo<MsgInfo> _msgTable;
CkRegisteredInfo<ChareInfo> _chareTable;
CkRegisteredInfo<MainInfo> _mainTable;
CkRegisteredInfo<ReadonlyInfo> _readonlyTable;
CkRegisteredInfo<ReadonlyMsgInfo> _readonlyMsgs;

static int __registerDone = 0;

void _registerInit(void)
{
  if(__registerDone)
    return;
  __registerDone = 1;
}

extern "C"
int CkRegisterMsg(const char *name, CkPackFnPtr pack, CkUnpackFnPtr unpack,
                  CkDeallocFnPtr dealloc, size_t size)
{
  return _msgTable.add(new MsgInfo(name, pack, unpack, dealloc, size));
}

extern "C"
void ckInvalidCallFn(void *msg,void *obj) {
  CkAbort("Charm++: Invalid entry method executed.  Perhaps there is an unregistered module?");
}

extern "C"
int CkRegisterEp(const char *name, CkCallFnPtr call, int msgIdx, int chareIdx,
	int ck_ep_flags)
{
  EntryInfo *e = new EntryInfo(name, call?call:ckInvalidCallFn, msgIdx, chareIdx);
  if (ck_ep_flags & CK_EP_NOKEEP) e->noKeep=true;
  if (ck_ep_flags & CK_EP_INTRINSIC) e->inCharm=true;
  if (ck_ep_flags & CK_EP_TRACEDISABLE) e->traceEnabled=false;
  if (ck_ep_flags & CK_EP_APPWORK) e->appWork=true;
#if ADAPT_SCHED_MEM
  if (ck_ep_flags & CK_EP_MEMCRITICAL){
     e->isMemCritical=true;
     if (CkMyRank()==0)
        numMemCriticalEntries++;
  }else{
    e->isMemCritical=false;
  }
#endif
  return _entryTable.add(e);
}

extern "C"
int CkRegisterChare(const char *name, size_t dataSz, ChareType chareType)
{
  return _chareTable.add(new ChareInfo(name, dataSz, chareType));
}

extern "C"
void CkRegisterChareInCharm(int chareIndex){
  _chareTable[chareIndex]->inCharm = true;
}

extern "C"
void CkRegisterGroupIrr(int chareIndex,int isIrr){
  _chareTable[chareIndex]->isIrr = isIrr;
}

extern "C"
void CkRegisterDefaultCtor(int chareIdx, int ctorEpIdx)
{
  _chareTable[chareIdx]->setDefaultCtor(ctorEpIdx);
}
extern "C"
void CkRegisterMigCtor(int chareIdx, int ctorEpIdx)
{
  _chareTable[chareIdx]->setMigCtor(ctorEpIdx);
}

extern "C"
int CkRegisterMainChare(int chareIdx, int entryIdx)
{
  int mIdx =  _mainTable.add(new MainInfo(chareIdx, entryIdx));
  _chareTable[chareIdx]->setMainChareType(mIdx);
  return mIdx;
}

extern "C"
void CkRegisterBase(int derivedIdx, int baseIdx)
{
  if (baseIdx!=-1)
    _chareTable[derivedIdx]->addBase(baseIdx);
}

extern "C"
void CkRegisterReadonly(const char *name,const char *type,
	int size, void *ptr,CkPupReadonlyFnPtr pup_fn)
{
  _readonlyTable.add(new ReadonlyInfo(name,type,size,ptr,pup_fn));
}

extern "C"
void CkRegisterReadonlyMsg(const char *name,const char *type,void **pMsg)
{
  _readonlyMsgs.add(new ReadonlyMsgInfo(name,type,pMsg));
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
void CkRegisterMessagePupFn(int epIndex,CkMessagePupFn m)
{
#if CMK_CHARMDEBUG
	_entryTable[epIndex]->messagePup=m;
#endif
}
extern "C"
int CkDisableTracing(int epIdx) {
	CmiLock(_smp_mutex);
	int oldStatus = _entryTable[epIdx]->traceEnabled;
	_entryTable[epIdx]->traceEnabled=false;
	CmiUnlock(_smp_mutex);
	return oldStatus;
}

extern "C"
void CkEnableTracing(int epIdx) {
	CmiLock(_smp_mutex);
	_entryTable[epIdx]->traceEnabled=true;
	CmiUnlock(_smp_mutex);
}

#if CMK_CHARMDEBUG
static void pupEntry(PUP::er &p,int index)
{
  EntryInfo *c=_entryTable[index];
  PCOMS(name)
  p.comment("index");
  p(index);
  PCOM(msgIdx)
  PCOM(chareIdx)
  PCOM(inCharm);
}

static void pupMsg(PUP::er &p,int i)
{
  MsgInfo *c=_msgTable[i];
  PCOMS(name) PCOM(size)
}
static void pupChare(PUP::er &p,int i)
{
  ChareInfo *c=_chareTable[i];
  PCOMS(name) PCOM(size)
  PCOM(defCtor) PCOM(migCtor)
  PCOM(numbases)
  PCOM(inCharm)
  p.comment("List of base classes:");
  p(c->bases,c->numbases);
}
static void pupMain(PUP::er &p,int i)
{
  MainInfo *c=_mainTable[i];
  PCOMS(name) PCOM(chareIdx) PCOM(entryIdx)
}
static void pupReadonly(PUP::er &p,int i)
{
  ReadonlyInfo *c=_readonlyTable[i];
  PCOMS(name) PCOMS(type) PCOM(size)
  p.comment("value");
  //c->pupData(p); Do not use puppers, just copy memory
  p((char *)c->ptr,c->size);
}
static void pupReadonlyMsg(PUP::er &p,int i)
{
  ReadonlyMsgInfo *c=_readonlyMsgs[i];
  PCOMS(name) PCOMS(type)
  p.comment("value");
  CkPupMessage(p,c->pMsg,0);
}
#endif
extern void CpdCharmInit(void);

void _registerDone(void)
{
#if CMK_CHARMDEBUG
  CpdListRegister(new CpdSimpleListAccessor("charm/entries",_entryTable.size(),pupEntry));
  CpdListRegister(new CpdSimpleListAccessor("charm/messages",_msgTable.size(),pupMsg));
  CpdListRegister(new CpdSimpleListAccessor("charm/chares",_chareTable.size(),pupChare));
  CpdListRegister(new CpdSimpleListAccessor("charm/mains",_mainTable.size(),pupMain));
  CpdListRegister(new CpdSimpleListAccessor("charm/readonly",_readonlyTable.size(),pupReadonly));
  CpdListRegister(new CpdSimpleListAccessor("charm/readonlyMsg",_readonlyMsgs.size(),pupReadonlyMsg));

  CpdCharmInit();
#endif
}

//Print a debugging version of this entry method index:
void CkPrintEntryMethod(int epIdx) {
	if (epIdx<=0 || epIdx>=(int)_entryTable.size())
		CkPrintf("INVALID ENTRY METHOD %d!",epIdx);
	else {
		EntryInfo *e=_entryTable[epIdx];
		CkPrintChareName(e->chareIdx);
		CkPrintf("::%s",e->name);
	}
}

//Print a debugging version of this chare index:
void CkPrintChareName(int chareIdx) {
	if (chareIdx<=0 || chareIdx>=(int)_chareTable.size())
		CkPrintf("INVALID CHARE INDEX %d!",chareIdx);
	else {
		ChareInfo *c=_chareTable[chareIdx];
		CkPrintf("%s",c->name);
	}
}




