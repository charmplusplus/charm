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
#include "ckarray.h"
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
}

int CkRegisterMsg(const char *name, CkPackFnPtr pack, CkUnpackFnPtr unpack,
                  CkDeallocFnPtr dealloc, size_t size)
{
  auto messageSize = _msgTable.size();
  return _msgTable.add(new MsgInfo(name, pack, unpack, dealloc, size));
}

void ckInvalidCallFn(void *msg,void *obj) {
  CkAbort("Charm++: Invalid entry method executed.  Perhaps there is an unregistered module?");
}

static
int CkRegisterEpInternal(const char *name, CkCallFnPtr call, int msgIdx, int chareIdx,
	int ck_ep_flags, bool isTemplated)
{
#if !CMK_CHARM4PY    // charm4py can support dynamic registration of Chares after program start
  if (__registerDone) {
    CkPrintf("Charm++: late entry method registration happened after init\nEntry point: %s, addr: %p\n", name, (void *)call);
    CkAbort("Did you forget to import a module or instantiate a templated entry method in a .ci file?\n");
  }
#endif
  EntryInfo *e = new EntryInfo(name, call?call:ckInvalidCallFn, msgIdx, chareIdx, isTemplated);
  if (ck_ep_flags & CK_EP_NOKEEP) e->noKeep=true;
  if (ck_ep_flags & CK_EP_INTRINSIC) e->inCharm=true;
  if (ck_ep_flags & CK_EP_TRACEDISABLE) e->traceEnabled=false;
  if (ck_ep_flags & CK_EP_APPWORK) e->appWork=true;
  if (ck_ep_flags & CK_EP_IMMEDIATE) e->isImmediate=true;
  if (ck_ep_flags & CK_EP_INLINE) e->isInline=true;
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

int CkRegisterEp(const char *name, CkCallFnPtr call, int msgIdx, int chareIdx, int ck_ep_flags)
{
  return CkRegisterEpInternal(name, call, msgIdx, chareIdx, ck_ep_flags, false /*=isTemplated*/);
}

int CkRegisterEpTemplated(const char *name, CkCallFnPtr call, int msgIdx, int chareIdx, int ck_ep_flags)
{
  return CkRegisterEpInternal(name, call, msgIdx, chareIdx, ck_ep_flags, true /*=isTemplated*/);
}

int CkRegisterChare(const char *name, size_t dataSz, ChareType chareType)
{
  return _chareTable.add(new ChareInfo(name, dataSz, chareType));
}

void CkRegisterArrayDimensions(int chareIndex, int ndims) {
  _chareTable[chareIndex]->ndims = ndims;
}

void CkRegisterChareInCharm(int chareIndex){
  _chareTable[chareIndex]->inCharm = true;
}

void CkRegisterGroupIrr(int chareIndex,int isIrr){
  _chareTable[chareIndex]->isIrr = (isIrr!=0);
}

#if CMK_CHARM4PY

// TODO give a unique name to entry methods when calling CkRegisterEp
// (no need has appeared for this so this is very low priority)

void CkRegisterGroupExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx) {
  int __idx = CkRegisterChare(s, sizeof(GroupExt), TypeGroup);
  CkRegisterBase(__idx, CkIndex_IrrGroup::__idx);

  int epIdxCtor = CkRegisterEp(s, GroupExt::__GroupExt, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterDefaultCtor(__idx, epIdxCtor);

  for (int i=emNamesStart + 1; i < emNamesStart+numEntryMethods; i++)
    {
      CkRegisterEp(emNames[i], GroupExt::__entryMethod, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
    }

  *chareIdx = __idx;
  *startEpIdx = epIdxCtor;
}

void CkRegisterSectionManagerExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx) {
  int __idx = CkRegisterChare(s, sizeof(SectionManagerExt), TypeGroup);
  CkRegisterBase(__idx, CkIndex_IrrGroup::__idx);

  int epIdxCtor = CkRegisterEp(s, SectionManagerExt::__SectionManagerExt, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterDefaultCtor(__idx, epIdxCtor);

  CkRegisterEp("sendToSection", SectionManagerExt::__sendToSection, CkMarshallMsg::__idx, __idx, 0);

  for (int i=emNamesStart+2; i < numEntryMethods+emNamesStart; i++)
    {
      CkRegisterEp(emNames[i], SectionManagerExt::__entryMethod, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
    }

  *chareIdx = __idx;
  *startEpIdx = epIdxCtor;
}

void CkRegisterArrayMapExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx) {
  int __idx = CkRegisterChare(s, sizeof(ArrayMapExt), TypeGroup);
  CkRegisterBase(__idx, CkIndex_IrrGroup::__idx);

  int epIdxCtor = CkRegisterEp(s, ArrayMapExt::__ArrayMapExt, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterDefaultCtor(__idx, epIdxCtor);

  for (int i=1; i < numEntryMethods; i++)
    CkRegisterEp(emNames[i], ArrayMapExt::__entryMethod, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);

  *chareIdx = __idx;
  *startEpIdx = epIdxCtor;
}

void CkRegisterArrayExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx) {
  int __idx = CkRegisterChare(s, sizeof(ArrayElemExt), TypeArray);
  CkRegisterBase(__idx, CkIndex_ArrayElement::__idx);

  int epIdxCtor = CkRegisterEp(s, ArrayElemExt::__ArrayElemExt, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterDefaultCtor(__idx, epIdxCtor);

  int epidx = CkRegisterEp(s, ArrayElemExt::__CkMigrateMessage, 0, __idx, 0);
  CkRegisterMigCtor(__idx, epidx);

  CkRegisterEp(s, ArrayElemExt::__AtSyncEntryMethod, 0, __idx, 0);

  for (int i=emNamesStart+3; i < emNamesStart+numEntryMethods; i++)
    {
      CkRegisterEp(emNames[i], ArrayElemExt::__entryMethod, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
    }

  *chareIdx = __idx;
  *startEpIdx = epIdxCtor;
}

#endif

void CkRegisterDefaultCtor(int chareIdx, int ctorEpIdx)
{
  _chareTable[chareIdx]->setDefaultCtor(ctorEpIdx);
}
void CkRegisterMigCtor(int chareIdx, int ctorEpIdx)
{
  _chareTable[chareIdx]->setMigCtor(ctorEpIdx);
}

int CkRegisterMainChare(int chareIdx, int entryIdx)
{
  int mIdx =  _mainTable.add(new MainInfo(chareIdx, entryIdx));
  _chareTable[chareIdx]->setMainChareType(mIdx);
  return mIdx;
}

#if CMK_CHARM4PY

void CkRegisterMainChareExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx) {
  int __idx = CkRegisterChare(s, sizeof(MainchareExt), TypeMainChare);
  CkRegisterBase(__idx, CkIndex_Chare::__idx);

  int epIdxCtor = CkRegisterEp(s, MainchareExt::__Ctor_CkArgMsg, CMessage_CkArgMsg::__idx, __idx, 0);
  CkRegisterMessagePupFn(epIdxCtor, (CkMessagePupFn)CkArgMsg::ckDebugPup);
  CkRegisterMainChare(__idx, epIdxCtor);

  for (int i=emNamesStart+1; i < emNamesStart+numEntryMethods; i++)
    {
      CkRegisterEp(emNames[i], MainchareExt::__entryMethod, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
    }

  *chareIdx = __idx;
  *startEpIdx = epIdxCtor;
}
#endif

void CkRegisterIfNotThere(int epIdx, CkArray_IfNotThere policy) {
  if ((epIdx >= 0) && (!__registerDone || CMK_CHARM4PY)) {
    _entryTable[epIdx]->ifNotThere = policy;
  }
}

void CkRegisterBase(int derivedIdx, int baseIdx)
{
  if (baseIdx!=-1)
    _chareTable[derivedIdx]->addBase(baseIdx);
}

int CkGetChareIdx(const char *name){
  for(int i=0; i<_chareTable.size(); i++){
    if(strcmp(name, _chareTable[i]->name)==0)
      return i;
  }
  return -1;
}

void CkRegisterReadonly(const char *name,const char *type,
	size_t size, void *ptr,CkPupReadonlyFnPtr pup_fn)
{
  _readonlyTable.add(new ReadonlyInfo(name,type,size,ptr,pup_fn));
}

#if CMK_CHARM4PY
void CkRegisterReadonlyExt(const char *name, const char *type, size_t msgSize, char *msg) {
  if (msgSize > 0) ReadOnlyExt::setData(msg, msgSize);
  CkRegisterReadonly(name, type, msgSize, ReadOnlyExt::ro_data, ReadOnlyExt::_roPup);
}
#endif

void CkRegisterReadonlyMsg(const char *name,const char *type,void **pMsg)
{
  _readonlyMsgs.add(new ReadonlyMsgInfo(name,type,pMsg));
}


void CkRegisterMarshallUnpackFn(int epIndex,CkMarshallUnpackFn m)
{
  _entryTable[epIndex]->marshallUnpack=m;
}

CkMarshallUnpackFn CkLookupMarshallUnpackFn(int epIndex)
{
  return _entryTable[epIndex]->marshallUnpack;
}
void CkRegisterMessagePupFn(int epIndex,CkMessagePupFn m)
{
#if CMK_CHARMDEBUG
	_entryTable[epIndex]->messagePup=m;
#endif
}
int CkDisableTracing(int epIdx) {
	CmiLock(_smp_mutex);
	int oldStatus = _entryTable[epIdx]->traceEnabled;
	_entryTable[epIdx]->traceEnabled=false;
	CmiUnlock(_smp_mutex);
	return oldStatus;
}

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
  if (CkMyRank() == 0)
    __registerDone = 1;

#if CMK_CHARMDEBUG
  CpdListRegister(new CpdSimpleListAccessor<EntryInfo>("charm/entries",&_entryTable,pupEntry));
  CpdListRegister(new CpdSimpleListAccessor<MsgInfo>("charm/messages",&_msgTable,pupMsg));
  CpdListRegister(new CpdSimpleListAccessor<ChareInfo>("charm/chares",&_chareTable,pupChare));
  CpdListRegister(new CpdSimpleListAccessor<MainInfo>("charm/mains",&_mainTable,pupMain));
  CpdListRegister(new CpdSimpleListAccessor<ReadonlyInfo>("charm/readonly",&_readonlyTable,pupReadonly));
  CpdListRegister(new CpdSimpleListAccessor<ReadonlyMsgInfo>("charm/readonlyMsg",&_readonlyMsgs,pupReadonlyMsg));
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




