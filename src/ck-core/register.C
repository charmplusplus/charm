/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
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

CkRegisteredInfo<EntryInfo> _entryTable;
CkRegisteredInfo<MsgInfo> _msgTable;
CkRegisteredInfo<ChareInfo> _chareTable;
CkRegisteredInfo<MainInfo> _mainTable;
CkRegisteredInfo<ReadonlyInfo> _readonlyTable;
CkRegisteredInfo<ReadonlyMsgInfo> _readonlyMsgs;

static int __registerDone = 0;;

void _registerInit(void)
{
  if(__registerDone)
    return;
  __registerDone = 1;
}

extern "C"
int CkRegisterMsg(const char *name, CkPackFnPtr pack, CkUnpackFnPtr unpack, 
                  size_t size)
{
  return _msgTable.add(new MsgInfo(name, pack, unpack, size));
}

extern "C"
int CkRegisterEp(const char *name, CkCallFnPtr call, int msgIdx, int chareIdx,
	int ck_ep_flags)
{
  EntryInfo *e = new EntryInfo(name, call, msgIdx, chareIdx);
  if (ck_ep_flags & CK_EP_NOKEEP) e->noKeep=CmiTrue;
  if (ck_ep_flags & CK_EP_INTRINSIC) e->inCharm=CmiTrue;
  return _entryTable.add(e);
}

extern "C"
int CkRegisterChare(const char *name, int dataSz)
{
  return _chareTable.add(new ChareInfo(name, dataSz));
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
  return _mainTable.add(new MainInfo(chareIdx, entryIdx));
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
int CkDisableTracing(int epIdx) {
	int oldStatus = _entryTable[epIdx]->traceEnabled;
	_entryTable[epIdx]->traceEnabled=CmiFalse;
	return oldStatus;
}

extern "C" 
void CkEnableTracing(int epIdx) {
	_entryTable[epIdx]->traceEnabled=CmiTrue;
}

//These pup functions are used by the CpdList interface, below
#define PCOM(field) p.comment(#field); p(c->field);
#define PCOMS(field) p.comment(#field); p((char *)c->field,strlen(c->field));
static void pupEntry(PUP::er &p,int i)
{
  EntryInfo *c=_entryTable[i];
  PCOMS(name) 
  PCOM(msgIdx) 
  PCOM(chareIdx)
  if (c->inCharm == CmiTrue)
    p.comment("System Entry Point");
  else
    p.comment("User Entry Point");
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
  c->pupData(p);
}
static void pupReadonlyMsg(PUP::er &p,int i)
{
  ReadonlyMsgInfo *c=_readonlyMsgs[i];
  PCOMS(name) PCOMS(type)
  p.comment("value");
  CkPupMessage(p,c->pMsg,0);
}

class GroupIterator : public CkLocIterator {
private:
   PUP::er &p;
public:
   GroupIterator(PUP::er &_p) :p(_p){}
   ~GroupIterator() {}
   void addLocation (CkLocation & loc)
   {
     p.comment("Element details");
     const CkArrayIndex &idx = loc.getIndex();
     const int * idxData = idx.data();
     char buf[128];
     char * temp = buf;
     for(int i=0; i < idx.nInts; i++)
     {
        sprintf(temp, "%s%d",i==0?"":":", idxData[i]);
        temp += strlen(temp);
     } 
     p(buf, strlen(buf));
     //loc.pup(p);
   }
};




static void pupArray(PUP::er &p, int i)
{
  IrrGroup * c;
  c = (CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i])).getObj();
  GroupIterator itr(p);
  char buf[128];
  if (c->isLocMgr())
  {
   //int groupID = (((CkLocMgr *)c)->getGroupID()).idx;
   p.comment("Array");
   p(i);
   ((CkLocMgr*)(c))->iterate(itr);
    
  }
  else
  {
    p.comment("Group");
    p.comment("Not an Array Location Mgr");
  }
}

CpvDeclare(int, groupTableSize);

extern void CpdCharmInit(void);

void _registerDone(void)
{
  CpdListRegister(new CpdSimpleListAccessor("charm/entries",_entryTable.size(),pupEntry));
  CpdListRegister(new CpdSimpleListAccessor("charm/messages",_msgTable.size(),pupMsg));
  CpdListRegister(new CpdSimpleListAccessor("charm/chares",_chareTable.size(),pupChare));
  CpdListRegister(new CpdSimpleListAccessor("charm/mains",_mainTable.size(),pupMain));
  CpdListRegister(new CpdSimpleListAccessor("charm/readonly",_readonlyTable.size(),pupReadonly));
  CpdListRegister(new CpdSimpleListAccessor("charm/readonlyMsg",_readonlyMsgs.size(),pupReadonlyMsg));
 
  CpdListRegister(new CpdSimpleListAccessor("charm/arrayelements", CkpvAccess(_groupIDTable)->length(), pupArray));
#if CMK_CCS_AVAILABLE
  CpdCharmInit();
#endif
}

//Print a debugging version of this entry method index:
void CkPrintEntryMethod(int epIdx) {
	if (epIdx<=0 || epIdx>=_entryTable.size()) 
		CkPrintf("INVALID ENTRY METHOD %d!",epIdx);
	else {
		EntryInfo *e=_entryTable[epIdx];
		CkPrintChareName(e->chareIdx);
		CkPrintf("::%s",e->name);
	}
}

//Print a debugging version of this chare index:
void CkPrintChareName(int chareIdx) {
	if (chareIdx<=0 || chareIdx>=_chareTable.size())
		CkPrintf("INVALID CHARE INDEX %d!",chareIdx);
	else {
		ChareInfo *c=_chareTable[chareIdx];
		CkPrintf("%s",c->name);
	}
}




