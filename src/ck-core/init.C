#include "ck.h"

UChar _defaultQueueing = CK_QUEUEING_FIFO;
UInt  _printCS = 0;
UInt  _printSS = 0;
UInt  _numGroups = 0;
UInt  _numInitMsgs = 0;
int   _infoIdx;
int   _charmHandlerIdx;
int   _initHandlerIdx;
int   _bocHandlerIdx;
int   _qdHandlerIdx;

CkOutStream ckout;
CkErrStream ckerr;
CkInStream  ckin;

CpvDeclare(void*,       _currentChare);
CpvDeclare(int,         _currentGroup);
CpvDeclare(GroupTable*, _groupTable);
CpvDeclare(Stats*, _myStats);
CpvDeclare(MsgPool*, _msgPool);

CpvDeclare(_CkOutStream*, _ckout);
CpvDeclare(_CkErrStream*, _ckerr);

CpvStaticDeclare(UInt,  _numInitsRecd);
CpvStaticDeclare(PtrQ*, _buffQ);
CpvStaticDeclare(PtrQ*, _bocInitQ);

static int    _exitHandlerIdx;
static Stats** _allStats = 0;
static UInt   _numStatsRecd = 0;
static int    _exitStarted = 0;

static inline int _parseCommandLineOpts(int argc, char **argv)
{
  int found;
  while(*argv) {
    found = 0;
    if(strcmp(*argv, "+cs")==0) {
      _printCS = 1; found = 1;
    } else if(strcmp(*argv, "+ss")==0) {
      _printSS = 1; found = 1;
    } else if(strcmp(*argv, "+fifo")==0) {
      _defaultQueueing = CK_QUEUEING_FIFO; found = 1;
    } else if(strcmp(*argv, "+lifo")==0) {
      _defaultQueueing = CK_QUEUEING_LIFO; found = 1;
    } else if(strcmp(*argv, "+ififo")==0) {
      _defaultQueueing = CK_QUEUEING_IFIFO; found = 1;
    } else if(strcmp(*argv, "+ilifo")==0) {
      _defaultQueueing = CK_QUEUEING_ILIFO; found = 1;
    } else if(strcmp(*argv, "+bfifo")==0) {
      _defaultQueueing = CK_QUEUEING_BFIFO; found = 1;
    } else if(strcmp(*argv, "+blifo")==0) {
      _defaultQueueing = CK_QUEUEING_BLIFO; found = 1;
    }
    if(found) {
      argc--;
      char **next = argv;
      while(*next) {
        *next = *(next+1);
        next++;
      }
    } else
      argv++;
  }
  return argc;
}

static void _bufferHandler(void *msg)
{
  CmiGrabBuffer(&msg);
  CpvAccess(_buffQ)->enq(msg);
}

static void _discardHandler(envelope *env)
{
  CmiGrabBuffer((void **)&env);
  CmiFree(env);
}


static inline void _printStats(void)
{
  int i;
  if(_printSS || _printCS) {
    Stats *total = new Stats();
    for(i=0;i<CkNumPes();i++)
      total->combine(_allStats[i]);
    CkPrintf("Charm Kernel Summary Statistics:\n");
    for(i=0;i<CkNumPes();i++) {
      CkPrintf("Proc %d: [%d created, %d processed]\n", i, 
               _allStats[i]->getCharesCreated(),
               _allStats[i]->getCharesProcessed());
    }
    CkPrintf("Total Chares: [%d created, %d processed]\n",
             total->getCharesCreated(), total->getCharesProcessed());
  }
  if(_printCS) {
    CkPrintf("Charm Kernel Detailed Statistics:\n");
    CkPrintf("PE\tCC\tCP\tFCC\tFCP\tGC\tGP\tFGC\tFGP\n");
    for(i=0;i<CkNumPes();i++) {
      CkPrintf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",i,
               _allStats[i]->getCharesCreated(),
               _allStats[i]->getCharesProcessed(),
               _allStats[i]->getForCharesCreated(),
               _allStats[i]->getForCharesProcessed(),
               _allStats[i]->getGroupsCreated(),
               _allStats[i]->getGroupsProcessed(),
               _allStats[i]->getGroupMsgsCreated(),
               _allStats[i]->getGroupMsgsProcessed());
    }
  }
}

static inline void _sendStats(void)
{
  envelope *env = UsrToEnv(CpvAccess(_myStats));
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _exitHandlerIdx);
  CmiSyncSendAndFree(0, env->getTotalsize(), env);
}

static void _exitHandler(envelope *env)
{
  CmiGrabBuffer((void **)&env);
  switch(env->getMsgtype()) {
    case ExitMsg:
      assert(CkMyPe()==0);
      if(_exitStarted) {
        CmiFree(env);
        return;
      }
      _exitStarted = 1; 
      CmiNumberHandler(_charmHandlerIdx,(CmiHandler)_discardHandler);
      CmiNumberHandler(_bocHandlerIdx, (CmiHandler)_discardHandler);
      env->setMsgtype(ReqStatMsg);
      env->setSrcPe(CkMyPe());
      CmiSyncBroadcastAllAndFree(env->getTotalsize(), env);
      break;
    case ReqStatMsg:
      CmiNumberHandler(_charmHandlerIdx,(CmiHandler)_discardHandler);
      CmiNumberHandler(_bocHandlerIdx, (CmiHandler)_discardHandler);
      CmiFree(env);
      _sendStats();
      if(CkMyPe())
        CsdExitScheduler();
      break;
    case StatMsg:
      assert(CkMyPe()==0);
      _allStats[env->getSrcPe()] = (Stats*) EnvToUsr(env);
      _numStatsRecd++;
      if(_numStatsRecd==CkNumPes()) {
        _printStats();
        CsdExitScheduler();
      }
      break;
    default:
      CmiAbort("Internal Error: Unknown-msg-type. Contact Developers.\n");
  }
}

static inline void _processBufferedBocInits(void)
{
  register envelope *env;
  CmiNumberHandler(_bocHandlerIdx, (CmiHandler)_processHandler);
  while(env=(envelope *)CpvAccess(_bocInitQ)->deq()) {
    if(env->isPacked() && _msgTable[env->getMsgIdx()]->unpack)
      env = UsrToEnv(_msgTable[env->getMsgIdx()]->unpack(EnvToUsr(env)));
    _processBocInitMsg(env);
  }
}

static inline void _processBufferedMsgs(void)
{
  CmiNumberHandler(_charmHandlerIdx,(CmiHandler)_processHandler);
  envelope *env;
  while(env=(envelope*)CpvAccess(_buffQ)->deq()) {
    CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), env);
  }
}

static inline void _initDone(void)
{
  CmiNumberHandler(_exitHandlerIdx, (CmiHandler)_exitHandler);
  _processBufferedBocInits();
  _processBufferedMsgs();
}

static inline void _processROMsgMsg(envelope *env)
{
  *((char **)(_readonlyMsgs[env->getRoIdx()]->pMsg))=(char *)EnvToUsr(env);
}

static inline void _processRODataMsg(envelope *env)
{
  register int i;
  register char *msg = (char *)EnvToUsr(env);
  for(i=0;i<_numReadonlies;i++) {
    memcpy(_readonlyTable[i]->ptr, msg, _readonlyTable[i]->size);
    msg += _readonlyTable[i]->size;
  }
  CmiFree(env);
}

static void _initHandler(void *msg)
{
  assert(CkMyPe()!=0);
  CpvAccess(_numInitsRecd)++;
  CmiGrabBuffer(&msg);
  register envelope *env = (envelope *) msg;
  switch (env->getMsgtype()) {
    case BocInitMsg:
      CpvAccess(_qd)->process();
      CpvAccess(_bocInitQ)->enq(msg);
      break;
    case ROMsgMsg:
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processROMsgMsg(env);
      break;
    case RODataMsg:
      CpvAccess(_qd)->process();
      _numInitMsgs = env->getCount();
      _processRODataMsg(env);
      break;
    default:
      CmiAbort("Internal Error: Unknown-msg-type. Contact Developers.\n");
  }
  if(_numInitMsgs&&(CpvAccess(_numInitsRecd)==_numInitMsgs))
    _initDone();
}

extern "C"
void CkExit(void) 
{
  CmiNumberHandler(_charmHandlerIdx,(CmiHandler)_discardHandler);
  CmiNumberHandler(_bocHandlerIdx, (CmiHandler)_discardHandler);
  if(CkMyPe()==0) {
    if(_exitStarted)
      return;
    envelope *env = _allocEnv(ReqStatMsg);
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _exitHandlerIdx);
    CmiSyncBroadcastAllAndFree(env->getTotalsize(), env);
  } else {
    envelope *env = _allocEnv(ExitMsg);
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _exitHandlerIdx);
    CmiSyncSendAndFree(0, env->getTotalsize(), env);
  }
}

static void _nullFn(void *, void *)
{
  CmiAbort("Null-Method Called. Program may have Unregistered Module!!\n");
}

void _initCharm(int argc, char **argv)
{
  CpvInitialize(PtrQ*,_buffQ);
  CpvInitialize(PtrQ*,_bocInitQ);
  CpvInitialize(void*, _currentChare);
  CpvInitialize(int, _currentGroup);
  CpvInitialize(GroupTable*, _groupTable);
  CpvInitialize(UInt, _numInitsRecd);
  CpvInitialize(QdState*, _qd);
  CpvInitialize(MsgPool*, _msgPool);

  CpvInitialize(_CkOutStream*, _ckout);
  CpvInitialize(_CkErrStream*, _ckerr);

  CpvInitialize(Stats*, _myStats);

  CpvAccess(_buffQ) = new PtrQ();
  CpvAccess(_bocInitQ) = new PtrQ();
  CpvAccess(_groupTable) = new GroupTable();
  CpvAccess(_qd) = new QdState();
  CpvAccess(_numInitsRecd) = 0;

  CpvAccess(_ckout) = new _CkOutStream();
  CpvAccess(_ckerr) = new _CkErrStream();

  _charmHandlerIdx = CmiRegisterHandler((CmiHandler)_bufferHandler);
  _initHandlerIdx = CmiRegisterHandler((CmiHandler)_initHandler);
  _exitHandlerIdx = CmiRegisterHandler((CmiHandler)_bufferHandler);
  _bocHandlerIdx = CmiRegisterHandler((CmiHandler)_initHandler);
  _qdHandlerIdx = CmiRegisterHandler((CmiHandler)_qdHandler);
  _infoIdx = CldRegisterInfoFn(_infoFn);

#if CMK_DEBUG_MODE
  handlerArrayRegister(_charmHandlerIdx);
  handlerArrayRegister(_initHandlerIdx);
#endif

  _futuresModuleInit(); // part of futures implementation is a converse module
  if(CmiMyRank()==0) {
    argc = _parseCommandLineOpts(argc, argv);
    _registerInit();
    CkRegisterMsg("System", 0, 0, 0, sizeof(int));
    CkRegisterChare("null", 0);
    CkRegisterEp("null", _nullFn, 0, 0);
    _registerCkFutures();
    CkRegisterMainModule();
  }
  CpvAccess(_msgPool) = new MsgPool();
  CpvAccess(_myStats) = new Stats();
  CmiNodeBarrier();
  if(CmiMyPe()==0) {
    _allStats = new Stats*[CkNumPes()];
    register int i;
    for(i=0;i<_numMains;i++) {
      register int size = _chareTable[_mainTable[i]->chareIdx]->size;
      register void *obj = malloc(size);
      CpvAccess(_currentChare) = obj;
      register CkArgMsg *msg = (CkArgMsg *)CkAllocMsg(0, sizeof(CkArgMsg), 0);
      msg->argc = argc;
      msg->argv = argv;
      _entryTable[_mainTable[i]->entryIdx]->call(msg, obj);
    }
    CpvAccess(_myStats)->recordCreateChare(_numMains);
    CpvAccess(_myStats)->recordProcessChare(_numMains);
    for(i=0;i<_numReadonlyMsgs;i++) {
      register void *roMsg = (void *) *((char **)(_readonlyMsgs[i]->pMsg));
      if(roMsg==0)
        continue;
      register envelope *env = UsrToEnv(roMsg);
      env->setSrcPe(CkMyPe());
      env->setMsgtype(ROMsgMsg);
      env->setRoIdx(i);
      CmiSetHandler(env, _initHandlerIdx);
      CldEnqueue(CLD_BROADCAST, env, _infoIdx);
      CpvAccess(_qd)->create(CkNumPes()-1);
      _numInitMsgs++;
    }
    register int roSize = 0;
    for(i=0;i<_numReadonlies;i++)
      roSize += _readonlyTable[i]->size;
    register envelope *env = _allocEnv(RODataMsg, roSize);
    register char *tmp;
    for(tmp=(char *)EnvToUsr(env), i=0;i<_numReadonlies;i++) {
      memcpy(tmp, _readonlyTable[i]->ptr, _readonlyTable[i]->size);
      tmp += _readonlyTable[i]->size;
    }
    env->setCount(++_numInitMsgs);
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _initHandlerIdx);
    CmiSyncBroadcastAndFree(env->getTotalsize(), env);
    CpvAccess(_qd)->create(CkNumPes()-1);
    _initDone();
  }
}

GroupTable::GroupTable() 
{ 
  for(int i=0;i<MAXBINS;i++) 
    bins[i] = 0;
}
