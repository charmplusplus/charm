/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ck.h"
#include "trace.h"

#define  DEBUGF(x)   /* printf x */ 

UChar _defaultQueueing = CK_QUEUEING_FIFO;

UInt  _printCS = 0;
UInt  _printSS = 0;

UInt  _numGroups = 0;
UInt  _numNodeGroups = 0;
UInt  _numInitMsgs = 0;
UInt  _numInitNodeMsgs = 0;
int   _infoIdx;
int   _charmHandlerIdx;
int   _initHandlerIdx;
int   _bocHandlerIdx;
int   _nodeBocHandlerIdx;
int   _qdHandlerIdx;
int   _triggerHandlerIdx;
int   _triggersSent = 0;

CmiNodeLock _nodeLock;

CkOutStream ckout;
CkErrStream ckerr;
CkInStream  ckin;

CpvDeclare(void*,       _currentChare);
CpvDeclare(int,         _currentChareType);
CpvDeclare(CkGroupID,   _currentGroup);
CpvDeclare(CkGroupID,   _currentNodeGroup);
CpvDeclare(GroupTable, _groupTable);
GroupTable* _nodeGroupTable = 0;

CpvDeclare(Stats*, _myStats);
CpvDeclare(MsgPool*, _msgPool);

CpvDeclare(_CkOutStream*, _ckout);
CpvDeclare(_CkErrStream*, _ckerr);

CpvStaticDeclare(int,  _numInitsRecd); /* UInt changed to int */
CpvStaticDeclare(PtrQ*, _buffQ);
CpvStaticDeclare(PtrVec*, _bocInitVec);

static PtrVec* _nodeBocInitVec;

static int    _exitHandlerIdx;

static Stats** _allStats = 0;

static int   _numStatsRecd = 0;
static int    _exitStarted = 0;

#ifndef CMK_OPTIMIZE
#define _STATS_ON(x) (x) = 1
#else
#define _STATS_ON(x) \
          CmiPrintf("stats unavailable in optimized version. ignoring...\n"); 
#endif

static inline void _parseCommandLineOpts(char **argv)
{
  if (CmiGetArgFlag(argv,"+cs"))
      _STATS_ON(_printCS);
  if (CmiGetArgFlag(argv,"+ss"))
      _STATS_ON(_printSS);
  if (CmiGetArgFlag(argv,"+fifo"))
      _defaultQueueing = CK_QUEUEING_FIFO;
  if (CmiGetArgFlag(argv,"+lifo"))
      _defaultQueueing = CK_QUEUEING_LIFO; 
  if (CmiGetArgFlag(argv,"+ififo"))
      _defaultQueueing = CK_QUEUEING_IFIFO; 
  if (CmiGetArgFlag(argv,"+ilifo"))
      _defaultQueueing = CK_QUEUEING_ILIFO;
  if (CmiGetArgFlag(argv,"+bfifo"))
      _defaultQueueing = CK_QUEUEING_BFIFO; 
  if (CmiGetArgFlag(argv,"+blifo"))
      _defaultQueueing = CK_QUEUEING_BLIFO;
}

static void _bufferHandler(void *msg)
{
  CpvAccess(_buffQ)->enq(msg);
}

static void _discardHandler(envelope *env)
{
  CmiFree(env);
}

#ifndef CMK_OPTIMIZE
static inline void _printStats(void)
{
  int i;
  if(_printSS || _printCS) {
    Stats *total = new Stats();
    _MEMCHECK(total);
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
    CkPrintf("PE\tCC\tCP\tFCC\tFCP\tGC\tNGC\tGP\tNGP\tFGC\tFNGC\tFGP\tFNGP\n");
    for(i=0;i<CkNumPes();i++) {
      CkPrintf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",i,
               _allStats[i]->getCharesCreated(),
               _allStats[i]->getCharesProcessed(),
               _allStats[i]->getForCharesCreated(),
               _allStats[i]->getForCharesProcessed(),
               _allStats[i]->getGroupsCreated(),
               _allStats[i]->getNodeGroupsCreated(),
               _allStats[i]->getGroupsProcessed(),
               _allStats[i]->getNodeGroupsProcessed(),
               _allStats[i]->getGroupMsgsCreated(),
               _allStats[i]->getNodeGroupMsgsCreated(),
               _allStats[i]->getGroupMsgsProcessed(),
               _allStats[i]->getNodeGroupMsgsProcessed());
    }
  }
}
#else
static inline void _printStats(void) {}
#endif

static inline void _sendStats(void)
{
#ifndef CMK_OPTIMIZE
  envelope *env = UsrToEnv(CpvAccess(_myStats));
#else
  envelope *env = _allocEnv(StatMsg);
#endif
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _exitHandlerIdx);
  CmiSyncSendAndFree(0, env->getTotalsize(), env);
}

static void _exitHandler(envelope *env)
{
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
      CmiNumberHandler(_nodeBocHandlerIdx, (CmiHandler)_discardHandler);
      env->setMsgtype(ReqStatMsg);
      env->setSrcPe(CkMyPe());
      CmiSyncBroadcastAllAndFree(env->getTotalsize(), env);
      break;
    case ReqStatMsg:
      CmiNumberHandler(_charmHandlerIdx,(CmiHandler)_discardHandler);
      CmiNumberHandler(_bocHandlerIdx, (CmiHandler)_discardHandler);
      CmiNumberHandler(_nodeBocHandlerIdx, (CmiHandler)_discardHandler);
      CmiFree(env);
      _sendStats();
      if(CkMyPe())
        CsdExitScheduler();
      break;
    case StatMsg:
      assert(CkMyPe()==0);
#ifndef CMK_OPTIMIZE
      _allStats[env->getSrcPe()] = (Stats*) EnvToUsr(env);
#endif
      _numStatsRecd++;
      if(_numStatsRecd==CkNumPes()) {
        _printStats();
        _TRACE_END_COMPUTATION();
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
  register int i = 0;
  register int len = CpvAccess(_bocInitVec)->length();
  register void **vec = CpvAccess(_bocInitVec)->getVec();
  for(i=0; i<len; i++) {
    env = (envelope *) vec[i];
    if(env==0) continue;
    if(env->isPacked() && _msgTable[env->getMsgIdx()]->unpack) {
      _TRACE_BEGIN_UNPACK();
      env = UsrToEnv(_msgTable[env->getMsgIdx()]->unpack(EnvToUsr(env)));
      _TRACE_END_UNPACK();
    }
    _processBocInitMsg(env);
  }
  delete CpvAccess(_bocInitVec);
}

static inline void _processBufferedNodeBocInits(void)
{
  register envelope *env;
  CmiNumberHandler(_nodeBocHandlerIdx, (CmiHandler)_processHandler);
  register int i = 0;
  register int len = _nodeBocInitVec->length();
  register void **vec = _nodeBocInitVec->getVec();
  for(i=0; i<len; i++) {
    env = (envelope *) vec[i];
    if(env==0) continue;
    if(env->isPacked() && _msgTable[env->getMsgIdx()]->unpack) {
      _TRACE_BEGIN_UNPACK();
      env = UsrToEnv(_msgTable[env->getMsgIdx()]->unpack(EnvToUsr(env)));
      _TRACE_END_UNPACK();
    }
    _processNodeBocInitMsg(env);
  }
  delete _nodeBocInitVec;
}

static inline void _processBufferedMsgs(void)
{
  CmiNumberHandler(_charmHandlerIdx,(CmiHandler)_processHandler);
  envelope *env;
  while(NULL!=(env=(envelope*)CpvAccess(_buffQ)->deq())) {
    if(env->getMsgtype()==NewChareMsg || env->getMsgtype()==NewVChareMsg) {
      if(env->isForAnyPE())
        CldEnqueue(CLD_ANYWHERE, env, _infoIdx);
      else
        CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), env);
    } else {
      CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), env);
    }
  }
}

static int _charmLoadEstimator(void)
{
  return CpvAccess(_buffQ)->length();
}

static void _sendTriggers(void)
{
  int i, num, first;
  CmiLock(_nodeLock);
  if (_triggersSent == 0) 
  {
    _triggersSent++;
    num = CmiMyNodeSize();
    register envelope *env = _allocEnv(RODataMsg);
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _triggerHandlerIdx);
    first = CmiNodeFirst(CmiMyNode());
    for (i=0; i < num; i++)
      if(first+i != CkMyPe())
	CmiSyncSend(first+i, env->getTotalsize(), env);
    CmiFree(env);
  }
  CmiUnlock(_nodeLock);
}

static inline void _initDone(void)
{
  if (!_triggersSent) _sendTriggers(); 
  CmiNumberHandler(_triggerHandlerIdx, (CmiHandler)_discardHandler); 
  CmiNumberHandler(_exitHandlerIdx, (CmiHandler)_exitHandler);
  _processBufferedBocInits();
  if(CmiMyRank() == 0) {
    _processBufferedNodeBocInits();
  }
  DEBUGF(("Reached CmiNodeBarrier(), pe = %d, rank = %d\n", CmiMyPe(), CmiMyRank()));
  CmiNodeBarrier();
  DEBUGF(("Crossed CmiNodeBarrier(), pe = %d, rank = %d\n", CmiMyPe(), CmiMyRank()));
  _processBufferedMsgs();
}

static void _triggerHandler(envelope *env)
{
  if (_numInitMsgs && CpvAccess(_numInitsRecd) + _numInitNodeMsgs == _numInitMsgs)
  {
    DEBUGF(("Calling Init Done from _triggerHandler\n"));
	_initDone();
  }
  CmiFree(env);
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
  register envelope *env = (envelope *) msg;
  switch (env->getMsgtype()) {
    case BocInitMsg:
      CpvAccess(_numInitsRecd)++;
      CpvAccess(_qd)->process();
      CpvAccess(_bocInitVec)->insert(env->getGroupNum(), msg);
      break;
    case NodeBocInitMsg:
      CmiLock(_nodeLock);
      _numInitNodeMsgs++;
      _nodeBocInitVec->insert(env->getGroupNum(), msg);
      CmiUnlock(_nodeLock);
      CpvAccess(_qd)->process();
      break;
    case ROMsgMsg:
      CpvAccess(_numInitsRecd)++;
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processROMsgMsg(env);
      break;
    case RODataMsg:
      CpvAccess(_numInitsRecd)+=2;  /*++;*/
      CpvAccess(_qd)->process();
      _numInitMsgs = env->getCount();
      _processRODataMsg(env);
      break;
    default:
      CmiAbort("Internal Error: Unknown-msg-type. Contact Developers.\n");
  }
  if(_numInitMsgs&&(CpvAccess(_numInitsRecd)+_numInitNodeMsgs==_numInitMsgs)) {
    _initDone();
  }
}

extern "C"
void CkExit(void) 
{
  CmiNumberHandler(_charmHandlerIdx,(CmiHandler)_discardHandler);
  CmiNumberHandler(_bocHandlerIdx, (CmiHandler)_discardHandler);
  CmiNumberHandler(_nodeBocHandlerIdx, (CmiHandler)_discardHandler);
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

#if CMK_DEBUG_MODE
int getCharmMsgHandlers(int *handleArray)
{
  *(handleArray) = _charmHandlerIdx;
  *(handleArray+1) = _initHandlerIdx;
  return(2);
}

char* getEnvInfo(envelope *env)
{
  char *returnInfo;
  int size;
  int chareIndex;
  int epIndex = env->getEpIdx();
  size = strlen(_entryTable[epIndex]->name)+1;
  chareIndex = _entryTable[epIndex]->chareIdx;
  size += strlen(_chareTable[chareIndex]->name)+1;
  
  returnInfo = (char *)malloc((size + 2) * sizeof(char));
  _MEMCHECK(returnInfo);
  strcpy(returnInfo, _entryTable[epIndex]->name);
  strcat(returnInfo, "%");
  strcat(returnInfo, _chareTable[chareIndex]->name);
  strcat(returnInfo, "#");
  return(returnInfo);
}

char* makeCharmSymbolTableInfo(void)
{
  int i, chareIndex;
  int size;
  char *returnInfo;
   
  size = _numEntries * 100;
  returnInfo = (char *)malloc(size * sizeof(char));
  _MEMCHECK(returnInfo);
  strcpy(returnInfo, "");
  for(i = 0; i < _numEntries; i++){
    strcat(returnInfo, "EP : ");
    strcat(returnInfo, _entryTable[i]->name);
    strcat(returnInfo, " ");
    strcat(returnInfo, "ChareName : ");
    chareIndex = _entryTable[i]->chareIdx;
    strcat(returnInfo, _chareTable[chareIndex]->name);
    strcat(returnInfo, "#");
  }

  return(returnInfo);
}

int getEpIdx(char *msg)
{
  envelope *env;
  
  env = (envelope *)msg;
  return(env->getEpIdx());
}

static char* fHeader(char* msg)
{
  return(getEnvInfo((envelope *)msg));
}

static const char *_contentStr = 
"Contents not known in this implementation"
;

static char* fContent(char *msg)
{
  char *temp;

  temp = (char *)malloc(strlen(_contentStr) + 1);
  _MEMCHECK(temp);
  strcpy(temp, _contentStr);
  return(temp);
}
#endif

extern void _registerLBDatabase(void);
extern void _ckModuleInit(void);

void _initCharm(int argc, char **argv)
{
	CpvInitialize(PtrQ*,_buffQ);
	CpvInitialize(PtrVec*,_bocInitVec);
	CpvInitialize(void*, _currentChare);
	CpvInitialize(int,   _currentChareType);
	CpvInitialize(CkGroupID, _currentGroup);
	CpvInitialize(CkGroupID, _currentNodeGroup);
	CpvInitialize(GroupTable, _groupTable);
	CpvInitialize(int, _numInitsRecd);
	CpvInitialize(QdState*, _qd);
	CpvInitialize(MsgPool*, _msgPool);

	CpvInitialize(_CkOutStream*, _ckout);
	CpvInitialize(_CkErrStream*, _ckerr);

	CpvInitialize(Stats*, _myStats);
	
	CpvAccess(_groupTable).init();
	CpvAccess(_buffQ) = new PtrQ();
	_MEMCHECK(CpvAccess(_buffQ));
	CpvAccess(_bocInitVec) = new PtrVec();
	_MEMCHECK(CpvAccess(_bocInitVec));
	
	if(CmiMyRank()==0) 
	{
		_nodeLock = CmiCreateLock();
		_nodeGroupTable = new GroupTable();
		_MEMCHECK(_nodeGroupTable);
		_nodeBocInitVec = new PtrVec();
		_MEMCHECK(_nodeBocInitVec);
	}
  
	CmiNodeBarrier();
	CpvAccess(_qd) = new QdState();
	_MEMCHECK(CpvAccess(_qd));
	CpvAccess(_numInitsRecd) = -1;  /*0;*/

	CpvAccess(_ckout) = new _CkOutStream();
	_MEMCHECK(CpvAccess(_ckout));
	CpvAccess(_ckerr) = new _CkErrStream();
	_MEMCHECK(CpvAccess(_ckerr));

	_charmHandlerIdx = CmiRegisterHandler((CmiHandler)_bufferHandler);
	_initHandlerIdx = CmiRegisterHandler((CmiHandler)_initHandler);
	_exitHandlerIdx = CmiRegisterHandler((CmiHandler)_bufferHandler);
	_bocHandlerIdx = CmiRegisterHandler((CmiHandler)_initHandler);
	_nodeBocHandlerIdx = CmiRegisterHandler((CmiHandler)_initHandler);
	_qdHandlerIdx = CmiRegisterHandler((CmiHandler)_qdHandler);
	_infoIdx = CldRegisterInfoFn((CldInfoFn)_infoFn);
	_triggerHandlerIdx = CmiRegisterHandler((CmiHandler)_triggerHandler);
	_ckModuleInit();

	CthSetSuspendable(CthSelf(), 0);

	CldRegisterEstimator((CldEstimator)_charmLoadEstimator);

#if CMK_DEBUG_MODE
	handlerArrayRegister(_charmHandlerIdx, (hndlrIDFunction)fHeader, 
		                   (hndlrIDFunction)fContent);
	handlerArrayRegister(_initHandlerIdx, (hndlrIDFunction)fHeader, 
		                   (hndlrIDFunction)fContent);
#endif

	_futuresModuleInit(); // part of futures implementation is a converse module
	if(CmiMyRank()==0) 
	{
		_parseCommandLineOpts(argv);
		_registerInit();
		CkRegisterMsg("System", 0, 0, 0, sizeof(int));
		CkRegisterChare("null", 0);
		CkRegisterEp("null", (CkCallFnPtr)_nullFn, 0, 0);
		_registerCkFutures();
		_registerCkArray();
		_registertempo();
		_registerwaitqd();
		_registerLBDatabase();
		CkRegisterMainModule();
	}

	_TRACE_BEGIN_COMPUTATION();
	CpvAccess(_myStats) = new Stats();
	_MEMCHECK(CpvAccess(_myStats));
	CpvAccess(_msgPool) = new MsgPool();
	_MEMCHECK(CpvAccess(_msgPool));
	CmiNodeBarrier();

	if(CmiMyPe()==0) 
	{
		_allStats = new Stats*[CkNumPes()];
		_MEMCHECK(_allStats);
		register int i;
		for(i=0;i<_numMains;i++) 
		{
			register int size = _chareTable[_mainTable[i]->chareIdx]->size;
			register void *obj = malloc(size);
			_MEMCHECK(obj);
			CpvAccess(_currentChare) = obj;
			CpvAccess(_currentChareType) = _mainTable[i]->chareIdx;
			register CkArgMsg *msg = (CkArgMsg *)CkAllocMsg(0, sizeof(CkArgMsg), 0);
			msg->argc = CmiGetArgc(argv);
			msg->argv = argv;
			_entryTable[_mainTable[i]->entryIdx]->call(msg, obj);
		}

		_STATS_RECORD_CREATE_CHARE_N(_numMains);
		_STATS_RECORD_PROCESS_CHARE_N(_numMains);
		for(i=0;i<_numReadonlyMsgs;i++) 
		{
			register void *roMsg = (void *) *((char **)(_readonlyMsgs[i]->pMsg));
			if(roMsg==0)
				continue;
			register envelope *env = UsrToEnv(roMsg);
			register int msgIdx = env->getMsgIdx();
			env->setSrcPe(CkMyPe());
			env->setMsgtype(ROMsgMsg);
			env->setRoIdx(i);
			CmiSetHandler(env, _initHandlerIdx);
			if (!env->isPacked() &&  _msgTable[msgIdx]->pack)
				_packFn((void **) &env);
			CmiSyncBroadcast(env->getTotalsize(), env);
			if (env->isPacked() && _msgTable[msgIdx]->unpack)
				_unpackFn((void **) &env);
			CpvAccess(_qd)->create(CkNumPes()-1);
			_numInitMsgs++;
		}
		register int roSize = 0;
		for(i=0;i<_numReadonlies;i++)
			roSize += _readonlyTable[i]->size;
		register envelope *env = _allocEnv(RODataMsg, roSize);
		register char *tmp;
		for(tmp=(char *)EnvToUsr(env), i=0;i<_numReadonlies;i++) 
		{
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

#if CMK_DEBUG_MODE
	symbolTableFnArrayRegister(_charmHandlerIdx, _numEntries,
				     (symbolTableFunction) makeCharmSymbolTableInfo,
				     (indirectionFunction) getEpIdx);
#endif

}

GroupTable::GroupTable() { }

void GroupTable::enqmsg(CkGroupID n, void *msg)
{
	if (tab[n].pending==NULL)
		tab[n].pending=new PtrQ();
	tab[n].pending->enq(msg);
}

void GroupTable::add(CkGroupID n, void *obj)
{
	tab[n].obj=obj;
	if (tab[n].pending==NULL)
		tab[n].pending=new PtrQ();
}



// this is needed because on o2k, f90 programs have to have main in
// fortran90.
extern "C" void fmain_(int *argc,char _argv[][80],int length[])
{
  int i;
  char **argv = new char*[*argc+2];

  for(i=0;i <= *argc;i++) {
    if (length[i] < 100) {
      _argv[i][length[i]]='\0';
      argv[i] = &(_argv[i][0]);
    } else {
      argv[i][0] = '\0';
    }
  }
  argv[*argc+1]=0;

  ConverseInit(*argc, argv, (CmiStartFn) _initCharm, 0, 0);
}

