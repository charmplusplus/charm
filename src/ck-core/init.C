/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ck.h"
#include "trace.h"

#define  DEBUGF(x)    // CmiPrintf x;

UChar _defaultQueueing = CK_QUEUEING_FIFO;

UInt  _printCS = 0;
UInt  _printSS = 0;

UInt  _numInitMsgs = 0;
UInt  _numExpectInitMsgs = 0;
UInt  _numInitNodeMsgs = 0;
int   _infoIdx;
int   _charmHandlerIdx;
int   _initHandlerIdx;
int   _bocHandlerIdx;
int   _nodeBocHandlerIdx;
int   _qdHandlerIdx;
int   _triggerHandlerIdx;
int   _triggersSent = 0;
int   _mainDone = 0;

CmiNodeLock _nodeLock;

CkOutStream ckout;
CkErrStream ckerr;
CkInStream  ckin;

CkpvDeclare(void*,       _currentChare);
CkpvDeclare(int,         _currentChareType);
CkpvDeclare(CkGroupID,   _currentGroup);
CkpvDeclare(CkGroupID,   _currentNodeGroup);
CkpvDeclare(GroupTable, _groupTable);
CkpvDeclare(UInt, _numGroups);
UInt _numNodeGroups;
GroupTable* _nodeGroupTable = 0;

CkpvDeclare(Stats*, _myStats);
CkpvDeclare(MsgPool*, _msgPool);

CkpvDeclare(_CkOutStream*, _ckout);
CkpvDeclare(_CkErrStream*, _ckerr);

CkpvStaticDeclare(int,  _numInitsRecd); /* UInt changed to int */
CkpvStaticDeclare(PtrQ*, _buffQ);
CkpvStaticDeclare(PtrVec*, _bocInitVec);

static PtrVec* _nodeBocInitVec;

static int    _exitHandlerIdx;

static Stats** _allStats = 0;

static int   _numStatsRecd = 0;
static int   _exitStarted = 0;

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
  DEBUGF(("[%d] _bufferHandler called.\n", CkMyPe()));
  CkpvAccess(_buffQ)->enq(msg);
}

static void _discardHandler(envelope *env)
{
  DEBUGF(("[%d] _discardHandler called.\n", CkMyPe()));
  CmiFree(env);
}

#ifndef CMK_OPTIMIZE
static inline void _printStats(void)
{
  DEBUGF(("[%d] _printStats\n", CkMyPe()));
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
    CkPrintf("Charm Kernel Detailed Statistics (R=requested P=processed):\n\n");

    CkPrintf("         Create    Mesgs     Create    Mesgs     Create    Mesgs\n");
    CkPrintf("         Chare     for       Group     for       Nodegroup for\n");
    CkPrintf("PE   R/P Mesgs     Chares    Mesgs     Groups    Mesgs     Nodegroups\n");
    CkPrintf("---- --- --------- --------- --------- --------- --------- ----------\n");

    for(i=0;i<CkNumPes();i++) {
      CkPrintf("%4d  R  %9d %9d %9d %9d %9d %9d\n      P  %9d %9d %9d %9d %9d %9d\n",i,
               _allStats[i]->getCharesCreated(),
               _allStats[i]->getForCharesCreated(),
               _allStats[i]->getGroupsCreated(),
               _allStats[i]->getGroupMsgsCreated(),
               _allStats[i]->getNodeGroupsCreated(),
               _allStats[i]->getNodeGroupMsgsCreated(),
               _allStats[i]->getCharesProcessed(),
               _allStats[i]->getForCharesProcessed(),
               _allStats[i]->getGroupsProcessed(),
               _allStats[i]->getGroupMsgsProcessed(), 
               _allStats[i]->getNodeGroupsProcessed(),
	       _allStats[i]->getNodeGroupMsgsProcessed());
    }
  }
}
#else
static inline void _printStats(void) {}
#endif

static inline void _sendStats(void)
{
  DEBUGF(("[%d] _sendStats\n", CkMyPe()));
#ifndef CMK_OPTIMIZE
  envelope *env = UsrToEnv(CkpvAccess(_myStats));
#else
  envelope *env = _allocEnv(StatMsg);
#endif
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _exitHandlerIdx);
  CmiSyncSendAndFree(0, env->getTotalsize(), (char *)env);
}

static void _exitHandler(envelope *env)
{
  DEBUGF(("exitHandler called on %d msgtype: %d\n", CkMyPe(), env->getMsgtype()));
  switch(env->getMsgtype()) {
    case ExitMsg:
      CkAssert(CkMyPe()==0);
      if(_exitStarted) {
        CmiFree(env);
        return;
      }
      _exitStarted = 1; 
      CkNumberHandler(_charmHandlerIdx,(CmiHandler)_discardHandler);
      CkNumberHandler(_bocHandlerIdx, (CmiHandler)_discardHandler);
      CkNumberHandler(_nodeBocHandlerIdx, (CmiHandler)_discardHandler);
      env->setMsgtype(ReqStatMsg);
      env->setSrcPe(CkMyPe());
      CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char *)env);
      break;
    case ReqStatMsg:
      DEBUGF(("ReqStatMsg on %d\n", CkMyPe()));
      CkNumberHandler(_charmHandlerIdx,(CmiHandler)_discardHandler);
      CkNumberHandler(_bocHandlerIdx, (CmiHandler)_discardHandler);
      CkNumberHandler(_nodeBocHandlerIdx, (CmiHandler)_discardHandler);
      CmiFree(env);
      _sendStats();
      _mainDone = 1; // This is needed because the destructors for
                     // readonly variables will be called when the program
		     // exits. If the destructor is called while _mainDone
		     // is 0, it will assume that the readonly variable was
		     // declared locally. On all processors other than 0, 
		     // _mainDone is never set to 1 before the program exits.
      if(CkMyPe())
        ConverseExit();
      break;
    case StatMsg:
      CkAssert(CkMyPe()==0);
#ifndef CMK_OPTIMIZE
      _allStats[env->getSrcPe()] = (Stats*) EnvToUsr(env);
#endif
      _numStatsRecd++;
      DEBUGF(("StatMsg on %d with %d\n", CkMyPe(), _numStatsRecd));
      if(_numStatsRecd==CkNumPes()) {
        _printStats();
        ConverseExit();
      }
      break;
    default:
      CmiAbort("Internal Error(_exitHandler): Unknown-msg-type. Contact Developers.\n");
  }
}

static inline void _processBufferedBocInits(void)
{
  register envelope *env;
  CkNumberHandler(_bocHandlerIdx, (CmiHandler)_processHandler);
  register int i = 0;
  register int len = CkpvAccess(_bocInitVec)->length();
  register void **vec = CkpvAccess(_bocInitVec)->getVec();
  for(i=0; i<len; i++) {
    env = (envelope *) vec[i];
    if(env==0) continue;
    if(env->isPacked()) 
      CkUnpackMessage(&env);
    _processBocInitMsg(env);
  }
  delete CkpvAccess(_bocInitVec);
}

static inline void _processBufferedNodeBocInits(void)
{
  register envelope *env;
  CkNumberHandler(_nodeBocHandlerIdx, (CmiHandler)_processHandler);
  register int i = 0;
  register int len = _nodeBocInitVec->length();
  register void **vec = _nodeBocInitVec->getVec();
  for(i=0; i<len; i++) {
    env = (envelope *) vec[i];
    if(env==0) continue;
    if(env->isPacked())
      CkUnpackMessage(&env);
    _processNodeBocInitMsg(env);
  }
  delete _nodeBocInitVec;
}

static inline void _processBufferedMsgs(void)
{
  CkNumberHandler(_charmHandlerIdx,(CmiHandler)_processHandler);
  envelope *env;
  while(NULL!=(env=(envelope*)CkpvAccess(_buffQ)->deq())) {
    if(env->getMsgtype()==NewChareMsg || env->getMsgtype()==NewVChareMsg) {
      if(env->isForAnyPE())
        CldEnqueue(CLD_ANYWHERE, env, _infoIdx);
      else
        CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), (char *)env);
    } else {
      CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), (char *)env);
    }
  }
}

static int _charmLoadEstimator(void)
{
  return CkpvAccess(_buffQ)->length();
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
	CmiSyncSend(first+i, env->getTotalsize(), (char *)env);
    CmiFree(env);
  }
  CmiUnlock(_nodeLock);
}

static inline void _initDone(void)
{
  DEBUGF(("[%d] _initDone.\n", CkMyPe()));
  if (!_triggersSent) _sendTriggers(); 
  CkNumberHandler(_triggerHandlerIdx, (CmiHandler)_discardHandler); 
  if(CkMyRank() == 0) {
    _processBufferedNodeBocInits();
  }
  CmiNodeBarrier(); // wait for all nodegroups to be created
  _processBufferedBocInits();
  DEBUGF(("Reached CmiNodeBarrier(), pe = %d, rank = %d\n", CkMyPe(), CkMyRank()));
  CmiNodeBarrier();
  DEBUGF(("Crossed CmiNodeBarrier(), pe = %d, rank = %d\n", CkMyPe(), CkMyRank()));
  _processBufferedMsgs();
}

static void _triggerHandler(envelope *env)
{
  if (_numExpectInitMsgs && CkpvAccess(_numInitsRecd) + _numInitNodeMsgs == _numExpectInitMsgs)
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
  CkAssert(CkMyPe()!=0);
  register envelope *env = (envelope *) msg;
  switch (env->getMsgtype()) {
    case BocInitMsg:
      CkpvAccess(_numInitsRecd)++;
      CpvAccess(_qd)->process();
      CkpvAccess(_bocInitVec)->insert(env->getGroupNum().idx, msg);
      break;
    case NodeBocInitMsg:
      CmiLock(_nodeLock);
      _numInitNodeMsgs++;
      _nodeBocInitVec->insert(env->getGroupNum().idx, msg);
      CmiUnlock(_nodeLock);
      CpvAccess(_qd)->process();
      break;
    case ROMsgMsg:
      CkpvAccess(_numInitsRecd)++;
      CpvAccess(_qd)->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processROMsgMsg(env);
      break;
    case RODataMsg:
      CkpvAccess(_numInitsRecd)+=2;  /*++;*/
      CpvAccess(_qd)->process();
      _numExpectInitMsgs = env->getCount();
      _processRODataMsg(env);
      break;
    default:
      CmiAbort("Internal Error: Unknown-msg-type. Contact Developers.\n");
  }
  if(_numExpectInitMsgs&&(CkpvAccess(_numInitsRecd)+_numInitNodeMsgs==_numExpectInitMsgs)) {
    _initDone();
  }
}

// CkExit: start the termination process, but
//   then drop into the scheduler so the user's
//   method never returns (which would be confusing).
extern "C"
void CkExit(void) 
{
  CkNumberHandler(_charmHandlerIdx,(CmiHandler)_discardHandler);
  CkNumberHandler(_bocHandlerIdx, (CmiHandler)_discardHandler);
  CkNumberHandler(_nodeBocHandlerIdx, (CmiHandler)_discardHandler);
  DEBUGF(("[%d] CkExit - _exitStarted:%d %d\n", CkMyPe(), _exitStarted, _exitHandlerIdx));
  if(CkMyPe()==0) {
    if(_exitStarted)
      CsdScheduler(-1);
    envelope *env = _allocEnv(ReqStatMsg);
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _exitHandlerIdx);
    CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char *)env);
  } else {
    envelope *env = _allocEnv(ExitMsg);
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _exitHandlerIdx);
    CmiSyncSendAndFree(0, env->getTotalsize(), (char *)env);
  }
#if ! CMK_BLUEGENE_THREAD
  _TRACE_END_EXECUTE();
  //Wait for stats, which will call ConverseExit when finished:
  CsdScheduler(-1);
#endif
}

static void _nullFn(void *, void *)
{
  CmiAbort("Null-Method Called. Program may have Unregistered Module!!\n");
}

extern void _registerLBDatabase(void);
extern void _registerExternalModules(char **argv);
extern void _ckModuleInit(void);
extern void _loadbalancerInit();

void _initCharm(int argc, char **argv)
{

	CkpvInitialize(PtrQ*,_buffQ);
	CkpvInitialize(PtrVec*,_bocInitVec);
	CkpvInitialize(void*, _currentChare);
	CkpvInitialize(int,   _currentChareType);
	CkpvInitialize(CkGroupID, _currentGroup);
	CkpvInitialize(CkGroupID, _currentNodeGroup);
	CkpvInitialize(GroupTable, _groupTable);
	CkpvInitialize(UInt, _numGroups);
//	CkpvInitialize(UInt, _numNodeGroups);
	CkpvInitialize(int, _numInitsRecd);
	CpvInitialize(QdState*, _qd);
	CkpvInitialize(MsgPool*, _msgPool);

	CkpvInitialize(_CkOutStream*, _ckout);
	CkpvInitialize(_CkErrStream*, _ckerr);

	CkpvInitialize(Stats*, _myStats);

	CkpvAccess(_groupTable).init();
	CkpvAccess(_numGroups) = 1; // make 0 an invalid group number
	_numNodeGroups = 1;
	CkpvAccess(_buffQ) = new PtrQ();
	_MEMCHECK(CkpvAccess(_buffQ));
	CkpvAccess(_bocInitVec) = new PtrVec();
	_MEMCHECK(CkpvAccess(_bocInitVec));
	
	if(CkMyRank()==0) 
	{
		_nodeLock = CmiCreateLock();
		_nodeGroupTable = new GroupTable();
		_MEMCHECK(_nodeGroupTable);
		_nodeGroupTable->init();
		_nodeBocInitVec = new PtrVec();
		_MEMCHECK(_nodeBocInitVec);
	}
  
	CmiNodeBarrier();
#ifdef __BLUEGENE__
	if(CkMyRank()==0) 
#endif
	{
	CpvAccess(_qd) = new QdState();
	_MEMCHECK(CpvAccess(_qd));
        }
	CkpvAccess(_numInitsRecd) = -1;  /*0;*/

	CkpvAccess(_ckout) = new _CkOutStream();
	_MEMCHECK(CkpvAccess(_ckout));
	CkpvAccess(_ckerr) = new _CkErrStream();
	_MEMCHECK(CkpvAccess(_ckerr));

	_charmHandlerIdx = CkRegisterHandler((CmiHandler)_bufferHandler);
	_initHandlerIdx = CkRegisterHandler((CmiHandler)_initHandler);
	_exitHandlerIdx = CkRegisterHandler((CmiHandler)_exitHandler);
	_bocHandlerIdx = CkRegisterHandler((CmiHandler)_initHandler);
	_nodeBocHandlerIdx = CkRegisterHandler((CmiHandler)_initHandler);
#ifdef __BLUEGENE__
	if(CkMyRank()==0) 
#endif
	{
	_qdHandlerIdx = CmiRegisterHandler((CmiHandler)_qdHandler);
        }
	_infoIdx = CldRegisterInfoFn((CldInfoFn)_infoFn);
	_triggerHandlerIdx = CkRegisterHandler((CmiHandler)_triggerHandler);
	_ckModuleInit();

	CthSetSuspendable(CthSelf(), 0);

	CldRegisterEstimator((CldEstimator)_charmLoadEstimator);

	_futuresModuleInit(); // part of futures implementation is a converse module
	_loadbalancerInit();
	if(CkMyRank()==0) 
	{
		_parseCommandLineOpts(argv);
		_registerInit();
		CkRegisterMsg("System", 0, 0, 0, sizeof(int));
		CkRegisterChare("null", 0);
		CkIndex_Chare::__idx=CkRegisterChare("Chare", sizeof(Chare));
		CkIndex_Group::__idx=CkRegisterChare("Group", sizeof(Group));
		CkRegisterEp("null", (CkCallFnPtr)_nullFn, 0, 0);
		_registerCkFutures();
		_registerCkArray();
		_registerCkCallback();
		_registertempo();
		_registerwaitqd();
		_registerLBDatabase();
		_registercharisma();
		_registerExternalModules(argv);
		CkRegisterMainModule();
	}

#if CMK_TRACE_IN_CHARM
        // initialize trace module in ck
        traceCharmInit(argv);
#endif
	_TRACE_BEGIN_COMPUTATION();
	CkpvAccess(_myStats) = new Stats();
	_MEMCHECK(CkpvAccess(_myStats));
	CkpvAccess(_msgPool) = new MsgPool();
	_MEMCHECK(CkpvAccess(_msgPool));
	CmiNodeBarrier();

	if(CkMyPe()==0) 
	{
		_allStats = new Stats*[CkNumPes()];
		_MEMCHECK(_allStats);
		register int i;
		for(i=0;i<_numMains;i++) 
		{
			register int size = _chareTable[_mainTable[i]->chareIdx]->size;
			register void *obj = malloc(size);
			_MEMCHECK(obj);
			CkpvAccess(_currentChare) = obj;
			CkpvAccess(_currentChareType) = _mainTable[i]->chareIdx;
			register CkArgMsg *msg = (CkArgMsg *)CkAllocMsg(0, sizeof(CkArgMsg), 0);
			msg->argc = CmiGetArgc(argv);
			msg->argv = argv;
			_entryTable[_mainTable[i]->entryIdx]->call(msg, obj);
		}
                _mainDone = 1;

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
				CkPackMessage(&env);
			CmiSyncBroadcast(env->getTotalsize(), (char *)env);
			if (env->isPacked() && _msgTable[msgIdx]->unpack)
				CkUnpackMessage(&env);
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
		CmiSyncBroadcastAndFree(env->getTotalsize(), (char *)env);
		CpvAccess(_qd)->create(CkNumPes()-1);
		_initDone();
	}

}

#ifdef __BLUEGENE__

#if  CMK_BLUEGENE_THREAD
void BgEmulatorInit(int argc, char **argv) 
{
  BgSetWorkerThreadStart(_initCharm);
}
void BgNodeStart(int argc, char **argv) {}
#else
void BgEmulatorInit(int argc, char **argv) {}
void BgNodeStart(int argc, char **argv)
{
  _initCharm(argc, argv);
}
#endif
#endif

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

