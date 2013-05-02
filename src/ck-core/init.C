/**
\addtogroup CkInit
\brief Controls the Charm++ startup process.

This file runs the entire Charm++ startup process.

The process begins with every processor finishing the 
Converse startup process and calling _initCharm.
This routine runs almost the entire Charm++ setup process.
It begins by setting up various Cpvs and subsystems.

The rank 0 processor of each node then does
the Charm++ registration, by calling the various _register
routines.  

Now processor 0:
<ol>
<li>Creates each mainchare, by allocating the chares
 and calling their constructors with argc/argv.
 This typically results in a number of chare/group creations.
<li>Sends off all readonly data to the other processors.
</ol>
After _initCharm, processor 0 immediately begins work.

The other processors, however, must wait until they recieve 
the readonly data and all group creations.  They do this by 
setting the _charmHandlerIdx to a special "_bufferHandler"
which simply saves all normal messages into a special queue.  

As the startup data (readonlies and group creations) streams
into _initHandler, it counts messages until it is fully 
initialized, then calls _initDone to clean out the queues 
and resume normal operation.  

Upon resume of normal operation, the user code is guaranteed that
all readonlies (both data and messages) have been set consistently
on all processors, and that the constructors for all nodegroups
and groups allocated from a mainchare have been called.

It is not guaranteed the order in which (node)groups allocated
outside of a mainchare are constructed, nor that the construction
will happen before other messages have been delivered by the scheduler.

Even though not exposed to the final users, the creation order of
groups and nodegroups allocated in mainchares is deterministic and
respects the following points:
<ul>
<li>On all processors, there is no guarantee of the order of creation
between (node)groups allocated from different mainchares;
<li>On processor zero, within a mainchare, all (node)groups are created
in the order specified by the source code (strictly), including array
allocation of initial elements;
<li>On processors other than zero, within a mainchare, the order
specified by the source code is maintained between different nodegroups
and between different groups;
<li>On processors other than zero, the ordering between groups and
nodegroups is NOT maintained, as all nodegroups are created before any
group is created.
</ul>

This process should not have race conditions, but it can
never be excluded...
*/
/*@{*/

#include "ckcheckpoint.h"
#include "ck.h"
#include "trace.h"
#include "CkCheckpoint.decl.h"
#include <sstream>
#include <limits.h>

void CkRestartMain(const char* dirname, CkArgMsg* args);

#define  DEBUGF(x)     //CmiPrintf x;

#include "TopoManager.h"

UChar _defaultQueueing = CK_QUEUEING_FIFO;

UInt  _printCS = 0;
UInt  _printSS = 0;

/**
 * This value has the number of total initialization message a processor awaits.
 * It is received on nodes other than zero together with the ROData message.
 * Even though it is shared by all processors it is ok: it doesn't matter when and
 * by who it is set, provided that it becomes equal to the number of awaited messages
 * (which is always at least one ---the readonly data message).
 */
UInt  _numExpectInitMsgs = 0;
/**
 * This number is used only by processor zero to count how many messages it will
 * send out for the initialization process. After the readonly data message is sent
 * (containing this counter), its value becomes irrelevant.
 */
UInt  _numInitMsgs = 0;
/**
 * Count the number of nodegroups that have been created in mainchares.
 * Since the nodegroup creation is executed by a single processor in a
 * given node, this value must be seen by all processors in a node.
 */
CksvDeclare(UInt,_numInitNodeMsgs);
int   _infoIdx;
int   _charmHandlerIdx;
int   _initHandlerIdx;
int   _roRestartHandlerIdx;
int   _bocHandlerIdx;
int   _qdHandlerIdx;
int   _qdCommHandlerIdx;
int   _triggerHandlerIdx;
int   _mainDone = 0;
CksvDeclare(int, _triggersSent);

CkOutStream ckout;
CkErrStream ckerr;
CkInStream  ckin;

CkpvDeclare(void*,       _currentChare);
CkpvDeclare(int,         _currentChareType);
CkpvDeclare(CkGroupID,   _currentGroup);
CkpvDeclare(void*,       _currentNodeGroupObj);
CkpvDeclare(CkGroupID,   _currentGroupRednMgr);
CkpvDeclare(GroupTable*, _groupTable);
CkpvDeclare(GroupIDTable*, _groupIDTable);
CkpvDeclare(CmiImmediateLockType, _groupTableImmLock);
CkpvDeclare(UInt, _numGroups);

CkpvDeclare(CkCoreState *, _coreState);

CksvDeclare(UInt, _numNodeGroups);
CksvDeclare(GroupTable*, _nodeGroupTable);
CksvDeclare(GroupIDTable, _nodeGroupIDTable);
CksvDeclare(CmiImmediateLockType, _nodeGroupTableImmLock);
CksvDeclare(CmiNodeLock, _nodeLock);
CksvStaticDeclare(PtrVec*,_nodeBocInitVec);
CkpvDeclare(int, _charmEpoch);

CkpvDeclare(bool, _destroyingNodeGroup);


CkpvDeclare(Stats*, _myStats);
CkpvDeclare(MsgPool*, _msgPool);

CkpvDeclare(_CkOutStream*, _ckout);
CkpvDeclare(_CkErrStream*, _ckerr);

CkpvStaticDeclare(int,  _numInitsRecd);
CkpvStaticDeclare(int,  _initdone);
CkpvStaticDeclare(PtrQ*, _buffQ);
CkpvStaticDeclare(PtrVec*, _bocInitVec);

//for interoperability
extern void _libExitHandler(envelope *env);
extern int _libExitHandlerIdx;
CpvCExtern(int,interopExitFlag);

#if CMK_SHRINK_EXPAND
//for shrink expand cleanup
int _ROGroupRestartHandlerIdx;
const char* _shrinkexpand_basedir;
#endif
/*
	FAULT_EVAC
*/
CpvCExtern(char *, _validProcessors);
CkpvDeclare(char ,startedEvac);

int    _exitHandlerIdx;

#if CMK_WITH_STATS
static Stats** _allStats = 0;
#endif
static int   _exitStarted = 0;

static InitCallTable _initCallTable;

#if CMK_WITH_STATS
#define _STATS_ON(x) (x) = 1
#else
#define _STATS_ON(x) \
          if (CkMyPe()==0) CmiPrintf("stats unavailable in optimized version. ignoring...\n");
#endif

// fault tolerance
typedef void (*CkFtFn)(const char *, CkArgMsg *);
static CkFtFn  faultFunc = NULL;
static char* _restartDir;

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
int teamSize=1;
int chkptPeriod=1000;
bool fastRecovery = false;
int parallelRecovery = 1;
extern int BUFFER_TIME; //time spent waiting for buffered messages
#endif

// flag for killing processes 
extern int killFlag;
// file specifying the processes to be killed
extern char *killFile;
// function for reading the kill file
void readKillFile();
#if CMK_MESSAGE_LOGGING
// flag for disk checkpoint
extern int diskCkptFlag;
#endif

int _defaultObjectQ = 0;            // for obejct queue
int _ringexit = 0;		    // for charm exit
int _ringtoken = 8;
extern int _messageBufferingThreshold;


/*
	FAULT_EVAC

	flag which marks whether or not to trigger the processor shutdowns
*/
static int _raiseEvac=0;
static char *_raiseEvacFile;
void processRaiseEvacFile(char *raiseEvacFile);

extern bool useNodeBlkMapping;

extern "C" int quietMode;
extern "C" int quietModeRequested;

// Modules are required to register command line opts they will parse. These
// options are stored in the _optSet, and then when parsing command line opts
// users will be warned about options starting with a '+' that are not in this
// table. This usually implies that they are attempting to use a Charm++ option
// without having compiled Charm++ to use the module that options belongs to.
std::set<std::string> _optSet;
void _registerCommandLineOpt(const char* opt) {
  // The command line options are only checked during init on PE0, so this makes
  // thread safety easy.
  if (CkMyPe() == 0) {
    _optSet.insert(opt);
  }
}

static inline void _parseCommandLineOpts(char **argv)
{
  if (CmiGetArgFlagDesc(argv,"+cs", "Print extensive statistics at shutdown"))
      _STATS_ON(_printCS);
  if (CmiGetArgFlagDesc(argv,"+ss", "Print summary statistics at shutdown"))
      _STATS_ON(_printSS);
  if (CmiGetArgFlagDesc(argv,"+fifo", "Default to FIFO queuing"))
      _defaultQueueing = CK_QUEUEING_FIFO;
  if (CmiGetArgFlagDesc(argv,"+lifo", "Default to LIFO queuing"))
      _defaultQueueing = CK_QUEUEING_LIFO;
  if (CmiGetArgFlagDesc(argv,"+ififo", "Default to integer-prioritized FIFO queuing"))
      _defaultQueueing = CK_QUEUEING_IFIFO;
  if (CmiGetArgFlagDesc(argv,"+ilifo", "Default to integer-prioritized LIFO queuing"))
      _defaultQueueing = CK_QUEUEING_ILIFO;
  if (CmiGetArgFlagDesc(argv,"+bfifo", "Default to bitvector-prioritized FIFO queuing"))
      _defaultQueueing = CK_QUEUEING_BFIFO;
  if (CmiGetArgFlagDesc(argv,"+blifo", "Default to bitvector-prioritized LIFO queuing"))
      _defaultQueueing = CK_QUEUEING_BLIFO;
  if (CmiGetArgFlagDesc(argv,"+objq", "Default to use object queue for every obejct"))
  {
#if CMK_OBJECT_QUEUE_AVAILABLE
      _defaultObjectQ = 1;
      if (CkMyPe()==0)
        CmiPrintf("Charm++> Create object queue for every Charm object.\n");
#else
      CmiAbort("Charm++> Object queue not enabled, recompile Charm++ with CMK_OBJECT_QUEUE_AVAILABLE defined to 1.");
#endif
  }

#if CMK_SHRINK_EXPAND
  if (!CmiGetArgStringDesc(argv, "+shrinkexpand_basedir", (char **)&_shrinkexpand_basedir,
                           "Checkpoint directory used for shrink-expand (defaults to /dev/shm)"))
      _shrinkexpand_basedir = "/dev/shm";
#endif

  if(CmiGetArgString(argv,"+restart",&_restartDir))
      faultFunc = CkRestartMain;
#if __FAULT__
  if (CmiGetArgIntDesc(argv,"+restartaftercrash",&CpvAccess(_curRestartPhase),"restarting this processor after a crash")){	
# if CMK_MEM_CHECKPOINT
      faultFunc = CkMemRestart;
# endif
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
      faultFunc = CkMlogRestart;
#endif
      CmiPrintf("[%d] Restarting after crash \n",CmiMyPe());
  }
#if CMK_MESSAGE_LOGGING
	// reading +ftc_disk flag
	if (CmiGetArgFlagDesc(argv, "+ftc_disk", "Disk Checkpointing")) {
		diskCkptFlag = 1;
    }
#endif
  // reading the killFile
  if(CmiGetArgStringDesc(argv,"+killFile", &killFile,"Generates SIGKILL on specified processors")){
    if(faultFunc == NULL){
      //do not read the killfile if this is a restarting processor
      killFlag = 1;
      if(CmiMyPe() == 0){
        printf("[%d] killFlag set to 1 for file %s\n",CkMyPe(),killFile);
      }
    }
  }
#endif

  // shut down program in ring fashion to allow projections output w/o IO error
  if (CmiGetArgIntDesc(argv,"+ringexit",&_ringtoken, "Program exits in a ring fashion")) 
  {
    _ringexit = 1;
    if (CkMyPe()==0)
      CkPrintf("Charm++> Program shutdown in token ring (%d).\n", _ringtoken);
    if (_ringtoken > CkNumPes())  _ringtoken = CkNumPes();
  }
	/*
		FAULT_EVAC

		if the argument +raiseevac is present then cause faults
	*/
	if(CmiGetArgStringDesc(argv,"+raiseevac", &_raiseEvacFile,"Generates processor evacuation on random processors")){
		_raiseEvac = 1;
	}
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	if(!CmiGetArgIntDesc(argv,"+teamSize",&teamSize,"Set the team size for message logging")){
        teamSize = 1;
    }
    if(!CmiGetArgIntDesc(argv,"+chkptPeriod",&chkptPeriod,"Set the checkpoint period for the message logging fault tolerance algorithm in seconds")){
        chkptPeriod = 100;
    }
	if(CmiGetArgIntDesc(argv,"+fastRecovery", &parallelRecovery, "Parallel recovery with message logging protocol")){
        fastRecovery = true;
    }
#endif

        if (!CmiGetArgIntDesc(argv, "+messageBufferingThreshold",
                              &_messageBufferingThreshold,
                              "Message size above which the runtime will buffer messages directed at unlocated array elements")) {
          _messageBufferingThreshold = INT_MAX;
        }

	/* Anytime migration flag */
	_isAnytimeMigration = true;
	if (CmiGetArgFlagDesc(argv,"+noAnytimeMigration","The program does not require support for anytime migration")) {
	  _isAnytimeMigration = false;
	}
	
	_isNotifyChildInRed = true;
	if (CmiGetArgFlagDesc(argv,"+noNotifyChildInReduction","The program has at least one element per processor for each charm array created")) {
	  _isNotifyChildInRed = false;
	}

	_isStaticInsertion = false;
	if (CmiGetArgFlagDesc(argv,"+staticInsertion","Array elements are only inserted at construction")) {
	  _isStaticInsertion = true;
	}

        useNodeBlkMapping = false;
        if (CmiGetArgFlagDesc(argv,"+useNodeBlkMapping","Array elements are block-mapped in SMP-node level")) {
          useNodeBlkMapping = true;
        }

#if ! CMK_WITH_CONTROLPOINT
	// Display a warning if charm++ wasn't compiled with control point support but user is expecting it
	if( CmiGetArgFlag(argv,"+CPSamplePeriod") || 
	    CmiGetArgFlag(argv,"+CPSamplePeriodMs") || 
	    CmiGetArgFlag(argv,"+CPSchemeRandom") || 
	    CmiGetArgFlag(argv,"+CPExhaustiveSearch") || 
	    CmiGetArgFlag(argv,"+CPAlwaysUseDefaults") || 
	    CmiGetArgFlag(argv,"+CPSimulAnneal") || 
	    CmiGetArgFlag(argv,"+CPCriticalPathPrio") || 
	    CmiGetArgFlag(argv,"+CPBestKnown") || 
	    CmiGetArgFlag(argv,"+CPSteering") || 
	    CmiGetArgFlag(argv,"+CPMemoryAware") || 
	    CmiGetArgFlag(argv,"+CPSimplex") || 
	    CmiGetArgFlag(argv,"+CPDivideConquer") || 
	    CmiGetArgFlag(argv,"+CPLDBPeriod") || 
	    CmiGetArgFlag(argv,"+CPLDBPeriodLinear") || 
	    CmiGetArgFlag(argv,"+CPLDBPeriodQuadratic") || 
	    CmiGetArgFlag(argv,"+CPLDBPeriodOptimal") || 
	    CmiGetArgFlag(argv,"+CPDefaultValues") || 
	    CmiGetArgFlag(argv,"+CPGatherAll") || 
	    CmiGetArgFlag(argv,"+CPGatherMemoryUsage") || 
	    CmiGetArgFlag(argv,"+CPGatherUtilization") || 
	    CmiGetArgFlag(argv,"+CPSaveData") || 
	    CmiGetArgFlag(argv,"+CPNoFilterData") || 
	    CmiGetArgFlag(argv,"+CPLoadData") || 
	    CmiGetArgFlag(argv,"+CPDataFilename")    )
	  {	    
	    CkAbort("You specified a control point command line argument, but compiled charm++ without control point support.\n");
	  }
#endif
       
}

static void _bufferHandler(void *msg)
{
  DEBUGF(("[%d] _bufferHandler called.\n", CkMyPe()));
  CkpvAccess(_buffQ)->enq(msg);
}

static void _discardHandler(envelope *env)
{
//  MESSAGE_PHASE_CHECK(env);

  DEBUGF(("[%d] _discardHandler called.\n", CkMyPe()));
#if CMK_MEM_CHECKPOINT
  //CkPrintf("[%d] _discardHandler called!\n", CkMyPe());
  if (CkInRestarting()) CpvAccess(_qd)->process();
#endif
  CmiFree(env);
}

#if CMK_WITH_STATS
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

typedef struct _statsHeader
{
  int n;
} statsHeader;

static void * mergeStats(int *size, void *data, void **remote, int count)
{
  envelope *newData;
  statsHeader *dataMsg = (statsHeader*)EnvToUsr((envelope*) data), *newDataMsg;
  int nPes = dataMsg->n, currentIndex = 0;

  for (int i = 0; i < count; ++i)
  {
    nPes += ((statsHeader *)EnvToUsr((envelope *)remote[i]))->n;
  }

  newData = _allocEnv(StatMsg, sizeof(statsHeader) + sizeof(Stats)*nPes);
  *size = newData->getTotalsize();
  newDataMsg = (statsHeader *)EnvToUsr(newData);
  newDataMsg->n = nPes;

  statsHeader *current = dataMsg;
  Stats *currentStats = (Stats*)(current + 1), *destination = (Stats*)(newDataMsg + 1);
  memcpy(destination + currentIndex, currentStats, sizeof(Stats) * current->n);
  currentIndex += current->n;

  for (int i = 0; i < count; ++i)
  {
    current = ((statsHeader *)EnvToUsr((envelope *)remote[i]));
    currentStats = (Stats *)(current + 1);
    memcpy(destination + currentIndex, currentStats, sizeof(Stats) * current->n);
    currentIndex += current->n;
  }

  CmiFree(data);
  return newData;
}

static inline void _sendStats(void)
{
  DEBUGF(("[%d] _sendStats\n", CkMyPe()));
  envelope *env = _allocEnv(StatMsg, sizeof(statsHeader) + sizeof(Stats));
  statsHeader* msg = (statsHeader*)EnvToUsr(env);
  msg->n = 1;
  memcpy(msg+1, CkpvAccess(_myStats), sizeof(Stats));
  CmiSetHandler(env, _exitHandlerIdx);
  CmiReduce(env, env->getTotalsize(), mergeStats);
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
extern void _messageLoggingExit();
#endif

#if __FAULT__
//CpvExtern(int, CldHandlerIndex);
//extern "C" void CldHandler(char *);
extern int index_skipCldHandler;
extern void _skipCldHandler(void *converseMsg);

void _discard_charm_message()
{
  CkNumberHandler(_charmHandlerIdx,_discardHandler);
//  CkNumberHandler(CpvAccess(CldHandlerIndex), _discardHandler);
  CkNumberHandler(index_skipCldHandler, _discardHandler);
}

void _resume_charm_message()
{
  CkNumberHandlerEx(_charmHandlerIdx, _processHandler, CkpvAccess(_coreState));
//  CkNumberHandler(CpvAccess(CldHandlerIndex), CldHandler);
  CkNumberHandler(index_skipCldHandler, _skipCldHandler);
}
#endif

static void _exitHandler(envelope *env)
{
  DEBUGF(("exitHandler called on %d msgtype: %d\n", CkMyPe(), env->getMsgtype()));
  switch(env->getMsgtype()) {
    case StartExitMsg:
      CkAssert(CkMyPe()==0);
      if (!_CkExitFnVec.isEmpty()) {
        CkExitFn fn = _CkExitFnVec.deq();
        fn();
        break;
      }
      // else goto next
    case ExitMsg:
      CkAssert(CkMyPe()==0);
      if(_exitStarted) {
        CmiFree(env);
        return;
      }
      _exitStarted = 1;
      CkNumberHandler(_charmHandlerIdx,_discardHandler);
      CkNumberHandler(_bocHandlerIdx, _discardHandler);
#if !CMK_BIGSIM_THREAD
      env->setMsgtype(ReqStatMsg);
      env->setSrcPe(CkMyPe());
      // if exit in ring, instead of broadcasting, send in ring
      if (_ringexit){
	DEBUGF(("[%d] Ring Exit \n",CkMyPe()));
        const int stride = CkNumPes()/_ringtoken;
        int pe = 0;
        while (pe<CkNumPes()) {
          CmiSyncSend(pe, env->getTotalsize(), (char *)env);
          pe += stride;
        }
        CmiFree(env);
      }else{
	CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char *)env);
      }
#else
      CmiFree(env);
      ConverseExit();
#endif
      break;
    case ReqStatMsg:
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
      _messageLoggingExit();
#endif
      DEBUGF(("ReqStatMsg on %d\n", CkMyPe()));
      CkNumberHandler(_charmHandlerIdx,_discardHandler);
      CkNumberHandler(_bocHandlerIdx, _discardHandler);
      /*FAULT_EVAC*/
      if(CmiNodeAlive(CkMyPe())){
#if CMK_WITH_STATS
         _sendStats();
#endif
      _mainDone = 1; // This is needed because the destructors for
                     // readonly variables will be called when the program
		     // exits. If the destructor is called while _mainDone
		     // is 0, it will assume that the readonly variable was
		     // declared locally. On all processors other than 0, 
		     // _mainDone is never set to 1 before the program exits.
#if CMK_TRACE_ENABLED
      if (_ringexit) traceClose();
#endif
    }
      if (_ringexit) {
        int stride = CkNumPes()/_ringtoken;
        int pe = CkMyPe()+1;
        if (pe < CkNumPes() && pe % stride != 0)
          CmiSyncSendAndFree(pe, env->getTotalsize(), (char *)env);
        else
          CmiFree(env);
      }
      else
        CmiFree(env);
#if CMK_SHRINK_EXPAND
      ConverseCleanup();
#endif
      //everyone exits here - there may be issues with leftover messages in the queue
#if !CMK_WITH_STATS
      DEBUGF(("[%d] Calling converse exit from ReqStatMsg \n",CkMyPe()));
      ConverseExit();
      if(CharmLibInterOperate)
	CpvAccess(interopExitFlag) = 1;
#endif
      break;
#if CMK_WITH_STATS
    case StatMsg:
    {
      CkAssert(CkMyPe()==0);
      statsHeader* header = (statsHeader*)EnvToUsr(env);
      int n = header->n;
      Stats* currentStats = (Stats*)(header + 1);
      for (int i = 0; i < n; ++i)
      {
        _allStats[currentStats->getPe()] = currentStats;
	currentStats++;
      }
      DEBUGF(("StatMsg on %d with %d\n", CkMyPe(), n));
			/*FAULT_EVAC*/
      _printStats();
      // broadcast to all others that they can now exit
      envelope* env = _allocEnv(StatDoneMsg);
      CmiSetHandler(env, _exitHandlerIdx);
      CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char*)env);
    }
    break;

    case StatDoneMsg:
      DEBUGF(("[%d] Calling converse exit from StatDoneMsg \n",CkMyPe()));
      ConverseExit();
      if (CharmLibInterOperate)
        CpvAccess(interopExitFlag) = 1;
      break;
#endif
    default:
      CmiAbort("Internal Error(_exitHandler): Unknown-msg-type. Contact Developers.\n");
  }
}

#if CMK_SHRINK_EXPAND
void _ROGroupRestartHandler(void * msg){
  CkResumeRestartMain((char *)msg);
}
#endif

/**
 * Create all groups in this processor (not called on processor zero).
 * Notice that only groups created in mainchares are processed here;
 * groups created later are processed as regular messages.
 */
static inline void _processBufferedBocInits(void)
{
  CkCoreState *ck = CkpvAccess(_coreState);
  CkNumberHandlerEx(_bocHandlerIdx,_processHandler, ck);
  register int i = 0;
  PtrVec &inits=*CkpvAccess(_bocInitVec);
  register int len = inits.size();
  for(i=1; i<len; i++) {
    envelope *env = inits[i];
    if(env==0) {
#if CMK_SHRINK_EXPAND
      if(_inrestart){
        CkPrintf("_processBufferedBocInits: empty message in restart, ignoring\n");
        break;
      }
      else
        CkAbort("_processBufferedBocInits: empty message");
#else
      CkAbort("_processBufferedBocInits: empty message");
#endif
    }
    if(env->isPacked())
      CkUnpackMessage(&env);
    _processBocInitMsg(ck,env);
  }
  delete &inits;
}

/**
 * Create all nodegroups in this node (called only by rank zero, and never on node zero).
 * Notice that only nodegroups created in mainchares are processed here;
 * nodegroups created later are processed as regular messages.
 */
static inline void _processBufferedNodeBocInits(void)
{
  CkCoreState *ck = CkpvAccess(_coreState);
  register int i = 0;
  PtrVec &inits=*CksvAccess(_nodeBocInitVec);
  register int len = inits.size();
  for(i=1; i<len; i++) {
    envelope *env = inits[i];
    if(env==0) CkAbort("_processBufferedNodeBocInits: empty message");
    if(env->isPacked())
      CkUnpackMessage(&env);
    _processNodeBocInitMsg(ck,env);
  }
  delete &inits;
}

static inline void _processBufferedMsgs(void)
{
  CkNumberHandlerEx(_charmHandlerIdx, _processHandler, CkpvAccess(_coreState));
  envelope *env;
  while(NULL!=(env=(envelope*)CkpvAccess(_buffQ)->deq())) {
    if(env->getMsgtype()==NewChareMsg || env->getMsgtype()==NewVChareMsg) {
      if(env->isForAnyPE())
        _CldEnqueue(CLD_ANYWHERE, env, _infoIdx);
      else
        _processHandler((void *)env, CkpvAccess(_coreState));
    } else {
      _processHandler((void *)env, CkpvAccess(_coreState));
    }
  }
}

static int _charmLoadEstimator(void)
{
  return CkpvAccess(_buffQ)->length();
}

/**
 * This function is used to send other processors on the same node a signal so
 * they can check if their _initDone can be called: the reason for this is that
 * the check at the end of _initHandler can fail due to a missing message containing
 * a Nodegroup creation. When that message arrives only one processor will receive
 * it, and thus if no notification is sent to the other processors in the node, they
 * will never proceed.
 */
static void _sendTriggers(void)
{
  int i, num, first;
  CmiImmediateLock(CksvAccess(_nodeGroupTableImmLock));
  if (CksvAccess(_triggersSent) == 0)
  {
    CksvAccess(_triggersSent)++;
    num = CmiMyNodeSize();
    register envelope *env = _allocEnv(RODataMsg); // Notice that the type here is irrelevant
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _triggerHandlerIdx);
    first = CmiNodeFirst(CmiMyNode());
    for (i=0; i < num; i++)
      if(first+i != CkMyPe())
	CmiSyncSend(first+i, env->getTotalsize(), (char *)env);
    CmiFree(env);
  }
  CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));
}

/**
 * This function (not a handler) is called once and only once per processor.
 * It signals the processor that the initialization is done and regular messages
 * can be processed.
 *
 * On processor zero it is called by _initCharm, on all other processors either
 * by _initHandler or _triggerHandler (cannot be both).
 * When fault-tolerance is active, it is called by the fault-tolerance scheme itself.
 */
void _initDone(void)
{
  if (CkpvAccess(_initdone)) return;
  CkpvAccess(_initdone) ++;
  DEBUGF(("[%d] _initDone.\n", CkMyPe()));
  if (!CksvAccess(_triggersSent)) _sendTriggers();
  CkNumberHandler(_triggerHandlerIdx, _discardHandler);
  CmiNodeBarrier();
  if(CkMyRank() == 0) {
    _processBufferedNodeBocInits();
    quietMode = 0; // re-enable CmiPrintf's if they were disabled
  }
  CmiNodeBarrier(); // wait for all nodegroups to be created
  _processBufferedBocInits();
  DEBUGF(("Reached CmiNodeBarrier(), pe = %d, rank = %d\n", CkMyPe(), CkMyRank()));
  CmiNodeBarrier();
  DEBUGF(("Crossed CmiNodeBarrier(), pe = %d, rank = %d\n", CkMyPe(), CkMyRank()));
  _processBufferedMsgs();
  CkpvAccess(_charmEpoch)=1;
}

/**
 * Converse handler receiving a signal from another processors in the same node.
 * (On _sendTrigger there is the explanation of why this is necessary)
 * Simply check if with the NodeGroup processed by another processor we reached
 * the expected count. Notice that it can only be called before _initDone: after
 * _initDone, a message destined for this handler will go instead to the _discardHandler.
 */
static void _triggerHandler(envelope *env)
{
  if (_numExpectInitMsgs && CkpvAccess(_numInitsRecd) + CksvAccess(_numInitNodeMsgs) == _numExpectInitMsgs)
  {
    DEBUGF(("Calling Init Done from _triggerHandler\n"));
    _initDone();
  }
  if (env!=NULL) CmiFree(env);
}

static inline void _processROMsgMsg(envelope *env)
{
  if(!CmiMyRank()) {
    *((char **)(_readonlyMsgs[env->getRoIdx()]->pMsg))=(char *)EnvToUsr(env);
  }
}

static inline void _processRODataMsg(envelope *env)
{
  //Unpack each readonly:
  if(!CmiMyRank()) {
    PUP::fromMem pu((char *)EnvToUsr(env));
    for(size_t i=0;i<_readonlyTable.size();i++) {
      _readonlyTable[i]->pupData(pu);
    }
  }
  CmiFree(env);
}

/**
 * This is similar to the _initHandler, only that the Groups and Nodegroups are
 * initialized from disk, so only one single message is expected.
 *
 * It is unclear how Readonly Messages are treated during restart... (if at all considered)
 */
static void _roRestartHandler(void *msg)
{
  CkAssert(CkMyPe()!=0);
  register envelope *env = (envelope *) msg;
  CkpvAccess(_numInitsRecd)++;
  _numExpectInitMsgs = env->getCount();
  _processRODataMsg(env);
  // in SMP, potentially there us a race condition between rank0 calling
  // initDone, which sendTriggers, and PE 0 calls bdcastRO which broadcast
  // readonlys
  // if this readonly message arrives later, we need to call trigger again
  // to trigger initDone() on all ranks
  // we therefore needs to make sure initDone() is exactly 
  _triggerHandler(NULL);
}

/**
 * This handler is used only during initialization. It receives messages from
 * processor zero regarding Readonly Data (in one single message), Readonly Messages,
 * Groups, and Nodegroups.
 * The Readonly Data message also contains the total number of messages expected
 * during the initialization phase.
 * For Groups and Nodegroups, only messages with epoch=0 (meaning created from within
 * a mainchare) are buffered for special creation, the other messages are buffered
 * together with all the other regular messages by _bufferHandler (and will be flushed
 * after all the initialization messages have been processed).
 */
static void _initHandler(void *msg, CkCoreState *ck)
{
  CkAssert(CkMyPe()!=0);
  register envelope *env = (envelope *) msg;
  
  if (ck->watcher!=NULL) {
    if (!ck->watcher->processMessage(&env,ck)) return;
  }
  
  switch (env->getMsgtype()) {
    case BocInitMsg:
      if (env->getGroupEpoch()==0) {
        CkpvAccess(_numInitsRecd)++;
	// _processBocInitMsg already handles QD
        //CpvAccess(_qd)->process();
        CkpvAccess(_bocInitVec)->insert(env->getGroupNum().idx, env);
      } else _bufferHandler(msg);
      break;
    case NodeBocInitMsg:
      if (env->getGroupEpoch()==0) {
        CmiImmediateLock(CksvAccess(_nodeGroupTableImmLock));
        CksvAccess(_numInitNodeMsgs)++;
        CksvAccess(_nodeBocInitVec)->insert(env->getGroupNum().idx, env);
        CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));
        CpvAccess(_qd)->process();
      } else _bufferHandler(msg);
      break;
    case ROMsgMsg:
      CkpvAccess(_numInitsRecd)++;
      CpvAccess(_qd)->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processROMsgMsg(env);
      break;
    case RODataMsg:
      CkpvAccess(_numInitsRecd)++;
      CpvAccess(_qd)->process();
      _numExpectInitMsgs = env->getCount();
      _processRODataMsg(env);
      break;
    default:
      CmiAbort("Internal Error: Unknown-msg-type. Contact Developers.\n");
  }
  DEBUGF(("[%d,%.6lf] _numExpectInitMsgs %d CkpvAccess(_numInitsRecd)+CksvAccess(_numInitNodeMsgs) %d+%d\n",CmiMyPe(),CmiWallTimer(),_numExpectInitMsgs,CkpvAccess(_numInitsRecd),CksvAccess(_numInitNodeMsgs)));
  if(_numExpectInitMsgs&&(CkpvAccess(_numInitsRecd)+CksvAccess(_numInitNodeMsgs)==_numExpectInitMsgs)) {
    _initDone();
  }
}

#if 0
/*****************************************
 *          no longer needed
 * ***************************************/
extern "C"
void _CkExit(void) 
{
  CmiAssert(CkMyPe() == 0);
  // Shuts down Converse handlers for the upper layers on this processor
  //
  CkNumberHandler(_charmHandlerIdx,_discardHandler);
  CkNumberHandler(_bocHandlerIdx, _discardHandler);
  DEBUGF(("[%d] CkExit - _exitStarted:%d %d\n", CkMyPe(), _exitStarted, _exitHandlerIdx));

  if(CkMyPe()==0) {
    if(_exitStarted)
      CsdScheduler(-1);
    envelope *env = _allocEnv(ReqStatMsg);
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _exitHandlerIdx);
		/*FAULT_EVAC*/
    CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char *)env);
  } else {
    envelope *env = _allocEnv(ExitMsg);
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _exitHandlerIdx);
    CmiSyncSendAndFree(0, env->getTotalsize(), (char *)env);
  }
#if ! CMK_BIGSIM_THREAD
  _TRACE_END_EXECUTE();
  //Wait for stats, which will call ConverseExit when finished:
  CsdScheduler(-1);
#endif
}
#endif

#if CMK_SHRINK_EXPAND
extern "C"
void CkCleanup()
{
	// always send to PE 0
	envelope *env = _allocEnv(StartExitMsg);
	env->setSrcPe(CkMyPe());
	CmiSetHandler(env, _exitHandlerIdx);
	CmiSyncSendAndFree(0, env->getTotalsize(), (char *)env);
}
#endif

CkQ<CkExitFn> _CkExitFnVec;

// triger exit on PE 0,
// which traverses _CkExitFnVec to call every registered user exit functions.
// Every user exit functions should end with CkExit() to continue the chain
extern "C"
void CkExit(void)
{
	/*FAULT_EVAC*/
  DEBUGF(("[%d] CkExit called \n",CkMyPe()));
    // always send to PE 0
  envelope *env = _allocEnv(StartExitMsg);
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _exitHandlerIdx);
  CmiSyncSendAndFree(0, env->getTotalsize(), (char *)env);

#if ! CMK_BIGSIM_THREAD
  _TRACE_END_EXECUTE();
  //Wait for stats, which will call ConverseExit when finished:
	if(!CharmLibInterOperate)
  CsdScheduler(-1);
#endif
}

/* This is a routine called in case the application is closing due to a signal.
   Tear down structures that must be cleaned up even when unclean exit happens.
   It is called by the machine layer whenever some problem occurs (it is thus up
   to the machine layer to call this function). */
extern "C"
void EmergencyExit(void) {
#ifndef __BIGSIM__
  /* Delete _coreState to force any CkMessageWatcher to close down. */
  if (CkpvAccess(_coreState) != NULL) {
    delete CkpvAccess(_coreState);
    CkpvAccess(_coreState) = NULL;
  }
#endif
}

static void _nullFn(void *, void *)
{
  CmiAbort("Null-Method Called. Program may have Unregistered Module!!\n");
}

extern void _registerLBDatabase(void);
extern void _registerMetaBalancer(void);
extern void _registerPathHistory(void);
#if CMK_WITH_CONTROLPOINT
extern void _registerControlPoints(void);
#endif
extern void _registerTraceControlPoints();
extern void _registerExternalModules(char **argv);
extern void _ckModuleInit(void);
extern void _loadbalancerInit();
extern void _metabalancerInit();
extern void _initChareTables();
#if CMK_MEM_CHECKPOINT
extern void init_memcheckpt(char **argv);
#endif
extern "C" void initCharmProjections();
extern "C" void CmiInitCPUTopology(char **argv);
extern "C" void CmiInitCPUAffinity(char **argv);
extern "C" void CmiInitMemAffinity(char **argv);
extern "C" void CmiInitPxshm(char **argv);

//extern "C" void CldCallback();

void _registerInitCall(CkInitCallFn fn, int isNodeCall)
{
  if (isNodeCall) _initCallTable.initNodeCalls.enq(fn);
  else _initCallTable.initProcCalls.enq(fn);
}

void InitCallTable::enumerateInitCalls()
{
  int i;
#ifdef __BIGSIM__
  if(BgNodeRank()==0)        // called only once on an emulating node
#else
  if(CkMyRank()==0) 
#endif
  {
    for (i=0; i<initNodeCalls.length(); i++) initNodeCalls[i]();
  }
  // initproc may depend on initnode calls.
  CmiNodeAllBarrier();
  for (i=0; i<initProcCalls.length(); i++) initProcCalls[i]();
}

CpvCExtern(int, cpdSuspendStartup);
extern "C" void CpdFreeze(void);

extern int _dummy_dq;

extern "C" void initQd(char **argv)
{
	CpvInitialize(QdState*, _qd);
	CpvAccess(_qd) = new QdState();
	if (CmiMyRank() == 0) {
#if !defined(CMK_CPV_IS_SMP) && !CMK_SHARED_VARS_UNIPROCESSOR
	CpvAccessOther(_qd, 1) = new QdState(); // for i/o interrupt
#endif
	}
	_qdHandlerIdx = CmiRegisterHandler((CmiHandler)_qdHandler);
	_qdCommHandlerIdx = CmiRegisterHandler((CmiHandler)_qdCommHandler);
        if (CmiGetArgIntDesc(argv,"+qd",&_dummy_dq, "QD time in seconds")) {
          if (CmiMyPe()==0)
            CmiPrintf("Charm++> Fake QD using %d seconds.\n", _dummy_dq);
        }
}

#if CMK_BIGSIM_CHARM && CMK_CHARMDEBUG
void CpdBgInit();
#endif
void CpdBreakPointInit();

/**
  This is the main charm setup routine.  It's called
  on all processors after Converse initialization.
  This routine gets passed to Converse from "main.C".
  
  The main purpose of this routine is to set up the objects
  and Ckpv's used during a regular Charm run.  See the comment
  at the top of the file for overall flow.
*/
void _initCharm(int unused_argc, char **argv)
{ 
	int inCommThread = (CmiMyRank() == CmiMyNodeSize());

	DEBUGF(("[%d,%.6lf ] _initCharm started\n",CmiMyPe(),CmiWallTimer()));

	CkpvInitialize(size_t *, _offsets);
	CkpvAccess(_offsets) = new size_t[32];
	CkpvInitialize(PtrQ*,_buffQ);
	CkpvInitialize(PtrVec*,_bocInitVec);
	CkpvInitialize(void*, _currentChare);
	CkpvInitialize(int,   _currentChareType);
	CkpvInitialize(CkGroupID, _currentGroup);
	CkpvInitialize(void *, _currentNodeGroupObj);
	CkpvInitialize(CkGroupID, _currentGroupRednMgr);
	CkpvInitialize(GroupTable*, _groupTable);
	CkpvInitialize(GroupIDTable*, _groupIDTable);
	CkpvInitialize(CmiImmediateLockType, _groupTableImmLock);
        CkpvInitialize(bool, _destroyingNodeGroup);
        CkpvAccess(_destroyingNodeGroup) = false;
	CkpvInitialize(UInt, _numGroups);
	CkpvInitialize(int, _numInitsRecd);
	CkpvInitialize(int, _initdone);
	CkpvInitialize(char**, Ck_argv); CkpvAccess(Ck_argv)=argv;
	CkpvInitialize(MsgPool*, _msgPool);
	CkpvInitialize(CkCoreState *, _coreState);
	/*
		Added for evacuation-sayantan
	*/
#ifndef __BIGSIM__
	CpvInitialize(char *,_validProcessors);
#endif
	CkpvInitialize(char ,startedEvac);
	CpvInitialize(int,serializer);

	_initChareTables();            // for checkpointable plain chares

	CksvInitialize(UInt, _numNodeGroups);
	CksvInitialize(GroupTable*, _nodeGroupTable);
	CksvInitialize(GroupIDTable, _nodeGroupIDTable);
	CksvInitialize(CmiImmediateLockType, _nodeGroupTableImmLock);
	CksvInitialize(CmiNodeLock, _nodeLock);
	CksvInitialize(PtrVec*,_nodeBocInitVec);
	CksvInitialize(UInt,_numInitNodeMsgs);
	CkpvInitialize(int,_charmEpoch);
	CkpvAccess(_charmEpoch)=0;
	CksvInitialize(int, _triggersSent);
	CksvAccess(_triggersSent) = 0;

	CkpvInitialize(_CkOutStream*, _ckout);
	CkpvInitialize(_CkErrStream*, _ckerr);
	CkpvInitialize(Stats*, _myStats);

	CkpvAccess(_groupIDTable) = new GroupIDTable(0);
	CkpvAccess(_groupTable) = new GroupTable;
	CkpvAccess(_groupTable)->init();
	CkpvAccess(_groupTableImmLock) = CmiCreateImmediateLock();
	CkpvAccess(_numGroups) = 1; // make 0 an invalid group number
	CkpvAccess(_buffQ) = new PtrQ();
	CkpvAccess(_bocInitVec) = new PtrVec();

	CkpvAccess(_currentNodeGroupObj) = NULL;

	if(CkMyRank()==0)
	{
	  	CksvAccess(_numNodeGroups) = 1; //make 0 an invalid group number
          	CksvAccess(_numInitNodeMsgs) = 0;
		CksvAccess(_nodeLock) = CmiCreateLock();
		CksvAccess(_nodeGroupTable) = new GroupTable();
		CksvAccess(_nodeGroupTable)->init();
		CksvAccess(_nodeGroupTableImmLock) = CmiCreateImmediateLock();
		CksvAccess(_nodeBocInitVec) = new PtrVec();
	}

	CkCallbackInit();
	
	CmiNodeAllBarrier();

#if ! CMK_BIGSIM_CHARM
	initQd(argv);         // bigsim calls it in ConverseCommonInit
#endif

	CkpvAccess(_coreState)=new CkCoreState();

	CkpvAccess(_numInitsRecd) = 0;
	CkpvAccess(_initdone) = 0;

	CkpvAccess(_ckout) = new _CkOutStream();
	CkpvAccess(_ckerr) = new _CkErrStream();

	_charmHandlerIdx = CkRegisterHandler(_bufferHandler);
	_initHandlerIdx = CkRegisterHandlerEx(_initHandler, CkpvAccess(_coreState));
	_roRestartHandlerIdx = CkRegisterHandler(_roRestartHandler);
	_exitHandlerIdx = CkRegisterHandler(_exitHandler);
	//added for interoperabilitY
	_libExitHandlerIdx = CkRegisterHandler(_libExitHandler);
	_bocHandlerIdx = CkRegisterHandlerEx(_initHandler, CkpvAccess(_coreState));
#if CMK_SHRINK_EXPAND
	// for shrink expand cleanup
	_ROGroupRestartHandlerIdx = CkRegisterHandler(_ROGroupRestartHandler);
#endif

#ifdef __BIGSIM__
	if(BgNodeRank()==0) 
#endif
	_infoIdx = CldRegisterInfoFn((CldInfoFn)_infoFn);

	_triggerHandlerIdx = CkRegisterHandler(_triggerHandler);
	_ckModuleInit();

	CldRegisterEstimator((CldEstimator)_charmLoadEstimator);

	_futuresModuleInit(); // part of futures implementation is a converse module
	_loadbalancerInit();
        _metabalancerInit();
	
#if CMK_MEM_CHECKPOINT
        init_memcheckpt(argv);
#endif

	initCharmProjections();
#if CMK_TRACE_IN_CHARM
        // initialize trace module in ck
        traceCharmInit(argv);
#endif
 	
    CkpvInitialize(int, envelopeEventID);
    CkpvAccess(envelopeEventID) = 0;
	CkMessageWatcherInit(argv,CkpvAccess(_coreState));
	
	/**
	  The rank-0 processor of each node calls the 
	  translator-generated "_register" routines. 
	  
	  _register routines call the charm.h "CkRegister*" routines,
	  which record function pointers and class information for
	  all Charm entities, like Chares, Arrays, and readonlies.
	  
	  There's one _register routine generated for each
	  .ci file.  _register routines *must* be called in the 
	  same order on every node, and *must not* be called by 
	  multiple threads simultaniously.
	*/
#ifdef __BIGSIM__
	if(BgNodeRank()==0) 
#else
	if(CkMyRank()==0)
#endif
	{
		SDAG::registerPUPables();
		CmiArgGroup("Charm++",NULL);
		_parseCommandLineOpts(argv);
		_registerInit();
		CkRegisterMsg("System", 0, 0, CkFreeMsg, sizeof(int));
		CkRegisterChareInCharm(CkRegisterChare("null", 0, TypeChare));
		CkIndex_Chare::__idx=CkRegisterChare("Chare", sizeof(Chare), TypeChare);
		CkRegisterChareInCharm(CkIndex_Chare::__idx);
		CkIndex_Group::__idx=CkRegisterChare("Group", sizeof(Group), TypeGroup);
                CkRegisterChareInCharm(CkIndex_Group::__idx);
		CkRegisterEp("null", (CkCallFnPtr)_nullFn, 0, 0, 0+CK_EP_INTRINSIC);
		
		/**
		  These _register calls are for the built-in
		  Charm .ci files, like arrays and load balancing.
		  If you add a .ci file to charm, you'll have to 
		  add a call to the _register routine here, or make
		  your library into a "-module".
		*/
		_registerCkFutures();
		_registerCkArray();
		_registerLBDatabase();
    _registerMetaBalancer();
		_registerCkCallback();
		_registertempo();
		_registerwaitqd();
		_registerCkCheckpoint();
#if CMK_MEM_CHECKPOINT
		_registerCkMemCheckpoint();
#endif


		/*
		  Setup Control Point Automatic Tuning Framework.

		  By default it is enabled as a part of charm, 
		  however it won't enable its tracing module 
		  unless a +CPEnableMeasurements command line argument
		  is specified. See trace-common.C for more info

		  Thus there should be no noticable overhead to 
		  always having the control point framework linked
		  in.
		  
		*/
#if CMK_WITH_CONTROLPOINT
		_registerPathHistory();
		_registerControlPoints();
		_registerTraceControlPoints();
#endif


		/**
		  CkRegisterMainModule is generated by the (unique)
		  "mainmodule" .ci file.  It will include calls to 
		  register all the .ci files.
		*/
		CkRegisterMainModule();

		/**
		  _registerExternalModules is actually generated by 
		  charmc at link time (as "moduleinit<pid>.C").  
		  
		  This generated routine calls the _register functions
		  for the .ci files of libraries linked using "-module".
		  This funny initialization is most useful for AMPI/FEM
		  programs, which don't have a .ci file and hence have
		  no other way to control the _register process.
		*/
		_registerExternalModules(argv);
		
		_registerDone();
	}
	/* The following will happen on every virtual processor in BigEmulator, not just on once per real processor */
	if (CkMyRank() == 0) {
	  CpdBreakPointInit();
	}
	CmiNodeAllBarrier();

	// Execute the initcalls registered in modules
	_initCallTable.enumerateInitCalls();

#if CMK_CHARMDEBUG
	CpdFinishInitialization();
#endif

	//CmiNodeAllBarrier();

	CkpvAccess(_myStats) = new Stats();
	CkpvAccess(_msgPool) = new MsgPool();

	CmiNodeAllBarrier();

#if !(__FAULT__)
	CmiBarrier();
	CmiBarrier();
	CmiBarrier();
#endif
#if CMK_SMP_TRACE_COMMTHREAD
	_TRACE_BEGIN_COMPUTATION();	
#else
 	if (!inCommThread) {
	  _TRACE_BEGIN_COMPUTATION();
	}
#endif

#ifdef ADAPT_SCHED_MEM
    if(CkMyRank()==0){
	memCriticalEntries = new int[numMemCriticalEntries];
	int memcnt=0;
	for(int i=0; i<_entryTable.size(); i++){
	    if(_entryTable[i]->isMemCritical){
		memCriticalEntries[memcnt++] = i;
	    }
	}
    }
#endif

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    _messageLoggingInit();
#endif

#ifndef __BIGSIM__
	/*
		FAULT_EVAC
	*/
	CpvAccess(_validProcessors) = new char[CkNumPes()];
	for(int vProc=0;vProc<CkNumPes();vProc++){
		CpvAccess(_validProcessors)[vProc]=1;
	}
	_ckEvacBcastIdx = CkRegisterHandler(_ckEvacBcast);
	_ckAckEvacIdx = CkRegisterHandler(_ckAckEvac);
#endif
	CkpvAccess(startedEvac) = 0;
	CpvAccess(serializer) = 0;

	evacuate = 0;
	CcdCallOnCondition(CcdSIGUSR1,(CcdVoidFn)CkDecideEvacPe,0);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_)) 
    CcdCallOnCondition(CcdSIGUSR2,(CcdVoidFn)CkMlogRestart,0);
#endif

	if(_raiseEvac){
		processRaiseEvacFile(_raiseEvacFile);
		/*
		if(CkMyPe() == 2){
		//	CcdCallOnConditionKeep(CcdPERIODIC_10s,(CcdVoidFn)CkDecideEvacPe,0);
			CcdCallFnAfter((CcdVoidFn)CkDecideEvacPe, 0, 10000);
		}
		if(CkMyPe() == 3){
			CcdCallFnAfter((CcdVoidFn)CkDecideEvacPe, 0, 10000);
		}*/
	}	
    
    if (CkMyRank() == 0) {
      TopoManager_init();
    }
    CmiNodeAllBarrier();

    if (!_replaySystem) {
        CkFtFn  faultFunc_restart = CkRestartMain;
        if (faultFunc == NULL || faultFunc == faultFunc_restart) {         // this is not restart from memory
            // these two are blocking calls for non-bigsim
#if ! CMK_BIGSIM_CHARM
	  CmiInitCPUAffinity(argv);
          CmiInitMemAffinity(argv);
#endif
        }
        CmiInitCPUTopology(argv);
#if CMK_SHARED_VARS_POSIX_THREADS_SMP
        if (CmiCpuTopologyEnabled()) {
            int *pelist;
            int num;
            CmiGetPesOnPhysicalNode(0, &pelist, &num);
#if !CMK_MULTICORE && !CMK_SMP_NO_COMMTHD
            // Count communication threads, if present
            // XXX: Assuming uniformity of node size here
            num += num/CmiMyNodeSize();
#endif
            if (!_Cmi_forceSpinOnIdle && num > CmiNumCores())
            {
              if (CmiMyPe() == 0)
                CmiPrintf("\nCharm++> Warning: the number of SMP threads (%d) is greater than the number of physical cores (%d), so threads will sleep while idling. Use +CmiSpinOnIdle or +CmiSleepOnIdle to control this directly.\n\n", num, CmiNumCores());
              CmiLock(CksvAccess(_nodeLock));
              if (! _Cmi_sleepOnIdle) _Cmi_sleepOnIdle = 1;
              CmiUnlock(CksvAccess(_nodeLock));
            }
        }
#endif
    }

    if(CmiMyPe() == 0) {
        char *topoFilename;
        if(CmiGetArgStringDesc(argv,"+printTopo",&topoFilename,"topo file name")) 
        {
            std::stringstream sstm;
            sstm << topoFilename << "." << CmiMyPartition();
            std::string result = sstm.str();
            FILE *fp;
            fp = fopen(result.c_str(), "w");
            if (fp == NULL) {
              CkPrintf("Error opening %s file, writing to stdout\n", topoFilename);
              fp = stdout;
            }
	    TopoManager_printAllocation(fp);
            fclose(fp);
        }
    }

#if CMK_USE_PXSHM && ( CMK_CRAYXE || CMK_CRAYXC ) && CMK_SMP
      // for SMP on Cray XE6 (hopper) it seems pxshm has to be initialized
      // again after cpuaffinity is done
    if (CkMyRank() == 0) {
      CmiInitPxshm(argv);
    }
    CmiNodeAllBarrier();
#endif

    //CldCallback();
#if CMK_BIGSIM_CHARM && CMK_CHARMDEBUG
      // Register the BG handler for CCS. Notice that this is put into a variable shared by
      // the whole real processor. This because converse needs to find it. We check that all
      // virtual processors register the same index for this handler.
    CpdBgInit();
#endif

	if (faultFunc) {
#if CMK_WITH_STATS
		if (CkMyPe()==0) _allStats = new Stats*[CkNumPes()];
#endif
		if (!inCommThread) {
                  CkArgMsg *msg = (CkArgMsg *)CkAllocMsg(0, sizeof(CkArgMsg), 0);
                  msg->argc = CmiGetArgc(argv);
                  msg->argv = argv;
                  faultFunc(_restartDir, msg);
                  CkFreeMsg(msg);
                }
	}else if(CkMyPe()==0){
#if CMK_WITH_STATS
		_allStats = new Stats*[CkNumPes()];
#endif
		register size_t i, nMains=_mainTable.size();

		// Check CkArgMsg and warn if it contains any args starting with '+'.
		// These args may be args intended for Charm++ but because of the specific
		// build, were not parsed by the RTS.
		int count = 0;
		int argc = CmiGetArgc(argv);
		for (int i = 1; i < argc; i++) {
			// The +vp option for TCharm is a special case that needs to be checked
			// separately, because the number passed does not need a space after
			// the vp, and the option can be specified with a '+' or a '-'.
			if (strncmp(argv[i],"+vp",3) == 0) {
				if (_optSet.count("+vp") == 0) {
					count++;
					CmiPrintf("WARNING: %s is a TCharm command line argument, but you have not compiled with TCharm\n", argv[i]);
				}
			} else if (strncmp(argv[i],"-vp",3) == 0) {
				CmiPrintf("WARNING: %s is no longer valid because -vp has been deprecated. Please use +vp.\n", argv[i]);
			} else if (argv[i][0] == '+' && _optSet.count(argv[i]) == 0) {
				count++;
				CmiPrintf("WARNING: %s is a command line argument beginning with a '+' but was not parsed by the RTS.\n", argv[i]);
			} else if (argv[i][0] == '+' && _optSet.count(argv[i]) != 0) {
				fprintf(stderr,"%s is used more than once. Please remove duplicate arguments.\n", argv[i]);
				CmiAbort("Bad command-line argument\n");
			}
		}
		if (count) {
			CmiPrintf("If any of the above arguments were intended for the RTS you may need to recompile Charm++ with different options.\n");
		}

		for(i=0;i<nMains;i++)  /* Create all mainchares */
		{
			register int size = _chareTable[_mainTable[i]->chareIdx]->size;
			register void *obj = malloc(size);
			_MEMCHECK(obj);
			_mainTable[i]->setObj(obj);
			CkpvAccess(_currentChare) = obj;
			CkpvAccess(_currentChareType) = _mainTable[i]->chareIdx;
			register CkArgMsg *msg = (CkArgMsg *)CkAllocMsg(0, sizeof(CkArgMsg), 0);
			msg->argc = CmiGetArgc(argv);
			msg->argv = argv;
      quietMode = 0;  // allow printing any mainchare user messages
			_entryTable[_mainTable[i]->entryIdx]->call(msg, obj);
      if (quietModeRequested) quietMode = 1;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
            CpvAccess(_currentObj) = (Chare *)obj;
#endif
		}
                _mainDone = 1;

		_STATS_RECORD_CREATE_CHARE_N(nMains);
		_STATS_RECORD_PROCESS_CHARE_N(nMains);




		for(i=0;i<_readonlyMsgs.size();i++) /* Send out readonly messages */
		{
			register void *roMsg = (void *) *((char **)(_readonlyMsgs[i]->pMsg));
			if(roMsg==0)
				continue;
			//Pack the message and send it to all other processors
			register envelope *env = UsrToEnv(roMsg);
			env->setSrcPe(CkMyPe());
			env->setMsgtype(ROMsgMsg);
			env->setRoIdx(i);
			CmiSetHandler(env, _initHandlerIdx);
			CkPackMessage(&env);
			CmiSyncBroadcast(env->getTotalsize(), (char *)env);
			CpvAccess(_qd)->create(CkNumPes()-1);

			//For processor 0, unpack and re-set the global
			CkUnpackMessage(&env);
			_processROMsgMsg(env);
			_numInitMsgs++;
		}

		//Determine the size of the RODataMessage
		PUP::sizer ps;
		for(i=0;i<_readonlyTable.size();i++) _readonlyTable[i]->pupData(ps);

		//Allocate and fill out the RODataMessage
		envelope *env = _allocEnv(RODataMsg, ps.size());
		PUP::toMem pp((char *)EnvToUsr(env));
		for(i=0;i<_readonlyTable.size();i++) _readonlyTable[i]->pupData(pp);

		env->setCount(++_numInitMsgs);
		env->setSrcPe(CkMyPe());
		CmiSetHandler(env, _initHandlerIdx);
		DEBUGF(("[%d,%.6lf] RODataMsg being sent of size %d \n",CmiMyPe(),CmiWallTimer(),env->getTotalsize()));
		CmiSyncBroadcastAndFree(env->getTotalsize(), (char *)env);
		CpvAccess(_qd)->create(CkNumPes()-1);
		_initDone();
	}

	DEBUGF(("[%d,%d%.6lf] inCommThread %d\n",CmiMyPe(),CmiMyRank(),CmiWallTimer(),inCommThread));
	// when I am a communication thread, I don't participate initDone.
        if (inCommThread) {
                CkNumberHandlerEx(_bocHandlerIdx, _processHandler, CkpvAccess(_coreState));
                CkNumberHandlerEx(_charmHandlerIdx, _processHandler, CkpvAccess(_coreState));
                _processBufferedMsgs();
        }

#if CMK_CHARMDEBUG
        // Should not use CpdFreeze inside a thread (since this processor is really a user-level thread)
       if (CpvAccess(cpdSuspendStartup))
       { 
          //CmiPrintf("In Parallel Debugging mode .....\n");
          CpdFreeze();
       }
#endif


#if __FAULT__
	if(killFlag){                                                  
                readKillFile();                                        
        }
#endif

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

// user callable function to register an exit function, this function
// will perform task of collecting of info from all pes to pe0, and call
// CkExit() on pe0 again to recursively traverse the registered exitFn.
// see trace-summary for an example.
void registerExitFn(CkExitFn fn)
{
#if CMK_SHRINK_EXPAND
  CkAbort("registerExitFn is called when shrink-expand is enabled!");
#else
  _CkExitFnVec.enq(fn);
#endif
}

/*@}*/
