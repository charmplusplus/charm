#include "CkLoop.h"
#if !defined(_WIN32)
#include <unistd.h>
#include <pthread.h>
#endif

#if !USE_CONVERSE_NOTIFICATION
#include "qd.h"
#endif

#if CMK_NODE_QUEUE_AVAILABLE
void CmiPushNode(void *msg);
#endif

#define CKLOOP_USECHARM 1
#define CKLOOP_PTHREAD 2
#define CKLOOP_NOOP 3

/*====Beginning of pthread-related variables and impelementation====*/
#if !CMK_SMP && !defined(_WIN32)
static CMK_THREADLOCAL pthread_cond_t thdCondition; //the signal var of each pthread to be notified
static CMK_THREADLOCAL pthread_mutex_t thdLock; //the lock associated with the condition variables
#endif

static FuncCkLoop *mainHelper = NULL;
static int mainHelperPhyRank = 0;
static int numPhysicalPEs = 0;
static CurLoopInfo *pthdLoop = NULL; //the pthread-version is always synchronized
#if !defined(_WIN32)
static pthread_mutex_t **allLocks = NULL;
static pthread_cond_t **allConds = NULL;
static pthread_t *ndhThreads = NULL;
#endif
static volatile int gCrtCnt = 0;
static volatile int exitFlag = 0;

#if CMK_OS_IS_LINUX
#include <sys/syscall.h>
#endif

static int HelperOnCore() {
#if CMK_OS_IS_LINUX
    char fname[64];
    sprintf(fname, "/proc/%d/task/%ld/stat", getpid(), syscall(SYS_gettid));
    FILE *ifp = fopen(fname, "r");
    if (ifp == NULL) return -1;
    fseek(ifp, 0, SEEK_SET);
    char str[128];
    for (int i=0; i<39; i++) fscanf(ifp, "%127s", str);
    fclose(ifp);
    return atoi(str);
#else
    return -1;
#endif
}

static void *ndhThreadWork(void *id) {
#if !CMK_SMP && !defined(_WIN32)
    size_t myId = (size_t) id;

    //further improvement of this affinity setting!!
    int myPhyRank = (myId+mainHelperPhyRank)%numPhysicalPEs;
    //CkPrintf("thread[%d]: affixed to rank %d\n", myId, myPhyRank);
    myPhyRank = myId;
    CmiSetCPUAffinity(myPhyRank);

    pthread_mutex_init(&thdLock, NULL);
    pthread_cond_init(&thdCondition, NULL);

    allLocks[myId-1] = &thdLock;
    allConds[myId-1] = &thdCondition;

    __sync_add_and_fetch(&gCrtCnt, 1);

    while (1) {
        //CkPrintf("thread[%ld]: on core %d with main %d\n", myId, HelperOnCore(), mainHelperPhyRank);
        if (exitFlag) break;
        pthread_mutex_lock(&thdLock);
        pthread_cond_wait(&thdCondition, &thdLock);
        pthread_mutex_unlock(&thdLock);
        /* kids ID range: [1 ~ numHelpers-1] */
        if (mainHelper->getSchedPolicy() == CKLOOP_TREE) {
            //notify my children
            int myKid = myId*TREE_BCAST_BRANCH+1;
            for (int i=0; i<TREE_BCAST_BRANCH; i++, myKid++) {
                if (myKid >= mainHelper->getNumHelpers()) break;
                //all locks and conditions exclude the main thread, so index needs to be subtracted by one
                pthread_mutex_lock(allLocks[myKid-1]);
                pthread_cond_signal(allConds[myKid-1]);
                pthread_mutex_unlock(allLocks[myKid-1]);
            }
        }
        pthdLoop->stealWork();
    }
    return NULL;
#else
    return NULL;
#endif
}

void FuncCkLoop::createPThreads() {
#if !defined(_WIN32)
    int numThreads = numHelpers - 1;
    allLocks = (pthread_mutex_t **)malloc(sizeof(void *)*numThreads);
    allConds = (pthread_cond_t **)malloc(sizeof(void *)*numThreads);
    memset(allLocks, 0, sizeof(void *)*numThreads);
    memset(allConds, 0, sizeof(void *)*numThreads);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    ndhThreads = new pthread_t[numThreads];
    mainHelperPhyRank = CmiOnCore();
    numPhysicalPEs = CmiNumCores();
    if (mainHelperPhyRank == -1) mainHelperPhyRank = 0;
    for (int i=1; i<=numThreads; i++) {
        pthread_create(ndhThreads+i, &attr, ndhThreadWork, (void *)(intptr_t)i);
    }
    while (gCrtCnt != numThreads); //wait for all threads to finish creation
#endif
}

void FuncCkLoop::exit() {
#if !defined(_WIN32)
    if (mode == CKLOOP_PTHREAD) {
        exitFlag = 1;
        for (int i=0; i<numHelpers-1; i++)
            pthread_join(ndhThreads[i], NULL);
        delete [] ndhThreads;
        free(allLocks);
        free(allConds);
        delete pthdLoop;
    }
#endif
}

/*====End of pthread-related variables and impelementation====*/


/* Note: Those event ids should be unique globally!! */
#define CKLOOP_TOTAL_WORK_EVENTID  139
#define CKLOOP_FINISH_SIGNAL_EVENTID 143
#define CKLOOP_STATIC_CHUNK_WORK 998
#define CKLOOP_DYNAMIC_CHUNK_WORK 999

static FuncCkLoop *globalCkLoop = NULL;

FuncCkLoop::FuncCkLoop(int mode_, int numThreads_) {
  init(mode_, numThreads_);
}

void FuncCkLoop::init(int mode_, int numThreads_) {
  traceRegisterUserEvent("ckloop total work",CKLOOP_TOTAL_WORK_EVENTID);
  traceRegisterUserEvent("ckloop finish signal",CKLOOP_FINISH_SIGNAL_EVENTID);

  mode = mode_;
  loop_info_inited_lock = CmiCreateLock();

  CmiAssert(globalCkLoop==NULL);
  globalCkLoop = this;

  if (mode == CKLOOP_USECHARM) {
      //CkPrintf("FuncCkLoop created on node %d\n", CkMyNode());
      numHelpers = CkMyNodeSize();
      helperPtr = new FuncSingleHelper *[numHelpers];
#if CMK_NODE_QUEUE_AVAILABLE
      schedPolicy = (numHelpers >= USE_TREE_BROADCAST_THRESHOLD ? CKLOOP_NODE_QUEUE : CKLOOP_LIST);
#else
      schedPolicy =  (numHelpers >= USE_TREE_BROADCAST_THRESHOLD ? CKLOOP_TREE : CKLOOP_LIST);
#endif
      int pestart = CkNodeFirst(CkMyNode());

      for (int i=0; i<numHelpers; i++) {
          CkChareID helper;
          CProxy_FuncSingleHelper::ckNew(&helper, pestart+i);
      }
  } else if (mode == CKLOOP_PTHREAD) {
      helperPtr = NULL;

      numHelpers = numThreads_;
      schedPolicy =  (numHelpers >= USE_TREE_BROADCAST_THRESHOLD ? CKLOOP_TREE : CKLOOP_LIST);
      pthdLoop = new CurLoopInfo(FuncCkLoop::MAX_CHUNKS);
      mainHelper = this;
      createPThreads();
  }
}

FuncCkLoop::FuncCkLoop(CkMigrateMessage *m) : CBase_FuncCkLoop(m) {
}

int FuncCkLoop::MAX_CHUNKS = 64;

#if CMK_TRACE_ENABLED
#define TRACE_START(id) _start = CmiWallTimer()
#define TRACE_BRACKET(id) traceUserBracketEvent(id,_start,CmiWallTimer())
#else
#define TRACE_START(id)
#define TRACE_BRACKET(id)
#endif

#define ALLOW_MULTIPLE_UNSYNC 1
void FuncCkLoop::parallelizeFunc(HelperFn func, int paramNum, void * param,
                                     int numChunks, int lowerRange,
                                     int upperRange, int sync,
                                     void *redResult, REDUCTION_TYPE type,
                                     CallerFn cfunc,
                                     int cparamNum, void * cparam) {

    double _start; //may be used for tracing

    if (numChunks > MAX_CHUNKS) {
        numChunks = MAX_CHUNKS;
    }

    /* "stride" determines the number of loop iterations to be done in each chunk
     * for chunk indexed at 0 to remainder-1, stride is "unit+1";
     * for chunk indexed at remainder to numChunks-1, stride is "unit"
    int stride;
     */
    CurLoopInfo *curLoop = NULL;

    //for using nodequeue
    TRACE_START(CKLOOP_TOTAL_WORK_EVENTID);

    /* Setting numChunks to 0 guarantees that func (and cfunc, if it exists)
     * will be executed on the calling PE.
     *
     * Setting numChunks to 1 guarantees that func will be executed on the
     * calling PE if cfunc does not exist. Otherwise (not in this block,
     * see below) it attempts to offload func while executing cfunc on the
     * calling PE, in an attempt to overlap their executions.
     */
    if (mode == CKLOOP_NOOP || numChunks + !!cfunc < 2) {
      func(lowerRange, upperRange, redResult, paramNum, param);
      if (cfunc != NULL) {
        cfunc(cparamNum, cparam);
      }
      return;
    } else if (mode == CKLOOP_USECHARM) {
        FuncSingleHelper *thisHelper = helperPtr[CkMyRank()];
#if USE_CONVERSE_NOTIFICATION
#if ALLOW_MULTIPLE_UNSYNC
        ConverseNotifyMsg *notifyMsg = thisHelper->getNotifyMsg();
#else
        ConverseNotifyMsg *notifyMsg = thisHelper->notifyMsg;
#endif
        curLoop = (CurLoopInfo *)(notifyMsg->ptr);
        curLoop->set(numChunks, func, lowerRange, upperRange, paramNum, param);
#if CMK_TRACE_ENABLED
        envelope *env = CpvAccess(dummyEnv);
#endif
        CmiMemoryReadFence();
        if (schedPolicy == CKLOOP_NODE_QUEUE) {
#if CMK_NODE_QUEUE_AVAILABLE
            notifyMsg->queueID = NODE_Q;
#if CMK_TRACE_ENABLED
            int loopTimes = CkMyNodeSize();
            _TRACE_CREATION_N(env, loopTimes);
            notifyMsg->eventID = env->getEvent();
#endif
            notifyMsg->srcRank = CmiMyRank();
            CmiPushNode((void *)(notifyMsg));
#else
            CkAbort("SchedPolicy, CKLOOP_NODE_QUEUE is not available on this environment\n");
#endif
        }
        else if (schedPolicy == CKLOOP_TREE) {
            int loopTimes = TREE_BCAST_BRANCH>(CmiMyNodeSize()-1)?CmiMyNodeSize()-1:TREE_BCAST_BRANCH;
            //just implicit binary tree
            int pe = CmiMyRank()+1;
            notifyMsg->queueID = NODE_Q;
#if CMK_TRACE_ENABLED
            _TRACE_CREATION_N(env, loopTimes);
            notifyMsg->eventID =env->getEvent();
#endif
            for (int i=0; i<loopTimes; i++, pe++) {
                if (pe >= CmiMyNodeSize()) pe -= CmiMyNodeSize();
                CmiPushPE(pe, (void *)(notifyMsg));
            }
        } else { /* schedPolicy == CKLOOP_LIST */
            notifyMsg->queueID = PE_Q;
#if CMK_TRACE_ENABLED
            _TRACE_CREATION_N(env, numHelpers-1);
            notifyMsg->eventID = env->getEvent();
#endif
            for (int i=CmiMyRank()+1; i<numHelpers; i++) {
                if (CpvAccessOther(isHelperOn, i))
                    CmiPushPE(i, (void *)(notifyMsg));
            }
            for (int i=0; i<CmiMyRank(); i++) {
                if (CpvAccessOther(isHelperOn, i))
                    CmiPushPE(i, (void *)(notifyMsg));
            }
        }
#else /* We don't support scheduling policy on node queue in this case */
#if ALLOW_MULTIPLE_UNSYNC
        curLoop = thisHelper->getNewTask();
#else
        curLoop = thisHelper->taskBuffer[0];
#endif
        curLoop->set(numChunks, func, lowerRange, upperRange, paramNum, param);
        CpvAccess(_qd)->create(numHelpers-1);
        CmiMemoryReadFence();
        if (schedPolicy == CKLOOP_TREE) {
            int loopTimes = TREE_BCAST_BRANCH>(CmiMyNodeSize()-1)?CmiMyNodeSize()-1:TREE_BCAST_BRANCH;
            //just implicit binary tree
            int pe = CmiMyRank()+1;
            for (int i=0; i<loopTimes; i++, pe++) {
                if (pe >= CmiMyNodeSize()) pe -= CmiMyNodeSize();
                CharmNotifyMsg *one = thisHelper->getNotifyMsg();
                one->ptr = (void *)curLoop;
                envelope *env = UsrToEnv(one);
                env->setObjPtr(thisHelper->ckGetChareID().objPtr);
                CmiPushPE(pe, (void *)(env));
            }
        } else {
            for (int i=CmiMyRank()+1; i<numHelpers; i++) {
                if (!CpvAccessOther(isHelperOn, i)) continue;
                CharmNotifyMsg *one = thisHelper->getNotifyMsg();
                one->ptr = (void *)curLoop;
                envelope *env = UsrToEnv(one);
                env->setObjPtr(thisHelper->ckGetChareID().objPtr);
                //CkPrintf("[%d] sending a msg %p (env=%p) to [%d]\n", CmiMyRank(), one, env, i);
                CmiPushPE(i, (void *)(env));
            }
            for (int i=0; i<CmiMyRank(); i++) {
                if (!CpvAccessOther(isHelperOn, i)) continue;
                CharmNotifyMsg *one = thisHelper->getNotifyMsg();
                one->ptr = (void *)curLoop;
                envelope *env = UsrToEnv(one);
                env->setObjPtr(thisHelper->ckGetChareID().objPtr);
                //CkPrintf("[%d] sending a msg %p (env=%p) to [%d]\n", CmiMyRank(), one, env, i);
                CmiPushPE(i, (void *)(env));
            }
        }
#endif
    } else if (mode == CKLOOP_PTHREAD) {

#if !defined(_WIN32)
        int numThreads = numHelpers-1;
        curLoop = pthdLoop;
        curLoop->set(numChunks, func, lowerRange, upperRange, paramNum, param);
        int numNotices = numThreads;
        if (schedPolicy == CKLOOP_TREE) {
            numNotices = TREE_BCAST_BRANCH>=numThreads?numThreads:TREE_BCAST_BRANCH;
        }
        for (int i=0; i<numNotices; i++) {
            pthread_mutex_lock(allLocks[i]);
            pthread_cond_signal(allConds[i]);
            pthread_mutex_unlock(allLocks[i]);
        }
        //in this mode, it's always synced
        sync = 1;
#endif
    }

    // Call the function on the caller PE before it starts working on chunks
    if (cfunc != NULL) {
      cfunc(cparamNum, cparam);
    }

    if(curLoop) curLoop->stealWork();
    TRACE_BRACKET(CKLOOP_TOTAL_WORK_EVENTID);

    //CkPrintf("[%d]: parallelize func %p with [%d ~ %d] divided into %d chunks using loop=%p\n", CkMyPe(), func, lowerRange, upperRange, numChunks, curLoop);

    TRACE_START(CKLOOP_FINISH_SIGNAL_EVENTID);
    curLoop->waitLoopDone(sync);
    TRACE_BRACKET(CKLOOP_FINISH_SIGNAL_EVENTID);

    //CkPrintf("[%d]: finished parallelize func %p with [%d ~ %d] divided into %d chunks using loop=%p\n", CkMyPe(), func, lowerRange, upperRange, numChunks, curLoop);

    if (type!=CKLOOP_NONE)
        reduce(curLoop->getRedBufs(), redResult, type, numChunks);
    return;
}

CpvStaticDeclare(int, chunkHandler);
CpvStaticDeclare(int, hybridHandler);

void FuncCkLoop::parallelizeFuncHybrid(float staticFraction, HelperFn func, int paramNum, void * param,
                                     int numChunks, int lowerRange,
                                     int upperRange, int sync,
                                     void *redResult, REDUCTION_TYPE type,
                                     CallerFn cfunc,
                                     int cparamNum, void * cparam) {
  double _start; //may be used for tracing
  if (numChunks > MAX_CHUNKS) {
    numChunks = MAX_CHUNKS;
  }

#ifdef CMK_PAPI_PROFILING
  int num_hwcntrs = 2;
  int Events[2] = {PAPI_L2_DCM, PAPI_L3_DCM}; // Obtain L1 and L2 data cache misses.
  long_long values[2];
#endif

#ifdef CMK_PAPI_PROFILING
  if (PAPI_start_counters(Events, num_hwcntrs) != PAPI_OK) CkPrintf("Error with creating event set \n");
#endif

  //for using nodequeue
  TRACE_START(CKLOOP_TOTAL_WORK_EVENTID);
  /*
 CkPrintf("debug [%d]:, CkMyRank=%d: funchybrid. \n", CkMyPe(), CkMyRank());
  FuncSingleHelper *thisHelper = helperPtr[CkMyRank()];
#if ALLOW_MULTIPLE_UNSYNC
  CkPrintf("debug: %d: funchybrid. 1b thisHelper: %d \n", CkMyPe(), (long) thisHelper);
      ConverseNotifyMsg *notifyMsg = thisHelper->getNotifyMsg();
CkPrintf("ptr %d\n", (long) notifyMsg->ptr);
#else
  CkPrintf("debug: %d: funchybrid. 1c \n", CkMyPe());
      ConverseNotifyMsg *notifyMsg = thisHelper->notifyMsg;
#endif
  CkPrintf("debug: %d: funchybrid. 2 \n", CkMyPe());
      curLoop = (CurLoopInfo *)(notifyMsg->ptr);
  */

  //curLoop = new CurLoopInfo(FuncCkLoop::MAX_CHUNKS);

  CurLoopInfo* curLoop = new CurLoopInfo(numHelpers);
  curLoop->set(numChunks, func, lowerRange, upperRange, paramNum, param);
  curLoop->setStaticFraction(staticFraction);
  curLoop->setReductionType(type);
  void** redBufs = curLoop->getRedBufs();
  if(type == CKLOOP_INT_SUM)
    for(int i=0; i<numHelpers; i++)
      *((int*)redBufs[i]) = 0;
  else if((type == CKLOOP_DOUBLE_SUM) || (type == CKLOOP_DOUBLE_MAX))
    for(int i=0; i<numHelpers; i++)
      *((double*)redBufs[i]) = 0.0;
  else if(type == CKLOOP_FLOAT_SUM)
    for(int i=0; i<numHelpers; i++)
      *((float*)redBufs[i]) = 0.0;
  LoopChunkMsg* msg = new LoopChunkMsg;
  msg->loopRec = curLoop;
  CmiSetHandler(msg, CpvAccess(hybridHandler));
  for (int i=1; i<numHelpers; i++) {
    CmiPushPE(i, (void*)msg); // New work coming, send message to other ranks to notify the ranks.
  }
  // Call the function on the caller PE before it starts working on chunks
  if (cfunc != NULL) {
    cfunc(cparamNum, cparam); //user code
  }
  curLoop->doWorkForMyPe();
  // Rank 0 processor continues dequeuing its dynamic chunks from its own task queue.
#if CMK_SMP && CMK_TASKQUEUE
  while(1) {
      void* msg = TaskQueuePop((TaskQueue)CpvAccess(CsdTaskQueue));
      if (msg == NULL) break;
      CmiHandleMessage(msg);
  }
#endif
  // TODO: Should core 0 steal?
  // If so, do randomized steals in a loop until all chunks are done.
  curLoop->waitLoopDoneHybrid(1);
  // NOTE: use 1 in parameter of function waitLoopDone to force exit.

  // CkPrintf("DEBUG: Exiting loop : numChunks = %d \t numStaticChunksCompleted = %d \t numDynamicChunksFired = %d \t numDynamicChunksCompleted = %d \t \n", numChunks, curLoop->numStaticChunksCompleted, curLoop->numDynamicChunksFired, curLoop->numDynamicChunksCompleted);

  TRACE_BRACKET(CKLOOP_TOTAL_WORK_EVENTID);

#ifdef CMK_PAPI_PROFILING
 if (PAPI_stop_counters(values, num_hwcntrs) != PAPI_OK)   CkPrintf("Error with stopping counters!\n");
#endif

#ifdef CMK_PAPI_PROFILING
    if (PAPI_read_counters(values, num_hwcntrs) != PAPI_OK)  CkPrintf("Error with reading counters!\n");
#endif

  //CkPrintf("[%d]: parallelize func %p with [%d ~ %d] divided into %d chunks using loop=%p\n", CkMyPe(), func, lowerRange, upperRange, numChunks, curLoop);
  TRACE_START(CKLOOP_FINISH_SIGNAL_EVENTID);
  TRACE_BRACKET(CKLOOP_FINISH_SIGNAL_EVENTID);
  //CkPrintf("[%d]: finished parallelize func %p with [%d ~ %d] divided into %d chunks using loop=%p\n", CkMyPe(), func, lowerRange, upperRange, numChunks, curLoop);
  if (type!=CKLOOP_NONE)
    reduce(curLoop->getRedBufs(), redResult, type, numHelpers);

  delete curLoop;
  delete msg;
}

#define COMPUTE_REDUCTION(T) {\
    for(int i=0; i<numChunks; i++) {\
     result += *((T *)(redBufs[i])); \
     /*CkPrintf("CkLoop Reduce: %d\n", result);*/ \
    }\
}
#define COMPUTE_REDUCTION_MAX(T) {\
    for(int i=0; i<numChunks; i++) {\
     if( *((T *)(redBufs[i])) > result ) result = *((T *)(redBufs[i])); \
     /*CkPrintf("CkLoop Reduce: %d\n", result);*/ \
    }\
}

void FuncCkLoop::destroyHelpers() {
  int pe = CmiMyRank()+1;
  for (int i = 0; i < numHelpers; i++) {
    if (pe >= CmiMyNodeSize()) pe -= CmiMyNodeSize();
    DestroyNotifyMsg *tmp = new DestroyNotifyMsg;
    envelope *env = UsrToEnv(tmp);
    env->setMsgtype(ForChareMsg);
    env->setEpIdx(CkIndex_FuncSingleHelper::destroyMyself());
    env->setSrcPe(CkMyPe());
    CmiSetHandler(env, _charmHandlerIdx);
    CmiPushPE(pe, (void *)(env));
  }
}

void FuncCkLoop::reduce(void **redBufs, void *redBuf, REDUCTION_TYPE type, int numChunks) {
    switch (type) {
    case CKLOOP_INT_SUM: {
        int result=0;
        COMPUTE_REDUCTION(int)
        *((int *)redBuf) = result;
        break;
    }
    case CKLOOP_FLOAT_SUM: {
        float result=0;
        COMPUTE_REDUCTION(float)
        *((float *)redBuf) = result;
        break;
    }
    case CKLOOP_DOUBLE_SUM: {
        double result=0;
        COMPUTE_REDUCTION(double)
        *((double *)redBuf) = result;
        break;
    }
    case CKLOOP_DOUBLE_MAX: {
        double result=0;
        COMPUTE_REDUCTION_MAX(double)
        *((double *)redBuf) = result;
        break;
    }
    default:
        break;
    }
}

void FuncCkLoop::registerHelper(HelperNotifyMsg* msg) {
  helperPtr[msg->srcRank] = msg->localHelper;
  msg->localHelper->thisCkLoop = this;
  delete msg;
}

void FuncCkLoop::pup(PUP::er &p) {
  p|mode;
  p|numHelpers;
  if (p.isUnpacking()) {
    init(mode, numHelpers);
  }
}

static int _ckloopEP;
CpvStaticDeclare(int, NdhStealWorkHandler);
static void RegisterCkLoopHdlrs() {
    CpvInitialize(int, NdhStealWorkHandler);

    // The following four lines are for the hybrid static/dynamic scheduler.
    CpvInitialize(int, hybridHandler);
    CpvAccess(hybridHandler) = CmiRegisterHandler((CmiHandler)hybridHandlerFunc);
    CpvInitialize(int, chunkHandler);
    CpvAccess(chunkHandler) = CmiRegisterHandler((CmiHandler)executeChunk);

#if CMK_TRACE_ENABLED
    CpvInitialize(envelope*, dummyEnv);
    CpvAccess(dummyEnv) = envelope::alloc(ForChareMsg,0,0); //Msgtype is the same as the one used for TRACE_BEGIN_EXECUTED_DETAILED
#endif
    CpvAccess(NdhStealWorkHandler) = CmiRegisterHandler((CmiHandler)SingleHelperStealWork);
      if(CkMyRank()==0) {
        int _ckloopMsg = CkRegisterMsg("ckloop_converse_msg", 0, 0, 0, 0);
        int _ckloopChare = CkRegisterChare("ckloop_converse_chare", 0, TypeInvalid);
        CkRegisterChareInCharm(_ckloopChare);
        _ckloopEP = CkRegisterEp("CkLoop", (CkCallFnPtr)SingleHelperStealWork, _ckloopMsg, _ckloopChare, 0+CK_EP_INTRINSIC);
      }
}

extern int _charmHandlerIdx;

FuncSingleHelper::FuncSingleHelper() {
    CmiAssert(globalCkLoop!=NULL);
    thisCkLoop = globalCkLoop;
    totalHelpers = globalCkLoop->numHelpers;
    funcckproxy = globalCkLoop->thisProxy;
    schedPolicy = globalCkLoop->schedPolicy;

    createNotifyMsg();

    globalCkLoop->helperPtr[CkMyRank()] = this;
}

void FuncSingleHelper::createNotifyMsg() {
#if USE_CONVERSE_NOTIFICATION
    notifyMsgBufSize = TASK_BUFFER_SIZE;
#else
    notifyMsgBufSize = TASK_BUFFER_SIZE*totalHelpers;
#endif

    nextFreeNotifyMsg = 0;
#if USE_CONVERSE_NOTIFICATION
    notifyMsg = (ConverseNotifyMsg *)malloc(sizeof(ConverseNotifyMsg)*notifyMsgBufSize);
    for (int i=0; i<notifyMsgBufSize; i++) {
        ConverseNotifyMsg *tmp = notifyMsg+i;
        if (schedPolicy == CKLOOP_NODE_QUEUE || schedPolicy == CKLOOP_TREE) {
            tmp->srcRank = CmiMyRank();
        } else {
            tmp->srcRank = -1;
        }
        tmp->ptr = (void *)(new CurLoopInfo(FuncCkLoop::MAX_CHUNKS));
        CmiSetHandler(tmp, CpvAccess(NdhStealWorkHandler));
    }
#else
    nextFreeTaskBuffer = 0;
    notifyMsg = (CharmNotifyMsg **)malloc(sizeof(CharmNotifyMsg *)*notifyMsgBufSize);
    for (int i=0; i<notifyMsgBufSize; i++) {
        CharmNotifyMsg *tmp = new(sizeof(int)*8)CharmNotifyMsg; //allow msg priority bits
        notifyMsg[i] = tmp;
        if (schedPolicy == CKLOOP_NODE_QUEUE || schedPolicy == CKLOOP_TREE) {
            tmp->srcRank = CmiMyRank();
        } else {
            tmp->srcRank = -1;
        }
        tmp->ptr = NULL;
        envelope *env = UsrToEnv(tmp);
        env->setMsgtype(ForChareMsg);
        env->setEpIdx(CkIndex_FuncSingleHelper::stealWork(NULL));
        env->setSrcPe(CkMyPe());
        CmiSetHandler(env, _charmHandlerIdx);
        //env->setObjPtr has to be called when a notification msg is sent
    }
    taskBuffer = (CurLoopInfo **)malloc(sizeof(CurLoopInfo *)*TASK_BUFFER_SIZE);
    for (int i=0; i<TASK_BUFFER_SIZE; i++) {
        taskBuffer[i] = new CurLoopInfo(FuncCkLoop::MAX_CHUNKS);
    }
#endif
}

void FuncSingleHelper::stealWork(CharmNotifyMsg *msg) {
#if !USE_CONVERSE_NOTIFICATION
    int srcRank = msg->srcRank;
    CurLoopInfo *loop = (CurLoopInfo *)msg->ptr;
    if (srcRank >= 0) {
        //means using tree-broadcast to send the notification msg
        int relPE = CmiMyRank()-msg->srcRank;
        if (relPE<0) relPE += CmiMyNodeSize();

        //CmiPrintf("Rank[%d]: got msg from src %d with relPE %d\n", CmiMyRank(), msg->srcRank, relPE);
        relPE=relPE*TREE_BCAST_BRANCH+1;
        for (int i=0; i<TREE_BCAST_BRANCH; i++, relPE++) {
            if (relPE >= CmiMyNodeSize()) break;
            int pe = (relPE + msg->srcRank)%CmiMyNodeSize();
            if (!CpvAccessOther(isHelperOn, pe)) continue;
            //CmiPrintf("Rank[%d]: send msg to dst %d (relPE: %d) from src %d\n", CmiMyRank(), pe, relPE, msg->srcRank);
            CharmNotifyMsg *newone = getNotifyMsg();
            newone->ptr = (void *)loop;
            envelope *env = UsrToEnv(newone);
            env->setObjPtr(thisCkLoop->helperPtr[pe]->ckGetChareID().objPtr);
            CmiPushPE(pe, (void *)env);
        }
    }
    loop->stealWork();
#endif
}

void SingleHelperStealWork(ConverseNotifyMsg *msg) {
    int srcRank = msg->srcRank;
    CurLoopInfo *loop = (CurLoopInfo *)msg->ptr;

    if (srcRank >= 0 && !loop->isFree()) {
        if (msg->queueID == NODE_Q && globalCkLoop->getSchedPolicy() == CKLOOP_NODE_QUEUE ) {
            msg->queueID = PE_Q;
            int myRank = CmiMyRank();
            if ( srcRank == myRank ) return;  // already done
            /* We don't push a message to PE where its helper bit is disabled. */
            for (int i=srcRank+1; i<CmiMyNodeSize(); i++) {
                if ( i == myRank ) continue;
                if (!CpvAccessOther(isHelperOn, i)) continue;
                CmiPushPE(i, (void *)(msg));
            }
            for (int i=0; i<srcRank; i++) {
                if ( i == myRank ) continue;
                if (!CpvAccessOther(isHelperOn, i)) continue;
                CmiPushPE(i, (void *)(msg));
            }
        }
        else if (globalCkLoop->getSchedPolicy() == CKLOOP_TREE) {
          int relPE = CmiMyRank()-msg->srcRank;
          if (relPE<0) relPE += CmiMyNodeSize();

          //means using tree-broadcast to send the notification msg
          //CmiPrintf("Rank[%d]: got msg from src %d with relPE %d\n", CmiMyRank(), msg->srcRank, relPE);
          relPE=relPE*TREE_BCAST_BRANCH+1;
          for (int i=0; i<TREE_BCAST_BRANCH; i++, relPE++) {
              if (relPE >= CmiMyNodeSize()) break;
              int pe = (relPE + msg->srcRank)%CmiMyNodeSize();
              //CmiPrintf("Rank[%d]: send msg to dst %d (relPE: %d) from src %d\n", CmiMyRank(), pe, relPE, msg->srcRank);
              /* This message is passed to children in the tree even when their helper bit is off. */
              CmiPushPE(pe, (void *)msg);
          }
        }
    }
    /* If this PE's helper bit is off, we should not steal work.
     * This handles NODE_QUEUE / TREE cases and cases when helper bit is off
     * after the master in NODE_QUEUE or parent in TREE pushes this message to this PE's queue
     * */
    if (!CpvAccess(isHelperOn)) return;
#if CMK_TRACE_ENABLED
    unsigned int event = msg->eventID;
    _TRACE_BEGIN_EXECUTE_DETAILED(event, ForChareMsg, _ckloopEP,
      CkNodeFirst(CkMyNode())+srcRank, sizeof(ConverseNotifyMsg), NULL, NULL);
#endif
    loop->stealWork();
#if CMK_TRACE_ENABLED
    _TRACE_END_EXECUTE();
#endif
}

void CurLoopInfo::stealWork() {
    //indicate the current work hasn't been initialized
    //or the old work has finished.
    CmiLock(loop_info_inited_lock);
    if (inited == 0) {
      CmiUnlock(loop_info_inited_lock);
      return;
    }

    int nextChunkId = getNextChunkIdx();
    if (nextChunkId >= numChunks) {
      CmiUnlock(loop_info_inited_lock);
      return;
    }

    CmiUnlock(loop_info_inited_lock);
    int execTimes = 0;

    int first, last;
    int unit = (upperIndex-lowerIndex+1)/numChunks;
    int remainder = (upperIndex-lowerIndex+1)-unit*numChunks;
    int markIdx = remainder*(unit+1);

    while (nextChunkId < numChunks) {
      if (nextChunkId < remainder) {
        first = lowerIndex+(unit+1)*nextChunkId;
        last = first+unit;
      } else {
        first = lowerIndex+(nextChunkId - remainder)*unit + markIdx;
        last = first+unit-1;
      }

      if (first < lowerIndex || first > upperIndex || last < lowerIndex || last > upperIndex) {
        CkPrintf("Error in CurLoopInfo::stealWork() node %d pe %d lowerIndex %d upperIndex %d numChunks %d first %d last %d\n",
          CkMyNode(), CkMyPe(), lowerIndex, upperIndex, numChunks, first, last);
        CkAbort("Indices of CkLoop incorrect. There maybe a race condition!\n");
      }

        fnPtr(first, last, redBufs[nextChunkId], paramNum, param);
        execTimes++;
        nextChunkId = getNextChunkIdx();
    }
    reportFinished(execTimes);
}

//======================================================================//
//   End of functions related with FuncSingleHelper                     //
//======================================================================//

CProxy_FuncCkLoop CkLoop_Init(int numThreads) {
    int mode;
#if CMK_SMP
    mode = CKLOOP_USECHARM;
#if USE_CONVERSE_NOTIFICATION
    CkPrintf("CkLoopLib is used in SMP with simple dynamic scheduling (converse-level notification)\n");
#else
    CkPrintf("CkLoopLib is used in SMP with simple dynamic scheduling (charm-level notification)\n");
#endif
#elif defined(_WIN32)
    mode = CKLOOP_NOOP;
#else
    mode = CKLOOP_PTHREAD;
    CkPrintf("CkLoopLib is used with extra %d pthreads via a simple dynamic scheduling\n", numThreads);
    CmiAssert(numThreads>0);
#endif
    return CProxy_FuncCkLoop::ckNew(mode, numThreads);
}

void CkLoop_Exit(CProxy_FuncCkLoop ckLoop) {
    ckLoop.exit();
}

void hybridHandlerFunc(LoopChunkMsg *msg)
{
  CurLoopInfo* loop = msg->loopRec;
  loop->doWorkForMyPe(); // Do or enqueue work for the current loop for my PE.
}

void CurLoopInfo::doWorkForMyPe() {
  int numHelpers = CmiMyNodeSize();
  int myRank = CmiMyRank();
  if (upperIndex-lowerIndex < numHelpers)
    numHelpers = upperIndex-lowerIndex;
  int myStaticBegin = lowerIndex + myRank*(upperIndex - lowerIndex)/numHelpers;

  int myDynamicBegin = myStaticBegin + ((upperIndex - lowerIndex)/numHelpers)*staticFraction;
  int lastDynamic = lowerIndex + (myRank+1)*(upperIndex - lowerIndex)/numHelpers;
  if(lastDynamic > upperIndex) lastDynamic = upperIndex; // for the last PE.

  int i, j = 0;

  // TODO: make numChunks smaller as needed.

  chunkSize = (upperIndex - lowerIndex)/numChunks;
  if(chunkSize == 0) chunkSize = 1;
  LoopChunkMsg* msgBlock = new LoopChunkMsg[1 + (lastDynamic - myDynamicBegin)/chunkSize];
  // TODO: msgBlock should be freed when the whole CkLoop loop is done and all stealers have finished using it.

  // Enqueue dynamic work first, because before a PE starts work on static, other PE's should be ready to steal from its dynamic.
  // TODO: the order of enqueues should be reversed since the task queue is run as a stack.
  /* for (i=myDynamicBegin, j=0; i<lastDynamic; i+=chunkSize, j++)
   {
     LoopChunkMsg* msg = (LoopChunkMsg*)(&(msgBlock[j]));
     msg->startIndex = i;
      msg->endIndex = i + chunkSize > lastDynamic ? lastDynamic : i+chunkSize;
      msg->loopRec = this;
      CmiSetHandler(msg, CpvAccess(chunkHandler));
      CsdTaskEnqueue(msg);
      } */

  //TODO: test with    400  404

  // TODO: size : 402 / 404 4 threads.
  // TODO: size:
#if CMK_SMP && CMK_TASKQUEUE
  for (i=lastDynamic, j=0; i>myDynamicBegin; i-=chunkSize, j++)
    {
      LoopChunkMsg* msg = (LoopChunkMsg*)(&(msgBlock[j]));
      //  msg->startIndex = i;
      // msg->endIndex = i + chunkSize > lastDynamic ? lastDynamic : i+chunkSize;
      msg->endIndex = i;
      msg->startIndex = i - chunkSize  < myDynamicBegin ? myDynamicBegin  :  i - chunkSize;
      msg->loopRec = this;
      CmiSetHandler(msg, CpvAccess(chunkHandler));
      CsdTaskEnqueue(msg);
    }
#endif

  double _start; //may be used for tracing
  TRACE_START(CKLOOP_STATIC_CHUNK_WORK);
  // do PE's static part
  double x = 0.0;
  fnPtr(myStaticBegin, myDynamicBegin, (void*) &x, paramNum, param);
  TRACE_BRACKET(CKLOOP_STATIC_CHUNK_WORK);
  // TODO: the code block below may not be needed since the hybrid scheduler doesn't use finishFlag.
  /* int tmp  = (myDynamicBegin - myStaticBegin)/chunkSize;
     finishFlag+= tmp; */

  //CkPrintf("DEBUG: chunk [%d:\t %d] function returned %f for reduction\n" , myStaticBegin, myDynamicBegin, x);

  localReduce(x, type);
  numDynamicChunksFired += j;
  numStaticRegionsCompleted++;
}

void executeChunk(LoopChunkMsg *msg) {
  double _start; //may be used for tracing
  TRACE_START(CKLOOP_DYNAMIC_CHUNK_WORK);
  // This is the function executed when a task is dequeued from the task queue in the hybrid scheduler.
  CurLoopInfo* linfo = msg->loopRec;
  linfo->runChunk(msg->startIndex, msg->endIndex);
  TRACE_BRACKET(CKLOOP_DYNAMIC_CHUNK_WORK);
  /* free(msg); */ // Free not needed since we are using a block allocation in doWorkForMyPe().
}

void CkLoop_Parallelize(HelperFn func,
                            int paramNum, void * param,
                            int numChunks, int lowerRange, int upperRange,
                            int sync,
                            void *redResult, REDUCTION_TYPE type,
                            CallerFn cfunc,
                            int cparamNum, void* cparam) {
    if ( numChunks > upperRange - lowerRange + 1 ) numChunks = upperRange - lowerRange + 1;
    globalCkLoop->parallelizeFunc(func, paramNum, param, numChunks, lowerRange,
        upperRange, sync, redResult, type, cfunc, cparamNum, cparam);
}

/**
*
* Author: Vivek Kale
* Contributors: Harshitha Menon, Karthik Senthil Kumar
*
* The CkLoop_Hybrid library is a mode of CkLoop that incorporates specific
* adaptive scheduling strategies aimed at providing a tradeoff between dynamic
* load balance and spatial locality. It is used in a build of Charm++ where all
* chares are placed on core 0 of each node (called the drone-mode, or
* all-drones-mode). It incorporates a strategy called staggered static-dynamic
* scheduling (from dissertation work of Vivek Kale). The iteration space is
* first tentatively divided approximately equally to all available PEs. Each
* PE's share of the iteration space is divided into a static portion, specified by
* the staticFraction parameter below, and the remaining dynamic portion. The dynamic
* portion of a PE is divided into chunks of specified chunksize, and enqueued in
* the task-queue associated with that PE. Each PE works on its static portion,
* and then on its own task queue (thus preserving spatial locality, as well as
* persistence of allocations across outer iterations), and after finishing that,
* steals work from other PE’s task queues.
*
* CkLoop_Hybrid support requires the SMP mode of Charm++ and the additional flags
* --enable-drone-mode and --enable-task-queue to be passed as build options when
* Charm++ is built.
**/

void CkLoop_ParallelizeHybrid(float staticFraction,
             HelperFn func,
             int paramNum, void * param,
             int numChunks, int lowerRange, int upperRange,
             int sync,
             void *redResult, REDUCTION_TYPE type,
             CallerFn cfunc,
             int cparamNum, void* cparam) {
#if CMK_SMP && CMK_TASKQUEUE
  if (0 != CkMyRank()) CkAbort("CkLoop_ParallelizeHybrid() must be called from rank 0 PE on a node.\n");
  if (numChunks > upperRange - lowerRange + 1) numChunks = upperRange - lowerRange + 1;
  // Not doing anything with loop history for now.
  globalCkLoop->parallelizeFuncHybrid(staticFraction, func, paramNum, param, numChunks, lowerRange, upperRange, sync, redResult, type, cfunc, cparamNum, cparam);
#else
  globalCkLoop->parallelizeFunc(func, paramNum, param, numChunks, lowerRange,
        upperRange, sync, redResult, type, cfunc, cparamNum, cparam);
#endif
}

void CkLoop_SetSchedPolicy(CkLoop_sched schedPolicy) {
  globalCkLoop->setSchedPolicy(schedPolicy);
  std::atomic_thread_fence(std::memory_order_release);
}

void CkLoop_DestroyHelpers() {
  globalCkLoop->destroyHelpers();
}
#include "CkLoop.def.h"
