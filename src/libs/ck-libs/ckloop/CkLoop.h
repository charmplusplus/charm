#ifndef _CKLOOP_H
#define _CKLOOP_H
#include <assert.h>

#include "converse.h"
#include "taskqueue.h"
#include "charm++.h"
#include "CkLoopAPI.h"
#include <atomic>
#define USE_TREE_BROADCAST_THRESHOLD 8
#define TREE_BCAST_BRANCH (4)

/* 1. Using converse-level msg, then the msg is always of highest priority.
 * And the notification msg comes from the singlehelper where the loop parallelization
 * is initiated.
 *
 * 2. Using charm-level msg, then the msg could be set with different priorities.
 * However, the notification msg comes from the singlehelper where the parallelized
 * loop is executed.
 *
 * */
#define USE_CONVERSE_NOTIFICATION 1

 CmiNodeLock loop_info_inited_lock;

#if CMK_TRACE_ENABLED
CpvDeclare(envelope*, dummyEnv);
#endif
CpvCExtern(int, isHelperOn);
class FuncSingleHelper;

class CurLoopInfo {
    friend class FuncSingleHelper;

private:
    float staticFraction;
    std::atomic<int> curChunkIdx;
    int numChunks;
    int chunkSize;
    REDUCTION_TYPE type; // only used in hybrid mode
    HelperFn fnPtr;
    int lowerIndex;
    int upperIndex;
    int paramNum;
    void *param;
    //limitation: only allow single variable reduction of size numChunks!!!
    void **redBufs;
    char *bufSpace;

    std::atomic<int> finishFlag;

    //a tag to indicate whether the task for this new loop has been inited
    //this tag is needed to prevent other helpers to run the old task
    std::atomic<int> inited;

    // For Hybrid mode:
    std::atomic<int> numStaticRegionsCompleted{0};
    std::atomic<int> numDynamicChunksCompleted{0};
    std::atomic<int> numDynamicChunksFired{0};

public:
    CurLoopInfo(int maxChunks):numChunks(0),fnPtr(NULL), lowerIndex(-1), upperIndex(0),
            paramNum(0), param(NULL), curChunkIdx(-1), finishFlag(0), redBufs(NULL), bufSpace(NULL), inited(0) {
        redBufs = new void *[maxChunks];
        bufSpace = new char[maxChunks * CMI_CACHE_LINE_SIZE];
        for (int i=0; i<maxChunks; i++) redBufs[i] = (void *)(bufSpace+i*CMI_CACHE_LINE_SIZE);
    }

    ~CurLoopInfo() {
        delete [] redBufs;
        delete [] bufSpace;
    }

    void set(int nc, HelperFn f, int lIdx, int uIdx, int numParams, void *p) {        /*
      * The locking is to handle a rare data-racing case here. The current loop is
      * about to finish (just before setting inited to 0; A helper (say B)
      * just enters the stealWork and passes the inited check. The helper
      * (say A) is very fast, and starts the next loop, and happens enter
      * into the middle of this function. Then helper B will face corrupted
      * task info as it is trying to execute the old loop task!
      * In reality for user cases, this case happens very rarely!! -Chao Mei
      */
        CmiLock(loop_info_inited_lock);
        numChunks = nc;
        fnPtr = f;
        lowerIndex = lIdx;
        upperIndex = uIdx;
        paramNum = numParams;
        param = p;
        curChunkIdx = -1;
        finishFlag = 0;
        //needs to be set last
        inited = 1;
        CmiUnlock(loop_info_inited_lock);
    }

    void setReductionType(REDUCTION_TYPE p) {
      type = p;
    }

    void setStaticFraction(float _staticFraction) {
      staticFraction = _staticFraction;
    }

    #define LOCALSUM(T) *((T*) redBufs[CmiMyRank()]) += (T) x;

    void localReduce(double x, REDUCTION_TYPE type) {
      // TODO: add more data types here
      switch(type)
        {
        case CKLOOP_INT_SUM: {
          LOCALSUM(int)
        break;
        }
        case CKLOOP_FLOAT_SUM: {
          LOCALSUM(float)
        break;
        }
        case CKLOOP_DOUBLE_SUM: {
          LOCALSUM(double)
        break;
        }
        case CKLOOP_DOUBLE_MAX: {
          if( *((double *)(redBufs[CmiMyRank()])) < x ) *((double *)(redBufs[CmiMyRank()])) = x;
          break;
        }
        default:
          break;
        }
    }

     //This function is called from hybrid scheduler functions.
     void runChunk(int sInd, int eInd) {
       int myRank = CmiMyRank();
       int numHelpers = CmiMyNodeSize();
       int nextPesStaticBegin = lowerIndex + (myRank+1)*(upperIndex - lowerIndex)/numHelpers;
       double x;  // Just allocating an 8-byte scalar.
       fnPtr(sInd, eInd, (void*) &x, paramNum, param); // Calling user's function to do one chunk of iterations.

       // "Add" *x to *(redBufs[myRank]). The meaning of "Add" depends on the type.
       localReduce(x, type);
       numDynamicChunksCompleted++;
     }

    void waitLoopDone(int sync) {
        //while(!__sync_bool_compare_and_swap(&finishFlag, numChunks, 0));
        if (sync) while (finishFlag.load(std::memory_order_relaxed)!=numChunks);
        std::atomic_thread_fence(std::memory_order_acquire);
       //finishFlag = 0;
        CmiLock(loop_info_inited_lock);
        inited = 0;
        CmiUnlock(loop_info_inited_lock);
    }

    void waitLoopDoneHybrid(int sync) {
        int count = 0;
        int numHelpers = CmiMyNodeSize();
        if (sync)
        while ((numStaticRegionsCompleted != numHelpers) || (numDynamicChunksCompleted != numDynamicChunksFired))
        {
            // debug print in case the function is stuck in an infinite loop.
            // count++; if ((count % 100000) == 0);
            // printf("DEBUG: nsrc= %d \t ndcf = %d \t ndcc = %d \n" , (int) numStaticRegionsCompleted, (int) numDynamicChunksFired, (int) numDynamicChunksCompleted);
        };
        CmiLock(loop_info_inited_lock);
        inited = 0;
        CmiUnlock(loop_info_inited_lock);
    }

    int getNextChunkIdx() {
        return curChunkIdx.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    void reportFinished(int counter) {
        if (counter==0) return;
        finishFlag.fetch_add(counter, std::memory_order_release);
    }

    int isFree() {
      int fin = finishFlag.load(std::memory_order_acquire) == numChunks;
      return fin;
    }

    void **getRedBufs() {
        return redBufs;
    }

    void stealWork();
    void doWorkForMyPe();
};

// To be used for hybridHandler and chunkHandler.
typedef struct loopChunkMsg
{
  char hdr[CmiMsgHeaderSizeBytes];
  CurLoopInfo* loopRec;
  int startIndex;
  int endIndex;
} LoopChunkMsg;

/* FuncCkLoop is a nodegroup object */

typedef enum CkLoop_queueID { NODE_Q=0, PE_Q} CkLoop_queueID;

typedef struct converseNotifyMsg {
    char core[CmiMsgHeaderSizeBytes];
    int srcRank;
    unsigned int eventID;
    CkLoop_queueID queueID; /* indiciate which queue this message come from
                    (e.g Node/PE Queue) */
    void *ptr;
} ConverseNotifyMsg;

class CharmNotifyMsg: public CMessage_CharmNotifyMsg {
public:
    int srcRank;
    void *ptr; //the loop info
};

class HelperNotifyMsg: public CMessage_HelperNotifyMsg {
public:
  int srcRank;
  FuncSingleHelper *localHelper;
};

class DestroyNotifyMsg: public CMessage_DestroyNotifyMsg {};

class FuncCkLoop : public CBase_FuncCkLoop {
    friend class FuncSingleHelper;

public:
    static int MAX_CHUNKS;
private:
    int mode;

    int numHelpers; //in pthread mode, the counter includes itself
    FuncSingleHelper **helperPtr; /* ptrs to the FuncSingleHelpers it manages */
    CkLoop_sched schedPolicy;

public:
    FuncCkLoop(int mode_, int numThreads_);

    FuncCkLoop(CkMigrateMessage *m);

    ~FuncCkLoop() {
#if CMK_TRACE_ENABLED
      int i;
      for (i = 0; i < CkMyNodeSize(); i++)
        CmiFree(CpvAccessOther(dummyEnv,i));
#endif
      CmiDestroyLock(loop_info_inited_lock);
        delete [] helperPtr;
    }

    // This entry method is used during restart. When the helper chares are
    // restarted, the FuncCkLoop node group need not be constructed. So the
    // helper chares send message to the node proxy on their node to register
    // themselves.
    void registerHelper(HelperNotifyMsg* msg);

    void createPThreads();
    void exit();
    void init(int mode_, int numThreads_);

    int getNumHelpers() {
        return numHelpers;
    }
    CkLoop_sched getSchedPolicy() {
        return schedPolicy;
    }
    void setSchedPolicy(CkLoop_sched schedPolicy) {
#if !CMK_NODE_QUEUE_AVAILABLE && CMK_ERROR_CHECKING
      if (schedPolicy == CKLOOP_NODE_QUEUE)
        CkAbort("SchedPolicy, CKLOOP_NODE_QUEUE is not available on this environment\n");
#endif
      this->schedPolicy = schedPolicy;
    }
    void parallelizeFunc(HelperFn func, /* the function that finishes a partial work on another thread */
                         int paramNum, void * param, /* the input parameters for the above func */
                         int numChunks, /* number of chunks to be partitioned */
                         int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */
                         int sync=1, /* whether the flow will continue until all chunks have finished */
                         void *redResult=NULL, REDUCTION_TYPE type=CKLOOP_NONE, /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
                         CallerFn cfunc=NULL, /* the caller PE will call this function before starting to work on the chunks */
                         int cparamNum=0, void* cparam=NULL /* the input parameters to the above function */
                        );
    void parallelizeFuncHybrid(float sf,
               HelperFn func, /* the function that finishes a partial work on another thread */
               int paramNum, void * param, /* the input parameters for the above func */
               int numChunks, /* number of chunks to be partitioned. Note that some of the chunks may be subsumed into a large static section. */
               int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */
               int sync=1, /* whether the flow will continue until all chunks have finished */
               void *redResult=NULL, REDUCTION_TYPE type=CKLOOP_NONE, /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
               CallerFn cfunc=NULL, /* the caller PE will call this function before starting to work on the chunks */
               int cparamNum=0, void* cparam=NULL /* the input parameters to the above function */
               );
    void destroyHelpers();
    void reduce(void **redBufs, void *redBuf, REDUCTION_TYPE type, int numChunks);
    void pup(PUP::er &p);
};

void executeChunk(LoopChunkMsg* msg);
void SingleHelperStealWork(ConverseNotifyMsg *msg);
void hybridHandlerFunc(LoopChunkMsg *msg);

/* FuncSingleHelper is a chare located on every core of a node */
//allowing arbitrary combination of sync and unsync parallelizd loops
#define TASK_BUFFER_SIZE (3)
class FuncSingleHelper: public CBase_FuncSingleHelper {
    friend class FuncCkLoop;
private:
    int totalHelpers;
    int notifyMsgBufSize;

    FuncCkLoop *thisCkLoop;
    CProxy_FuncCkLoop funcckproxy;
    CkLoop_sched schedPolicy;

#if USE_CONVERSE_NOTIFICATION
    //this msg is shared across all SingleHelpers
    ConverseNotifyMsg *notifyMsg;
#else
    //acted as a msg buffer for charm-level notification msgs sent to other
    //SingleHelpers. At each sending,
    //1. the msg destination chare (SingleHelper) has to be set.
    //2. the associated loop info has to be set.
    CharmNotifyMsg **notifyMsg;
    CurLoopInfo **taskBuffer;
    int nextFreeTaskBuffer;
#endif
    int nextFreeNotifyMsg;

public:
    FuncSingleHelper();

    ~FuncSingleHelper() {
#if USE_CONVERSE_NOTIFICATION
        for (int i=0; i<notifyMsgBufSize; i++) {
            ConverseNotifyMsg *tmp = notifyMsg+i;
            CurLoopInfo *loop = (CurLoopInfo *)(tmp->ptr);
            delete loop;
        }
        free(notifyMsg);
#else
        for (int i=0; i<notifyMsgBufSize; i++) delete notifyMsg[i];
        free(notifyMsg);
        for (int i=0; i<TASK_BUFFER_SIZE; i++) delete taskBuffer[i];
        free(taskBuffer);
#endif
    }
#if USE_CONVERSE_NOTIFICATION
    ConverseNotifyMsg *getNotifyMsg() {
        while (1) {
            ConverseNotifyMsg *cur = notifyMsg+nextFreeNotifyMsg;
            CurLoopInfo *loop = (CurLoopInfo *)(cur->ptr);
            nextFreeNotifyMsg = (nextFreeNotifyMsg+1)%notifyMsgBufSize;
            if (loop->isFree()) return cur;
        }
        return NULL;
    }
#else
    CharmNotifyMsg *getNotifyMsg() {
        while (1) {
            CharmNotifyMsg *cur = notifyMsg[nextFreeNotifyMsg];
            CurLoopInfo *loop = (CurLoopInfo *)(cur->ptr);
            nextFreeNotifyMsg = (nextFreeNotifyMsg+1)%notifyMsgBufSize;
            if (loop==NULL || loop->isFree()) return cur;
        }
        return NULL;
    }
    CurLoopInfo *getNewTask() {
        while (1) {
            CurLoopInfo *cur = taskBuffer[nextFreeTaskBuffer];
            nextFreeTaskBuffer = (nextFreeTaskBuffer+1)%TASK_BUFFER_SIZE;
            if (cur->isFree()) return cur;
        }
        return NULL;
    }
#endif

    void stealWork(CharmNotifyMsg *msg);
    void destroyMyself() {
      delete this;
    }

    FuncSingleHelper(CkMigrateMessage *m) : CBase_FuncSingleHelper(m) {}

 private:
    void createNotifyMsg();

};

#endif
