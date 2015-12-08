#ifndef _CKLOOP_H
#define _CKLOOP_H
#include <assert.h>

#include "charm++.h"
#include "CkLoopAPI.h"

#define USE_TREE_BROADCAST_THRESHOLD 8
#define TREE_BCAST_BRANCH (4)
#define CACHE_LINE_SIZE 64
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

class FuncSingleHelper;

class CurLoopInfo {
    friend class FuncSingleHelper;

private:
    volatile int curChunkIdx;
    int numChunks;
    HelperFn fnPtr;
    int lowerIndex;
    int upperIndex;
    int paramNum;
    void *param;
    //limitation: only allow single variable reduction of size numChunks!!!
    void **redBufs;
    char *bufSpace;

    volatile int finishFlag;

    //a tag to indicate whether the task for this new loop has been inited
    //this tag is needed to prevent other helpers to run the old task
    int inited;

public:
    CurLoopInfo(int maxChunks):numChunks(0),fnPtr(NULL), lowerIndex(-1), upperIndex(0),
            paramNum(0), param(NULL), curChunkIdx(-1), finishFlag(0), redBufs(NULL), bufSpace(NULL), inited(0) {
        redBufs = new void *[maxChunks];
        bufSpace = new char[maxChunks * CACHE_LINE_SIZE];
        for (int i=0; i<maxChunks; i++) redBufs[i] = (void *)(bufSpace+i*CACHE_LINE_SIZE);
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

    void waitLoopDone(int sync) {
        //while(!__sync_bool_compare_and_swap(&finishFlag, numChunks, 0));
        if (sync) while (finishFlag!=numChunks);
        //finishFlag = 0;
        CmiLock(loop_info_inited_lock);
        inited = 0;
        CmiUnlock(loop_info_inited_lock);
    }
    int getNextChunkIdx() {
#if defined(_WIN32)
#if CMK_SMP
        int next_chunk_id;
        CmiLock(cmiMemoryLock);
        curChunkIdx=curChunkIdx+1;
        next_chunk_id = curChunkIdx;
        CmiUnlock(cmiMemoryLock);
        return next_chunk_id;
#else
        curChunkIdx++;
        return curChunkIdx;
#endif
#else
        return __sync_add_and_fetch(&curChunkIdx, 1);
#endif
    }
    void reportFinished(int counter) {
        if (counter==0) return;
#if defined(_WIN32)
#if CMK_SMP
        CmiLock(cmiMemoryLock);
        finishFlag=finishFlag+counter;
        CmiUnlock(cmiMemoryLock);
#else
        finishFlag=finishFlag+counter;
#endif
#else
        __sync_add_and_fetch(&finishFlag, counter);
#endif
    }

    int isFree() {
        return finishFlag == numChunks;
    }

    void **getRedBufs() {
        return redBufs;
    }

    void stealWork();
};

/* FuncCkLoop is a nodegroup object */

typedef struct converseNotifyMsg {
    char core[CmiMsgHeaderSizeBytes];
    int srcRank;
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
    int useTreeBcast;

public:
    FuncCkLoop(int mode_, int numThreads_);

    FuncCkLoop(CkMigrateMessage *m);

    ~FuncCkLoop() {
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
    int needTreeBcast() {
        return useTreeBcast;
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
    void destroyHelpers();
    void reduce(void **redBufs, void *redBuf, REDUCTION_TYPE type, int numChunks);
    void pup(PUP::er &p);
};

void SingleHelperStealWork(ConverseNotifyMsg *msg);

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
    int useTreeBcast;

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
