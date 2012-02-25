#ifndef _NODEHELPER_H
#define _NODEHELPER_H
#include <assert.h>

#include "charm++.h"
#include "NodeHelperAPI.h"

#define USE_TREE_BROADCAST_THRESHOLD 8
#define TREE_BCAST_BRANCH (4)
#define CACHE_LINE_SIZE 64

class FuncSingleHelper;

class CurLoopInfo{
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
    paramNum(0), param(NULL), curChunkIdx(-1), finishFlag(0), redBufs(NULL), bufSpace(NULL), inited(0) 
	{
		redBufs = new void *[maxChunks];
		bufSpace = new char[maxChunks * CACHE_LINE_SIZE];
        for(int i=0; i<maxChunks; i++) redBufs[i] = (void *)(bufSpace+i*CACHE_LINE_SIZE);
	}
    
    ~CurLoopInfo() { 
		delete [] redBufs; 
		delete [] bufSpace;
	}
    
    void set(int nc, HelperFn f, int lIdx, int uIdx, int numParams, void *p){        /*
      * WARNING: there's a rare data-racing case here. The current loop is
      * about to finish (just before setting inited to 0; A helper (say B) 
      * just enters the stealWork and passes the inited check. The helper 
      * (say A) is very fast, and starts the next loop, and happens enter
      * into the middle of this function. Then helper B will face corrupted
      * task info as it is trying to execute the old loop task!
      * In reality for user cases, this case happens very rarely!! -Chao Mei
      */
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
    }
      
    void waitLoopDone(int sync){
        //while(!__sync_bool_compare_and_swap(&finishFlag, numChunks, 0));
	if(sync) while(finishFlag!=numChunks);
	//finishFlag = 0;
        inited = 0;
    }
    int getNextChunkIdx(){
        return __sync_add_and_fetch(&curChunkIdx, 1);
    }
    void reportFinished(int counter){
	if(counter==0) return;
        __sync_add_and_fetch(&finishFlag, counter);
    }
    
    int isFree() { return finishFlag == numChunks; }
    
	void **getRedBufs() { return redBufs; }
	
    void stealWork();
};

/* FuncNodeHelper is a nodegroup object */

typedef struct converseNotifyMsg{
    char core[CmiMsgHeaderSizeBytes];
    int srcRank;
    void *ptr;
}ConverseNotifyMsg;

class FuncNodeHelper : public CBase_FuncNodeHelper {
    friend class FuncSingleHelper;
	
public:
    static int MAX_CHUNKS;
private:    
    int numHelpers;    
    FuncSingleHelper **helperPtr; /* ptrs to the FuncSingleHelpers it manages */
	int useTreeBcast;
    
public:
	FuncNodeHelper();
    ~FuncNodeHelper() {
        delete [] helperPtr;
    }
    
    void parallelizeFunc(HelperFn func, /* the function that finishes a partial work on another thread */
                        int paramNum, void * param, /* the input parameters for the above func */
                        int numChunks, /* number of chunks to be partitioned */
                        int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */                        
			int sync=1, /* whether the flow will continue until all chunks have finished */
                        void *redResult=NULL, REDUCTION_TYPE type=NODEHELPER_NONE /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
                        );
    void reduce(void **redBufs, void *redBuf, REDUCTION_TYPE type, int numChunks);
};

void SingleHelperStealWork(ConverseNotifyMsg *msg);

/* FuncSingleHelper is a chare located on every core of a node */
//allowing arbitrary combination of sync and unsync parallelizd loops
#define MSG_BUFFER_SIZE (3)
class FuncSingleHelper: public CBase_FuncSingleHelper {
	friend class FuncNodeHelper;
private: 
    FuncNodeHelper *thisNodeHelper;
    ConverseNotifyMsg *notifyMsg;
    int nextFreeNotifyMsg;
    //CurLoopInfo *curLoop; /* Points to the current loop that is being processed */
    
public:
    FuncSingleHelper(size_t ndhPtr) {
        thisNodeHelper = (FuncNodeHelper *)ndhPtr;
        CmiAssert(thisNodeHelper!=NULL);        
        int stealWorkHandler = CmiRegisterHandler((CmiHandler)SingleHelperStealWork);
        
	nextFreeNotifyMsg = 0;
        notifyMsg = (ConverseNotifyMsg *)malloc(sizeof(ConverseNotifyMsg)*MSG_BUFFER_SIZE);
        for(int i=0; i<MSG_BUFFER_SIZE; i++){
            ConverseNotifyMsg *tmp = notifyMsg+i;
            if(thisNodeHelper->useTreeBcast){
                tmp->srcRank = CmiMyRank();
            }else{
                tmp->srcRank = -1;
            }            
            tmp->ptr = (void *)(new CurLoopInfo(FuncNodeHelper::MAX_CHUNKS));
            CmiSetHandler(tmp, stealWorkHandler);
        }
        thisNodeHelper->helperPtr[CkMyRank()] = this;
    }

    ~FuncSingleHelper() {
        for(int i=0; i<MSG_BUFFER_SIZE; i++){
            ConverseNotifyMsg *tmp = notifyMsg+i;
            CurLoopInfo *loop = (CurLoopInfo *)(tmp->ptr);
            delete loop;
        }
        free(notifyMsg);
    }
    
    ConverseNotifyMsg *getNotifyMsg(){
        while(1){
            ConverseNotifyMsg *cur = notifyMsg+nextFreeNotifyMsg;
            CurLoopInfo *loop = (CurLoopInfo *)(cur->ptr);
            nextFreeNotifyMsg = (nextFreeNotifyMsg+1)%MSG_BUFFER_SIZE;
            if(loop->isFree()) return cur;
        }
        return NULL;
    }
    
    FuncSingleHelper(CkMigrateMessage *m) {}		
};

#endif
