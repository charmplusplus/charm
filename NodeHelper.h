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
    
    void set(int nc, HelperFn f, int lIdx, int uIdx, int numParams, void *p){        
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
      
    void waitLoopDone(){
        while(!__sync_bool_compare_and_swap(&finishFlag, numChunks, 0));
        inited = 0;
    }
    int getNextChunkIdx(){
        return __sync_add_and_fetch(&curChunkIdx, 1);
    }
    void reportFinished(int counter){
	if(counter==0) return;
        __sync_add_and_fetch(&finishFlag, counter);
    }
    
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
                        int msgPriority, /* the priority of the intra-node msg, and node-level msg */
                        int numChunks, /* number of chunks to be partitioned */
                        int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */                        
                        void *redResult=NULL, REDUCTION_TYPE type=NODEHELPER_NONE /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
                        );
    void reduce(void **redBufs, void *redBuf, REDUCTION_TYPE type, int numChunks);
};

void SingleHelperStealWork(ConverseNotifyMsg *msg);

/* FuncSingleHelper is a chare located on every core of a node */
class FuncSingleHelper: public CBase_FuncSingleHelper {
	friend class FuncNodeHelper;
private: 
    FuncNodeHelper *thisNodeHelper;
    ConverseNotifyMsg *notifyMsg;
    CurLoopInfo *curLoop; /* Points to the current loop that is being processed */
    
public:
    FuncSingleHelper(size_t ndhPtr) {
        thisNodeHelper = (FuncNodeHelper *)ndhPtr;
        CmiAssert(thisNodeHelper!=NULL);        
        int stealWorkHandler = CmiRegisterHandler((CmiHandler)SingleHelperStealWork);
        curLoop = new CurLoopInfo(FuncNodeHelper::MAX_CHUNKS);
        
        notifyMsg = (ConverseNotifyMsg *)malloc(sizeof(ConverseNotifyMsg));
        if(thisNodeHelper->useTreeBcast){
            notifyMsg->srcRank = CmiMyRank();
        }else{
            notifyMsg->srcRank = -1;
        }
        notifyMsg->ptr = (void *)curLoop;
        CmiSetHandler(notifyMsg, stealWorkHandler);
        thisNodeHelper->helperPtr[CkMyRank()] = this;
    }

    ~FuncSingleHelper() {
        delete curLoop;
        delete notifyMsg;
    }
    
    FuncSingleHelper(CkMigrateMessage *m) {}		
};

#endif
