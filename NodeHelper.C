#include "NodeHelper.h"

FuncNodeHelper::FuncNodeHelper()
{  
#if CMK_SMP	
    //CkPrintf("FuncNodeHelper created on node %d\n", CkMyNode());
         
    traceRegisterUserEvent("nodehelper total work",20);
    traceRegisterUserEvent("nodehlelper finish signal",21);
    
	numHelpers = CkMyNodeSize();
	helperPtr = new FuncSingleHelper *[numHelpers];
	useTreeBcast = (numHelpers >= USE_TREE_BROADCAST_THRESHOLD);
	
	int pestart = CkNodeFirst(CkMyNode());
		
	for (int i=0; i<numHelpers; i++) {
        CkChareID helper;
        CProxy_FuncSingleHelper::ckNew((size_t)this, &helper, pestart+i);
	}	
#endif
}

int FuncNodeHelper::MAX_CHUNKS = 64;

#if CMK_TRACE_ENABLED
#define TRACE_START(id) _start = CmiWallTimer()
#define TRACE_BRACKET(id) traceUserBracketEvent(id,_start,CmiWallTimer())
#else
#define TRACE_START(id)
#define TRACE_BRACKET(id)
#endif

#define ALLOW_MULTIPLE_UNSYNC 1
void FuncNodeHelper::parallelizeFunc(HelperFn func, int paramNum, void * param, 
                                    int numChunks, int lowerRange, 
				    int upperRange, int sync,
                                    void *redResult, REDUCTION_TYPE type) {
                                        
    double _start; //may be used for tracing
    
    if(numChunks > MAX_CHUNKS){ 
        CkPrintf("NodeHelper[%d]: WARNING! chunk is set to MAX_CHUNKS=%d\n", CmiMyPe(), MAX_CHUNKS);
        numChunks = MAX_CHUNKS;
    }
	
    /* "stride" determines the number of loop iterations to be done in each chunk
     * for chunk indexed at 0 to remainder-1, stride is "unit+1";
     * for chunk indexed at remainder to numChunks-1, stride is "unit"
     */
     int stride;
    
    //for using nodequeue
	TRACE_START(20);
	
	FuncSingleHelper *thisHelper = helperPtr[CkMyRank()];
#if ALLOW_MULTIPLE_UNSYNC
    ConverseNotifyMsg *notifyMsg = thisHelper->getNotifyMsg();
#else
    ConverseNotifyMsg *notifyMsg = thisHelper->notifyMsg;
#endif
    CurLoopInfo *curLoop = (CurLoopInfo *)(notifyMsg->ptr);
	curLoop->set(numChunks, func, lowerRange, upperRange, paramNum, param);	
	if(useTreeBcast){		
		int loopTimes = TREE_BCAST_BRANCH>(CmiMyNodeSize()-1)?CmiMyNodeSize()-1:TREE_BCAST_BRANCH;
		//just implicit binary tree
		int pe = CmiMyRank()+1;        
		for(int i=0; i<loopTimes; i++, pe++){
			if(pe >= CmiMyNodeSize()) pe -= CmiMyNodeSize();
			CmiPushPE(pe, (void *)(notifyMsg));    
		}
	}else{
		for (int i=0; i<numHelpers; i++) {
			if (i!=CkMyRank()) CmiPushPE(i, (void *)(notifyMsg));            
		}
	}
    
	curLoop->stealWork();
	TRACE_BRACKET(20);
	
	TRACE_START(21);                
	curLoop->waitLoopDone(sync);
	TRACE_BRACKET(21);        

    if (type!=NODEHELPER_NONE)
        reduce(curLoop->getRedBufs(), redResult, type, numChunks);            
    return;
}

#define COMPUTE_REDUCTION(T) {\
    for(int i=0; i<numChunks; i++) {\
     result += *((T *)(redBufs[i])); \
     /*CkPrintf("Nodehelper Reduce: %d\n", result);*/ \
    }\
}

void FuncNodeHelper::reduce(void **redBufs, void *redBuf, REDUCTION_TYPE type, int numChunks) {
    switch(type){
        case NODEHELPER_INT_SUM:
        {
            int result=0;
            COMPUTE_REDUCTION(int)
            *((int *)redBuf) = result;
            break;
        }
        case NODEHELPER_FLOAT_SUM:
        {
            float result=0;
            COMPUTE_REDUCTION(float)
            *((float *)redBuf) = result;
            break;
        }
        case NODEHELPER_DOUBLE_SUM:
        {
            double result=0;
            COMPUTE_REDUCTION(double)
            *((double *)redBuf) = result;
            break;
        }
        default:
        break;
    }
}

CpvStaticDeclare(int, NdhStealWorkHandler);
static void RegisterNodeHelperHdlrs(){
    CpvInitialize(int, NdhStealWorkHandler);
    CpvAccess(NdhStealWorkHandler) = CmiRegisterHandler((CmiHandler)SingleHelperStealWork);
}

FuncSingleHelper::FuncSingleHelper(size_t ndhPtr) {
    thisNodeHelper = (FuncNodeHelper *)ndhPtr;
    CmiAssert(thisNodeHelper!=NULL);
        
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
        CmiSetHandler(tmp, CpvAccess(NdhStealWorkHandler));
    }
    thisNodeHelper->helperPtr[CkMyRank()] = this;
}


void SingleHelperStealWork(ConverseNotifyMsg *msg){
	
	int srcRank = msg->srcRank;
	
	if(srcRank >= 0){
		//means using tree-broadcast to send the notification msg
		
		//int numHelpers = CmiMyNodeSize(); //the value of "numHelpers" should be obtained somewhere else
		int relPE = CmiMyRank()-msg->srcRank;
		if(relPE<0) relPE += CmiMyNodeSize();
		
		//CmiPrintf("Rank[%d]: got msg from src %d with relPE %d\n", CmiMyRank(), msg->srcRank, relPE);
		relPE=relPE*TREE_BCAST_BRANCH+1;
		for(int i=0; i<TREE_BCAST_BRANCH; i++, relPE++){
			if(relPE >= CmiMyNodeSize()) break;
			int pe = (relPE + msg->srcRank)%CmiMyNodeSize();
			//CmiPrintf("Rank[%d]: send msg to dst %d (relPE: %d) from src %d\n", CmiMyRank(), pe, relPE, msg->srcRank);
			CmiPushPE(pe, (void *)msg);
		}
	}
    CurLoopInfo *loop = (CurLoopInfo *)msg->ptr;
    loop->stealWork();
}

void CurLoopInfo::stealWork(){
    //indicate the current work hasn't been initialized
    //or the old work has finished.
    if(inited == 0) return;
    
    int first, last;
    int unit = (upperIndex-lowerIndex+1)/numChunks;
    int remainder = (upperIndex-lowerIndex+1)-unit*numChunks;
    int markIdx = remainder*(unit+1);
    
    int nextChunkId = getNextChunkIdx();
    int execTimes = 0;
    while(nextChunkId < numChunks){
        if(nextChunkId < remainder){
            first = (unit+1)*nextChunkId;
            last = first+unit;
        }else{
            first = (nextChunkId - remainder)*unit + markIdx;
            last = first+unit-1;
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

CProxy_FuncNodeHelper NodeHelper_Init(){
    CkPrintf("NodeHelperLib is used in SMP with a simple dynamic scheduling but not using node-level queue\n");
    return CProxy_FuncNodeHelper::ckNew();
}

void NodeHelper_Parallelize(CProxy_FuncNodeHelper nodeHelper, HelperFn func, 
                        int paramNum, void * param, 
                        int numChunks, int lowerRange, int upperRange,
			int sync,
                        void *redResult, REDUCTION_TYPE type)
{
    nodeHelper[CkMyNode()].ckLocalBranch()->parallelizeFunc(func, paramNum, param, numChunks, lowerRange, upperRange, sync, redResult, type);
}

#include "NodeHelper.def.h"
