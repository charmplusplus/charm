#include "NodeHelper.h"
#include <pthread.h>

/*====Beginning of pthread-related variables and impelementation====*/
//__thread is not portable, but it works almost everywhere if pthread works
//After C++11, this should be thread_local
static __thread pthread_cond_t thdCondition; //the signal var of each pthread to be notified
static __thread pthread_mutex_t thdLock; //the lock associated with the condition variables

static FuncNodeHelper *mainHelper = NULL;
static int mainHelperPhyRank = 0;
static int numPhysicalPEs = 0;
static CurLoopInfo *pthdLoop = NULL; //the pthread-version is always synchronized
static pthread_mutex_t **allLocks = NULL;
static pthread_cond_t **allConds = NULL;
static pthread_t *ndhThreads = NULL;
static volatile int gCrtCnt = 0;
static volatile int exitFlag = 0;

#if CMK_OS_IS_LINUX
#include <sys/syscall.h>
#endif

static int HelperOnCore(){
#if CMK_OS_IS_LINUX
	char fname[64];
	sprintf(fname, "/proc/%d/task/%ld/stat", getpid(), syscall(SYS_gettid));
	FILE *ifp = fopen(fname, "r");
	if(ifp == NULL) return -1;
	fseek(ifp, 0, SEEK_SET);
	char str[128];
	for(int i=0; i<39; i++) fscanf(ifp, "%s", str);
	fclose(ifp);
	return atoi(str);
#else
	return -1;
#endif
}

static void *ndhThreadWork(void *id) {
	size_t myId = (size_t) id;
	
	//further improvement of this affinity setting!!
	int myPhyRank = (myId+mainHelperPhyRank)%numPhysicalPEs;
	//printf("thread[%d]: affixed to rank %d\n", myId, myPhyRank);
	myPhyRank = myId;
	CmiSetCPUAffinity(myPhyRank);
	
	pthread_mutex_init(&thdLock, NULL);
	pthread_cond_init(&thdCondition, NULL);
	
	allLocks[myId-1] = &thdLock;
	allConds[myId-1] = &thdCondition;
	
	__sync_add_and_fetch(&gCrtCnt, 1);
	
	while(1){
		//printf("thread[%ld]: on core %d with main %d\n", myId, HelperOnCore(), mainHelperPhyRank);
		if(exitFlag) break;
		pthread_mutex_lock(&thdLock);
		pthread_cond_wait(&thdCondition, &thdLock);
		pthread_mutex_unlock(&thdLock);
		/* kids ID range: [1 ~ numHelpers-1] */
		if(mainHelper->needTreeBcast()){
			//notify my children
			int myKid = myId*TREE_BCAST_BRANCH+1;
			for(int i=0; i<TREE_BCAST_BRANCH; i++, myKid++){
				if(myKid >= mainHelper->getNumHelpers()) break;
				//all locks and conditions exclude the main thread, so index needs to be subtracted by one
				pthread_mutex_lock(allLocks[myKid-1]);
				pthread_cond_signal(allConds[myKid-1]);
				pthread_mutex_unlock(allLocks[myKid-1]);
			}
		}
		pthdLoop->stealWork();
	}
}

void FuncNodeHelper::createPThreads() {
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
	if(mainHelperPhyRank == -1) mainHelperPhyRank = 0;
	for(int i=1; i<=numThreads; i++){
		pthread_create(ndhThreads+i, &attr, ndhThreadWork, (void *)i);
	}
	while(gCrtCnt != numThreads); //wait for all threads to finish creation
}

void FuncNodeHelper::exit(){
	if(mode == NODEHELPER_PTHREAD){
		exitFlag = 1;             
		for(int i=0; i<numHelpers-1; i++)
			pthread_join(ndhThreads[i], NULL);
		delete [] ndhThreads;
		free(allLocks);
		free(allConds);
		delete pthdLoop;
	}
}

/*====End of pthread-related variables and impelementation====*/


/* Note: Those event ids should be unique globally!! */
#define NDH_TOTAL_WORK_EVENTID  139
#define NDH_FINISH_SIGNAL_EVENTID 143

FuncNodeHelper::FuncNodeHelper(int mode_, int numThreads_)
{  
    traceRegisterUserEvent("nodehelper total work",NDH_TOTAL_WORK_EVENTID);
    traceRegisterUserEvent("nodehlelper finish signal",NDH_FINISH_SIGNAL_EVENTID);

    mode = mode_;

    if(mode == NODEHELPER_USECHARM){
        //CkPrintf("FuncNodeHelper created on node %d\n", CkMyNode());                     
        numHelpers = CkMyNodeSize();
        helperPtr = new FuncSingleHelper *[numHelpers];
        useTreeBcast = (numHelpers >= USE_TREE_BROADCAST_THRESHOLD);
        
        int pestart = CkNodeFirst(CkMyNode());
            
        for (int i=0; i<numHelpers; i++) {
            CkChareID helper;
            CProxy_FuncSingleHelper::ckNew((size_t)this, &helper, pestart+i);
        }
    }else if(mode == NODEHELPER_PTHREAD){
		helperPtr = NULL;
		
		numHelpers = numThreads_;
		useTreeBcast = (numHelpers >= USE_TREE_BROADCAST_THRESHOLD);
		pthdLoop = new CurLoopInfo(FuncNodeHelper::MAX_CHUNKS);
		mainHelper = this;
        createPThreads();
    }
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
	 CurLoopInfo *curLoop = NULL;
    
    //for using nodequeue
	TRACE_START(NDH_TOTAL_WORK_EVENTID);
	if(mode == NODEHELPER_USECHARM){	
		FuncSingleHelper *thisHelper = helperPtr[CkMyRank()];
	#if ALLOW_MULTIPLE_UNSYNC
		ConverseNotifyMsg *notifyMsg = thisHelper->getNotifyMsg();
	#else
		ConverseNotifyMsg *notifyMsg = thisHelper->notifyMsg;
	#endif
		curLoop = (CurLoopInfo *)(notifyMsg->ptr);
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
	}else if(mode == NODEHELPER_PTHREAD){
		int numThreads = numHelpers-1;
		curLoop = pthdLoop;
		curLoop->set(numChunks, func, lowerRange, upperRange, paramNum, param);
		int numNotices = numThreads;
		if(useTreeBcast){
			numNotices = TREE_BCAST_BRANCH>=numThreads?numThreads:TREE_BCAST_BRANCH;
		}
		for(int i=0; i<numNotices; i++){
			pthread_mutex_lock(allLocks[i]);
			pthread_cond_signal(allConds[i]);
			pthread_mutex_unlock(allLocks[i]);
		}
		//in this mode, it's always synced
		sync = 1;
	}
	
	curLoop->stealWork();
	TRACE_BRACKET(NDH_TOTAL_WORK_EVENTID);
	
	TRACE_START(NDH_FINISH_SIGNAL_EVENTID);
	curLoop->waitLoopDone(sync);
	TRACE_BRACKET(NDH_FINISH_SIGNAL_EVENTID);

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

CProxy_FuncNodeHelper NodeHelper_Init(int mode, int numThreads){
    if(mode == NODEHELPER_USECHARM){
        CkPrintf("NodeHelperLib is used in SMP with a simple dynamic scheduling but not using node-level queue\n");
    }else if(mode==NODEHELPER_PTHREAD){
        CkPrintf("NodeHelperLib is used with extra %d pthreads via a simple dynamic scheduling\n", numThreads);
        CmiAssert(numThreads>0);
    }
    return CProxy_FuncNodeHelper::ckNew(mode, numThreads);
}

void NodeHelper_Exit(CProxy_FuncNodeHelper nodeHelper){
	nodeHelper.exit();
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
