#include "NodeHelper.h"

//=======Beginning of pthread version of scheduling which is used in non-SMP =======//
#if !CMK_SMP
NodeQueue Q;

//vars local to spawned threads
//Note: __thread is not portable, but works pretty much anywhere pthreads work.
// after C++11 this should be thread_local
__thread pthread_mutex_t lock;
__thread pthread_cond_t condition;

//vars to the main flow (master thread)
pthread_t * threads;
pthread_mutex_t **allLocks;
pthread_cond_t **allConds;

//global barrier
pthread_mutex_t gLock;
pthread_cond_t gCond;
pthread_barrier_t barr;
//testing counter
volatile int finishedCnt;

void * threadWork(void * id) {
    long my_id =(long) id;
    //printf("thread :%ld\n",my_id);
    CmiSetCPUAffinity(my_id+1);

    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&condition, NULL);

    allLocks[my_id] = &lock;
    allConds[my_id] = &condition;

    while (1) {
        pthread_mutex_lock(&lock);
        pthread_cond_wait(&condition,&lock);
        pthread_mutex_unlock(&lock);
        void * r;
        Task * one;
        CmiLock(Q->lock);
        CqsDequeue(Q->nodeQ,&r);
        CmiUnlock(Q->lock);
        one=(Task *)r;

        while (one) {
            //printf("starttime:%lf,id:%ld,proc:%d\n",CmiWallTimer(),my_id,CkMyPe());
            (one->fnPtr)(one->first, one->last, (void *)(one->redBuf), one->paramNum, one->param);
            pthread_barrier_wait(&barr);
            //one->setFlag();
            //printf
            //printf("endtime:%lf,id:%ld\n",CmiWallTimer(),my_id);

            //Testing
            //AtomicIncrement(finishedCnt);
            if (my_id==0)
                finishedCnt=4;
            //printf("finishedCnt = %d\n", finishedCnt);

            CmiLock((Q->lock));
            CqsDequeue(Q->nodeQ,&r);
            CmiUnlock((Q->lock));
            one=(Task *)r;

        }
    }

}

void FuncNodeHelper::createThread() {
    int threadNum = numThds;
    pthread_attr_t attr;
    finishedCnt=0;
    pthread_barrier_init(&barr,NULL,threadNum);
    allLocks = (pthread_mutex_t **)malloc(sizeof(void *)*threadNum);
    allConds = (pthread_cond_t **)malloc(sizeof(void *)*threadNum);
    memset(allLocks, 0, sizeof(void *)*threadNum);
    memset(allConds, 0, sizeof(void *)*threadNum);

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
    Q=(NodeQueue )malloc(sizeof(struct SimpleQueue));
    Q->nodeQ=CqsCreate();
    Q->lock=CmiCreateLock();
    /*for(int i=0;i<threadNum;i++){
    	//Q[i]=
    	Q[i]=(NodeQueue)malloc(sizeof(struct SimpleQueue));
    	Q[i]->nodeQ=CqsCreate();
    	Q[i]->lock=CmiCreateLock();
    }*/
    threads=(pthread_t *)malloc(threadNum*sizeof(pthread_t));


    //create queue;
    for (int i=0; i<threadNum; i++)
        pthread_create(&threads[i],&attr,threadWork,(void *)i);
}
#endif //end of !CMK_SMP (definitions for vars and functions used in non-SMP)

//=======End of pthread version of static scheduling=======//

FuncNodeHelper::FuncNodeHelper(int mode_,int numThds_):
    mode(mode_), numThds(numThds_)
{   
    
    //CkPrintf("FuncNodeHelper created on node %d\n", CkMyNode());
         
    traceRegisterUserEvent("assign work",20);
    traceRegisterUserEvent("finish signal",21);
    
#if CMK_SMP
    if (mode==NODEHELPER_DYNAMIC || 
        mode==NODEHELPER_STATIC
        || mode==NODEHELPER_CHARE_DYNAMIC) {
        numHelpers = CkMyNodeSize();
        helperArr = new CkChareID[numHelpers];
        helperPtr = new FuncSingleHelper *[numHelpers];
        
        notifyMsgs = (ConverseNotifyMsg *)malloc(sizeof(ConverseNotifyMsg)*numHelpers);
        
        int pestart = CkNodeFirst(CkMyNode());
        for (int i=0; i<numHelpers; i++) {
            CProxy_FuncSingleHelper::ckNew(thisgroup, &helperArr[i], pestart+i);
            helperPtr[i] = NULL;
        }
        for (int i=0; i<numHelpers; i++) {
            CProxy_FuncSingleHelper helpProxy(helperArr[i]);
            helpProxy.reportCreated();
        }
    }
#else
    CmiAssert(mode==NODEHELPER_PTHREAD);
    createThread();    
#endif
}

/* Used for dynamic scheduling as it's a node-level msg */
/* So this function will be executed on any PE of this node */
void FuncNodeHelper::send(Task * msg) {
    (msg->fnPtr)(msg->first,msg->last,(void *)(msg->redBuf),msg->paramNum, msg->param);
    CmiNodeLock lock = helperPtr[msg->originRank]->reqLock;
    CmiLock(lock);
    helperPtr[msg->originRank]->counter++;
    CmiUnlock(lock);
}

int FuncNodeHelper::MAX_CHUNKS = 64;

#if CMK_TRACE_ENABLED
#define TRACE_START(id) _start = CmiWallTimer()
#define TRACE_BRACKET(id) traceUserBracketEvent(id,_start,CmiWallTimer())
#else
#define TRACE_START(id)
#define TRACE_BRACKET(id)
#endif

void FuncNodeHelper::parallelizeFunc(HelperFn func, int paramNum, void * param, 
                                    int msgPriority, int numChunks, int lowerRange, int upperRange, 
                                    void *redResult, REDUCTION_TYPE type) {
                                        
    double _start; //may be used for tracing
    
    if(numChunks > MAX_CHUNKS){ 
        CkPrintf("NodeHelper[%d]: WARNING! chunk is set to MAX_CHUNKS=%d\n", CmiMyPe(), MAX_CHUNKS);
        numChunks = MAX_CHUNKS;
    }
        
    Task **task = helperPtr[CkMyRank()]->getTasksMem();
    
    /* "stride" determines the number of loop iterations to be done in each chunk
     * for chunk indexed at 0 to remainder-1, stride is "unit+1";
     * for chunk indexed at remainder to numChunks-1, stride is "unit"
     */
     int stride;
    
    //for using nodequeue
#if CMK_SMP
    if (mode==NODEHELPER_DYNAMIC) {
        int first = lowerRange;
        int unit = (upperRange-lowerRange+1)/numChunks;
        int remainder = (upperRange-lowerRange+1)-unit*numChunks;        
        CProxy_FuncNodeHelper fh(thisgroup);

        TRACE_START(20);        
        stride = unit+1;
        for (int i=0; i<remainder; i++, first+=stride) {          
            task[i]->init(func, first, first+stride-1, CkMyRank(), paramNum, param);
            *((int *)CkPriorityPtr(task[i]))=msgPriority;
            CkSetQueueing(task[i],CK_QUEUEING_IFIFO);
            fh[CkMyNode()].send(task[i]);
        }
        
        stride = unit;
        for(int i=remainder; i<numChunks; i++, first+=stride) {
            task[i]->init(func, first, first+stride-1, CkMyRank(), paramNum, param);
            *((int *)CkPriorityPtr(task[i]))=msgPriority;
            CkSetQueueing(task[i],CK_QUEUEING_IFIFO);
            fh[CkMyNode()].send(task[i]);
        }
        TRACE_BRACKET(20);
        
        TRACE_START(21);
        FuncSingleHelper *fs = helperPtr[CmiMyRank()];
        while (fs->counter!=numChunks)
            CsdScheduleNodePoll();
        //CkPrintf("counter:%d,master:%d\n",counter[master],master);
        fs->counter = 0;
        TRACE_BRACKET(21);        
    } else if (mode==NODEHELPER_STATIC) {
        int first = lowerRange;
        int unit = (upperRange-lowerRange+1)/numChunks;
        int remainder = (upperRange-lowerRange+1)-unit*numChunks;

        TRACE_START(20);
                
        stride = unit+1;
        for (int i=0; i<remainder; i++, first+=stride) {
            task[i]->init(func, first, first+stride-1, 0, CkMyRank(),paramNum, param);            
            helperPtr[i%numHelpers]->enqueueWork(task[i]);
        }
        
        stride = unit;
        for (int i=remainder; i<numChunks; i++, first+=stride) {
            task[i]->init(func, first, first+stride-1, 0, CkMyRank(),paramNum, param);            
            helperPtr[i%numHelpers]->enqueueWork(task[i]);
        }
        
#if USE_CONVERSE_MSG
        
        for (int i=0; i<numHelpers; i++) {
            if (i!=CkMyRank()) {
                CmiPushPE(i, (void *)(notifyMsgs+i));
            }
        }
#else
        CkEntryOptions entOpts;
        entOpts.setPriority(msgPriority);

        for (int i=0; i<numHelpers; i++) {
            if (i!=CkMyRank()) {
                CProxy_FuncSingleHelper helpProxy(helperArr[i]);                
                helpProxy.processWork(0, &entOpts);
            }
        }    
#endif        
        helperPtr[CkMyRank()]->processWork(0);
        
        TRACE_BRACKET(20);
        
        TRACE_START(21);
                
        while(!__sync_bool_compare_and_swap(&(helperPtr[CkMyRank()]->counter), numChunks, 0));
        //waitDone(task,numChunks);
        
        TRACE_BRACKET(21);
    }else if(mode == NODEHELPER_CHARE_DYNAMIC){
        TRACE_START(20);
        
        FuncSingleHelper *thisHelper = helperPtr[CkMyRank()];
        CurLoopInfo *curLoop = thisHelper->curLoop;
        curLoop->set(numChunks, func, lowerRange, upperRange, paramNum, param);
        
        for (int i=0; i<numHelpers; i++) {
            if (i!=CkMyRank()) {
                notifyMsgs[i].ptr = (void *)curLoop;
                CmiPushPE(i, (void *)(notifyMsgs+i));
            }
        }        
        curLoop->stealWork();
        TRACE_BRACKET(20);
        
        TRACE_START(21);                
        curLoop->waitLoopDone();
        TRACE_BRACKET(21);        
    }
#else
//non-SMP case
/* Only works in the non-SMP case */
    CmiAssert(mode == NODEHELPER_PTHREAD);
    
    TRACE_START(20);
    stride = unit+1;
    for (int i=0; i<remainder; i++, first+=stride) {
        task[i]->init(func, first, first+stride-1, 0, CkMyRank(),paramNum, param);            
        CmiLock((Q->lock));
        unsigned int t=(int)(CmiWallTimer()*1000);
        CqsEnqueueGeneral((Q->nodeQ), (void *)task[i],CQS_QUEUEING_IFIFO,0,&t);
        CmiUnlock((Q->lock));
    }
    
    stride = unit;
    for (int i=remainder; i<numChunks; i++, first+=stride) {
        task[i]->init(func, first, first+stride-1, 0, CkMyRank(),paramNum, param);            
        CmiLock((Q->lock));
        unsigned int t=(int)(CmiWallTimer()*1000);
        CqsEnqueueGeneral((Q->nodeQ), (void *)task[i],CQS_QUEUEING_IFIFO,0,&t);
        CmiUnlock((Q->lock));
    }    
    //signal the thread
    for (int i=0; i<threadNum; i++) {
        pthread_mutex_lock(allLocks[i]);
        pthread_cond_signal(allConds[i]);
        pthread_mutex_unlock(allLocks[i]);
    }
    TRACE_BRACKET(20);
    
    TRACE_START(21);
    //wait for the result
    waitThreadDone(numChunks);
    TRACE_BRACKET(21);
#endif

    if (type!=NODEHELPER_NONE)
        reduce(task, redResult, type, numChunks);            
    return;
}

#define COMPUTE_REDUCTION(T) {\
    for(int i=0; i<numChunks; i++) {\
     result += *((T *)(thisReq[i]->redBuf)); \
     /*CkPrintf("Nodehelper Reduce: %d\n", result);*/ \
    }\
}

void FuncNodeHelper::reduce(Task ** thisReq, void *redBuf, REDUCTION_TYPE type, int numChunks) {
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

#if CMK_SMP
void FuncNodeHelper::waitDone(Task ** thisReq,int chunck) {
    int flag = 1,i;
    while (1) {
        for (i=0; i<chunck; i++)
            flag = flag & thisReq[i]->isFlagSet();
        if (flag) break;
        flag = 1;
    }
}
#else
void FuncNodeHelper::waitThreadDone(int chunck) {
    while (finishedCnt!=chunck);
    finishedCnt=0;
}
#endif

void FuncNodeHelper::printMode(int mode) {
    switch(mode){
        case NODEHELPER_PTHREAD:
            CkPrintf("NodeHelperLib is used in non-SMP using pthread with a simple dynamic scheduling\n");
            break;
        case NODEHELPER_DYNAMIC:
            CkPrintf("NodeHelperLib is used in SMP with a simple dynamic scheduling\n");
            break;
        case NODEHELPER_STATIC:
            CkPrintf("NodeHelperLib is used in SMP with a simple static scheduling\n");
            break;
        case NODEHELPER_CHARE_DYNAMIC:
            CkPrintf("NodeHelperLib is used in SMP with a simple dynamic scheduling but not using node-level queue\n");
            break;
        default:
            CkPrintf("ERROR: NodeHelperLib is used in unknown mode\n");
    }
}

void NotifySingleHelper(ConverseNotifyMsg *msg){
    FuncSingleHelper *h = (FuncSingleHelper *)msg->ptr;
    h->processWork(0);
}

void SingleHelperStealWork(ConverseNotifyMsg *msg){
    CurLoopInfo *loop = (CurLoopInfo *)msg->ptr;
    loop->stealWork();
}

//======================================================================//
// Functions regarding helpers that parallelize a single function on a  //
// single node (like OpenMP)                                            // 
//======================================================================//
void FuncSingleHelper::processWork(int filler) {
    Task *one = NULL; // = (WorkReqEntry *)SimpleQueuePop(reqQ);    
    void *tmp;
    
    CmiLock(reqLock);
    CqsDequeue(reqQ, &tmp);
    CmiUnlock(reqLock);    

    one = (Task *)tmp;
    while (one) {
        (one->fnPtr)(one->first,one->last,(void *)(one->redBuf), one->paramNum, one->param);
        //int *partial = (int *)(one->redBuf);
        //CkPrintf("SingleHelper[%d]: partial=%d\n", CkMyRank(), *partial);
        
        //one->setFlag();
        __sync_add_and_fetch(&(thisNodeHelper->helperPtr[one->originRank]->counter), 1);
        
        
        CmiLock(reqLock);
        CqsDequeue(reqQ, &tmp);
        one = (Task *)tmp;
        CmiUnlock(reqLock);
    }
}

void CurLoopInfo::stealWork(){
    int first, last;
    int unit = (upperIndex-lowerIndex+1)/numChunks;
    int remainder = (upperIndex-lowerIndex+1)-unit*numChunks;
    int markIdx = remainder*(unit+1);
    
    int nextChunkId = getNextChunkIdx();
    while(nextChunkId < numChunks){
        if(nextChunkId < remainder){
            first = (unit+1)*nextChunkId;
            last = first+unit;
        }else{
            first = (nextChunkId - remainder)*unit + markIdx;
            last = first+unit-1;
        }
                
        fnPtr(first, last, redBufs[nextChunkId], paramNum, param);
        reportFinished();
        
        nextChunkId = getNextChunkIdx();
    }
}

//======================================================================//
//   End of functions related with FuncSingleHelper                     //
//======================================================================//

CProxy_FuncNodeHelper NodeHelper_Init(int mode,int numThds){
    FuncNodeHelper::printMode(mode);
    return CProxy_FuncNodeHelper::ckNew(mode, numThds);
}

void NodeHelper_Parallelize(CProxy_FuncNodeHelper nodeHelper, HelperFn func, 
                        int paramNum, void * param, int msgPriority,
                        int numChunks, int lowerRange, int upperRange, 
                        void *redResult, REDUCTION_TYPE type)
{
    nodeHelper[CkMyNode()].ckLocalBranch()->parallelizeFunc(func, paramNum, param, msgPriority, numChunks, lowerRange, upperRange, redResult, type);
}

#include "NodeHelper.def.h"
