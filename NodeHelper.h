#ifndef _NODEHELPER_H
#define _NODEHELPER_H

#include <pthread.h>
#include <assert.h>

#include "charm++.h"
#include "NodeHelperAPI.h"
#include "queueing.h"

#define USE_CONVERSE_MSG 1

/* The following only works on X86_64 platform */
#define AtomicIncrement(someInt)  __asm__ __volatile__("lock incl (%0)" :: "r" (&(someInt)))


typedef struct SimpleQueue {
    Queue nodeQ;
    pthread_mutex_t * lock;
}* NodeQueue;

class Task:public CMessage_Task {
public:
    HelperFn fnPtr;
    int first;
    int last;
    int originRank;
    int flag;    
    int paramNum;
    void *param;
    
    //limitation: only allow single variable reduction!!!
    char redBuf[sizeof(double)];
    
    //make sure 32-byte aligned so that each task doesn't cross cache lines
    //char padding[32-(sizeof(int)*7+sizeof(void *)*2)%32];
    
    Task():fnPtr(NULL), param(NULL), paramNum(0) {}

    void init(HelperFn fn,int first_,int last_,int rank){
        fnPtr = fn;
        first = first_;
        last = last_;
        originRank = rank;
    }

    void init(HelperFn fn,int first_,int last_,int flag_,int rank) {
        init(fn, first_, last_, rank);
        flag=flag_;
    }
    
    void init(HelperFn fn,int first_,int last_,int rank, int paramNum_, void *param_) {
        init(fn, first_, last_, rank); 
        paramNum=paramNum_;
        param=param_;
    }
    
    void init(HelperFn fn,int first_,int last_,int flag_,int rank, int paramNum_, void *param_) {
        init(fn, first_, last_, rank, paramNum_, param_);
        flag=flag_;
    }
    
    void setFlag() {
        flag=1;
    }
    int isFlagSet() {
        return flag;
    }
};


/* FuncNodeHelper is a nodegroup object */
class FuncSingleHelper;
#if USE_CONVERSE_MSG
typedef struct converseNotifyMsg{
    char core[CmiMsgHeaderSizeBytes];
    FuncSingleHelper *ptr;
}ConverseNotifyMsg;
#endif

class FuncNodeHelper : public CBase_FuncNodeHelper {
    friend class FuncSingleHelper;
private:
#if USE_CONVERSE_MSG
    ConverseNotifyMsg *notifyMsgs;         
#endif

public:
    static int MAX_CHUNKS;
    static void printMode(int mode);

public:
    int numHelpers;
    int mode; /* determine whether using dynamic or static scheduling */
    
    int numThds; /* only used for pthread version in non-SMP case, the expected #pthreads to be created */
    
    CkChareID *helperArr; /* chare ids to the FuncSingleHelpers it manages */
    FuncSingleHelper **helperPtr; /* ptrs to the FuncSingleHelpers it manages */
    
    ~FuncNodeHelper() {
        delete [] helperArr;
        delete [] helperPtr;
    #if USE_CONVERSE_MSG
        delete [] notifyMsgs;
    #endif
    }

    /* handler is only useful when converse msg is used to initiate tasks on the pseudo-thread */
    void oneHelperCreated(int hid, CkChareID cid, FuncSingleHelper* cptr, int handler=0) {
        helperArr[hid] = cid;
        helperPtr[hid] = cptr;
#if USE_CONVERSE_MSG        
        notifyMsgs[hid].ptr = cptr;
        CmiSetHandler(&(notifyMsgs[hid]), handler);
#endif
    }

#if CMK_SMP
    void  waitDone(Task ** thisReq,int chunck);
#else	
    void waitThreadDone(int chunck);
    void createThread();
#endif
	
	/* mode_: PTHREAD only available in non-SMP, while STATIC/DYNAMIC are available in SMP */
	/* numThds: the expected number of pthreads to be spawned */
    FuncNodeHelper(int mode_, int numThds_);
    
    void parallelizeFunc(HelperFn func, /* the function that finishes a partial work on another thread */
                        int paramNum, void * param, /* the input parameters for the above func */
                        int msgPriority, /* the priority of the intra-node msg, and node-level msg */
                        int numChunks, /* number of chunks to be partitioned */
                        int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */                        
                        void *redResult=NULL, REDUCTION_TYPE type=NODEHELPER_NONE /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
                        );
    void send(Task *);
    void reduce(Task **thisReq, void *redBuf, REDUCTION_TYPE type, int numChunks);
};

#if USE_CONVERSE_MSG
void NotifySingleHelper(ConverseNotifyMsg *msg);
#endif

/* FuncSingleHelper is a chare located on every core of a node */
class FuncSingleHelper: public CBase_FuncSingleHelper {
	friend class FuncNodeHelper;
private: 
    /* BE CAREFUL ABOUT THE FILEDS LAYOUT CONSIDERING CACHE EFFECTS */
    volatile int counter;
    int notifyHandler;
    CkGroupID nodeHelperID;
    FuncNodeHelper *thisNodeHelper;
    Queue reqQ; /* The queue may be simplified for better performance */
    
    /* The following two vars are for usage of detecting completion in dynamic scheduling */
    CmiNodeLock reqLock;
    /* To reuse such Task memory as each SingleHelper (i.e. a PE) will only
     * process one node-level parallelization at a time */
    Task **tasks; /* Note the Task type is a message */
public:
    FuncSingleHelper(CkGroupID nid):nodeHelperID(nid) {
        reqQ = CqsCreate();

        reqLock = CmiCreateLock();
        counter = 0;
        
        tasks = new Task *[FuncNodeHelper::MAX_CHUNKS];
        for(int i=0; i<FuncNodeHelper::MAX_CHUNKS; i++) tasks[i] = new (8*sizeof(int)) Task();
        
        CProxy_FuncNodeHelper fh(nodeHelperID);
        thisNodeHelper = fh[CkMyNode()].ckLocalBranch();
        CmiAssert(thisNodeHelper!=NULL);
        
        notifyHandler = 0;
    #if USE_CONVERSE_MSG
        notifyHandler = CmiRegisterHandler((CmiHandler)NotifySingleHelper);
    #endif
    }

    ~FuncSingleHelper() {
        for(int i=0; i<FuncNodeHelper::MAX_CHUNKS; i++) delete tasks[i];
        delete [] tasks;        
		CmiDestroyLock(reqLock);
    }
    
    FuncSingleHelper(CkMigrateMessage *m) {}
    
    Task **getTasksMem() { return tasks; }
    
    void enqueueWork(Task *one) {
		unsigned int t = 0; /* default priority */
        CmiLock(reqLock);
        CqsEnqueueGeneral(reqQ, (void *)one,CQS_QUEUEING_IFIFO,0,&t);
        //SimpleQueuePush(reqQ, (char *)one);
        CmiUnlock(reqLock);
    }
    void processWork(int filler); /* filler is here in order to use CkEntryOptions for setting msg priority */
    void reportCreated() {
        //CkPrintf("Single helper %d is created on rank %d\n", CkMyPe(), CkMyRank());        
        thisNodeHelper->oneHelperCreated(CkMyRank(), thishandle, this, notifyHandler);
    }
};

#endif
