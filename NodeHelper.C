#include "NodeHelper.h"
#define THRESHOLD 100
#define WPSTHRESHOLD 400
#define SMP_SUM 1

NodeQueue Q;

//vars local to spawned threads
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

void * threadWork(void * id){
	long my_id =(long) id;
	//printf("thread :%ld\n",my_id);
	CmiSetCPUAffinity(my_id+1);
	
	pthread_mutex_init(&lock, NULL);
	pthread_cond_init(&condition, NULL);
	
	allLocks[my_id] = &lock;
	allConds[my_id] = &condition;
	
	while(1){
		pthread_mutex_lock(&lock);
		pthread_cond_wait(&condition,&lock);
		pthread_mutex_unlock(&lock);
		void * r;
		Task * one;
		CmiLock(Q->lock);
		CqsDequeue(Q->nodeQ,&r);
		CmiUnlock(Q->lock);
		one=(Task *)r;
	    
		while(one) {
			//printf("starttime:%lf,id:%ld,proc:%d\n",CmiWallTimer(),my_id,CkMyPe());
			(one->fnPtr)(one->first,one->last,one->result,one->paramNum, one->param);
			pthread_barrier_wait(&barr);
			//one->setFlag();
			//printf
			//printf("endtime:%lf,id:%ld\n",CmiWallTimer(),my_id);
			
			//Testing
			//AtomicIncrement(finishedCnt);
			if(my_id==0)
				finishedCnt=4;
			//printf("finishedCnt = %d\n", finishedCnt);
			
			CmiLock((Q->lock));
			CqsDequeue(Q->nodeQ,&r);
			CmiUnlock((Q->lock));
			one=(Task *)r;
			
	 	}	
	}
	
}

FuncNodeHelper::FuncNodeHelper(int mode_o,int nElements, int threadNum_o){
	mode=mode_o;
	threadNum=threadNum_o;
	numHelpers = CkMyNodeSize();
		traceRegisterUserEvent("assign work",20);	
		traceRegisterUserEvent("finish signal",21);	
#if CMK_SMP
	if(mode==1){
		counter=new int[nElements];
		reqLock=new  pthread_mutex_t *[nElements];
		for(int i=0;i<nElements;i++){
			counter[i]=0;
			reqLock[i] = CmiCreateLock();
		}
	}else if(mode==2){
		helperArr = new CkChareID[numHelpers];
		helperPtr = new FuncSingleHelper *[numHelpers];
		int pestart = CkNodeFirst(CkMyNode());
		for(int i=0; i<numHelpers; i++) {
			CProxy_FuncSingleHelper::ckNew(i, thisgroup, &helperArr[i], pestart+i);    
			helperPtr[i] = NULL;
		}
		for(int i=0; i<numHelpers; i++) {
			CProxy_FuncSingleHelper helpProxy(helperArr[i]);
			helpProxy.reportCreated();
		}
	}
#endif
}
void FuncNodeHelper::createThread(){

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
		for(int i=0;i<threadNum;i++)
			pthread_create(&threads[i],&attr,threadWork,(void *)i);
		
}
void FuncNodeHelper::send(Task * msg){
	(msg->fnPtr)(msg->first,msg->last,msg->result,msg->paramNum, msg->param);
	CmiLock(reqLock[msg->master]);
	counter[msg->master]++;
	CmiUnlock(reqLock[msg->master]);
}
int FuncNodeHelper::parallelizeFunc(HelperFn func, int wps,unsigned int t, int master,int chunck,int time,int paramNum, void * param, int reduction, int type){
	int result=0;
	if(chunck==0){
		if(time!=0)
			chunck=(double)(time/THRESHOLD)+0.5;
		else
			chunck=(double)(wps/WPSTHRESHOLD)+0.5;
	}
	int unit=((double)wps)/(double)chunck+0.5;
	//printf("use %d chuncks for testcase %d\n",chunck,master);
	Task **task=new Task *[chunck];
    //for using nodequeue 
#if CMK_SMP
	if(mode==1){
#if 0
//Note: CsdScheduleNodePoll has not been in the charm yet, so currently disable it
		CProxy_FuncNodeHelper fh(thisgroup);
		double _start=CmiWallTimer();
		for(int i=0; i<chunck; i++) {
			int first=unit*i;
			int last= (i==chunck-1)?wps:(i+1)*unit-1;
			task[i]=new (8*sizeof(int)) Task(func,first,last,master, paramNum, param);
			*((int *)CkPriorityPtr(task[i]))=t;
			CkSetQueueing(task[i],CK_QUEUEING_IFIFO);
			fh[CkMyNode()].send(task[i]);
			//send(task[i]);
		}
		traceUserBracketEvent(20,_start,CmiWallTimer());
		_start=CmiWallTimer();
		while(counter[master]!=chunck)
			CsdScheduleNodePoll();
		//CkPrintf("counter:%d,master:%d\n",counter[master],master);
		traceUserBracketEvent(21,_start,CmiWallTimer());
		counter[master]=0;
		/*for(int i=0;i<chunck;i++){
			result+=task[i]->result;
		}*/
#endif
	}
	else if(mode==2){
	// for not using node queue
		for(int i=0; i<chunck; i++) {
		  	int first=unit*i;
          		int last= (i==chunck-1)?wps:(i+1)*unit-1;
    	  		task[i] = new Task(func,first,last,0,master,paramNum, param);
          		//task[i]->UnsetFlag();
          		helperPtr[i%CkMyNodeSize()]->enqueueWork(task[i],t);
    		}
   		for(int i=0; i<numHelpers; i++) {        
        		if(i!=CkMyRank()){
           			CProxy_FuncSingleHelper helpProxy(helperArr[i]);
            			helpProxy.processWork();
        		}
    		}
   		helperPtr[CkMyRank()]->processWork();
  
   		waitDone(task,chunck);
   		result=0;
		
   		/*for(int i=0;i<chunck;i++){
	     	result+=(task[i]->result);
    		}*/
	}
#else
	if(mode==0){
		for(int i=0;i<chunck;i++)
		{
			int first = unit*i;
			int last=(i==chunck-1)?wps:(i+1)*unit-1;
			task[i]=new Task(func,first,last,0,master, paramNum, param);
			CmiLock((Q->lock));
			unsigned int t=(int)(CmiWallTimer()*1000);
			CqsEnqueueGeneral((Q->nodeQ), (void *)task[i],CQS_QUEUEING_IFIFO,0,&t);
			CmiUnlock((Q->lock));				
		}
		//signal the thread
		for(int i=0;i<threadNum;i++){
			pthread_mutex_lock(allLocks[i]);
			pthread_cond_signal(allConds[i]);
			pthread_mutex_unlock(allLocks[i]);
		}
		//wait for the result
		waitThreadDone(chunck);
		//for(int i=0;i<threadNum;i++)
		//	pthread_join(threads[i],NULL);
		/*result=0;
		for(int i=0;i<chunck;i++)
			result+=task[i]->result;
		*/
	}
#endif	
 	if(reduction==1)
		result=reduce(task, type,chunck);
	delete task;
	return result;
}

//======================================================================
// Functions regarding helpers that parallelize a single function on a
// single node (like OpenMP)
void FuncSingleHelper::processWork(){
    	//CmiLock(reqLock);
	void *r;
    Task *one; // = (WorkReqEntry *)SimpleQueuePop(reqQ);
	CmiLock(reqLock);	
	CqsDequeue(reqQ,&r);
	CmiUnlock(reqLock);
	one=(Task *)r;
    
    while(one) {
        (one->fnPtr)(one->first,one->last,one->result, one->paramNum, one->param);
        one->setFlag();
		CmiLock(reqLock);
		CqsDequeue(reqQ,&r);
		CmiUnlock(reqLock);
		one=(Task *)r;
		
    }    
}

void FuncSingleHelper::reportCreated(){
    CProxy_FuncNodeHelper fh(nodeHelperID);
    CProxy_FuncSingleHelper thisproxy(thishandle);
    fh[CkMyNode()].ckLocalBranch()->oneHelperCreated(id, thishandle, this);
}

void FuncNodeHelper::waitDone(Task ** thisReq,int chunck){
    int flag = 1,i;
    while(1) {
        for(i=0; i<chunck; i++) 
            flag = flag & thisReq[i]->isFlagSet();
        if(flag) break;
        flag = 1;
    }
}
int FuncNodeHelper::reduce(Task ** thisReq, int type,int chunck){
	int result=0,i;
	if(type==SMP_SUM){
		for(i=0;i<chunck;i++){
			result+=thisReq[i]->result;
		//CkPrintf("result:%d\n",result);
		}
	}
	return result;
}
void FuncNodeHelper::waitThreadDone(int chunck){
	while(finishedCnt!=chunck);
	finishedCnt=0;
}

#include "NodeHelper.def.h"
