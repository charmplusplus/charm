#ifndef _NODEHELPER_H
#define _NODEHELPER_H

#include <pthread.h>

#include "charm++.h"
#include "NodeHelper.decl.h"
#include <assert.h>
#include "queueing.h"
#include <converse.h>
#define AtomicIncrement(someInt)  __asm__ __volatile__("lock incl (%0)" :: "r" (&(someInt)))
#define SMP_SUM 1
typedef void (*HelperFn)(int first,int last, int &result, int paramNum, void * param_o);

typedef struct SimpleQueue
{
	Queue nodeQ;
	pthread_mutex_t * lock;
}* NodeQueue;
class Task:public CMessage_Task{
public:
	HelperFn fnPtr;
	int first;
	int last;
	int result;
	int master;
	int flag;
	int reduction;
	int paramNum;
	void * param;
	Task(HelperFn fn,int first_o,int last_o,int master_o){
		fnPtr=fn;
		first=first_o;
		last=last_o;
		master=master_o;
	}
	
	Task(HelperFn fn,int first_o,int last_o,int flag_o,int master_o){
		fnPtr=fn;
		first=first_o;
		last=last_o;
		flag=flag_o;
		master=master_o;
	}
	Task(HelperFn fn,int first_o,int last_o,int master_o, int paramNum_o, void * param_o){
		fnPtr=fn;
		first=first_o;
		last=last_o;
		master=master_o;
		//reduction=reduction_o;
		paramNum=paramNum_o;
		param=param_o;
	}
	Task(HelperFn fn,int first_o,int last_o,int flag_o,int master_o, int paramNum_o, void * param_o){
		fnPtr=fn;
		first=first_o;
		last=last_o;
		master=master_o;
		flag=flag_o;
		//reduction=reduction_o;
		paramNum=paramNum_o;
		param=param_o;
	}
	void setFlag(){
		flag=1;
	}
	int isFlagSet(){
		return flag;
	}
};



class FuncSingleHelper: public CBase_FuncSingleHelper{
private:
    CkGroupID nodeHelperID;
    int id;
    Queue reqQ;
    pthread_mutex_t* reqLock;

public:
    FuncSingleHelper(int myid, CkGroupID nid):id(myid),nodeHelperID(nid){
 	//CkPrintf("Single helper %d is created on rank %d\n", myid, CkMyRank());
        reqQ = CqsCreate();
        reqLock = CmiCreateLock();
    }

    ~FuncSingleHelper(){}
    FuncSingleHelper(CkMigrateMessage *m){}
    void enqueueWork(Task *one,unsigned int t){
        //CmiLock(reqLock);
		//unsigned int t;
		//t=(int)(CmiWallTimer()*1000);
		CmiLock(reqLock);
		CqsEnqueueGeneral(reqQ, (void *)one,CQS_QUEUEING_IFIFO,0,&t);
        //SimpleQueuePush(reqQ, (char *)one);
        	CmiUnlock(reqLock);
    }
    void processWork();
    void reportCreated();
};
class FuncNodeHelper : public CBase_FuncNodeHelper{  

public:
	int numHelpers;
	int mode;
	int * counter;
	int threadNum;
	pthread_mutex_t** reqLock;
	CkChareID *helperArr;
    	FuncSingleHelper **helperPtr;
	~FuncNodeHelper(){
        	delete [] helperArr;
        	delete [] helperPtr;
    	}
    
    	void oneHelperCreated(int hid, CkChareID cid, FuncSingleHelper* cptr){
        	helperArr[hid] = cid;
        	helperPtr[hid] = cptr;
    	}
    	
	void  waitDone(Task ** thisReq,int chunck);
	void waitThreadDone(int chunck);
	void createThread();
	FuncNodeHelper(int mode,int elements, int threadNum);
	int parallelizeFunc(HelperFn func, int wps,unsigned int t, int master,int chunck,int time, int paramNum, void * param, int reduction, int type);
	void send(Task *);
	int reduce(Task ** thisReq, int type, int chunck);

};

	
#endif
