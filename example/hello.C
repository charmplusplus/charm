#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "hello.h"

#include "hello.decl.h"


CProxy_Main mainProxy;
CProxy_TestInstance allTestsProxy;
CProxy_FuncNodeHelper nodeHelperProxy;

int iteration;
int flag;// ith iteration
#define MODE 1
int threadNum;
int useNodeQueue;

#define THRESHOLD 100
extern "C" void doCalc(int first,int last, int & result, int paramNum, void * param);

	void work(int start, int end, int &result){
		result=0;
		for(int i=start;i<=end;i++){
			result+=(int)(sqrt(1+cos(i*1.57)));
		}
		//return result;
	}

extern "C" void doCalc(int first, int last, int &result, int paramNum, void * param) {
    	result=0;
	double tstart = CmiWallTimer();
	work(first, last, result);
	tstart = CmiWallTimer() - tstart;
    //printf("start=%d, end=%d, time: %f,result: %d on proc %d\n",first, last, tstart,result,CkMyPe());
}
/*mainchare*/


Main::Main(CkArgMsg* m) {
	//number of elements
        totalElems = 2;
	flag=0;
		for(int i=0;i<4;i++){
			time[i]=0;

			chunck[i]=0;
		}
	//process command line
        if (m->argc >1 ) 
			processCommandLine(m->argc,m->argv);
		else 
			CkPrintf("usage -t(time) -c(chunk) -n(nodequeue) -a (num of tests) -p (thread)\n");
        delete m;
	/*choose mode and initialization
	 mode=0, use pthread;
	 mode=1, use NODEQUEUE;
	 mode=2, not use NODEQUEUE*/
			nodeHelperProxy = CProxy_FuncNodeHelper::ckNew(MODE,totalElems,threadNum);
			
	if(MODE==0){
		
      		FuncNodeHelper *nth = nodeHelperProxy[CkMyNode()].ckLocalBranch();
		nth->createThread();
	}
		int result;
		wps=0;
		calTime=0.05;

        mainProxy = thishandle;
        nodesFinished = 0;
	CkPrintf("useNodeQueue:%d\n",useNodeQueue);
        CProxy_cyclicMap myMap = CProxy_cyclicMap::ckNew();
        CkArrayOptions opts(totalElems);
        opts.setMap(myMap);
        allTestsProxy = CProxy_TestInstance::ckNew(opts);


        //Start the computation
        CkPrintf("Running Hello on %d processors for %d test elements\n",
                 CkNumPes(),totalElems);

        //serial version
		
		calibrated=-1;

		const double end_time = CmiWallTimer()+calTime;
		wps = 0;

		while(CmiWallTimer() < end_time) {
			work(0,100,result);
			wps+=100;
		}
		

		for(int i=0; i < 2; i++) {
			const double start_time = CmiWallTimer();
			work(0,(int)wps,result);
			const double end_time = CmiWallTimer();
			const double correction = calTime / (end_time-start_time);
			wps *= correction;
		}
		calibrated = (int)(wps/calTime);
		printf("calibrated 1: %d\n", calibrated);
		double starttime=CmiWallTimer();
        	work(0,calibrated,result);
		double endtime = CmiWallTimer();
        	printf("Serial time = %lf,result:%d\n",endtime-starttime,result);
		starttime=CmiWallTimer();
        	work(0,calibrated/2500,result);
		endtime = CmiWallTimer();
        	printf("Serial time for 400 micro second= %lf,result:%d\n",endtime-starttime,result);

		
        	CmiSetCPUAffinity(0);
		
		//use node initialization
			
		CkStartQD(CkIndex_Main::doTests((CkQdMsg *)0), &thishandle);
    };
void Main::done(void) {
    nodesFinished++;
    if (nodesFinished < totalElems){
		return;
	}
	else{
		flag++;
		nodesFinished=0;
		if(flag<iteration){
			
			for(int i=0;i<totalElems;i++){
				//CkPrintf("time:%d\n",time[i]);
				allTestsProxy[i].doTest(flag,(calibrated/(1e6/time[i])),chunck[i],time[i]);
			}
			return;
		}
	}
        CkPrintf("All done\n");
        CkExit();
};
    
    void Main:: doTests(CkQdMsg *msg){

		delete msg;
		int wps=calibrated;
		for(int i=0;i<totalElems;i++){
			allTestsProxy[i].doTest(0,(wps/(1e6/time[i])),chunck[i],time[i]);
			//allTestsProxy[8].doTest(0,(wps/(1e6/time[0])),chunck[0],time[0]);
		}
    };
    
    void Main::processCommandLine(int argc,char ** argv){
		int i;
		int j=0;
		int f=0;
		for(i=0;i<argc;i++){
			if(argv[i][0]=='-'){
				switch(argv[i][1]){
					case 't': time[j++]=atoi(argv[++i]);
							  break;
					case 'c': chunck[f++]=atoi(argv[++i]);
							  break;
					case 'n': iteration=atoi(argv[++i]);
							  break;
					case 'a': totalElems=atoi(argv[++i]);
							  break;
					case 'p': threadNum=atoi(argv[++i]);
							  break;
				}
			}
		}
	}



int cmpDFunc(const void *a, const void *b){
        double n1 = *(double *)a;
        double n2 = *(double *)b;
        if(n1<n2) return -1;
        if(n1>n2) return 1;
        return 0;
}

TestInstance::TestInstance() { 
	CkPrintf("test case %d is created on proc %d node %d\n", thisIndex, CkMyPe(),CkMyNode());
		result=0;
		flag=0;
	allTimes=(double *)malloc(sizeof(double)*iteration);
	fflush(stdout);
    }
void TestInstance::doTest(int flag,int wps,int chunck,int time){
      //printf("On proc %d node %d, begin parallel execution for test case %d %dth iteration\n", CkMyPe(), CkMyNode(), thisIndex,flag);    
      	//CkPrintf("wps :%d, flag:%d\n",wps,flag); 
		timerec = CmiWallTimer();
		 int result;
		 if(chunck==0&&time<=THRESHOLD){
			 work(0,wps,result);
			 //printf("On proc %d node %d, Parallel time with %d helpers: %lf for test case %d, result: %d\n", CkMyPe(), CkMyNode(), 0,(CmiWallTimer()-timerec)*1e6, thisIndex,result);
			 
		 }
		 else{

      			FuncNodeHelper *nth = nodeHelperProxy[CkMyNode()].ckLocalBranch();
       			unsigned int t;
			t=(int)(CmiWallTimer()*1000);
			result=nth->parallelizeFunc(doCalc,wps,t,thisIndex,chunck,time,0,NULL, 1,SMP_SUM);
			//printf("On proc %d node %d, Parallel time with %d helpers: %lf for test case %d, result: %d\n", CkMyPe(), CkMyNode(), nth->numHelpers,(CmiWallTimer()-timerec)*1e6, thisIndex,result);
			
		 }
		allTimes[flag]=(CmiWallTimer()-timerec)*1e6;
		if(flag==iteration-1){
			qsort(allTimes,iteration,sizeof(double),cmpDFunc);
			double sumTimes=0.0;
			for(int i=0; i<iteration-3; i++) sumTimes += allTimes[i];
			CkPrintf("result:%d,avg iteration time: %.6f [%.6f, %.6f, %.6f] (us)\n",result, sumTimes/(iteration-3), allTimes[0], allTimes[iteration/2], allTimes[iteration-1]);
		}
		mainProxy.done();
		  

}

#include "hello.def.h"

