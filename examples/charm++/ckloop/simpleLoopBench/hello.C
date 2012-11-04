#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "hello.h"

#include "hello.decl.h"

#include <omp.h>

#define TEST_REPEAT_TIMES 100

CProxy_Main mainProxy;
CProxy_TestInstance allTestsProxy;
CProxy_FuncCkLoop ckLoopProxy;
int totalElems; //the number of test instances
int loopTimes;
int numChunks;

int threadNum; //number of threads to be used in non-SMP

int cmpDFunc(const void *a, const void *b) {
    double n1 = *(double *)a;
    double n2 = *(double *)b;
    if (n1<n2) return -1;
    if (n1>n2) return 1;
    return 0;
}

void work(int start, int end, void *result) {
    int tmp=0;
    for (int i=start; i<=end; i++) {
        tmp+=(int)(sqrt(1+cos(i*1.57)));
    }
    *(int *)result = tmp;
    
   //CkPrintf("From rank[%d]: start=%d, end=%d, result=%d\n", CkMyRank(), start, end, tmp);
   //fflush(stdout);
}

int openMPWork(int start, int end) {
    int result = 0;
    
    #pragma omp parallel for reduction (+:result)
    for(int i=start; i<=end; i++) {
        result += (int)(sqrt(1+cos(i*1.57)));
    }
    
    return result;
}

extern "C" void doCalc(int first, int last, void *result, int paramNum, void * param) {    
    //double tstart = CkWallTimer();
    
	work(first, last, result);
    
	//tstart = CkWallTimer() - tstart;
    //printf("start=%d, end=%d, time: %f,result: %d on proc %d\n",first, last, tstart,result,CkMyPe());
}

/*mainchare*/
Main::Main(CkArgMsg* m) {
    
	//default values	
    totalElems = 1;
	numChunks = CkMyNodeSize();
	loopTimes = 1000;
	
    mainStep = 0;
	numElemFinished = 0;
	
    curTestMode = 0;
    
    //process command line
    if (m->argc >1 ){
        processCommandLine(m->argc,m->argv);
	}
    else{		
		CkPrintf("Usage: -t(#iterations) -c(#chunks) -a(#test instances) -m(running mode, 1 for use Charm threads; 2 for use pthreads )  -p(#threads)\n");
	}
    delete m;
	
    omp_set_num_threads(numChunks);    
    
	mainTimes = new double[TEST_REPEAT_TIMES];
	memset(mainTimes, 0, sizeof(double)*TEST_REPEAT_TIMES);
	
	CkPrintf("Using CkLoop Lib: nodesize=%d\n", CkMyNodeSize());
	CkPrintf("Testcase info: %d test instances where the loop iterates %d times, each work is partitioned into %d tasks\n", totalElems, loopTimes, numChunks);
	
	ckLoopProxy = CkLoop_Init(threadNum);
	//ckLoopProxy = CkLoop_Init();
    mainProxy = thishandle;
    
	//create test instances
    CProxy_cyclicMap myMap = CProxy_cyclicMap::ckNew();
    CkArrayOptions opts(totalElems);
    opts.setMap(myMap);
    allTestsProxy = CProxy_TestInstance::ckNew(opts);

    //serial version
	int result;
	double starttime, endtime;
	for(int i=0; i<3; i++){
		starttime = CkWallTimer();
		work(0, loopTimes, &result);		
		endtime = CkWallTimer();
		CkPrintf("Calibration %d: the loop takes %.3f us with result %d\n", i+1,  (endtime-starttime)*1e6, result);
	}
	int results[5];
	starttime = CkWallTimer();
	for(int i=0; i<5; i++) work(0, loopTimes, results+i);
	endtime = CkWallTimer();
	double avgtime = (endtime-starttime)*1e6/5; //in the unit of us
	CkPrintf("Calibration: avg time %.3f us of 5 consecutive runs, so a 100us-loop will iterate %d times\n", avgtime, (int)(loopTimes*100.0/avgtime));
		
    CmiSetCPUAffinity(0);
    CkStartQD(CkIndex_Main::doTests((CkQdMsg *)0), &thishandle);
};

void Main::done(void) {
    numElemFinished++;
    if (numElemFinished < totalElems) {
        return;
    } else {
		mainTimes[mainStep] = (CkWallTimer() - timestamp)*1e6;
        mainStep++;
        numElemFinished=0;
        if (mainStep < TEST_REPEAT_TIMES) {
			doTests(NULL);
            return;
        }
    }	
    
	//do some final output
	allTestsProxy[0].reportSts();
};

void Main::exitTest(){
	//do some final output
	qsort(mainTimes, TEST_REPEAT_TIMES, sizeof(double), cmpDFunc);
	double sum = 0.0;
	for(int i=0; i<TEST_REPEAT_TIMES-3; i++) sum += mainTimes[i];
	int maxi = TEST_REPEAT_TIMES;
	CkPrintf("Global timestep info: avg time: %.3f [%.3f, %.3f, %.3f] (us)\n", sum/(maxi-3), mainTimes[0], mainTimes[maxi/2], mainTimes[maxi-1]);
	
    if(curTestMode == 0){
	    CkPrintf("Charm++ CkLoop Test done\n\n");
        curTestMode++;
        mainStep = 0;
        numElemFinished = 0;
        doTests(NULL);
    }else if(curTestMode == 1){        
        CkPrintf("OpenMP Test done\n");
        CkExit();
    }
}

void Main::doTests(CkQdMsg *msg) {
    delete msg;

    //CkPrintf("===========Starting mainstep %d===========\n", mainStep);    

    if(mainStep == 0){
        if(curTestMode == 0){
            CkPrintf("===Start CkLoop Test===\n");
        }else if(curTestMode == 1){
            int numthds = 0;
            int openmpid;
            #pragma omp parallel private(openmpid)
            {
                openmpid = omp_get_thread_num();
                if(openmpid == 0) numthds = omp_get_num_threads();
            }
            CkPrintf("===Start OpenMP Test with %d threads===\n", numthds);
        }
    }
    
	timestamp = CkWallTimer(); //record the start time of the whole test
    for (int i=0; i<totalElems; i++) {
        allTestsProxy[i].doTest(mainStep, curTestMode);
        //allTestsProxy[8].doTest(mainStep, curTestMode);
    }
};

void Main::processCommandLine(int argc,char ** argv) {
    for (int i=0; i<argc; i++) {
        if (argv[i][0]=='-') {
            switch (argv[i][1]) {
            case 't':
                loopTimes = atoi(argv[++i]);
                break;
            case 'c':
                numChunks = atoi(argv[++i]);
                break;
            case 'a':
                totalElems = atoi(argv[++i]);
                break;
            case 'p':
                threadNum = atoi(argv[++i]);
                break;
            }
        }
    }
}


TestInstance::TestInstance() {
    CkPrintf("test case %d is created on proc %d node %d\n", thisIndex, CkMyPe(),CkMyNode());
    
	hasTest = 0; 
	allTimes = new double[TEST_REPEAT_TIMES];
	allResults = new int[TEST_REPEAT_TIMES];
	memset(allTimes, 0, sizeof(double)*TEST_REPEAT_TIMES);
	memset(allResults, 0, sizeof(int)*TEST_REPEAT_TIMES);
}

void TestInstance::doTest(int curstep, int curTestMode) {
    //printf("On proc %d node %d, begin parallel execution for test case %d %dth iteration\n", CkMyPe(), CkMyNode(), thisIndex,flag);
	hasTest = 1;
	int result;
	
    double timerec = CkWallTimer();
    
    if(curTestMode == 0){
	    CkLoop_Parallelize(doCalc, 0, NULL, numChunks, 0, loopTimes-1, 1, &result, CKLOOP_INT_SUM);
    }else if(curTestMode == 1){
        result = openMPWork(0, loopTimes-1);
    }
    
    allTimes[curstep]=(CkWallTimer()-timerec)*1e6;
	allResults[curstep] = result;
	
    mainProxy.done();
}

void TestInstance::reportSts(){
	if(hasTest){
		//do sts output
		qsort(allTimes, TEST_REPEAT_TIMES, sizeof(double), cmpDFunc);
		double sum = 0.0;
		for(int i=0; i<TEST_REPEAT_TIMES-3; i++) sum += allTimes[i];
		
		double avgResult = 0.0;
		for(int i=0; i<TEST_REPEAT_TIMES; i++) avgResult += allResults[i];
		avgResult /= TEST_REPEAT_TIMES;
		
		int maxi = TEST_REPEAT_TIMES;
		CkPrintf("Test instance[%d]: result:%.3f, avg time: %.3f [%.3f, %.3f, %.3f] (us)\n",thisIndex, avgResult, sum/(maxi-3), allTimes[0], allTimes[maxi/2], allTimes[maxi-1]);	    
    }
	
	if(thisIndex == totalElems-1) mainProxy.exitTest();
	else thisProxy[thisIndex+1].reportSts();
}

#include "hello.def.h"

