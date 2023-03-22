#include <stdlib.h>
#include <unistd.h>
#include <papi.h>
#include "throughput.decl.h"
#include "PowerLogger.C"

CProxy_Main mainProxy;
CProxy_Block block;
CProxy_PowerLogger pLog;

#define TEMP 1

static int writeFreq(int cpu, int freq){
  FILE *f;
  char path[300];
  sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",cpu%4);
  f=fopen(path,"w");
  if (f==NULL) {
    printf("[%d] FILE OPEN ERROR in temp :%s\n",CkMyPe(),path);
    return 0;
  }
  else
  {
    char write_freq[10];
    sprintf(write_freq, "%d", freq*100000);
    fputs(write_freq,f);
    fclose(f);
    //return 1;
  }
  sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",cpu%4+4);
  f=fopen(path,"w");
  if (f==NULL) {
    printf("[%d] FILE OPEN ERROR in temp :%s\n",CkMyPe(),path);
    return 0;
  }
  else
  {
    char write_freq[10];
    sprintf(write_freq, "%d", freq*100000);
    fputs(write_freq,f);
    fclose(f);
    return 1;
  }


}

void example_dgemm(int M, int N, int K, double alpha,
                   const double *A, const double *B, double *C) {
                   //const double * __restrict__ A, const double *__restrict__ B, double *__restrict__ C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A[i*K + k] * B[k*N + j];
      }
      C[N*i + j] = C[N*i + j] + alpha*sum;
    }
  }
}

class Main : public CBase_Main {
  unsigned int blockSize, iteration;
public:
  Main(CkArgMsg* m) {
    if (m->argc > 2) {
      blockSize = atoi(m->argv[1]);
      iteration = atoi(m->argv[2]);
    } else {
      CkAbort("Usage: ./throughput blockSize iteration");
    }
    CkPrintf("Running throughput with blockSize:%d iteration:%d\n", blockSize, iteration);

    mainProxy = thisProxy;
    pLog = CProxy_PowerLogger::ckNew(4);

    block = CProxy_Block::ckNew(blockSize, iteration, CkCallback(CkReductionTarget(Main, ready), thisProxy));
  }

  void ready(CkReductionMsg *msg) {
       block.run(CkCallback(CkReductionTarget(Main, done), thisProxy));
  }

  void done(CkReductionMsg *msg) {
    CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
    FILE* myfile;
    myfile = fopen("output_log_temp","w");
    fprintf(myfile, "GroupID\t Before_temp\t After_temp\t Avg_temp\t Time \tPAPI_TOT_CYC\t PAPI_REF_CYC \t PAPI_L3_TCM \t PAPI_L2_TCM \t Host_Name - Block_Size: %d\n", blockSize);
    while(current != NULL)
    {
      double* result = (double*) &current->data;
      fprintf(myfile, "%d\t %d\t %d\t %d\t %f\t %f\t %f\t %f\t %f\t  %s\n", (int)result[0], (int)result[1], (int)result[2], (int)result[3], result[4],
		result[5], result[6], result[7], result[8],
	 	(char*)&(result[9]));
      current = current->next();
    }
    fclose(myfile);
    CkExit();
  }
};

class Block : public CBase_Block {
  unsigned int blockSize, iteration, curIteration;
  double time;
//  double * __restrict__ dataA;
//  double * __restrict__ dataB;
//  double * __restrict__ dataC;
  double * dataA;
  double * dataB;
  double * dataC;
  double avg_temp;
  int temp_points;
  double cur_temp;
  public:
  Block(unsigned int blockSize_, unsigned int iteration_, CkCallback ready)
    : blockSize(blockSize_), iteration(iteration_)
  {
    unsigned int elems = blockSize * blockSize;
    curIteration=0;
#if 1
    posix_memalign((void**)&dataA, 2*1024*1024, elems*sizeof(double)); //new double[elems];
    posix_memalign((void**)&dataB, 2*1024*1024, (elems+16)*sizeof(double)); //new double[elems];
    posix_memalign((void**)&dataC, 2*1024*1024, (elems+32)*sizeof(double)); //new double[elems];
    //dataB = dataB + 16;
    //dataC = dataC + 32;
#else
    dataA = new double[elems];
    dataB = new double[elems];
    dataC = new double[elems];
#endif

    for (int i = 0; i < elems; ++i){
      dataA[i] = drand48();
      dataB[i] = drand48();
      dataC[i] = 0;
    }
    avg_temp = 0; temp_points = 0;
    contribute(ready);
  }
  Block(){
  }
  void run(CkCallback cb)
  {
    for(int i=0; i<0; i++){
    	example_dgemm(blockSize, blockSize, blockSize, 1.0, dataA, dataB, dataC); // warmup
    }
    double startTime = CkWallTimer();
    double startCpuTime = CmiCpuTimer();
    double before_temp = getTemp(CkMyNode());
    double last_time_interval = CkWallTimer();
    for(int i=0; i<10; i++){
      example_dgemm(blockSize, blockSize, blockSize, 1.0, dataA, dataB, dataC);
    }
    curIteration+=10;
    if(curIteration >= iteration){
        double after_temp = getTemp(CkMyNode());
        double totalTime = CkWallTimer() - startTime;
        double totalCpuTime = CmiCpuTimer() - startCpuTime;
        for(int i=0; i<0; i++){
          example_dgemm(blockSize, blockSize, blockSize, 1.0, dataA, dataB, dataC); //close up
        }

        char* msg = (char *)malloc(328);
        ((double *)msg)[0] = CkMyNode();
        ((double *)msg)[1] = before_temp;
        ((double *)msg)[2] = after_temp;
        ((double *)msg)[3] = (avg_temp/(double)temp_points);
        ((double *)msg)[4] = totalTime;
        ((double *)msg)[5] = 0;
        ((double *)msg)[6] = 0;
        ((double *)msg)[7] = 0;
        ((double *)msg)[8] = 0;
        char host_name[256];
        if(gethostname(&(msg[72]), sizeof(host_name)) != 0) CkAbort("gethostname error!");
        contribute(328, msg, CkReduction::set, cb);
    }
    else{
        block[CkMyPe()].run(cb);
    }
  }

  Block(CkMigrateMessage*) {}

};

#include "throughput.def.h"
