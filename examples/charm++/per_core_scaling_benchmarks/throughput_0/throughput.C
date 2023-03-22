#include <stdlib.h>
#include <unistd.h>
#include <papi.h>
#include "throughput.decl.h"
#include "PowerLogger.C"

CProxy_Main mainProxy;
CProxy_PowerLogger pLog;
int procMap[8] = {1,2,3,4,5,6,7,8}; //this maps the core numbers to the temp files
//valid only for Edison

#define TEMP 1
#define PAPI 1
/*
static float getTemp(int cpu)
{
  char val[10];
  FILE *f;
  char path[300];
  sprintf(path,"/sys/devices/platform/coretemp.%d/temp%d_input",0,procMap[cpu%4]);
  f=fopen(path,"r");
  if (f==NULL) {
    printf("[%d] FILE OPEN ERROR in temp :%s\n",CkMyPe(),path);
  }
  else
  {
    fgets(val,10,f);
    fclose(f);
  }
  return atof(val)/1000;
}
*/
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
  CProxy_Block block;
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
  unsigned int blockSize, iteration;
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
#if TEMP
  FILE * temp_log;
#endif
#if PAPI
  long_long values[4];
  long_long elapsed_cyc;
#endif
  public:
  Block(unsigned int blockSize_, unsigned int iteration_, CkCallback ready)
    : blockSize(blockSize_), iteration(iteration_)
  {
    unsigned int elems = blockSize * blockSize;
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
#if PAPI
    setenv("PAT_RT_PERFCTR", "PAPI_TOT_CYC,PAPI_REF_CYC,PAPI_L3_TCM,PAPI_L2_TCM", 1);
    for (int i = 0; i<4; ++i) values[i] = 0;
#endif

#if TEMP
    char file_name[100];
    sprintf(file_name, "%d_temp_log", CkMyNode());
    temp_log = fopen(file_name,"w");
    fprintf(temp_log, "Time\tTemperature\tIntervalTime\tTOTAL_CYC\tREF_CYC\n");
#endif
    avg_temp = 0; temp_points = 0;
    contribute(ready);
  }
  Block(){
#if TEMP
    fclose(temp_log);
#endif
  }
  void run(CkCallback cb)
  {
#if PAPI
    int retval, Events[4] = {PAPI_TOT_CYC, PAPI_REF_CYC, PAPI_L3_TCM,PAPI_L2_TCM};
#endif
    for(int j=0; j<4; ++j) values[j] = 0;
    for(int i=0; i<0; i++){
    	example_dgemm(blockSize, blockSize, blockSize, 1.0, dataA, dataB, dataC); // warmup
    }
#if PAPI
    //PAPI START
/*
    if (PAPI_start_counters (Events, 4) != PAPI_OK) {
      printf("Error starting counters\n");
      CkExit();
    }
    elapsed_cyc = PAPI_get_real_cyc();
*/
#endif
	//writeFreq(CkMyNode(), 12);
    double startTime = CkWallTimer();
    double startCpuTime = CmiCpuTimer();
    FILE* temp_file;
    double before_temp = getTemp(CkMyNode());
    double last_time_interval = CkWallTimer();
    for(int i=0; i<iteration; i++){
      example_dgemm(blockSize, blockSize, blockSize, 1.0, dataA, dataB, dataC);
#if TEMP
      if(i%1 == 0 && i!= 0){
        temp_points++;
        cur_temp = getTemp(CkMyNode());
        avg_temp += cur_temp;
		/*if(PAPI_accum_counters(values, 4) != PAPI_OK) {
	 	 printf("Error reading counters\n");
	  	 CkExit();
		}*/
		fprintf(temp_log, "%f\t %f\t %f\t %lld\t %lld\t\n", CkWallTimer(), cur_temp, CkWallTimer()-last_time_interval, values[0], values[1]);
		last_time_interval = CkWallTimer();
        for (int j = 0; j<4; ++j) values[j] = 0;
      }
	  //if(i%100 == 0)
	  //      writeFreq(CkMyNode(), 35);
	  //}// else if(i%50 == 0 ){
	//	writeFreq(CkMyNode(), 35);
	  //}
#endif
    }

    double after_temp = getTemp(CkMyNode());
    double totalTime = CkWallTimer() - startTime;
    double totalCpuTime = CmiCpuTimer() - startCpuTime;
#if PAPI
    //PAPI END
    /*if (PAPI_stop_counters (values, 4) != PAPI_OK) {
      printf("Error stopping counters\n");
      CkExit();
    }
    elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;
    double cycles_error=100.0*((double)values[1] - (double)elapsed_cyc)/(double)elapsed_cyc;
    if ((cycles_error>5.0) || (cycles_error<-5.0)) {
  	printf("Error of %.2f%%\n",cycles_error);
	//test_warn( __FILE__, __LINE__, "validation", 0 );
    }*/
#endif

    for(int i=0; i<iteration; i++){
      example_dgemm(blockSize, blockSize, blockSize, 1.0, dataA, dataB, dataC); //close up
    }

#if !PAPI
    char* msg = (char *)malloc(296);
    ((double *)msg)[0] = CkMyNode();
    ((double *)msg)[1] = before_temp;
    ((double *)msg)[2] = after_temp;
    ((double *)msg)[3] = (avg_temp/(double)temp_points);
    ((double *)msg)[4] = totalTime;
    char host_name[256];
    if(gethostname(&(msg[40]), sizeof(host_name)) != 0) CkAbort("gethostname error!");
    contribute(296, msg, CkReduction::set, cb);
#else
    char* msg = (char *)malloc(328);
    ((double *)msg)[0] = CkMyNode();
    ((double *)msg)[1] = before_temp;
    ((double *)msg)[2] = after_temp;
    ((double *)msg)[3] = (avg_temp/(double)temp_points);
    ((double *)msg)[4] = totalTime;
    ((double *)msg)[5] = values[0];
    ((double *)msg)[6] = values[1];
    ((double *)msg)[7] = values[2];
    ((double *)msg)[8] = values[3];
    char host_name[256];
    if(gethostname(&(msg[72]), sizeof(host_name)) != 0) CkAbort("gethostname error!");
    contribute(328, msg, CkReduction::set, cb);
#endif
  }

  Block(CkMigrateMessage*) {}

};

#include "throughput.def.h"
