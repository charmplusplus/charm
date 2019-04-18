#include <stdlib.h>
#include <unistd.h>
#include "main.decl.h"
#include "PowerLogger.C"
CProxy_PowerLogger pLog;
CProxy_Block block;
CProxy_Main mainProxy;

int freqs[16] = {1200000, 1400000, 1500000, 1700000,    
             1900000, 2000000, 2200000, 2300000,    
             2500000, 2700000, 2800000, 3000000,    
             3200000, 3300000, 3500000, 3501000};   

int changeFreq(int level, FILE* f, FILE* f2){ 
    //check if current frequency is already at the desired level                
    //CkPrintf("[%d,%d]: Change freq to level %d : %d\n", CkMyPe(),CkMyPe()+4, level, freqs[level]);
    //FILE *f, *f2;                                                               
    //char path[300], sibling_path[300];                                          
    //sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",CkMyPe()%4);
    //sprintf(sibling_path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",CkMyPe()%4+4);
    //f=fopen(path,"w");                                                          
    //f2=fopen(sibling_path,"w");                                                 
    //if (f==NULL || f2==NULL) {                                                  
    //    printf("[%d] FILE OPEN ERROR: %s\n", CkMyPe());                   
    //    return 0;                                                               
    //} else {                                                                    
        char write_freq[10];                                                    
        sprintf(write_freq, "%d", freqs[level]);                                
        fputs(write_freq,f);                                                    
        fputs(write_freq,f2);                                                   
        fseek(f, 0, SEEK_SET);
        fseek(f2, 0, SEEK_SET);
        //fclose(f);                                                              
        //fclose(f2);                                                             
        return 1;                                                               
    //}                                                                           
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
void memops(int X, int Y, double** &in, const double f) {
    double** new_grid = new double*[X];
    for(int i=0; i<X; i++){
        new_grid[i] = new double[Y];
        memcpy(new_grid[i], in, sizeof(double)*Y);
    }
    for(int i=0; i<X; i++)
        delete in[i];
    delete in;

    in = new_grid;
}

void structured_grid(int X, int Y, double** &in, const double f) {
    double** new_grid = new double*[X+2];
    for(int i=0; i<X+2; i++)
        new_grid[i] = new double[Y];

    for(int i=0; i<X+2; i++)
        for(int j=0; j<Y; j++)
            new_grid[i][j] = 0;
   
    for(int i=2; i<X-1; i++)
        for(int j=1; j<Y-1; j++)
            new_grid[i][j] = (in[i-1][j]+in[i+1][j]+in[i][j-1]+in[i][j+1]+in[i][j]) * f;

    for(int i=0; i<X; i++)
        delete in[i];
    delete in;

    in = new_grid;
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
    //pLog = CProxy_PowerLogger::ckNew(4);

    block = CProxy_Block::ckNew(blockSize, iteration, CkCallback(CkReductionTarget(Main, ready), thisProxy));
  }

  void ready(CkReductionMsg *msg) {
       block.run2(CkCallback(CkReductionTarget(Main, done), thisProxy));
  }

  void done(CkReductionMsg *msg) {
      CkExit();
  }
};

class Block : public CBase_Block {
  unsigned int blockSize, iteration, curIteration;
  int fLevel;
  double startTime, startEnergy;
  double accumTime, accumEnergy;
  double * dataA;
  double * dataB;
  double * dataC;
  double ** dataD;
  FILE *f, *f2;                                                               
  char path[300], sibling_path[300];                                          
  public:
  Block(unsigned int blockSize_, unsigned int iteration_, CkCallback ready)
    : blockSize(blockSize_), iteration(iteration_)
  {
    unsigned int elems = blockSize * blockSize;
    curIteration=0;
    srand(CkMyPe());
    posix_memalign((void**)&dataA, 2*1024*1024, elems*sizeof(double)); //new double[elems];
    posix_memalign((void**)&dataB, 2*1024*1024, (elems+16)*sizeof(double)); //new double[elems];
    posix_memalign((void**)&dataC, 2*1024*1024, (elems+32)*sizeof(double)); //new double[elems];

    dataD = new double*[blockSize*64];
    for(int i=0; i<blockSize*64; i++)
        dataD[i] = new double[blockSize*64];
    startTime=0; startEnergy=0;
    accumTime=0; accumEnergy=0;
    contribute(ready);
    sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",CkMyPe()%4);
    sprintf(sibling_path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",CkMyPe()%4+4);
    f=fopen(path,"w");                                                          
    f2=fopen(sibling_path,"w");                                                 
    if (f==NULL || f2==NULL) {                                                  
        printf("[%d] FILE OPEN ERROR: %s\n", CkMyPe(), path);                   
    }
    fLevel=14;
    changeFreq(fLevel, f, f2);
  }
  Block(){
  }
  void run(CkCallback cb)
  {
    if(fLevel >= 0 && curIteration%1==0 && curIteration != 0){
        if(CkMyPe()==0){
            CkPrintf("[%d] F: %d ElapsedTime: %f TotalEnergy: %f \n",CkMyPe(), freqs[fLevel], accumTime, accumEnergy);
            if(fLevel==0) CkExit(); 
        }
        fLevel=fLevel-1;
        changeFreq(fLevel, f, f2);
        accumTime=accumEnergy=0;
    }
    //else if(CkMyPe()==0 && curIteration >= iteration) CkExit();
    for(int i=0;i<2;i++)
      usleep(500);

    //if(curIteration == 0 && CkMyPe()==0){
        startTime=CkWallTimer();
	    cpupower_(&startEnergy, 0);
    //}
        
    for(int i=0; i<5; i++){
        structured_grid(blockSize, blockSize, dataD, 0.2);
        //memops(blockSize*64, blockSize*64, dataD, 0.2);
    }
    curIteration+=10;

        double endEnergy=0,divisor=0; double mem_unit=0;
	    cpupower_(&endEnergy, 0);
		getpowerunit_(&divisor, &mem_unit);
        accumEnergy+=(endEnergy-startEnergy)/divisor;
        accumTime+=CkWallTimer()-startTime;

    //if(curIteration >= iteration && CkMyPe()==0){
    //    double endEnergy=0,divisor=0; double mem_unit=0;
	//    cpupower_(&endEnergy, 0);
	//	getpowerunit_(&divisor, &mem_unit);
    //    CkPrintf("[%d] DONE! ElapsedTime: %f, TotalEnergy: %f \n",CkMyPe(), CkWallTimer()-startTime, (endEnergy-startEnergy)/divisor);
    //    CkExit();
    //    //contribute(cb);
    //}
    //else{
        int r=rand()%2;
        //if(r)
            block[CkMyPe()].run(cb);
        //else
        //    block[CkMyPe()].run2(cb);
    //}
  }
  void run2(CkCallback cb)
  {
    if(fLevel >= 0 && curIteration%1==0 & curIteration != 0){
        if(CkMyPe()==0){
            CkPrintf("[%d] F: %d ElapsedTime: %f TotalEnergy: %f \n",CkMyPe(), freqs[fLevel], accumTime, accumEnergy);
            if(fLevel==0) CkExit();
        }
        fLevel=fLevel-1;
        //changeFreq(fLevel, f, f2);
        accumTime=accumEnergy=0;
    }
    for(int i=0;i<2;i++)
      usleep(500);

    //CkPrintf("[%d] Run2-iter %d, ElapsedTime %f \n",CkMyPe(), curIteration, CkWallTimer()-startTime);
    //if(CkMyPe()==0){
        startTime=CkWallTimer();
	    cpupower_(&startEnergy, 0);
    //}
    //changeFreq(8, f, f2); //25
    //changeFreq(7, f, f2); //23
    //for(int i=0; i<2; i++){ // T0.005 w 120 size
    //for(int i=0; i<4; i++){
        example_dgemm(blockSize, blockSize, blockSize, 1.0, dataA, dataB, dataC);
    //}
    curIteration+=10;
    //if(CkMyPe()==0){
        double endEnergy=0,divisor=0; double mem_unit=0;
	    cpupower_(&endEnergy, 0);
		getpowerunit_(&divisor, &mem_unit);
        accumEnergy+=(endEnergy-startEnergy)/divisor;
        accumTime+=CkWallTimer()-startTime;
        //CkPrintf("[%d] F: %d, ElapsedTime: %f, TotalEnergy: %f \n",CkMyPe(), freqs[fLevel], CkWallTimer()-startTime, (endEnergy-startEnergy)/divisor);
        //CkExit();
        //contribute(cb);
    //}
    //else{
        int r=rand()%2;
        //if(r)
        //    block[CkMyPe()].run(cb);
        //else
            block[CkMyPe()].run2(cb);
    //}
  }


  Block(CkMigrateMessage*) {}

};

#include "main.def.h"
