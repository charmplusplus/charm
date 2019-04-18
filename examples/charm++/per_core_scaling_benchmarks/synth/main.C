#include <stdlib.h>
#include <unistd.h>
#include "main.decl.h"
#include "PowerLogger.C"
CProxy_PowerLogger pLog;
CProxy_Block block;
CProxy_Main mainProxy;

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
  double startTime;
  double startEnergy;
  double * dataA;
  double * dataB;
  double * dataC;
  double ** dataD;
  int blockSizeD;
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
    blockSizeD = blockSize*64;
    dataD = new double*[blockSizeD];
    for(int i=0; i<blockSizeD; i++)
        dataD[i] = new double[blockSizeD];
    startTime=0;
    startEnergy=0;
    contribute(ready);
  }
  Block(){
  }
  void run(CkCallback cb)
  {
    //while(1){}
    CkPrintf("[%d] Run-iter %d, ElapsedTime %f \n",CkMyPe(), curIteration, CkWallTimer()-startTime);
    if(curIteration == 0 && CkMyPe()==0){
        startTime=CkWallTimer();
	    cpupower_(&startEnergy, 0);
    }
        
    for(int i=0; i<9; i++){
        //structured_grid(blockSize, blockSize, dataD, 0.2);
        memops(blockSizeD, blockSizeD, dataD, 0.2);
    }
    curIteration+=10;
    if(curIteration >= iteration && CkMyPe()==0){
        double endEnergy=0,divisor=0; double mem_unit=0;
	    cpupower_(&endEnergy, 0);
		getpowerunit_(&divisor, &mem_unit);
        CkPrintf("[%d] DONE! ElapsedTime: %f, TotalEnergy: %f \n",CkMyPe(), CkWallTimer()-startTime, (endEnergy-startEnergy)/divisor);
        CkExit();
        //contribute(cb);
    }
    else{
        int r=rand()%2;
        if(r)
            block[CkMyPe()].run(cb);
        else
            block[CkMyPe()].run2(cb);
    }
  }
  void run2(CkCallback cb)
  {
    CkPrintf("[%d] Run2-iter %d, ElapsedTime %f \n",CkMyPe(), curIteration, CkWallTimer()-startTime);
    if(curIteration == 100 && CkMyPe()==0){
        startTime=CkWallTimer();
	    cpupower_(&startEnergy, 0);
    }
    for(int i=0; i<400; i++){
        example_dgemm(blockSize, blockSize, blockSize, 1.0, dataA, dataB, dataC);
    }
    curIteration+=10;
    if(curIteration >= iteration && CkMyPe()==0){
        double endEnergy=0,divisor=0; double mem_unit=0;
	    cpupower_(&endEnergy, 0);
		getpowerunit_(&divisor, &mem_unit);
        CkPrintf("[%d] DONE! ElapsedTime: %f, TotalEnergy: %f \n",CkMyPe(), CkWallTimer()-startTime, (endEnergy-startEnergy)/divisor);
        CkExit();
        //contribute(cb);
    }
    else{
        int r=rand()%2;
        if(r)
            block[CkMyPe()].run(cb);
        else
            block[CkMyPe()].run2(cb);
    }
  }


  Block(CkMigrateMessage*) {}

};

#include "main.def.h"
