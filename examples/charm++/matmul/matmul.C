#include "matmul.decl.h"
#include "rand48_replacement.h"
#include "papi.C"
CProxy_Main mainProxy;
/* readonly */ CProxy_PAPI_grp papi_arr;
#if 0
/*readonly*/ CProxy_InputBlock inputProxy_a;
/*readonly*/ CProxy_InputBlock inputProxy_b;
#endif
static int print = 0;
void example_dgemm(int M, int N, int K, double alpha,
                   double *A, double *B, double *C) {
  double kernl_start_time;
  if(print == 1) {
    kernl_start_time = CkWallTimer();
  }
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A[i*K + k] * B[k*N + j];
      }
      C[N*i + j] = C[N*i + j] + alpha*sum;
    }
  }
  if(print == 1) {
    CkPrintf("\nKernel time = %lf s", CkWallTimer()-kernl_start_time);
  }
}

class Main : public CBase_Main {
  double startTime;
  unsigned int blockSize, numBlocks, dun;
  CProxy_Block a, b, c;
  PAPI_grp* papi;
public:
  Main(CkArgMsg* m) {
    dun = 0;
    if (m->argc > 2) {
      blockSize = atoi(m->argv[1]);
      numBlocks = atoi(m->argv[2]);
    } else {
      CkPrintf("Usage: matmul blockSize numBlocks");
      CkExit(1);
    }

    mainProxy = thisProxy;
    papi_arr = CProxy_PAPI_grp::ckNew();
#if 0
    CProxy_MatMap map = CProxy_MatMap::ckNew(numBlocks, numBlocks);
    CkArrayOptions opts(numBlocks, numBlocks);
    opts.setMap(map);
#endif
//    inputProxy_a = CProxy_InputBlock::ckNew(true, blockSize, numBlocks, opts);//numBlocks, numBlocks);
//    inputProxy_b = CProxy_InputBlock::ckNew(true, blockSize, numBlocks, opts);//numBlocks, numBlocks);
    c = CProxy_Block::ckNew(false, blockSize, numBlocks, numBlocks, numBlocks);

    set_active_pes(CkNodeSize(CkMyNode()));
    set_active_redn_pes(CkNodeSize(CkMyNode()));
    startTime = CkWallTimer();

    c.pdgemmSendInput(c, true);
    c.pdgemmSendInput(c, false);
    c.pdgemmRun(1.0, CkCallback(CkReductionTarget(Main, done), thisProxy));
  }
  
  void done() {
    double endTime = CkWallTimer();
  //  if(dun==1) {
    //  print = 1;
      papi = papi_arr.ckLocalBranch();
 ///   } else print = 0;
    CkPrintf("D1[Trial %d]Matrix multiply of %u blocks with %u elements each (%u^2) finished in %f seconds\n", papi->iter, 
             numBlocks, blockSize, numBlocks*blockSize, endTime - startTime);
    startTime = endTime;
    
 //   if(dun==2) papi->start_energy(CkWallTimer());
 //   if(dun==3) papi->stop_energy(CkWallTimer());
    c.pdgemmSendInput(c, true);
    c.pdgemmSendInput(c, false);

    c.pdgemmRun(1.0, CkCallback(CkReductionTarget(Main, done), thisProxy));
    if(dun++ == 80)
    CkExit();
  }
};

class Block : public CBase_Block {
  unsigned int blockSize, numBlocks, block;
  int iteration,ack;
  int dunb;
  double start_time;
  double* data;
  double* inpA_data;
  double* inpB_data;
  PAPI_grp* papi;
  Block_SDAG_CODE
  public:
  Block(bool randomInit, unsigned int blockSize_, unsigned int numBlocks_)
    : blockSize(blockSize_), numBlocks(numBlocks_)
  {
    unsigned int elems = blockSize * blockSize;
    usesAtSync = true;
    dunb = 0;
    ack = 0;
    iteration = 0;
    papi = papi_arr.ckLocalBranch();
    inpA_data = new double[elems];
    inpB_data = new double[elems];
    data = new double[elems];
    for (int i = 0; i < elems; ++i) {
      inpA_data[i] = drand48();
      inpB_data[i] = drand48();
      data[i] = 0;
    }

  }

  void pup(PUP::er &p)
    {
      p(dunb);
      p(ack);
      p(blockSize);
      p(numBlocks);
      p(block);
      p(iteration);
      p(start_time);
      int elems = blockSize * blockSize;
      if(p.isUnpacking()) {
        data = new double[elems];
        inpA_data = new double[elems];
        inpB_data = new double[elems];
      }
      PUParray(p, data, elems);
      PUParray(p, inpA_data, elems);
      PUParray(p, inpB_data, elems);
    }

  void doBeforeAtSync() {
    ack++;
    if(ack==numBlocks*numBlocks) {
      ack = 0;
      thisProxy.doAtSync();
    }
  }
  void doAtSync() {
    AtSync();
  }
  void resume(){
    ack++;
    if(ack==numBlocks*numBlocks) {
      CkPrintf("D2[Trial %d]Matrix multiply of %u blocks with %u elements each (%u^2) finished in %f seconds\n", papi->iter,
             numBlocks, blockSize, numBlocks*blockSize);
    dunb++;

      ack = 0;
      thisProxy.pdgemmSendInput(thisProxy, true);
      thisProxy.pdgemmSendInput(thisProxy, false);

      thisProxy.pdgemmRun(1.0, CkCallback(CkReductionTarget(Main, done), mainProxy));
    }
  }
  void resume0(){
      CkPrintf("D2[Trial %d]Matrix multiply of %u blocks with %u elements each (%u^2) finished in %f seconds\n", papi->iter,
             numBlocks, blockSize, numBlocks*blockSize, CkWallTimer()-start_time);
      start_time = CkWallTimer();
    dunb++;

      thisProxy.pdgemmSendInput(thisProxy, true);
      thisProxy.pdgemmSendInput(thisProxy, false);

      thisProxy.pdgemmRun(1.0, CkCallback(CkReductionTarget(Main, done), mainProxy));
  }
  Block(CkMigrateMessage*) {}
  void ResumeFromSync() {
    papi = papi_arr.ckLocalBranch();
    if(CkMyPe()==0 && papi->wakeup == 1) papi->wakeup = 0;
    CkCallback cb(CkReductionTarget(Block, resume0), thisProxy(0,0));
    contribute(cb);
    //thisProxy(0,0).resume();
   
  }
};

#include "matmul.def.h"
