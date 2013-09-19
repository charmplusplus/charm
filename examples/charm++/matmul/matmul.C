#include "matmul.decl.h"
#include "rand48_replacement.h"

CProxy_Main mainProxy;

void example_dgemm(int M, int N, int K, double alpha,
                   double *A, double *B, double *C) {
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
  double startTime;
  unsigned int blockSize, numBlocks;
  CProxy_Block a, b, c;
public:
  Main(CkArgMsg* m) {
    if (m->argc > 2) {
      blockSize = atoi(m->argv[1]);
      numBlocks = atoi(m->argv[2]);
    } else {
      CkAbort("Usage: matmul blockSize numBlocks");
    }

    mainProxy = thisProxy;

    a = CProxy_Block::ckNew(true, blockSize, numBlocks, numBlocks, numBlocks);
    b = CProxy_Block::ckNew(true, blockSize, numBlocks, numBlocks, numBlocks);
    c = CProxy_Block::ckNew(false, blockSize, numBlocks, numBlocks, numBlocks);

    startTime = CkWallTimer();

    a.pdgemmSendInput(c, true);
    b.pdgemmSendInput(c, false);
    c.pdgemmRun(1.0, CkCallback(CkReductionTarget(Main, done), thisProxy));
  }

  void done() {
    double endTime = CkWallTimer();
    CkPrintf("Matrix multiply of %u blocks with %u elements each (%u^2) finished in %f seconds\n", 
             numBlocks, blockSize, numBlocks*blockSize, endTime - startTime);
    CkExit();
  }
};

class Block : public CBase_Block {
  unsigned int blockSize, numBlocks, block;
  double* data;
  Block_SDAG_CODE
  public:
  Block(bool randomInit, unsigned int blockSize_, unsigned int numBlocks_)
    : blockSize(blockSize_), numBlocks(numBlocks_)
  {
    unsigned int elems = blockSize * blockSize;
    data = new double[elems];
    if(randomInit)
      for (int i = 0; i < elems; ++i)
        data[i] = drand48();
    else
      for (int i = 0; i < elems; ++i)
        data[i] = 0;

  }

  Block(CkMigrateMessage*) {}

};

#include "matmul.def.h"
