#include "matmul.decl.h"
#include "ckcoro.h"
#include <random>

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
      CkPrintf("Usage: matmul blockSize numBlocks");
      CkExit(1);
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

  void done(double checksum) {
    double endTime = CkWallTimer();
    CkPrintf("Matrix multiply of %u blocks with %u elements each (%u^2) finished in %f seconds\n",
             numBlocks, blockSize, numBlocks*blockSize, endTime - startTime);
    CkPrintf("matmul: checksum %.6f\n", checksum);
    CkExit();
  }
};

class Block : public CBase_Block {
  Block_CORO_CODE
  unsigned int blockSize, numBlocks;
  double* data;
  // Outs of the wait in step(): members, as every wait out is. The tasks of
  // the forall share them; a task's code from its wait to its end runs
  // without a suspension, so one task uses its values before another
  // task's wait overwrites them. The pointers borrow the delivered messages
  // until the task's next wait (there is none: the task ends).
  int blockIdA, blockIdB;
  double *blockA, *blockB;
  unsigned int M, KA, KB, N;

  public:
  Block(bool randomInit, unsigned int blockSize_, unsigned int numBlocks_)
    : blockSize(blockSize_), numBlocks(numBlocks_)
  {
    unsigned int elems = blockSize * blockSize;
    data = new double[elems];
    // Random input that depends only on the element's index, so a run's
    // checksum does not depend on the number of PEs or on the order in
    // which the elements are constructed.
    if(randomInit) {
      std::minstd_rand gen(thisIndex.x * numBlocks + thisIndex.y + 1);
      std::uniform_real_distribution<double> uniform(0.0, 1.0);
      for (int i = 0; i < elems; ++i)
        data[i] = uniform(gen);
    } else
      for (int i = 0; i < elems; ++i)
        data[i] = 0;

  }

  // Allocates nothing: pup allocates data when unpacking.
  Block(CkMigrateMessage* m)
    : CBase_Block(m), blockSize(0), numBlocks(0), data(nullptr) {}
  ~Block() { delete[] data; }

  void pup(PUP::er &p) {
    p | blockSize;
    p | numBlocks;
    if (p.isUnpacking()) data = new double[blockSize * blockSize];
    PUParray(p, data, blockSize * blockSize);
  }

  void pdgemmSendInput(CProxy_Block output, bool aOrB) {
    if (aOrB)
      output((thisIndex.x-thisIndex.y+numBlocks)%numBlocks, thisIndex.y).inputA(0, data, blockSize, blockSize);
    else
      output(thisIndex.x, (thisIndex.y-thisIndex.x+numBlocks)%numBlocks).inputB(0, data, blockSize, blockSize);
  }

  // The body of forall [block]:
  //   when inputA[block](...), inputB[block](...) serial { ... }
  ck::coro::task<void> step(int block, double alpha) {
    CK_CORO_TASK();
    CK_WHEN_ALL(CK_ALT(inbox::inputA(block), blockIdA, blockA, M, KA),
                CK_ALT(inbox::inputB(block), blockIdB, blockB, KB, N));
    CkAssert(KA == KB);

    // This DGEMM makes no attempt at optimization. For
    // benchmarking, it should be replaced with a library
    // call. The local version is kept for easy portability.
    example_dgemm(M, N, KA,
                  alpha,
                  blockA, blockB, data);

    if (blockIdA != numBlocks) {
      int destX = (thisIndex.x + 1) % numBlocks;
      int destY = thisIndex.y;
      thisProxy(destX, destY).inputA(blockIdA+1, blockA, M, KA);
    }
    if (blockIdB != numBlocks) {
      int destX = thisIndex.x;
      int destY = (thisIndex.y + 1) % numBlocks;
      thisProxy(destX, destY).inputB(blockIdB+1, blockB, KB, N);
    }
    CK_CORO_TASK_END();
  }

  ck::coroutine pdgemmRun(double alpha, CkCallback done) {
    CK_COROUTINE();
    CK_FORALL(block, 0, (int)numBlocks, step(block, alpha));
    {
      double sum = 0.0;
      for (unsigned int i = 0; i < blockSize*blockSize; ++i) sum += data[i];
      contribute(sizeof(double), &sum, CkReduction::sum_double, done);
    }
    CK_COROUTINE_END();
  }
};

#include "matmul.def.h"
