/*
 * 3D Jacobi relaxation with the step written as a coroutine entry method
 * (ckcoro.h): the counterpart of tests/charm++/jacobi3d-sdag, whose doStep
 * is an SDAG entry. Same decomposition, messages and arithmetic.
 *
 *   ./jacobi3d arrayDim blockDim [maxIter [ckptFreq [lbFreq]]] [-chkpt ITER DIR]
 *
 * Every ckptFreq iterations the elements report to Main, which starts the
 * next round (as jacobi3d-sdag does). With lbFreq > 0 every element calls
 * AtSync every lbFreq iterations inside its coroutine and waits for
 * ResumeFromSync at a when site, so with +balancer the elements migrate while
 * their coroutines are suspended. With -chkpt, after iteration ITER the
 * elements ask Main for a checkpoint to DIR and wait at a when site for the
 * resume; a later run with +restart DIR continues from there. The final line
 * "jacobi3d: N iterations, checksum C" is the same for every PE count, with
 * or without migration, rebuild (+ck_fakemigrate) and restart.
 */
#include <string>
#include "jacobi3d.decl.h"
#include "ckcoro.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int arrayDimZ;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;
/*readonly*/ int blockDimZ;
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;
/*readonly*/ int num_chare_z;
/*readonly*/ int maxIter;
/*readonly*/ int ckptFreq;
/*readonly*/ int lbFreq;
/*readonly*/ int chkptIter;
/*readonly*/ std::string chkptDir;

#define wrap_x(a) (((a)+num_chare_x)%num_chare_x)
#define wrap_y(a) (((a)+num_chare_y)%num_chare_y)
#define wrap_z(a) (((a)+num_chare_z)%num_chare_z)

#define index(a, b, c) ( (a)*(blockDimY+2)*(blockDimZ+2) + (b)*(blockDimZ+2) + (c) )

#define PRINT_FREQ 10
#define CKP_FREQ 100
#define MAX_ITER 400
#define WARM_ITER 5
#define LEFT 1
#define RIGHT 2
#define TOP 3
#define BOTTOM 4
#define FRONT 5
#define BACK 6
#define DIVIDEBY7 0.14285714285714285714

static double startTime;

class ghostMsg : public CMessage_ghostMsg {
  public:
    int dir;
    int height;
    int width;
    double *gh;
    ghostMsg(int _d, int _h, int _w) : dir(_d), height(_h), width(_w) {}
};

class Main : public CBase_Main {
  public:
    CProxy_Jacobi array;
    int iterations;

    Main(CkArgMsg *m) {
      iterations = 0;
      mainProxy = thisProxy;
      maxIter = MAX_ITER;
      ckptFreq = CKP_FREQ;
      lbFreq = 0;
      chkptIter = -1;
      std::vector<char *> pos;
      for (int i = 1; i < m->argc; i++) {
        if (strcmp(m->argv[i], "-chkpt") == 0 && i + 2 < m->argc) {
          chkptIter = atoi(m->argv[i + 1]);
          chkptDir = m->argv[i + 2];
          i += 2;
        } else if (m->argv[i][0] != '+') { // runtime options stay in argv
          pos.push_back(m->argv[i]);
        }
      }
      if (pos.size() < 2 || pos.size() > 5) {
        CkPrintf("%s arrayDim blockDim [maxIter [ckptFreq [lbFreq]]] [-chkpt ITER DIR]\n", m->argv[0]);
        CkAbort("Abort");
      }
      arrayDimX = arrayDimY = arrayDimZ = atoi(pos[0]);
      blockDimX = blockDimY = blockDimZ = atoi(pos[1]);
      if (pos.size() > 2) maxIter = atoi(pos[2]);
      if (pos.size() > 3) ckptFreq = atoi(pos[3]);
      if (pos.size() > 4) lbFreq = atoi(pos[4]);
      delete m;

      if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
        CkAbort("array_size %% block_size != 0!");
      num_chare_x = arrayDimX / blockDimX;
      num_chare_y = arrayDimY / blockDimY;
      num_chare_z = arrayDimZ / blockDimZ;

      CkPrintf("Running Jacobi (coroutine entry) on %d processors with (%d, %d, %d) chares\n",
               CkNumPes(), num_chare_x, num_chare_y, num_chare_z);
      CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
      CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);
      CkPrintf("Iterations: %d, report every %d, load balance every %d, checkpoint after %d to %s\n",
               maxIter, ckptFreq, lbFreq, chkptIter, chkptDir.empty() ? "(none)" : chkptDir.c_str());

      array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z);
      startTime = CmiWallTimer();
      array.doStep();
    }

    Main(CkMigrateMessage *m) : CBase_Main(m) {}

    void pup(PUP::er &p) {
      p | array;
      p | iterations;
    }

    // Every ckptFreq iterations each element contributes its checksum.
    void report(long long sum) {
      iterations += ckptFreq;
      if (iterations < maxIter) {
        array.doStep();
      } else {
        double endTime = CmiWallTimer();
        CkPrintf("Time elapsed per iteration: %f\n", (endTime - startTime) / (maxIter - 1 - WARM_ITER));
        CkPrintf("jacobi3d: %d iterations, checksum %lld\n", maxIter, sum);
        CkExit();
      }
    }

    // All elements are suspended at their checkpoint site.
    void checkpoint() {
      CkPrintf("Checkpoint after iteration %d to %s\n", chkptIter, chkptDir.c_str());
      CkStartCheckpoint(chkptDir.c_str(), CkCallback(CkIndex_Main::checkpointed(), thisProxy));
    }

    // The checkpoint's callback: after the checkpoint in this run, and as the
    // first message of a run restarted from it (+restart DIR).
    void checkpointed() {
      CkPrintf("Resuming after the checkpoint on %d PEs\n", CkNumPes());
      array.resume();
    }
};

class Jacobi : public CBase_Jacobi {
  Jacobi_CORO_CODE          // the coroutine state and the inbox entries
  public:
    int iterations;
    int imsg;
    double *temperature;
    double *new_temperature;
    double timing, average;

    Jacobi() {
      temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
      new_temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
      for (int i = 0; i < blockDimX+2; ++i)
        for (int j = 0; j < blockDimY+2; ++j)
          for (int k = 0; k < blockDimZ+2; ++k)
            temperature[index(i, j, k)] = 0.0;
      iterations = 0;
      imsg = 0;
      timing = average = 0.0;
      constrainBC();
      usesAtSync = true;
    }

    Jacobi(CkMigrateMessage *m) : CBase_Jacobi(m) {}

    ~Jacobi() {
      delete [] temperature;
      delete [] new_temperature;
    }

    void pup(PUP::er &p) {
      p | iterations;
      p | imsg;
      p | timing;
      p | average;
      if (p.isUnpacking()) {
        temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
        new_temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
      }
      p((char *)temperature, (blockDimX+2) * (blockDimY+2) * (blockDimZ+2) * sizeof(double));
    }

    // One step: the SDAG doStep of jacobi3d-sdag as a coroutine.
    ck::coroutine doStep() {
      CK_COROUTINE();
      if (lbFreq > 0 && iterations > 0 && iterations % lbFreq == 0) {
        AtSync();
        CK_WHEN(inbox::ResumeFromSync());
      }
      if (iterations == chkptIter) {
        contribute(CkCallback(CkReductionTarget(Main, checkpoint), mainProxy));
        CK_WHEN(inbox::resume());
      }
      begin_iteration();
      for (imsg = 0; imsg < 6; imsg++) {
        CK_WHEN_BLOCK(inbox::receiveGhosts(iterations), gmsg) {  // when receiveGhosts[iterations](ghostMsg *gmsg) {
          processGhosts(gmsg);
        }
      }
      check_and_compute();
      CK_END();
    }

    void begin_iteration() {
      if (thisIndex.x == 0 && thisIndex.y == 0 && thisIndex.z == 0) {
        if (iterations % PRINT_FREQ == 0) {
          average = timing;
          timing = CmiWallTimer();
          average = (timing - average) / (double)PRINT_FREQ;
          CkPrintf("time=%.2f it=%d avg=%.4f\n", timing, iterations, average);
        }
      }
      iterations++;

      ghostMsg *leftMsg = new (blockDimY*blockDimZ) ghostMsg(RIGHT, blockDimY, blockDimZ);
      ghostMsg *rightMsg = new (blockDimY*blockDimZ) ghostMsg(LEFT, blockDimY, blockDimZ);
      ghostMsg *topMsg = new (blockDimX*blockDimZ) ghostMsg(BOTTOM, blockDimX, blockDimZ);
      ghostMsg *bottomMsg = new (blockDimX*blockDimZ) ghostMsg(TOP, blockDimX, blockDimZ);
      ghostMsg *frontMsg = new (blockDimX*blockDimY) ghostMsg(BACK, blockDimX, blockDimY);
      ghostMsg *backMsg = new (blockDimX*blockDimY) ghostMsg(FRONT, blockDimX, blockDimY);

      CkSetRefNum(leftMsg, iterations);
      CkSetRefNum(rightMsg, iterations);
      CkSetRefNum(topMsg, iterations);
      CkSetRefNum(bottomMsg, iterations);
      CkSetRefNum(frontMsg, iterations);
      CkSetRefNum(backMsg, iterations);

      for (int j = 0; j < blockDimY; ++j)
        for (int k = 0; k < blockDimZ; ++k) {
          leftMsg->gh[k*blockDimY+j] = temperature[index(1, j+1, k+1)];
          rightMsg->gh[k*blockDimY+j] = temperature[index(blockDimX, j+1, k+1)];
        }
      for (int i = 0; i < blockDimX; ++i)
        for (int k = 0; k < blockDimZ; ++k) {
          topMsg->gh[k*blockDimX+i] = temperature[index(i+1, 1, k+1)];
          bottomMsg->gh[k*blockDimX+i] = temperature[index(i+1, blockDimY, k+1)];
        }
      for (int i = 0; i < blockDimX; ++i)
        for (int j = 0; j < blockDimY; ++j) {
          frontMsg->gh[j*blockDimX+i] = temperature[index(i+1, j+1, 1)];
          backMsg->gh[j*blockDimX+i] = temperature[index(i+1, j+1, blockDimZ)];
        }

      thisProxy(wrap_x(thisIndex.x-1), thisIndex.y, thisIndex.z).receiveGhosts(leftMsg);
      thisProxy(wrap_x(thisIndex.x+1), thisIndex.y, thisIndex.z).receiveGhosts(rightMsg);
      thisProxy(thisIndex.x, wrap_y(thisIndex.y-1), thisIndex.z).receiveGhosts(topMsg);
      thisProxy(thisIndex.x, wrap_y(thisIndex.y+1), thisIndex.z).receiveGhosts(bottomMsg);
      thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z-1)).receiveGhosts(frontMsg);
      thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z+1)).receiveGhosts(backMsg);
    }

    void processGhosts(ghostMsg *m) {
      int height = m->height;
      int width = m->width;
      switch (m->dir) {
        case LEFT:
          for (int j = 0; j < height; ++j)
            for (int k = 0; k < width; ++k)
              temperature[index(0, j+1, k+1)] = m->gh[k*height+j];
          break;
        case RIGHT:
          for (int j = 0; j < height; ++j)
            for (int k = 0; k < width; ++k)
              temperature[index(blockDimX+1, j+1, k+1)] = m->gh[k*height+j];
          break;
        case TOP:
          for (int i = 0; i < height; ++i)
            for (int k = 0; k < width; ++k)
              temperature[index(i+1, 0, k+1)] = m->gh[k*height+i];
          break;
        case BOTTOM:
          for (int i = 0; i < height; ++i)
            for (int k = 0; k < width; ++k)
              temperature[index(i+1, blockDimY+1, k+1)] = m->gh[k*height+i];
          break;
        case FRONT:
          for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
              temperature[index(i+1, j+1, blockDimZ+1)] = m->gh[j*height+i];
          break;
        case BACK:
          for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
              temperature[index(i+1, j+1, 0)] = m->gh[j*height+i];
          break;
        default:
          CkAbort("ERROR\n");
      }
      delete m;
    }

    void check_and_compute() {
      compute_kernel();
      double *tmp = temperature;
      temperature = new_temperature;
      new_temperature = tmp;
      constrainBC();

      if (iterations % ckptFreq == 0 || iterations > maxIter) {
        // An order-independent checksum of the interior cells.
        long long sum = 0;
        for (int i = 1; i < blockDimX+1; ++i)
          for (int j = 1; j < blockDimY+1; ++j)
            for (int k = 1; k < blockDimZ+1; ++k)
              sum += llround(temperature[index(i, j, k)] * 1e6);
        contribute(sizeof(long long), &sum, CkReduction::sum_long_long,
                   CkCallback(CkReductionTarget(Main, report), mainProxy));
      } else {
        thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).doStep();
      }
    }

    void compute_kernel() {
      for (int i = 1; i < blockDimX+1; ++i)
        for (int j = 1; j < blockDimY+1; ++j)
          for (int k = 1; k < blockDimZ+1; ++k)
            new_temperature[index(i, j, k)] = (temperature[index(i-1, j, k)]
                + temperature[index(i+1, j, k)]
                + temperature[index(i, j-1, k)]
                + temperature[index(i, j+1, k)]
                + temperature[index(i, j, k-1)]
                + temperature[index(i, j, k+1)]
                + temperature[index(i, j, k)]) * DIVIDEBY7;
    }

    void constrainBC() {
      for (int i = 1; i < blockDimX+1; ++i)
        for (int k = 1; k < blockDimZ+1; ++k)
          temperature[index(i, 1, k)] = 255.0;
      for (int j = 1; j < blockDimY+1; ++j)
        for (int k = 1; k < blockDimZ+1; ++k)
          temperature[index(1, j, k)] = 255.0;
      for (int i = 1; i < blockDimX+1; ++i)
        for (int j = 1; j < blockDimY+1; ++j)
          temperature[index(i, j, 1)] = 255.0;
    }
};

#include "jacobi3d.def.h"
