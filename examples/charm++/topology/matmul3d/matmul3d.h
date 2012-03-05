/** \file matmul3d.h
 *  Author: Abhinav S Bhatele
 *  Date Created: March 13th, 2008
 *
 */

// Read-only global variables

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Compute compute;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int arrayDimZ;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;
/*readonly*/ int blockDimZ;
/*readonly*/ int torusDimX;
/*readonly*/ int torusDimY;
/*readonly*/ int torusDimZ;

/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;
/*readonly*/ int num_chare_z;
/*readonly*/ int num_chares;
/*readonly*/ int subBlockDimXz;
/*readonly*/ int subBlockDimYx;
/*readonly*/ int subBlockDimXy;

static unsigned long next = 1;

int myrand(int numpes) {
  next = next * 1103515245 + 12345;
  return((unsigned)(next/65536) % numpes);
}

#define USE_TOPOMAP	0
#define USE_BLOCKMAP	1

#define NUM_ITER	21

#define MAX_LIMIT       9999999999.0

double startTime;
double firstTime;
double endTime;

/** \class Main
 *
 */
class Main : public CBase_Main {
  public:
    int numIterations;

    Main(CkArgMsg* m);
    void done();
};

/** \class Compute
 *
 */
class Compute: public CBase_Compute {
  public:
    float *A, *B, *C;
    int countA, countB, countC;

    Compute();
    Compute(CkMigrateMessage* m);
    ~Compute();

    void beginCopying();
    void resetArrays();
    void sendA();
    void sendB();
    void sendC();
    void receiveA(int indexZ, float *data, int size);
    void receiveB(int indexX, float *data, int size);
    void receiveC(float *data, int size, int who);
    void doWork();
};

/** \class ComputeMap
 *
 */
class ComputeMap: public CBase_ComputeMap {
  public:
    int X, Y, Z;
    int *mapping;

    ComputeMap(int x, int y, int z, int tx, int ty, int tz);
    ~ComputeMap();
    int procNum(int, const CkArrayIndex &idx);
};

#include "matmul3d.def.h"
