/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** \file matmul3d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: March 13th, 2008
 *
 */

#include "matmul3d.decl.h"
#include "matmul3d.h"
#include "TopoManager.h"

Main::Main(CkArgMsg* m) {
  if ( (m->argc != 3) && (m->argc != 7) && (m->argc != 10) ) {
    CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
    CkPrintf("OR %s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z]\n", m->argv[0]);
    CkPrintf("OR %s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z] [torus_dim_X] [torus_dim_Y] [torus_dim_Z] \n", m->argv[0]);
    CkAbort("Abort");
  }

  // store the main proxy
  mainProxy = thisProxy;

  // get the size of the global array, size of each chare and size of the torus [optional]
  if(m->argc == 3) {
    arrayDimX = arrayDimY = arrayDimZ = atoi(m->argv[1]);
    blockDimX = blockDimY = blockDimZ = atoi(m->argv[2]);
  }
  else if (m->argc == 7) {
    arrayDimX = atoi(m->argv[1]);
    arrayDimY = atoi(m->argv[2]);
    arrayDimZ = atoi(m->argv[3]);
    blockDimX = atoi(m->argv[4]);
    blockDimY = atoi(m->argv[5]);
    blockDimZ = atoi(m->argv[6]);
  } else {
    arrayDimX = atoi(m->argv[1]);
    arrayDimY = atoi(m->argv[2]);
    arrayDimZ = atoi(m->argv[3]);
    blockDimX = atoi(m->argv[4]);
    blockDimY = atoi(m->argv[5]);
    blockDimZ = atoi(m->argv[6]);
    torusDimX = atoi(m->argv[7]);
    torusDimY = atoi(m->argv[8]);
    torusDimZ = atoi(m->argv[9]);
  }

  if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
    CkAbort("array_size_X % block_size_X != 0!");
  if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0)
    CkAbort("array_size_Y % block_size_Y != 0!");
  if (arrayDimZ < blockDimZ || arrayDimZ % blockDimZ != 0)
    CkAbort("array_size_Z % block_size_Z != 0!");

  num_chare_x = arrayDimX / blockDimX;
  num_chare_y = arrayDimY / blockDimY;
  num_chare_z = arrayDimZ / blockDimZ;
  num_chares = num_chare_x * num_chare_y * num_chare_z;

  subBlockDimX = blockDimX/num_chare_z;
  subBlockDimY = blockDimY/num_chare_x;
  subBlockDimZ = blockDimZ/num_chare_y;

  // print info
  CkPrintf("Running Matrix Multiplication on %d processors with (%d, %d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z);
  CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
  CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);

  // Create new array of worker chares
#if USE_TOPOMAP || USE_RRMAP || USE_RNDMAP
  CkPrintf("Topology Mapping is being done ... %d %d %d\n", USE_TOPOMAP, USE_RRMAP, USE_RNDMAP);
  CProxy_ComputeMap map = CProxy_ComputeMap::ckNew(num_chare_x, num_chare_y, num_chare_z);
  CkArrayOptions opts(num_chare_x, num_chare_y, num_chare_z);
  opts.setMap(map);
  compute = CProxy_Compute::ckNew(opts);
#else
  compute = CProxy_Compute::ckNew(num_chare_x, num_chare_y, num_chare_z);
#endif

  // CkPrintf("Total Hops: %d\n", hops);

  // Start the computation
  doneCount=0;
  startTime = CmiWallTimer();
  compute.beginCopying();
}

void Main::done() {
  doneCount++;
  if(doneCount == num_chares) {
    endTime = CmiWallTimer();
    CkPrintf("TIME %f secs\n", endTime - startTime);
    CkExit();
  }
}

// Constructor, initialize values
Compute::Compute() {
  // each compute will only hold blockDimX/num_chare_z no. of rows to begin with
  A = new float[blockDimX*blockDimY];
  B = new float[blockDimY*blockDimZ];
  C = new float[blockDimX*blockDimZ];
  memset(A, 1, sizeof(float)*(blockDimX*blockDimY));
  memset(B, 2, sizeof(float)*(blockDimY*blockDimZ));
  memset(C, 0, sizeof(float)*(blockDimX*blockDimZ));

  // counters to keep track of how many messages have been received
  countA = 0;
  countB = 0;
}

Compute::Compute(CkMigrateMessage* m) {

}

Compute::~Compute() {
  delete [] A;
  delete [] B;
  delete [] C;
}

void Compute::beginCopying() {
  sendA();
  sendB();
}

void Compute::sendA() {
  int indexZ = thisIndex.z;
  float *dataA = new float[subBlockDimX * blockDimY];
  for(int i=0; i<subBlockDimX; i++)
    for(int j=0; j<blockDimY; j++)
      dataA[i*blockDimY + j] = A[indexZ*subBlockDimX*blockDimY + i*blockDimY + j];

  for(int k=0; k<num_chare_z; k++)
    if(k != indexZ)
      compute(thisIndex.x, thisIndex.y, k).receiveA(indexZ, dataA, subBlockDimX * blockDimY);
}

void Compute::sendB() {
  int indexX = thisIndex.x;
  float *dataB = new float[subBlockDimY * blockDimZ];
  for(int j=0; j<subBlockDimY; j++)
    for(int k=0; k<blockDimZ; k++)
      dataB[j*blockDimZ + k] = B[indexX*subBlockDimY*blockDimZ + j*blockDimZ + k];

  for(int i=0; i<num_chare_x; i++)
    if(i != indexX)
      compute(i, thisIndex.y, thisIndex.z).receiveB(indexX, dataB, subBlockDimY * blockDimZ);
}

void Compute::receiveA(int indexZ, float *data, int size) {
  for(int i=0; i<subBlockDimX; i++)
    for(int j=0; j<blockDimY; j++)
      A[indexZ*subBlockDimX*blockDimY + i*blockDimY + j] = data[i*blockDimY + j];
  countA++;
  if(countA == num_chare_z-1)
    doWork();
}

void Compute::receiveB(int indexX, float *data, int size) {
  for(int j=0; j<subBlockDimY; j++)
    for(int k=0; k<blockDimZ; k++)
      B[indexX*subBlockDimY*blockDimZ + j*blockDimZ + k] = data[j*blockDimZ + k];
  if(countB == num_chare_x-1)
    doWork();
}

void Compute::doWork() {
  if(countA == num_chare_z-1 && countA == num_chare_z-1) {
    for(int i=0; i<blockDimX; i++)
      for(int j=0; j<blockDimY; j++)
	for(int k=0; k<blockDimZ; k++)
	  C[i*blockDimZ+k] = A[i*blockDimY+j] * B[j*blockDimZ+k];
  }
}

ComputeMap::ComputeMap(int x, int y, int z) {

}

ComputeMap::~ComputeMap() {
  for (int i=0; i<X; i++) {
    for(int j=0; j<Y; j++)
      delete [] mapping[i][j];
    delete [] mapping[i];
  }
  delete [] mapping;
}

int ComputeMap::procNum(int, const CkArrayIndex &idx) {
  int *index = (int *)idx.data();
  return mapping[index[0]][index[1]][index[2]];
}


