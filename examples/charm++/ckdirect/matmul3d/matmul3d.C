/** \file matmul3d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: April 01st, 2008
 *
 */

#include <stdlib.h>
#include "ckdirect.h"
#include "matmul3d.decl.h"
#include "matmul3d.h"
#include "TopoManager.h"
#if CMK_BLUEGENEP || CMK_VERSION_BLUEGENE
#include "essl.h"
#endif
#include "rand48_replacement.h"

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

  subBlockDimXz = blockDimX/num_chare_z;
  subBlockDimYx = blockDimY/num_chare_x;
  subBlockDimXy = blockDimX/num_chare_y;

  // print info
  CkPrintf("Running Matrix Multiplication on %d processors with (%d, %d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z);
  CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
  CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);

  // Create new array of worker chares
#if USE_TOPOMAP
  CkPrintf("Topology Mapping is being done ...\n");
  CProxy_ComputeMap map = CProxy_ComputeMap::ckNew(num_chare_x, num_chare_y, num_chare_z, 1, 1, 1);
  CkArrayOptions opts(num_chare_x, num_chare_y, num_chare_z);
  opts.setMap(map);
  compute = CProxy_Compute::ckNew(opts);
#elif USE_BLOCKMAP
  CkPrintf("Block Mapping is being done ...\n");
  CProxy_ComputeMap map = CProxy_ComputeMap::ckNew(num_chare_x, num_chare_y, num_chare_z, torusDimX, torusDimY, torusDimZ);
  CkArrayOptions opts(num_chare_x, num_chare_y, num_chare_z);
  opts.setMap(map);
  compute = CProxy_Compute::ckNew(opts);
#else
  compute = CProxy_Compute::ckNew(num_chare_x, num_chare_y, num_chare_z);
#endif

  // CkPrintf("Total Hops: %d\n", hops);

  // Start the computation
  numIterations = 0;
  startTime = CkWallTimer();
#if USE_CKDIRECT
  compute.setupChannels();
#else
  compute.beginCopying();
#endif
}

void Main::done() {
  numIterations++;
  if(numIterations == 1) {
    firstTime = CkWallTimer();
#if USE_CKDIRECT
    CkPrintf("FIRST ITER TIME %f secs\n", firstTime - setupTime);
#else
    CkPrintf("FIRST ITER TIME %f secs\n", firstTime - startTime);
#endif
    compute.resetArrays();
  } else {
    if(numIterations == NUM_ITER) {
      endTime[numIterations-2] = CkWallTimer() - firstTime;
      double sum = 0;
      for(int i=0; i<NUM_ITER-1; i++)
	sum += endTime[i];
#if USE_CKDIRECT
      CkPrintf("AVG TIME %f secs\n", sum/(NUM_ITER-1));
#else
      CkPrintf("AVG TIME %f secs\n", sum/(NUM_ITER-1));
#endif
      CkExit();
    } else {
      endTime[numIterations-2] = CkWallTimer() - firstTime;
      compute.resetArrays();
    }
  }
}

void Main::resetDone() {
  firstTime = CkWallTimer();
  compute.beginCopying();
}

void Main::setupDone() {
  setupTime = CkWallTimer();
  CkPrintf("SETUP TIME %f secs\n", setupTime - startTime);
  compute.beginCopying();
}

// Constructor, initialize values
Compute::Compute() {
  // each compute will only hold blockDimX/num_chare_z no. of rows to begin with
  A = new float[blockDimX*blockDimY];
  B = new float[blockDimY*blockDimZ];
  C = new float[blockDimX*blockDimZ];
#if USE_CKDIRECT
  tmpC = new float[blockDimX*blockDimZ];
#endif

  int indexX = thisIndex.x;
  int indexY = thisIndex.y;
  int indexZ = thisIndex.z;

  float tmp;

  for(int i=indexZ*subBlockDimXz; i<(indexZ+1)*subBlockDimXz; i++)
    for(int j=0; j<blockDimY; j++) {
      tmp = (float)drand48();
      while(tmp > MAX_LIMIT || tmp < (-1)*MAX_LIMIT)
        tmp = (float)drand48();

      A[i*blockDimY + j] = tmp;
  }

  for(int j=indexX*subBlockDimYx; j<(indexX+1)*subBlockDimYx; j++)
    for(int k=0; k<blockDimZ; k++) {
      tmp = (float)drand48();
      while(tmp > MAX_LIMIT || tmp < (-1)*MAX_LIMIT)
        tmp = (float)drand48();

      B[j*blockDimZ + k] = tmp;
  }

  for(int i=0; i<blockDimX; i++)
    for(int k=0; k<blockDimZ; k++) {
      C[i*blockDimZ + k] = 0.0;
#if USE_CKDIRECT
      tmpC[i*blockDimZ + k] = 0.0;
#endif
    }

  // counters to keep track of how many messages have been received
  countA = 0;
  countB = 0;
  countC = 0;

#if USE_CKDIRECT
  sHandles = new infiDirectUserHandle[num_chare_x + num_chare_y + num_chare_z];
  rHandles = new infiDirectUserHandle[num_chare_x + num_chare_y + num_chare_z];
#endif
}

Compute::Compute(CkMigrateMessage* m) {

}

Compute::~Compute() {
  delete [] A;
  delete [] B;
  delete [] C;
#if USE_CKDIRECT
  delete [] tmpC;
#endif
}

void Compute::beginCopying() {
  sendA();
  sendB();
}

void Compute::resetArrays() {
  int indexX = thisIndex.x;
  int indexY = thisIndex.y;
  int indexZ = thisIndex.z;
  
  float tmp;
  
  for(int i=indexZ*subBlockDimXz; i<(indexZ+1)*subBlockDimXz; i++)
    for(int j=0; j<blockDimY; j++) {
      tmp = (float)drand48(); 
      while(tmp > MAX_LIMIT || tmp < (-1)*MAX_LIMIT)
        tmp = (float)drand48();

      A[i*blockDimY + j] = tmp;
  }

  for(int j=indexX*subBlockDimYx; j<(indexX+1)*subBlockDimYx; j++)
    for(int k=0; k<blockDimZ; k++) {
      tmp = (float)drand48();
      while(tmp > MAX_LIMIT || tmp < (-1)*MAX_LIMIT)
        tmp = (float)drand48();

      B[j*blockDimZ + k] = tmp;
  }

  for(int i=0; i<blockDimX; i++)
    for(int k=0; k<blockDimZ; k++) {
      C[i*blockDimZ + k] = 0.0;
#if USE_CKDIRECT
      tmpC[i*blockDimZ + k] = 0.0;
#endif
    }

  contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::resetDone(), mainProxy));
}

void Compute::sendA() {
  int indexZ = thisIndex.z;

  for(int k=0; k<num_chare_z; k++)
    if(k != indexZ) {
#if USE_CKDIRECT
      CkDirect_put(&sHandles[num_chare_x + num_chare_y + k]);
#else
      // use a local pointer for chares on the same processor
      Compute* c = compute(thisIndex.x, thisIndex.y, k).ckLocal();
      if(c != NULL)
	c->receiveA(indexZ, &A[indexZ*subBlockDimXz*blockDimY], subBlockDimXz * blockDimY);
      else
	compute(thisIndex.x, thisIndex.y, k).receiveA(indexZ, &A[indexZ*subBlockDimXz*blockDimY], subBlockDimXz * blockDimY);
#endif
    }
}

void Compute::sendB() {
  int indexX = thisIndex.x;

  for(int i=0; i<num_chare_x; i++)
    if(i != indexX) {
#if USE_CKDIRECT
      CkDirect_put(&sHandles[i]);
#else
      // use a local pointer for chares on the same processor
      Compute* c = compute(i, thisIndex.y, thisIndex.z).ckLocal();
      if(c != NULL)
	c->receiveB(indexX, &B[indexX*subBlockDimYx*blockDimZ], subBlockDimYx * blockDimZ);
      else
	compute(i, thisIndex.y, thisIndex.z).receiveB(indexX, &B[indexX*subBlockDimYx*blockDimZ], subBlockDimYx * blockDimZ);
#endif
    }
}

void Compute::sendC() {
  int indexY = thisIndex.y;
  for(int j=0; j<num_chare_y; j++) {
    if(j != indexY) {
#if USE_CKDIRECT
      CkDirect_put(&sHandles[num_chare_x + j]);
#else
      // use a local pointer for chares on the same processor
      Compute *c = compute(thisIndex.x, j, thisIndex.z).ckLocal();
      if(c != NULL)
	c->receiveC(&C[j*subBlockDimXy*blockDimZ], subBlockDimXy * blockDimZ, 1);
      else
	compute(thisIndex.x, j, thisIndex.z).receiveC(&C[j*subBlockDimXy*blockDimZ], subBlockDimXy * blockDimZ, 1);
#endif
    }
  }
}

void Compute::receiveA(int indexZ, float *data, int size) {
  for(int i=0; i<subBlockDimXz; i++)
    for(int j=0; j<blockDimY; j++)
      A[indexZ*subBlockDimXz*blockDimY + i*blockDimY + j] = data[i*blockDimY + j];
  countA++;
  if(countA == num_chare_z-1 && countB == num_chare_x-1)
    doWork();
}

void Compute::receiveB(int indexX, float *data, int size) {
  for(int j=0; j<subBlockDimYx; j++)
    for(int k=0; k<blockDimZ; k++)
      B[indexX*subBlockDimYx*blockDimZ + j*blockDimZ + k] = data[j*blockDimZ + k];
  countB++;
  if(countA == num_chare_z-1 && countB == num_chare_x-1)
    doWork();
}

void Compute::receiveC(float *data, int size, int who) {
  int indexY = thisIndex.y;
  if(who) {
    for(int i=0; i<subBlockDimXy; i++)
      for(int k=0; k<blockDimZ; k++)
	C[indexY*subBlockDimXy*blockDimZ + i*blockDimZ + k] += data[i*blockDimZ + k];
  }
  countC++;
  if(countC == num_chare_y) {
    /*char name[30];
    sprintf(name, "%s_%d_%d_%d", "C", thisIndex.x, thisIndex.y, thisIndex.z);
    FILE *fp = fopen(name, "w");
    for(int i=0; i<subBlockDimXy; i++) {
      for(int k=0; k<blockDimZ; k++)
	fprintf(fp, "%f ", C[indexY*subBlockDimXy*blockDimZ + i*blockDimZ + k]);
      fprintf(fp, "\n");
    }
    fclose(fp);*/

    // counters to keep track of how many messages have been received
    countA = 0;
    countB = 0;
    countC = 0;

    contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::done(), mainProxy));
    // mainProxy.done();
  }
}

void Compute::receiveC() {
  int indexX = thisIndex.x;
  int indexY = thisIndex.y;
  int indexZ = thisIndex.z;

  // copy C from tmpC to the correct location
  for(int j=0; j<num_chare_y; j++) {
    if( j != indexY) {
      for(int i=0; i<subBlockDimXy; i++)
	for(int k=0; k<blockDimZ; k++)
	  C[indexY*subBlockDimXy*blockDimZ + i*blockDimZ + k] += tmpC[j*subBlockDimXy*blockDimZ + i*blockDimZ + k];
    }
  }
  /*char name[30];
  sprintf(name, "%s_%d_%d_%d", "C", thisIndex.x, thisIndex.y, thisIndex.z);
  FILE *fp = fopen(name, "w");
  for(int i=0; i<subBlockDimXy; i++) {
    for(int k=0; k<blockDimZ; k++)
      fprintf(fp, "%f ", C[indexY*subBlockDimXy*blockDimZ + i*blockDimZ + k]);
    fprintf(fp, "\n");
  }
  fclose(fp);
  CkPrintf("%d_%d_%d\n", thisIndex.x, thisIndex.y, thisIndex.z);
  for(int i=0; i<subBlockDimXy; i++) {
    for(int k=0; k<blockDimZ; k++)
      CkPrintf("%f ", C[indexY*subBlockDimXy*blockDimZ + i*blockDimZ + k]);
    CkPrintf("\n");
  }*/

  // call ready for the buffers
  for(int i=0; i<num_chare_x; i++)
    if(i != indexX)
      CkDirect_ready(&rHandles[i]);

  for(int j=0; j<num_chare_y; j++)
    if(j != indexY)
      CkDirect_ready(&rHandles[num_chare_x + j]);
  
  for(int k=0; k<num_chare_z; k++)
    if(k != indexZ)
      CkDirect_ready(&rHandles[num_chare_x + num_chare_y + k]);

  // counters to keep track of how many messages have been received
  countA = 0;
  countB = 0;
  countC = 0;

  contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::done(), mainProxy));
  // mainProxy.done();
}

void Compute::doWork() {
  if(countA == num_chare_z-1 && countB == num_chare_x-1) {

#if CMK_BLUEGENEP || CMK_VERSION_BLUEGENE
    const char trans = 'N';
    const double alpha = 1.0;
    const double beta = 0.0;
    
    sgemm(&trans, &trans, blockDimX, blockDimZ, blockDimY, alpha, A, blockDimX, B, blockDimY, beta, C, blockDimX);
#else
    for(int i=0; i<blockDimX; i++)
      for(int j=0; j<blockDimY; j++)
	for(int k=0; k<blockDimZ; k++)
	  C[i*blockDimZ+k] += A[i*blockDimY+j] * B[j*blockDimZ+k];
#endif

#if USE_CKDIRECT
    callBackRcvdC((void *)this);
#else
    receiveC(&C[(thisIndex.y)*subBlockDimXy*blockDimZ], subBlockDimXy*blockDimZ, 0);
#endif
    sendC();
  }
}

void Compute::setupChannels() {
  int mype = CkMyPe();
  int x = thisIndex.x;
  int y = thisIndex.y;
  int z = thisIndex.z;

  for(int k=0; k<num_chare_z; k++)
    if(k != z)
      thisProxy(x, y, k).notifyReceiver(mype, thisIndex, SENDA);

  for(int i=0; i<num_chare_x; i++)
    if(i != x)
      thisProxy(i, y, z).notifyReceiver(mype, thisIndex, SENDB);

  for(int j=0; j<num_chare_y; j++)
    if(j != y)
      thisProxy(x, j, z).notifyReceiver(mype, thisIndex, SENDC);

}

void Compute::notifyReceiver(int pe, CkIndex3D index, int arr) {
  // --- B --- | --- C --- | --- A ---
  if(arr == SENDA) {
    rHandles[num_chare_x + num_chare_y + index.z] = CkDirect_createHandle(pe, &A[(index.z)*subBlockDimXz*blockDimY], sizeof(float)*subBlockDimXz*blockDimY, Compute::callBackRcvdA, (void *)this, OOB);
    thisProxy(index.x, index.y, index.z).recvHandle(rHandles[num_chare_x + num_chare_y + index.z], thisIndex.z, arr);
  }

  if(arr == SENDB) {
    rHandles[index.x] = CkDirect_createHandle(pe, &B[(index.x)*subBlockDimYx*blockDimZ], sizeof(float)*subBlockDimYx*blockDimZ, Compute::callBackRcvdB, (void *)this, OOB);
    thisProxy(index.x, index.y, index.z).recvHandle(rHandles[index.x], thisIndex.x, arr);
  }

  if(arr == SENDC) {
    rHandles[num_chare_x + index.y] = CkDirect_createHandle(pe, &tmpC[(index.y)*subBlockDimXy*blockDimZ], sizeof(float)*subBlockDimXy*blockDimZ, Compute::callBackRcvdC, (void *)this, OOB);
    thisProxy(index.x, index.y, index.z).recvHandle(rHandles[num_chare_x + index.y], thisIndex.y, arr);
  }
}

void Compute::recvHandle(infiDirectUserHandle shdl, int index, int arr) {
  // --- B --- | --- C --- | --- A ---
  if(arr == SENDA) {
    sHandles[num_chare_x + num_chare_y + index] = shdl;
    CkDirect_assocLocalBuffer(&sHandles[num_chare_x + num_chare_y + index], &A[thisIndex.z*subBlockDimXz*blockDimY], sizeof(float)*subBlockDimXz*blockDimY);
    countA++;
  }

  if(arr == SENDB) {
    sHandles[index] = shdl;
    CkDirect_assocLocalBuffer(&sHandles[index], &B[thisIndex.x*subBlockDimYx*blockDimZ], sizeof(float)*subBlockDimYx*blockDimZ);
    countB++;
  }

  if(arr == SENDC) {
    sHandles[num_chare_x + index] = shdl;
    CkDirect_assocLocalBuffer(&sHandles[num_chare_x + index], &C[index*subBlockDimXy*blockDimZ], sizeof(float)*subBlockDimXy*blockDimZ);
    countC++;
  }

  if(countA == num_chare_z-1 && countB == num_chare_x-1 && countC == num_chare_y-1) {
    countA = 0;
    countB = 0;
    countC = 0;
    contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::setupDone(), mainProxy));
    // mainProxy.setupDone();
  }
}


void Compute::callBackRcvdA(void *arg) {
  Compute *obj = (Compute *)arg;
  obj->countA++;
  if(obj->countA == num_chare_z-1 && obj->countB == num_chare_x-1)
    obj->thisProxy(obj->thisIndex.x, obj->thisIndex.y, obj->thisIndex.z).doWork();
}

void Compute::callBackRcvdB(void *arg) {
  Compute *obj = (Compute *)arg;
  obj->countB++;
  if(obj->countA == num_chare_z-1 && obj->countB == num_chare_x-1)
    obj->thisProxy(obj->thisIndex.x, obj->thisIndex.y, obj->thisIndex.z).doWork();
}

void Compute::callBackRcvdC(void *arg) {
  Compute *obj = (Compute *)arg;
  obj->countC++;
  if(obj->countC == num_chare_y)
    obj->thisProxy(obj->thisIndex.x, obj->thisIndex.y, obj->thisIndex.z).receiveC();
}

ComputeMap::ComputeMap(int x, int y, int z, int tx, int ty, int tz) {
  X = x;
  Y = y;
  Z = z;
  mapping = new int[X*Y*Z];

  TopoManager tmgr;
  int dimX, dimY, dimZ, dimT;

#if USE_TOPOMAP
  dimX = tmgr.getDimNX();
  dimY = tmgr.getDimNY();
  dimZ = tmgr.getDimNZ();
  dimT = tmgr.getDimNT();
#elif USE_BLOCKMAP
  dimX = tx;
  dimY = ty;
  dimZ = tz;
  dimT = 1;
#endif

  // we are assuming that the no. of chares in each dimension is a 
  // multiple of the torus dimension
  int numCharesPerPe = X*Y*Z/CkNumPes();

  int numCharesPerPeX = X / dimX;
  int numCharesPerPeY = Y / dimY;
  int numCharesPerPeZ = Z / dimZ;

  if(dimT < 2) {    // one core per node
    if(CkMyPe()==0) CkPrintf("DATA: %d %d %d %d : %d %d %d\n", dimX, dimY, dimZ, dimT, numCharesPerPeX, numCharesPerPeY, numCharesPerPeZ);
    for(int i=0; i<dimX; i++)
      for(int j=0; j<dimY; j++)
        for(int k=0; k<dimZ; k++)
          for(int ci=i*numCharesPerPeX; ci<(i+1)*numCharesPerPeX; ci++)
            for(int cj=j*numCharesPerPeY; cj<(j+1)*numCharesPerPeY; cj++)
              for(int ck=k*numCharesPerPeZ; ck<(k+1)*numCharesPerPeZ; ck++) {
#if USE_TOPOMAP
                mapping[ci*Y*Z + cj*Z + ck] = tmgr.coordinatesToRank(i, j, k);
#elif USE_BLOCKMAP
                mapping[ci*Y*Z + cj*Z + ck] = i + j*dimX + k*dimX*dimY;
#endif
              }
  } else {          // multiple cores per node
    // In this case, we split the chares in the X dimension among the
    // cores on the same node.
    numCharesPerPeX /= dimT;
      if(CkMyPe()==0) CkPrintf("%d %d %d : %d %d %d %d : %d %d %d \n", x, y, z, dimX, dimY, dimZ, dimT, numCharesPerPeX, numCharesPerPeY, numCharesPerPeZ);
      for(int i=0; i<dimX; i++)
        for(int j=0; j<dimY; j++)
          for(int k=0; k<dimZ; k++)
            for(int l=0; l<dimT; l++)
              for(int ci=(dimT*i+l)*numCharesPerPeX; ci<(dimT*i+l+1)*numCharesPerPeX; ci++)
                for(int cj=j*numCharesPerPeY; cj<(j+1)*numCharesPerPeY; cj++)
                  for(int ck=k*numCharesPerPeZ; ck<(k+1)*numCharesPerPeZ; ck++) {
                    mapping[ci*Y*Z + cj*Z + ck] = tmgr.coordinatesToRank(i, j, k, l);
                  }
  }
}

ComputeMap::~ComputeMap() {
  delete [] mapping;
}

int ComputeMap::procNum(int, const CkArrayIndex &idx) {
  int *index = (int *)idx.data();
  return mapping[index[0]*Y*Z + index[1]*Z + index[2]];
}

