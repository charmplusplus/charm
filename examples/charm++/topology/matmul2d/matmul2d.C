#include "matmul2d.decl.h"
#include "matmul2d.h"
#include "TopoManager.h"

Main::Main(CkArgMsg* m) {
  if ( (m->argc != 3) && (m->argc != 5) && (m->argc != 7) ) {
    CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
    CkPrintf("OR %s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y]\n", m->argv[0]);
    CkPrintf("OR %s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y] [torus_dim_X] [torus_dim_Y]\n", m->argv[0]);
    CkAbort("Abort");
  }

  // store the main proxy
  mainProxy = thisProxy;

  // get the size of the global array, size of each chare and size of the torus [optional]
  if(m->argc == 3) {
    arrayDimX = arrayDimY = atoi(m->argv[1]);
    blockDimX = blockDimY = atoi(m->argv[2]);
  }
  /*
  else if (m->argc == 5) {
    arrayDimX = atoi(m->argv[1]);
    arrayDimY = atoi(m->argv[2]);
    blockDimX = atoi(m->argv[3]);
    blockDimY = atoi(m->argv[4]);
  } else if (m->argc == 7){
    arrayDimX = atoi(m->argv[1]);
    arrayDimY = atoi(m->argv[2]);
    blockDimX = atoi(m->argv[3]);
    blockDimY = atoi(m->argv[4]);
    torusDimX = atoi(m->argv[5]);
    torusDimY = atoi(m->argv[6]);
  }
  */
  else{
    CkAbort("Square matrices only: matmul2d array_dim block_dim\n");
  }

  if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
    CkAbort("array_size_X % block_size_X != 0!");
  if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0)
    CkAbort("array_size_Y % block_size_Y != 0!");

  num_chare_x = arrayDimX / blockDimX;
  num_chare_y = arrayDimY / blockDimY;

  // print info
  CkPrintf("Running Matrix Multiplication on %d processors with (%d, %d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y);
  CkPrintf("Array Dimensions: %d %d\n", arrayDimX, arrayDimY);
  CkPrintf("Block Dimensions: %d %d\n", blockDimX, blockDimY);
  CkPrintf("Chare-array Dimensions: %d %d\n", num_chare_x, num_chare_y);

  // Create new array of worker chares
  compute = CProxy_Compute::ckNew(num_chare_x, num_chare_y);

  // Start the computation
  startTime = CmiWallTimer();
  compute.compute();
}

void Main::done(){
  endTime = CmiWallTimer();
  CkPrintf("Fin: %f sec\n", endTime-startTime);
  CkExit();
}

// Constructor, initialize values
Compute::Compute() {

  for(int i = 0; i < 2; i++){
    A[i] = new float[blockDimX*blockDimY];
    B[i] = new float[blockDimX*blockDimY];
  }
  C = new float[blockDimX*blockDimY];

  for(int i = 0; i < blockDimX*blockDimY; i++){
    A[0][i] = MAGIC_A;
    B[0][i] = MAGIC_B;
  }
  step = 0;  
  row = thisIndex.y;
  col = thisIndex.x;

  whichLocal = 0;
  remaining = 2;
  iteration = 0;
}

Compute::Compute(CkMigrateMessage* m) {
}

Compute::~Compute() {
  delete [] A[0];
  delete [] A[1];
  delete [] B[0];
  delete [] B[1];
  delete [] C;
  delete [] C;
}

void Compute::compute(){
  for(int i = 0; i < blockDimX; i++){
    for(int j = 0; j < blockDimX; j++){
      C[i*blockDimX+j] = 0;
      for(int k = 0; k < blockDimX; k++){
        C[i*blockDimX+j] += A[whichLocal][i*blockDimX+k] + B[whichLocal][k*blockDimX+j];
      }
    }
  }
  remaining = 2;
  iteration++;
  if(iteration == num_chare_x){
    contribute(0,0,CkReduction::concat, CkCallback(CkIndex_Main::done(), mainProxy));
  }
  else{
    contribute(0,0,CkReduction::concat,CkCallback(CkIndex_Compute::resumeFromBarrier(), thisProxy));
  }
}

void Compute::resumeFromBarrier(){
  // At this point, everyone has used their A and B buffers
  int newBuf = 1-whichLocal;
  // We must put our own 
  // 1. First put A

  if(num_chare_x == 0 || num_chare_y ==0)
    CkPrintf("(%d,%d): 0 divisor\n", thisIndex.y, thisIndex.x);
  /*
  CkPrintf("(%d,%d): A nbr: (%d,%d)\n", thisIndex.y, thisIndex.x, row, (col-1+num_chare_x)%num_chare_x);
  CkPrintf("(%d,%d): B nbr: (%d,%d)\n", thisIndex.y, thisIndex.x, (row-1+num_chare_y)%num_chare_y, col);
  */

  Compute *c = thisProxy((col-1+num_chare_x)%num_chare_x, row).ckLocal();
  if(c == 0){
    thisProxy((col-1+num_chare_x)%num_chare_x, row).recvBlockA(A[whichLocal], blockDimX, newBuf);
  }
  else{
    c->recvBlockA(A[whichLocal], blockDimX, newBuf);
  }
  // 2. Then put B
  c =  thisProxy(col, (row-1+num_chare_y)%num_chare_y).ckLocal();
  if(c == 0){
    thisProxy(col, (row-1+num_chare_y)%num_chare_y).recvBlockB(B[whichLocal], blockDimX, newBuf);
  }
  else{
    c->recvBlockB(B[whichLocal], blockDimX, newBuf);
  }
  // toggle between local buffers
  whichLocal = newBuf;
}

void Compute::recvBlockA(float *block, int size, int whichBuf){
  memcpy(A[whichBuf], block, size);
  remaining--;
  if(remaining == 0){
    compute();
  }
}

void Compute::recvBlockB(float *block, int size, int whichBuf){
  memcpy(B[whichBuf], block, size);
  remaining--;
  if(remaining == 0){
    compute();
  }
}

#include "matmul2d.def.h"
