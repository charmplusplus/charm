#include "matmul2d.decl.h"
#include "matmul2d.h"
#include "TopoManager.h"
/*
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
*/
#include "stdio.h"
#include "stdlib.h"

Main::Main(CkArgMsg* m) {
  if (m->argc != 4){ 
    CkPrintf("%s [N] [K] [num_chares_per_dim]\n", m->argv[0]);
    CkAbort("Abort");
  }
  else {
    N = atoi(m->argv[1]);
    K = atoi(m->argv[2]);
    num_chares_per_dim = atoi(m->argv[3]);
    T = N/num_chares_per_dim;
  }

  // store the main proxy
  mainProxy = thisProxy;

  // get the size of the global array, size of each chare and size of the torus [optional]
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

  if (N < T || N % T != 0)
    CkAbort("N % T != 0!");
  if (K < T || K % T != 0)
    CkAbort("K % T != 0!");

  // print info
  CkPrintf("Running Matrix Multiplication on %d processors with (%d, %d) chares\n", CkNumPes(), num_chares_per_dim, num_chares_per_dim);
  CkPrintf("Array Dimensions: %d %d\n", N, K);
  CkPrintf("Block Dimensions: %dx%d, %dx%d\n", T, K/num_chares_per_dim, K/num_chares_per_dim, T);
  CkPrintf("Chare-array Dimensions: %d %d\n", num_chares_per_dim, num_chares_per_dim);

  // Create new array of worker chares
  compute = CProxy_Compute::ckNew(num_chares_per_dim, num_chares_per_dim);

  // Start the computation
  startTime = CkWallTimer();
  if(num_chares_per_dim == 1){
    compute(0,0).compute();
  }
  else{
    compute.compute();
    //compute.start();
  }
}

void Main::done(){
  endTime = CkWallTimer();
  CkPrintf("Fin: %f sec\n", endTime-startTime);
  CkExit();
}

// Constructor, initialize values
Compute::Compute() {

  int s1 = (K/num_chares_per_dim)*T;
  int s2 = T*T;
  for(int i = 0; i < 2; i++){
    A[i] = new float[s1];
    B[i] = new float[s1];
  }
  C = new float[s2];

  for(int i = 0; i < s1; i++){
    A[0][i] = MAGIC_A;
    B[0][i] = MAGIC_B;
  }
  for(int i = 0; i < s2; i++){
    C[i] = 0;
  }
    
  step = 0;  
  row = thisIndex.y;
  col = thisIndex.x;

  whichLocal = 0;
  remaining = 2;
  iteration = 0;

  //comps = 0;
}

Compute::Compute(CkMigrateMessage* m) {
}

Compute::~Compute() {
  delete [] A[0];
  delete [] A[1];
  delete [] B[0];
  delete [] B[1];
  delete [] C;
}

void Compute::start(){
  int newBuf = 1 - whichLocal;
  /*
  //1. send A
  Compute *c = thisProxy((col-row+num_chare_x)%num_chare_x, row).ckLocal();
  if(c == 0){
    thisProxy((col-row+num_chare_x)%num_chare_x, row).recvBlockA(A[whichLocal], blockDimX*blockDimY, newBuf);
  }
  else{
    c->recvBlockA(A[whichLocal], blockDimX*blockDimY, newBuf);
  }
  
  //2. send B
  c = thisProxy(col, (row-col+num_chare_y)%num_chare_y).ckLocal();
  if(c == 0){
    thisProxy(col, (row-col+num_chare_y)%num_chare_y).recvBlockB(B[whichLocal], blockDimX*blockDimY, newBuf);
  }
  else{
    c->recvBlockB(B[whichLocal], blockDimX*blockDimY, newBuf);
  }
  */
  whichLocal = newBuf;
}

void Compute::compute(){
  //int count = 0;
#ifdef USE_OPT_ROUTINES
  const char trans = 'N';
  const double alpha = 1.0;
  const double beta = 0.0;

  sgemm(&trans, &trans, blockDimX, blockDimZ, blockDimY, alpha, A, blockDimX, B, blockDimY, beta, C, blockDimX);
#else
  int i, j, k;

  float *thisa = A[whichLocal];
  float *thisb = B[whichLocal];

  for(i = 0; i < T; i++){
    for(j = 0; j < T; j++){
      float sum = 0.0;
      for(k = 0; k < (K/num_chares_per_dim); k++){
        sum += thisa[i*(K/num_chares_per_dim)+k]*thisb[k*(T)+j];
        //comps++;
      }
      C[i*T+j] += sum;
    }
  }
#endif

  remaining = 2;
  iteration++;
  if(iteration == num_chares_per_dim){
#ifdef MATMUL2D_WRITE_FILE
    // create a file
    //ostringstream oss;
    char buf[128];
    
    sprintf(buf, "mat.%d.%d", row, col);
    //oss << "mat." << row << "." << col;
    //ofstream ofs(oss.str().c_str());
    FILE *fp = fopen(buf, "w");

    for(int i = 0; i < blockDimX; i++){
      for(int j = 0; j < blockDimX; j++){
        //ofs << C[i*blockDimX+j];
        fprintf(fp, "%f ", C[i*blockDimX+j]);
      }
      //ofs << endl;
      fprintf(fp, "\n");

    }
    fclose(fp);
#endif
    contribute(0,0,CkReduction::concat, CkCallback(CkIndex_Main::done(), mainProxy));
    CkPrintf("[%d,%d] comps: %d iter: %d\n", thisIndex.x, thisIndex.y, -1, iteration);
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

  //if(num_chare_x == 0 || num_chare_y ==0)
  //  CkPrintf("(%d,%d): 0 divisor\n", thisIndex.y, thisIndex.x);
#ifdef MATMUL2D_DEBUG
  CkPrintf("(%d,%d): A nbr: (%d,%d), iteration: %d\n", thisIndex.y, thisIndex.x, row, (col-1+num_chare_x)%num_chare_x, iteration);
#endif
  /*
  CkPrintf("(%d,%d): B nbr: (%d,%d)\n", thisIndex.y, thisIndex.x, (row-1+num_chare_y)%num_chare_y, col);
  */

  int size = (K/num_chares_per_dim)*T;

  //Compute *c = thisProxy((col-1+num_chares_per_dim)%num_chares_per_dim, row).ckLocal();
  //if(c == 0){
  thisProxy((col-1+num_chares_per_dim)%num_chares_per_dim, row).recvBlockA(A[whichLocal], size, newBuf);
  //}
  //else{
  //  c->recvBlockA(A[whichLocal], blockDimX*blockDimY, newBuf);
  //}
  // 2. Then put B
  //c =  thisProxy(col, (row-1+num_chare_y)%num_chare_y).ckLocal();
  //if(c == 0){
  thisProxy(col, (row-1+num_chares_per_dim)%num_chares_per_dim).recvBlockB(B[whichLocal], size, newBuf);
  //}
  //else{
  //  c->recvBlockB(B[whichLocal], blockDimX*blockDimY, newBuf);
  //}
  // toggle between local buffers
  whichLocal = newBuf;
}

void Compute::recvBlockA(float *block, int size, int whichBuf){
  memcpy(A[whichBuf], block, sizeof(float)*size);
  remaining--;
  if(remaining == 0){
    compute();
  }
}

void Compute::recvBlockB(float *block, int size, int whichBuf){
  memcpy(B[whichBuf], block, sizeof(float)*size);
  remaining--;
  if(remaining == 0){
    compute();
  }
}

ComputeMap::ComputeMap(int _adx, int _ady){
  arrayDimX = _adx;
  arrayDimY = _ady;

  map = new int[_adx*_ady];
  
}

int ComputeMap::procNum(int arrayHdl, const CkArrayIndex &idx){
  int *index = (int *)(idx.data());
  int row = index[1];
  int col = index[0];

  return map[row*arrayDimX + col];
}
#include "matmul2d.def.h"
