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
  startTime = CmiWallTimer();
  compute.beginCopying();
}

// Constructor, initialize values
Compute::Compute() {

}

Compute::Compute(CkMigrateMessage* m) {

}

Compute::~Compute() {

}

void Compute::beginCopying() {

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


