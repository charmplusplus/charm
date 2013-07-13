#include "partitioning_strategies.h"
#include "TopoManager.h"
#include "converse.h"

#ifdef __cplusplus
#include <queue>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>

using namespace std;

/** 
 *  Author: Harshitha Menon 
 *  Contact: gplkrsh2@illinois.edu 
 */
static int gray_encode(int i) {
  //cout << "gray_encode " << i << " " << (i^(i/2)) << endl;
  return i ^ (i/2);
}

static int gray_decode(int n) {
  int sh = 1;
  int div;
  while (true) {
    div = n >> sh;
    n ^= div;
    if (div <= 1) {
      return n;
    }
    sh <<=1;
  }
}

static void initial_start_end(int nChunks, int dim, int& start, int& end) {
  start = 0;
  ////cout << "-nchunks - 1 mod dim " << ((-nChunks-1)%dim) << endl;
  int mod_val = (-nChunks - 1) % dim;
  if (mod_val < 0) {
    mod_val += dim;
  }
  end = pow(2, mod_val);
}

static int pack_index(vector<int> chunks, int dim) {
  int p = pow(2, dim);
  int chunk_size = chunks.size();
  int val = 0;
  for (int i = 0;i < chunk_size; i++) {
    val += chunks[i] * p;
  }
  return val;
}

static vector<int> transpose_bits(vector<int> srcs, int nDests) {
  int nSrcs = srcs.size();
  vector<int> dests;
  dests.resize(nDests);
  int dest = 0;
  for (int j = nDests - 1; j > -1; j--) {
    dest = 0;
    ////cout << "nSrcs " << nSrcs << endl;
    for (int k = 0; k < nSrcs; k++) {
      dest = dest * 2 + srcs[k] % 2;
      srcs[k] /= 2;
      ////cout << "dest " << dest << endl;
    }
    dests[j] = dest;
  }
  return dests;
}

static vector<int> pack_coords(vector<int> coord_chunks, int dim) {
  return transpose_bits(coord_chunks, dim);
}

static void unpack_index(int i, int dim, vector<int>& chunks) {
  int p = pow(2, dim);
  int nChunks = max(1, int(ceil(double(log(i+1))/log(p))));
  //cout << "num chunks " << nChunks << endl;
  chunks.resize(nChunks); 
  for (int j = nChunks-1; j > -1; j--) {
    chunks[j] = i % p;
    i /= p;
    ////cout << "chunks[" << j << "] " << chunks[j] << endl;
  }
  //cout << "chunk size " << chunks.size() << endl;
}

static int gray_encode_travel(int start, int end, int mask, int i) {
  int travel_bit = start ^ end;
  int modulus = mask + 1;
  int g = gray_encode(i) * (travel_bit * 2);
  //cout << "start " << start << " end " << end << "travel_bits " << travel_bit << " modulus " << modulus << " g " << g << endl;
  return ((g | (g/modulus) ) & mask) ^ start;
}

static int gray_decode_travel(int start, int end, int mask, int i) {
  int travel_bit = start ^ end;
  int modulus = mask + 1;
  int rg = (i ^ start) * ( modulus/(travel_bit * 2));
  return gray_decode((rg | (rg / modulus)) & mask);
}

static void child_start_end(int parent_start, int parent_end, int mask, int i, int&
    child_start, int& child_end) {
  int start_i = max(0, (i-1)&~1);
  //cout << "start_i " << start_i << endl;
  int end_i = min(mask, (i+1)|1);
  //cout << "end_i " << end_i << endl;
  child_start = gray_encode_travel(parent_start, parent_end, mask, start_i);
  child_end = gray_encode_travel(parent_start, parent_end, mask, end_i);
  //cout << "child_start " << child_start << " child end " << child_end << endl;
}

static vector<int> int_to_Hilbert(int i, int dim) {
  int nChunks, mask, start, end;
  vector<int> index_chunks;
  unpack_index(i, dim, index_chunks);
  nChunks = index_chunks.size();
  //cout << "int to hilbert of " << i << " in dim " << dim << " size " << nChunks << endl;
  mask = pow(2, dim) - 1;
  //cout << "mask " << mask << endl;
  initial_start_end(nChunks, dim, start, end);
  //cout << "start " << start << " end " << end << endl;
  vector<int> coord_chunks;
  coord_chunks.resize(nChunks);
  for (int j = 0; j < nChunks; j++) {
    i = index_chunks[j];
    coord_chunks[j] = gray_encode_travel(start, end, mask, i);
    //cout << "coord_chuunk " << j << " : " << coord_chunks[j] << endl;
    ////cout << "going for child start end" << endl;
    child_start_end(start, end, mask, i, start, end);
    //cout << "child start end " << start << "  " << end << endl;
  }
  //return pack_index(index_chunks, dim);
  //cout << "coords size " << coord_chunks.size() << endl;
  return pack_coords(coord_chunks, dim);
}

/** \brief A function to traverse the given processors, and get a hilbert list
 */
void getHilbertList(int * procList)
{
  vector<int> hcoords;

  int numDims;
  int *dims, *pdims;
  int *ranks, numranks;

  TopoManager_getDimCount(&numDims);

  dims = new int[numDims+1];
  pdims = new int[numDims+1];

  TopoManager_getDims(dims);
  ranks = new int[dims[numDims]];

  //our hilbert only works for power of 2
  int maxDim = dims[0];
  for(int i = 1; i < numDims; i++) {
    if(maxDim < dims[i]) maxDim = dims[i];
  }

  int pow2 = 1;
  while(maxDim>pow2)
    pow2 *= 2;

  int cubeDim = pow2;

  for(int i = 1; i < numDims; i++) {
    cubeDim *= pow2;
  }

  int currPos = 0;
  for(int i = 0; i < cubeDim; i++) {
    hcoords = int_to_Hilbert(i,numDims);

    for(int i = 0; i < numDims; i++) {
      if(hcoords[i] >= dims[i]) continue;
    }

    //check if physical node is allocatd to us
    for(int i = 0; i < numDims; i++) {
      pdims[i] = hcoords[i]; 
    }
    TopoManager_getRanks(&numranks, ranks, pdims);
    if(numranks == 0) continue; 

    //check if both chips on the gemini were allocated to us
    for(int j = 0; j < numranks; j++) {
      procList[currPos++] = ranks[j];
    }
  }

  CmiAssert(currPos == CmiNumNodes());

  delete [] dims;
  delete [] pdims;
  delete [] ranks;
}

/** \brief A function to traverse the given processors, and get a planar list
 */
void * getPlanarList(int *procList) 
{
  int numDims;
  int *dims, *pdims;
  int *ranks, numranks;
  int phynodes;

  TopoManager_getDimCount(&numDims);

  dims = new int[numDims+1];
  pdims = new int[numDims+1];

  TopoManager_getDims(dims);
  ranks = new int[dims[numDims]];

  phynodes = 1;
  for(int i = 0; i < numDims; i++) {
    phynodes *= dims[i];
    pdims[i] = 0;
  }

  int currPos = 0;
  for(int i = 0; i < phynodes; i++) {

    TopoManager_getRanks(&numranks, ranks, pdims);
    for(int j = numDims - 1; j > -1; j--) {
      pdims[j] = (pdims[j]+1) % dims[j];
      if(pdims[j] != 0) break;
    }
    if(numranks == 0) continue; 

    for(int j = 0; j < numranks; j++) {
      procList[currPos++] = ranks[j];
    }
  }

  CmiAssert(currPos == CmiNumNodes());

  delete [] dims;
  delete [] pdims;
  delete [] ranks;
}

#endif
