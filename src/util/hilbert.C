#include "hilbert.h"

#ifdef __cplusplus
#include <queue>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <math.h>

using namespace std;

#ifndef PARTITION_TOPOLOGY_VERBOSE
#define PARTITION_TOPOLOGY_VERBOSE 0
#endif

/** 
 *  Author: Harshitha Menon, Nikhil Jain, Yanhua Sun
 *  Contact: gplkrsh2@illinois.edu, nikhil@illinois.edu, sun51@illinois.edu
 *
 *  More details about this implementation of the Hilbert curve can be found
 *  from https://github.com/straup/gae-spacetimeid/blob/master/hilbert.py
 *  and this is a C++ implementation of what is given there.
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
  end = pow((double)2, (double)mod_val);
}

static int pack_index(vector<int> chunks, int dim) {
  int p = pow((double)2, (double)dim);
  int chunk_size = chunks.size();
  int val = 0;
  for (int i = 0;i < chunk_size; i++) {
      val = val*p + chunks[i] ;
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

static vector<int> unpack_coords(const vector<int> &coords, int dim) {
    int biggest = 0;
    for(int i=0; i<coords.size(); i++)
    {
        if(coords[i] > biggest)
            biggest = coords[i];
    }
    int nChunks = max(1, int( ceil( log( (double)biggest + 1)/log(2.0f) ) ) );
    return transpose_bits( coords, nChunks );
}


static void unpack_index(int i, int dim, vector<int>& chunks) {
  int p = pow((double)2, (double)dim);
  int nChunks = max(1, int(ceil(double(log((double)i+1))/log((double)p))));
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

vector<int> int_to_Hilbert(int i, int dim) {
  int nChunks, mask, start, end;
  vector<int> index_chunks;
  unpack_index(i, dim, index_chunks);
  nChunks = index_chunks.size();
  //cout << "int to hilbert of " << i << " in dim " << dim << " size " << nChunks << endl;
  mask = pow((double)2, (double)dim) - 1;
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
  return pack_coords(coord_chunks, dim);
}

int Hilbert_to_int(const vector<int>& coords, int dim)
{
    vector<int> coord_chunks =  unpack_coords( coords, dim );
    int i;
    int nChunks = coord_chunks.size();
    int mask = pow((double)2, (double)dim) - 1;
    int start, end;
    initial_start_end(nChunks, dim, start, end);
    vector<int> index_chunks;
    index_chunks.resize(nChunks);
    for (int j = 0; j < nChunks; j++) {
        i = gray_decode_travel( start, end, mask, coord_chunks[ j ] );
        index_chunks[ j ] = i;
        child_start_end(start, end, mask, i, start, end);
    }

    return pack_index( index_chunks, dim );
}

#endif
