#include "partitioning_strategies.h"
#include "TopoManager.h"
#include "converse.h"

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
 *  Author: Harshitha Menon, Nikhil Jain 
 *  Contact: gplkrsh2@illinois.edu, nikhil@illinois.edu
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

static vector<int> int_to_Hilbert(int i, int dim) {
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
void getPlanarList(int *procList) 
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

namespace {

// class to re-order dimensions in decreasing size
struct TopoManagerWrapper {
  TopoManager tmgr;
  int a_dim, b_dim, c_dim, d_dim, e_dim;
  int a_rot, b_rot, c_rot, d_rot, e_rot;
  int a_mod, b_mod, c_mod, d_mod, e_mod;
  int fixnode(int node) {  // compensate for lame fallback topology information
    return node; // CmiNodeOf(CmiGetFirstPeOnPhysicalNode(CmiPhysicalNodeID(CmiFirstPe(node))));
  }
  TopoManagerWrapper() {
#if CMK_BLUEGENEQ
    int na=tmgr.getDimNA();
    int nb=tmgr.getDimNB();
    int nc=tmgr.getDimNC();
    int nd=tmgr.getDimND();
    int ne=tmgr.getDimNE();
#else
    int na=tmgr.getDimNX();
    int nb=tmgr.getDimNY();
    int nc=tmgr.getDimNZ();
    int nd=1;
    int ne=1;
#endif
    std::vector<int> a_flags(na);
    std::vector<int> b_flags(nb);
    std::vector<int> c_flags(nc);
    std::vector<int> d_flags(nd);
    std::vector<int> e_flags(ne);
    for ( int i=0; i<na; ++i ) { a_flags[i] = 0; }
    for ( int i=0; i<nb; ++i ) { b_flags[i] = 0; }
    for ( int i=0; i<nc; ++i ) { c_flags[i] = 0; }
    for ( int i=0; i<nd; ++i ) { d_flags[i] = 0; }
    for ( int i=0; i<ne; ++i ) { e_flags[i] = 0; }
    int nnodes = CmiNumNodes();
    for ( int node=0; node<nnodes; ++node ) {
      int a,b,c,d,e,t;
#if CMK_BLUEGENEQ
      tmgr.rankToCoordinates(CmiNodeFirst(fixnode(node)),a,b,c,d,e,t);
#else
      tmgr.rankToCoordinates(CmiNodeFirst(fixnode(node)),a,b,c,t);
      d=0; e=0;
#endif
      if ( a < 0 || a >= na ) CmiAbort("inconsistent torus topology!");
      if ( b < 0 || b >= nb ) CmiAbort("inconsistent torus topology!");
      if ( c < 0 || c >= nc ) CmiAbort("inconsistent torus topology!");
      if ( d < 0 || d >= nd ) CmiAbort("inconsistent torus topology!");
      if ( e < 0 || e >= ne ) CmiAbort("inconsistent torus topology!");
      a_flags[a] = 1;
      b_flags[b] = 1;
      c_flags[c] = 1;
      d_flags[d] = 1;
      e_flags[e] = 1;
    }
    std::basic_ostringstream<char> iout;
    iout << "Charm++> " << "TORUS A SIZE " << na << " USING";
    for ( int i=0; i<na; ++i ) { if ( a_flags[i] ) iout << " " << i; }
    iout << "\n" ;
    iout << "Charm++> " << "TORUS B SIZE " << nb << " USING";
    for ( int i=0; i<nb; ++i ) { if ( b_flags[i] ) iout << " " << i; }
    iout << "\n" ;
    iout << "Charm++> " << "TORUS C SIZE " << nc << " USING";
    for ( int i=0; i<nc; ++i ) { if ( c_flags[i] ) iout << " " << i; }
    iout << "\n" ;
#if CMK_BLUEGENEQ
    iout << "Charm++> " << "TORUS D SIZE " << nd << " USING";
    for ( int i=0; i<nd; ++i ) { if ( d_flags[i] ) iout << " " << i; }
    iout << "\n" ;
    iout << "Charm++> " << "TORUS E SIZE " << ne << " USING";
    for ( int i=0; i<ne; ++i ) { if ( e_flags[i] ) iout << " " << i; }
    iout << "\n" ;
#endif
    // find most compact representation of our subset
    a_rot = b_rot = c_rot = d_rot = e_rot = 0;
    a_mod = na; b_mod = nb; c_mod = nc; d_mod = nd; e_mod = ne;
#if CMK_BLUEGENEQ
    if ( tmgr.absA(na) == 0 ) // torus
#else
    if ( tmgr.absX(na) == 0 ) // torus
#endif
      for ( int i=0, gaplen=0, gapstart=0; i<2*na; ++i ) {
        if ( a_flags[i%na] ) gapstart = i+1;
        else if ( i - gapstart >= gaplen ) {
          a_rot = 2*na-i-1; gaplen = i - gapstart;
        }
      }
#if CMK_BLUEGENEQ
    if ( tmgr.absB(nb) == 0 ) // torus
#else
    if ( tmgr.absY(nb) == 0 ) // torus
#endif
      for ( int i=0, gaplen=0, gapstart=0; i<2*nb; ++i ) {
        if ( b_flags[i%nb] ) gapstart = i+1;
        else if ( i - gapstart >= gaplen ) {
          b_rot = 2*nb-i-1; gaplen = i - gapstart;
        }
      }
#if CMK_BLUEGENEQ
    if ( tmgr.absC(nc) == 0 ) // torus
#else
    if ( tmgr.absZ(nc) == 0 ) // torus
#endif
      for ( int i=0, gaplen=0, gapstart=0; i<2*nc; ++i ) {
        if ( c_flags[i%nc] ) gapstart = i+1;
        else if ( i - gapstart >= gaplen ) {
          c_rot = 2*nc-i-1; gaplen = i - gapstart;
        }
      }
#if CMK_BLUEGENEQ
    if ( tmgr.absD(nd) == 0 ) // torus
      for ( int i=0, gaplen=0, gapstart=0; i<2*nd; ++i ) {
        if ( d_flags[i%nd] ) gapstart = i+1;
        else if ( i - gapstart >= gaplen ) {
          d_rot = 2*nd-i-1; gaplen = i - gapstart;
        }
      }
    if ( tmgr.absE(ne) == 0 ) // torus
      for ( int i=0, gaplen=0, gapstart=0; i<2*ne; ++i ) {
        if ( e_flags[i%ne] ) gapstart = i+1;
        else if ( i - gapstart >= gaplen ) {
          e_rot = 2*ne-i-1; gaplen = i - gapstart;
        }
      }
#endif
    // order dimensions by length
    int a_min=na, a_max=-1;
    int b_min=nb, b_max=-1;
    int c_min=nc, c_max=-1;
    int d_min=nd, d_max=-1;
    int e_min=ne, e_max=-1;
    for ( int node=0; node<nnodes; ++node ) {
      int a,b,c,d,e,t;
#if CMK_BLUEGENEQ
      tmgr.rankToCoordinates(CmiNodeFirst(fixnode(node)),a,b,c,d,e,t);
#else
      tmgr.rankToCoordinates(CmiNodeFirst(fixnode(node)),a,b,c,t);
      d=0; e=0;
#endif
      a = (a+a_rot)%a_mod;
      b = (b+b_rot)%b_mod;
      c = (c+c_rot)%c_mod;
      d = (d+d_rot)%d_mod;
      e = (e+e_rot)%e_mod;
      if ( a < a_min ) a_min = a;
      if ( b < b_min ) b_min = b;
      if ( c < c_min ) c_min = c;
      if ( d < d_min ) d_min = d;
      if ( e < e_min ) e_min = e;
      if ( a > a_max ) a_max = a;
      if ( b > b_max ) b_max = b;
      if ( c > c_max ) c_max = c;
      if ( d > d_max ) d_max = d;
      if ( e > e_max ) e_max = e;
    }
    int a_len = a_max - a_min + 1;
    int b_len = b_max - b_min + 1;
    int c_len = c_max - c_min + 1;
    int d_len = d_max - d_min + 1;
    int e_len = e_max - e_min + 1;
    int lensort[5];
    lensort[0] = (a_len << 3) + 4;
    lensort[1] = (b_len << 3) + 3;
    lensort[2] = (c_len << 3) + 2;
    lensort[3] = (d_len << 3) + 1;
    lensort[4] = (e_len << 3) + 0;
    // printf("TopoManagerWrapper lensort before %d %d %d %d %d\n", lensort[0] & 7, lensort[1] & 7, lensort[2] & 7, lensort[3] & 7, lensort[4] & 7);
    std::sort(lensort, lensort+5);
    // printf("TopoManagerWrapper lensort after %d %d %d %d %d\n", lensort[0] & 7, lensort[1] & 7, lensort[2] & 7, lensort[3] & 7, lensort[4] & 7);
    for ( int i=0; i<5; ++i ) { if ( (lensort[i] & 7) == 4 ) a_dim = 4-i; }
    for ( int i=0; i<5; ++i ) { if ( (lensort[i] & 7) == 3 ) b_dim = 4-i; }
    for ( int i=0; i<5; ++i ) { if ( (lensort[i] & 7) == 2 ) c_dim = 4-i; }
    for ( int i=0; i<5; ++i ) { if ( (lensort[i] & 7) == 1 ) d_dim = 4-i; }
    for ( int i=0; i<5; ++i ) { if ( (lensort[i] & 7) == 0 ) e_dim = 4-i; }
    iout << "Charm++> " << "TORUS MINIMAL MESH SIZE IS " << a_len << " BY " << b_len << " BY " << c_len
#if CMK_BLUEGENEQ
    << " BY " << d_len << " BY " << e_len
#endif
    << "\n" ;
    if ( CmiMyNodeGlobal() == 0 ) printf("%s",iout.str().c_str());
    // printf("TopoManagerWrapper dims %d %d %d %d %d\n", a_dim, b_dim, c_dim, d_dim, e_dim);
  }
  void coords(int node, int *crds) {
    int a,b,c,d,e,t;
#if CMK_BLUEGENEQ
    tmgr.rankToCoordinates(CmiNodeFirst(fixnode(node)),a,b,c,d,e,t);
#else
    tmgr.rankToCoordinates(CmiNodeFirst(fixnode(node)),a,b,c,t);
    d=0; e=0;
#endif
    crds[a_dim] = (a+a_rot)%a_mod;
    crds[b_dim] = (b+b_rot)%b_mod;
    crds[c_dim] = (c+c_rot)%c_mod;
    crds[d_dim] = (d+d_rot)%d_mod;
    crds[e_dim] = (e+e_rot)%e_mod;
  }
  int coord(int node, int dim) {
    int crds[5];
    coords(node,crds);
    return crds[dim];
  }
  struct node_sortop_topo {
    TopoManagerWrapper &tmgr;
    const int *sortdims;
    node_sortop_topo(TopoManagerWrapper &t, int *d) : tmgr(t), sortdims(d) {}
    bool operator() (int node1, int node2) const {
      int crds1[5], crds2[5];
      tmgr.coords(node1,crds1);
      tmgr.coords(node2,crds2);
      for ( int i=0; i<5; ++i ) {
        int d = sortdims[i];
        if ( crds1[d] != crds2[d] ) return ( crds1[d] < crds2[d] );
      }
      // int pn1 = CmiPhysicalNodeID(CmiNodeFirst(node1));
      // int pn2 = CmiPhysicalNodeID(CmiNodeFirst(node2));
      // if ( pn1 != pn2 ) return ( pn1 < pn2 );
      return ( node1 < node2 );
    }
  };
  void sortLongest(int *node_begin, int *node_end) {
    if ( node_begin == node_end ) return;
    int tmins[5], tmaxs[5], tlens[5], sortdims[5];
    coords(*node_begin, tmins);
    coords(*node_begin, tmaxs);
    for ( int *nodeitr = node_begin; nodeitr != node_end; ++nodeitr ) {
      int tvals[5];
      coords(*nodeitr, tvals);
      for ( int i=0; i<5; ++i ) {
        if ( tvals[i] < tmins[i] ) tmins[i] = tvals[i];
        if ( tvals[i] > tmaxs[i] ) tmaxs[i] = tvals[i];
      }
    }
    for ( int i=0; i<5; ++i ) {
      tlens[i] = ((tmaxs[i] - tmins[i] + 1) << 3) + (4-i);
    }
    std::sort(tlens, tlens+5);
    for ( int i=0; i<5; ++i ) {
      sortdims[4-i] = 4 - (tlens[i] & 7);
    }
    if ( PARTITION_TOPOLOGY_VERBOSE && CmiMyNodeGlobal() == 0 )
      printf("sorting %d(%d) %d(%d) %d(%d)\n", sortdims[0], tlens[4]>>3, sortdims[1], tlens[3]>>3, sortdims[2], tlens[2]>>3);
    std::sort(node_begin,node_end,node_sortop_topo(*this,sortdims));
    int *nodes = node_begin;
    int nnodes = node_end - node_begin;
  }
};

void recursive_bisect(
  int part_begin, int part_end,
  int *node_begin, int *node_end,
  TopoManagerWrapper &tmgr
  ) {

  if ( part_end - part_begin == 1 ) {
    if ( CmiPartitionSize(part_begin) != node_end - node_begin ) {
      CmiAbort("partitioning algorithm size mismatch in recursive_bisect()");
    }
    tmgr.sortLongest(node_begin, node_end);
    // std::sort(node_begin,node_end);
    if ( PARTITION_TOPOLOGY_VERBOSE && CmiMyNodeGlobal() == 0 ) {
      int crds[5];
      tmgr.coords(*node_begin, crds);
      printf("partitioning node %5d at %5d %5d %5d %5d %5d nodes %5d\n",
               *node_begin,
               crds[0], crds[1], crds[2], crds[3], crds[4],
               node_end - node_begin);
    }
    return;
  }

  if ( PARTITION_TOPOLOGY_VERBOSE && CmiMyNodeGlobal() == 0 )
    printf("recursive_bisect %d %d %d\n", part_begin, part_end, node_end-node_begin);

  int nnodes = node_end - node_begin;
  int nsplit = (nnodes+1) / 2;
  int ncount = 0;
  int part_split = part_begin;
  for ( int p = part_begin; p < part_end; ++p ) {
    int ps = CmiPartitionSize(p);
    if ( abs(ncount+ps-nsplit) < abs(ncount-nsplit) ) part_split = p+1;
    else break;
    ncount += ps;
  }
  if ( part_split == part_begin || part_split == part_end )
    CmiAbort("partitioning algorithm failure in recursive_bisect()");

  int *node_split = node_begin + ncount;

  tmgr.sortLongest(node_begin, node_end);

  // recurse
  recursive_bisect(
    part_begin, part_split, node_begin, node_split, tmgr);
  recursive_bisect(
    part_split, part_end, node_split, node_end, tmgr);
}

} // anonymous namespace

/** \brief A function to traverse the given processors, and get a recursive bisection list
 */
void getRecursiveBisectionList(int numparts, int *procList)
{
  int n = CmiNumNodes();
  for(int i = 0; i < n; i++) {
    procList[i] = i;
  }
  TopoManagerWrapper tmgr;
  recursive_bisect(
    0, numparts, procList, procList + n, tmgr);
  if ( PARTITION_TOPOLOGY_VERBOSE && CmiMyNodeGlobal() == 0 ) {
    int crds[5];
    for ( int i=0,p=0,ip=0; i<n; ++i,++ip ) {
      if ( p == numparts ) break;  // this shouldn't happen
      if ( ip == CmiPartitionSize(p) ) { ++p; ip=0; }
      tmgr.coords(procList[i],crds);
      printf("procList[%5d] part[%3d] %5d (%2d %2d %2d %2d %2d)\n",
        i, p, procList[i],
        crds[0], crds[1], crds[2], crds[3], crds[4], crds[5]);
    }
  }
}


#endif
