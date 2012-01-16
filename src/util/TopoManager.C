/** \file TopoManager.C
 *  Author: Abhinav S Bhatele
 *  Date Created: March 19th, 2007
 *
 *  This would be the top level interface for all topology managers we
 *  will write for different machines (cray, bg/l ... for tori, meshes ...)
 *  Current plan is to have functionality for Blue Gene/L, Cray XT3,
 *  BigSim and non-topo machines.
 */

#include "TopoManager.h"

TopoManager::TopoManager() {
#if CMK_BLUEGENEL
  dimX = bgltm.getDimX();
  dimY = bgltm.getDimY();
  dimZ = bgltm.getDimZ();

  dimNX = bgltm.getDimNX();
  dimNY = bgltm.getDimNY();
  dimNZ = bgltm.getDimNZ();
  dimNT = bgltm.getDimNT();

  procsPerNode = bgltm.getProcsPerNode();
  int *torus;
  torus = bgltm.isTorus();
  torusX = torus[0];
  torusY = torus[1];
  torusZ = torus[2];
  torusT = torus[3];

#elif CMK_BLUEGENEP
  dimX = bgptm.getDimX();
  dimY = bgptm.getDimY();
  dimZ = bgptm.getDimZ();

  dimNX = bgptm.getDimNX();
  dimNY = bgptm.getDimNY();
  dimNZ = bgptm.getDimNZ();
  dimNT = bgptm.getDimNT();

  procsPerNode = bgptm.getProcsPerNode();
  int *torus;
  torus = bgptm.isTorus();
  torusX = torus[0];
  torusY = torus[1];
  torusZ = torus[2];
  torusT = torus[3];

#elif XT3_TOPOLOGY
  dimX = xt3tm.getDimX();
  dimY = xt3tm.getDimY();
  dimZ = xt3tm.getDimZ();

  dimNX = xt3tm.getDimNX();
  dimNY = xt3tm.getDimNY();
  dimNZ = xt3tm.getDimNZ();
  dimNT = xt3tm.getDimNT();

  procsPerNode = xt3tm.getProcsPerNode();
  int *torus;
  torus = xt3tm.isTorus();
  torusX = torus[0];
  torusY = torus[1];
  torusZ = torus[2];
  torusT = torus[3];

#elif XT4_TOPOLOGY || XT5_TOPOLOGY
  dimX = xttm.getDimX();
  dimY = xttm.getDimY();
  dimZ = xttm.getDimZ();

  dimNX = xttm.getDimNX();
  dimNY = xttm.getDimNY();
  dimNZ = xttm.getDimNZ();
  dimNT = xttm.getDimNT();

  procsPerNode = xttm.getProcsPerNode();
  int *torus;
  torus = xttm.isTorus();
  torusX = torus[0];
  torusY = torus[1];
  torusZ = torus[2];
  torusT = torus[3];

#else
  dimX = CmiNumPes();
  dimY = 1;
  dimZ = 1;

  dimNX = dimX;
  dimNY = 1;
  dimNZ = 1;

  dimNT = procsPerNode = 1;
  torusX = true;
  torusY = true;
  torusZ = true;
  torusT = false;
#endif

#if CMK_BIGSIM_CHARM
  BgGetSize(&dimNX, &dimNY, &dimNZ);

  dimNT = procsPerNode = BgGetNumWorkThread();
  dimX = dimNX * procsPerNode;
  dimY = dimNY;
  dimZ = dimNZ;

  torusX = true;
  torusY = true;
  torusZ = true;
  torusT = false;
#endif

  numPes = dimNX * dimNY * dimNZ * dimNT;
}

TopoManager::TopoManager(int NX, int NY, int NZ, int NT) : dimNX(NX), dimNY(NY), dimNZ(NZ), dimNT(NT) {
  /* we rashly assume only one dimension is expanded */
  procsPerNode = dimNT;
  dimX = dimNX * dimNT;
  dimY = dimNY;
  dimZ = dimNZ;
  torusX = true;
  torusY = true;
  torusZ = true;
  numPes = dimNX * dimNY * dimNZ * dimNT;
}

int TopoManager::hasMultipleProcsPerNode() const {
  if(procsPerNode == 1)
    return 0;
  else
    return 1;
}

void TopoManager::rankToCoordinates(int pe, int &x, int &y, int &z) {
  CmiAssert( pe >= 0 && pe < numPes );
#if CMK_BLUEGENEL
  bgltm.rankToCoordinates(pe, x, y, z);
#elif CMK_BLUEGENEP
  bgptm.rankToCoordinates(pe, x, y, z);
#elif XT3_TOPOLOGY || XT4_TOPOLOGY || XT5_TOPOLOGY
  CmiAbort("This function should not be called on Cray XT machines\n");
#else
  if(dimY > 1){
    // Assumed TXYZ
    x = pe % dimX;
    y = (pe % (dimX * dimY)) / dimX;
    z = pe / (dimX * dimY);
  }
  else {
    x = pe; 
    y = 0; 
    z = 0;
  }
#endif

#if CMK_BIGSIM_CHARM
  if(dimY > 1){
    // Assumed TXYZ
    x = pe % dimX;
    y = (pe % (dimX * dimY)) / dimX;
    z = pe / (dimX * dimY);
  }
  else {
    x = pe; 
    y = 0; 
    z = 0;
  }
#endif
}

void TopoManager::rankToCoordinates(int pe, int &x, int &y, int &z, int &t) {
  CmiAssert( pe >= 0 && pe < numPes );
#if CMK_BLUEGENEL
  bgltm.rankToCoordinates(pe, x, y, z, t);
#elif CMK_BLUEGENEP
  bgptm.rankToCoordinates(pe, x, y, z, t);
#elif XT3_TOPOLOGY
  xt3tm.rankToCoordinates(pe, x, y, z, t);
#elif XT4_TOPOLOGY || XT5_TOPOLOGY
  xttm.rankToCoordinates(pe, x, y, z, t);
#else
  if(dimNY > 1) {
    t = pe % dimNT;
    x = (pe % (dimNT*dimNX)) / dimNT;
    y = (pe % (dimNT*dimNX*dimNY)) / (dimNT*dimNX);
    z = pe / (dimNT*dimNX*dimNY);
  } else {
    t = pe % dimNT;
    x = (pe % (dimNT*dimNX)) / dimNT;
    y = 0;
    z = 0;
  }
#endif

#if CMK_BIGSIM_CHARM
  if(dimNY > 1) {
    t = pe % dimNT;
    x = (pe % (dimNT*dimNX)) / dimNT;
    y = (pe % (dimNT*dimNX*dimNY)) / (dimNT*dimNX);
    z = pe / (dimNT*dimNX*dimNY);
  } else {
    t = pe % dimNT;
    x = (pe % (dimNT*dimNX)) / dimNT;
    y = 0;
    z = 0;
  }
#endif
}

int TopoManager::coordinatesToRank(int x, int y, int z) {
  CmiAssert( x>=0 && x<dimX && y>=0 && y<dimY && z>=0 && z<dimZ );
#if CMK_BIGSIM_CHARM
  if(dimY > 1)
    return x + y*dimX + z*dimX*dimY;
  else
    return x;
#endif

#if CMK_BLUEGENEL
  return bgltm.coordinatesToRank(x, y, z);
#elif CMK_BLUEGENEP
  return bgptm.coordinatesToRank(x, y, z);
#elif XT3_TOPOLOGY || XT4_TOPOLOGY || XT5_TOPOLOGY
  CmiAbort("This function should not be called on Cray XT machines\n");
  return -1;
#else
  if(dimY > 1)
    return x + y*dimX + z*dimX*dimY;
  else
    return x;
#endif
}

int TopoManager::coordinatesToRank(int x, int y, int z, int t) {
  CmiAssert( x>=0 && x<dimNX && y>=0 && y<dimNY && z>=0 && z<dimNZ && t>=0 && t<dimNT );
#if CMK_BIGSIM_CHARM
  if(dimNY > 1)
    return t + (x + (y + z*dimNY) * dimNX) * dimNT;
  else
    return t + x * dimNT;
#endif

#if CMK_BLUEGENEL
  return bgltm.coordinatesToRank(x, y, z, t);
#elif CMK_BLUEGENEP
  return bgptm.coordinatesToRank(x, y, z, t);
#elif XT3_TOPOLOGY
  return xt3tm.coordinatesToRank(x, y, z, t);
#elif XT4_TOPOLOGY || XT5_TOPOLOGY
  return xttm.coordinatesToRank(x, y, z, t);
#else
  if(dimNY > 1)
    return t + (x + (y + z*dimNY) * dimNX) * dimNT;
  else
    return t + x * dimNT;
#endif
}

int TopoManager::getHopsBetweenRanks(int pe1, int pe2) {
  CmiAssert( pe1 >= 0 && pe1 < numPes );
  CmiAssert( pe2 >= 0 && pe2 < numPes );
  int x1, y1, z1, x2, y2, z2, t1, t2;
  rankToCoordinates(pe1, x1, y1, z1, t1);
  rankToCoordinates(pe2, x2, y2, z2, t2);
  return (absX(x2-x1)+absY(y2-y1)+absZ(z2-z1));
}

void TopoManager::sortRanksByHops(int pe, int *pes, int *idx, int n) {
  /* The next three lines appear to do nothing other than waste time.
     int minHops = getHopsBetweenRanks(pe, pes[0]);
     int minIdx = 0;
     int nowHops, tmp;
  */
  for(int i=0;i<n;i++)
    idx[i] = i;
  quicksort(pe, pes, idx, 0, n-1);
}

/*
int TopoManager::pickClosestRank(int mype, int *pes, int n) {
#if CMK_BLUEGENEL
  return(bgltm->pickClosestRank(mype, pes, n));
#elif XT3_TOPOLOGY
#else 
  return(pickClosestRank(mype,pes,n));
#endif
}
*/

int TopoManager::pickClosestRank(int mype, int *pes, int n){
  int minHops = getHopsBetweenRanks(mype, pes[0]);
  int minIdx=0;
  int nowHops; 
  for(int i=1; i<n; i++) {
    nowHops = getHopsBetweenRanks(mype, pes[i]);
    if(nowHops < minHops) {
      minHops = nowHops;
      minIdx=i;
    }
  }
  return minIdx;
}

int TopoManager::areNeighbors(int pe1, int pe2, int pe3, int distance) {
  int pe1_x, pe1_y, pe1_z, pe1_t;
  int pe2_x, pe2_y, pe2_z, pe2_t;
  int pe3_x, pe3_y, pe3_z, pe3_t;

  rankToCoordinates(pe1, pe1_x, pe1_y, pe1_z, pe1_t);
  rankToCoordinates(pe2, pe2_x, pe2_y, pe2_z, pe2_t);
  rankToCoordinates(pe3, pe3_x, pe3_y, pe3_z, pe3_t);

  if ( (absX(pe1_x - (pe2_x+pe3_x)/2) + absY(pe1_y - (pe2_y+pe3_y)/2) + absZ(pe1_z - (pe2_z+pe3_z)/2)) <= distance )
    return 1;
  else
    return 0;
}

void TopoManager::quicksort(int pe, int *pes, int *arr, int left, int right) {
  if(left<right) {
    int split = partition(pe, pes, arr, left, right);
    quicksort(pe, pes, arr, left, split);
    quicksort(pe, pes, arr, split+1, right);
  }
}

int TopoManager::partition(int pe, int *pes, int *idx, int left, int right) {
  int val = getHopsBetweenRanks(pe, pes[idx[(left+right)/2]]);
  int lm = left-1;
  int rm = right+1;
  for(;;) {
    do
      rm--;
    while(getHopsBetweenRanks(pe, pes[idx[rm]]) > val);
    do
      lm++;
    while(getHopsBetweenRanks(pe, pes[idx[lm]]) < val);
    if(lm < rm) {
      int tmp = idx[rm];
      idx[rm] = idx[lm];
      idx[lm] = tmp;
    }
    else
      return rm;
  }
}

