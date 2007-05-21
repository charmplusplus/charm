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

int TopoManager::hasMultipleProcsPerNode() {
#ifdef CMK_VERSION_BLUEGENE
  if(procsPerNode==1)
    return 0;
  else
    return 1;
#elif CMK_XT3
  return 0;
#else
  return 0;
#endif
}

void TopoManager::rankToCoordinates(int pe, int &x, int &y, int &z) {
#ifdef CMK_VERSION_BLUEGENE
  bgltm->getCoordinatesByRank(pe, x, y, z);
#elif CMK_XT3

#else
  if(dimY>0){
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

int TopoManager::coordinatesToRank(int x, int y, int z) {
#ifdef CMK_VERSION_BLUEGENE
  return bgltm->coords2rank(x, y, z);
#elif CMK_XT3

#else
  if(dimY > 0)
    return x + y*dimX + z*dimX*dimY;
  else
    return x;
#endif
}

int TopoManager::getHopsBetweenRanks(int pe1, int pe2) {
  int x1, y1, z1, x2, y2, z2;
  rankToCoordinates(pe1, x1, y1, z1);
  rankToCoordinates(pe2, x2, y2, z2);
  return (absX(x2-x1)+absY(y2-y1)+absZ(z2-z1));
}

void TopoManager::sortRanksByHops(int pe, int *pes, int *idx, int n) {
  int minHops = getHopsBetweenRanks(pe, pes[0]);
  int minIdx = 0;
  int nowHops, tmp;
  for(int i=0;i<n;i++)
    idx[i] = i;
  quicksort(pe, pes, idx, 0, n-1);
}

/*
int TopoManager::pickClosestRank(int mype, int *pes, int n) {
#ifdef CMK_VERSION_BLUEGENE
  return(bgltm->pickClosestRank(mype, pes, n));
#elif CMK_XT3
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
#ifdef CMK_VERSION_BLUEGENE
  return(bgltm->isNeighborOfBoth(pe1, pe2, pe3, distance));
#elif CMK_XT3
#else 
  if(abs(pe1-pe2) + abs(pe2-pe3) <= distance)
    return 1;
  else
    return 0;
#endif
}

/*int TopoManager::getConeNumberForRank(int pe) {
#ifdef CMK_VERSION_BLUEGENE
  return(bgltm->getConeNumberForRank(pe));
#elif CMK_XT3
#else 
    return 0;
#endif
}*/

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
