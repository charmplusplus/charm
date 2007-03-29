/** \file TopoManager.C
 *  Author: Abhinav S Bhatele
 *  Date Created: March 19th, 2007
 *
 *  This would be the top level interface for all topology managers we
 *  will write for different machines (cray, bg/l ... for tori, meshes ...)
 *  Current plan is to have functionality for Blue Gene/L, Cray XT3,
 *  BigSim and non-topo machines.
 */

#include "ck.h"
#include "TopoManager.h"

TopoManager::TopoManager() {
#ifdef CMK_VERSION_BLUEGENE
  bgltm = BGLTorusManager::getObject();
  
  dimX = bgltm->getXSize();
  dimY = bgltm->getYSize();
  dimZ = bgltm->getZSize();

  dimNX = bgltm->getXNodeSize();
  dimNY = bgltm->getYNodeSize();
  dimNZ = bgltm->getZNodeSize();

  if(bgltm->isVnodeMode())
    procsPerNode = 2;
  else
    procsPerNode = 1;

#elif CMK_XT3

#else
  dimX = CkNumPes();
  dimY = 1;
  dimZ = 1;

  dimNX = dimX;
  dimNY = 1;
  dimNZ = 1;

  procsPerNode = 1;
#endif
}

TopoManager::~TopoManager() {

}

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
  x = pe; y = 1; z = 1;
#endif
}

int TopoManager::coordinatesToRank(int x, int y, int z) {
#ifdef CMK_VERSION_BLUEGENE
  return bgltm->coords2rank(x, y, z);
#elif CMK_XT3

#else
  return x;
#endif
}

int TopoManager::getHopsBetweenRanks(int pe1, int pe2) {
#ifdef CMK_VERSION_BLUEGENE
  return bgltm->getHopsBetweenRanks(pe1, pe2);
#elif CMK_XT3

#else
  return abs(pe1-pe2);
#endif
}

void TopoManager::sortRanksByHops(int pe, int *pes, int *idx, int n) {
  int minHops = getHopsBetweenRanks(pe, pes[0]);
  int minIdx = 0;
  int nowHops, tmp;
  for(int i=0;i<n;i++)
    idx[i] = i;
  quicksort(pe, pes, idx, 0, n-1);
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
