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
#include "partitioning_strategies.h"

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

#elif CMK_BLUEGENEQ
  dimX = bgqtm.getDimX();
  dimY = bgqtm.getDimY();
  dimZ = bgqtm.getDimZ();

  dimNX = bgqtm.getDimNX();
  dimNY = bgqtm.getDimNY();
  dimNZ = bgqtm.getDimNZ();
  dimNT = bgqtm.getDimNT();
  
  dimNA = bgqtm.getDimNA();
  dimNB = bgqtm.getDimNB();
  dimNC = bgqtm.getDimNC();
  dimND = bgqtm.getDimND();
  dimNE = bgqtm.getDimNE();

  procsPerNode = bgqtm.getProcsPerNode();
  int *torus;
  torus = bgqtm.isTorus();
  torusA = torus[0];
  torusB = torus[1];
  torusC = torus[2]; 
  torusD = torus[3];
  torusE = torus[4];

#elif XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
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

  numPes = CmiNumPes();
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
#if CMK_BLUEGENEQ
  torusA = true;
  torusB = true;
  torusC = true;
  torusD = true;
  torusE = true;
#endif
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
#elif XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
	int t;
  xttm.rankToCoordinates(pe, x, y, z, t);
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
#elif CMK_BLUEGENEQ
  bgqtm.rankToCoordinates(pe, x, y, z, t);
#elif XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
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

#if CMK_BLUEGENEQ
void TopoManager::rankToCoordinates(int pe, int &a, int &b, int &c, int &d, int &e, int &t) {
  CmiAssert( pe >= 0 && pe < numPes );
  bgqtm.rankToCoordinates(pe, a, b, c, d, e, t);
}
#endif

int TopoManager::coordinatesToRank(int x, int y, int z) {
  if(!( x>=0 && x<dimX && y>=0 && y<dimY && z>=0 && z<dimZ ))
    return -1;
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
#elif XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
  return xttm.coordinatesToRank(x, y, z, 0);
#else
  if(dimY > 1)
    return x + y*dimX + z*dimX*dimY;
  else
    return x;
#endif
}

int TopoManager::coordinatesToRank(int x, int y, int z, int t) {
  if(!( x>=0 && x<dimNX && y>=0 && y<dimNY && z>=0 && z<dimNZ && t>=0 && t<dimNT ))
    return -1;
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
#elif CMK_BLUEGENEQ
  return bgqtm.coordinatesToRank(x, y, z, t);
#elif XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
  return xttm.coordinatesToRank(x, y, z, t);
#else
  if(dimNY > 1)
    return t + (x + (y + z*dimNY) * dimNX) * dimNT;
  else
    return t + x * dimNT;
#endif
}

#if CMK_BLUEGENEQ
int TopoManager::coordinatesToRank(int a, int b, int c, int d, int e, int t) {
  if(!( a>=0 && a<dimNA && b>=0 && b<dimNB && c>=0 && c<dimNC && d>=0 && d<dimND && e>=0 && e<dimNE && t>=0 && t<dimNT ))
    return -1;
  return bgqtm.coordinatesToRank(a, b, c, d, e, t);
}
#endif

int TopoManager::getHopsBetweenRanks(int pe1, int pe2) {
  CmiAssert( pe1 >= 0 && pe1 < numPes );
  CmiAssert( pe2 >= 0 && pe2 < numPes );
#if CMK_BLUEGENEQ
  int a1, b1, c1, d1, e1, t1, a2, b2, c2, d2, e2, t2; 
  rankToCoordinates(pe1, a1, b1, c1, d1, e1, t1);
  rankToCoordinates(pe2, a2, b2, c2, d2, e2, t2);
  return (absA(a2-a1)+absB(b2-b1)+absC(c2-c1)+absD(d2-d1)+absE(e2-e1));  
#else
  int x1, y1, z1, x2, y2, z2, t1, t2;
  rankToCoordinates(pe1, x1, y1, z1, t1);
  rankToCoordinates(pe2, x2, y2, z2, t2);
  return (absX(x2-x1)+absY(y2-y1)+absZ(z2-z1));
#endif
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
#if CMK_BLUEGENEQ
  int pe1_a, pe1_b, pe1_c, pe1_d, pe1_e, pe1_t;
  int pe2_a, pe2_b, pe2_c, pe2_d, pe2_e, pe2_t;
  int pe3_a, pe3_b, pe3_c, pe3_d, pe3_e, pe3_t;
  rankToCoordinates(pe1, pe1_a, pe1_b, pe1_c, pe1_d, pe1_e, pe1_t);
  rankToCoordinates(pe2, pe2_a, pe2_b, pe2_c, pe2_d, pe2_e, pe2_t);
  rankToCoordinates(pe3, pe3_a, pe3_b, pe3_c, pe3_d, pe3_e, pe3_t);

  if ( (absA(pe1_a - (pe2_a+pe3_a)/2) + absB(pe1_b - (pe2_b+pe3_b)/2)+absC(pe1_c - (pe2_c+pe3_c)/2)+absD(pe1_d - (pe2_d+pe3_d)/2)+absE(pe1_e - (pe2_e+pe3_e)/2)) <= distance )
    return 1;
  else
    return 0;
#else
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
#endif
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

#if CMK_BLUEGENEQ
void TopoManager::printAllocation(FILE *fp)
{
	int i,a,b,c,d,e,t;
	fprintf(fp, "Topology Info-\n");
	fprintf(fp, "NumPes -  %d\n", numPes);
	fprintf(fp, "Dims - %d %d %d %d %d\n",dimNA,dimNB,dimNC,dimND,dimNE);
	fprintf(fp, "Rank - a b c d e t\n");
	for(i=0; i<numPes; i++) {
		rankToCoordinates(i,a,b,c,d,e,t);
		fprintf(fp, "%d/%d - %d/%d - %d %d %d %d %d %d\n",CmiGetPeGlobal(i,CmiMyPartition()),CmiGetNodeGlobal(CmiNodeOf(i),CmiMyPartition()),i,CmiNodeOf(i),a,b,c,d,e,t);
	}
}
#else
void TopoManager::printAllocation(FILE *fp)
{
	int i,x,y,z,t;
	fprintf(fp, "Topology Info-\n");
	fprintf(fp, "NumPes -  %d\n", numPes);
	fprintf(fp, "Dims - %d %d %d\n",dimNX,dimNY,dimNZ);
	fprintf(fp, "GlobalPe/GlobalNode - LocalPe/LocalNode - x y z t\n");
	for(i=0; i<numPes; i++) {
		rankToCoordinates(i,x,y,z,t);
		fprintf(fp, "%d/%d - %d/%d - %d %d %d %d\n",CmiGetPeGlobal(i,CmiMyPartition()),CmiGetNodeGlobal(CmiNodeOf(i),CmiMyPartition()),i,CmiNodeOf(i),x,y,z,t);
	}
}
#endif

#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
extern "C" void craynid_init();
extern "C" void craynid_reset();
extern "C" void craynid_free();
#endif

CmiNodeLock _topoLock = 0;
TopoManager *_tmgr = NULL;

extern "C" void TopoManager_init()
{
  if(_topoLock == 0) {
    _topoLock = CmiCreateLock();
#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
    craynid_init();
#endif
  }
}

extern "C" void TopoManager_reset() {
  CmiLock(_topoLock);
#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
  craynid_reset();
#endif
  if(_tmgr) delete _tmgr;
  _tmgr = new TopoManager;
  CmiUnlock(_topoLock);
}

extern "C" void TopoManager_free() {
  CmiLock(_topoLock);
  if(_tmgr) delete _tmgr;
  _tmgr = NULL;
#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
  craynid_free();
#endif
  CmiUnlock(_topoLock);
}

extern "C" void TopoManager_printAllocation(FILE *fp) {
  if(_tmgr == NULL) { TopoManager_reset(); }
  _tmgr->printAllocation(fp);
}

extern "C" void TopoManager_getDimCount(int *ndims) {
#if CMK_BLUEGENEQ
  *ndims = 5;
#else
  *ndims = 3;
#endif
}

extern "C" void TopoManager_getDims(int *dims) {
  if(_tmgr == NULL) { TopoManager_reset(); }
#if CMK_BLUEGENEQ
  dims[0] = _tmgr->getDimNA();
  dims[1] = _tmgr->getDimNB();
  dims[2] = _tmgr->getDimNC();
  dims[3] = _tmgr->getDimND();
  dims[4] = _tmgr->getDimNE();
  dims[5] = _tmgr->getDimNT()/CmiMyNodeSize();
#else 
  dims[0] = _tmgr->getDimNX();
  dims[1] = _tmgr->getDimNY();
  dims[2] = _tmgr->getDimNZ();
  dims[3] = _tmgr->getDimNT()/CmiMyNodeSize();
#endif
}

extern "C" void TopoManager_getCoordinates(int rank, int *coords) {
  if(_tmgr == NULL) { TopoManager_reset(); }
  int t;
#if CMK_BLUEGENEQ
  _tmgr->rankToCoordinates(CmiNodeFirst(rank),coords[0],coords[1],coords[2],coords[3],coords[4],t);
#else
  _tmgr->rankToCoordinates(CmiNodeFirst(rank),coords[0],coords[1],coords[2],t);
#endif
}

extern "C" void TopoManager_getPeCoordinates(int rank, int *coords) {
  if(_tmgr == NULL) { TopoManager_reset(); }
#if CMK_BLUEGENEQ
  _tmgr->rankToCoordinates(rank,coords[0],coords[1],coords[2],coords[3],coords[4],coords[5]);
#else
  _tmgr->rankToCoordinates(rank,coords[0],coords[1],coords[2],coords[3]);
#endif
}

void TopoManager_getRanks(int *rank_cnt, int *ranks, int *coords) {
  if(_tmgr == NULL) { TopoManager_reset(); }
  int rank, numRanks = _tmgr->getDimNT()/CmiMyNodeSize();
  *rank_cnt = 0;
  for(int t = 0; t < _tmgr->getDimNT(); t += CmiMyNodeSize()) {
#if CMK_BLUEGENEQ
    rank = _tmgr->coordinatesToRank(coords[0],coords[1],coords[2],coords[3],coords[4],t);
#else
    rank = _tmgr->coordinatesToRank(coords[0],coords[1],coords[2],t);
#endif
    if(rank != -1) {
      ranks[*rank_cnt] = CmiNodeOf(rank);
      *rank_cnt = *rank_cnt + 1;
    }
  }
}

extern "C" void TopoManager_getPeRank(int *rank, int *coords) {
  if(_tmgr == NULL) { TopoManager_reset(); }
#if CMK_BLUEGENEQ
  *rank = _tmgr->coordinatesToRank(coords[0],coords[1],coords[2],coords[3],coords[4],coords[5]);
#else
  *rank = _tmgr->coordinatesToRank(coords[0],coords[1],coords[2],coords[3]);
#endif
}

extern "C" void TopoManager_getHopsBetweenPeRanks(int pe1, int pe2, int *hops) {
  if(_tmgr == NULL) { TopoManager_reset(); }
  *hops = _tmgr->getHopsBetweenRanks(pe1, pe2);
}

extern "C" void TopoManager_createPartitions(int scheme, int numparts, int *nodeMap) {
  if(scheme == 0) {
    if(!CmiMyNodeGlobal()) {
      CmiPrintf("Charm++> Using rank ordered division (scheme 0) for topology aware partitions\n");
    }
    int i;
    for(i = 0; i < CmiNumNodes(); i++) {
      nodeMap[i] = i;
    }
  } else if(scheme == 1) {
    if(!CmiMyNodeGlobal()) {
      CmiPrintf("Charm++> Using planar division (scheme 1) for topology aware partitions\n");
    }
    getPlanarList(nodeMap);
  } else if(scheme == 2) {
    if(!CmiMyNodeGlobal()) {
      CmiPrintf("Charm++> Using hilber curve (scheme 2) for topology aware partitions\n");
    }
    getHilbertList(nodeMap);
  } else if(scheme == 3) {
    if(!CmiMyNodeGlobal()) {
      CmiPrintf("Charm++> Using recursive bisection (scheme 3) for topology aware partitions\n");
    }
    getRecursiveBisectionList(numparts,nodeMap);
  } else {
    CmiAbort("Specified value for topology scheme is not supported\n");
  }
}

#if CMK_BLUEGENEQ

#include "spi/include/kernel/process.h"
#include "spi/include/kernel/location.h"
#include <firmware/include/personality.h>

BGQTorusManager::BGQTorusManager() {
  order[0] = 5;
  order[1] = 4;
  order[2] = 3;
  order[3] = 2;
  order[4] = 1;
  order[5] = 0;

  int numPes = CmiNumPes();
  procsPerNode = Kernel_ProcessCount();
  thdsPerProc = CmiMyNodeSize();
  hw_NT = procsPerNode*thdsPerProc;

  if(CmiNumPartitions() > 1) {
    dimA = rn_NA = numPes/hw_NT;
    dimB = dimC = dimD = dimE = 1;
    rn_NB = rn_NC = rn_ND = rn_NE = 1;
    torus[0] = torus[1] = torus[2] = torus[3] = torus[4] = 0;
    dims[0] = rn_NA;
    dims[1] = rn_NB;
    dims[2] = rn_NC;
    dims[3] = rn_ND;
    dims[4] = rn_NE;
    dims[5] = hw_NT;
    return;
  }

  Personality_t pers;
  Kernel_GetPersonality(&pers, sizeof(pers));

  hw_NA = pers.Network_Config.Anodes;
  hw_NB = pers.Network_Config.Bnodes;
  hw_NC = pers.Network_Config.Cnodes;
  hw_ND = pers.Network_Config.Dnodes;
  hw_NE = pers.Network_Config.Enodes;

  mapping = getenv("RANK_ORDER");
  if(mapping != NULL) {
    sscanf(mapping,"%d %d %d %d %d %d",&order[5],&order[4],&order[3],&order[2],&order[1],&order[0]);
  }
  //printf("Mapping %d %d %d %d %d %d\n",order[0],order[1],order[2],order[3],order[4],order[5]);

  rn_NA = hw_NA;
  rn_NB = hw_NB;
  rn_NC = hw_NC;
  rn_ND = hw_ND;
  rn_NE = hw_NE;

  int max_t = 0;
  if(rn_NA * rn_NB * rn_NC * rn_ND * rn_NE != numPes/hw_NT) {
    rn_NA = rn_NB = rn_NC = rn_ND =rn_NE =0;
    int rn_NT=0;
    int min_a, min_b, min_c, min_d, min_e, min_t;
    min_a = min_b = min_c = min_d = min_e = min_t = (~(-1));
    int tmp_t, tmp_a, tmp_b, tmp_c, tmp_d, tmp_e;
    uint64_t numentries;
    BG_CoordinateMapping_t *coord;

    int nranks=numPes/thdsPerProc;
    coord = (BG_CoordinateMapping_t *) malloc(sizeof(BG_CoordinateMapping_t)*nranks);
    Kernel_RanksToCoords(sizeof(BG_CoordinateMapping_t)*nranks, coord, &numentries);

    for(int c = 0; c < nranks; c++) {
      tmp_a = coord[c].a;
      tmp_b = coord[c].b;
      tmp_c = coord[c].c;
      tmp_d = coord[c].d;
      tmp_e = coord[c].e;
      tmp_t = coord[c].t;

      if(tmp_a > rn_NA) rn_NA = tmp_a;
      if(tmp_a < min_a) min_a = tmp_a;
      if(tmp_b > rn_NB) rn_NB = tmp_b;
      if(tmp_b < min_b) min_b = tmp_b;
      if(tmp_c > rn_NC) rn_NC = tmp_c;
      if(tmp_c < min_c) min_c = tmp_c;
      if(tmp_d > rn_ND) rn_ND = tmp_d;
      if(tmp_d < min_d) min_d = tmp_d;
      if(tmp_e > rn_NE) rn_NE = tmp_e;
      if(tmp_e < min_e) min_e = tmp_e;
      if(tmp_t > rn_NT) rn_NT = tmp_t;
      if(tmp_t < min_t) min_t = tmp_t;
    }
    rn_NA = rn_NA - min_a + 1;
    rn_NB = rn_NB - min_b + 1;
    rn_NC = rn_NC - min_c + 1;
    rn_ND = rn_ND - min_d + 1;
    rn_NE = rn_NE - min_e + 1;
    procsPerNode = rn_NT - min_t + 1;
    hw_NT = procsPerNode * thdsPerProc;
  }

  dimA = rn_NA;
  dimB = rn_NB;
  dimC = rn_NC;
  dimD = rn_ND;
  dimE = rn_NE;
  dimA = dimA * hw_NT;	// assuming TABCDE

  dims[0] = rn_NA;
  dims[1] = rn_NB;
  dims[2] = rn_NC;
  dims[3] = rn_ND;
  dims[4] = rn_NE;
  dims[5] = hw_NT;

  torus[0] = ((rn_NA % 4) == 0)? true:false;
  torus[1] = ((rn_NB % 4) == 0)? true:false;
  torus[2] = ((rn_NC % 4) == 0)? true:false;
  torus[3] = ((rn_ND % 4) == 0)? true:false;
  torus[4] = true;

}

#endif
