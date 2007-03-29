/** \file TopoManager.h
 *  Author: Abhinav S Bhatele
 *  Date Created: March 19th, 2007
 *
 *  This would be the top level interface for all topology managers we
 *  will write for different machines (cray, bg/l ... for tori, meshes ...)
 *  Current plan is to have functionality for Blue Gene/L, Cray XT3,
 *  BigSim and non-topo machines.
 */

#ifndef _TOPO_MANAGER_H_
#define _TOPO_MANAGER_H_

#ifdef CMK_VERSION_BLUEGENE
#include "bgltorus.h"
#endif
#ifdef CMK_XT3
#include "CrayTorus.h"
#endif

class TopoManager {
  private:
    int dimX;	// dimension of the allocation in X (processors)
    int dimY;	// dimension of the allocation in Y (processors)
    int dimZ;	// dimension of the allocation in Z (processors)
    int dimNX;	// dimension of the allocation in X (nodes)
    int dimNY;	// dimension of the allocation in Y (nodes)
    int dimNZ;	// dimension of the allocation in Z (nodes)
    int procsPerNode;

#ifdef CMK_VERSION_BLUEGENE
  BGLTorusManager *bgltm;
#elif CMK_XT3
  CrayTorusManager *crtm;
#endif

  public:
    TopoManager();
    ~TopoManager();

    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }

    inline int getDimNX() { return dimNX; }
    inline int getDimNY() { return dimNY; }
    inline int getDimNZ() { return dimNZ; }

    int hasMultipleProcsPerNode();
 
    void rankToCoordinates(int pe, int &x, int &y, int &z);
    int coordinatesToRank(int x, int y, int z);
    int getHopsBetweenRanks(int pe1, int pe2);
    void sortRanksByHops(int pe, int *pes, int *idx, int n); 

  private:
    void quicksort(int pe, int *pes, int *arr, int left, int right);
    int partition(int pe, int *pes, int *idx, int left, int right);

};

#endif //_TOPO_MANAGER_H_
