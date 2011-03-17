/** \file TopoManager.h
 *  Author: Abhinav S Bhatele
 *  Date Created: March 19th, 2007
 *
 *  This would be the top level interface for all topology managers we
 *  will write for different machines (Cray, IBM ... for tori, meshes ...)
 *  Current we have functionality for Blue Gene, Cray XT, BigSim and 
 *  non-topo machines.
 *
 */

#ifndef _TOPO_MANAGER_H_
#define _TOPO_MANAGER_H_

#include "charm.h"

#if CMK_BLUEGENEL
#include "BGLTorus.h"
#elif CMK_BLUEGENEP
#include "BGPTorus.h"
#elif XT3_TOPOLOGY
#include "XT3Torus.h"
#elif XT4_TOPOLOGY || XT5_TOPOLOGY
#include "XTTorus.h"
#endif

#if CMK_BLUEGENE_CHARM
#include "blue.h"
#endif

class TopoManager {
  public:
    TopoManager();
    TopoManager(int NX, int NY, int NZ, int NT);
    ~TopoManager() { }

    inline int getDimNX() const { return dimNX; }
    inline int getDimNY() const { return dimNY; }
    inline int getDimNZ() const { return dimNZ; }
    inline int getDimNT() const { return dimNT; }

    inline int getProcsPerNode() const { return procsPerNode; }

    int hasMultipleProcsPerNode() const;
    void rankToCoordinates(int pe, int &x, int &y, int &z, int &t);
    int coordinatesToRank(int x, int y, int z, int t);
    int getHopsBetweenRanks(int pe1, int pe2);
    void sortRanksByHops(int pe, int *pes, int *idx, int n);
    int pickClosestRank(int mype, int *pes, int n);
    int areNeighbors(int pe1, int pe2, int pe3, int distance);

    /** The next 5 functions are only there for backward compatibility
    and should not be used */
    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }
    void rankToCoordinates(int pe, int &x, int &y, int &z);
    int coordinatesToRank(int x, int y, int z);

    inline int absX(int x) {
      int px = abs(x);
      int sx = dimNX - px;
      CmiAssert(sx>=0);
      if(torusX)
        return ((px>sx) ? sx : px);
      else
        return px;
    }
    
    inline int absY(int y) {
      int py = abs(y);
      int sy = dimNY - py;
      CmiAssert(sy>=0);
      if(torusY)
        return ((py>sy) ? sy : py);
      else
        return py;
    }

    inline int absZ(int z) {
      int pz = abs(z);
      int sz = dimNZ - pz;
      CmiAssert(sz>=0);
      if(torusZ)
        return ((pz>sz) ? sz : pz);
      else
        return pz;
    }

  private:
    int dimX;	// dimension of the allocation in X (no. of processors)
    int dimY;	// dimension of the allocation in Y (no. of processors)
    int dimZ;	// dimension of the allocation in Z (no. of processors)
    int dimNX;	// dimension of the allocation in X (no. of nodes)
    int dimNY;	// dimension of the allocation in Y (no. of nodes)
    int dimNZ;	// dimension of the allocation in Z (no. of nodes)
    int dimNT;  // dimension of the allocation in T (no. of processors per node)
    int numPes;
    int torusX, torusY, torusZ, torusT;
    int procsPerNode;

#if CMK_BLUEGENEL
    BGLTorusManager bgltm;
#elif CMK_BLUEGENEP
    BGPTorusManager bgptm;
#elif XT3_TOPOLOGY
    XT3TorusManager xt3tm;
#elif XT4_TOPOLOGY || XT5_TOPOLOGY
    XTTorusManager xttm;
#endif

    void quicksort(int pe, int *pes, int *arr, int left, int right);
    int partition(int pe, int *pes, int *idx, int left, int right);
};

#endif //_TOPO_MANAGER_H_
