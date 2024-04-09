/** \file TopoManager.h
 *  Author: Abhinav S Bhatele
 *  Date Created: March 19th, 2007
 *
 *  This would be the top level interface for all topology managers we
 *  will write for specific machines.
 *  Currently we have functionality for Cray XT/XE,
 *  and non-topo machines.
 */

#ifndef _TOPO_MANAGER_H_
#define _TOPO_MANAGER_H_

#include "topomanager_config.h"

#ifndef __TPM_STANDALONE__
#include "converse.h"
#else
#include "tpm_standalone.h"
#include <string.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** basic initialization */
#ifndef __TPM_STANDALONE__
void TopoManager_init();
#else
void TopoManager_init(int numpes);
void TopoManager_setNumThreads(int t);  // num threads per process, NOTE that, if used, this must be called before TopoManager_init
#endif
/** redo the initialization */
void TopoManager_reset();
/** free the current occupant */
void TopoManager_free();
/** print allocation */
void TopoManager_printAllocation(FILE *fp);
/** get the number physical dimensions */
void TopoManager_getDimCount(int *ndims);
/** get the length of dimensions, last one is core/process/thread count */
void TopoManager_getDims(int *dims);
/** get coordinates of a logical node */
void TopoManager_getCoordinates(int rank, int *coords);
/** get coordinate of a PE, includes an additional dim */
void TopoManager_getPeCoordinates(int rank, int *coords);
/** get ranks of logical nodes at a coordinate */
void TopoManager_getRanks(int *rank_cnt, int *ranks, int *coords);
/** get rank of PE at a coordinate */
void TopoManager_getPeRank(int *rank, int *coords);
/** get hops betweens Pes */
void TopoManager_getHopsBetweenPeRanks(int pe1, int pe2, int *hops);
#ifndef __TPM_STANDALONE__
/** topoaware partition using scheme s */
void TopoManager_createPartitions(int scheme, int numparts, int *nodeMap);
#endif

#if defined(__cplusplus)
}

#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
#include "XTTorus.h"
#endif


#include <vector>

class TopoManager {
  public:
    TopoManager();
    TopoManager(int NX, int NY, int NZ, int NT);
    ~TopoManager() { }

    /***
     * Access singleton instance of TopoManager
     * NOTE: this should only be called after TopoManager_init() has been called
     * (in Charm++ TopoManager_init() is called during initialization)
     */
    static TopoManager *getTopoManager();

    inline int getDimNX() const { return dimNX; }
    inline int getDimNY() const { return dimNY; }
    inline int getDimNZ() const { return dimNZ; }
    inline int getDimNT() const { return dimNT; }
    inline int getNumDims() const {
      return 3;
    }
    inline int getDimSize(unsigned int i) const {
      CmiAssert(i < 3);
      switch (i) {
        case 0: return getDimNX();
        case 1: return getDimNY();
        case 2: return getDimNZ();
        default: return -1;
      }
    }
    inline bool haveTopologyInfo() const {
#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
      return true;
#else
      return false;
#endif
    }

    inline int getProcsPerNode() const { return procsPerNode; }

    inline bool hasMultipleProcsPerNode() const { return (procsPerNode > 1); }
    void rankToCoordinates(int pe, std::vector<int> &coords) const;
    void rankToCoordinates(int pe, int &x, int &y, int &z, int &t) const;
    void rankToCoordinates(int pe, int &a, int &b, int &c, int &d, int &e, int &t) const;
    /**
     * Return pe at specified coordinates, or -1 if doesn't exist
     */
    int coordinatesToRank(int x, int y, int z, int t) const;
    /**
     * Return pe at specified coordinates, or -1 if doesn't exist
     */
    int coordinatesToRank(int a, int b, int c, int d, int e, int t) const;
    int getHopsBetweenRanks(int pe1, int pe2) const;
    int getHopsBetweenRanks(int *pe1, int pe2) const;
    void sortRanksByHops(int pe, int *pes, int *idx, int n) const;
    void sortRanksByHops(int *pe, int *pes, int *idx, int n) const;
    int pickClosestRank(int mype, int *pes, int n) const;
    int areNeighbors(int pe1, int pe2, int pe3, int distance) const;
    void printAllocation(FILE *fp) const;

    /** The next 5 functions are only there for backward compatibility
    and should not be used */
    inline int getDimX() const { return dimX; }
    inline int getDimY() const { return dimY; }
    inline int getDimZ() const { return dimZ; }
    void rankToCoordinates(int pe, int &x, int &y, int &z) const;
    int coordinatesToRank(int x, int y, int z) const;

    inline int absX(int x) const {
      int px = abs(x);
      int sx = dimNX - px;
      CmiAssert(sx>=0);
      if(torusX)
        return ((px>sx) ? sx : px);
      else
        return px;
    }
    
    inline int absY(int y) const {
      int py = abs(y);
      int sy = dimNY - py;
      CmiAssert(sy>=0);
      if(torusY)
        return ((py>sy) ? sy : py);
      else
        return py;
    }

    inline int absZ(int z) const {
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
#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
    XTTorusManager xttm;
#endif
};
#endif
#endif //_TOPO_MANAGER_H_
