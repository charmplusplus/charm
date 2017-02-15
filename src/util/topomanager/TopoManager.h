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

#if CMK_BLUEGENEQ
#include "BGQTorus.h"
#elif XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
#include "XTTorus.h"
#endif

#if CMK_BIGSIM_CHARM
#include "blue.h"
#endif

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
#if CMK_BLUEGENEQ
    inline int getDimNA() const { return dimNA; }
    inline int getDimNB() const { return dimNB; }
    inline int getDimNC() const { return dimNC; }
    inline int getDimND() const { return dimND; }
    inline int getDimNE() const { return dimNE; }
#endif
    inline int getDimNT() const { return dimNT; }

    inline int getProcsPerNode() const { return procsPerNode; }

    inline bool hasMultipleProcsPerNode() const { return (procsPerNode > 1); }
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
#if CMK_BLUEGENEQ
    inline int absA(int a) const {
      int pa = abs(a);
      int sa = dimNA - pa;
      CmiAssert(sa>=0);
      if(torusA)
        return ((pa>sa) ? sa : pa);
      else
        return pa;
    }

    inline int absB(int b) const {
      int pb = abs(b);
      int sb = dimNB - pb;
      CmiAssert(sb>=0);
      if(torusB)
        return ((pb>sb) ? sb : pb);
      else
        return pb;
    }

    inline int absC(int c) const {
      int pc = abs(c);
      int sc = dimNC - pc;
      CmiAssert(sc>=0);
      if(torusC)
        return ((pc>sc) ? sc : pc);
      else
        return pc;
    }

    inline int absD(int d) const {
      int pd = abs(d);
      int sd = dimND - pd;
      CmiAssert(sd>=0);
      if(torusD)
        return ((pd>sd) ? sd : pd);
      else
        return pd;
    }

    inline int absE(int e) const {
      int pe = abs(e);
      int se = dimNE - pe;
      CmiAssert(se>=0);
        return ((pe>se) ? se : pe);
    }
#endif
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
#if CMK_BLUEGENEQ
    int dimNA, dimNB, dimNC, dimND, dimNE;
    int torusA, torusB, torusC, torusD, torusE;
#endif
    int procsPerNode;
#if CMK_BLUEGENEQ
    BGQTorusManager bgqtm;
#elif XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY
    XTTorusManager xttm;
#endif
};
#endif
#endif //_TOPO_MANAGER_H_
