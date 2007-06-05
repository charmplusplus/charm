/** \file BGPTorus.h
 *  Author: Abhinav S Bhatele
 *  Date created: May 21st, 2007  
 *  
 */

#ifndef _BGP_TORUS_H_
#define _BGP_TORUS_H_

#include "converse.h"

#if CMK_BLUEGENEP

#include "dcmf.h"

class BGPTorusManager {
  private:
    DCMF_Hardware_t bgp_hwt;
    int dimX;   // dimension of the allocation in X (processors)
    int dimY;   // dimension of the allocation in Y (processors)
    int dimZ;   // dimension of the allocation in Z (processors)
    int dimNX;  // dimension of the allocation in X (nodes)
    int dimNY;  // dimension of the allocation in Y (nodes)
    int dimNZ;  // dimension of the allocation in Z (nodes)
    int torus[4];
    int procsPerNode;

  public:
    BGPTorusManager() {
      DCMF_Hardware(&bgp_hwt);
      dimNX = bgp_hwt.xSize;
      dimNY = bgp_hwt.ySize;
      dimNZ = bgp_hwt.zSize;
 
      dimX = dimNX;
      dimY = dimNY;
      dimZ = dimNZ;
 
      if(bgp_hwt.tSize != 1) {
        dimX = dimX * bgp_hwt.tSize;
      }
      procsPerNode = bgp_hwt.tSize;

      torus[0] = bgp_hwt.xTorus;
      torus[1] = bgp_hwt.yTorus;
      torus[2] = bgp_hwt.zTorus;
      torus[3] = bgp_hwt.tTorus;
    }

    ~BGPTorusManager() { }

    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }

    inline int getDimNX() { return dimNX; }
    inline int getDimNY() { return dimNY; }
    inline int getDimNZ() { return dimNZ; }

    inline int getProcsPerNode() { return procsPerNode; }

    inline int* isTorus() { return torus; }

    inline void rankToCoordinates(int pe, int &x, int &y, int &z) {
      x = pe % dimX;
      y = (pe % (dimX*dimY)) / dimX;
      z = pe / (dimX*dimY);
    }

    inline int coordinatesToRank(int x, int y, int z) {
      return x + y*dimX + z*dimX*dimY;
    }

};

#endif // CMK_BLUEGENEP
#endif //_BGP_TORUS_H_
