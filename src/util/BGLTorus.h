/** \file BGLTorus.h
 *  Author: Abhinav S Bhatele
 *  Date created: June 28th, 2007  
 *  The previous file bgltorus.h was cleaned up.
 */

#ifndef _BGL_TORUS_H_
#define _BGL_TORUS_H_

#include <string.h>
#include "converse.h"

#if CMK_BLUEGENEL

#include <bglpersonality.h>

extern "C" int rts_get_personality(struct BGLPersonality *dst, unsigned size);
extern "C" int rts_coordinatesForRank(unsigned logicalRank, unsigned *x,
                                  unsigned *y, unsigned *z, unsigned *t);

class BGLTorusManager {
  private:
    BGLPersonality bgl_p;
    int dimX;	// dimension of the allocation in X (no. of processors)
    int dimY;	// dimension of the allocation in Y (no. of processors)
    int dimZ;	// dimension of the allocation in Z (no. of processors)
    int dimNX;	// dimension of the allocation in X (no. of nodes)
    int dimNY;	// dimension of the allocation in Y (no. of nodes)
    int dimNZ;	// dimension of the allocation in Z (no. of nodes)
    int dimNT;  // dimension of the allocation in T (no. of processors per node)
   
    int torus[4];
    int procsPerNode;
    char *mapping;

  public:
    BGLTorusManager() {
      if(CmiNumPartitions() > 1) {
        dimNX = dimX = CmiNumPes();
        dimNY = dimNZ = dimNT = 1;
        dimY = dimZ = 1;
        procsPerNode = 1;
        torus[0] = torus[1] = torus[2] = torus[3] = 0;
        return;
      }
      int size = sizeof(BGLPersonality);
      int i = rts_get_personality(&bgl_p, size);

      dimNX = bgl_p.xSize;
      dimNY = bgl_p.ySize;
      dimNZ = bgl_p.zSize;
   
      if(bgl_p.opFlags & BGLPERSONALITY_OPFLAGS_VIRTUALNM)
        dimNT = 2;
      else
	dimNT = 1;

      // If we are using lesser no. of processors than the total partition
      // we allocated, then we need to do some arithmetic:

      int numPes = CmiNumPes();
      int numNodes = numPes;
      if(dimNT==2) numNodes = numPes / 2;

      int max_t = 0;
      if(dimNX * dimNY * dimNZ != numNodes) {
        dimNX = dimNY = dimNZ = 0;
        int min_x, min_y, min_z;
        min_x = min_y = min_z = numPes;
        unsigned int tmp_t, tmp_x, tmp_y, tmp_z;      
        for(int c = 0; c < numPes; c++) {
	  rts_coordinatesForRank(c, &tmp_x, &tmp_y, &tmp_z, &tmp_t);

	  if(tmp_x > dimNX) dimNX = tmp_x;
          if(tmp_x < min_x) min_x = tmp_x;
	  if(tmp_y > dimNY) dimNY = tmp_y;
          if(tmp_y < min_y) min_y = tmp_y;
	  if(tmp_z > dimNZ) dimNZ = tmp_z;
          if(tmp_z < min_z) min_z = tmp_z;

	  if(tmp_t > max_t) max_t = tmp_t;
        }
	 
	dimNX = dimNX - min_x + 1;
	dimNY = dimNY - min_y + 1;
	dimNZ = dimNZ - min_z + 1;
      }

      dimX = dimNX;
      dimY = dimNY;
      dimZ = dimNZ;

      if(dimX * dimY * dimZ != numNodes) {
        dimX = dimX * (max_t + 1);	// assuming TXYZ
        procsPerNode = max_t;
      }
      else if(dimNT == 2) {
	dimX = dimX * dimNT;		// assuming TXYZ
	procsPerNode = dimNT;
      }

      torus[0] = bgl_p.isTorusX();
      torus[1] = bgl_p.isTorusY();
      torus[2] = bgl_p.isTorusZ();
      torus[3] = 0;
      
      mapping = getenv("BGLMPI_MAPPING");
    }

    ~BGLTorusManager() { 
     }

    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }

    inline int getDimNX() { return dimNX; }
    inline int getDimNY() { return dimNY; }
    inline int getDimNZ() { return dimNZ; }
    inline int getDimNT() { return dimNT; }

    inline int getProcsPerNode() { return procsPerNode; }

    inline int* isTorus() { return torus; }

    inline void rankToCoordinates(int pe, int &x, int &y, int &z) {
      x = pe % dimX;
      y = (pe % (dimX*dimY)) / dimX;
      z = pe / (dimX*dimY);
    }

    inline void rankToCoordinates(int pe, int &x, int &y, int &z, int &t) {
      if(mapping==NULL || (mapping!=NULL && mapping[0]=='X')) {
        x = pe % dimNX;
        y = (pe % (dimNX*dimNY)) / dimNX;
        z = (pe % (dimNX*dimNY*dimNZ)) / (dimNX*dimNY);
        t = pe / (dimNX*dimNY*dimNZ);
      } else {
        t = pe % dimNT;
        x = (pe % (dimNT*dimNX)) / dimNT;
        y = (pe % (dimNT*dimNX*dimNY)) / (dimNT*dimNX);
        z = pe / (dimNT*dimNX*dimNY);
      }
    }

    inline int coordinatesToRank(int x, int y, int z) {
      return x + (y + z*dimY) * dimX;
    }

    inline int coordinatesToRank(int x, int y, int z, int t) {
      if(mapping==NULL || (mapping!=NULL && mapping[0]=='X'))
        return x + (y + (z + t*dimNZ) * dimNY) * dimNX;
      else
        return t + (x + (y + z*dimNY) * dimNX) * dimNT;
    }

    inline int getNodeID(int pe) {
      int t, x, y, z;
      t = pe % dimNT;
      x = (pe % (dimNT*dimNX)) / dimNT;
      y = (pe % (dimNT*dimNX*dimNY)) / (dimNT*dimNX);
      z = pe / (dimNT*dimNX*dimNY);
      return x + y*dimNX + z*dimNX*dimNY;
    }
};

#endif // CMK_BLUEGENEL
#endif //_BGL_TORUS_H_
