
#ifndef _BGL_TOROUS_H__
#define _BGL_TOROUS_H__

#if CMK_VERSION_BLUEGENE

#include "converse.h"

#include <bglpersonality.h>
#include <rts.h>
#include <stdlib.h>

class BGLTorousManager;

class BGLTorousManager {

  BGLPersonality my_bg;
  int xsize, ysize, zsize; 

 public:
  
  //Assumes an TXYZ Mapping
  BGLTorousManager() {
    int size = sizeof(BGLPersonality);
    rts_get_personality(&my_bg, size);

    xsize = my_bg.xSize;
    ysize = my_bg.ySize;
    zsize = my_bg.zSize;  

    if(my_bg.opFlags & BGLPERSONALITY_OPFLAGS_VIRTUALNM)
      xsize *= 2;

    //CmiPrintf("BGL Torous Constructor %d,%d,%d\n", xsize, ysize, zsize);
  }

  inline int getXSize() { return xsize;}
  inline int getYSize() { return ysize;}
  inline int getZSize() { return zsize;}

  static inline BGLTorousManager *getObject();

  inline void getMyCoordinates(int &X, int &Y, int &Z) {
    /*
      X = my_bg.xCoord;
      Y = my_bg.yCoord;
      Z = my_bg.zCoord;
    */

    X = CmiMyPe() % xsize;
    Y = (CmiMyPe() % (xsize * ysize)) / xsize;
    Z = CmiMyPe() / (xsize * ysize);
  } 
  
  static int isNeighborByCoord(int x1, int y1, int z1, int x2, int y2, int z2, int dist) {
    if(abs(x1 - x2) <= dist && 
       abs(y1 - y2) <= dist && 
       abs(z1 - z2) <= dist)
      return 1;
    
    return 0;
  }

  inline void getCoordinatesByRank(int pe, int &x, int &y, int &z) {

    x = pe % xsize;
    y = (pe % (xsize * ysize)) / xsize;
    z = pe / (xsize * ysize);
  }
  
  //Assumes a TXYZ mapping
  inline int isMyNeighbor(int pe, int dist=2) {
    int x,y,z;
    int pe_x, pe_y, pe_z;
    
    getMyCoordinates(x,y,z);
    getCoordinatesByRank(pe, pe_x, pe_y, pe_z);
    
    return isNeighborByCoord(x, y, z, pe_x, pe_y, pe_z, dist); 
  }

  inline int neighbors(int pe1, int pe2, int dist=2) {
    int pe1_x, pe1_y, pe1_z;
    int pe2_x, pe2_y, pe2_z;
    
    getCoordinatesByRank(pe1, pe1_x, pe1_y, pe1_z);
    getCoordinatesByRank(pe2, pe2_x, pe2_y, pe2_z);
    
    return isNeighborByCoord(pe1_x, pe1_y, pe1_z, pe2_x, pe2_y, pe2_z, dist); 
  }

  inline int isNeighborOfBoth(int pe1, int pe2, int pe3, int dist=2) {
    int pe1_x, pe1_y, pe1_z;
    int pe2_x, pe2_y, pe2_z;
    int pe3_x, pe3_y, pe3_z;
    
    getCoordinatesByRank(pe1, pe1_x, pe1_y, pe1_z);
    getCoordinatesByRank(pe2, pe2_x, pe2_y, pe2_z);
    getCoordinatesByRank(pe3, pe3_x, pe3_y, pe3_z);
    
    return isNeighborByCoord(pe1_x, pe1_y, pe1_z, (pe2_x+pe3_x)/2, 
			     (pe2_y+pe3_y)/2, (pe2_z+pe3_z)/2, dist); 
  }

  inline int coords2rank(int x, int y, int z) {
    return x + y * xsize + z * xsize * ysize;
  }
};

CpvExtern(BGLTorousManager *, tmanager); 

BGLTorousManager *BGLTorousManager::getObject() {
  return CpvAccess(tmanager);
}  

#endif
#endif
