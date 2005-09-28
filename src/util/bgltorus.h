
#ifndef _BGL_TOROUS_H__
#define _BGL_TOROUS_H__

#if CMK_VERSION_BLUEGENE

#include "converse.h"

#include <bglpersonality.h>
#include <rts.h>
#include <stdlib.h>

class BGLTorusManager;

class BGLTorusManager {

  BGLPersonality my_bg;
  int xsize, ysize, zsize;   //size in processors
  int nxsize, nysize, nzsize;  //size in nodes
  int isVN;

 public:
  
  //Assumes an TXYZ Mapping
  BGLTorusManager() {
    int size = sizeof(BGLPersonality);
    rts_get_personality(&my_bg, size);

    isVN = 0;

    xsize = my_bg.xSize;
    ysize = my_bg.ySize;
    zsize = my_bg.zSize;  

    nxsize = xsize;
    nysize = ysize;
    nzsize = zsize;
    
    if(my_bg.opFlags & BGLPERSONALITY_OPFLAGS_VIRTUALNM) {
      isVN = 1;
      xsize *= 2;
    }

    //CmiPrintf("BGL Torus Constructor %d,%d,%d\n", xsize, ysize, zsize);
  }

  inline int getXSize() { return xsize;}
  inline int getYSize() { return ysize;}
  inline int getZSize() { return zsize;}

  inline int getXNodeSize() { return nxsize;}
  inline int getYNodeSize() { return nysize;}
  inline int getZNodeSize() { return nzsize;}

  inline int isVnodeMode() { return isVN;}

  static inline BGLTorusManager *getObject();

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

  inline int getHopsToCoordinates(int x1, int y1, int z1) {
    int x,y,z;
    getMyCoordinates(x,y,z);
    return (abs(x1-x)+abs(y1-y)+abs(z1-z));
  }
  inline int getHopsToRank(int pe){
    int pe_x, pe_y, pe_z;
    getCoordinatesByRank(pe, pe_x, pe_y, pe_z);
    return getHopsToCoordinates(pe_x,pe_y,pe_z);
  }

  /* return my cone number 0-5, self=-1 */
  inline int getConeNumberForRank(int pe){
    int x,y,z,x1,y1,z1;
    int dx,dy,dz;
    getMyCoordinates(x,y,z);
    getCoordinatesByRank(pe, x1, y1, z1);
    dx=x1-x;
    dy=y1-y;
    dz=z1-z;
    if(dx==0 && dy==0 && dz==0) return -1;
    if(abs(dx)>=abs(dy) && abs(dx)>=abs(dz)) return (dx>0)?0:1;
    if(abs(dy)> abs(dx) && abs(dy)>=abs(dz)) return (dy>0)?2:3;
    if(abs(dz)> abs(dx) && abs(dz)> abs(dy)) return (dz>0)?4:5;
    CkAssert(-1);
  }

  /* sort pes by ascending order of hops to me */
  inline void sortRanksByHops(int *pes, int n){
    int i,j,tmp;
    for (i=0; i<n-1; i++)
      for (j=0; j<n-1-i; j++)
        if (getHopsToRank(pes[j+1]) < getHopsToRank(pes[j])){
          tmp=pes[j+1]; pes[j+1]=pes[j]; pes[j]=tmp;
        }
  }

  inline int getHopsBetweenRanks(int pe, int pe1){
    int x,y,z,x1,y1,z1;
    getCoordinatesByRank(pe,x,y,z);
    getCoordinatesByRank(pe1,x1,y1,z1);
    return (abs(x1-x)+abs(y1-y)+abs(z1-z));
  }
  inline int pickClosestRank(int mype, int *pes, int n){
    int minHops=getHopsBetweenRanks(mype,pes[0]);
    int minIdx=0;
    int nowHops; 
    for(int i=1;i<n;i++){
      nowHops = getHopsBetweenRanks(mype,pes[i]);
      if(nowHops<minHops){
        minHops=nowHops; minIdx=i;
      }
    }
    return minIdx;
  }
};

CpvExtern(BGLTorusManager *, tmanager); 

BGLTorusManager *BGLTorusManager::getObject() {
  if(CpvAccess(tmanager) == NULL)
    CpvAccess(tmanager) = new BGLTorusManager();
  return CpvAccess(tmanager);
}  

#endif
#endif
