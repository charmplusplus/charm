
#ifndef _BGL_TOROUS_H__
#define _BGL_TOROUS_H__

#if CMK_VERSION_BLUEGENE

#include "converse.h"

#if 0
#include <bgml.h>
#endif

#include <bglpersonality.h>

extern "C" int rts_get_personality(struct BGLPersonality *dst, unsigned size);
extern "C" int rts_coordinatesForRank(unsigned logicalRank, unsigned *x, 
				  unsigned *y, unsigned *z, unsigned *t);

/*#include <rts.h>*/
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

    unsigned numnodes = CmiNumPes();
    if(my_bg.opFlags & BGLPERSONALITY_OPFLAGS_VIRTUALNM) 
      numnodes = CmiNumPes() / 2;
    
    unsigned max_t = 0;
    if(xsize * ysize * zsize != numnodes) {
      xsize = ysize = zsize = 0;
      
      int mx,my,mz;        //min values for x,y.z
      mx = my = mz = CmiNumPes();
      unsigned tmpx, tmpy, tmpz, tmpt;
      for(int count = 0; count < CmiNumPes(); count ++) {
	rts_coordinatesForRank(count, &tmpx, &tmpy, &tmpz, &tmpt);
	
	if(tmpx > xsize)
	  xsize = tmpx;
	
	if(tmpx < mx)
	  mx = tmpx;
	
	if(tmpy > ysize)
	  ysize = tmpy;
	
	if(tmpy < my)
	  my = tmpy;
	
	if(tmpz > zsize)
	  zsize = tmpz;
	
	if(tmpz < mz)
	  mz = tmpz;

	if(tmpt > max_t)
	  max_t = tmpt;
      }

      xsize = xsize - mx + 1;
      ysize = ysize - my + 1;
      zsize = zsize - mz + 1;
    }
    
    nxsize = xsize;
    nysize = ysize;
    nzsize = zsize;
    
    if(xsize * ysize * zsize != numnodes) {
      xsize *= max_t + 1;   //Assuming TXYZ
      isVN = max_t;
    }
    else if(my_bg.opFlags & BGLPERSONALITY_OPFLAGS_VIRTUALNM) {
      isVN = 1;
      xsize *= 2;      //Assuming TXYZ
    }    
    
    //CmiPrintf("BGL Torus Constructor %d,%d,%d  nodes %d,%d,%d\n", xsize, ysize, zsize, nxsize, nysize, nzsize);
  }
  
  inline int getXSize() { return xsize;}
  inline int getYSize() { return ysize;}
  inline int getZSize() { return zsize;}

  inline int getXNodeSize() { return nxsize;}
  inline int getYNodeSize() { return nysize;}
  inline int getZNodeSize() { return nzsize;}

  inline int isVnodeMode() { return isVN;}

  static inline BGLTorusManager *getObject();

  inline void getCoordinatesByRank(int pe, int &x, int &y, int &z) {
    //Assumed TXYZ
    x = pe % xsize;
    y = (pe % (xsize * ysize)) / xsize;
    z = pe / (xsize * ysize);    
  }
  
  inline void getMyCoordinates(int &X, int &Y, int &Z) {    
    getCoordinatesByRank (CmiMyPe(), X, Y, Z);
  } 
  
  inline int absx(int n){
    int an = abs(n);
    int aan = xsize - an;
    CmiAssert(aan>=0);
    return ((an>aan)?aan:an);
  }
  inline int absy(int n){
    int an = abs(n);
    int aan = ysize - an;
    CmiAssert(aan>=0);
    return ((an>aan)?aan:an);
  }

  inline int absz(int n){
    int an = abs(n);
    int aan = zsize - an;
    
    if (aan < 0)
      printf ("aan = %d, an = %d, nz = %d \n", an, aan, zsize);

    CmiAssert(aan>=0);
    return ((an>aan)?aan:an);
  }
  
  int isNeighborByCoord(int x1, int y1, int z1, int x2, int y2, int z2, int dist) {
#if 0
    if(absx(x1 - x2) <= dist && 
       absy(y1 - y2) <= dist && 
       absz(z1 - z2) <= dist)
      return 1;
#endif

    if( (absx(x1 - x2) + absy(y1 - y2) + absz(z1 - z2)) <= dist )
      return 1;

    return 0;
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
    return (absx(x1-x)+absy(y1-y)+absz(z1-z));
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
    if(absx(dx)>=absy(dy) && absx(dx)>=absz(dz)) return (dx>0)?0:1;
    if(absy(dy)> absx(dx) && absy(dy)>=absz(dz)) return (dy>0)?2:3;
    if(absz(dz)> absx(dx) && absz(dz)> absy(dy)) return (dz>0)?4:5;
    return -1;
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
    return (absx(x1-x)+absy(y1-y)+absz(z1-z));
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
  /*  
  inline void sortIndexByHops(int pe, int *pes, int *idx, int n){
    int minHops = getHopsBetweenRanks(pe,pes[0]);
    int minIdx = 0;
    int nowHops, tmp; 
    for(int i=0;i<n;i++){
      idx[i] = i;
    }
    for (int i=0; i<n-1; i++)
      for (int j=0; j<n-1-i; j++)
        if (getHopsBetweenRanks(pe, pes[idx[j+1]]) < getHopsBetweenRanks(pe, pes[idx[j]])){
          tmp=idx[j+1]; idx[j+1]=idx[j]; idx[j]=tmp;
        }
  }
};
  */
  inline void sortIndexByHops(int pe, int *pes, int *idx, int n)
    {
	int minHops = getHopsBetweenRanks(pe,pes[0]);
	int minIdx = 0;
	int nowHops, tmp;
	for(int i=0;i<n;i++){
	  idx[i] = i;
	}
	quicksort(pe, pes, idx, 0, n-1);
      }
    void quicksort(int pe, int *pes, int *arr, int left, int right)
      {
	if(left<right)
	  {
	    int split = partition(pe, pes, arr, left, right);
	    quicksort(pe, pes, arr, left, split);
	    quicksort(pe, pes, arr, split+1, right);
	  }
      }

    int partition(int pe, int *pes, int *idx, int left, int right)
      {
	int val = getHopsBetweenRanks(pe, pes[idx[(left+right)/2]]);
	int lm = left-1;
	int rm = right+1;
	for(;;)
	  {
	    do
	      rm--;
	    while(getHopsBetweenRanks(pe, pes[idx[rm]]) > val);
	    do
	      lm++;
	    while(getHopsBetweenRanks(pe, pes[idx[lm]]) < val);
	    if(lm < rm)
	      {
		int tmp = idx[rm];
		idx[rm] = idx[lm];
		idx[lm] = tmp;
	      }
	    else
	      return rm;
	  }
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
