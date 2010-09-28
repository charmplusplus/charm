/** \file XTTorus.h
 *  Author: Abhinav S Bhatele
 *  Date created: August 19th, 2008
 *  
 */

#ifndef _XT_TORUS_H_
#define _XT_TORUS_H_

#include "converse.h"
#include <stdlib.h>
#include <stdio.h>

#if XT4_TOPOLOGY || XT5_TOPOLOGY

// XDIM, YDIM, ZDIM and MAXNID depend on a specific Cray installation.
// Please do NOT expect things to work if you use this code on a new
// Cray machine.

#if XT4_TOPOLOGY
#define MAXNID 14000
#define XDIM 21
#define YDIM 16
#define ZDIM 24
#define TDIM 4

#elif XT5_TOPOLOGY
#define MAXNID 17000
#define XDIM 25
#define YDIM 32
#define ZDIM 24
#define TDIM 12

#endif

extern "C" int *pid2nid;
extern "C" int nid2pid[MAXNID][TDIM];
extern "C" int pidtonid(int numpes);
extern "C" int getMeshCoord(int nid, int *x, int *y, int *z);

struct loc {
  int x;
  int y;
  int z;
  int t;
};

class XTTorusManager {
  private:
    int dimX;	// dimension of the allocation in X (processors)
    int dimY;	// dimension of the allocation in Y (processors)
    int dimZ;	// dimension of the allocation in Z (processors)
    int dimNX;	// dimension of the allocation in X (nodes)
    int dimNY;	// dimension of the allocation in Y (nodes)
    int dimNZ;	// dimension of the allocation in Z (nodes)
    int dimNT;  // number of processors per node (2 for XT3)

    int torus[4];
    int procsPerNode;   // number of cores per node
    
    int coords2pid[XDIM][YDIM][ZDIM][TDIM];     // coordinates to rank
    struct loc *pid2coords;                     // rank to coordinates
    struct loc origin;

  public:
    XTTorusManager() {
      int nid = 0, oldnid = -1, lx, ly, lz;
      int i, j, k, l;
      int minX=XDIM, minY=YDIM, minZ=ZDIM, minT=0, maxX=0, maxY=0, maxZ=0;
      pid2coords = (struct loc*)malloc(sizeof(struct loc) * CmiNumPes());

      // fill the nid2pid and pid2nid data structures
      pidtonid(CmiNumPes());

      for(i=0; i<XDIM; i++)
	for(j=0; j<YDIM; j++)
	  for(k=0; k<ZDIM; k++)
	    for(l=0; l<TDIM; l++)
	      coords2pid[i][j][k][l] = -1;

      dimNT = 1;			// assume SN mode first
      // now fill the coords2pid and pid2coords data structures
      for(int i=0; i<CmiNumPes(); i++)
      {
        nid = pid2nid[i];
	if (nid != oldnid)
	  getMeshCoord(nid, &lx, &ly, &lz);
	oldnid = nid;

        pid2coords[i].x = lx;      
        pid2coords[i].y = ly;
        pid2coords[i].z = lz;

	l = 0;
	while(coords2pid[lx][ly][lz][l] != -1)
	  l++;
	coords2pid[lx][ly][lz][l] = i;
	pid2coords[i].t = l;

        if (lx<minX) minX = lx; if (lx>maxX) maxX = lx;
        if (ly<minY) minY = ly; if (ly>maxY) maxY = ly;
        if (lz<minZ) minZ = lz; if (lz>maxZ) maxZ = lz;
      }

      // set the origin as the element on the lower end of the torus
      origin.x =  minX;
      origin.y =  minY;
      origin.z =  minZ;
      origin.t =  minT;
      
      // assuming a contiguous allocation find the dimensions of 
      // the torus
      dimNX = maxX - minX + 1;
      dimNY = maxY - minY + 1;
      dimNZ = maxZ - minZ + 1;
      procsPerNode = dimNT;
      dimX = dimNX * dimNT;
      dimY = dimNY;
      dimZ = dimNZ;

      for(l=0; l<TDIM; l++) {
	if(coords2pid[minX][minY][minZ][l] == -1)
	  break;
      }
      dimNT = l;

      // we get a torus only if the size of the dimension is the biggest
      torus[0] = 0;		// Jaguar is a mesh in X dimension always
      torus[1] = (dimNY == YDIM) ? 1 : 0;
      torus[2] = (dimNZ == ZDIM) ? 1 : 0;
      torus[3] = 0;
    }

    ~XTTorusManager() { }

    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }

    inline int getDimNX() { return dimNX; }
    inline int getDimNY() { return dimNY; }
    inline int getDimNZ() { return dimNZ; }
    inline int getDimNT() { return dimNT; }

    inline int getProcsPerNode() { return procsPerNode; }
    
    inline int* isTorus() { return torus; }

    inline void rankToCoordinates(int pe, int &x, int &y, int &z, int &t) {
      x = pid2coords[pe].x - origin.x; 
      y = pid2coords[pe].y - origin.y; 
      z = pid2coords[pe].z - origin.z; 
      t = pid2coords[pe].t - origin.t; 
    }

    inline void realRankToCoordinates(int pe, int &x, int &y, int &z, int &t) {
      x = pid2coords[pe].x; 
      y = pid2coords[pe].y; 
      z = pid2coords[pe].z; 
      t = pid2coords[pe].t; 
    }

    inline int coordinatesToRank(int x, int y, int z, int t) {
      return coords2pid[x+origin.x][y+origin.y][z+origin.z][t+origin.t];
    }
};

#endif // XT4_TOPOLOGY || XT5_TOPOLOGY
#endif //_XT_TORUS_H_
