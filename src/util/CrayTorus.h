/** \file CrayTorus.h
 *  Author: Abhinav S Bhatele
 *  Date created: March 19th, 2007  
 *  
 *  This file makes use of a static table we obtain from a file available
 *  on Bigben.
 */

#ifndef _CRAY_TORUS_H_
#define _CRAY_TORUS_H_

#include "converse.h"
#include <stdlib.h>
#include <stdio.h>

#if CMK_XT3

#define XDIM 11
#define YDIM 12
#define ZDIM 16
#define TDIM 2
#define MAXNID 2784

extern "C" int *pid2nid;
extern "C" int nid2pid[MAXNID][2];
extern "C" int pidtonid(int numpes);

struct loc {
  int x;
  int y;
  int z;
  int t;
};

class CrayTorusManager {
  private:
    int dimX;	// dimension of the allocation in X (processors)
    int dimY;	// dimension of the allocation in Y (processors)
    int dimZ;	// dimension of the allocation in Z (processors)
    int dimNX;	// dimension of the allocation in X (nodes)
    int dimNY;	// dimension of the allocation in Y (nodes)
    int dimNZ;	// dimension of the allocation in Z (nodes)
    int dimNT;  // number of processors per node (2 for XT3)

    int procsPerNode;
    
    int coords2pid[XDIM][YDIM][ZDIM][TDIM];
    struct loc *pid2coords;

  public:
    CrayTorusManager() {
      // load data from CrayNeighborTable
      FILE *fp = fopen("/usr/users/4/abhatele/work/charm/src/util/CrayNeighbourTable", "r");
      int temp, nid, pid0, pid1, num, lx, ly, lz;
      int minX=XDIM, minY=YDIM, minZ=ZDIM, maxX=0, maxY=0, maxZ=0;
      char header[50];
      pid2coords = (struct loc*)malloc(sizeof(struct loc) * CmiNumPes());
      pidtonid(CmiNumPes());
      
      // skip header
      for(int i=0; i<10;i++)
        temp = fscanf(fp, "%s", header);
        
      // read the lines one at a time and fill the two arrays
      // for pid2coords and coords2pid
      for(int i=0; i<2112;i++)
      {
        temp = fscanf(fp, "%d%d%d%d%d%d%d%d%d%d", &nid, &num, &num, &num, &num, &num, &num, &lx, &ly, &lz);
        pid0 = nid2pid[nid][0];
        if(pid0 != -1) {
          pid2coords[pid0].x = lx;
          pid2coords[pid0].y = ly;
          pid2coords[pid0].z = lz;
          pid2coords[pid0].t = 0;
          coords2pid[lx][ly][lz][0] = pid0;
          
          if(lx<minX) minX = lx; if(lx>maxX) maxX = lx;
          if(ly<minY) minY = ly; if(ly>maxY) maxY = ly;
          if(lz<minZ) minZ = lz; if(lz>maxZ) maxZ = lz;

          //printf("%d %d %d %d %d\n", pid0, lx, ly, lz, 0);
        }
        pid1 = nid2pid[nid][1];
        if(pid1 != -1) {
          pid2coords[pid1].x = lx;
          pid2coords[pid1].y = ly;
          pid2coords[pid1].z = lz;
          pid2coords[pid1].t = 1;
          coords2pid[lx][ly][lz][1] = pid1;

          if(lx<minX) minX = lx; if(lx>maxX) maxX = lx;
          if(ly<minY) minY = ly; if(ly>maxY) maxY = ly;
          if(lz<minZ) minZ = lz; if(lz>maxZ) maxZ = lz;
          
          //printf("%d %d %d %d %d\n", pid1, lx, ly, lz, 1);
        }
      }
      fclose(fp); 

      // assuming a contiguous allocation find the dimensions of 
      // the torus
      dimNX = maxX - minX;
      dimNY = maxY - minY;
      dimNZ = maxZ - minZ;
      dimNT = procsPerNode = 2;
      dimX = dimNX * dimNT;
      dimY = dimNY; dimZ = dimNZ;
    }

    ~CrayTorusManager() { }

    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }

    inline int getDimNX() { return dimNX; }
    inline int getDimNY() { return dimNY; }
    inline int getDimNZ() { return dimNZ; }
    inline int getDimNT() { return dimNT; }

    inline int getProcsPerNode() { return procsPerNode; }
    
    inline void rankToCoordinates(int pe, int &x, int &y, int &z, int &t) {
      x = pid2coords[pe].x; 
      y = pid2coords[pe].y; 
      z = pid2coords[pe].z; 
      t = pid2coords[pe].t; 
    }
    
    inline int coordinatesToRank(int x, int y, int z, int t) {
      return coords2pid[x][y][z][t];
    }
};

#endif // CMK_XT3
#endif //_CRAY_TORUS_H_
