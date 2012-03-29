/** \file XT3Torus.h
 *  Author: Abhinav S Bhatele
 *  Date created: March 19th, 2007  
 *  
 *  This file makes use of a static table we obtain from a file available
 *  on Bigben. Bigben does a XYZT mapping by default. Two steps to obtaining
 *  pid2coords and coords2pid are:
 *
 *  1. Get the nid for each pid using the nidpid_map call (also store this 
 *     information as pid's (two) corresponding to a given nid
 *  2. For each nid in the CrayNeighborTablefile, get the coordinates and
 *     assign the corresponding pids in the nid2pid data structure with 
 *     these coordinates ('t' coord as 0 and 1)
 */

#ifndef _CRAY_TORUS_H_
#define _CRAY_TORUS_H_

#include "converse.h"

#include <stdlib.h>
#include <stdio.h>

#if XT3_TOPOLOGY

#define CPU_FACTOR 1
/*
#define XDIM 11
#define YDIM 12
#define ZDIM 16
#define TDIM 2
#define MAXNID 2784
*/
extern "C" int *pid2nid;
extern "C" int pidtonid(int numpes);
extern "C" int getMeshCoord(int nid, int *x, int *y, int *z);
extern "C" void getDimension(int *,int *, int *, int *);

struct loc {
  int x;
  int y;
  int z;
  int t;
};

class XT3TorusManager {
  private:
    int dimX;	// dimension of the allocation in X (processors)
    int dimY;	// dimension of the allocation in Y (processors)
    int dimZ;	// dimension of the allocation in Z (processors)
    int dimNX;	// dimension of the allocation in X (nodes)
    int dimNY;	// dimension of the allocation in Y (nodes)
    int dimNZ;	// dimension of the allocation in Z (nodes)
    int dimNT;  // number of processors per node (2 for XT3)
    int xDIM, yDIM, ZDIM, maxNID;
  
    int torus[4];
    int procsPerNode;   // number of cores per node
    char mapping[10];
    
    int ****coords2pid;     // coordinates to rank
    struct loc *pid2coords;                     // rank to coordinates
    struct loc origin;

  public:
    XT3TorusManager() {
      int nid = 0, oldnid = -1, lx, ly, lz;
			int numCores;
      int minX, minY, minZ, minT=0, maxX=0, maxY=0, maxZ=0;

      int numPes = CmiNumPes();
      pid2coords = (struct loc*)malloc(sizeof(struct loc) * numPes);

      // first fill the nid2pid and pid2nid data structures
      pidtonid(numPes);
			getDimension(&maxNID,&xDIM,&yDIM,&zDIM);
      minX=xDIM, minY=yDIM, minZ=zDIM;
			numCores = CmiNumCores()*CPU_FACTOR;
			
			coords2pid = (int ****)malloc(xDIM*sizeof(int***));
			for(i=0; i<xDIM; i++) {
				coords2pid[i] = (int ***)malloc(yDIM*sizeof(int**));
				for(j=0; j<yDIM; j++) {
					coords2pid[i][j] = (int **)malloc(zDIM*sizeof(int*));
					for(k=0; k<zDIM; k++) {
						coords2pid[i][j][k] = (int *)malloc(numCores*sizeof(int*));
					}
				}
			}

      for(i=0; i<xDIM; i++)
        for(j=0; j<yDIM; j++)
          for(k=0; k<zDIM; k++)
            for(l=0; l<numCores; l++)
              coords2pid[i][j][k][l] = -1;
 
      dimNT = 1;	
      // now fill the coords2pid and pid2coords data structures
      for(int i=0; i<numPes; i++)
      {
        nid = pid2nid[i];
	if (nid != oldnid)
	  getMeshCoord(nid, &lx, &ly, &lz);
	oldnid = nid;

        pid2coords[i].x = lx;      
        pid2coords[i].y = ly;
        pid2coords[i].z = lz;

        if(lx >= XDIM) printf("ERROR in X %d lx %d ly %d lz %d\n", i, lx, ly, lz);
        if(ly >= YDIM) printf("ERROR in Y %d lx %d ly %d lz %d\n", i, lx, ly, lz);
        if(lz >= ZDIM) printf("ERROR in Z %d lx %d ly %d lz %d\n", i, lx, ly, lz);

	if (coords2pid[lx][ly][lz][0] == -1) {
	  coords2pid[lx][ly][lz][0] = i;
	  pid2coords[i].t = 0;
	} else {
	  dimNT = 2;			// 2 cores per node
          coords2pid[lx][ly][lz][1] = i;
	  pid2coords[i].t = 1;
	}
          
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

      // we get a torus only if the size of the dimension is the biggest
      torus[0] = (dimNX == xDIM) ? 1 : 0;
      torus[1] = (dimNY == yDIM) ? 1 : 0;
      torus[2] = (dimNZ == zDIM) ? 1 : 0;
      torus[3] = 0;

      if(dimNT == 2) {
        int pe1 = coordinatesToRank(0, 0, 0, 0);
        int pe2 = coordinatesToRank(0, 0, 0, 1);
        if(pe2 == pe1 +1)
          sprintf(mapping, "%s", "TXYZ");
        else
          sprintf(mapping, "%s", "XYZT");
      }
    }

    ~XT3TorusManager() { 
			int i,j,k;
			free(pid2coords); 
			for(i=0; i<xDIM; i++) {
				for(j=0; j<yDIM; j++) {
					for(k=0; k<zDIM; k++) {
						free(coords2pid[i][j][k]);
					}
					free(coords2pid[i][j]);
				}
				free(coords2pid[i]);
			}
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

#endif // XT3_TOPOLOGY
#endif //_CRAY_TORUS_H_
