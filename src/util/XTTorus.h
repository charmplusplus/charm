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

#if XE6_TOPOLOGY
#define CPU_FACTOR 2
#else
#define CPU_FACTOR 1
#endif

#if CMK_HAS_RCALIB
#include <rca_lib.h>
#endif

#if XT4_TOPOLOGY || XT5_TOPOLOGY || XE6_TOPOLOGY

extern "C" int *pid2nid;
extern "C" int pidtonid(int numpes);
extern "C" int getMeshCoord(int nid, int *x, int *y, int *z);
extern "C" void getDimension(int *,int *, int *, int *);
#if CMK_HAS_RCALIB
extern "C" rca_mesh_coord_t  *rca_coords;
#endif

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
    int dimNT;  // number of processors per node 
    int xDIM, yDIM, zDIM, maxNID;

    int torus[4];
    int procsPerNode;   // number of cores per node

    int ****coords2pid;     // coordinates to rank
    struct loc *pid2coords;                     // rank to coordinates

  public:
    XTTorusManager() {
      int nid = 0, oldnid = -1, lx, ly, lz;
      int i, j, k, l;
      int numCores;

      int numPes = CmiNumPes();
      pid2coords = (struct loc*)malloc(sizeof(struct loc) * numPes);
      _MEMCHECK(pid2coords);

      // fill the nid2pid and pid2nid data structures
      pidtonid(numPes);
      getDimension(&maxNID,&xDIM,&yDIM,&zDIM);
      numCores = CmiNumCores()*CPU_FACTOR;

      coords2pid = (int ****)malloc(xDIM*sizeof(int***));
      _MEMCHECK(coords2pid);
      for(i=0; i<xDIM; i++) {
		coords2pid[i] = (int ***)malloc(yDIM*sizeof(int**));
                _MEMCHECK(coords2pid[i]);
		for(j=0; j<yDIM; j++) {
			coords2pid[i][j] = (int **)malloc(zDIM*sizeof(int*));
                        _MEMCHECK(coords2pid[i][j]);
			for(k=0; k<zDIM; k++) {
				coords2pid[i][j][k] = (int *)malloc(numCores*sizeof(int*));
                                _MEMCHECK(coords2pid[i][j][k]);
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
      for(i=0; i<numPes; i++)
      {
        nid = pid2nid[i];
        if (nid != oldnid) {
          int ret = getMeshCoord(nid, &lx, &ly, &lz);
          CmiAssert(ret != -1);
        }
        oldnid = nid;

        pid2coords[i].x = lx;      
        pid2coords[i].y = ly;
        pid2coords[i].z = lz;

        l = 0;
        while(coords2pid[lx][ly][lz][l] != -1)
          l++;
        CmiAssert(l<numCores);
        coords2pid[lx][ly][lz][l] = i;
        pid2coords[i].t = l;
	if((l+1) > dimNT)
		dimNT = l+1;
      }

      // assuming a contiguous allocation find the dimensions of 
      // the torus
      dimNX = xDIM;
      dimNY = yDIM;
      dimNZ = zDIM;
      procsPerNode = dimNT;
      dimX = dimNX * dimNT;
      dimY = dimNY;
      dimZ = dimNZ;

      // we get a torus only if the size of the dimension is the biggest
      torus[0] = 1;		
      torus[1] = 1;
      torus[2] = 1;
      torus[3] = 1;
    }

    ~XTTorusManager() { 
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
        free(coords2pid);
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
      x = pid2coords[pe].x;
      y = pid2coords[pe].y;
      z = pid2coords[pe].z;
      t = pid2coords[pe].t;
    }

    inline int coordinatesToRank(int x, int y, int z, int t) {
      return coords2pid[x][y][z][t];
    }
};

#endif // XT4_TOPOLOGY || XT5_TOPOLOGY ||XE6_TOPOLOGY
#endif //_XT_TORUS_H_
