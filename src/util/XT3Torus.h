 /*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
 
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

    int torus[4];
    int procsPerNode;   // number of cores per node
    
    int coords2pid[XDIM][YDIM][ZDIM][TDIM];     // coordinates to rank
    struct loc *pid2coords;                     // rank to coordinates
    struct loc origin;

  public:
    CrayTorusManager() {
      // open file to load data from CrayNeighborTable
      FILE *fp = fopen("/usr/users/4/abhatele/work/charm/src/util/CrayNeighbourTable", "r");

      int temp, nid, pid0, pid1, num, lx, ly, lz;
      int minX=XDIM, minY=YDIM, minZ=ZDIM, minT=0, maxX=0, maxY=0, maxZ=0;
      char header[50];
      pid2coords = (struct loc*)malloc(sizeof(struct loc) * CmiNumPes());

      // first fill the nid2pid and pid2nid data structures
      pidtonid(CmiNumPes());
      
      // skip header
      for(int i=0; i<10;i++)
        temp = fscanf(fp, "%s", header);

      dimNT = 1; 			// assume SN mode first        
      // now read the lines one at a time and fill the two arrays
      // for pid2coords and coords2pid
      for(int i=0; i<2112;i++)
      {
        temp = fscanf(fp, "%d%d%d%d%d%d%d%d%d%d", &nid, &num, &num, &num, &num, &num, &num, &lx, &ly, &lz);

        // look for the first position (0) on the node for a pid
	// this will be assigned 't' of 0
        pid0 = nid2pid[nid][0];
        
        if (pid0 != -1) {		// if the pid exists
          pid2coords[pid0].x = lx;      
          pid2coords[pid0].y = ly;
          pid2coords[pid0].z = lz;
          pid2coords[pid0].t = 0;	// give it position 0 on the node
          coords2pid[lx][ly][lz][0] = pid0;
          
          if(lx<minX) minX = lx; if(lx>maxX) maxX = lx;
          if(ly<minY) minY = ly; if(ly>maxY) maxY = ly;
          if(lz<minZ) minZ = lz; if(lz>maxZ) maxZ = lz;
        }

        // look for the second position (1) on the node for a pid
	// this will be assigned 't' of 1
        pid1 = nid2pid[nid][1];
        if (pid1 != -1) {		// if the pid exists
	  dimNT = 2;			// we are running in VN mode
          pid2coords[pid1].x = lx;
          pid2coords[pid1].y = ly;
          pid2coords[pid1].z = lz;
          pid2coords[pid1].t = 1;	// give it position 1 on the node
          coords2pid[lx][ly][lz][1] = pid1;

          if(lx<minX) minX = lx; if(lx>maxX) maxX = lx;
          if(ly<minY) minY = ly; if(ly>maxY) maxY = ly;
          if(lz<minZ) minZ = lz; if(lz>maxZ) maxZ = lz;
        }
      }
      fclose(fp); 

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
      torus[0] = (dimNX == XDIM) ? 1 : 0;
      torus[1] = (dimNY == YDIM) ? 1 : 0;
      torus[2] = (dimNZ == ZDIM) ? 1 : 0;
      torus[3] = 0;
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
