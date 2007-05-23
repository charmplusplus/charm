/** \file CrayTorus.h
 *  Author: Abhinav S Bhatele
 *  Date created: March 19th, 2007  
 *  
 *  This file makes use of a static routing we obtain from a file available
 *  on Bigben.
 */

#ifndef _CRAY_TORUS_H_
#define _CRAY_TORUS_H_

#include "converse.h"
#include <stdlib.h>

#if CMK_XT3

#include <stdio.h>
//#include <catamount/cnos_mpi_os.h>
#define XDIM 11
#define YDIM 12
#define ZDIM 16
#define MAXNID 2783

struct loc {
  int x;
  int y;
  int z;
};

class CrayTorusManager {
  private:
    int dimX;	// dimension of the allocation in X (processors)
    int dimY;	// dimension of the allocation in Y (processors)
    int dimZ;	// dimension of the allocation in Z (processors)
    int dimNX;	// dimension of the allocation in X (nodes)
    int dimNY;	// dimension of the allocation in Y (nodes)
    int dimNZ;	// dimension of the allocation in Z (nodes)

    //cnos_nidpid_map_t* nidpid;
    int coords2nid[XDIM][YDIM][ZDIM];
    struct loc nid2coords[MAXNID];

  public:
    CrayTorusManager() {
      FILE *fp = fopen("CrayNeighbourTable", "r");
      int temp, nid, num, lx, ly, lz;
      char header[50];
      for(int i=0; i<10;i++)
        temp = fscanf(fp, "%s", header);
      for(int i=0; i<2112;i++)
      {
        temp = fscanf(fp, "%d%d%d%d%d%d%d%d%d%d", &nid, &num, &num, &num, &num, &num, &num, &lx, &ly, &lz);
        //printf("%d %d %d %d %d %d %d %d %d %d\n", nid, num, num, num, num, num, num, lx, ly, lz);
        nid2coords[nid].x = lx;
        nid2coords[nid].y = ly;
        nid2coords[nid].z = lz;
        coords2nid[lx][ly][lz] = nid;
      }
      fclose(fp); 
    }

    ~CrayTorusManager() { }

    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }

    inline int getDimNX() { return dimNX; }
    inline int getDimNY() { return dimNY; }
    inline int getDimNZ() { return dimNZ; }

    inline void rankToCoordinates(int pe, int &x, int &y, int &z) {
      x = nid2coords[pe].x; 
      y = nid2coords[pe].y; 
      z = nid2coords[pe].z; 
    }

    inline int coordinatesToRank(int x, int y, int z) {
      return coords2nid[x][y][z];
    }
};

#endif // CMK_XT3
#endif //_CRAY_TORUS_H_
