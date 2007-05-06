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
    CrayTorusManager();
    ~CrayTorusManager();

    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }

    inline int getDimNX() { return dimNX; }
    inline int getDimNY() { return dimNY; }
    inline int getDimNZ() { return dimNZ; }

    void rankToCoordinates(int pe, int &x, int &y, int &z);
    int coordinatesToRank(int x, int y, int z);
};

#endif // CMK_XT3
#endif //_CRAY_TORUS_H_
