/** \file CrayTorus.h
 *  Author: Abhinav S Bhatele
 *  Date created: March 19th, 2007  
 *  
 *  This file makes use of a static routing we obtain from a file available
 *  on Bigben.
 */

#ifndef _CRAY_TORUS_H_
#define _CRAY_TORUS_H_

#ifdef CMK_XT3
class CrayTorusManager {
  private:
    int dimX;	// dimension of the allocation in X (processors)
    int dimY;	// dimension of the allocation in Y (processors)
    int dimZ;	// dimension of the allocation in Z (processors)
    int dimNX;	// dimension of the allocation in X (nodes)
    int dimNY;	// dimension of the allocation in Y (nodes)
    int dimNZ;	// dimension of the allocation in Z (nodes)

  public:
    CrayTorusManager();
    ~CrayTorusManager();

    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }

    inline int getDimNX() { return dimNX; }
    inline int getDimNY() { return dimNY; }
    inline int getDimNZ() { return dimNZ; }

    inline void rankToCoordinates(int pe, int &x, int &y, int &z);
    inline int coordinatesToRank(int x, int y, int z);
    inline int getHopsBetweenRanks(int pe1, int pe2);
    inline void sortRanksByHops(int pe, int *pes, int *idx, int n); 

};

#endif // CMK_XT3
#endif //_CRAY_TORUS_H_
