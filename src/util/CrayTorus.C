/** \file CrayTorus.C
 *  Author: Abhinav S Bhatele
 *  Date Created: March 22nd, 2007
 */

#include "CrayTorus.h"

#if CMK_XT3

CrayTorusManager::CrayTorusManager() {
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

CrayTorusManager::~CrayTorusManager() {

}

void CrayTorusManager::rankToCoordinates(int pe, int &x, int &y, int &z) {
  x = nid2coords[pe].x; 
  y = nid2coords[pe].y; 
  z = nid2coords[pe].z; 
}

int CrayTorusManager::coordinatesToRank(int x, int y, int z) {
  return coords2nid[x][y][z];
}

#endif
