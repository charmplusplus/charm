
/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <math.h>

#include "charm++.h"
#include "LBTopology.h"

// ring

LBTOPO_MACRO(LBTopo_ring);

int LBTopo_ring::max_neighbors()
{
  if (npes > 2) return 2;
  else return (npes-1);
}

void LBTopo_ring::neighbors(int mype, int* _n, int &nb)
{
  nb = 0;
  if (npes>1) _n[nb++] = (mype + npes -1) % npes;
  if (npes>2) _n[nb++] = (mype + 1) % npes;
}

//  MESH 2D

LBTOPO_MACRO(LBTopo_mesh2d);

LBTopo_mesh2d::LBTopo_mesh2d(int p): LBTopology(p) 
{
  width = (int)sqrt(p);
  if (width * width < npes) width++;
}

int LBTopo_mesh2d::max_neighbors()
{
  return 4;
}

int LBTopo_mesh2d::goodcoor(int x, int y)
{
  if (x<0 && x>=width) return -1;
  if (y<0 && y>=width) return -1;
  int next = x*width + y;
  if (next<npes && next>=0) return next;
  return -1;
}

void LBTopo_mesh2d::neighbors(int mype, int* _n, int &nb)
{
  nb=0;
  int x = mype/width;
  int y = mype%width;
  for (int i=-1; i<=1; i+=2) {
    int x1 = x+i;
    int next = goodcoor(x1, y);
    if (next!=-1) _n[nb++] = next;
    int y1 = y+i;
    next = goodcoor(x, y1);
    if (next!=-1) _n[nb++] = next;
  }
}


//  MESH 3D

LBTOPO_MACRO(LBTopo_mesh3d);

LBTopo_mesh3d::LBTopo_mesh3d(int p): LBTopology(p) 
{
  width = (int)sqrt(p);
  width = 1;
  while ( (width+1) * (width+1) * (width+1) < npes) width++;
  if (width * width * width < npes) width++;
}

int LBTopo_mesh3d::max_neighbors()
{
  return 6;
}

int LBTopo_mesh3d::goodcoor(int x, int y, int z)
{
  if (x<0 && x>=width) return -1;
  if (y<0 && y>=width) return -1;
  if (z<0 && z>=width) return -1;
  int next = x*width*width + y*width + z;
  if (next<npes && next>=0) return next;
  return -1;
}

void LBTopo_mesh3d::neighbors(int mype, int* _n, int &nb)
{
  nb=0;
  int x = mype/(width*width);
  int k = mype%(width*width);
  int y = k/width;
  int z = k%width;
  for (int i=-1; i<=1; i+=2) {
    int x1 = x+i;
    int next = goodcoor(x1, y, z);
    if (next!=-1) _n[nb++] = next;
    int y1 = y+i;
    next = goodcoor(x, y1, z);
    if (next!=-1) _n[nb++] = next;
    int z1 = z+i;
    next = goodcoor(x, y, z1);
    if (next!=-1) _n[nb++] = next;
  }
}

//

class LBTopoMap {
public:
  char *name;
  LBtopoFn fn;
  LBTopoMap(char *s, LBtopoFn f): name(s), fn(f) {}
};
static CkVec<LBTopoMap *>  lbTopoMap;


void registerLBTopos()
{
  if (lbTopoMap.length()==0) {
  lbTopoMap.push_back(new LBTopoMap("ring", createLBTopo_ring));
  lbTopoMap.push_back(new LBTopoMap("mesh2d", createLBTopo_mesh2d));
  lbTopoMap.push_back(new LBTopoMap("mesh3d", createLBTopo_mesh3d));
  }
}

LBtopoFn LBTopoLookup(char *name)
{
  for (int i=0; i<lbTopoMap.length(); i++) {
    if (strcmp(name, lbTopoMap[i]->name)==0) return lbTopoMap[i]->fn;
  }
  return NULL;
}

