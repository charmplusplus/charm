
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
#include "LBDatabase.h"
#include "topology.h"

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

LBTOPO_MACRO(LBTopo_torus2d);

LBTopo_torus2d::LBTopo_torus2d(int p): LBTopology(p) 
{
  width = (int)sqrt(p*1.0);
  if (width * width < npes) width++;
}

int LBTopo_torus2d::max_neighbors()
{
  return 4;
}

int LBTopo_torus2d::goodcoor(int x, int y)
{
  if (x<0 || x>=width) return -1;
  if (y<0 || y>=width) return -1;
  int next = x*width + y;
  if (next<npes && next>=0) return next;
  return -1;
}

static int checkuniq(int *arr, int nb, int val) {
  for (int i=0;i<nb;i++) if (arr[i]==val) return 0;
  return 1;
}

void LBTopo_torus2d::neighbors(int mype, int* _n, int &nb)
{
  int next;
  int x = mype/width;
  int y = mype%width;
  nb=0;
  for (int i=-1; i<=1; i+=2) {
    int x1 = x+i;
    if (x1 == -1) {
      x1 = width-1;
      while (goodcoor(x1, y)==-1) x1--;
    }
    else if (goodcoor(x1, y) == -1) x1=0;
    next = goodcoor(x1, y);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;

    int y1 = y+i;
    if (y1 == -1) {
      y1 = width-1;
      while (goodcoor(x, y1)==-1) y1--;
    }
    else if (goodcoor(x, y1) == -1) y1=0;
    next = goodcoor(x, y1);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;
  }
}


//  MESH 3D

LBTOPO_MACRO(LBTopo_torus3d);

LBTopo_torus3d::LBTopo_torus3d(int p): LBTopology(p) 
{
  width = 1;
  while ( (width+1) * (width+1) * (width+1) <= npes) width++;
  if (width * width * width < npes) width++;
}

int LBTopo_torus3d::max_neighbors()
{
  return 6;
}

int LBTopo_torus3d::goodcoor(int x, int y, int z)
{
  if (x<0 || x>=width) return -1;
  if (y<0 || y>=width) return -1;
  if (z<0 || z>=width) return -1;
  int next = x*width*width + y*width + z;
  if (next<npes && next>=0) return next;
  return -1;
}

void LBTopo_torus3d::neighbors(int mype, int* _n, int &nb)
{
  int x = mype/(width*width);
  int k = mype%(width*width);
  int y = k/width;
  int z = k%width;
  int next;
  nb=0;
  for (int i=-1; i<=1; i+=2) {
    int x1 = x+i;
    if (x1 == -1) {
      x1 = width-1;
      while (goodcoor(x1, y, z)==-1) x1--;
    }
    else if (goodcoor(x1, y, z) == -1) x1=0;
    next = goodcoor(x1, y, z);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;

    int y1 = y+i;
    if (y1 == -1) {
      y1 = width-1;
      while (goodcoor(x, y1, z)==-1) y1--;
    }
    else if (goodcoor(x, y1, z) == -1) y1=0;
    next = goodcoor(x, y1, z);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;

    int z1 = z+i;
    if (z1 == -1) {
      z1 = width-1;
      while (goodcoor(x, y, z1)==-1) z1--;
    }
    else if (goodcoor(x, y, z1) == -1) z1=0;
    next = goodcoor(x, y, z1);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;
  }
}

// dense graph

LBTOPO_MACRO(LBTopo_graph);

int LBTopo_graph::max_neighbors()
{
  return (int)(sqrt(1.0*CmiNumPes())+0.5);
}

extern "C" void gengraph(int, int, int, int *, int *, int);

void LBTopo_graph::neighbors(int mype, int* na, int &nb)
{
  gengraph(CmiNumPes(), (int)(sqrt(1.0*CmiNumPes())+0.5), 234, na, &nb, 0);
}

//

class LBTopoMap {
public:
  char *name;
  LBtopoFn fn;
  LBTopoMap(char *s, LBtopoFn f): name(s), fn(f) {}
};
static CkVec<LBTopoMap *>  lbTopoMap;


extern "C"
void registerLBTopos()
{
  if (lbTopoMap.length()==0) {
  lbTopoMap.push_back(new LBTopoMap("ring", createLBTopo_ring));
  lbTopoMap.push_back(new LBTopoMap("torus2d", createLBTopo_torus2d));
  lbTopoMap.push_back(new LBTopoMap("torus3d", createLBTopo_torus3d));
  lbTopoMap.push_back(new LBTopoMap("graph", createLBTopo_graph));
  }
}

extern "C"
LBtopoFn LBTopoLookup(char *name)
{
  for (int i=0; i<lbTopoMap.length(); i++) {
    if (strcmp(name, lbTopoMap[i]->name)==0) return lbTopoMap[i]->fn;
  }
  return NULL;
}

// C wrapper functions
extern "C" void getTopoNeighbors(void *topo, int myid, int* narray, int *n)
{
  ((LBTopology*)topo)->neighbors(myid, narray, *n);
}

extern "C" int getTopoMaxNeighbors(void *topo)
{
  return ((LBTopology*)topo)->max_neighbors();
}

extern "C" void printoutTopo()
{
  for (int i=0; i<lbTopoMap.length(); i++) {
    CmiPrintf("	%s\n", lbTopoMap[i]->name);
  }
}

