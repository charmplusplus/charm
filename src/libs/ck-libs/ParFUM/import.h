#ifndef __PARFUM_IMPORT_H
#define __PARFUM_IMPORT_H
#include "ParFUM.h"
#include "ParFUM_internals.h"

#define msbInt(var,key){\
	int *ptr = (int *)&key; \
	var = *(ptr+endian);\
}

extern int getFloatFormat(void);

inline int coordEqual(double *key1, double *key2, int dim) {
  static const int endian = getFloatFormat();
  int maxUlps=200;
  int xIntDiff, yIntDiff, zIntDiff;
  // Make int coords lexicographically ordered as twos-complement ints
  int x1Int,x2Int;
  msbInt(x1Int,key1[0]);
  msbInt(x2Int,key2[0]);
  if (x1Int < 0) x1Int = 0x80000000 - x1Int;
  if (x2Int < 0) x2Int = 0x80000000 - x2Int;
  xIntDiff = abs(x1Int - x2Int);
  
  if (dim > 1) {
    int y1Int,y2Int;
    msbInt(y1Int,key1[1]);
    msbInt(y2Int,key2[1]);
    if (y1Int < 0) y1Int = 0x80000000 - y1Int;
    if (y2Int < 0) y2Int = 0x80000000 - y2Int;
    yIntDiff = abs(y1Int - y2Int);
  }
  
  if (dim > 2) {
    int z1Int,z2Int;
    msbInt(z1Int,key1[2]);
    msbInt(z2Int,key2[2]);
    if (z1Int < 0) z1Int = 0x80000000 - z1Int;
    if (z2Int < 0) z2Int = 0x80000000 - z2Int;
    zIntDiff = abs(z1Int - z2Int);
  }
  
  if (dim == 1) return (xIntDiff<=maxUlps);
  else if (dim == 2) return((xIntDiff<=maxUlps) && (yIntDiff<=maxUlps));
  else if (dim == 3)
    return((xIntDiff<=maxUlps) && (yIntDiff<=maxUlps) && (zIntDiff<=maxUlps));
}

void ParFUM_desharing(int meshid);
void ParFUM_deghosting(int meshid);
void ParFUM_generateGlobalNodeNumbers(int fem_mesh);
void ParFUM_recreateSharedNodes(int meshid, int dim);

void ParFUM_createComm(int meshid, int dim);

void ParFUM_import_nodes(int meshid, int numNodes, double *nodeCoords, int dim);
void ParFUM_import_elems(int meshid, int numElems, int nodesPer, int *conn, int type);
#endif


