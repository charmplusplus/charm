#ifndef __PARFUM_IMPORT_H
#define __PARFUM_IMPORT_H
#include "ParFUM.h"
#include "ParFUM_internals.h"


#define msbInt(var,key){\
	int *ptr = (int *)&key; \
	var = *(ptr+endian);\
}

extern int getFloatFormat(void);

inline int coordEqual(double *key1, double *key2) {
	static const int endian = getFloatFormat();
  int maxUlps=200;
  // Make int coords lexicographically ordered as twos-complement ints
  int x1Int,x2Int;
	msbInt(x1Int,key1[0]);
	msbInt(x2Int,key2[0]);
  if (x1Int < 0) x1Int = 0x80000000 - x1Int;
  if (x2Int < 0) x2Int = 0x80000000 - x2Int;
	
  int y1Int,y2Int;
	msbInt(y1Int,key1[1]);
	msbInt(y2Int,key2[1]);
  if (y1Int < 0) y1Int = 0x80000000 - y1Int;
  if (y2Int < 0) y2Int = 0x80000000 - y2Int;
  
	int z1Int,z2Int;
	msbInt(z1Int,key1[2]);
	msbInt(z2Int,key2[2]);
  if (z1Int < 0) z1Int = 0x80000000 - z1Int;
  if (z2Int < 0) z2Int = 0x80000000 - z2Int;

  int xIntDiff = abs(x1Int - x2Int);
  int yIntDiff = abs(y1Int - y2Int);
  int zIntDiff = abs(z1Int - z2Int);
  return((xIntDiff<=maxUlps) && (yIntDiff<=maxUlps) && (zIntDiff<=maxUlps));
}

void ParFUM_desharing(int meshid);
void ParFUM_deghosting(int meshid);
void ParFUM_generateGlobalNodeNumbers(int fem_mesh);
void ParFUM_recreateSharedNodes(int meshid);

void ParFUM_createComm(int meshid);
#endif


