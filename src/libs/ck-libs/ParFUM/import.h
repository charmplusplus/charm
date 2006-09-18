#ifndef __PARFUM_IMPORT_H
#define __PARFUM_IMPORT_H
#include "ParFUM.h"
#include "ParFUM_internals.h"
#include <vector>

#define msbInt(var,key){\
	int *ptr = (int *)&key; \
	var = *(ptr+endian);\
}

extern int getFloatFormat(void);


/**
 * Returns:
 * 0  if key1 == key2
 * 1  if key1 > key2
 * -1 if key1 < key2
 */
inline int coordCompare(const double *key1, const double *key2, int dim) {
  static const int endian = getFloatFormat();
  int maxUlps=200;
  for(int ii=0; ii<dim; ii++) {
    int a, b;
    msbInt(a,key1[ii]);
    msbInt(b,key2[ii]);
    if (a < 0) a = 0x80000000 - a;  //FIXME: hardcoded value assumes certian precision.
    if (b < 0) b = 0x80000000 - b;
    int diff = a-b;
    if (abs(diff) > maxUlps) {
      if (diff < 0) {
	//b greater than a
	return 1;
      } else {
	return -1;
      }
    }
  }
  return 0;
} 

inline int coordEqual(const double *key1, const double *key2, int dim) {
  return coordCompare(key1, key2, dim)==0;
}

inline int coordLessThan(const double *key1, const double *key2, int dim) {
  return coordCompare(key1, key2, dim)==1;
}

void ParFUM_desharing(int meshid);
void ParFUM_deghosting(int meshid);
void ParFUM_generateGlobalNodeNumbers(int fem_mesh);
void ParFUM_recreateSharedNodes(int meshid, int dim);

void ParFUM_createComm(int meshid, int dim);

void ParFUM_import_nodes(int meshid, int numNodes, double *nodeCoords, int dim);
void ParFUM_import_elems(int meshid, int numElems, int nodesPer, int *conn, int type);
void ParFUM_findMatchingCoords(int dim, int extent_a, double* a,
			       int extent_b, double *b, 
			       std::vector<int>& matches_a,
			       std::vector<int>& matches_b
			       );
#endif


