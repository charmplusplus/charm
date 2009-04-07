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
#if TERRY_BIT_COMPARE
  // This version contains some bit operations that confused Isaac, so he wrote the other implementation below.
  static const int endian = getFloatFormat();
  int maxUlps=100;
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
#else
  CkAssert(dim==3);
  const double threshold = 0.0001;
  for(int i=0; i<dim; i++) {
    const double a = key1[i];
    const double b = key2[i];
    const double diff = a-b;
    if (diff < -1.0*threshold){
      return 1;
    }
    else if (diff > threshold){
      return -1;
    }
  }
  return 0;
#endif
} 

inline int coordEqual(const double *key1, const double *key2, int dim) {
  return coordCompare(key1, key2, dim)==0;
}

inline int coordLessThan(const double *key1, const double *key2, int dim) {
  return coordCompare(key1, key2, dim)==1;
}

void ParFUM_desharing(int meshid);
void ParFUM_deghosting(int meshid);
void ParFUM_generateGlobalNodeNumbers(int fem_mesh, MPI_Comm comm);
void ParFUM_recreateSharedNodes(int meshid, int dim, int nParts);

void ParFUM_createComm(int meshid, int dim, MPI_Comm comm);

void ParFUM_import_nodes(int meshid, int numNodes, double *nodeCoords, int dim);
void ParFUM_import_elems(int meshid, int numElems, int nodesPer, int *conn, int type);
void ParFUM_findMatchingCoords(int dim, int extent_a, double* a,
			       int extent_b, double *b, 
			       std::vector<int>& matches_a,
			       std::vector<int>& matches_b
			       );

void sortNodes(double *nodes, double *sorted_nodes, int *sorted_ids, int numNodes, int dim);

#endif


