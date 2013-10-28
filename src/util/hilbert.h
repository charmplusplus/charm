#ifndef _HILBERT_H
#define _HILBERT_H

#include <vector>
using namespace std;

/*
 *  map a point with index of i on Hilbert walk in dim dimensions to its coordinate, return this coord
 */
extern vector<int> int_to_Hilbert(int i, int dim) ;

/*
 *  map a point with coordinate 'coords' to its linearized index on Hilbert walk , return this value 
 */
extern int Hilbert_to_int(const vector<int>& coords, int dim);

#endif
