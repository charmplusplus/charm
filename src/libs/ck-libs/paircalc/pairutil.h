#ifndef __PAIRUTIL_H__
#define __PAIRUTIL_H__

// some codes are borrowed from Orion's serial version.

#include "charm++.h"
#include "ckcomplex.h"

/*
struct Index4D {unsigned short w,x,y,z;};

class CkArrayIndexIndex4D: public CkArrayIndex {
 public:
  unsigned short index[4];
  CkArrayIndexIndex4D(unsigned short i0,
		      unsigned short i1,
		      unsigned short i2,
		      unsigned short i3)
    {index[0] = i0; index[1] = i1; index[2] = i2; index[3] = i3; nInts = 2;}
};

typedef ArrayElementT<Index4D> ArrayElement4D;
*/

// ComplexPoint: real and imaginary components and 3 coordinates
struct ComplexPt {
  double re, im;
  int x, y, z;
  inline ComplexPt() {re=im=0.0; x=y=z=0;}
  inline ComplexPt(double re_,double im_,int x_,int y_,int z_) {
    re=re_; im=im_;
    x=x_; y=y_; z=z_;
  }
};

#endif //__PAIRUTIL_H__


