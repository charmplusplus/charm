#ifndef __PAIRUTIL_H__
#define __PAIRUTIL_H__

// some codes are borrowed from Orion's serial version.

#include <stdio.h>
#include <stdlib.h>
#include <fftw.h>
#include <charm++.h>

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

// ComplexPoint: real and imaginary components and 3 coordinates
struct ComplexPt {
  double re, im;
  int x, y, z;
  ComplexPt() {re=im=0.0; x=y=z=0;}
  ComplexPt(double re_,double im_,int x_,int y_,int z_) {
    re=re_; im=im_;
    x=x_; y=y_; z=z_;
  }
};


class complex : public fftw_complex {
public:
  complex() {re=0; im=0;}
  explicit complex(fftw_real r) {re=r; im=0;}
  complex(fftw_real r,fftw_real i) {re=r; im=i;}
  double getMagSqr(void) const { return re*re+im*im; }
  inline complex operator+(complex a) { return complex(re+a.re,im+a.im); }
  inline complex conj(void) { return complex(re, -im); }
  inline void operator+=(complex a) { re+=a.re; im+=a.im; }
  inline bool operator==(complex a) const { return(re==a.re && im==a.im); }
  inline void operator*=(complex a) {        
    double treal, tim;
    treal = re * a.re - im * a.im;
    tim = re * a.im + im * a.re;
    re = treal;
    im = tim;
  }
  inline complex operator*(complex a) {
    return complex( re * a.re - im * a.im, re * a.im + im * a.re); }
  void pup(PUP::er &p) {
    p|re;
    p|im;
  }
};

#endif //__PAIRUTIL_H__


