#ifndef __PFFTUTIL_H__
#define __PFFTUTIL_H__

// some codes are borrowed from Orion's serial version.

#include <stdio.h>
#include <stdlib.h>
#include <fftw.h>
#include <charm++.h>

#define CAREFUL 1


class complex : public fftw_complex {
public:
  complex() {re=0; im=0;}
  explicit complex(fftw_real r) {re=r; im=0;}
  complex(fftw_real r,fftw_real i) {re=r; im=i;}
  double getMagSqr(void) const { return re*re+im*im; }
  complex operator+(complex a) { return complex(re+a.re,im+a.im); }
  complex conj(void) { return complex(re, -im); }
  void operator+=(complex a) { re+=a.re; im+=a.im; }
  void operator*=(complex a);
  complex operator*(complex a) {
    return complex( re * a.re - im * a.im, re * a.im + im * a.re); }
  void pup(PUP::er &p) {
    p|re;
    p|im;
  }
};

#endif //__PFFTUTIL_H__


